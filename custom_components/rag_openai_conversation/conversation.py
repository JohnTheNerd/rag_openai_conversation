"""Conversation support for OpenAI."""

from collections.abc import AsyncGenerator, Callable
from datetime import timedelta
from homeassistant.helpers.event import async_track_time_interval
import json
import copy
from typing import Any, Literal, cast

import numpy as np

import openai
from openai._streaming import AsyncStream
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition
from voluptuous_openapi import convert

from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    chat_session,
    device_registry as dr,
    config_validation as cv,
    entity_registry as er,
    intent,
    template,
    llm,
)
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

import random
import re

from thefuzz import fuzz
from thefuzz import process

from . import OpenAIConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_EMBEDDING_MODEL,
    CONF_ENABLE_RAG,
    CONF_MAX_TOKENS,
    CONF_MAX_MESSAGES,
    CONF_FUZZY_TOP_N,
    CONF_RAG_TOP_N,
    CONF_PROMPT,
    CONF_RAG_REFRESH_INTERVAL,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

IGNORED_TOOLS = ["get_home_state", "GetLiveContext"]


def _cosine_similarity(first, second):
    dot_product = np.dot(first, second)
    magnitude_product = np.sqrt(np.dot(first, first)) * np.sqrt(np.dot(second, second))
    return dot_product / magnitude_product


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = OpenAIConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


def _convert_content_to_param(
    content: conversation.Content,
) -> ChatCompletionMessageParam:
    """Convert any native chat message for this agent to the native format."""
    if content.role == "tool_result":
        assert type(content) is conversation.ToolResultContent
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json.dumps(content.tool_result),
        )
    if content.role != "assistant" or not content.tool_calls:  # type: ignore[union-attr]
        role = content.role
        if role == "system":
            role = "developer"
        return cast(
            ChatCompletionMessageParam,
            {"role": content.role, "content": content.content},  # type: ignore[union-attr]
        )

    # Handle the Assistant content including tool calls.
    assert type(content) is conversation.AssistantContent
    return ChatCompletionAssistantMessageParam(
        role="assistant",
        content=content.content,
        tool_calls=[
            ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                function=Function(
                    arguments=json.dumps(tool_call.tool_args),
                    name=tool_call.tool_name,
                ),
                type="function",
            )
            for tool_call in content.tool_calls
        ],
    )


async def _transform_stream(
    result: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform an OpenAI delta stream into HA format."""
    current_tool_call: dict | None = None

    async for chunk in result:
        LOGGER.debug("Received chunk: %s", chunk)
        choice = chunk.choices[0]

        if choice.finish_reason:
            if current_tool_call:
                tool_args = []
                if current_tool_call["tool_args"]:
                    tool_args = json.loads(current_tool_call["tool_args"])
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=current_tool_call["id"],
                            tool_name=current_tool_call["tool_name"],
                            tool_args=tool_args,
                        )
                    ]
                }

            break

        delta = chunk.choices[0].delta

        # We can yield delta messages not continuing or starting tool calls
        if current_tool_call is None and not delta.tool_calls:
            yield {  # type: ignore[misc]
                key: value
                for key in ("role", "content")
                if (value := getattr(delta, key)) is not None
            }
            continue

        # When doing tool calls, we should always have a tool call
        # object or we have gotten stopped above with a finish_reason set.
        if (
            not delta.tool_calls
            or not (delta_tool_call := delta.tool_calls[0])
            or not delta_tool_call.function
        ):
            continue

        if current_tool_call and delta_tool_call.index == current_tool_call["index"]:
            current_tool_call["tool_args"] += delta_tool_call.function.arguments or ""
            continue

        # We got tool call with new index, so we need to yield the previous
        if current_tool_call:
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call["id"],
                        tool_name=current_tool_call["tool_name"],
                        tool_args=json.loads(current_tool_call["tool_args"]),
                    )
                ]
            }

        current_tool_call = {
            "index": delta_tool_call.index,
            "id": delta_tool_call.id,
            "tool_name": delta_tool_call.function.name,
            "tool_args": delta_tool_call.function.arguments or "",
        }


class OpenAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """OpenAI conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OpenAIConfigEntry) -> None:
        """Initialize the agent."""
        super().__init__()
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._rag_documents = []
        self._rag_examples = []
        self._rag_update_unsub = None
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="OpenAI",
            model="ChatGPT",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    def _get_entity_examples(self, entity):
        examples = []
        exclude_state_domains = ["lock", "weather"]
        entity_domain = entity.get("entity_id").split(".")[0]
        if entity_domain not in exclude_state_domains:
            question = f"What is the state of {entity['name']}?"
            tool_call = None
            tool_call_params = None
            answer = f"The {entity['name']} is {entity['state']}."
            examples.append(
                {
                    "question": question,
                    "tool_call": tool_call,
                    "tool_call_params": tool_call_params,
                    "answer": answer,
                }
            )
        match entity_domain:
            case "button":
                question = f"Press the {entity['name']}."
                tool_call = "HassTurnOn"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been pressed."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
            case "climate":
                question = f"Turn the {entity['name']} on."
                tool_call = "HassTurnOn"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned on."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Turn the {entity['name']} off."
                tool_call = "HassTurnOff"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned off."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                temperature = random.randint(15, 30)
                question = f"Set the {entity['name']} to {temperature} degrees."
                tool_call = "HassClimateSetTemperature"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "temperature": temperature}
                )
                answer = f"The {entity['name']} was set to {temperature} degrees."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                if "temperature" in entity:
                    question = f"What is the temperature of the {entity['name']}?"
                    tool_call = None
                    tool_call_params = None
                    answer = f"The temperature of the {entity['name']} is {entity['temperature']} degrees."
                    examples.append(
                        {
                            "question": question,
                            "tool_call": tool_call,
                            "tool_call_params": tool_call_params,
                            "answer": answer,
                        }
                    )
            case "cover":
                question = f"Turn the {entity['name']} on."
                tool_call = "HassTurnOn"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned on."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Turn the {entity['name']} off."
                tool_call = "HassTurnOff"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned off."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                position = random.randint(5, 95)
                question = f"Set the {entity['name']} to {position}%."
                tool_call = "HassSetPosition"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "position": position}
                )
                answer = f"The {entity['name']} was set to {position}%."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
            case "fan":
                question = f"Turn the {entity['name']} on."
                tool_call = "HassTurnOn"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned on."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Turn the {entity['name']} off."
                tool_call = "HassTurnOff"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned off."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                position = random.randint(5, 95)
                question = f"Set the {entity['name']} to {position}%."
                tool_call = "HassSetPosition"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "position": position}
                )
                answer = f"The {entity['name']} was set to {position}%."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
            case "light":
                question = f"Turn the {entity['name']} on."
                tool_call = "HassTurnOn"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned on."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Turn the {entity['name']} off."
                tool_call = "HassTurnOff"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned off."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                colors = ["red", "green", "blue"]
                color = random.choice(colors)
                question = f"Make the {entity['name']} {color}."
                tool_call = "HassLightSet"
                tool_call_params = json.dumps(
                    {
                        "name": entity["name"],
                        "domain": entity_domain,
                        "color": color,
                    }
                )
                answer = f"The {entity['name']} is now {color}."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                brightness = random.randint(5, 95)
                question = f"Make the {entity['name']} brightness {brightness}%."
                tool_call = "HassLightSet"
                tool_call_params = json.dumps(
                    {
                        "name": entity["name"],
                        "domain": entity_domain,
                        "brightness": brightness,
                    }
                )
                answer = f"The {entity['name']} is now at {brightness}% brightness."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
            case "lock":
                question = f"Is the {entity['name']} locked?"
                tool_call = None
                tool_call_params = None
                if entity["state"] == "on":
                    answer = f"The {entity['name']} is locked."
                else:
                    answer = f"The {entity['name']} is unlocked."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Lock the {entity['name']}."
                tool_call = "HassTurnOn"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} is now locked."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Unlock the {entity['name']}."
                tool_call = "HassTurnOff"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} is now unlocked."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
            case "media_player":
                question = f"Turn the {entity['name']} on."
                tool_call = "HassTurnOn"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned on."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Turn the {entity['name']} off."
                tool_call = "HassTurnOff"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned off."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Unpause the {entity['name']}."
                tool_call = "HassMediaUnpause"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} is no longer paused."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Pause the {entity['name']}."
                tool_call = "HassMediaPause"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} is now paused."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Play the next song on {entity['name']}."
                tool_call = "HassMediaNext"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} skipped to the next song."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Play the previous song on {entity['name']}."
                tool_call = "HassMediaNext"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} is playing the previous song."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                volume = random.randint(5, 95)
                question = f"Set the {entity['name']} to volume {volume}%."
                tool_call = "HassSetVolume"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "volume_level": volume}
                )
                answer = (
                    f"The {entity['name']}'s volume is now set to {volume}% volume."
                )
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
            case "switch":
                question = f"Turn the {entity['name']} on."
                tool_call = "HassTurnOn"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned on."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Turn the {entity['name']} off."
                tool_call = "HassTurnOff"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been turned off."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
            case "vacuum":
                question = f"Start the {entity['name']}."
                tool_call = "HassVacuumStart"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} has been started."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
                question = f"Send the {entity['name']} back to the base."
                tool_call = "HassVacuumReturnToBase"
                tool_call_params = json.dumps(
                    {"name": entity["name"], "domain": entity_domain}
                )
                answer = f"The {entity['name']} is on its way back to the base."
                examples.append(
                    {
                        "question": question,
                        "tool_call": tool_call,
                        "tool_call_params": tool_call_params,
                        "answer": answer,
                    }
                )
            case "weather":
                pass

        return examples

    def _get_area_examples(self, area):
        examples = []
        area_entities = []
        device_registry = dr.async_get(self.hass)
        area_devices = {
            device.id
            for device in device_registry.devices.values()
            if device.area_id == area.id
        }

        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)
            if entity.device_id in area_devices:
                area_entities.append(entity)

        interesting_domains = [
            "button",
            "climate",
            "cover",
            "fan",
            "light",
            "lock",
            "media_player",
            "switch",
            "vacuum",
        ]

        for domain in interesting_domains:
            for entity in area_entities:
                if domain == entity.domain:
                    match entity.domain:
                        case "climate":
                            question = f"Turn the {area.name} climate control on."
                            tool_call = "HassTurnOn"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = (
                                f"The {area.name} climate control has been turned on."
                            )
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            question = f"Turn the {area.name} climate control off."
                            tool_call = "HassTurnOff"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = (
                                f"The {area.name} climate control has been turned off."
                            )
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            temperature = random.randint(15, 30)
                            question = f"Set the {area.name} to {temperature} degrees."
                            tool_call = "HassClimateSetTemperature"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "temperature": temperature,
                                }
                            )
                            answer = (
                                f"The {area.name} was set to {temperature} degrees."
                            )
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            break
                        case "cover":
                            question = f"Turn the {area.name} covers on."
                            tool_call = "HassTurnOn"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = f"The {area.name} covers have been turned on."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            question = f"Turn the {area.name} covers off."
                            tool_call = "HassTurnOff"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = f"The {area.name} covers have been turned off."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            position = random.randint(5, 95)
                            question = f"Set the {area.name} covers to {position}%."
                            tool_call = "HassSetPosition"
                            tool_call_params = json.dumps(
                                {"area": area.name, "position": position}
                            )
                            answer = f"The {area.name} covers were set to {position}%."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            break
                        case "fan":
                            question = f"Turn the {area.name} fans on."
                            tool_call = "HassTurnOn"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = f"The {area.name} fans have been turned on."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            question = f"Turn the {area.name} fans off."
                            tool_call = "HassTurnOff"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = f"The {area.name} fans have been turned off."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            position = random.randint(5, 95)
                            question = f"Set the {area.name} fans to {position}%."
                            tool_call = "HassSetPosition"
                            tool_call_params = json.dumps(
                                {"area": area.name, "position": position}
                            )
                            answer = f"The {area.name} fans were set to {position}%."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                        case "light":
                            question = f"Turn the {area.name} lights on."
                            tool_call = "HassTurnOn"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = f"The {area.name} lights are now been turned on."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            question = f"Turn the {area.name} lights off."
                            tool_call = "HassTurnOff"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = f"The {area.name} lights are now turned off."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            colors = ["red", "green", "blue"]
                            color = random.choice(colors)
                            question = f"Make the {area.name} lights {color}."
                            tool_call = "HassLightSet"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                    "color": color,
                                }
                            )
                            answer = f"The {area.name} lights are now {color}."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            brightness = random.randint(5, 95)
                            question = (
                                f"Make the {area.name} light brightness {brightness}%."
                            )
                            tool_call = "HassLightSet"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                    "brightness": brightness,
                                }
                            )
                            answer = f"The {area.name} lights are now at {brightness}% brightness."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                        case "lock":
                            question = f"Lock the {area.name}."
                            tool_call = "HassTurnOn"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = f"The {area.name} is now locked."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            question = f"Unlock the {area.name}."
                            tool_call = "HassTurnOff"
                            tool_call_params = json.dumps(
                                {
                                    "area": area.name,
                                    "domain": domain,
                                }
                            )
                            answer = f"The {area.name} is now unlocked."
                            examples.append(
                                {
                                    "question": question,
                                    "tool_call": tool_call,
                                    "tool_call_params": tool_call_params,
                                    "answer": answer,
                                }
                            )
                            break

        return examples

    def _render_example(self, example):
        """Render the example."""
        example_text = ""
        question = example.get("question")
        tool_call = example.get("tool_call")
        tool_call_params = example.get("tool_call_params")
        answer = example.get("answer")

        example_text += f"Example question: {question}\n"
        if tool_call:
            example_text += f"You should call the tool named {tool_call}\n"
        if tool_call_params:
            example_text += f"With the arguments {tool_call_params}\n"
        example_text += f"Example answer: {answer}"

        return template.Template(example_text, self.hass).async_render()

    def _get_exposed_entities(self):
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            entity_object = {
                "entity_id": entity_id,
                "name": state.name,
                "state": self.hass.states.get(entity_id).state
                + state.attributes.get("unit_of_measurement", ""),
                "aliases": aliases,
                "last_updated": state.last_updated,
                "device_id": entity.device_id,
            }

            if (
                "brightness" in state.attributes
                and state.attributes["brightness"] is not None
            ):
                entity_object["brightness"] = (
                    str(round(state.attributes["brightness"] / 255 * 100)) + "%"
                )
            if (
                "temperature" in state.attributes
                and state.attributes["temperature"] is not None
            ):
                entity_object["temperature"] = str(
                    round(state.attributes["temperature"])
                ) + state.attributes.get("unit_of_measurement", "")
            if (
                "current_temperature" in state.attributes
                and state.attributes["current_temperature"] is not None
            ):
                entity_object["temperature"] = str(
                    round(state.attributes["current_temperature"])
                ) + state.attributes.get("unit_of_measurement", "")
            if (
                "volume_level" in state.attributes
                and state.attributes["volume_level"] is not None
            ):
                entity_object["volume_level"] = (
                    str(round(state.attributes["volume_level"] * 100)) + "%"
                )
            exposed_entities.append(entity_object)
        return exposed_entities

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )
        if self.entry.options.get(CONF_ENABLE_RAG):
            options = self.entry.options
            self._rag_update_unsub = async_track_time_interval(
                self.hass,
                self._update_rag_documents,
                timedelta(minutes=options.get(CONF_RAG_REFRESH_INTERVAL, 60)),
            )
            self.entry.async_on_unload(self._rag_update_unsub)

    async def async_will_remove_from_hass(self) -> None:
        """Clean up when removing from Home Assistant."""
        if self._rag_update_unsub:
            self._rag_update_unsub()
        await super().async_will_remove_from_hass()

    async def _update_rag_documents(self, now=None):
        """Update RAG documents and examples with subentries."""

        clients = self.entry.runtime_data
        rag_client = clients.rag
        options = self.entry.options
        exposed_entities = sorted(
            self._get_exposed_entities(), key=lambda x: x["last_updated"]
        )

        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        entity_registry = er.async_get(self.hass)

        unnecessary_entity_attributes = [
            "entity_id",
            "aliases",
            "area_id",
            "name",
            "device_id",
        ]

        documents = []
        examples = []

        # Populate with areas and floors, alongside the entities inside that are exposed

        entities_with_areas = set()
        for area in area_registry.areas.values():
            area_devices = [
                device.id
                for device in device_registry.devices.values()
                if device.area_id == area.id
            ]
            area_entities = []
            for device_id in area_devices:
                device = device_registry.async_get(device_id)
                if device:
                    for exposed_entity in exposed_entities:
                        if exposed_entity["device_id"] == device.id:
                            area_entities.append(exposed_entity)
                            entities_with_areas.add(exposed_entity["entity_id"])
            area_entities = sorted(
                area_entities,
                key=lambda x: x["last_updated"],
                reverse=True,
            )

            description = f"""

Entities in area {area.name}:
"""
            prompt_extension = description
            if area_entities:
                for entity in area_entities:
                    description += f"- {entity['name']}\n"
                    prompt_extension += f"- {entity['name']}:\n"
                    for key, value in entity.items():
                        if key not in unnecessary_entity_attributes:
                            prompt_extension += f"{key}: {value}\n"
                documents.append(
                    {"description": description, "prompt": prompt_extension}
                )

        found_entities_without_areas = False
        description = """

Entities that do not have an area:
"""
        prompt_extension = description
        for entity in exposed_entities:
            if entity.get("entity_id") not in entities_with_areas:
                found_entities_without_areas = True
                description += f"- {entity['name']}\n"
                prompt_extension += f"\n- {entity['name']}:\n"
                for key, value in entity.items():
                    if key not in unnecessary_entity_attributes:
                        prompt_extension += f"{key}: {value}\n"
        if found_entities_without_areas:
            documents.append(
                {
                    "description": description,
                    "prompt": prompt_extension,
                }
            )

        # Get all subentries
        for subentry in self.entry.subentries.values():
            match subentry.subentry_type:
                case "document":
                    documents.append(subentry.data)
                case "example":
                    example = dict(subentry.data)
                    example["text"] = self._render_example(example)
                    examples.append(example)

        rag_documents = []
        for document in documents:
            description = template.Template(
                document.get("description"), self.hass
            ).async_render(
                {
                    "exposed_entities": exposed_entities,
                }
            )
            embedding = await rag_client.embeddings.create(
                model=options.get(CONF_EMBEDDING_MODEL), input=description
            )
            rag_documents.append(
                {
                    "description": description,
                    "embedding": embedding,
                    "prompt_template": document.get("prompt"),
                }
            )
        self._rag_documents = rag_documents

        for entity in exposed_entities:
            entity_examples = self._get_entity_examples(entity)
            for example in entity_examples:
                example["text"] = self._render_example(example)
                examples.append(example)

        for area in area_registry.areas.values():
            area_examples = self._get_area_examples(area)
            for example in area_examples:
                example["text"] = self._render_example(example)
                examples.append(example)

        for example in examples:
            example["embedding"] = await rag_client.embeddings.create(
                model=options.get(CONF_EMBEDDING_MODEL), input=example["question"]
            )

        self._rag_examples = examples

        LOGGER.debug("Updated RAG documents with %d entries", len(self._rag_documents))
        LOGGER.debug("Updated RAG examples with %d entries", len(self._rag_examples))

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        with (
            chat_session.async_get_chat_session(
                self.hass, user_input.conversation_id
            ) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.entry.options

        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[ChatCompletionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
                if tool.name not in IGNORED_TOOLS
            ]

        clients = self.entry.runtime_data
        llm_client = clients.llm

        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        messages = [_convert_content_to_param(content) for content in chat_log.content]
        if options.get(CONF_MAX_MESSAGES, 10) > 0 and len(messages) > options.get(
            CONF_MAX_MESSAGES, 10
        ):
            # keep first message and last few messages
            messages = [messages[0]] + messages[-options.get(CONF_MAX_MESSAGES, 10) :]
        if options.get(CONF_ENABLE_RAG):
            if not self._rag_documents or not self._rag_examples:
                await self._update_rag_documents()
            rag_client = clients.rag
            documents = self._rag_documents
            examples = self._rag_examples
            exposed_entities = self._get_exposed_entities()

            current_time_prompt = template.Template(
                'Today\'s date is {{ now().strftime("%Y-%m-%d") }}. Current time is {{ now().strftime("%H:%M:%S") }}.',
                self.hass,
            ).async_render(
                {
                    "exposed_entities": exposed_entities,
                }
            )

            prompt_embedding = await rag_client.embeddings.create(
                model=options.get(CONF_EMBEDDING_MODEL),
                input="\n\n".join(
                    [
                        message["content"]
                        for message in messages[1:]
                        if message["content"]
                    ]
                ),
            )
            prompt_embedding = np.array(prompt_embedding.data[0].embedding)

            selected_documents = []

            for document in documents:
                document["similarity"] = _cosine_similarity(
                    prompt_embedding, document["embedding"].data[0].embedding
                )

            # Sort the list of tuples based on cosine similarity in descending order
            sorted_documents = sorted(
                documents, key=lambda x: x["similarity"], reverse=True
            )

            # Select the top documents based on cosine similarity
            selected_documents.extend(
                sorted_documents[: options.get(CONF_RAG_TOP_N, 3)]
            )

            # Log the top documents for debugging purposes
            for i, doc in enumerate(selected_documents, start=1):
                LOGGER.info(
                    f"Top {i} Document: {doc['description']} (Similarity: {doc['similarity']}): {doc['prompt_template']}"
                )

            document_choices = [document["description"] for document in documents]

            top_fuzzy_matches = process.extract(
                "\n\n".join(
                    [
                        message["content"]
                        for message in messages[1:]
                        if message["content"]
                    ]
                ),
                document_choices,
                limit=options.get(CONF_FUZZY_TOP_N, 3),
                scorer=fuzz.partial_token_sort_ratio,
            )

            for match in top_fuzzy_matches:
                LOGGER.info(f"Fuzzy match: {match[0]} (Score: {match[1]})")
                for document in documents:
                    if document["description"] == match[0]:
                        for selected_document in selected_documents:
                            if (
                                selected_document["description"]
                                == document["description"]
                            ):
                                break
                        else:
                            document["similarity"] = match[1]
                            selected_documents.append(document)

            area_prompt = "An overview of this smart home:\n\n" + "\n".join(
                [
                    template.Template(doc["prompt_template"], self.hass).async_render(
                        {
                            "exposed_entities": exposed_entities,
                        }
                    )
                    for doc in selected_documents
                ]
            )

            selected_examples = []

            question_choices = [example["question"] for example in examples]

            top_fuzzy_matches = process.extract(
                "\n\n".join(
                    [
                        message["content"]
                        for message in messages[1:]
                        if message["content"]
                    ]
                ),
                question_choices,
                limit=options.get(CONF_FUZZY_TOP_N, 3),
                scorer=fuzz.partial_token_sort_ratio,
            )

            for match in top_fuzzy_matches:
                LOGGER.info(f"Fuzzy match: {match[0]} (Score: {match[1]})")
                for example in examples:
                    if example["question"] == match[0]:
                        for selected_example in selected_examples:
                            if selected_example["question"] == example["question"]:
                                break
                        else:
                            example["similarity"] = match[1]
                            selected_examples.append(example)

            for example in examples:
                example["similarity"] = _cosine_similarity(
                    prompt_embedding, example["embedding"].data[0].embedding
                )

            # Sort the list of tuples based on cosine similarity in descending order
            sorted_examples = sorted(
                examples, key=lambda x: x["similarity"], reverse=True
            )

            # Select the top examples based on cosine similarity
            selected_examples.extend(sorted_examples[: options.get(CONF_RAG_TOP_N, 3)])

            # Log the top examples for debugging purposes
            for i, example in enumerate(selected_examples, start=1):
                LOGGER.info(
                    f"Top {i} Example: {example['text']} (Similarity: {example['similarity']})"
                )

            example_prompt = "Examples:\n\n" + "\n\n".join(
                [
                    template.Template(example["text"], self.hass).async_render(
                        {
                            "exposed_entities": exposed_entities,
                        }
                    )
                    for example in selected_examples
                ]
            )

            messages[0]["content"] = (
                template.Template(options.get(CONF_PROMPT), self.hass).async_render(
                    {
                        "exposed_entities": exposed_entities,
                    }
                )
                + "\n\n"
                + area_prompt
                + "\n\n"
                + example_prompt
                + "\n\n"
                + current_time_prompt
            )

        LOGGER.info(messages[0]["content"])

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args = {
                "model": model,
                "messages": messages,
                "tools": tools or NOT_GIVEN,
                "max_completion_tokens": options.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "user": chat_log.conversation_id,
                "stream": True,
            }

            if model.startswith("o"):
                model_args["reasoning_effort"] = options.get(
                    CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                )

            try:
                result = await llm_client.chat.completions.create(**model_args)
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by OpenAI: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to OpenAI: %s", err)
                raise HomeAssistantError("Error talking to OpenAI") from err

            messages.extend(
                [
                    _convert_content_to_param(content)
                    async for content in chat_log.async_add_delta_content_stream(
                        user_input.agent_id, _transform_stream(result)
                    )
                ]
            )

            if not chat_log.unresponded_tool_results:
                break

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)
