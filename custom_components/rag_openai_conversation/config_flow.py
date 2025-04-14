"""Config flow for OpenAI Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any
from homeassistant.core import callback
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
    ConfigSubentryFlow,
    SubentryFlowResult,
)

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import (
    CONF_LLM_HASS_API,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    DEFAULT_CONF_LLM_BASE_URL,
    DEFAULT_CONF_RAG_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_EMBEDDING_MODEL,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_RAG_API_KEY,
    CONF_RAG_BASE_URL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_DESCRIPTION,
    CONF_QUESTION,
    CONF_TOOL_CALL,
    CONF_TOOL_CALL_ARGS,
    CONF_ANSWER,
    CONF_TITLE,
    CONF_ENABLE_RAG,
    CONF_MAX_MESSAGES,
    CONF_FUZZY_TOP_N,
    CONF_RAG_TOP_N,
    CONF_RAG_REFRESH_INTERVAL,
    CONF_REASONING_EFFORT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    UNSUPPORTED_MODELS,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_LLM_BASE_URL, default=DEFAULT_CONF_LLM_BASE_URL): str,
        vol.Required(CONF_LLM_BASE_URL): str,
        vol.Required(CONF_LLM_API_KEY): str,
        vol.Optional(CONF_RAG_BASE_URL, default=DEFAULT_CONF_RAG_BASE_URL): str,
        vol.Required(CONF_RAG_BASE_URL): str,
        vol.Required(CONF_RAG_API_KEY): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    llm_client = openai.AsyncOpenAI(
        api_key=data[CONF_LLM_API_KEY],
        base_url=data[CONF_LLM_BASE_URL],
    )
    rag_client = openai.AsyncOpenAI(
        api_key=data[CONF_RAG_API_KEY],
        base_url=data[CONF_RAG_BASE_URL],
    )
    await hass.async_add_executor_job(llm_client.with_options(timeout=10.0).models.list)
    await hass.async_add_executor_job(rag_client.with_options(timeout=10.0).models.list)


class OpenAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 1

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "document": DocumentSubentryFlowHandler,
            "example": ExampleSubentryFlowHandler,
        }

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title="ChatGPT",
                data=user_input,
                options=RECOMMENDED_OPTIONS,
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OpenAIOptionsFlow(config_entry)


class OpenAIOptionsFlow(OptionsFlow):
    """OpenAI config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options
        errors: dict[str, str] = {}

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if user_input[CONF_LLM_HASS_API] == "none":
                    user_input.pop(CONF_LLM_HASS_API)

                if user_input.get(CONF_CHAT_MODEL) in UNSUPPORTED_MODELS:
                    errors[CONF_CHAT_MODEL] = "model_not_supported"
                else:
                    return self.async_create_entry(title="", data=user_input)
            else:
                # Re-render the options again, now with the recommended options shown/hidden
                self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

                options = {
                    CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                    CONF_PROMPT: user_input[CONF_PROMPT],
                    CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
                }

        schema = openai_config_option_schema(self.hass, options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
        )


class DocumentSubentryFlowHandler(ConfigSubentryFlow):
    """Handle document subentries."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle user step to add document."""
        errors = {}

        if user_input is not None:
            return self.async_create_entry(
                title=user_input[CONF_TITLE],
                data={
                    CONF_TITLE: user_input[CONF_TITLE],
                    CONF_DESCRIPTION: user_input.get(CONF_DESCRIPTION),
                    CONF_PROMPT: user_input[CONF_PROMPT],
                },
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TITLE): str,
                    vol.Required(CONF_DESCRIPTION): TemplateSelector(),
                    vol.Required(CONF_PROMPT): TemplateSelector(),
                }
            ),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfigure step."""
        current_subentry = self._get_reconfigure_subentry()
        current_title = current_subentry.data.get(CONF_TITLE, "")
        current_description = current_subentry.data.get(CONF_DESCRIPTION, "")
        current_prompt = current_subentry.data.get(CONF_PROMPT, "")

        if user_input is not None:
            return self.async_update_and_abort(
                entry=self._get_entry(),
                subentry=self._get_reconfigure_subentry(),
                data={
                    CONF_TITLE: user_input[CONF_TITLE],
                    CONF_PROMPT: user_input[CONF_PROMPT],
                    CONF_DESCRIPTION: user_input[CONF_DESCRIPTION],
                },
                title=user_input[CONF_TITLE],
            )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TITLE, default=current_title): str,
                    vol.Required(
                        CONF_DESCRIPTION, default=current_description
                    ): TemplateSelector(),
                    vol.Required(
                        CONF_PROMPT, default=current_prompt
                    ): TemplateSelector(),
                }
            ),
        )


class ExampleSubentryFlowHandler(ConfigSubentryFlow):
    """Handle example subentries."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle user step to add example."""
        errors = {}

        if user_input is not None:
            return self.async_create_entry(
                title=user_input[CONF_TITLE],
                data={
                    CONF_TITLE: user_input[CONF_TITLE],
                    CONF_QUESTION: user_input.get(CONF_QUESTION),
                    CONF_TOOL_CALL: user_input.get(CONF_TOOL_CALL),
                    CONF_TOOL_CALL_ARGS: user_input.get(CONF_TOOL_CALL_ARGS),
                    CONF_ANSWER: user_input.get(CONF_ANSWER),
                },
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TITLE): str,
                    vol.Required(CONF_QUESTION): TemplateSelector(),
                    vol.Optional(CONF_TOOL_CALL): TemplateSelector(),
                    vol.Optional(CONF_TOOL_CALL_ARGS): TemplateSelector(),
                    vol.Required(CONF_ANSWER): TemplateSelector(),
                }
            ),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfigure step."""
        current_subentry = self._get_reconfigure_subentry()
        current_title = current_subentry.data.get(CONF_TITLE, "")
        current_question = current_subentry.data.get(CONF_QUESTION, "")
        current_tool_call = current_subentry.data.get(CONF_TOOL_CALL, "")
        current_tool_call_arguments = current_subentry.data.get(CONF_TOOL_CALL_ARGS, "")
        current_answer = current_subentry.data.get(CONF_ANSWER, "")

        if user_input is not None:
            return self.async_update_and_abort(
                entry=self._get_entry(),
                subentry=self._get_reconfigure_subentry(),
                data={
                    CONF_TITLE: user_input[CONF_TITLE],
                    CONF_QUESTION: user_input.get(CONF_QUESTION),
                    CONF_TOOL_CALL: user_input.get(CONF_TOOL_CALL),
                    CONF_TOOL_CALL_ARGS: user_input.get(CONF_TOOL_CALL_ARGS),
                    CONF_ANSWER: user_input.get(CONF_ANSWER),
                },
                title=user_input[CONF_TITLE],
            )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TITLE, default=current_title): str,
                    vol.Required(
                        CONF_QUESTION, default=current_question
                    ): TemplateSelector(),
                    vol.Optional(
                        CONF_TOOL_CALL, default=current_tool_call
                    ): TemplateSelector(),
                    vol.Optional(
                        CONF_TOOL_CALL_ARGS, default=current_tool_call_arguments
                    ): TemplateSelector(),
                    vol.Required(
                        CONF_ANSWER, default=current_answer
                    ): TemplateSelector(),
                }
            ),
        )


def openai_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> VolDictType:
    """Return a schema for OpenAI completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_MAX_MESSAGES,
                description={"suggested_value": options.get(CONF_MAX_MESSAGES)},
                default=10,
            ): int,
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_REASONING_EFFORT,
                description={"suggested_value": options.get(CONF_REASONING_EFFORT)},
                default=RECOMMENDED_REASONING_EFFORT,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=["low", "medium", "high"],
                    translation_key="reasoning_effort",
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_ENABLE_RAG,
                description={"suggested_value": options.get(CONF_ENABLE_RAG)},
                default=False,
            ): BooleanSelector(),
            vol.Optional(
                CONF_EMBEDDING_MODEL,
                description={"suggested_value": options.get(CONF_EMBEDDING_MODEL)},
                default="",
            ): str,
            vol.Optional(
                CONF_RAG_REFRESH_INTERVAL,
                description={"suggested_value": options.get(CONF_RAG_REFRESH_INTERVAL)},
                default=60,
            ): int,
            vol.Optional(
                CONF_RAG_TOP_N,
                description={"suggested_value": options.get(CONF_RAG_TOP_N)},
                default=3,
            ): int,
            vol.Optional(
                CONF_FUZZY_TOP_N,
                description={"suggested_value": options.get(CONF_FUZZY_TOP_N)},
                default=3,
            ): int,
        }
    )
    return schema
