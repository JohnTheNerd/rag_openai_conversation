"""Constants for the RAG OpenAI Conversation integration."""

import logging

DOMAIN = "rag_openai_conversation"
LOGGER = logging.getLogger(__package__)

CONF_ENABLE_RAG = "enable_rag"
CONF_RAG_TOP_N = "rag_top_n"
CONF_FUZZY_TOP_N = "fuzzy_top_n"
CONF_MAX_MESSAGES = "max_messages"
CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_DESCRIPTION = "description"
CONF_TITLE = "title"
CONF_QUESTION = "question"
CONF_TOOL_CALL = "tool_call"
CONF_TOOL_CALL_ARGS = "tool_call_args"
CONF_ANSWER = "answer"
CONF_CHAT_MODEL = "chat_model"
CONF_EMBEDDING_MODEL = "embedding_model"
CONF_RAG_REFRESH_INTERVAL = "rag_refresh_interval"
RECOMMENDED_CHAT_MODEL = "gpt-4o-mini"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
CONF_REASONING_EFFORT = "reasoning_effort"
RECOMMENDED_REASONING_EFFORT = "low"

DEFAULT_CONF_LLM_BASE_URL = "https://api.openai.com/v1"
CONF_LLM_BASE_URL = "llm_base_url"
CONF_LLM_API_KEY = "llm_api_key"
DEFAULT_CONF_RAG_BASE_URL = "https://api.openai.com/v1"
CONF_RAG_BASE_URL = "embedding_base_url"
CONF_RAG_API_KEY = "embedding_api_key"

UNSUPPORTED_MODELS = [
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-preview",
    "o1-preview-2024-09-12",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-realtime-preview-2024-12-17",
]
