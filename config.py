"""
Shared configuration for Scythe Context Engine.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Literal

from ollama import Client
from cache import Cache
from openrouter_client import OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
ProviderType = Literal["openrouter", "ollama"]


def _load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"Configuration file '{CONFIG_FILE}' not found. "
            "Run 'python create_config.py' to create it."
        )
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file '{CONFIG_FILE}': {e}")


_config = _load_config()

# Cache
cache = Cache()
CACHE_TTL = _config["cache"]["ttl_seconds"]

# Provider Constants
PROVIDER: ProviderType = _config["provider"]

# Client Initialization
ollama_client = Client()
_openrouter_client: Optional[OpenRouterClient] = None
if _config["openrouter"]["api_key"]:
    _openrouter_client = OpenRouterClient(
        api_key=_config["openrouter"]["api_key"],
        api_base=_config["openrouter"]["api_base"],
        timeout_seconds=_config["openrouter"]["timeout_seconds"],
    )


def _get_openrouter_options(
    base_options: Optional[Dict[str, Any]], mode: Literal["chat", "embedding"]
) -> Dict[str, Any]:
    """Helper to inject provider whitelists into OpenRouter requests."""
    options = (base_options or {}).copy()
    key = f"{mode}_provider_whitelist"
    whitelist = _config["openrouter"].get(key)

    if whitelist:
        options.setdefault("provider", {})["only"] = whitelist
    return options


def _require_openrouter() -> OpenRouterClient:
    if _openrouter_client is None:
        raise ValueError("OpenRouter API key missing in config.")
    return _openrouter_client


def embed_texts(texts: Sequence[str], model: Optional[str] = None) -> List[List[float]]:
    chosen_model = model or _default_embedding_model()

    if PROVIDER == "openrouter":
        client = _require_openrouter()
        options = _get_openrouter_options(None, "embedding")
        try:
            return client.embed_texts(list(texts), chosen_model, options=options)
        except OpenRouterError as exc:
            logger.error("OpenRouter embeddings failed: %s", exc)
            raise

    # Ollama branch
    # Note: Using required_permissions=['network'] if calling real API
    response = ollama_client.embed(model=chosen_model, input=list(texts))
    if "embeddings" not in response:
        raise ValueError("Ollama response missing 'embeddings' key.")
    return response["embeddings"]


def embed_single(text: str, model: Optional[str] = None) -> List[float]:
    """Generate an embedding for a single text."""
    return embed_texts([text], model)[0]


def chat_completion(
    messages: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Any:
    chosen_model = model or _default_chat_model()

    if PROVIDER == "openrouter":
        client = _require_openrouter()
        req_options = _get_openrouter_options(options, "chat")
        try:
            return client.chat_completion(
                messages=list(messages),
                model=chosen_model,
                response_format=response_format,
                options=req_options,
            )
        except OpenRouterError as exc:
            logger.error("OpenRouter chat completion failed: %s", exc)
            raise

    return ollama_client.chat(
        model=chosen_model,
        messages=list(messages),
        format=response_format,
        options=options or {},
    )


def generate_text(
    prompt: str,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    chosen_model = model or _default_chat_model()

    if PROVIDER == "openrouter":
        client = _require_openrouter()
        req_options = _get_openrouter_options(options, "chat")
        try:
            return client.generate_text(prompt, chosen_model, options=req_options)
        except OpenRouterError as exc:
            logger.error("OpenRouter text generation failed: %s", exc)
            raise

    response = ollama_client.generate(
        model=chosen_model, prompt=prompt, options=options or {}
    )
    content = response.get("response")
    if not isinstance(content, str):
        raise ValueError("Ollama response missing text content.")
    return content


def _default_chat_model() -> str:
    return (
        _config["openrouter"]["chat_model"]
        if PROVIDER == "openrouter"
        else _config["ollama"]["summarization_model"]
    )


def _default_embedding_model() -> str:
    return (
        _config["openrouter"]["embedding_model"]
        if PROVIDER == "openrouter"
        else _config["ollama"]["embedding_model"]
    )


def build_structured_output_format(
    schema: Dict[str, Any], schema_name: str
) -> Optional[Dict[str, Any]]:
    if PROVIDER == "openrouter":
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
    # Ollama accepts the raw schema for its 'format' parameter
    return schema


def extract_chat_content(response: Any) -> Optional[str]:
    # Extract from Dict (OpenRouter/OpenAI style)
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content")

    # Extract from Object (Ollama SDK style)
    message = getattr(response, "message", None)
    if message:
        return getattr(message, "content", None)

    return None


# Exported Constants
SUMMARIZATION_MODEL = _default_chat_model()
EMBEDDING_MODEL = _default_embedding_model()
SUPPORTED_LANGS = _config["indexing"]["supported_languages"]
IGNORED_DIRS = set(_config["indexing"]["ignored_dirs"])
IGNORED_FILES = set(_config["indexing"]["ignored_files"])
