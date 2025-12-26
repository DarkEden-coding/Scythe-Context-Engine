"""
Shared configuration for Scythe Context Engine.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from ollama import Client

from cache import Cache
from openrouter_client import OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)

# Cache configuration
cache = Cache()

# Cache TTL in seconds (1 hour)
CACHE_TTL = 3600 * 24

# Provider configuration
PROVIDER = "openrouter"

# Ollama configuration
ollama_client = Client()
OLLAMA_SUMMARIZATION_MODEL = "gemma3:1b"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# OpenRouter configuration
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = (
    "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"
)
OPENROUTER_CHAT_MODEL = "openai/gpt-oss-20b"
OPENROUTER_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
OPENROUTER_TIMEOUT_SECONDS = 60
OPENROUTER_chat_PROVIDER_WHITELIST = [
    "hyperbolic"
]  # List of allowed provider slugs, e.g., ["openai", "anthropic"]
OPENROUTER_embedding_PROVIDER_WHITELIST = [
    "nebius"
]  # List of allowed provider slugs, e.g., ["openai", "anthropic"]

_openrouter_client: Optional[OpenRouterClient] = None
if OPENROUTER_API_KEY:
    _openrouter_client = OpenRouterClient(
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_API_BASE,
        timeout_seconds=OPENROUTER_TIMEOUT_SECONDS,
    )


def embed_texts(texts: Sequence[str], model: Optional[str] = None) -> List[List[float]]:
    """Generate embeddings for a collection of texts."""
    chosen_model = model or _default_embedding_model()
    if PROVIDER == "openrouter":
        client = _require_openrouter_client()
        # Prepare options with provider whitelist if configured
        request_options = {}
        if OPENROUTER_embedding_PROVIDER_WHITELIST:
            request_options.setdefault("provider", {})
            request_options["provider"]["only"] = (
                OPENROUTER_embedding_PROVIDER_WHITELIST
            )
        try:
            return client.embed_texts(
                list(texts), chosen_model, options=request_options
            )
        except OpenRouterError as exc:
            logger.error("OpenRouter embeddings failed: %s", exc)
            raise
    response = ollama_client.embed(model=chosen_model, input=list(texts))
    embeddings = response.get("embeddings")
    if embeddings is None:
        raise ValueError("Ollama embedding response missing embeddings key.")
    return embeddings


def embed_single(text: str, model: Optional[str] = None) -> List[float]:
    """Generate an embedding for a single text."""
    return embed_texts([text], model)[0]


def chat_completion(
    messages: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Any:
    """Send a chat completion request to the active provider."""
    chosen_model = model or _default_chat_model()
    if PROVIDER == "openrouter":
        return _chat_completion_openrouter(
            messages, chosen_model, response_format, options
        )
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
    """Generate text using the active provider."""
    chosen_model = model or _default_chat_model()
    if PROVIDER == "openrouter":
        client = _require_openrouter_client()
        # Prepare options with provider whitelist if configured
        request_options = options.copy() if options else {}
        if OPENROUTER_chat_PROVIDER_WHITELIST:
            request_options.setdefault("provider", {})
            request_options["provider"]["only"] = OPENROUTER_chat_PROVIDER_WHITELIST
        try:
            return client.generate_text(prompt, chosen_model, options=request_options)
        except OpenRouterError as exc:
            logger.error("OpenRouter text generation failed: %s", exc)
            raise
    response = ollama_client.generate(
        model=chosen_model, prompt=prompt, options=options or {}
    )
    content = response.get("response")
    if not isinstance(content, str):
        raise ValueError("Ollama text generation response missing content.")
    return content


def _default_chat_model() -> str:
    """Return the default chat model for the active provider."""
    if PROVIDER == "openrouter":
        return OPENROUTER_CHAT_MODEL
    return OLLAMA_SUMMARIZATION_MODEL


def _default_embedding_model() -> str:
    """Return the default embedding model for the active provider."""
    if PROVIDER == "openrouter":
        return OPENROUTER_EMBEDDING_MODEL
    return OLLAMA_EMBEDDING_MODEL


def _require_openrouter_client() -> OpenRouterClient:
    """Return the configured OpenRouter client or raise if unavailable."""
    if _openrouter_client is None:
        raise ValueError("OpenRouter API key is required when provider is openrouter.")
    return _openrouter_client


def _chat_completion_openrouter(
    messages: Sequence[Dict[str, Any]],
    model: str,
    response_format: Optional[Dict[str, Any]],
    options: Optional[Dict[str, Any]],
) -> Any:
    """Execute an OpenRouter chat completion with error handling."""
    client = _require_openrouter_client()
    # Prepare options with provider whitelist if configured
    request_options = options.copy() if options else {}
    if OPENROUTER_chat_PROVIDER_WHITELIST:
        request_options.setdefault("provider", {})
        request_options["provider"]["only"] = OPENROUTER_chat_PROVIDER_WHITELIST

    try:
        return client.chat_completion(
            messages=list(messages),
            model=model,
            response_format=response_format,
            options=request_options,
        )
    except OpenRouterError as exc:
        logger.error("OpenRouter chat completion failed: %s", exc)
        raise


def _extract_content_from_dict_response(response: Any) -> Optional[str]:
    """Extract message content when the provider returns a dict."""
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            return _extract_content_from_message(choices[0].get("message"))
    return None


def _extract_content_from_message(message: Any) -> Optional[str]:
    """Extract the content field from a chat message structure."""
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else None
    content_attr = getattr(message, "content", None)
    return content_attr if isinstance(content_attr, str) else None


def build_structured_output_format(
    schema: Dict[str, Any], schema_name: str
) -> Optional[Dict[str, Any]]:
    """Return provider-specific structured output configuration."""
    if PROVIDER == "openrouter":
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
    return schema


def extract_chat_content(response: Any) -> Optional[str]:
    """Extract the textual content from a chat completion response."""
    content = _extract_content_from_dict_response(response)
    if content is not None:
        return content
    message = getattr(response, "message", None)
    return _extract_content_from_message(message)


SUMMARIZATION_MODEL = _default_chat_model()
EMBEDDING_MODEL = _default_embedding_model()


# Supported programming languages and document types for indexing
SUPPORTED_LANGS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
    ".md": "markdown",
}

# Ignored directories during indexing (skip these when walking the repo)
IGNORED_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv", "static", ".cudavenv", "build"}

# Ignored files during indexing (skip files matching these patterns)
IGNORED_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "Gemfile.lock",
    "Cargo.lock",
}
