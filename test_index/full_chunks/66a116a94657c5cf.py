def _require_openrouter_client() -> OpenRouterClient:
    """Return the configured OpenRouter client or raise if unavailable."""
    if _openrouter_client is None:
        raise ValueError("OpenRouter API key is required when provider is openrouter.")
    return _openrouter_client