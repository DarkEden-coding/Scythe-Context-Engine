def _default_chat_model() -> str:
    """Return the default chat model for the active provider."""
    if PROVIDER == "openrouter":
        return OPENROUTER_CHAT_MODEL
    return OLLAMA_SUMMARIZATION_MODEL