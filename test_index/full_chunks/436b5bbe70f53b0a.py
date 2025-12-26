def _default_embedding_model() -> str:
    """Return the default embedding model for the active provider."""
    if PROVIDER == "openrouter":
        return OPENROUTER_EMBEDDING_MODEL
    return OLLAMA_EMBEDDING_MODEL