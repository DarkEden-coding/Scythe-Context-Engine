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