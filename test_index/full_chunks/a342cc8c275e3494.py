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