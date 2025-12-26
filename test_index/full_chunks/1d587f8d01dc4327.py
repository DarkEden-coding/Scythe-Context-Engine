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