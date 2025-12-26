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