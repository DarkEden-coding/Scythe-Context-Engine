def extract_chat_content(response: Any) -> Optional[str]:
    """Extract the textual content from a chat completion response."""
    content = _extract_content_from_dict_response(response)
    if content is not None:
        return content
    message = getattr(response, "message", None)
    return _extract_content_from_message(message)