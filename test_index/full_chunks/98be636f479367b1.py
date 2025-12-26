def _extract_content_from_message(message: Any) -> Optional[str]:
    """Extract the content field from a chat message structure."""
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else None
    content_attr = getattr(message, "content", None)
    return content_attr if isinstance(content_attr, str) else None