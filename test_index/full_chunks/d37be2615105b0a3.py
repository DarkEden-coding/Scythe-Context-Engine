def _extract_content_from_dict_response(response: Any) -> Optional[str]:
    """Extract message content when the provider returns a dict."""
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            return _extract_content_from_message(choices[0].get("message"))
    return None