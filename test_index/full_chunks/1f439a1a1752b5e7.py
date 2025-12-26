def build_structured_output_format(
    schema: Dict[str, Any], schema_name: str
) -> Optional[Dict[str, Any]]:
    """Return provider-specific structured output configuration."""
    if PROVIDER == "openrouter":
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
    return schema