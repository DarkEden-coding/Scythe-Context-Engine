 _replace_chunk_placeholders(
    response: str, top_chunks: List[Dict], index_prefix: str
) -> str:
    """Replace {chunk0}, {chunk1}, etc. placeholders with actual chunk content.

    Args:
        response: LLM response containing placeholder references.
        top_chunks: List of top-scoring chunks for replacement.
        index_prefix: Path prefix for the FAISS index files (needed to load full code).

    Returns:
        Response string with placeholders replaced by formatted chunk content.
    """
    result = response
    for i, chunk in enumerate(top_chunks):
        placeholder = f"{{chunk{i}}}"
        if placeholder in result:
            formatted_chunk = _render_context_sections([chunk], index_prefix)
            result = result.replace(placeholder, formatted_chunk.strip())
    return result


