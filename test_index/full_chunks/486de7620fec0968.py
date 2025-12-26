 _render_context_sections(
    top_chunks: List[Dict], index_prefix: Optional[str] = None
) -> str:
    """Format the top chunks into a combined context string.

    Args:
        top_chunks: List of top-scoring chunks to format.
        index_prefix: Path prefix for the FAISS index files (needed to load full code).

    Returns:
        Formatted context string with file locations and code blocks.
    """
    sections = []
    for chunk in top_chunks:
        metadata = chunk["metadata"]
        level = metadata.get("level")
        if level == "code_chunk":
            chunk_id = metadata.get("chunk_id")
            full_code = ""
            if chunk_id and index_prefix:
                full_code = load_full_chunk(chunk_id, index_prefix)
            else:
                full_code = "[Full code not available]"

            location = (
                f"Function: {metadata.get('function_name', 'unknown')}, "
                f"File: {metadata['file']}, "
                f"Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}, "
                f"Type: {metadata.get('type', 'unknown')}"
            )
            sections.append(f"**{location}**\n```\n{full_code}\n```")
        elif level == "file_summary":
            sections.append(f"**File Summary: {metadata['file']}**\n{chunk['text']}")
        elif level == "folder_summary":
            sections.append(
                f"**Folder: {metadata.get('folder', '?')}**\n{chunk['text']}"
            )
        elif level == "document":
            chunk_id = metadata.get("chunk_id")
            full_content = ""
            if chunk_id and index_prefix:
                full_content = load_full_chunk(chunk_id, index_prefix)
            else:
                full_content = "[Full content not available]"

            sections.append(f"**Document: {metadata['file']}**\n```\n{full_content}\n```")
        else:
            sections.append(f"**{chunk['text']}**")
    return "\n\n---\n\n".join(sections)


