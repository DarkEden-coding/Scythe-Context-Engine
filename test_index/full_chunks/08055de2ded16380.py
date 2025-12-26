 _build_refinement_prompt(query: str, top_chunks: List[Dict]) -> str:
    """Build the refinement prompt for extracting essential context.

    Args:
        query: The original search query.
        top_chunks: List of top-scoring chunks for reference.

    Returns:
        Formatted prompt string for LLM context refinement.
    """

    chunk_references = []
    for i, chunk in enumerate(top_chunks):
        metadata = chunk["metadata"]
        level = metadata.get("level")
        if level == "code_chunk":
            ref = f"{{chunk{i}}}: {metadata['file']} lines {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}"
        elif level == "file_summary":
            ref = f"{{chunk{i}}}: File summary for {metadata['file']}"
        elif level == "folder_summary":
            ref = f"{{chunk{i}}}: Folder summary for {metadata.get('folder', '?')}"
        elif level == "document":
            ref = f"{{chunk{i}}}: Document {metadata['file']}"
        else:
            ref = f"{{chunk{i}}}: {chunk['text'][:50]}..."
        chunk_references.append(ref)

    available_chunks = "\n".join(chunk_references)

    return (
        f'Extract ONLY the essential code/context needed for: "{query}"\n\n'
        "Make sure to include:\n"
        "- For every code snippet, include the exact file path and line numbers\n"
        "- Include relevant file/folder summaries when helpful\n"
        "- Call out key functions, classes, or patterns with their locations\n\n"
        "IMPORTANT: Instead of copying code directly, reference chunks using placeholders like {chunk0}, {chunk1}, etc.\n"
        "Each placeholder will be automatically replaced with the full code snippet and metadata.\n\n"
        "Make sure to return output in markdown format.\n\n"
        "Do not make any code change recommendations or suggestions, only provide context to a model down the line that will make the code changes.\n\n"
        f"Available chunks:\n{available_chunks}\n\n"
        "Essential context (concise):"
    )


