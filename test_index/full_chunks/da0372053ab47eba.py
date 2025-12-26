 _build_rerank_prompt(rerank_chunks: List[Dict], query: str) -> str:
    """Create the reranking prompt from chunk metadata.

    Args:
        rerank_chunks: Chunks to be reranked by the LLM.
        query: The original search query.

    Returns:
        Formatted prompt string for LLM reranking.
    """
    chunk_sections = []
    for index, chunk in enumerate(rerank_chunks):
        metadata = chunk["metadata"]
        header = f"[Chunk {index}]"
        level = metadata.get("level")
        if level == "code_chunk":
            header += (
                f" Function: {metadata.get('function_name', '?')} "
                f"File: {metadata['file']} "
                f"(lines {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')})"
            )
        elif level == "file_summary":
            header += f" File Summary: {metadata['file']}"
        elif level == "folder_summary":
            header += f" Folder: {metadata.get('folder', '?')}"
        elif level == "document":
            header += f" Document: {metadata['file']}"
        chunk_sections.append(f"{header}\n{chunk['text']}\n")

    combined_chunks = "\n---\n".join(chunk_sections)
    max_index = max(len(rerank_chunks) - 1, 0)
    return (
        f'Rate the relevance (0-10) of each code chunk to the query: "{query}"\n\n'
        f"Chunks to rank (metadata only):\n\n{combined_chunks}\n"
        f'Provide rankings as a list of objects with "chunk_id" (integer; 0-{max_index}) '
        'and "score" (number 0-10).\nReturn ONLY valid JSON matching this schema:\n'
        '{"rankings": [{"chunk_id": 0, "score": 8.5}, {"chunk_id": 1, "score": 3.2}]}'
    )


