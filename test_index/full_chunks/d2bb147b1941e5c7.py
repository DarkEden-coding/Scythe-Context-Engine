def categorize_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize chunks by their level type.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Dictionary grouping chunks by level (code_chunk, file_summary, folder_summary)
    """
    categorized = defaultdict(list)
    for chunk in chunks:
        level = chunk.get("metadata", {}).get("level", "unknown")
        categorized[level].append(chunk)
    return categorized