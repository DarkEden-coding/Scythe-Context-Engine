def load_full_chunk(chunk_id: str, index_prefix: str) -> str:
    """Load the full chunk code from disk.

    Args:
        chunk_id: Unique identifier for the chunk.
        index_prefix: Directory prefix where the index is stored.

    Returns:
        The complete source code of the chunk.
    """
    full_chunks_dir = Path(index_prefix) / "full_chunks"

    # Try all supported language extensions plus .txt as fallback
    extensions_to_try = list(SUPPORTED_LANGS.keys()) + [".txt", ".md"]

    for extension in extensions_to_try:
        chunk_file = full_chunks_dir / f"{chunk_id}{extension}"
        if chunk_file.exists():
            with open(chunk_file, "r", encoding="utf-8") as f:
                return f.read()

    return f"[Chunk {chunk_id} not found]"