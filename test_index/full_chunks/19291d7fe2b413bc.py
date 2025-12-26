def save_full_chunk(
    chunk_id: str, full_code: str, output_prefix: str, extension: str = ".txt"
):
    """Save the full chunk code to disk.

    Args:
        chunk_id: Unique identifier for the chunk.
        full_code: The complete source code of the chunk.
        output_prefix: Directory prefix for output files.
        extension: File extension for the saved chunk.
    """
    full_chunks_dir = Path(output_prefix) / "full_chunks"
    full_chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_file = full_chunks_dir / f"{chunk_id}{extension}"
    with open(chunk_file, "w", encoding="utf-8") as f:
        f.write(full_code)