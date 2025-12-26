def display_code_chunk(chunk: Dict[str, Any], index: int) -> None:
    """
    Display a code chunk with its metadata.

    Args:
        chunk: Chunk dictionary containing text and metadata
        index: Index number for display
    """
    metadata = chunk.get("metadata", {})
    file_path = metadata.get("file", "unknown")
    start_line = metadata.get("start_line", "?")
    end_line = metadata.get("end_line", "?")
    chunk_type = metadata.get("type", "unknown")

    print(f"\n--- Code Chunk #{index} ---")
    print(f"File: {file_path}")
    print(f"Type: {chunk_type}")
    print(f"Lines: {start_line}-{end_line}")
    print("Content:")
    print("-" * 50)
    print(chunk.get("text", ""))
    print("-" * 50)