def display_file_summary(chunk: Dict[str, Any], index: int) -> None:
    """
    Display a file summary chunk with its metadata.

    Args:
        chunk: Chunk dictionary containing text and metadata
        index: Index number for display
    """
    metadata = chunk.get("metadata", {})
    file_path = metadata.get("file", "unknown")

    print(f"\n--- File Summary #{index} ---")
    print(f"File: {file_path}")
    print("Summary:")
    print("-" * 50)
    print(chunk.get("text", ""))
    print("-" * 50)