def display_folder_summary(chunk: Dict[str, Any], index: int) -> None:
    """
    Display a folder summary chunk with its metadata.

    Args:
        chunk: Chunk dictionary containing text and metadata
        index: Index number for display
    """
    metadata = chunk.get("metadata", {})
    folder_path = metadata.get("folder", "unknown")

    print(f"\n--- Folder Summary #{index} ---")
    print(f"Folder: {folder_path}")
    print("Summary:")
    print("-" * 50)
    print(chunk.get("text", ""))
    print("-" * 50)