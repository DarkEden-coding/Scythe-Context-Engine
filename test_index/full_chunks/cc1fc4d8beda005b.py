def display_chunks(chunks_file_path: str) -> None:
    """
    Load and display chunks from a pickle file in a organized manner.

    Args:
        chunks_file_path: Path to the chunks.pkl file
    """
    chunks = load_chunks(chunks_file_path)

    if not chunks:
        print("No chunks to display.")
        return

    print(f"Loaded {len(chunks)} chunks from {chunks_file_path}")
    print("=" * 80)

    categorized = categorize_chunks(chunks)

    # Display code chunks
    if "code_chunk" in categorized:
        print(f"\n📄 CODE CHUNKS ({len(categorized['code_chunk'])} total)")
        print("=" * 80)
        for index, chunk in enumerate(categorized["code_chunk"], 1):
            display_code_chunk(chunk, index)

    # Display file summaries
    if "file_summary" in categorized:
        print(f"\n📁 FILE SUMMARIES ({len(categorized['file_summary'])} total)")
        print("=" * 80)
        for index, chunk in enumerate(categorized["file_summary"], 1):
            display_file_summary(chunk, index)

    # Display folder summaries
    if "folder_summary" in categorized:
        print(f"\n📂 FOLDER SUMMARIES ({len(categorized['folder_summary'])} total)")
        print("=" * 80)
        for index, chunk in enumerate(categorized["folder_summary"], 1):
            display_folder_summary(chunk, index)

    # Display unknown chunks if any
    if "unknown" in categorized:
        print(f"\n❓ UNKNOWN CHUNKS ({len(categorized['unknown'])} total)")
        print("=" * 80)
        for index, chunk in enumerate(categorized["unknown"], 1):
            print(f"\n--- Unknown Chunk #{index} ---")
            print(f"Metadata: {chunk.get('metadata', {})}")
            print("Content:")
            print("-" * 50)
            print(chunk.get("text", ""))
            print("-" * 50)


if __nam