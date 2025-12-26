def save_metadata_json(metadata: Dict, output_prefix: str):
    """Save metadata mapping to JSON file.

    Args:
        metadata: Dictionary containing chunk metadata.
        output_prefix: Directory prefix for output files.
    """
    metadata_file = Path(output_prefix) / "chunk_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)