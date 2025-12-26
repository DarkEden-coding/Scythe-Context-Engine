def load_metadata_json(index_prefix: str) -> Dict:
    """Load metadata mapping from JSON file.

    Args:
        index_prefix: Directory prefix where the index is stored.

    Returns:
        Dictionary containing chunk metadata.
    """
    metadata_file = Path(index_prefix) / "chunk_metadata.json"

    if not metadata_file.exists():
        return {}

    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)