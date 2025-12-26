def load_chunks(chunks_file_path: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a pickle file.

    Args:
        chunks_file_path: Path to the chunks.pkl file

    Returns:
        List of chunk dictionaries containing 'text' and 'metadata' keys
    """
    try:
        with open(chunks_file_path, "rb") as file_handle:
            chunks = pickle.load(file_handle)
        return chunks
    except FileNotFoundError:
        print(f"Error: File '{chunks_file_path}' not found.")
        return []
    except Exception as error:
        print(f"Error loading chunks: {error}")
        return []