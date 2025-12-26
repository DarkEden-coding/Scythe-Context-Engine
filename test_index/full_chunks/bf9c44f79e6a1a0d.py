def generate_chunk_id(file_path: str, start_line: int, end_line: int) -> str:
    """Generate a unique chunk ID based on file path and line numbers.

    Args:
        file_path: Relative path to the source file.
        start_line: Starting line number of the chunk.
        end_line: Ending line number of the chunk.

    Returns:
        Unique chunk identifier string.
    """
    content = f"{file_path}:{start_line}:{end_line}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]