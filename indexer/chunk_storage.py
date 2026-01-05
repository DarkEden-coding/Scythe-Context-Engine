"""
Chunk storage utilities for metadata-only RAG.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict

from config.config import SUPPORTED_LANGS


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


def load_full_chunk(chunk_id: str, index_prefix: str) -> str:
    """Load the full chunk code from disk.

    Args:
        chunk_id: Unique identifier for the chunk.
        index_prefix: Directory prefix where the index is stored.

    Returns:
        The complete source code of the chunk with non-ASCII characters stripped.
    """
    full_chunks_dir = Path(index_prefix) / "full_chunks"

    # Try all supported language extensions plus .txt as fallback
    extensions_to_try = list(SUPPORTED_LANGS.keys()) + [".txt", ".md"]

    for extension in extensions_to_try:
        chunk_file = full_chunks_dir / f"{chunk_id}{extension}"
        if chunk_file.exists():
            with open(chunk_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Strip non-ASCII characters to prevent encoding issues
                return ''.join(char for char in content if ord(char) < 128)

    return f"[Chunk {chunk_id} not found]"


def save_metadata_json(metadata: Dict, output_prefix: str):
    """Save metadata mapping to JSON file.

    Args:
        metadata: Dictionary containing chunk metadata.
        output_prefix: Directory prefix for output files.
    """
    metadata_file = Path(output_prefix) / "chunk_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


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
