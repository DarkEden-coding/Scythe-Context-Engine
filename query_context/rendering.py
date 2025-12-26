"""Rendering functionality for query context."""

from typing import Dict, List, Optional

from indexer.chunk_storage import load_full_chunk


def _load_chunk_content(chunk_id: str, index_prefix: Optional[str]) -> str:
    """Load full chunk content if available."""
    if chunk_id and index_prefix:
        return load_full_chunk(chunk_id, index_prefix)
    return "[Content not available]"


def _render_code_chunk(chunk: Dict, index_prefix: Optional[str]) -> str:
    """Render a code chunk section."""
    metadata = chunk["metadata"]
    full_code = _load_chunk_content(metadata.get("chunk_id"), index_prefix)

    location = (
        f"Function: {metadata.get('function_name', 'unknown')}, "
        f"File: {metadata['file']}, "
        f"Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}, "
        f"Type: {metadata.get('type', 'unknown')}"
    )
    return f"**{location}**\n```\n{full_code}\n```"


def _render_file_summary(chunk: Dict, index_prefix: Optional[str] = None) -> str:
    """Render a file summary section."""
    metadata = chunk["metadata"]
    return f"**File Summary: {metadata['file']}**\n{chunk['text']}"


def _render_folder_summary(chunk: Dict, index_prefix: Optional[str] = None) -> str:
    """Render a folder summary section."""
    metadata = chunk["metadata"]
    return f"**Folder: {metadata.get('folder', '?')}**\n{chunk['text']}"


def _render_document(chunk: Dict, index_prefix: Optional[str]) -> str:
    """Render a document section."""
    metadata = chunk["metadata"]
    full_content = _load_chunk_content(metadata.get("chunk_id"), index_prefix)
    return f"**Document: {metadata['file']}**\n```\n{full_content}\n```"


def _render_context_sections(
    top_chunks: List[Dict], index_prefix: Optional[str] = None
) -> str:
    """Format the top chunks into a combined context string.

    Args:
        top_chunks: List of top-scoring chunks to format.
        index_prefix: Path prefix for the FAISS index files (needed to load full code).

    Returns:
        Formatted context string with file locations and code blocks.
    """
    renderers = {
        "code_chunk": _render_code_chunk,
        "file_summary": _render_file_summary,
        "folder_summary": _render_folder_summary,
        "document": _render_document,
    }

    sections = []
    for chunk in top_chunks:
        level = chunk["metadata"].get("level")
        renderer = renderers.get(level)
        if renderer:
            sections.append(renderer(chunk, index_prefix))
        else:
            sections.append(f"**{chunk['text']}**")

    return "\n\n---\n\n".join(sections)
