"""
Data models for the Scythe Context Engine.
"""

from typing import Optional
from pydantic import BaseModel


class FileSummary(BaseModel):
    """Structured summary of a file."""

    summary: str


class FolderSummary(BaseModel):
    """Structured summary of a folder."""

    purpose: str


class FunctionMetadata(BaseModel):
    """Metadata for a single function chunk in metadata-only RAG."""

    chunk_id: str
    function_name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    summary: str
    full_code_path: str
    node_type: str
