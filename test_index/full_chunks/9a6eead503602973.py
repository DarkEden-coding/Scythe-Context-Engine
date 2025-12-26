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