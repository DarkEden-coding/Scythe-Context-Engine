def embed_single(text: str, model: Optional[str] = None) -> List[float]:
    """Generate an embedding for a single text."""
    return embed_texts([text], model)[0]