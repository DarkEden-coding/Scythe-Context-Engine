def collect_successful_chunks_and_embeddings(
    chunks: List[Dict], embeddings: List
) -> tuple:
    """Filter out chunks that failed to embed and return successful ones.

    Args:
        chunks: List of all chunks.
        embeddings: List of embeddings (some may be None for failed batches).

    Returns:
        Tuple containing (successful_chunks, successful_embeddings).
    """
    missing_count = embeddings.count(None)
    if missing_count > 0:
        print(f"Warning: {missing_count} chunks failed to embed and will be skipped")
        # Filter out chunks that couldn't be embedded
        successful_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        successful_chunks = [chunks[i] for i in successful_indices]
        successful_embeddings = [embeddings[i] for i in successful_indices]
        return successful_chunks, successful_embeddings

    return chunks, embeddings