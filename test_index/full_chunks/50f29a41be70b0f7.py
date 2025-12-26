def embed_batch_with_retry(batch_idx: int, batch_texts: List[str]) -> tuple:
    """Embed a single batch with retry logic.

    Args:
        batch_idx: Index of the batch being processed.
        batch_texts: List of text strings to embed.

    Returns:
        Tuple containing (batch_idx, embeddings, success) where embeddings is None if failed.
    """
    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            batch_embs = embed_texts(batch_texts, model=EMBEDDING_MODEL)
            return batch_idx, batch_embs, True  # Success
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"Embedding batch {batch_idx} failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(retry_delay * (2**attempt))  # Exponential backoff
            else:
                print(
                    f"Embedding batch {batch_idx} failed permanently after {max_retries} attempts, skipping: {e}"
                )
                return batch_idx, None, False  # Failure - skip this batch

    # This should never be reached, but satisfies type checker
    return batch_idx, None, False