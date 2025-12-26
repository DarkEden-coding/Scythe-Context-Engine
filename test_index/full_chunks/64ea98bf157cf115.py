def create_faiss_index(chunks: List[Dict]) -> tuple:
    """Create FAISS index from embedded chunks using multithreading.

    Args:
        chunks: List of chunk dictionaries to embed and index.

    Returns:
        Tuple containing (faiss_index, embedding_dimension).
    """
    texts = [c["text"] for c in chunks]

    # Batch embed with nomic
    batch_size = 32
    embeddings = [None] * len(texts)  # Pre-allocate to maintain order

    # Prepare batches with indices
    batches = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batches.append((i // batch_size, batch_texts))

    # Thread-safe embeddings list
    embeddings_lock = threading.Lock()

    def collect_embedding_result(future):
        """Collect results from completed embedding futures."""
        batch_idx, batch_embs, success = future.result()
        if success and batch_embs is not None:
            start_idx = batch_idx * batch_size
            with embeddings_lock:
                for j, emb in enumerate(batch_embs):
                    embeddings[start_idx + j] = emb

    # Process batches with 32 threads
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(embed_batch_with_retry, batch_idx, batch_texts)
            for batch_idx, batch_texts in batches
        ]

        # Use tqdm to track progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            collect_embedding_result(future)

    # Handle missing embeddings (failed batches)
    chunks, embeddings = collect_successful_chunks_and_embeddings(chunks, embeddings)

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

    # Build FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # type: ignore  # Inner product = cosine (normalized)
    index.add(embeddings)  # type: ignore

    return index, d