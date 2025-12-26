 query_context(
    query: str,
    index_prefix: str,
    top_k: int = 20,
    output_k: int = 5,
    no_cache: bool = False,
):
    """Main query pipeline for retrieving context from indexed repository.

    Args:
        query: The search query string.
        index_prefix: Path prefix for the FAISS index files.
        top_k: Number of top chunks to retrieve initially.
        output_k: Number of chunks to include in final output.
        no_cache: If True, skip semantic caching.

    Returns:
        Refined context string containing relevant code and information.
    """

    print(f"🔍 Query: {query}")

    # Load index

    print("📂 Loading index...")

    index = faiss.read_index(f"{index_prefix}/index.faiss")

    with open(f"{index_prefix}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(f"{index_prefix}/meta.json", "r") as f:
        meta = json.load(f)

    print(f"📊 Index: {meta['total_chunks']} chunks")

    # Embed query

    print("🔢 Embedding query...")

    query_emb = embed_single(query, model=EMBEDDING_MODEL)

    query_emb = np.array(query_emb, dtype="float32")

    query_emb = query_emb.reshape(
        1, -1
    )  # Reshape to (1, d) for normalization and search
    faiss.normalize_L2(query_emb)

    # Search

    print(f"🎯 Searching (top-{top_k})...")

    scores, indices = index.search(query_emb, top_k)

    # Filter low scores

    results = []

    for i, idx in enumerate(indices[0]):
        if scores[0][i] > 0.5:  # Threshold
            chunk = chunks[idx]

            chunk["score"] = float(scores[0][i])

            results.append(chunk)

    print(f"✅ Found {len(results)} relevant chunks")

    if not results:
        return "No relevant context found."

    # Check cache

    cached = check_cache(query, results[:5]) if not no_cache else None

    if cached:
        return cached

    # Rerank + extract

    print("🤖 Reranking with LLM...")

    refined = rerank_and_extract(results, query, index_prefix, output_k)

    # Cache result

    store_cache(query, results[:5], refined) if not no_cache else None

    return refined


if __name__ == "__ma