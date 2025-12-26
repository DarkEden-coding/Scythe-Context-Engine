 store_cache(query: str, top_chunks: List[Dict], refined: str):
    """Store refined context in cache.

    Args:
        query: The search query string.
        top_chunks: List of top-scoring chunks used to generate the refined context.
        refined: The refined context string to cache.
    """

    try:
        cache_input = query + "|".join([c["text"][:100] for c in top_chunks])

        cache_key = hashlib.sha256(cache_input.encode()).hexdigest()

        cache.set(f"context_cache:{cache_key}", refined, CACHE_TTL)

    except Exception as e:
        print(f"Cache store failed: {e}")


