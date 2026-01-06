"""Cache functionality for query context."""

import hashlib
import sys
from typing import Dict, List, Optional

from config.config import CACHE_TTL, cache


def check_cache(query: str, top_chunks: List[Dict]) -> Optional[str]:
    """Check semantic cache for prior refined context.

    Args:
        query: The search query string.
        top_chunks: List of top-scoring chunks to include in cache key.

    Returns:
        Cached refined context string if found, None otherwise.
    """

    try:
        # Cache key: hash(query + chunk texts)

        cache_input = query + "|".join([c["text"][:100] for c in top_chunks])

        cache_key = hashlib.sha256(cache_input.encode()).hexdigest()

        cached = cache.get(f"context_cache:{cache_key}")

        if cached:
            print("Cache HIT", file=sys.stderr)

            return cached

        return None

    except Exception:
        return None


def store_cache(query: str, top_chunks: List[Dict], refined: str):
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
        print(f"Cache store failed: {e}", file=sys.stderr)
