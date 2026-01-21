"""Cache functionality for query context."""

import hashlib
import sys
import time
from typing import Dict, List, Optional

from config.config import CACHE_TTL, cache
from utils.logger import log_event


def check_cache(query: str, top_chunks: List[Dict]) -> Optional[str]:
    """Check semantic cache for prior refined context.

    Args:
        query: The search query string.
        top_chunks: List of top-scoring chunks to include in cache key.

    Returns:
        Cached refined context string if found, None otherwise.
    """
    cache_check_start_time = time.time()

    try:
        # Cache key: hash(query + chunk texts)

        cache_input = query + "|".join([c["text"][:100] for c in top_chunks])

        cache_key = hashlib.sha256(cache_input.encode()).hexdigest()

        cached = cache.get(f"context_cache:{cache_key}")

        cache_duration_ms = (time.time() - cache_check_start_time) * 1000

        if cached:
            print("Cache HIT", file=sys.stderr)
            log_event(
                event="cache_hit",
                level="INFO",
                phase="query",
                component="cache",
                message="Semantic cache hit",
                data={
                    "cache_key": cache_key[:8],
                    "result_length": len(cached),
                },
                duration_ms=cache_duration_ms,
            )
            return cached

        log_event(
            event="cache_miss",
            level="INFO",
            phase="query",
            component="cache",
            message="Semantic cache miss",
            data={
                "cache_key": cache_key[:8],
            },
            duration_ms=cache_duration_ms,
        )
        return None

    except Exception as e:
        cache_duration_ms = (time.time() - cache_check_start_time) * 1000
        log_event(
            event="cache_error",
            level="WARNING",
            phase="query",
            component="cache",
            message=f"Cache check error: {str(e)}",
            data={
                "operation": "check",
            },
            duration_ms=cache_duration_ms,
            error=e,
        )
        return None


def store_cache(query: str, top_chunks: List[Dict], refined: str):
    """Store refined context in cache.

    Args:
        query: The search query string.
        top_chunks: List of top-scoring chunks used to generate the refined context.
        refined: The refined context string to cache.
    """
    cache_store_start_time = time.time()

    try:
        cache_input = query + "|".join([c["text"][:100] for c in top_chunks])

        cache_key = hashlib.sha256(cache_input.encode()).hexdigest()

        cache.set(f"context_cache:{cache_key}", refined, CACHE_TTL)

        cache_store_duration_ms = (time.time() - cache_store_start_time) * 1000
        log_event(
            event="cache_store",
            level="INFO",
            phase="query",
            component="cache",
            message="Refined context stored in cache",
            data={
                "cache_key": cache_key[:8],
                "value_length": len(refined),
                "ttl": CACHE_TTL,
            },
            duration_ms=cache_store_duration_ms,
        )

    except Exception as e:
        cache_store_duration_ms = (time.time() - cache_store_start_time) * 1000
        print(f"Cache store failed: {e}", file=sys.stderr)
        log_event(
            event="cache_error",
            level="WARNING",
            phase="query",
            component="cache",
            message=f"Cache store error: {str(e)}",
            data={
                "operation": "store",
            },
            duration_ms=cache_store_duration_ms,
            error=e,
        )
