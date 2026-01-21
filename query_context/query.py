"""Main query functionality for retrieving context from indexed repository."""

import json
import pickle
import time

import faiss
import numpy as np

from config.config import EMBEDDING_MODEL, embed_single
from utils.logger import log_event
from .cache import check_cache, store_cache
from .refinement import rerank_and_extract


def query_context(
    query: str,
    index_prefix: str,
    top_k: int = 20,
    output_k: int = 5,
    no_cache: bool = False,
    token_limit: int = 15000,
    quiet: bool = False,
):
    """Main query pipeline for retrieving context from indexed repository.

    Args:
        query: The search query string.
        index_prefix: Path prefix for the FAISS index files.
        top_k: Number of top chunks to retrieve initially.
        output_k: Number of chunks to include in final output.
        no_cache: If True, skip semantic caching.
        token_limit: Maximum token count for the final output.
        quiet: If True, suppress progress output.

    Returns:
        Refined context string containing relevant code and information.
    """

    print(f"Query: {query}")

    # Load index
    index_load_start_time = time.time()

    if not quiet:
        print("Loading index...")

    log_event(
        event="index_load_start",
        level="INFO",
        phase="query",
        component="query_context",
        message="Loading index",
        data={"index_prefix": index_prefix},
    )

    index = faiss.read_index(f"{index_prefix}/index.faiss")

    with open(f"{index_prefix}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Strip non-ASCII characters from all chunks to prevent encoding issues
    def strip_non_ascii(text):
        if isinstance(text, str):
            return ''.join(char for char in text if ord(char) < 128)
        return text

    for chunk in chunks:
        if isinstance(chunk, dict):
            # Strip from text
            if 'text' in chunk:
                chunk['text'] = strip_non_ascii(chunk['text'])
            # Strip from metadata fields
            if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                for key, value in chunk['metadata'].items():
                    if isinstance(value, str):
                        chunk['metadata'][key] = strip_non_ascii(value)

    with open(f"{index_prefix}/meta.json", "r") as f:
        meta = json.load(f)

    print(f"Index: {meta['total_chunks']} chunks")

    index_load_duration_ms = (time.time() - index_load_start_time) * 1000
    log_event(
        event="index_loaded",
        level="INFO",
        phase="query",
        component="query_context",
        message="Index loaded successfully",
        data={
            "total_chunks": meta['total_chunks'],
            "embedding_dim": meta.get('embedding_dim', 0),
            "model": meta.get('model', 'unknown'),
        },
        duration_ms=index_load_duration_ms,
    )

    # Embed query
    query_embed_start_time = time.time()

    if not quiet:
        print("Embedding query...")

    log_event(
        event="query_embedding_start",
        level="INFO",
        phase="query",
        component="query_context",
        message="Starting query embedding",
        data={
            "query_length": len(query),
            "model": EMBEDDING_MODEL,
        },
    )

    query_emb = embed_single(query, model=EMBEDDING_MODEL)

    query_emb = np.array(query_emb, dtype="float32")

    query_emb = query_emb.reshape(
        1, -1
    )  # Reshape to (1, d) for normalization and search
    faiss.normalize_L2(query_emb)

    query_embed_duration_ms = (time.time() - query_embed_start_time) * 1000
    log_event(
        event="query_embedded",
        level="INFO",
        phase="query",
        component="query_context",
        message="Query embedding completed",
        data={
            "embedding_dim": query_emb.shape[1],
        },
        duration_ms=query_embed_duration_ms,
    )

    # Search
    search_start_time = time.time()

    if not quiet:
        print(f"Searching (top-{top_k})...")

    scores, indices = index.search(query_emb, top_k)

    # Filter low scores
    results = []

    for i, idx in enumerate(indices[0]):
        # Relaxed threshold to allow more potential matches for the reranker
        if scores[0][i] > 0.3:
            chunk = chunks[idx]
            chunk["score"] = float(scores[0][i])
            results.append(chunk)

    print(f"Found {len(results)} relevant chunks")

    search_duration_ms = (time.time() - search_start_time) * 1000
    # Extract actual scores from the filtered results
    score_values = [float(result.get("score", 0)) for result in results] if results else []
    log_event(
        event="faiss_search",
        level="INFO",
        phase="query",
        component="query_context",
        message="FAISS search completed",
        data={
            "top_k": top_k,
            "results_found": len(results),
            "min_score": min(score_values) if score_values else 0,
            "max_score": max(score_values) if score_values else 0,
            "avg_score": sum(score_values) / len(score_values) if score_values else 0,
        },
        duration_ms=search_duration_ms,
    )

    if not results:
        log_event(
            event="no_results",
            level="INFO",
            phase="query",
            component="query_context",
            message="No relevant chunks found",
            data={"threshold": 0.3},
        )
        return "No relevant context found."

    # Check cache
    cache_check_start_time = time.time()

    cached = check_cache(query, results[:5]) if not no_cache else None

    cache_check_duration_ms = (time.time() - cache_check_start_time) * 1000
    if cached:
        log_event(
            event="cache_hit",
            level="INFO",
            phase="query",
            component="query_context",
            message="Semantic cache hit",
            data={
                "result_length": len(cached),
            },
            duration_ms=cache_check_duration_ms,
        )
        return cached
    else:
        log_event(
            event="cache_miss",
            level="INFO",
            phase="query",
            component="query_context",
            message="Semantic cache miss",
            duration_ms=cache_check_duration_ms,
        )

    # Rerank + extract

    if not quiet:
        print("Reranking with LLM...")

    refined = rerank_and_extract(results, query, index_prefix, output_k, token_limit=token_limit)

    # Cache result

    store_cache(query, results[:5], refined) if not no_cache else None

    return refined
