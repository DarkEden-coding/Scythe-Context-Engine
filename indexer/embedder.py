"""
Embedding and FAISS index creation.
"""

import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from tqdm import tqdm

from config.config import EMBEDDING_MODEL, embed_texts
from utils.logger import log_event


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
                    f"Embedding batch {batch_idx} failed (attempt {attempt + 1}/{max_retries}): {e}",
                    file=sys.stderr
                )
                time.sleep(retry_delay * (2**attempt))  # Exponential backoff
            else:
                print(
                    f"Embedding batch {batch_idx} failed permanently after {max_retries} attempts, skipping: {e}",
                    file=sys.stderr
                )
                return batch_idx, None, False  # Failure - skip this batch

    # This should never be reached, but satisfies type checker
    return batch_idx, None, False


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
        print(f"Warning: {missing_count} chunks failed to embed and will be skipped", file=sys.stderr)
        # Filter out chunks that couldn't be embedded
        successful_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        successful_chunks = [chunks[i] for i in successful_indices]
        successful_embeddings = [embeddings[i] for i in successful_indices]
        return successful_chunks, successful_embeddings

    return chunks, embeddings


def create_faiss_index(chunks: List[Dict], quiet: bool = False) -> tuple:
    """Create FAISS index from embedded chunks using multithreading.

    Args:
        chunks: List of chunk dictionaries to embed and index.
        quiet: If True, suppress progress bars.

    Returns:
        Tuple containing (faiss_index, embedding_dimension).
    """
    embedding_start_time = time.time()

    # Identify chunks that need embedding
    chunks_to_embed = [c for c in chunks if "embedding" not in c]
    texts_to_embed = [c["text"] for c in chunks_to_embed]

    # Log start of embedding phase
    log_event(
        event="embedding_phase_start",
        level="INFO",
        phase="indexing",
        component="embedder",
        message="Starting embedding phase",
        data={
            "total_chunks": len(chunks),
            "chunks_to_embed": len(chunks_to_embed),
            "model": EMBEDDING_MODEL,
        },
    )

    if texts_to_embed:
        # Batch embed with nomic
        batch_size = 32
        new_embeddings = [None] * len(texts_to_embed)  # Pre-allocate to maintain order

        # Prepare batches with indices
        batches = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i : i + batch_size]
            batches.append((i // batch_size, batch_texts))

        # Log batch start
        log_event(
            event="embedding_batch_start",
            level="INFO",
            phase="indexing",
            component="embedder",
            message="Starting embedding batches",
            data={
                "batch_size": batch_size,
                "total_batches": len(batches),
            },
        )

        # Thread-safe embeddings list
        embeddings_lock = threading.Lock()

        def collect_embedding_result(future):
            """Collect results from completed embedding futures."""
            batch_idx, batch_embs, success = future.result()
            if success and batch_embs is not None:
                start_idx = batch_idx * batch_size
                with embeddings_lock:
                    for j, emb in enumerate(batch_embs):
                        new_embeddings[start_idx + j] = emb

        # Process batches with 32 threads
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(embed_batch_with_retry, batch_idx, batch_texts)
                for batch_idx, batch_texts in batches
            ]

            # Use tqdm to track progress (unless quiet mode)
            if quiet:
                for future in as_completed(futures):
                    collect_embedding_result(future)
            else:
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Embedding"
                ):
                    collect_embedding_result(future)

        # Assign new embeddings to chunks
        for i, chunk in enumerate(chunks_to_embed):
            if new_embeddings[i] is not None:
                chunk["embedding"] = new_embeddings[i]

        embedding_batch_duration_ms = (time.time() - embedding_start_time) * 1000
        log_event(
            event="embedding_batch_complete",
            level="INFO",
            phase="indexing",
            component="embedder",
            message="Completed embedding batches",
            data={
                "total_batches": len(batches),
                "embeddings_generated": len([e for e in new_embeddings if e is not None]),
            },
            duration_ms=embedding_batch_duration_ms,
        )

    # Filter out chunks that still don't have embeddings (failed to embed)
    successful_chunks = [c for c in chunks if "embedding" in c]
    missing_count = len(chunks) - len(successful_chunks)
    if missing_count > 0:
        print(f"Warning: {missing_count} chunks failed to embed and will be skipped", file=sys.stderr)
        log_event(
            event="embedding_skipped_chunks",
            level="WARNING",
            phase="indexing",
            component="embedder",
            message=f"{missing_count} chunks failed to embed",
            data={
                "skipped_chunks": missing_count,
                "successful_chunks": len(successful_chunks),
            },
        )

    # Update the chunks list in place if possible, or return the successful ones
    # For FAISS, we need the array of all successful embeddings
    all_embeddings = np.array([c["embedding"] for c in successful_chunks]).astype(
        "float32"
    )
    faiss.normalize_L2(all_embeddings)  # Normalize for cosine similarity

    # Build FAISS index
    d = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # type: ignore  # Inner product = cosine (normalized)
    index.add(all_embeddings)  # type: ignore

    embedding_total_duration_ms = (time.time() - embedding_start_time) * 1000
    log_event(
        event="embedding_phase_complete",
        level="INFO",
        phase="indexing",
        component="embedder",
        message="Embedding phase completed",
        data={
            "embedding_dimension": d,
            "indexed_chunks": len(successful_chunks),
        },
        duration_ms=embedding_total_duration_ms,
    )

    # Return the potentially filtered list of chunks as well
    return index, d, successful_chunks


def save_index(
    index, chunks: List[Dict], repo_path: str, output_prefix: str, embedding_dim: int
):
    """Save the FAISS index and metadata.

    Args:
        index: FAISS index object to save.
        chunks: List of chunk dictionaries.
        repo_path: Original repository path.
        output_prefix: Directory prefix for output files.
        embedding_dim: Dimension of the embeddings.
    """
    save_start_time = time.time()

    # Log index save start
    log_event(
        event="index_save_start",
        level="INFO",
        phase="indexing",
        component="embedder",
        message="Starting index save",
        data={
            "output_prefix": str(output_prefix),
            "chunks_count": len(chunks),
            "embedding_dim": embedding_dim,
        },
    )

    os.makedirs(output_prefix, exist_ok=True)

    # Save FAISS index
    index_file = f"{output_prefix}/index.faiss"
    faiss.write_index(index, index_file)  # type: ignore

    # Save chunks
    chunks_file = f"{output_prefix}/chunks.pkl"
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)

    # Save metadata
    meta_file = f"{output_prefix}/meta.json"
    with open(meta_file, "w") as f:
        json.dump(
            {
                "repo_path": repo_path,
                "total_chunks": len(chunks),
                "embedding_dim": embedding_dim,
                "model": "nomic-embed-text-v1.5",
            },
            f,
            indent=2,
        )

    # Calculate index size
    try:
        index_size_mb = Path(index_file).stat().st_size / (1024 * 1024)
    except Exception:
        index_size_mb = 0

    save_duration_ms = (time.time() - save_start_time) * 1000
    log_event(
        event="index_saved",
        level="INFO",
        phase="indexing",
        component="embedder",
        message="Index saved successfully",
        data={
            "total_chunks": len(chunks),
            "embedding_dim": embedding_dim,
            "index_size_mb": round(index_size_mb, 2),
            "output_prefix": str(output_prefix),
        },
        duration_ms=save_duration_ms,
    )
