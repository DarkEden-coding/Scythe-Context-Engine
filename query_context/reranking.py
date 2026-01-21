"""Reranking functionality for query context."""

import sys
import time
from typing import Dict, List

from config.config import (
    SUMMARIZATION_MODEL,
    build_structured_output_format,
    chat_completion,
    extract_chat_content,
)
from utils.logger import log_event
from .models import ChunkRanking


def _select_rerank_candidates(chunks: List[Dict]) -> List[Dict]:
    """Select the subset of chunks to rerank with the language model.

    Args:
        chunks: All retrieved chunks from the initial search.

    Returns:
        First 25 chunks to be reranked by the LLM.
    """
    return chunks[:25]


def _build_rerank_prompt(rerank_chunks: List[Dict], query: str) -> str:
    """Create the reranking prompt from chunk metadata.

    Args:
        rerank_chunks: Chunks to be reranked by the LLM.
        query: The original search query.

    Returns:
        Formatted prompt string for LLM reranking.
    """
    chunk_sections = []
    for index, chunk in enumerate(rerank_chunks):
        metadata = chunk["metadata"]
        header = f"[Chunk {index}]"
        level = metadata.get("level")
        if level == "code_chunk":
            header += (
                f" Function: {metadata.get('function_name', '?')} "
                f"File: {metadata['file']} "
                f"(lines {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')})"
            )
        elif level == "file_summary":
            header += f" File Summary: {metadata['file']}"
        elif level == "folder_summary":
            header += f" Folder: {metadata.get('folder', '?')}"
        elif level == "document":
            header += f" Document: {metadata['file']}"
        chunk_sections.append(f"{header}\n{chunk['text']}\n")

    combined_chunks = "\n---\n".join(chunk_sections)
    max_index = max(len(rerank_chunks) - 1, 0)
    return (
        f'Rate the relevance (0-10) of each chunk to the query: "{query}"\n\n'
        "Score based on:\n"
        "- How directly the function/file relates to the query topic\n"
        "- Whether the metadata (function name, file path, documentation) matches the query\n"
        "- Use your judgment, but do not infer what the code does beyond what's stated\n\n"
        f"Chunks to rank:\n\n{combined_chunks}\n"
        f'Provide rankings as JSON with "chunk_id" (integer; 0-{max_index}) and "score" (number 0-10).\n'
        "Return ONLY valid JSON:\n"
        '{"rankings": [{"chunk_id": 0, "score": 8.5}, {"chunk_id": 1, "score": 3.2}]}'
    )


def _score_chunks_with_model(rerank_chunks: List[Dict], query: str) -> List[tuple]:
    """Score chunks using the active provider.

    Args:
        rerank_chunks: Chunks to be scored by the LLM.
        query: The original search query.

    Returns:
        List of tuples containing (score, chunk) pairs.
    """
    reranking_start_time = time.time()

    log_event(
        event="reranking_start",
        level="INFO",
        phase="reranking",
        component="reranking",
        message="Starting chunk reranking",
        data={
            "chunks_to_rerank": len(rerank_chunks),
            "query_length": len(query),
            "model": SUMMARIZATION_MODEL,
        },
    )

    prompt = _build_rerank_prompt(rerank_chunks, query)
    try:
        response_format = build_structured_output_format(
            ChunkRanking.model_json_schema(), schema_name="chunk_ranking"
        )
        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=SUMMARIZATION_MODEL,
            response_format=response_format,
            options={"temperature": 0.1},
        )
        content = extract_chat_content(response)
        if not content:
            return [(5.0, chunk) for chunk in rerank_chunks]
        try:
            ranking_data = ChunkRanking.model_validate_json(content)
        except Exception:
            print("JSON parsing failed for ranking response, using default scores", file=sys.stderr)
            reranking_duration_ms = (time.time() - reranking_start_time) * 1000
            log_event(
                event="reranking_error",
                level="WARNING",
                phase="reranking",
                component="reranking",
                message="JSON parsing failed for ranking response",
                duration_ms=reranking_duration_ms,
            )
            return [(5.0, chunk) for chunk in rerank_chunks]
        scored: List[tuple] = []
        for ranking in ranking_data.rankings:
            chunk_id = ranking.chunk_id
            score = ranking.score
            if 0 <= chunk_id < len(rerank_chunks):
                scored.append((float(score), rerank_chunks[chunk_id]))
                print(f"Chunk {chunk_id}: Score {score}", file=sys.stderr)

        reranking_duration_ms = (time.time() - reranking_start_time) * 1000
        scores = [s[0] for s in scored] if scored else []
        log_event(
            event="chunks_scored",
            level="INFO",
            phase="reranking",
            component="reranking",
            message="Chunks scored successfully",
            data={
                "chunks_scored": len(scored),
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "avg_score": sum(scores) / len(scores) if scores else 0,
            },
            duration_ms=reranking_duration_ms,
        )
        return scored if scored else [(5.0, chunk) for chunk in rerank_chunks]

    except Exception as exc:
        reranking_duration_ms = (time.time() - reranking_start_time) * 1000
        print(f"Ranking failed: {exc}, using default scores", file=sys.stderr)
        log_event(
            event="reranking_error",
            level="ERROR",
            phase="reranking",
            component="reranking",
            message=f"Reranking failed: {str(exc)}",
            duration_ms=reranking_duration_ms,
            error=exc,
        )
        return [(5.0, chunk) for chunk in rerank_chunks]
