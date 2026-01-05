"""Refinement functionality for query context."""

import tiktoken
from typing import Dict, List

from config.config import SUMMARIZATION_MODEL, generate_text
from indexer.chunk_storage import load_full_chunk
from .rendering import _render_context_sections
from .reranking import _score_chunks_with_model, _select_rerank_candidates


def _count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken.

    Args:
        text: The text to count tokens in.

    Returns:
        Number of tokens in the text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _build_refinement_prompt(
    query: str, top_chunks: List[Dict], index_prefix: str, token_limit: int = 15000
) -> str:
    """Build the refinement prompt for extracting essential context.

    Args:
        query: The original search query.
        top_chunks: List of top-scoring chunks for reference.
        index_prefix: Path prefix for loading full chunk code.
        word_limit: Maximum word count for the final output.

    Returns:
        Formatted prompt string for LLM context refinement.
    """

    max_chunk_tokens = token_limit - 1000
    current_token_count = 0
    chunk_details = []

    for i, chunk in enumerate(top_chunks):
        metadata = chunk["metadata"]
        level = metadata.get("level")

        # Build chunk header
        chunk_content = ""
        if level == "code_chunk":
            header = f"[Chunk {i}] {metadata['file']} lines {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}"
            func_name = metadata.get("function_name", "unknown")
            if func_name != "unknown":
                header += f" | Function: {func_name}"

            # Load the actual code
            chunk_id = metadata.get("chunk_id")
            if chunk_id:
                code = load_full_chunk(chunk_id, index_prefix)
                # Truncate very long chunks to keep prompt manageable
                if len(code) > 1500:
                    code = code[:1500] + "\n... (truncated)"
                chunk_content = f"{header}\n```\n{code}\n```"
            else:
                chunk_content = f"{header}\n(Code not available)"

        elif level == "file_summary":
            header = f"[Chunk {i}] File Summary: {metadata['file']}"
            chunk_content = f"{header}\n{chunk['text']}"
        elif level == "folder_summary":
            header = f"[Chunk {i}] Folder: {metadata.get('folder', '?')}"
            chunk_content = f"{header}\n{chunk['text']}"
        elif level == "document":
            header = f"[Chunk {i}] Document: {metadata['file']}"
            chunk_id = metadata.get("chunk_id")
            if chunk_id:
                content = load_full_chunk(chunk_id, index_prefix)
                if len(content) > 1500:
                    content = content[:1500] + "\n... (truncated)"
                chunk_content = f"{header}\n```\n{content}\n```"
            else:
                chunk_content = f"{header}\n{chunk['text'][:500]}"
        else:
            chunk_content = f"[Chunk {i}] {chunk['text'][:200]}..."

        # Check if adding this chunk would exceed the limit
        chunk_tokens = _count_tokens(chunk_content)
        if current_token_count + chunk_tokens > max_chunk_tokens:
            break

        chunk_details.append(chunk_content)
        current_token_count += chunk_tokens

    available_chunks = "\n\n---\n\n".join(chunk_details)

    return (
        f'Provide a summary for: "{query}"\n\n'
        "First, write one paragraph about the overall structure of the provided code chunks and what a next LLM would need to know to understand or continue working with this code.\n\n"
        "Then, for each code chunk, provide a small description of what that chunk is (1-2 sentences).\n\n"
        "Format your response exactly as:\n"
        "Overall Summary:\n[paragraph]\n\n"
        "Chunk Descriptions:\n"
        "Chunk 0: [description]\n"
        "Chunk 1: [description]\n"
        "...\n\n"
        "Do not include any actual code in your response.\n\n"
        f"## Code Chunks to Analyze:\n\n{available_chunks}\n\n"
        "Your Response:"
    )


def rerank_and_extract(
    chunks: List[Dict], query: str, index_prefix: str, top_k: int = 5, token_limit: int = 15000
) -> str:
    """Use LLM to rerank chunks and extract exact context.

    Args:
        chunks: All retrieved chunks from the initial search.
        query: The search query.
        index_prefix: Path prefix for the FAISS index files (needed to load full code).
        top_k: Number of top chunks to include in final context.
        word_limit: Maximum word count for the final output.

    Returns:
        Refined context string with essential information extracted by LLM.
    """

    rerank_chunks = _select_rerank_candidates(chunks)
    scored_chunks = _score_chunks_with_model(rerank_chunks, query)
    scored_chunks.sort(reverse=True, key=lambda item: item[0])
    top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
    if not top_chunks:
        return "query invalid, no related chunks found\nTry with a different query. Or search for context yourself"
    refine_prompt = _build_refinement_prompt(query, top_chunks, index_prefix, token_limit)

    try:
        refined_text = generate_text(
            refine_prompt,
            model=SUMMARIZATION_MODEL,
            options={"temperature": 0.2},
        )
        rendered_chunks = _render_context_sections(top_chunks, index_prefix)
        combined = refined_text.strip() + "\n\n" + rendered_chunks
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = len(encoding.encode(combined))
        if tokens > token_limit:
            # Reduce chunks until under limit
            for i in range(len(top_chunks) - 1, -1, -1):
                reduced_chunks = top_chunks[:i + 1]
                rendered_reduced = _render_context_sections(reduced_chunks, index_prefix)
                combined_reduced = refined_text.strip() + "\n\n" + rendered_reduced
                tokens_reduced = len(encoding.encode(combined_reduced))
                if tokens_reduced <= token_limit:
                    return combined_reduced
            # If even one chunk is over, return just the summary
            return refined_text.strip()
        return combined

    except Exception:
        return _render_context_sections(top_chunks, index_prefix)
