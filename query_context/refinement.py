"""Refinement functionality for query context."""

from typing import Dict, List

from config import SUMMARIZATION_MODEL, generate_text
from .rendering import _render_context_sections
from .reranking import _score_chunks_with_model, _select_rerank_candidates


def _build_refinement_prompt(query: str, top_chunks: List[Dict]) -> str:
    """Build the refinement prompt for extracting essential context.

    Args:
        query: The original search query.
        top_chunks: List of top-scoring chunks for reference.

    Returns:
        Formatted prompt string for LLM context refinement.
    """

    chunk_references = []
    for i, chunk in enumerate(top_chunks):
        metadata = chunk["metadata"]
        level = metadata.get("level")
        if level == "code_chunk":
            ref = f"{{chunk{i}}}: {metadata['file']} lines {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}"
        elif level == "file_summary":
            ref = f"{{chunk{i}}}: File summary for {metadata['file']}"
        elif level == "folder_summary":
            ref = f"{{chunk{i}}}: Folder summary for {metadata.get('folder', '?')}"
        elif level == "document":
            ref = f"{{chunk{i}}}: Document {metadata['file']}"
        else:
            ref = f"{{chunk{i}}}: {chunk['text'][:50]}..."
        chunk_references.append(ref)

    available_chunks = "\n".join(chunk_references)

    return (
        f'Extract ONLY the essential code/context needed for: "{query}"\n\n'
        "Make sure to include:\n"
        "- For every code snippet, include the exact file path and line numbers\n"
        "- Include relevant file/folder summaries when helpful\n"
        "- Call out key functions, classes, or patterns with their locations\n\n"
        "IMPORTANT: Instead of copying code directly, reference chunks using placeholders like {chunk0}, {chunk1}, etc.\n"
        "Each placeholder will be automatically replaced with the full code snippet and metadata.\n\n"
        "Make sure to return output in markdown format.\n\n"
        "Do not make any code change recommendations or suggestions, only provide context to a model down the line that will make the code changes.\n\n"
        f"Available chunks:\n{available_chunks}\n\n"
        "Essential context (concise):"
    )


def _replace_chunk_placeholders(
    response: str, top_chunks: List[Dict], index_prefix: str
) -> str:
    """Replace {chunk0}, {chunk1}, etc. placeholders with actual chunk content.

    Args:
        response: LLM response containing placeholder references.
        top_chunks: List of top-scoring chunks for replacement.
        index_prefix: Path prefix for the FAISS index files (needed to load full code).

    Returns:
        Response string with placeholders replaced by formatted chunk content.
    """
    result = response
    for i, chunk in enumerate(top_chunks):
        placeholder = f"{{chunk{i}}}"
        if placeholder in result:
            formatted_chunk = _render_context_sections([chunk], index_prefix)
            result = result.replace(placeholder, formatted_chunk.strip())
    return result


def rerank_and_extract(
    chunks: List[Dict], query: str, index_prefix: str, top_k: int = 5
) -> str:
    """Use LLM to rerank chunks and extract exact context.

    Args:
        chunks: All retrieved chunks from the initial search.
        query: The search query.
        index_prefix: Path prefix for the FAISS index files (needed to load full code).
        top_k: Number of top chunks to include in final context.

    Returns:
        Refined context string with essential information extracted by LLM.
    """

    rerank_chunks = _select_rerank_candidates(chunks)
    scored_chunks = _score_chunks_with_model(rerank_chunks, query)
    scored_chunks.sort(reverse=True, key=lambda item: item[0])
    top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
    refine_prompt = _build_refinement_prompt(query, top_chunks)

    try:
        refined_text = generate_text(
            refine_prompt,
            model=SUMMARIZATION_MODEL,
            options={"temperature": 0.2},
        )
        final_text = _replace_chunk_placeholders(
            refined_text.strip(), top_chunks, index_prefix
        )
        return final_text

    except Exception:
        return _render_context_sections(top_chunks, index_prefix)
