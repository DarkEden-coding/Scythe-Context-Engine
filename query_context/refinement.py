"""Refinement functionality for query context."""

from typing import Dict, List

from config.config import SUMMARIZATION_MODEL, generate_text
from indexer.chunk_storage import load_full_chunk
from .rendering import _render_context_sections
from .reranking import _score_chunks_with_model, _select_rerank_candidates


def _count_words(text: str) -> int:
    """Count words in a text string.

    Args:
        text: The text to count words in.

    Returns:
        Number of words in the text.
    """
    return len(text.split())


def _build_refinement_prompt(
    query: str, top_chunks: List[Dict], index_prefix: str, word_limit: int = 5000
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

    max_chunk_words = word_limit - 1000
    current_word_count = 0
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
        chunk_words = _count_words(chunk_content)
        if current_word_count + chunk_words > max_chunk_words:
            break

        chunk_details.append(chunk_content)
        current_word_count += chunk_words

    available_chunks = "\n\n---\n\n".join(chunk_details)

    return (
        f'Create a comprehensive guide for: "{query}"\n\n'
        f"# OUTPUT LIMIT: Your response MUST NOT exceed {word_limit} words. Be concise and focused.\n\n"
        "# CRITICAL RULES - FOLLOW EXACTLY:\n\n"
        "1. **ANALYZE THE CODE PROVIDED**: You can see the actual implementation code below. Use it to understand:\n"
        "   - How functions call each other\n"
        "   - What imports/dependencies exist\n"
        "   - The actual flow of data and control\n"
        "   - How different files interact\n\n"
        "2. **INCLUDE ALL RELEVANT CHUNKS**: Look for chunks from the same file or related files. If multiple chunks are related (e.g., they call each other or share imports), explain how they work together.\n\n"
        "3. **ONLY USE PROVIDED CHUNKS**: You have access to exactly these chunks below. Do NOT invent, assume, or mention ANY functions, files, or patterns not explicitly shown in the code.\n\n"
        "4. **NO ASSUMPTIONS**: Do NOT assume:\n"
        "   - Architecture patterns unless explicitly shown in imports/code\n"
        "   - Auth mechanisms beyond what you see in the actual code\n"
        "   - State management patterns unless shown in imports/code\n"
        "   - Storage mechanisms unless you see the actual API calls (cookies, localStorage, etc.)\n\n"
        "5. **REFERENCE CHUNKS BY NUMBER**: Use {chunk0}, {chunk1}, etc. to reference code. Each chunk will be automatically expanded with its formatted code and metadata.\n\n"
        "6. **STRUCTURE YOUR RESPONSE**:\n"
        "   - Start with a brief overview of what these chunks show\n"
        "   - Group chunks by file/functionality\n"
        "   - For each chunk, use {chunkN} placeholder and explain:\n"
        "     * What the code does\n"
        "     * What it imports/depends on\n"
        "     * How it relates to other chunks (look for function calls, shared imports)\n"
        "   - Describe the flow between chunks if you can trace it from the code\n\n"
        f"7. **FORMAT**: Use markdown with clear sections. Be thorough and technical but stay within the {word_limit} word limit.\n\n"
        f"## Code Chunks to Analyze:\n\n{available_chunks}\n\n"
        "## Your Response (following all rules above):"
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
    chunks: List[Dict], query: str, index_prefix: str, top_k: int = 5, word_limit: int = 5000
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
    refine_prompt = _build_refinement_prompt(query, top_chunks, index_prefix, word_limit)

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
