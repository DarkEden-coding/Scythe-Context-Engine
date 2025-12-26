 rerank_and_extract(
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


