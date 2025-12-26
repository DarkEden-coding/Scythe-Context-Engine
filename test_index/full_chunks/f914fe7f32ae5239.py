 _select_rerank_candidates(chunks: List[Dict]) -> List[Dict]:
    """Select the subset of chunks to rerank with the language model.

    Args:
        chunks: All retrieved chunks from the initial search.

    Returns:
        First 15 chunks to be reranked by the LLM.
    """
    return chunks[:15]


