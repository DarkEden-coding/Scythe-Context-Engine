 _score_chunks_with_model(rerank_chunks: List[Dict], query: str) -> List[tuple]:
    """Score chunks using the active provider.

    Args:
        rerank_chunks: Chunks to be scored by the LLM.
        query: The original search query.

    Returns:
        List of tuples containing (score, chunk) pairs.
    """
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
            print("JSON parsing failed for ranking response, using default scores")
            return [(5.0, chunk) for chunk in rerank_chunks]
        scored: List[tuple] = []
        for ranking in ranking_data.rankings:
            chunk_id = ranking.chunk_id
            score = ranking.score
            if 0 <= chunk_id < len(rerank_chunks):
                scored.append((float(score), rerank_chunks[chunk_id]))
                print(f"Chunk {chunk_id}: Score {score}")
        return scored if scored else [(5.0, chunk) for chunk in rerank_chunks]

    except Exception as exc:
        print(f"Ranking failed: {exc}, using default scores")
        return [(5.0, chunk) for chunk in rerank_chunks]


