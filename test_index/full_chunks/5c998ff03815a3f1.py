class ChunkRanking(BaseModel):
    """Batch ranking of chunks with scores."""

    rankings: List[RankingItem]