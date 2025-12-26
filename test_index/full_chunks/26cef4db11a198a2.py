class RankingItem(BaseModel):
    """Single ranked item mapping a chunk id to a score."""

    chunk_id: int
    score: float