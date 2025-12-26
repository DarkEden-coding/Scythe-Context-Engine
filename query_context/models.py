"""Pydantic models for query context functionality."""

from pydantic import BaseModel


class RelevanceScore(BaseModel):
    """Relevance score for chunk ranking."""

    score: float


class RankingItem(BaseModel):
    """Single ranked item mapping a chunk id to a score."""

    chunk_id: int
    score: float


class ChunkRanking(BaseModel):
    """Batch ranking of chunks with scores."""

    rankings: list[RankingItem]
