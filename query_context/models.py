"""Pydantic models for query context functionality."""

from pydantic import BaseModel, ConfigDict


class RelevanceScore(BaseModel):
    """Relevance score for chunk ranking."""

    model_config = ConfigDict(extra="forbid")
    score: float


class RankingItem(BaseModel):
    """Single ranked item mapping a chunk id to a score."""

    model_config = ConfigDict(extra="forbid")
    chunk_id: int
    score: float


class ChunkRanking(BaseModel):
    """Batch ranking of chunks with scores."""

    model_config = ConfigDict(extra="forbid")
    rankings: list[RankingItem]
