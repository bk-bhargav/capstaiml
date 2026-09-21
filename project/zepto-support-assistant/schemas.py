"""Pydantic request/response models -- the structured-output contract."""
from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's question")


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(
        default_factory=list,
        description="IDs of the chunks used; empty for general_question answers",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
