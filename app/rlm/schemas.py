from __future__ import annotations

from pydantic import BaseModel, Field


class RLMEntityUpdate(BaseModel):
    entity_id: str
    canonical_name: str
    attributes: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RLMTransitionExtraction(BaseModel):
    """
    Structured output from the LLM-backed RLM update node.

    The LLM does not directly mutate RLMState. It proposes entity-state updates
    and notes; the runtime wraps them into RLMTransition and applies it.
    """

    added_entities: list[RLMEntityUpdate] = Field(default_factory=list)
    updated_entities: list[RLMEntityUpdate] = Field(default_factory=list)
    open_hypotheses: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
