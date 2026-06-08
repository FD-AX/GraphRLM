from pydantic import BaseModel, Field

from app.core.graph_models import EvidenceSpan, RelationCandidate, RelationEdge


class EntityState(BaseModel):
    entity_id: str
    canonical_name: str
    attributes: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class RLMState(BaseModel):
    document_id: str
    current_chunk_index: int = -1
    entities: dict[str, EntityState] = Field(default_factory=dict)
    open_hypotheses: list[str] = Field(default_factory=list)
    recent_chunk_ids: list[str] = Field(default_factory=list)
    recent_evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    relation_candidates: dict[str, RelationCandidate] = Field(default_factory=dict)
    unresolved_references: list[str] = Field(default_factory=list)


class RLMTransition(BaseModel):
    transition_id: str
    document_id: str
    from_chunk_id: str | None = None
    to_chunk_id: str
    from_chunk_index: int | None = None
    to_chunk_index: int

    added_entities: list[EntityState] = Field(default_factory=list)
    updated_entities: list[EntityState] = Field(default_factory=list)
    added_relations: list[RelationEdge] = Field(default_factory=list)
    added_relation_candidates: list[RelationCandidate] = Field(default_factory=list)
    added_evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
