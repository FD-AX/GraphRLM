from __future__ import annotations

from pydantic import BaseModel, Field


class EntityContext(BaseModel):
    entity_id: str
    canonical_name: str
    descriptions: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    vector: list[float] | None = None

    def as_text(self) -> str:
        parts = [
            f"entity: {self.canonical_name}",
            *self.descriptions,
            *self.attributes,
            *self.states,
        ]
        return "\n".join(part for part in parts if part)


class Observation(BaseModel):
    observation_id: str
    entity_id: str
    text: str
    observation_type: str = "evidence"
    normalized_predicate: str | None = None
    object_entity_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    chunk_id: str | None = None
    temporal_scope: str | None = None
    confidence: float = 1.0
    embedding_version: str | None = None
    vector: list[float] | None = None


class EntityEventObservation(BaseModel):
    entity_id: str
    event_id: str
    entity_role: str
    normalized_predicate: str
    counterpart_entity_ids: list[str] = Field(default_factory=list)
    unresolved_counterparts: list[str] = Field(default_factory=list)
    temporal_scope: str | None = None
    modality: str | None = None
    polarity: str = "positive"
    evidence_span_ids: list[str] = Field(default_factory=list)
    event_resolution_status: str
    event_materialization_status: str
    extractor_version: str | None = None
    resolver_version: str | None = None


class GraphTraversalCandidate(BaseModel):
    transition_id: str | None = None
    source_entity_id: str
    target_entity_id: str
    relation_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    local_context: str
    path_context: str = ""
    depth: int
    structural_confidence: float = 1.0
    vector: list[float] | None = None

    def as_text(self) -> str:
        return "\n".join(
            part
            for part in [
                self.path_context,
                f"transition: {self.source_entity_id} -> {self.target_entity_id}",
                self.local_context,
            ]
            if part
        )


class PathState(BaseModel):
    path_embedding: list[float] | None = None
    covered_entities: list[str] = Field(default_factory=list)
    covered_events: list[str] = Field(default_factory=list)
    accumulated_evidence: list[str] = Field(default_factory=list)
    unresolved_query_facets: list[str] = Field(default_factory=list)


class RankedItem(BaseModel):
    item_id: str
    score: float
    item: EntityContext | Observation | GraphTraversalCandidate
    score_parts: dict[str, float] = Field(default_factory=dict)


class TraversalPath(BaseModel):
    entity_ids: list[str]
    transitions: list[GraphTraversalCandidate] = Field(default_factory=list)
    state: PathState = Field(default_factory=PathState)
    score: float = 0.0

    @property
    def last_entity_id(self) -> str:
        return self.entity_ids[-1]

    def as_text(self) -> str:
        if not self.transitions:
            return " -> ".join(self.entity_ids)
        return "\n".join(transition.local_context for transition in self.transitions)

    def signature(self) -> tuple:
        return (
            tuple(sorted(set(self.entity_ids))),
            tuple(sorted(set(self.state.covered_events))),
            tuple(sorted(set(self.state.accumulated_evidence))),
        )


class RetrievalResult(BaseModel):
    query: str
    seed_entities: list[RankedItem]
    observations: list[RankedItem]
    paths: list[TraversalPath]
    evidence_span_ids: list[str]
