from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ProjectionEventObservation(BaseModel):
    event_id: str
    predicate: str
    entity_role: str
    surface_text: str | None = None
    counterpart_entity_ids: list[str] = Field(default_factory=list)
    unresolved_counterparts: list[str] = Field(default_factory=list)
    temporal_scope: str | None = None
    modality: str | None = None
    polarity: str = "positive"
    evidence_span_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    event_resolution_status: str | None = None
    event_materialization_status: str | None = None
    extractor_version: str | None = None
    resolver_version: str | None = None


class EncodingBlock(BaseModel):
    block_id: str
    block_type: str
    text: str
    evidence_span_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class EntityEncodingInput(BaseModel):
    entity_id: str
    document_id: str
    projection_version: str
    identity_blocks: list[EncodingBlock] = Field(default_factory=list)
    event_blocks: list[EncodingBlock] = Field(default_factory=list)
    state_attribute_blocks: list[EncodingBlock] = Field(default_factory=list)
    temporal_blocks: list[EncodingBlock] = Field(default_factory=list)
    evidence_blocks: list[EncodingBlock] = Field(default_factory=list)

    def as_text(self) -> str:
        blocks = (
            self.identity_blocks
            + self.event_blocks
            + self.state_attribute_blocks
            + self.temporal_blocks
            + self.evidence_blocks
        )
        return "\n".join(block.text for block in blocks if block.text)


class EntityContextSnapshot(BaseModel):
    snapshot_id: str
    entity_id: str
    document_id: str
    canonical_name: str
    alias_surfaces: list[str] = Field(default_factory=list)
    event_observations: list[ProjectionEventObservation] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)
    related_entity_ids: list[str] = Field(default_factory=list)
    unresolved_counterparts: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    projection_version: str
    extractor_versions: list[str] = Field(default_factory=list)
    resolver_versions: list[str] = Field(default_factory=list)
    encoding_input: EntityEncodingInput
    content_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PairEventLink(BaseModel):
    event_id: str
    predicate: str
    source_role: str | None = None
    target_role: str | None = None
    direction: Literal["source_to_target", "target_to_source", "shared"] = "shared"
    evidence_span_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    temporal_scope: str | None = None
    surface_texts: list[str] = Field(default_factory=list)


class EntityPairEncodingInput(BaseModel):
    pair_id: str
    document_id: str
    projection_version: str
    direct_interaction_blocks: list[EncodingBlock] = Field(default_factory=list)
    directional_blocks: list[EncodingBlock] = Field(default_factory=list)
    shared_context_blocks: list[EncodingBlock] = Field(default_factory=list)
    temporal_blocks: list[EncodingBlock] = Field(default_factory=list)
    evidence_blocks: list[EncodingBlock] = Field(default_factory=list)

    def as_text(self) -> str:
        blocks = (
            self.direct_interaction_blocks
            + self.directional_blocks
            + self.shared_context_blocks
            + self.temporal_blocks
            + self.evidence_blocks
        )
        return "\n".join(block.text for block in blocks if block.text)


class EntityPairContextSnapshot(BaseModel):
    pair_id: str
    snapshot_id: str
    document_id: str
    source_entity_id: str
    target_entity_id: str
    direct_shared_events: list[PairEventLink] = Field(default_factory=list)
    source_to_target_events: list[PairEventLink] = Field(default_factory=list)
    target_to_source_events: list[PairEventLink] = Field(default_factory=list)
    shared_locations: list[str] = Field(default_factory=list)
    shared_time_scopes: list[str] = Field(default_factory=list)
    relation_evidence_span_ids: list[str] = Field(default_factory=list)
    source_roles: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    indirect_graph_paths: list[list[str]] = Field(default_factory=list)
    projection_version: str
    encoding_input: EntityPairEncodingInput
    content_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
