from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


EntityType = Literal[
    "person",
    "place",
    "object",
    "organization",
    "event",
    "concept",
    "unknown",
]

MentionType = Literal[
    "named",
    "pronoun",
    "descriptor",
    "alias",
    "nominal",
    "object",
    "location",
]

MentionKind = Literal[
    "named_entity",
    "role_anchor",
    "descriptive_alias",
    "pronoun",
    "generic_nominal",
    "noise",
]

ResolutionStatus = Literal[
    "confirmed",
    "likely",
    "possible",
    "rejected",
    "unresolved",
]

EntityCreationDecision = Literal[
    "create_entity",
    "link_to_existing",
    "keep_as_mention_only",
    "drop",
]

TerminalAuthority = Literal[
    "heuristic_fallback",
    "entity_anchor_policy",
    "cross_chunk_coreference",
    "human_override",
]

ChunkExtractionStatus = Literal["PENDING", "PROCESSING", "DONE", "FAILED"]


class TextSpan(BaseModel):
    start_char: int
    end_char: int
    text: str


class ChunkNode(BaseModel):
    chunk_id: str
    document_id: str
    index: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    content_hash: str | None = None
    extraction_status: ChunkExtractionStatus = "PENDING"
    extraction_started_at: str | None = None
    extraction_finished_at: str | None = None
    extraction_error: str | None = None
    extraction_attempts: int = 0


class Mention(BaseModel):
    mention_id: str
    chunk_id: str
    text: str
    span: TextSpan
    entity_id: str | None = None
    reference_type: Literal["named", "pronoun", "nominal", "implicit"] = "named"


class RawMention(BaseModel):
    mention_id: str
    chunk_id: str
    text: str
    span: TextSpan
    normalized_text: str | None = None
    mention_type: MentionType = "nominal"
    mention_kind: MentionKind | None = None
    source: str = "gliner_relex"
    extractor_source: Literal["llm", "gliner_relex", "repair", "fallback", "legacy"] = "llm"
    extractor_version: str | None = None
    repaired_by: str | None = None
    repair_notes: list[str] = Field(default_factory=list)
    confidence: float | None = None
    semantic_payload: dict[str, Any] = Field(default_factory=dict)


class RawRelation(BaseModel):
    raw_relation_id: str
    chunk_id: str
    source_mention_id: str
    target_mention_id: str
    relation_span: str
    evidence_span_id: str
    relation_type: str | None = None
    confidence: float | None = None


class EventArgument(BaseModel):
    argument_id: str | None = None
    event_frame_id: str | None = None
    role: str
    mention_id: str
    entity_id: str | None = None
    surface_text: str | None = None
    evidence_span_id: str | None = None
    resolution_status: Literal["resolved", "unresolved", "mention_only"] = "unresolved"
    grounding_expectation: Literal[
        "entity_expected",
        "mention_only_allowed",
        "concept_allowed",
    ] = "entity_expected"
    argument_index: int = 0
    extractor_version: str | None = None
    resolver_version: str | None = None
    confidence: float | None = None


class EventFrame(BaseModel):
    event_frame_id: str
    chunk_id: str
    document_id: str | None = None
    predicate: str
    normalized_predicate: str | None = None
    event_type: str | None = None
    arguments: list[EventArgument] = Field(default_factory=list)
    evidence_span_id: str
    temporal_scope: str | None = None
    modality: str | None = None
    polarity: Literal["positive", "negative", "unknown"] = "positive"
    resolution_status: Literal["complete", "partial", "unresolved"] = "unresolved"
    materialization_status: Literal["valid", "degraded", "rejected"] = "valid"
    source: str = "event_frame_builder"
    extractor_version: str | None = None
    resolver_version: str | None = None
    confidence: float | None = None


class ResolutionFeature(BaseModel):
    name: str
    value: float
    evidence_span_id: str | None = None
    explanation: str | None = None


class ResolutionHypothesis(BaseModel):
    hypothesis_id: str
    mention_id: str
    hypothesis_type: Literal[
        "mention_to_known_entity",
        "mention_to_new_entity",
        "same_entity",
        "unresolved",
    ]
    candidate_entity_id: str | None = None
    candidate_entity_name: str | None = None
    confidence: float
    status: ResolutionStatus
    mention_kind: MentionKind | None = None
    entity_creation_decision: EntityCreationDecision = "keep_as_mention_only"
    final_entity_id: str | None = None
    candidate_entity_ids: list[str] = Field(default_factory=list)
    candidate_scores: dict[str, float] = Field(default_factory=dict)
    evidence_span_id: str | None = None
    previous_decision: EntityCreationDecision | None = None
    decision_stage: str | None = None
    authority: TerminalAuthority = "heuristic_fallback"
    is_terminal: bool = False
    reason: str
    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)
    features: list[ResolutionFeature] = Field(default_factory=list)
    resolution_run_id: str
    resolver_version: str
    policy_version: str
    extractor_version: str | None = None
    model_name: str | None = None
    chunking_version: str | None = None


class TerminalResolution(BaseModel):
    mention_id: str
    decision: EntityCreationDecision
    authority: TerminalAuthority
    confidence: float
    policy_version: str
    created_at_stage: str
    final_entity_id: str | None = None
    revisable_by_higher_authority: bool = True


class Entity(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: EntityType = "unknown"
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class Claim(BaseModel):
    claim_id: str
    chunk_id: str
    text: str
    subject_entity_ids: list[str] = Field(default_factory=list)
    evidence_span: TextSpan | None = None
    confidence: float = 0.0
    needs_verification: bool = False


class Event(BaseModel):
    event_id: str
    chunk_id: str
    event_type: str
    description: str
    participants: list[str] = Field(default_factory=list)
    evidence_span: TextSpan | None = None
    confidence: float = 0.0


class Relation(BaseModel):
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    chunk_id: str
    evidence_span: TextSpan | None = None
    confidence: float = 0.0


class ResolvedEntityRef(BaseModel):
    entity_id: str
    surface_form: str
    reference_type: Literal["named", "pronoun", "nominal", "implicit"]
    role: Literal["source", "target", "subject", "object", "participant", "unclear"]
    confidence: float = 0.0


class EvidenceSpan(BaseModel):
    span_id: str
    original_text: str
    normalized_text: str | None = None
    start_char: int
    end_char: int
    chunk_id: str
    document_id: str
    resolved_entities: list[ResolvedEntityRef] = Field(default_factory=list)


class RelationCandidate(BaseModel):
    relation_candidate_id: str
    source_entity_id: str
    target_entity_id: str
    relation_span: str
    evidence_span_id: str
    direction_hint: str | None = None
    direction_confidence: float = 0.0
    encode_as_latent_edge: bool = True
    symbolic_hint: str | None = None
    confidence: float = 0.0


class ProjectionSpace(BaseModel):
    projection_space_id: str
    name: str
    encoder_model: str
    vector_size: int
    distance_metric: Literal["cosine", "dot", "l2"] = "cosine"
    unit_type: Literal[
        "evidence_span",
        "relation_candidate",
        "entity_pair_context",
        "query_relation_intent",
    ] = "relation_candidate"
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LatentRelationEdge(BaseModel):
    edge_id: str
    source_entity_id: str
    target_entity_id: str
    relation_candidate_id: str
    evidence_span_id: str
    vector_ref: str
    projection_space_id: str
    direction: Literal["directed", "undirected"] = "directed"
    confidence: float = 0.0
    symbolic_hint: str | None = None
    vector: list[float] | None = None


class EntityPair(BaseModel):
    pair_id: str
    entity_a_id: str
    entity_b_id: str
    directed_edge_ids: list[str] = Field(default_factory=list)
    undirected_context_edge_ids: list[str] = Field(default_factory=list)


class DocumentNode(BaseModel):
    document_id: str
    title: str | None = None
    source_path: str | None = None
    metadata: dict = Field(default_factory=dict)


MentionNode = Mention
EntityNode = Entity
ClaimNode = Claim
EventNode = Event
RelationEdge = Relation
EvidenceSpanNode = EvidenceSpan


class LocalGraphPatch(BaseModel):
    chunk: ChunkNode
    raw_mentions: list[RawMention] = Field(default_factory=list)
    raw_relations: list[RawRelation] = Field(default_factory=list)
    event_frames: list[EventFrame] = Field(default_factory=list)
    resolution_hypotheses: list[ResolutionHypothesis] = Field(default_factory=list)
    terminal_resolutions: list[TerminalResolution] = Field(default_factory=list)
    entities: list[EntityNode] = Field(default_factory=list)
    mentions: list[MentionNode] = Field(default_factory=list)
    claims: list[ClaimNode] = Field(default_factory=list)
    events: list[EventNode] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    relation_candidates: list[RelationCandidate] = Field(default_factory=list)
    latent_relation_edges: list[LatentRelationEdge] = Field(default_factory=list)


LocalGraph = LocalGraphPatch
