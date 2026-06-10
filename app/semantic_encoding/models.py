from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


OwnerType = Literal[
    "entity",
    "entity_event",
    "entity_state",
    "entity_pair",
    "pair_interaction",
    "evidence",
    "record_fact",
]


class EncoderConfig(BaseModel):
    projection_dim: int | None = None
    embedding_dim: int | None = None
    interaction_dim: int | None = 64
    interaction_profile_mode: Literal["hashed_features", "contribution"] = "contribution"
    encoder_name: str = "hashing-bert-like"
    encoder_version: str = "graph_semantic_encoder_v1"
    transformer_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"
    max_graph_depth: int = 3
    beam_width: int = 3
    local_frontier_cap: int = 20
    navigation_similarity_weight: float = 0.70
    active_profile_weight: float = 0.0
    structural_confidence_weight: float = 0.15
    evidence_quality_weight: float = 0.10
    novelty_weight: float = 0.05


class GraphSemanticDocument(BaseModel):
    document_id: str
    semantic_document_id: str
    owner_type: OwnerType
    owner_id: str
    source_entity_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    text: str
    structural_features: dict = Field(default_factory=dict)
    projection_version: str
    content_hash: str


class GraphSemanticEmbedding(BaseModel):
    semantic_document_id: str
    embedding: list[float]
    encoder_name: str
    encoder_version: str
    embedding_dim: int
    content_hash: str
    actual_embedding_dim: int | None = None
    encoder_revision: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    pooling_strategy: str | None = None
    serializer_version: str | None = None
    normalize_embeddings: bool | None = None
    query_prefix: str = "query:"
    document_prefix: str = "document:"


class SemanticSearchResult(BaseModel):
    semantic_document_id: str
    owner_type: OwnerType
    owner_id: str
    score: float
    query_similarity: float
    contribution_top_components: list[dict] = Field(default_factory=list)
    source_entity_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    text: str
    score_parts: dict = Field(default_factory=dict)


class LatentTraversalState(BaseModel):
    query_text: str
    original_query_embedding: list[float]
    navigation_embedding: list[float]
    active_interaction_embedding: list[float]
    current_document_id: str
    current_owner_id: str
    visited_document_ids: list[str] = Field(default_factory=list)
    visited_entity_ids: list[str] = Field(default_factory=list)
    visited_event_ids: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    path_transition_ids: list[str] = Field(default_factory=list)
    depth: int = 0
    remaining_budget: int = 0


class LatentGraphTransition(BaseModel):
    transition_id: str
    source_owner_id: str
    target_owner_id: str
    source_document_id: str
    target_document_id: str
    transition_document_id: str
    transition_embedding: list[float]
    evidence_span_ids: list[str] = Field(default_factory=list)
    structural_type: str
    confidence: float = 1.0


class SemanticTraversalTraceStep(BaseModel):
    depth: int
    current_document_id: str
    current_owner_id: str
    selected_transition_id: str | None = None
    selected_document_id: str | None = None
    query_similarity: float
    active_profile_similarity: float
    final_score: float
    score_parts: dict = Field(default_factory=dict)
    contribution_top_components: list[dict] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    stop_reason: str | None = None


class SemanticTraversalTrace(BaseModel):
    query: str
    seed_results: list[SemanticSearchResult] = Field(default_factory=list)
    steps: list[SemanticTraversalTraceStep] = Field(default_factory=list)
    visited_document_ids: list[str] = Field(default_factory=list)
    visited_entity_ids: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    stop_reason: str
