from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


FactLensAuditMode = Literal[
    "flat_subclaim_verification",
    "graph_projection_verification",
    "graph_query_verification",
    "graph_shared_evidence",
    "graph_shared_evidence_without_cross_subclaim_edges",
    "graph_shared_evidence_shuffled_edges",
    "graph_shared_evidence_masked_required_fact",
    "graph_recursive_completeness",
]


class FactLensSubclaim(BaseModel):
    subclaim_id: str
    text: str
    evidence_span_ids: list[str] = Field(default_factory=list)
    graph_fact_ids: list[str] = Field(default_factory=list)
    verification_modes: list[FactLensAuditMode] = Field(default_factory=list)


class FactLensGraphEdge(BaseModel):
    edge_id: str
    source_subclaim_id: str
    target_subclaim_id: str
    relation: str
    graph_fact_id: str
    evidence_span_ids: list[str] = Field(default_factory=list)


class FactLensAuditResult(BaseModel):
    mode: FactLensAuditMode
    claim_id: str
    supported: bool
    final_verdict_accuracy: float
    subclaims_total: int
    subclaims_with_evidence: int
    subclaim_recall: float
    all_required_subclaims_verified: bool
    unsupported_subclaim_rate: float
    unsupported_verdict_rate: float
    shared_graph_facts: int
    reused_evidence_count: int
    shared_evidence_reuse: int
    cross_subclaim_edges: int
    contradictions_found: int
    complete_evidence_coverage: bool
    required_path_found: bool
    answer_derived_from_graph_path: bool
    tokens_per_verified_subclaim: float | None = None
    tokens_per_fully_verified_claim: float | None = None
    fully_verified_claims_per_10k_tokens: float = 0.0
    graph_contributed: bool
    graph_contribution_outcome: Literal["useful", "neutral", "misleading", "insufficient"]
    evidence_span_ids: list[str] = Field(default_factory=list)
    graph_fact_ids: list[str] = Field(default_factory=list)
    graph_edge_ids: list[str] = Field(default_factory=list)
    subclaim_coverage: list[dict] = Field(default_factory=list)
    shared_evidence_trace: list[dict] = Field(default_factory=list)
