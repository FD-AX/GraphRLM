from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel, Field


class GraphViewRef(BaseModel):
    document_id: str
    graph_version: str
    projection_version: str
    encoder_version: str


class SourceChunk(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    start_char: int = 0
    end_char: int | None = None


class TextEvidenceSpan(BaseModel):
    evidence_span_id: str
    document_id: str
    chunk_id: str
    start_char: int
    end_char: int
    text: str


class RetrievalArmResult(BaseModel):
    arm: Literal["graph_rlm", "text_rlm"]
    answer_candidate: str | None = None
    evidence_span_ids: list[str] = Field(default_factory=list)
    consulted_object_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    completeness: float = 0.0
    contradictions: list[str] = Field(default_factory=list)
    stop_reason: str
    trace_id: str
    trace: list[dict] = Field(default_factory=list)
    model_call_traces: list["ModelCallTrace"] = Field(default_factory=list)


class AnswerArbitrationResult(BaseModel):
    answer_candidate: str | None
    support_status: Literal[
        "both_supported",
        "graph_supported_text_unconfirmed",
        "text_supported_graph_missing",
        "contradiction",
        "insufficient_evidence",
    ]
    confidence: float
    evidence_span_ids: list[str] = Field(default_factory=list)
    selected_arm: Literal["graph_rlm", "text_rlm", "hybrid", "none"]
    rationale: str


class DualRLMResult(BaseModel):
    run_id: str
    query: str
    graph_view: GraphViewRef
    graph_result: RetrievalArmResult
    text_result: RetrievalArmResult
    arbitration: AnswerArbitrationResult
    graph_mutation_allowed: bool = False


class GraphRLMState(TypedDict):
    original_query: str
    query: str
    graph_view: GraphViewRef
    current_subquery: str
    current_owner_id: str | None
    current_document_ids: list[str]
    frontier_transition_ids: list[str]
    presented_frontier: list[dict]
    presented_evidence_ids: list[str]
    visited_owner_ids: list[str]
    visited_document_ids: list[str]
    visited_transition_ids: list[str]
    collected_evidence_ids: list[str]
    path: list[str]
    depth: int
    max_depth: int
    remaining_model_calls: int
    remaining_expansions: int
    last_decision: dict | None
    model_call_traces: list[dict]
    final_answer: str | None
    stop_reason: str | None
    # Backwards-compatible aliases used by earlier static arms/tests.
    current_owner_ids: list[str]
    semantic_document_ids: list[str]
    frontier_ids: list[str]
    graph_paths: list[list[str]]
    evidence_span_ids: list[str]
    recursion_depth: int
    remaining_calls: int


class TextRLMState(TypedDict):
    query: str
    document_id: str
    current_subquery: str
    retrieved_chunk_ids: list[str]
    evidence_span_ids: list[str]
    working_memory: list[str]
    visited_chunk_ids: list[str]
    recursion_depth: int
    remaining_calls: int
    stop_reason: str | None


class GraphRLMDecision(BaseModel):
    action: Literal[
        "inspect",
        "expand",
        "reformulate",
        "verify",
        "answer",
        "stop",
    ]
    selected_transition_ids: list[str] = Field(default_factory=list)
    subquery: str | None = None
    evidence_sufficient: bool = False
    evidence_gap: str | None = None
    confidence: float = 0.0
    decision_summary: str


class TextRLMResult(BaseModel):
    status: Literal[
        "evidence_found",
        "insufficient_evidence",
        "contradiction_found",
    ]
    answer_summary: str
    consulted_chunk_ids: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    entity_mentions: list[str] = Field(default_factory=list)
    event_candidates: list[dict] = Field(default_factory=list)
    relation_candidates: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    unresolved_questions: list[str] = Field(default_factory=list)


class DualRLMConfig(BaseModel):
    graph_top_k: int = 5
    text_top_k: int = 5
    text_window_radius: int = 1
    max_text_rounds: int = 2
    max_graph_rounds: int = 2
    max_graph_depth: int = 3
    max_graph_model_calls: int = 8
    max_graph_expansions: int = 4
    min_confidence: float = 0.25


class RLMGateway(Protocol):
    def decide_graph(self, state: GraphRLMState) -> GraphRLMDecision:
        ...

    def inspect_text(self, state: TextRLMState, chunks: list[SourceChunk]) -> TextRLMResult:
        ...

    @property
    def model_call_traces(self) -> list["ModelCallTrace"]:
        ...


class RetrievalArm(Protocol):
    def run(self, query: str, run_id: str) -> RetrievalArmResult:
        ...


class ModelCallTrace(BaseModel):
    call_id: str
    provider: str
    model_name: str
    model_role: str | None = None
    reasoning_effort: str | None = None
    purpose: str
    request_timestamp: datetime
    response_timestamp: datetime
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    response_id: str | None = None
    validated_output_type: str
    fallback_used: bool = False
    prompt_section_estimates: dict[str, dict[str, int]] = Field(default_factory=dict)
    prompt_section_hashes: dict[str, str] = Field(default_factory=dict)
