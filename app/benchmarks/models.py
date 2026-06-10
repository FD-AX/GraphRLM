from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


BenchmarkName = Literal["s_niah", "oolong", "oolong_pairs", "factlens", "musique"]
BenchmarkArmName = Literal[
    "direct_model",
    "flat_subclaim_verification",
    "graph_projection_verification",
    "graph_query_verification",
    "graph_shared_evidence",
    "graph_shared_evidence_without_cross_subclaim_edges",
    "graph_shared_evidence_shuffled_edges",
    "graph_shared_evidence_masked_required_fact",
    "graph_recursive_completeness",
    "scripted_single_search",
    "scripted_iterative_search",
    "model_graph_guided_rlm",
    "gold_search_goal_oracle",
    "generic_model_prompt",
    "contract_model_prompts",
    "rag",
    "raw_records_ops",
    "graph_rlm",
    "graph_rlm_hash_frontier",
    "graph_rlm_spam_worker_materialized",
    "graph_rlm_semantic_graph",
    "graph_rlm_semantic_graph_ops",
    "gold_fixture",
    "wrong_fixture",
    "musique_keyword",
    "musique_dense_topk",
    "musique_cross_encoder",
    "musique_mdr_iterative",
    "musique_mdr_pooled",
    "musique_text_rlm",
    "musique_graph_navigator",
    "musique_graph_navigator_active",
    "musique_graph_rlm",
]


class BenchmarkCase(BaseModel):
    benchmark: BenchmarkName
    benchmark_id: str
    dataset_origin: Literal["official", "local_generated", "fixture", "unknown"] = "unknown"
    task_id: str
    context: str
    question: str
    gold_answer: str | list[str]
    gold_evidence_span_ids: list[str] = Field(default_factory=list)
    gold_entities: list[str] = Field(default_factory=list)
    expected_hops: int | None = None
    answerable: bool = True
    context_tokens: int = 0
    benchmark_context_len: int | None = None
    measured_context_tokens: int = 0
    tokenizer_id: str = "whitespace"
    metadata: dict = Field(default_factory=dict)

    def arm_view(self) -> "BenchmarkCase":
        metadata = dict(self.metadata)
        native_fields = dict(metadata.get("native_fields") or {})
        native_fields.pop("context_window_text_with_labels", None)
        native_fields.pop("answer", None)
        metadata["native_fields"] = native_fields
        metadata.pop("source_record", None)
        return self.model_copy(update={"metadata": metadata})


class BenchmarkArmResult(BaseModel):
    prediction: str
    raw_response: str | None = None
    provider: str | None = None
    model_name: str | None = None
    model_role: str | None = None
    reasoning_effort: str | None = None
    worker_model_name: str | None = None
    worker_reasoning_effort: str | None = None
    experiment_id: str | None = None
    response_id: str | None = None
    total_tokens: int = 0
    fallback_used: bool = False
    error: str | None = None
    prompt_id: str | None = None
    arm_input_hash: str | None = None
    graph_source_hash: str | None = None
    evidence_span_ids: list[str] = Field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    stop_reason: str = "complete"
    trace_id: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    model_call_traces: list[dict[str, Any]] = Field(default_factory=list)


class BenchmarkScore(BaseModel):
    score_backend: str
    score_name: str
    score_value: float | None
    is_official_score: bool = False
    metadata: dict = Field(default_factory=dict)


class BenchmarkRunRecord(BaseModel):
    benchmark: BenchmarkName
    benchmark_id: str
    dataset_origin: str
    task_id: str
    context_tokens: int
    benchmark_context_len: int | None = None
    measured_context_tokens: int = 0
    tokenizer_id: str = "whitespace"
    arm: BenchmarkArmName
    prediction: str
    raw_response: str | None = None
    gold: str | list[str]
    scores: list[BenchmarkScore] = Field(default_factory=list)
    model_calls: int
    provider: str | None = None
    model_name: str | None = None
    model_role: str | None = None
    reasoning_effort: str | None = None
    worker_model_name: str | None = None
    worker_reasoning_effort: str | None = None
    experiment_id: str | None = None
    response_id: str | None = None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    cost_usd: float
    fallback_used: bool = False
    error: str | None = None
    prompt_id: str | None = None
    arm_input_hash: str | None = None
    graph_source_hash: str | None = None
    stop_reason: str
    trace_id: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    gold_evidence_span_ids: list[str] = Field(default_factory=list)
    model_call_traces: list[dict[str, Any]] = Field(default_factory=list)
    case_metadata: dict = Field(default_factory=dict)
    run_fingerprint: str
    failure_categories: list[str] = Field(default_factory=list)


class BenchmarkArm(Protocol):
    name: BenchmarkArmName

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        ...


class BenchmarkScorer(Protocol):
    score_backend: str

    def score(
        self,
        case: BenchmarkCase,
        prediction: BenchmarkArmResult,
    ) -> list[BenchmarkScore]:
        ...
