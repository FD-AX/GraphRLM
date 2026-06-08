from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.benchmarks import (
    DirectHeuristicArm,
    KeywordRAGArm,
    OOLONGLocalCompatibleScorer,
    SNIAHScorer,
    aggregate_records,
    case_from_record,
    load_jsonl_cases,
    make_s_niah_cases,
    run_benchmark_cases,
)
from app.benchmarks.oolong.semantic_graph import build_oolong_semantic_graph
from app.benchmarks.oolong.operations import (
    execute_oolong_operation,
    normalize_user_id,
    plan_oolong_operation,
)
from app.benchmarks.models import BenchmarkArmResult
from app.benchmarks.arms import GraphRLMSemanticGraphArm, GraphRLMSemanticGraphOpsArm, RawRecordsOpsArm
from app.benchmarks.artifact_validator import validate_benchmark_record
from app.benchmarks.metrics import build_evaluation_report
from app.benchmarks.metrics.postgres_projection import build_postgres_projection
from app.benchmarks.metrics.postgres_repository import AsyncBenchmarkMetricsRepository
from app.dual_rlm import GraphRLMDecision


def test_s_niah_adapter_and_scorer_exact_match() -> None:
    case = case_from_record(
        {
            "benchmark": "s_niah",
            "id": "needle_1",
            "haystack": "hay hay The secret code is needle-0001. hay",
            "question": "What is the secret code?",
            "needle": "needle-0001",
            "evidence_span_ids": ["span_needle"],
            "context_tokens": 8,
            "dataset_origin": "official",
        }
    )

    scores = SNIAHScorer().score(
        case,
        BenchmarkArmResult(prediction="needle-0001", evidence_span_ids=["span_needle"]),
    )

    assert case.task_id == "needle_1"
    assert case.benchmark == "s_niah"
    assert case.benchmark_id == "sniah"
    assert scores[0].score_backend == "sniah_official_exact_match"
    assert scores[0].score_value == 1.0
    assert scores[0].is_official_score is True


def test_oolong_and_pairs_adapters_accept_hf_like_records(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    rows = [
        {
            "id": "trec_1",
            "context_window_text": "question type: location\nanswer: place",
            "question": "Which type is most frequent?",
            "answer": "location",
            "dataset_origin": "official",
        },
        {
            "id": "pair_1",
            "records": [{"a": "x", "b": "y"}],
            "question": "Which pair is valid?",
            "answer": "x-y",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    oolong_cases = load_jsonl_cases(path, benchmark="oolong")
    pair_cases = load_jsonl_cases(path, benchmark="oolong_pairs")

    assert [case.benchmark for case in oolong_cases] == ["oolong", "oolong"]
    assert [case.benchmark for case in pair_cases] == ["oolong_pairs", "oolong_pairs"]
    assert "question type" in oolong_cases[0].context
    assert oolong_cases[0].metadata["native_fields"]["context_window_text"]
    assert oolong_cases[0].dataset_origin == "official"
    assert pair_cases[1].task_id == "pair_1"


def test_oolong_local_compatible_scorer_keeps_official_flag_false() -> None:
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": 110010000,
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": "one spam message",
            "question": "Which label is most common?",
            "answer": "spam",
            "dataset_origin": "official",
        }
    )
    scorer = OOLONGLocalCompatibleScorer()

    gold_scores = scorer.score(case, BenchmarkArmResult(prediction="spam"))
    wrong_scores = scorer.score(case, BenchmarkArmResult(prediction="ham"))

    assert gold_scores[0].score_backend == "oolong_local_compatible_v2"
    assert gold_scores[0].score_name == "task_score"
    assert gold_scores[0].score_value == 1.0
    assert gold_scores[0].is_official_score is False
    assert wrong_scores[0].score_value == 0.0


def test_oolong_local_compatible_scorer_accepts_required_answer_format() -> None:
    scorer = OOLONGLocalCompatibleScorer()
    label_case = case_from_record(
        {
            "benchmark": "oolong",
            "id": 1,
            "context_window_text": "context",
            "question": "Which label? Give answer using labels: ham, spam.",
            "answer": "['spam']",
            "answer_type": "ANSWER_TYPE.LABEL",
            "dataset_origin": "official",
        }
    )
    numeric_case = case_from_record(
        {
            "benchmark": "oolong",
            "id": 2,
            "context_window_text": "context",
            "question": "How many?",
            "answer": "[1]",
            "answer_type": "ANSWER_TYPE.NUMERIC",
            "dataset_origin": "official",
        }
    )

    assert scorer.score(label_case, BenchmarkArmResult(prediction="Label: spam"))[0].score_value == 1.0
    assert scorer.score(label_case, BenchmarkArmResult(prediction="Label: spam."))[0].score_value == 1.0
    assert (
        scorer.score(label_case, BenchmarkArmResult(prediction="Final answer: Label: spam."))[0].score_value
        == 1.0
    )
    assert scorer.score(label_case, BenchmarkArmResult(prediction="['spam']"))[0].score_value == 1.0
    assert scorer.score(label_case, BenchmarkArmResult(prediction="spam"))[0].score_value == 1.0
    assert scorer.score(label_case, BenchmarkArmResult(prediction="ham or spam"))[0].score_value == 0.0
    assert scorer.score(numeric_case, BenchmarkArmResult(prediction="Answer: 1"))[0].score_value == 1.0


def test_runner_passes_label_safe_case_to_arm() -> None:
    seen = {}

    class SpyArm:
        name = "direct_model"

        def run_case(self, case):
            seen["metadata"] = case.metadata
            return BenchmarkArmResult(prediction="spam")

    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": 1,
            "context_len": 1024,
            "context_window_text": "unlabelled context",
            "context_window_text_with_labels": "labelled context || Label: spam",
            "question": "Which label?",
            "answer": "spam",
            "dataset_origin": "official",
        }
    )

    run_benchmark_cases([case], [SpyArm()], scorers=[OOLONGLocalCompatibleScorer()])

    native_fields = seen["metadata"]["native_fields"]
    assert "context_window_text" in native_fields
    assert "context_window_text_with_labels" not in native_fields
    assert "answer" not in native_fields


def test_benchmark_runner_outputs_required_record_and_aggregate() -> None:
    cases = make_s_niah_cases(2, 128)
    records = run_benchmark_cases(cases, [DirectHeuristicArm(), KeywordRAGArm(top_k=1)])
    aggregate = aggregate_records(records)

    assert len(records) == 4
    first = records[0].model_dump()
    for key in [
        "benchmark",
        "task_id",
        "context_tokens",
        "arm",
        "prediction",
        "gold",
        "scores",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "cost_usd",
        "stop_reason",
        "trace_id",
        "trace",
        "case_metadata",
        "model_call_traces",
    ]:
        assert key in first
    assert all(not score["is_official_score"] for score in first["scores"])
    assert aggregate["groups"]


def test_model_config_is_recorded_in_run_fingerprint() -> None:
    case = make_s_niah_cases(1, 128)[0]

    class ConfiguredArm:
        name = "direct_model"

        def __init__(self, reasoning_effort: str) -> None:
            self.reasoning_effort = reasoning_effort

        def run_case(self, case):
            return BenchmarkArmResult(
                prediction="needle-0000",
                provider="openai",
                model_name="gpt-5",
                model_role="root",
                reasoning_effort=self.reasoning_effort,
                experiment_id="paper_aligned_gpt5_v1",
                response_id="resp_test",
                prompt_id="oolong_direct_v1",
                trace_id=f"configured:{self.reasoning_effort}",
                model_call_traces=[
                    {
                        "provider": "openai",
                        "model_name": "gpt-5",
                        "model_role": "root",
                        "reasoning_effort": self.reasoning_effort,
                        "purpose": "test",
                        "response_id": "resp_test",
                        "total_tokens": 1,
                        "fallback_used": False,
                    }
                ],
            )

    low_record = run_benchmark_cases([case], [ConfiguredArm("low")])[0]
    medium_record = run_benchmark_cases([case], [ConfiguredArm("medium")])[0]

    assert medium_record.reasoning_effort == "medium"
    assert medium_record.model_role == "root"
    assert medium_record.experiment_id == "paper_aligned_gpt5_v1"
    assert medium_record.model_call_traces[0]["reasoning_effort"] == "medium"
    assert low_record.run_fingerprint != medium_record.run_fingerprint


def test_nested_trace_payload_survives_record_serialization() -> None:
    case = make_s_niah_cases(1, 128)[0]

    class TraceArm:
        name = "direct_model"

        def run_case(self, case):
            return BenchmarkArmResult(
                prediction="needle-0000",
                trace=[
                    {
                        "graph_build": {"model_calls": 0, "latency_ms": 3},
                        "query": {"controller_calls": 1, "prompt_overhead_ratio": 0.5},
                    }
                ],
            )

    record = run_benchmark_cases([case], [TraceArm()])[0]
    restored = type(record).model_validate_json(record.model_dump_json())

    assert restored.trace[0]["graph_build"]["model_calls"] == 0
    assert restored.trace[0]["query"]["prompt_overhead_ratio"] == 0.5


def test_oolong_semantic_graph_builds_production_style_snapshots_without_label_leakage() -> None:
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "spam_case",
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": (
                "The following lines contain text messages.\n"
                "Date: Jan 01, 2026 || User: 1 || Instance: hello there\n"
                "Date: Jan 02, 2026 || User: 2 || Instance: win cash now\n"
            ),
            "context_window_text_with_labels": (
                "Date: Jan 01, 2026 || User: 1 || Instance: hello there || Label: ham\n"
                "Date: Jan 02, 2026 || User: 2 || Instance: win cash now || Label: spam\n"
            ),
            "question": "Which label is most common?",
            "answer": "['ham']",
            "answer_type": "ANSWER_TYPE.LABEL",
            "dataset_origin": "official",
        }
    )

    build = build_oolong_semantic_graph(
        case,
        record_labels={0: "ham", 1: "spam"},
        encoder_backend="hashing",
    )

    assert build.validation["graph_node_count"] > 0
    assert build.validation["stable_source_ids"] is True
    assert build.validation["evidence_spans_reference_public_context"] is True
    assert build.validation["labelled_context_leakage"] is False
    assert build.validation["materialized_observations"] > 0
    assert build.validation["aggregate_projection_count"] > 0
    assert len(build.embeddings) == len(build.semantic_documents)
    assert any("has_predicted_label" in document.text for document in build.semantic_documents)
    assert build.validation["record_fact_count"] == 2
    assert build.validation["canonical_user_id_count"] == 2
    assert build.validation["timestamp_normalized_count"] == 2
    assert build.validation["record_fact_candidate_count"] == 2
    assert build.validation["seed_strategy"] == "cosine_seed_structural_frontier"
    assert build.validation["frontier_strategy"] == "structural_overlap"
    assert build.validation["interaction_profile_enabled"] is False
    assert build.record_facts[0].user_id == "1"
    assert build.record_facts[0].label == "ham"
    assert any(
        document.owner_type == "record_fact"
        and "TYPE=record" in document.text
        and "USER_ID=1" in document.text
        and "LABEL=ham" in document.text
        and "DATE=2026-01-01" in document.text
        for document in build.semantic_documents
    )


def test_oolong_record_fact_count_operation_uses_canonical_user_and_distinct_record() -> None:
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "count_case",
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": (
                "Date: Mar 20, 2025 || User: 78591 || Instance: hello one\n"
                "Date: Mar 21, 2025 || User: 45630 || Instance: hello two\n"
                "Date: Mar 22, 2025 || User: 78591 || Instance: hello three\n"
            ),
            "question": (
                "only consider subset of instances associated with user IDs 45630. "
                "Among these users, how many data points should be label ham? Answer: number."
            ),
            "answer": "[1]",
            "answer_type": "ANSWER_TYPE.NUMBER",
            "dataset_origin": "official",
        }
    )
    build = build_oolong_semantic_graph(
        case,
        record_labels={0: "ham", 1: "ham", 2: "ham"},
        encoder_backend="hashing",
    )

    plan = plan_oolong_operation(case.question)
    result = execute_oolong_operation(plan, build.record_facts)

    assert normalize_user_id("user_45630") == "45630"
    assert plan is not None
    assert result.status == "complete"
    assert result.answer_text == "Answer: 1"
    assert result.matched_record_ids == [build.records[1].record_id]
    assert result.evidence_span_ids == [build.records[1].evidence_span_id]


def test_oolong_question_planner_builds_plans_from_paraphrases_without_case_metadata() -> None:
    count_plan = plan_oolong_operation("Among records belonging to user 45630, how many have label ham?")
    rank_plan = plan_oolong_operation("Which user is the 2nd most frequent?")
    temporal_plan = plan_oolong_operation("Was label ham more common before 2023-03-07 or after 2023-03-07?")
    label_plan = plan_oolong_operation("Which label is most common?")

    assert count_plan is not None
    assert count_plan.operation == "count_distinct"
    assert count_plan.subject_filter == {"user_id": "45630"}
    assert count_plan.predicate_filter == {"label": "ham"}
    assert count_plan.planner_source == "question_text"
    assert rank_plan is not None
    assert rank_plan.operation == "rank"
    assert rank_plan.group_by == "user_id"
    assert rank_plan.rank == 2
    assert temporal_plan is not None
    assert temporal_plan.operation == "compare_group_rate"
    assert temporal_plan.partition["boundary"] == "2023-03-07"
    assert label_plan is not None
    assert label_plan.operation == "label_mode"
    assert label_plan.target == "label"
    assert label_plan.sort == "count_desc"


def test_temporal_plan_is_not_executed_until_boundary_contract_is_confirmed() -> None:
    plan = plan_oolong_operation("Was label ham more common before 2023-03-07 or after 2023-03-07?")
    result = execute_oolong_operation(plan, [])

    assert result.status == "unsupported"
    assert "diagnostic-only" in result.reason


def test_oolong_user_rank_operation_returns_second_most_frequent_user() -> None:
    lines = []
    labels = {}
    index = 0
    for user_id, count in [("39230", 12), ("38371", 9), ("75774", 1)]:
        for _ in range(count):
            lines.append(f"Date: Feb 01, 2025 || User: {user_id} || Instance: message {index}")
            labels[index] = "ham"
            index += 1
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "rank_case",
            "context_len": 2048,
            "dataset": "spam",
            "context_window_text": "\n".join(lines),
            "question": "In the above data, which user is represented second most often?",
            "answer": "[38371]",
            "answer_type": "ANSWER_TYPE.NUMBER",
            "dataset_origin": "official",
        }
    )
    build = build_oolong_semantic_graph(case, record_labels=labels, encoder_backend="hashing")

    plan = plan_oolong_operation(case.question)
    result = execute_oolong_operation(plan, build.record_facts)

    assert plan is not None
    assert plan.operation == "rank"
    assert result.status == "complete"
    assert result.answer_text == "38371"
    assert len(result.matched_record_ids) == 9


def test_oolong_label_mode_operation_returns_most_common_label() -> None:
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "label_mode_case",
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": (
                "Date: Jan 01, 2026 || User: 1 || Instance: hello there\n"
                "Date: Jan 02, 2026 || User: 2 || Instance: win cash now\n"
                "Date: Jan 03, 2026 || User: 3 || Instance: free prize\n"
            ),
            "question": "Which label is most common?",
            "answer": "['spam']",
            "answer_type": "ANSWER_TYPE.LABEL",
            "dataset_origin": "official",
        }
    )
    build = build_oolong_semantic_graph(
        case,
        record_labels={0: "ham", 1: "spam", 2: "spam"},
        encoder_backend="hashing",
    )

    plan = plan_oolong_operation(case.question)
    result = execute_oolong_operation(plan, build.record_facts)

    assert plan is not None
    assert plan.operation == "label_mode"
    assert result.status == "complete"
    assert result.answer_text == "Label: spam"
    assert len(result.matched_record_ids) == 2


def test_graph_rlm_semantic_graph_executes_user_count_without_root_controller(monkeypatch) -> None:
    def fake_worker_labels(*args, **kwargs):
        return (
            {0: "ham", 1: "ham", 2: "ham"},
            [
                {
                    "provider": "openai",
                    "model_name": "gpt-5-mini",
                    "model_role": "worker",
                    "purpose": "spam_label_extraction",
                    "response_id": "resp_worker",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "fallback_used": False,
                }
            ],
        )

    class ForbiddenGateway:
        @property
        def model_call_traces(self):
            return []

        def decide_graph(self, state):
            raise AssertionError("deterministic count must not call the root controller")

    monkeypatch.setattr("app.benchmarks.arms._extract_worker_record_labels", fake_worker_labels)
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "count_case_arm",
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": (
                "Date: Mar 20, 2025 || User: 78591 || Instance: hello one\n"
                "Date: Mar 21, 2025 || User: 45630 || Instance: hello two\n"
                "Date: Mar 22, 2025 || User: 78591 || Instance: hello three\n"
            ),
            "question": (
                "only consider subset of instances associated with user IDs 45630. "
                "Among these users, how many data points should be label ham? Answer: number."
            ),
            "answer": "[1]",
            "answer_type": "ANSWER_TYPE.NUMBER",
            "dataset_origin": "official",
        }
    ).arm_view()
    arm = GraphRLMSemanticGraphArm(
        worker_model_name="gpt-5-mini",
        encoder_backend="hashing",
        gateway_factory=lambda: ForbiddenGateway(),
    )

    record = run_benchmark_cases([case], [arm], scorers=[OOLONGLocalCompatibleScorer()])[0]

    assert record.prediction == "Answer: 1"
    assert record.scores[0].score_value == 1.0
    assert record.stop_reason == "deterministic_operation_complete"
    assert record.model_calls == 1
    assert record.model_call_traces[0]["model_role"] == "worker"
    assert record.trace[0]["deterministic_operation"]["matched_record_ids"]


def test_raw_records_ops_executes_count_without_semantic_graph(monkeypatch) -> None:
    def fake_worker_labels(*args, **kwargs):
        return (
            {0: "ham", 1: "ham", 2: "spam"},
            [
                {
                    "provider": "openai",
                    "model_name": "gpt-5-mini",
                    "model_role": "worker",
                    "purpose": "spam_label_extraction",
                    "response_id": "resp_worker",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "fallback_used": False,
                }
            ],
        )

    monkeypatch.setattr("app.benchmarks.arms._extract_worker_record_labels", fake_worker_labels)
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "raw_count_case",
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": (
                "Date: Mar 20, 2025 || User: 78591 || Instance: hello one\n"
                "Date: Mar 21, 2025 || User: 45630 || Instance: hello two\n"
                "Date: Mar 22, 2025 || User: 45630 || Instance: hello three\n"
            ),
            "question": "Among records belonging to user 45630, how many have label ham?",
            "answer": "[1]",
            "answer_type": "ANSWER_TYPE.NUMBER",
            "dataset_origin": "official",
        }
    ).arm_view()
    arm = RawRecordsOpsArm(worker_model_name="gpt-5-mini")

    record = run_benchmark_cases([case], [arm], scorers=[OOLONGLocalCompatibleScorer()])[0]

    assert record.arm == "raw_records_ops"
    assert record.prediction == "Answer: 1"
    assert record.stop_reason == "raw_records_ops_complete"
    assert record.trace[0]["runtime_provenance"]["semantic_graph_used"] is False
    assert record.trace[0]["runtime_provenance"]["record_fact_count"] == 3


def test_graph_rlm_semantic_graph_executes_user_rank_without_root_controller(monkeypatch) -> None:
    def fake_worker_labels(*args, **kwargs):
        return ({index: "ham" for index in range(22)}, [])

    class ForbiddenGateway:
        @property
        def model_call_traces(self):
            return []

        def decide_graph(self, state):
            raise AssertionError("deterministic rank must not call the root controller")

    monkeypatch.setattr("app.benchmarks.arms._extract_worker_record_labels", fake_worker_labels)
    lines = []
    index = 0
    for user_id, count in [("39230", 12), ("38371", 9), ("75774", 1)]:
        for _ in range(count):
            lines.append(f"Date: Feb 01, 2025 || User: {user_id} || Instance: message {index}")
            index += 1
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "rank_case_arm",
            "context_len": 2048,
            "dataset": "spam",
            "context_window_text": "\n".join(lines),
            "question": "In the above data, which user is represented second most often?",
            "answer": "[38371]",
            "answer_type": "ANSWER_TYPE.NUMBER",
            "dataset_origin": "official",
        }
    ).arm_view()
    arm = GraphRLMSemanticGraphArm(
        worker_model_name="gpt-5-mini",
        encoder_backend="hashing",
        gateway_factory=lambda: ForbiddenGateway(),
    )

    record = run_benchmark_cases([case], [arm], scorers=[OOLONGLocalCompatibleScorer()])[0]

    assert record.prediction == "38371"
    assert record.scores[0].score_value == 1.0
    assert record.stop_reason == "deterministic_operation_complete"
    assert record.model_calls == 0


def test_graph_rlm_semantic_graph_ops_has_separate_arm_identity(monkeypatch) -> None:
    def fake_worker_labels(*args, **kwargs):
        return ({0: "ham"}, [])

    class ForbiddenGateway:
        @property
        def model_call_traces(self):
            return []

        def decide_graph(self, state):
            raise AssertionError("deterministic count must not call the root controller")

    monkeypatch.setattr("app.benchmarks.arms._extract_worker_record_labels", fake_worker_labels)
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "count_case_ops_arm",
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": "Date: Mar 21, 2025 || User: 45630 || Instance: hello two",
            "question": "Among records belonging to user 45630, how many have label ham?",
            "answer": "[1]",
            "answer_type": "ANSWER_TYPE.NUMBER",
            "dataset_origin": "official",
        }
    ).arm_view()
    arm = GraphRLMSemanticGraphOpsArm(
        worker_model_name="gpt-5-mini",
        encoder_backend="hashing",
        gateway_factory=lambda: ForbiddenGateway(),
    )

    record = run_benchmark_cases([case], [arm], scorers=[OOLONGLocalCompatibleScorer()])[0]

    assert record.arm == "graph_rlm_semantic_graph_ops"
    assert record.prompt_id == "oolong_graph_rlm_semantic_graph_ops_v1"
    assert record.stop_reason == "deterministic_operation_complete"
    assert record.prediction == "Answer: 1"


def test_graph_rlm_semantic_graph_ops_blocks_unsupported_analytic_fallback(monkeypatch) -> None:
    def fake_worker_labels(*args, **kwargs):
        return ({0: "ham", 1: "spam"}, [])

    class ForbiddenGateway:
        @property
        def model_call_traces(self):
            return []

        def decide_graph(self, state):
            raise AssertionError("unsupported analytic operation must not call the root controller")

    monkeypatch.setattr("app.benchmarks.arms._extract_worker_record_labels", fake_worker_labels)
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "temporal_unsupported_case",
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": (
                "Date: Mar 01, 2025 || User: 1 || Instance: hello\n"
                "Date: Mar 09, 2025 || User: 2 || Instance: win cash"
            ),
            "question": "Was label ham more common before 2025-03-07 or after 2025-03-07?",
            "answer": "['before']",
            "answer_type": "ANSWER_TYPE.CATEGORY",
            "dataset_origin": "official",
        }
    ).arm_view()
    arm = GraphRLMSemanticGraphOpsArm(
        worker_model_name="gpt-5-mini",
        encoder_backend="hashing",
        gateway_factory=lambda: ForbiddenGateway(),
    )

    record = run_benchmark_cases([case], [arm], scorers=[OOLONGLocalCompatibleScorer()])[0]

    assert record.stop_reason == "unsupported_analytic_operation"
    assert record.model_calls == 0
    assert record.prediction == ""
    operation = record.trace[0]["deterministic_operation"]
    guard = record.trace[0]["routing_guard"]
    assert operation["query_semantics"] == "unsupported_analytic_operation"
    assert operation["status"] == "unsupported"
    assert guard["recursive_graph_allowed"] is False


def test_graph_rlm_semantic_graph_no_model_wiring_uses_semantic_view() -> None:
    captured = {}

    class CaptureGateway:
        @property
        def model_call_traces(self):
            return []

        def decide_graph(self, state):
            captured["state"] = state
            return GraphRLMDecision(
                action="answer",
                evidence_sufficient=True,
                confidence=1.0,
                decision_summary="Semantic graph view captured.",
            )

    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "semantic_arm_case",
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": (
                "The following lines contain text messages.\n"
                "Date: Jan 01, 2026 || User: 1 || Instance: hello there\n"
                "Date: Jan 02, 2026 || User: 2 || Instance: win cash now\n"
            ),
            "context_window_text_with_labels": (
                "Date: Jan 01, 2026 || User: 1 || Instance: hello there || Label: ham\n"
                "Date: Jan 02, 2026 || User: 2 || Instance: win cash now || Label: spam\n"
            ),
            "question": "Which label is most common?",
            "answer": "['ham']",
            "answer_type": "ANSWER_TYPE.LABEL",
            "dataset_origin": "official",
        }
    ).arm_view()

    arm = GraphRLMSemanticGraphArm(
        model_name="gpt-5",
        reasoning_effort="medium",
        experiment_id="paper_aligned_gpt5_v1",
        encoder_backend="hashing",
        gateway_factory=lambda: CaptureGateway(),
    )
    record = run_benchmark_cases([case], [arm], scorers=[OOLONGLocalCompatibleScorer()])[0]
    provenance = record.trace[0]["runtime_provenance"]
    payload = captured["state"]
    frontier_text = json.dumps(payload["presented_frontier"], ensure_ascii=False)

    assert record.arm == "graph_rlm_semantic_graph"
    assert provenance["graph_builder_class"] == "OOLONGSemanticGraphBuilder"
    assert provenance["semantic_graph_used"] is True
    assert provenance["legacy_hash_frontier_used"] is False
    assert provenance["validation"]["labelled_context_leakage"] is False
    assert "evidence_" in frontier_text
    assert "Label:" not in frontier_text
    assert "context_window_text_with_labels" not in frontier_text


def test_artifact_validator_rejects_manifest_runtime_worker_mismatch() -> None:
    case = make_s_niah_cases(1, 128)[0]

    class TraceArm:
        name = "direct_model"

        def run_case(self, case):
            return BenchmarkArmResult(
                prediction="needle-0000",
                provider="openai",
                model_name="gpt-5",
                response_id="resp",
                total_tokens=1,
                experiment_id="exp",
                model_call_traces=[{"total_tokens": 1, "response_id": "resp"}],
            )

    record = run_benchmark_cases([case], [TraceArm()])[0]
    result = validate_benchmark_record(
        record,
        {"worker_model": "gpt-5-mini", "resolved_revision": "sha"},
    )

    assert result["artifact_valid"] is False
    assert "manifest_runtime_worker_mismatch" in result["invalid_reasons"]


def test_evaluation_report_applies_score_revision_without_rewriting_record(tmp_path: Path) -> None:
    case = case_from_record(
        {
            "benchmark": "oolong",
            "id": "113010009",
            "context_window_text": "context",
            "question": "Which label? Give answer using labels: ham, spam.",
            "answer": "['ham']",
            "answer_type": "ANSWER_TYPE.LABEL",
            "dataset_origin": "official",
        }
    )

    class VerboseArm:
        name = "direct_model"

        def run_case(self, case):
            return BenchmarkArmResult(
                prediction="Final answer: Label: ham.",
                provider="openai",
                model_name="gpt-5",
                response_id="resp",
                total_tokens=10,
                experiment_id="exp",
                model_call_traces=[
                    {
                        "provider": "openai",
                        "model_name": "gpt-5",
                        "model_role": "root",
                        "purpose": "test",
                        "response_id": "resp",
                        "total_tokens": 10,
                        "fallback_used": False,
                    }
                ],
            )

    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    record = run_benchmark_cases([case], [VerboseArm()], scorers=[OOLONGLocalCompatibleScorer()])[0]
    original_record = record.model_copy(
        update={
            "scores": [
                record.scores[0].model_copy(update={"score_backend": "oolong_local_compatible", "score_value": 0.0})
            ],
        }
    )
    (artifact_dir / "benchmark_records.jsonl").write_text(
        original_record.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "benchmark_manifest.json").write_text(
        json.dumps({"worker_model": None, "resolved_revision": "sha"}),
        encoding="utf-8",
    )
    (artifact_dir / "score_revision_oolong_local_compatible_v2.json").write_text(
        json.dumps(
            {
                "score_revisions": [
                    {
                        "task_id": "113010009",
                        "arm": "direct_model",
                        "model_rerun": False,
                        "rescored_scores": [
                            {
                                "score_backend": "oolong_local_compatible_v2",
                                "score_name": "task_score",
                                "score_value": 1.0,
                                "is_official_score": False,
                            }
                        ],
                        "rescore_reason": "test normalization",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_evaluation_report([artifact_dir])
    row = report["runs"][0]

    assert row["original_score"] == 0.0
    assert row["active_score"] == 1.0
    assert row["original_scorer"] == "oolong_local_compatible"
    assert row["active_scorer"] == "oolong_local_compatible_v2"
    assert row["model_rerun"] is False


def test_evaluation_report_113010009_golden_metrics() -> None:
    semantic_dir = Path("artifacts/oolong_synth_semantic_graph_real_113010009")
    surrogate_dir = Path("artifacts/oolong_synth_spam_8k_graph_rlm_gpt5mini_worker_task113010009_v3")
    if not semantic_dir.exists() or not surrogate_dir.exists():
        pytest.skip("Golden benchmark artifacts are not present in this checkout.")

    report = build_evaluation_report([surrogate_dir, semantic_dir])
    rows = {row["arm"]: row for row in report["runs"]}
    semantic = rows["graph_rlm_semantic_graph"]
    surrogate = rows["graph_rlm"]

    assert semantic["original_score"] == 0.0
    assert semantic["active_score"] == 1.0
    assert semantic["total_tokens"] == 40633
    assert semantic["root_tokens"] == 27935
    assert semantic["worker_tokens"] == 12698
    assert semantic["context_amplification"] == 11.215291

    assert surrogate["original_score"] == 0.0
    assert surrogate["active_score"] == 0.0
    assert surrogate["total_tokens"] == 98249
    assert surrogate["root_tokens"] == 85875
    assert surrogate["worker_tokens"] == 12374
    assert surrogate["context_amplification"] == 27.118134

    token_reduction = 1 - (semantic["total_tokens"] / surrogate["total_tokens"])
    assert round(token_reduction, 3) == 0.586


def test_postgres_projection_maps_113010009_metrics_idempotently() -> None:
    semantic_dir = Path("artifacts/oolong_synth_semantic_graph_real_113010009")
    surrogate_dir = Path("artifacts/oolong_synth_spam_8k_graph_rlm_gpt5mini_worker_task113010009_v3")
    if not semantic_dir.exists() or not surrogate_dir.exists():
        pytest.skip("Golden benchmark artifacts are not present in this checkout.")

    first = build_postgres_projection([surrogate_dir, semantic_dir])
    second = build_postgres_projection([surrogate_dir, semantic_dir])
    first_runs = {row["arm_name"]: row for row in first["benchmark_runs"]}
    second_runs = {row["arm_name"]: row for row in second["benchmark_runs"]}

    semantic = first_runs["graph_rlm_semantic_graph"]
    surrogate = first_runs["graph_rlm"]

    assert semantic["original_score"] == 0.0
    assert semantic["active_score"] == 1.0
    assert semantic["total_tokens"] == 40633
    assert semantic["execution_path_status"] == "research"
    assert semantic["research_eligible"] is True

    assert surrogate["original_score"] == 0.0
    assert surrogate["active_score"] == 0.0
    assert surrogate["total_tokens"] == 98249
    assert surrogate["execution_path_status"] == "surrogate"
    assert surrogate["research_eligible"] is False

    assert len(first["benchmark_runs"]) == 2
    assert len(first["benchmark_graph_metrics"]) == 2
    assert len(first["benchmark_score_revisions"]) >= 1
    assert sum(call["total_tokens"] for call in first["benchmark_model_calls"] if call["run_id"] == semantic["run_id"]) == 40633
    assert first_runs["graph_rlm_semantic_graph"]["run_id"] == second_runs["graph_rlm_semantic_graph"]["run_id"]
    assert first_runs["graph_rlm"]["run_id"] == second_runs["graph_rlm"]["run_id"]


@pytest.mark.asyncio
async def test_async_postgres_repository_upserts_projection_rows() -> None:
    class FakeAsyncConnection:
        def __init__(self) -> None:
            self.executed = []

        async def execute(self, sql, params=None):
            self.executed.append((sql, params or []))

    projection = {
        "benchmark_runs": [{"run_id": "run-1", "metadata": {"source": "artifact"}}],
        "benchmark_model_calls": [{"call_id": "call-1", "run_id": "run-1"}],
        "benchmark_score_revisions": [{"score_revision_id": "revision-1", "run_id": "run-1"}],
        "benchmark_graph_metrics": [{"run_id": "run-1", "extra_metrics": {"nodes": 3}}],
    }
    conn = FakeAsyncConnection()
    repository = AsyncBenchmarkMetricsRepository("postgresql://unused")
    repository._conn = conn

    await repository.upsert_projection(projection)

    sql_statements = [sql for sql, _ in conn.executed]
    assert len(sql_statements) == 4
    assert any("benchmark_metrics.benchmark_runs" in sql for sql in sql_statements)
    assert any("benchmark_metrics.benchmark_model_calls" in sql for sql in sql_statements)
    assert any("benchmark_metrics.benchmark_score_revisions" in sql for sql in sql_statements)
    assert any("benchmark_metrics.benchmark_graph_metrics" in sql for sql in sql_statements)
    assert all("ON CONFLICT" in sql for sql in sql_statements)
