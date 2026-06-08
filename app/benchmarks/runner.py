from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from math import ceil
from statistics import mean, median

from app.benchmarks.models import (
    BenchmarkArm,
    BenchmarkCase,
    BenchmarkRunRecord,
    BenchmarkScorer,
)
from app.benchmarks.proxy.scorers import default_proxy_scorers


def run_benchmark_cases(
    cases: list[BenchmarkCase],
    arms: list[BenchmarkArm],
    scorers: list[BenchmarkScorer] | None = None,
) -> list[BenchmarkRunRecord]:
    resolved_scorers = scorers or default_proxy_scorers()
    records = []
    for case in cases:
        for arm in arms:
            arm_case = case.arm_view()
            arm_result = arm.run_case(arm_case)
            scores = [
                score
                for scorer in resolved_scorers
                for score in scorer.score(case, arm_result)
            ]
            records.append(
                BenchmarkRunRecord(
                    benchmark=case.benchmark,
                    benchmark_id=case.benchmark_id,
                    dataset_origin=case.dataset_origin,
                    task_id=case.task_id,
                    context_tokens=case.context_tokens,
                    benchmark_context_len=case.benchmark_context_len,
                    measured_context_tokens=case.measured_context_tokens,
                    tokenizer_id=case.tokenizer_id,
                    arm=arm.name,
                    prediction=arm_result.prediction,
                    raw_response=arm_result.raw_response,
                    gold=case.gold_answer,
                    scores=scores,
                    model_calls=arm_result.model_calls,
                    provider=arm_result.provider,
                    model_name=arm_result.model_name,
                    model_role=arm_result.model_role,
                    reasoning_effort=arm_result.reasoning_effort,
                    worker_model_name=arm_result.worker_model_name,
                    worker_reasoning_effort=arm_result.worker_reasoning_effort,
                    experiment_id=arm_result.experiment_id,
                    response_id=arm_result.response_id,
                    input_tokens=arm_result.input_tokens,
                    output_tokens=arm_result.output_tokens,
                    total_tokens=arm_result.total_tokens,
                    latency_ms=arm_result.latency_ms,
                    cost_usd=arm_result.cost_usd,
                    fallback_used=arm_result.fallback_used,
                    error=arm_result.error,
                    prompt_id=arm_result.prompt_id,
                    arm_input_hash=arm_result.arm_input_hash,
                    graph_source_hash=arm_result.graph_source_hash,
                    stop_reason=arm_result.stop_reason,
                    trace_id=arm_result.trace_id,
                    trace=arm_result.trace,
                    evidence_span_ids=arm_result.evidence_span_ids,
                    gold_evidence_span_ids=case.gold_evidence_span_ids,
                    model_call_traces=arm_result.model_call_traces,
                    case_metadata=case.metadata,
                    run_fingerprint=_run_fingerprint(case, arm_result, scores),
                    failure_categories=_failure_categories(case, scores, arm_result.evidence_span_ids),
                )
            )
    return records


def aggregate_records(
    records: list[BenchmarkRunRecord],
    *,
    score_name: str = "exact_match",
) -> dict:
    grouped: dict[tuple[str, str, int | None], list[BenchmarkRunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.benchmark, record.arm, record.benchmark_context_len)].append(record)

    rows = []
    for (benchmark, arm, context_tokens), items in sorted(grouped.items()):
        score_values = [_score_value(item, score_name) for item in items]
        correct = [
            item
            for item, value in zip(items, score_values)
            if value is not None and value >= 1.0
        ]
        token_totals = [item.total_tokens or item.input_tokens + item.output_tokens for item in items]
        measured_tokens = [item.measured_context_tokens for item in items]
        rows.append(
            {
                "benchmark": benchmark,
                "arm": arm,
                "benchmark_context_len": context_tokens,
                "measured_context_tokens_mean": _mean(measured_tokens),
                "tokenizer_id": items[0].tokenizer_id,
                "runs_total": len(items),
                "score_name": score_name,
                "score_mean": _mean(value for value in score_values if value is not None),
                "official_score_count": sum(
                    1
                    for item in items
                    for score in item.scores
                    if score.score_name == score_name and score.is_official_score
                ),
                "evidence_recall_mean": _mean(
                    _score_value(item, "evidence_recall") or 0.0
                    for item in items
                ),
                "latency_mean_ms": _mean(item.latency_ms for item in items),
                "latency_p50_ms": _median(item.latency_ms for item in items),
                "latency_p95_ms": _p95([item.latency_ms for item in items]),
                "tokens_mean": _mean(token_totals),
                "tokens_total": sum(token_totals),
                "missing_response_id_count": sum(1 for item in items if item.provider and not item.response_id),
                "fallback_count": sum(1 for item in items if item.fallback_used),
                "missing_prediction_count": sum(1 for item in items if not item.prediction),
                "tokens_per_correct_answer": (
                    sum(token_totals) / len(correct) if correct else None
                ),
                "failure_categories": dict(
                    Counter(
                        category
                        for item in items
                        for category in item.failure_categories
                    )
                ),
            }
        )
    return {
        "groups": rows,
        "by_task_group": _dimension_breakdown(records, "task_group", score_name),
        "by_answer_type": _dimension_breakdown(records, "answer_type", score_name),
    }


def _failure_categories(
    case: BenchmarkCase,
    scores,
    evidence_span_ids: list[str],
) -> list[str]:
    failures = []
    exact = _score_value_from_scores(scores, "exact_match")
    if case.answerable and exact is not None and exact < 1.0:
        failures.append("wrong_answer")
    if case.gold_evidence_span_ids and not evidence_span_ids:
        failures.append("missing_evidence")
    evidence_recall = _score_value_from_scores(scores, "evidence_recall")
    if case.gold_evidence_span_ids and evidence_recall is not None and evidence_recall < 1.0:
        failures.append("partial_evidence_recall")
    return failures


def _mean(values) -> float:
    collected = list(values)
    return round(mean(collected), 6) if collected else 0.0


def _median(values) -> float:
    collected = list(values)
    return round(median(collected), 6) if collected else 0.0


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = max(ceil(0.95 * len(values)) - 1, 0)
    return float(values[index])


def _dimension_breakdown(
    records: list[BenchmarkRunRecord],
    dimension: str,
    score_name: str,
) -> list[dict]:
    grouped: dict[tuple[str, str], list[BenchmarkRunRecord]] = defaultdict(list)
    for record in records:
        native = record.case_metadata.get("native_fields", {})
        grouped[(record.arm, str(native.get(dimension)))].append(record)
    rows = []
    for (arm, value), items in sorted(grouped.items()):
        scores = [_score_value(item, score_name) for item in items]
        rows.append(
            {
                "arm": arm,
                dimension: value,
                "runs_total": len(items),
                "score_name": score_name,
                "score_mean": _mean(score for score in scores if score is not None),
            }
        )
    return rows


def _score_value(record: BenchmarkRunRecord, score_name: str) -> float | None:
    return _score_value_from_scores(record.scores, score_name)


def _score_value_from_scores(scores, score_name: str) -> float | None:
    for score in scores:
        if score.score_name == score_name:
            return score.score_value
    return None


def _run_fingerprint(case: BenchmarkCase, arm_result, scores) -> str:
    source = case.metadata.get("source", {})
    payload = {
        "dataset": source.get("dataset"),
        "dataset_revision": source.get("resolved_revision") or source.get("revision"),
        "requested_revision": source.get("revision"),
        "adapter_version": "benchmark_case_v2",
        "scorers": [
            {
                "backend": score.score_backend,
                "name": score.score_name,
                "official": score.is_official_score,
            }
            for score in scores
        ],
        "arm_version": arm_result.trace_id.split(":", 1)[0] if arm_result.trace_id else None,
        "model": arm_result.model_name,
        "model_role": arm_result.model_role,
        "reasoning_effort": arm_result.reasoning_effort,
        "worker_model": arm_result.worker_model_name,
        "worker_reasoning_effort": arm_result.worker_reasoning_effort,
        "experiment_id": arm_result.experiment_id,
        "prompt_version": arm_result.prompt_id,
        "tokenizer": case.tokenizer_id,
    }
    return sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
