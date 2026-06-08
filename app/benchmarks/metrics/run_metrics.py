from __future__ import annotations

from pathlib import Path

from app.benchmarks.artifact_validator import validate_benchmark_record
from app.benchmarks.metrics.score_revision import active_score, latest_revision_for_record, score_from_record
from app.benchmarks.models import BenchmarkRunRecord


def build_run_metric_row(
    record: BenchmarkRunRecord,
    *,
    artifact_dir: Path,
    manifest: dict,
    revisions: list[dict],
    score_name: str,
) -> dict:
    revision = latest_revision_for_record(revisions, record)
    validation = validate_benchmark_record(record, manifest)
    original = score_from_record(record, score_name)
    active = active_score(original, revision, score_name)
    return {
        "artifact_dir": str(artifact_dir),
        "task_id": record.task_id,
        "arm": record.arm,
        "benchmark": record.benchmark,
        "benchmark_context_len": record.benchmark_context_len,
        "measured_context_tokens": record.measured_context_tokens,
        "original_score": original["score_value"],
        "original_scorer": original["score_backend"],
        "active_score": active["score_value"],
        "active_scorer": active["score_backend"],
        "score_revision_reason": active["revision_reason"],
        "score_revision_path": active["revision_path"],
        "model_rerun": bool(active["model_rerun"]),
        "provider": record.provider,
        "model_name": record.model_name,
        "reasoning_effort": record.reasoning_effort,
        "worker_model_name": record.worker_model_name,
        "model_calls": record.model_calls,
        "total_tokens": record.total_tokens,
        "root_tokens": tokens_by_role(record, "root"),
        "worker_tokens": tokens_by_role(record, "worker"),
        "latency_ms": record.latency_ms,
        "root_input_tokens": input_tokens_by_role(record, "root"),
        "context_amplification": safe_div(record.total_tokens, record.measured_context_tokens),
        "root_input_amplification": safe_div(
            input_tokens_by_role(record, "root"),
            record.measured_context_tokens,
        ),
        "artifact_valid": validation["artifact_valid"],
        "invalid_reasons": validation["invalid_reasons"],
        "fallback_used": record.fallback_used,
        "response_id": record.response_id,
        "run_fingerprint": record.run_fingerprint,
        "prediction": record.prediction,
        "gold": record.gold,
    }


def tokens_by_role(record: BenchmarkRunRecord, role: str) -> int:
    return sum(
        int(trace.get("total_tokens") or 0)
        for trace in record.model_call_traces
        if trace.get("model_role") == role
    )


def input_tokens_by_role(record: BenchmarkRunRecord, role: str) -> int:
    return sum(
        int(trace.get("input_tokens") or 0)
        for trace in record.model_call_traces
        if trace.get("model_role") == role
    )


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 6)
