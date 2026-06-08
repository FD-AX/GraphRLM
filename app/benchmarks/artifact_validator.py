from __future__ import annotations

from app.benchmarks.models import BenchmarkRunRecord


def validate_benchmark_record(record: BenchmarkRunRecord, manifest: dict | None = None) -> dict:
    manifest = manifest or {}
    reasons: list[str] = []
    if record.provider == "openai" and not record.response_id:
        reasons.append("missing_response_id")
    if record.provider == "openai" and not record.model_call_traces:
        reasons.append("missing_model_call_traces")
    trace_tokens = sum(trace.get("total_tokens") or 0 for trace in record.model_call_traces)
    if record.model_call_traces and trace_tokens != record.total_tokens:
        reasons.append("total_tokens_trace_sum_mismatch")
    manifest_worker = manifest.get("worker_model")
    if manifest_worker != record.worker_model_name:
        reasons.append("manifest_runtime_worker_mismatch")
    if not record.experiment_id:
        reasons.append("missing_experiment_id")
    if manifest and not manifest.get("resolved_revision"):
        reasons.append("missing_resolved_revision")
    provenance = _runtime_provenance(record)
    if record.arm in {"graph_rlm_semantic_graph", "graph_rlm_semantic_graph_ops"}:
        if not provenance.get("semantic_graph_used"):
            reasons.append("semantic_graph_not_used")
        if provenance.get("legacy_hash_frontier_used") is not False:
            reasons.append("legacy_hash_frontier_used")
        if provenance.get("encoder_class") != "TransformerSemanticEncoder":
            reasons.append("semantic_graph_encoder_not_transformer")
        if provenance.get("graph_builder_class") != "OOLONGSemanticGraphBuilder":
            reasons.append("wrong_graph_builder")
    return {
        "artifact_valid": not reasons,
        "invalid_reasons": reasons,
        "trace_total_tokens": trace_tokens,
        "record_total_tokens": record.total_tokens,
        "runtime_provenance": provenance,
    }


def _runtime_provenance(record: BenchmarkRunRecord) -> dict:
    for step in record.trace:
        if "runtime_provenance" in step:
            return dict(step["runtime_provenance"])
    return {}
