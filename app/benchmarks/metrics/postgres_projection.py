from __future__ import annotations

import hashlib
import ast
import json
import uuid
from pathlib import Path
from statistics import mean

from app.benchmarks.metrics.artifacts import read_manifest, read_records, read_score_revisions
from app.benchmarks.metrics.registry import artifact_status, load_artifact_registry
from app.benchmarks.metrics.run_metrics import build_run_metric_row
from app.benchmarks.models import BenchmarkRunRecord
from app.benchmarks.oolong.operations import plan_oolong_operation


RUN_NAMESPACE = uuid.UUID("d9587d4a-26aa-4f75-b0c0-2df24d35b53e")


def build_postgres_projection(
    artifact_dirs: list[Path],
    *,
    registry_path: Path = Path("artifacts/artifact_registry.json"),
    score_name: str = "task_score",
) -> dict:
    registry = load_artifact_registry(registry_path)
    projection = {
        "benchmark_runs": [],
        "benchmark_model_calls": [],
        "benchmark_score_revisions": [],
        "benchmark_graph_metrics": [],
        "benchmark_graph_forensics": [],
        "benchmark_factlens_audits": [],
    }
    for artifact_dir in artifact_dirs:
        manifest = read_manifest(artifact_dir)
        revisions = read_score_revisions(artifact_dir)
        for record in read_records(artifact_dir):
            metric = build_run_metric_row(
                record,
                artifact_dir=artifact_dir,
                manifest=manifest,
                revisions=revisions,
                score_name=score_name,
            )
            run_id = stable_run_id(record)
            status = artifact_status(registry, artifact_dir, record.arm)
            runtime_provenance = _runtime_provenance(record)
            graph_validation = runtime_provenance.get("validation", {})
            source = record.case_metadata.get("source", {})
            native = record.case_metadata.get("native_fields", record.case_metadata)
            score_revision_id = _active_score_revision_id(revisions, record, run_id, score_name)
            projection["benchmark_runs"].append(
                {
                    "run_id": str(run_id),
                    "experiment_id": record.experiment_id,
                    "run_fingerprint": record.run_fingerprint,
                    "artifact_path": str(artifact_dir),
                    "artifact_hash": _artifact_hash(artifact_dir),
                    "benchmark_id": record.benchmark_id,
                    "dataset_name": source.get("dataset") or native.get("dataset"),
                    "dataset_origin": record.dataset_origin,
                    "dataset_subset": manifest.get("dataset_subset_filter") or native.get("dataset"),
                    "dataset_config": source.get("config"),
                    "dataset_split": manifest.get("split") or source.get("split"),
                    "requested_revision": manifest.get("requested_revision") or source.get("revision"),
                    "resolved_revision": manifest.get("resolved_revision") or source.get("resolved_revision"),
                    "task_id": record.task_id,
                    "task_group": native.get("task_group"),
                    "task_name": native.get("task"),
                    "answer_type": native.get("answer_type"),
                    "benchmark_context_len": record.benchmark_context_len,
                    "measured_context_tokens": record.measured_context_tokens,
                    "tokenizer_id": record.tokenizer_id,
                    "arm_name": record.arm,
                    "execution_path_status": status.get("status", "unknown"),
                    "research_eligible": bool(status.get("research_eligible", False)),
                    "graph_source": record.graph_source_hash,
                    "graph_builder_class": runtime_provenance.get("graph_builder_class"),
                    "encoder_class": runtime_provenance.get("encoder_class"),
                    "semantic_graph_used": bool(runtime_provenance.get("semantic_graph_used", False)),
                    "legacy_hash_frontier_used": bool(runtime_provenance.get("legacy_hash_frontier_used", False)),
                    "root_provider": record.provider,
                    "root_model": record.model_name,
                    "root_reasoning_effort": record.reasoning_effort,
                    "worker_provider": _provider_by_role(record, "worker"),
                    "worker_model": record.worker_model_name,
                    "recursive_model": _model_by_role(record, "recursive"),
                    "prompt_id": record.prompt_id,
                    "prediction": record.prediction,
                    "gold_answer": _json_compatible_answer(record.gold),
                    "original_score": metric["original_score"],
                    "original_scorer_backend": metric["original_scorer"],
                    "active_score": metric["active_score"],
                    "active_scorer_backend": metric["active_scorer"],
                    "score_revision_id": str(score_revision_id) if score_revision_id else None,
                    "model_rerun_for_score": metric["model_rerun"],
                    "model_calls": record.model_calls,
                    "root_calls": _call_count_by_role(record, "root"),
                    "worker_calls": _call_count_by_role(record, "worker"),
                    "recursive_calls": _call_count_by_role(record, "recursive"),
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "total_tokens": record.total_tokens,
                    "root_tokens": metric["root_tokens"],
                    "worker_tokens": metric["worker_tokens"],
                    "recursive_tokens": _tokens_by_role(record, "recursive"),
                    "latency_ms": record.latency_ms,
                    "graph_build_latency_ms": None,
                    "online_query_latency_ms": None,
                    "context_amplification": metric["context_amplification"],
                    "root_input_amplification": metric["root_input_amplification"],
                    "tokens_per_correct_answer": _tokens_per_correct(record.total_tokens, metric["active_score"]),
                    "artifact_valid": metric["artifact_valid"],
                    "invalid_reasons": metric["invalid_reasons"],
                    "fallback_used": record.fallback_used,
                    "response_ids_complete": _response_ids_complete(record),
                    "trace_token_match": _trace_token_match(record),
                    "labelled_context_leakage": bool(graph_validation.get("labelled_context_leakage", False)),
                    "stop_reason": record.stop_reason,
                    "graph_build_id": None,
                    "graph_reused": None,
                    "runtime_provenance": runtime_provenance,
                    "graph_metrics": graph_validation,
                    "metadata": {
                        "registry_status": status,
                        "manifest": manifest,
                        "source": source,
                    },
                }
            )
            projection["benchmark_model_calls"].extend(_model_call_rows(record, run_id))
            projection["benchmark_score_revisions"].extend(
                _score_revision_rows(revisions, record, run_id)
            )
            projection["benchmark_graph_metrics"].append(_graph_metrics_row(record, run_id, graph_validation))
            projection["benchmark_graph_forensics"].append(
                _graph_forensics_row(record, run_id, graph_validation, runtime_provenance)
            )
            factlens_row = _factlens_audit_row(record, run_id)
            if factlens_row:
                projection["benchmark_factlens_audits"].append(factlens_row)
    return projection


def stable_run_id(record: BenchmarkRunRecord) -> uuid.UUID:
    key = f"{record.experiment_id}:{record.run_fingerprint}:{record.task_id}:{record.arm}"
    return uuid.uuid5(RUN_NAMESPACE, key)


def _runtime_provenance(record: BenchmarkRunRecord) -> dict:
    for step in record.trace:
        if "runtime_provenance" in step:
            return dict(step["runtime_provenance"])
    return {}


def _artifact_hash(artifact_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(artifact_dir.glob("benchmark_*.json*")):
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_compatible_answer(value):
    if isinstance(value, (list, dict, int, float, bool)) or value is None:
        return value
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    return parsed


def _active_score_revision_id(
    revisions: list[dict],
    record: BenchmarkRunRecord,
    run_id: uuid.UUID,
    score_name: str,
) -> uuid.UUID | None:
    for revision in reversed(revisions):
        if str(revision.get("task_id")) == record.task_id and revision.get("arm") == record.arm:
            return _score_revision_id(record, revision, run_id, score_name)
    return None


def _score_revision_id(
    record: BenchmarkRunRecord,
    revision: dict,
    run_id: uuid.UUID,
    score_name: str | None,
) -> uuid.UUID:
    key = f"{run_id}:{record.task_id}:{record.arm}:{score_name}:{revision.get('rescored_scorer_version')}:{revision.get('rescore_reason')}"
    return uuid.uuid5(RUN_NAMESPACE, key)


def _provider_by_role(record: BenchmarkRunRecord, role: str) -> str | None:
    for trace in record.model_call_traces:
        if trace.get("model_role") == role:
            return trace.get("provider")
    return None


def _model_by_role(record: BenchmarkRunRecord, role: str) -> str | None:
    for trace in record.model_call_traces:
        if trace.get("model_role") == role:
            return trace.get("model_name")
    return None


def _call_count_by_role(record: BenchmarkRunRecord, role: str) -> int:
    return sum(1 for trace in record.model_call_traces if trace.get("model_role") == role)


def _tokens_by_role(record: BenchmarkRunRecord, role: str) -> int:
    return sum(
        int(trace.get("total_tokens") or 0)
        for trace in record.model_call_traces
        if trace.get("model_role") == role
    )


def _tokens_per_correct(total_tokens: int, active_score: float | None) -> float | None:
    if not active_score:
        return None
    return float(total_tokens) / active_score


def _response_ids_complete(record: BenchmarkRunRecord) -> bool:
    traces = [trace for trace in record.model_call_traces if trace.get("provider") == "openai"]
    return bool(traces) and all(trace.get("response_id") for trace in traces)


def _trace_token_match(record: BenchmarkRunRecord) -> bool:
    if not record.model_call_traces:
        return True
    return sum(int(trace.get("total_tokens") or 0) for trace in record.model_call_traces) == record.total_tokens


def _model_call_rows(record: BenchmarkRunRecord, run_id: uuid.UUID) -> list[dict]:
    rows = []
    for index, trace in enumerate(record.model_call_traces):
        call_id = trace.get("call_id") or f"{run_id}:{index}"
        rows.append(
            {
                "call_id": call_id,
                "run_id": str(run_id),
                "call_index": index,
                "provider": trace.get("provider") or record.provider or "unknown",
                "model_name": trace.get("model_name") or "unknown",
                "model_role": trace.get("model_role") or "unknown",
                "purpose": trace.get("purpose"),
                "reasoning_effort": trace.get("reasoning_effort"),
                "request_timestamp": trace.get("request_timestamp"),
                "response_timestamp": trace.get("response_timestamp"),
                "latency_ms": trace.get("latency_ms"),
                "input_tokens": int(trace.get("input_tokens") or 0),
                "output_tokens": int(trace.get("output_tokens") or 0),
                "total_tokens": int(trace.get("total_tokens") or 0),
                "response_id": trace.get("response_id"),
                "response_id_present": bool(trace.get("response_id")),
                "fallback_used": bool(trace.get("fallback_used", False)),
                "batch_start": trace.get("batch_start"),
                "batch_size": trace.get("batch_size"),
                "records_returned": trace.get("records_returned"),
                "trace_metadata": trace,
            }
        )
    return rows


def _score_revision_rows(revisions: list[dict], record: BenchmarkRunRecord, run_id: uuid.UUID) -> list[dict]:
    rows = []
    for revision in revisions:
        if str(revision.get("task_id")) != record.task_id or revision.get("arm") != record.arm:
            continue
        previous_scores = {
            score.get("score_name"): score
            for score in revision.get("original_scores", [])
            if score.get("score_name")
        }
        for score in revision.get("rescored_scores", []):
            previous = previous_scores.get(score.get("score_name"), {})
            rows.append(
                {
                    "score_revision_id": str(
                        _score_revision_id(record, revision, run_id, score.get("score_name"))
                    ),
                    "run_id": str(run_id),
                    "previous_backend": previous.get("score_backend"),
                    "previous_score": previous.get("score_value"),
                    "scorer_backend": score.get("score_backend"),
                    "scorer_version": revision.get("rescored_scorer_version"),
                    "score_name": score.get("score_name"),
                    "score_value": score.get("score_value"),
                    "reason": revision.get("rescore_reason") or "unspecified",
                    "model_rerun": bool(revision.get("model_rerun", False)),
                    "is_official_score": bool(score.get("is_official_score", False)),
                    "scorer_metadata": score.get("metadata", {}),
                }
            )
    return rows


def _graph_metrics_row(record: BenchmarkRunRecord, run_id: uuid.UUID, graph_validation: dict) -> dict:
    frontier_sizes = [
        len(step.get("frontier_transition_ids") or [])
        for step in record.trace
        if step.get("step") == "inspect_local"
    ]
    duplicate_evidence_count = sum(
        len(step.get("omitted_repeated_evidence_span_ids") or [])
        for step in record.trace
        if step.get("step") == "inspect_local"
    )
    return {
        "run_id": str(run_id),
        "semantic_document_count": graph_validation.get("graph_node_count"),
        "entity_snapshot_count": None,
        "pair_snapshot_count": None,
        "observation_count": graph_validation.get("materialized_observations"),
        "aggregate_projection_count": graph_validation.get("aggregate_projection_count"),
        "embedding_count": graph_validation.get("embedding_count"),
        "embedding_dim": graph_validation.get("embedding_dim"),
        "initial_frontier_size": frontier_sizes[0] if frontier_sizes else None,
        "mean_frontier_size": mean(frontier_sizes) if frontier_sizes else None,
        "max_frontier_size": max(frontier_sizes) if frontier_sizes else None,
        "visited_node_count": len(set(record.evidence_span_ids)),
        "controller_steps": sum(1 for step in record.trace if step.get("step") == "graph_decide"),
        "presented_evidence_count": len(set(record.evidence_span_ids)),
        "duplicate_evidence_count": duplicate_evidence_count,
        "worker_label_coverage": None,
        "graph_build_tokens": _tokens_by_role(record, "worker"),
        "graph_build_model_calls": _call_count_by_role(record, "worker"),
        "controller_payload_tokens_initial": None,
        "controller_payload_tokens_max": None,
        "repeated_context_tokens": None,
        "graph_selected_context_tokens": None,
        "full_context_tokens": record.measured_context_tokens,
        "context_reduction_ratio": None,
        "graph_answer": record.prediction,
        "rlm_answer": None,
        "agreement_status": None,
        "verifier_calls": None,
        "extra_metrics": graph_validation,
    }


def _graph_forensics_row(
    record: BenchmarkRunRecord,
    run_id: uuid.UUID,
    graph_validation: dict,
    runtime_provenance: dict,
) -> dict:
    graph_steps = [
        step
        for step in record.trace
        if step.get("step") in {"graph_seed", "graph_decide", "graph_expand", "graph_frontier", "inspect_local"}
    ]
    semantic_document_count = graph_validation.get("graph_node_count") or graph_validation.get("semantic_document_count")
    record_count = graph_validation.get("record_count") or graph_validation.get("record_fact_count")
    matched_node_ids = set()
    matched_edge_ids = set()
    evidence_ids = set(record.evidence_span_ids)
    prompted_document_ids = set()
    for step in graph_steps:
        prompted_document_ids.update(_collect_values_for_key(step, "semantic_document_id"))
        matched_node_ids.update(_collect_values_for_key(step, "owner_id"))
        matched_node_ids.update(_collect_values_for_key(step, "target_owner_id"))
        matched_edge_ids.update(_collect_values_for_key(step, "transition_id"))
        evidence_ids.update(_collect_values_for_key(step, "evidence_span_ids"))
        evidence_ids.update(_collect_values_for_key(step, "answer_evidence_span_ids"))
    worker_tokens = _tokens_by_role(record, "worker")
    root_tokens = _tokens_by_role(record, "root")
    graph_query_executed = bool(graph_steps) and root_tokens > 0
    semantic_graph_used = bool(runtime_provenance.get("semantic_graph_used", False))
    query_semantics = _query_semantics(record)
    execution_mode = _execution_mode(record, graph_query_executed, semantic_graph_used)
    graph_contribution = _graph_contribution(
        execution_mode=execution_mode,
        query_semantics=query_semantics,
        active_score=_score_value(record),
        graph_query_executed=graph_query_executed,
        matched_node_count=len(matched_node_ids),
        matched_edge_count=len(matched_edge_ids),
        evidence_id_count=len(evidence_ids),
    )
    return {
        "run_id": str(run_id),
        "query_semantics": query_semantics,
        "execution_mode": execution_mode,
        "graph_query_executed": graph_query_executed,
        "graph_repository_called": False,
        "uses_local_projection_list": semantic_graph_used and not graph_query_executed,
        "graph_contributed": graph_contribution["graph_contributed"],
        "graph_contribution_outcome": graph_contribution["graph_contribution_outcome"],
        "graph_contribution_reasons": graph_contribution["reasons"],
        "path_length": len(matched_edge_ids),
        "graph_query_reason": _graph_query_reason(record, graph_query_executed),
        "graph_query_dsl": None,
        "required_path_found": len(matched_edge_ids) > 0,
        "required_entities_found": len(matched_node_ids) > 0,
        "required_evidence_set_complete": _score_value(record) > 0 and len(evidence_ids) > 0,
        "answer_derived_from_graph_path": graph_contribution["graph_contributed"] and _score_value(record) > 0,
        "graph_rows_returned": len(graph_steps),
        "matched_node_count": len(matched_node_ids),
        "matched_edge_count": len(matched_edge_ids),
        "evidence_id_count": len(evidence_ids),
        "projection_fidelity": None,
        "semantic_documents_per_source_record": (
            float(semantic_document_count) / float(record_count)
            if semantic_document_count and record_count
            else None
        ),
        "prompted_documents_per_unique_source_record": (
            float(len(prompted_document_ids)) / float(len(evidence_ids))
            if prompted_document_ids and evidence_ids
            else 0.0
        ),
        "cold_total_tokens": record.total_tokens,
        "ingestion_tokens": worker_tokens,
        "warm_query_tokens": root_tokens if graph_query_executed else 0,
        "forensic_metadata": {
            "semantic_graph_used": semantic_graph_used,
            "legacy_hash_frontier_used": runtime_provenance.get("legacy_hash_frontier_used"),
            "graph_builder_class": runtime_provenance.get("graph_builder_class"),
            "encoder_class": runtime_provenance.get("encoder_class"),
            "note": "Projection-level forensic; exact field fidelity is produced by graph_ops_forensics artifact.",
        },
    }


def _execution_mode(record: BenchmarkRunRecord, graph_query_executed: bool, semantic_graph_used: bool) -> str:
    if record.arm == "direct_model":
        return "direct_model"
    if record.arm == "raw_records_ops":
        return "raw_record_ops"
    if record.arm == "graph_rlm_semantic_graph_ops":
        return "graph_query_ops" if graph_query_executed else "graph_projection_ops"
    if record.arm == "graph_rlm_semantic_graph":
        return "recursive_graph"
    if graph_query_executed:
        return "graph_query_ops"
    if semantic_graph_used:
        return "graph_projection_ops"
    return "unknown"


def _graph_contribution(
    *,
    execution_mode: str,
    query_semantics: str,
    active_score: float,
    graph_query_executed: bool,
    matched_node_count: int,
    matched_edge_count: int,
    evidence_id_count: int,
) -> dict:
    reasons = []
    if execution_mode not in {"graph_query_ops", "recursive_graph"} or not graph_query_executed:
        return {"graph_contributed": False, "graph_contribution_outcome": "neutral", "reasons": reasons}
    if matched_node_count:
        reasons.append("matched_graph_nodes")
    if matched_edge_count:
        reasons.append("matched_graph_transitions")
    if evidence_id_count:
        reasons.append("graph_linked_evidence")
    outcome = "neutral"
    if reasons and active_score > 0:
        outcome = "useful"
    elif reasons and query_semantics == "unsupported_analytic_operation":
        outcome = "insufficient"
    elif reasons:
        outcome = "misleading"
    return {"graph_contributed": bool(reasons), "graph_contribution_outcome": outcome, "reasons": reasons}


def _graph_query_reason(record: BenchmarkRunRecord, graph_query_executed: bool) -> str | None:
    if not graph_query_executed:
        return None
    for step in record.trace:
        operation = step.get("deterministic_operation") or step.get("operation")
        if isinstance(operation, dict) and operation.get("mode"):
            return f"deterministic_operation_mode={operation['mode']}; routed to graph controller"
    return "frontier-bound graph traversal step was present in trace"


def _query_semantics(record: BenchmarkRunRecord) -> str:
    for step in record.trace:
        operation = step.get("deterministic_operation") or step.get("operation")
        if isinstance(operation, dict) and operation.get("query_semantics"):
            return str(operation["query_semantics"])
    question = _record_question(record)
    plan = plan_oolong_operation(question) if question else None
    if plan is None:
        return "open_ended_semantic_search"
    if plan.operation == "compare_group_rate":
        return "unsupported_analytic_operation"
    if plan.operation in {"count_distinct", "rank", "label_mode"}:
        return "deterministic_operation"
    return "open_ended_semantic_search"


def _record_question(record: BenchmarkRunRecord) -> str:
    for container_key in ("source_record", "native_fields"):
        container = record.case_metadata.get(container_key)
        if isinstance(container, dict) and container.get("question"):
            return str(container["question"])
    if record.case_metadata.get("question"):
        return str(record.case_metadata["question"])
    return ""


def _score_value(record: BenchmarkRunRecord) -> float:
    for score in record.scores:
        if score.score_name == "task_score":
            return float(score.score_value or 0.0)
    return 0.0


def _collect_values_for_key(value, key: str) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                if isinstance(item_value, list):
                    found.update(str(item) for item in item_value)
                elif item_value:
                    found.add(str(item_value))
            else:
                found.update(_collect_values_for_key(item_value, key))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_values_for_key(item, key))
    return found


def _factlens_audit_row(record: BenchmarkRunRecord, run_id: uuid.UUID) -> dict | None:
    audit = _factlens_audit_payload(record)
    if not audit:
        return None
    return {
        "run_id": str(run_id),
        "mode": audit.get("mode") or record.arm,
        "claim_id": audit.get("claim_id") or record.task_id,
        "subclaims_total": int(audit.get("subclaims_total") or 0),
        "subclaims_with_evidence": int(audit.get("subclaims_with_evidence") or 0),
        "final_verdict_accuracy": float(audit.get("final_verdict_accuracy") or 0.0),
        "subclaim_recall": float(audit.get("subclaim_recall") or 0.0),
        "all_required_subclaims_verified": bool(audit.get("all_required_subclaims_verified", False)),
        "unsupported_subclaim_rate": float(audit.get("unsupported_subclaim_rate") or 0.0),
        "unsupported_verdict_rate": float(audit.get("unsupported_verdict_rate") or 0.0),
        "shared_graph_facts": int(audit.get("shared_graph_facts") or 0),
        "reused_evidence_count": int(audit.get("reused_evidence_count") or 0),
        "shared_evidence_reuse": int(audit.get("shared_evidence_reuse") or 0),
        "cross_subclaim_edges": int(audit.get("cross_subclaim_edges") or 0),
        "contradictions_found": int(audit.get("contradictions_found") or 0),
        "complete_evidence_coverage": bool(audit.get("complete_evidence_coverage", False)),
        "required_path_found": bool(audit.get("required_path_found", False)),
        "answer_derived_from_graph_path": bool(audit.get("answer_derived_from_graph_path", False)),
        "tokens_per_verified_subclaim": audit.get("tokens_per_verified_subclaim"),
        "tokens_per_fully_verified_claim": audit.get("tokens_per_fully_verified_claim"),
        "fully_verified_claims_per_10k_tokens": float(audit.get("fully_verified_claims_per_10k_tokens") or 0.0),
        "graph_contributed": bool(audit.get("graph_contributed", False)),
        "graph_contribution_outcome": audit.get("graph_contribution_outcome") or "neutral",
        "evidence_span_ids": audit.get("evidence_span_ids") or [],
        "graph_fact_ids": audit.get("graph_fact_ids") or [],
        "graph_edge_ids": audit.get("graph_edge_ids") or [],
        "audit_metadata": audit,
    }


def _factlens_audit_payload(record: BenchmarkRunRecord) -> dict | None:
    for step in record.trace:
        audit = step.get("factlens_audit")
        if isinstance(audit, dict):
            return audit
    return None
