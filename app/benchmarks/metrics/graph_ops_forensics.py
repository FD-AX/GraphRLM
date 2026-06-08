from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.benchmarks.adapters import oolong_case_from_record
from app.benchmarks.arms import _extract_date, _extract_field, _oolong_record_lines, _raw_record_facts_from_case
from app.benchmarks.metrics.artifacts import read_records
from app.benchmarks.models import BenchmarkCase, BenchmarkRunRecord
from app.benchmarks.oolong.operations import OOLONGRecordFact, plan_oolong_operation
from app.benchmarks.oolong.semantic_graph import build_oolong_semantic_graph


def build_graph_ops_forensics(
    artifact_dirs: list[Path],
    *,
    encoder_backend: str = "hashing",
) -> dict[str, Any]:
    records = [
        record
        for artifact_dir in artifact_dirs
        for record in read_records(artifact_dir)
    ]
    rows = []
    for record in records:
        case = _case_from_record(record)
        labels = _labels_from_record(record)
        raw_facts = _raw_record_facts_from_case(case.arm_view(), labels)
        graph = build_oolong_semantic_graph(
            case.arm_view(),
            record_labels=labels,
            encoder_backend=encoder_backend,
        )
        rows.append(_forensic_row(record, case, raw_facts, graph))
    return {
        "forensic_version": "graph_ops_forensics_v2",
        "encoder_backend": encoder_backend,
        "rows": rows,
        "summary": _summary(rows),
        "graph_query_run_details": _graph_query_run_details(rows),
    }


def write_graph_ops_forensics(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "graph_ops_forensics.json"
    markdown_path = output_dir / "graph_ops_forensics.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_graph_ops_forensics_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_graph_ops_forensics_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Graph Ops Forensics",
        "",
        "## Verdict",
        "",
        f"- Graph query executed rows: `{summary['graph_query_executed_count']}/{summary['rows_total']}`",
        f"- Graph contributed rows: `{summary['graph_contributed_count']}/{summary['rows_total']}`",
        f"- Query semantics: `{summary['query_semantics_counts']}`",
        f"- Graph contribution outcomes: `{summary['graph_contribution_outcome_counts']}`",
        f"- Mean projection fidelity: `{summary['field_fidelity_mean']:.3f}`",
        f"- Mean semantic documents per source record: `{summary['semantic_documents_per_record_mean']:.2f}`",
        f"- Mean prompted documents per unique source record: `{summary['prompted_documents_per_unique_source_record_mean']:.2f}`",
        f"- Warm query token estimate for graph ops: `{summary['warm_query_total_tokens']:,}`",
        f"- Cold graph token total: `{summary['cold_total_tokens']:,}`",
        "",
        "## Execution Mode Summary",
        "",
        "| Execution mode | Runs | Score sum | Tokens | Graph queries | Graph contributed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, item in summary["execution_mode_summary"].items():
        lines.append(
            f"| `{mode}` | {item['runs']} | {item['score_sum']:.3f} "
            f"| {item['tokens']:,} | {item['graph_query_runs']} | {item['graph_contributed_runs']} |"
        )
    lines.extend(
        [
            "",
            "## Graph Query Run Details",
            "",
        "| Task | Semantics | Question | Reason | Outcome | Query/DSL | Nodes | Edges | Evidence | Score | Tokens |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for detail in payload.get("graph_query_run_details", []):
        lines.append(
            f"| `{detail['task_id']}` | `{detail['query_semantics']}` | {detail['question']} | {detail['graph_query_reason']} "
            f"| `{detail['graph_contribution_outcome']}` "
            f"| `{detail['graph_query_dsl'] or 'frontier_traversal'}` "
            f"| {detail['matched_node_count']} | {detail['matched_edge_count']} "
            f"| {detail['evidence_id_count']} | {detail['active_score']:.3f} | {detail['total_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "This report reconstructs each case from immutable artifacts, rebuilds raw `RecordFact` values directly from public records, rebuilds graph-derived `RecordFact` values from `build_oolong_semantic_graph`, and compares the two. It also inspects traces for actual graph query execution versus local projection-list use.",
            "",
            "## Per Run",
            "",
            "| Task | Arm | Semantics | Execution mode | Graph query | Graph contributed | Outcome | Fidelity | Docs/record | Prompt docs/record | Cold tokens | Warm query tokens |",
            "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| `{row['task_id']}` | `{row['arm']}` | `{row['query_semantics']}` | `{row['execution_mode']}` "
            f"| {str(row['graph_use']['graph_query_executed']).lower()} "
            f"| {str(row['graph_contribution']['graph_contributed']).lower()} "
            f"| `{row['graph_contribution']['graph_contribution_outcome']}` "
            f"| {row['projection_fidelity']['field_fidelity']:.3f} "
            f"| {row['deduplication_amplification']['semantic_documents_per_source_record']:.2f} "
            f"| {row['deduplication_amplification']['prompted_documents_per_unique_source_record']:.2f} "
            f"| {row['cost']['cold_total_tokens']:,} "
            f"| {row['cost']['warm_query_tokens']:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def _forensic_row(record: BenchmarkRunRecord, case: BenchmarkCase, raw_facts: list[OOLONGRecordFact], graph) -> dict[str, Any]:
    graph_use = _graph_use(record)
    query_semantics = _query_semantics(record, case)
    execution_mode = _execution_mode(record, graph_use)
    active_score = _active_score(record)
    graph_contribution = _graph_contribution(graph_use, execution_mode, query_semantics, active_score)
    graph_success = _graph_specific_success_criteria(graph_use, graph_contribution, active_score)
    fidelity = _projection_fidelity(raw_facts, graph.record_facts)
    amplification = _deduplication_amplification(record, graph)
    cost = _cost_split(record)
    return {
        "task_id": record.task_id,
        "question": case.question,
        "arm": record.arm,
        "query_semantics": query_semantics,
        "execution_mode": execution_mode,
        "active_score": active_score,
        "graph_use": graph_use,
        "graph_contribution": graph_contribution,
        "graph_specific_success": graph_success,
        "cost": cost,
        "projection_fidelity": fidelity,
        "deduplication_amplification": amplification,
        "data_advantage": {
            "raw_records_input_fields": _raw_input_fields(case),
            "graph_record_fact_source": "build_oolong_semantic_graph.record_facts",
            "graph_fact_count": len(graph.record_facts),
            "raw_fact_count": len(raw_facts),
        },
    }


def _graph_use(record: BenchmarkRunRecord) -> dict[str, Any]:
    provenance = {}
    graph_steps = []
    for step in record.trace:
        if "runtime_provenance" in step:
            provenance = dict(step["runtime_provenance"])
        if step.get("step") in {"graph_seed", "graph_decide", "graph_expand", "graph_frontier", "inspect_local"}:
            graph_steps.append(step)
    matched_node_ids = []
    matched_edge_ids = []
    evidence_ids = []
    for step in graph_steps:
        for key in ("current_documents", "presented_frontier", "selected_transition_ids"):
            value = step.get(key)
            if isinstance(value, list):
                matched_node_ids.extend(str(item) for item in value if not isinstance(item, dict))
                for item in value:
                    if isinstance(item, dict):
                        matched_node_ids.extend(_nested_ids(item, ["semantic_document_id", "owner_id", "target_owner_id"]))
                        matched_edge_ids.extend(_nested_ids(item, ["transition_id"]))
                        evidence_ids.extend(_nested_ids(item, ["evidence_span_ids"]))
        decision = step.get("decision")
        if isinstance(decision, dict):
            matched_edge_ids.extend(_nested_ids(decision, ["selected_transition_ids"]))
            evidence_ids.extend(_nested_ids(decision, ["answer_evidence_span_ids", "evidence_span_ids"]))
    return {
        "semantic_graph_used": provenance.get("semantic_graph_used"),
        "graph_query_executed": bool(graph_steps),
        "graph_repository_called": False,
        "repository_or_query": None,
        "cypher_or_dsl": None,
        "graph_query_reason": _graph_query_reason(record, graph_steps),
        "path_length": len(set(matched_edge_ids)),
        "graph_rows_returned": len(graph_steps),
        "matched_node_ids": list(dict.fromkeys(matched_node_ids)),
        "matched_edge_ids": list(dict.fromkeys(matched_edge_ids)),
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
        "uses_local_projection_list": provenance.get("semantic_graph_used") is True and not graph_steps,
    }


def _execution_mode(record: BenchmarkRunRecord, graph_use: dict[str, Any]) -> str:
    if record.arm == "direct_model":
        return "direct_model"
    if record.arm == "raw_records_ops":
        return "raw_record_ops"
    if record.arm == "graph_rlm_semantic_graph_ops":
        if graph_use["graph_query_executed"]:
            return "graph_query_ops"
        return "graph_projection_ops"
    if record.arm == "graph_rlm_semantic_graph":
        return "recursive_graph"
    if graph_use["graph_query_executed"]:
        return "graph_query_ops"
    if graph_use["uses_local_projection_list"]:
        return "graph_projection_ops"
    return "unknown"


def _graph_contribution(
    graph_use: dict[str, Any],
    execution_mode: str,
    query_semantics: str,
    active_score: float,
) -> dict[str, Any]:
    reasons = []
    if execution_mode not in {"graph_query_ops", "recursive_graph"}:
        return {
            "graph_contributed": False,
            "graph_contribution_outcome": "neutral",
            "reasons": reasons,
            "note": "No graph query/traversal was executed; projection use alone is not counted as graph contribution.",
        }
    if graph_use["graph_query_executed"] and graph_use["matched_node_ids"]:
        reasons.append("matched_graph_nodes")
    if graph_use["matched_edge_ids"]:
        reasons.append("matched_graph_transitions")
    if graph_use["evidence_ids"]:
        reasons.append("graph_linked_evidence")
    if graph_use["path_length"] >= 2:
        reasons.append("multi_hop_path")
    outcome = "neutral"
    if reasons and active_score > 0:
        outcome = "useful"
    elif reasons and query_semantics == "unsupported_analytic_operation":
        outcome = "insufficient"
    elif reasons and active_score == 0:
        outcome = "misleading"
    return {
        "graph_contributed": bool(reasons),
        "graph_contribution_outcome": outcome,
        "reasons": reasons,
        "note": "Contribution requires an executed graph query/traversal plus matched nodes, transitions, evidence, or a multi-hop path.",
    }


def _graph_specific_success_criteria(
    graph_use: dict[str, Any],
    graph_contribution: dict[str, Any],
    active_score: float,
) -> dict[str, bool]:
    return {
        "required_path_found": graph_use["path_length"] > 0,
        "required_entities_found": bool(graph_use["matched_node_ids"]),
        "required_evidence_set_complete": active_score > 0 and bool(graph_use["evidence_ids"]),
        "answer_derived_from_graph_path": graph_contribution["graph_contributed"] and active_score > 0,
    }


def _graph_query_reason(record: BenchmarkRunRecord, graph_steps: list[dict[str, Any]]) -> str:
    if not graph_steps:
        return "not_executed"
    for step in record.trace:
        operation = step.get("deterministic_operation") or step.get("operation")
        if isinstance(operation, dict):
            mode = operation.get("mode")
            if mode:
                return f"deterministic_operation_mode={mode}; routed to graph controller"
    return "frontier-bound graph traversal step was present in trace"


def _active_score(record: BenchmarkRunRecord) -> float:
    for score in record.scores:
        if score.score_name == "task_score":
            return float(score.score_value or 0.0)
    return 0.0


def _query_semantics(record: BenchmarkRunRecord, case: BenchmarkCase) -> str:
    for step in record.trace:
        operation = step.get("deterministic_operation") or step.get("operation")
        if isinstance(operation, dict) and operation.get("query_semantics"):
            return str(operation["query_semantics"])
    plan = plan_oolong_operation(case.question)
    if plan is None:
        if record.arm in {"graph_rlm_semantic_graph", "graph_rlm_semantic_graph_ops"}:
            return "open_ended_semantic_search"
        return "deterministic_operation" if record.arm in {"raw_records_ops"} else "open_ended_semantic_search"
    if plan.operation == "compare_group_rate":
        return "unsupported_analytic_operation"
    if plan.operation in {"count_distinct", "rank", "label_mode"}:
        return "deterministic_operation"
    return "open_ended_semantic_search"


def _projection_fidelity(raw_facts: list[OOLONGRecordFact], graph_facts: list[OOLONGRecordFact]) -> dict[str, Any]:
    raw_by_index = {fact.record_index: fact for fact in raw_facts}
    graph_by_index = {fact.record_index: fact for fact in graph_facts}
    indexes = sorted(set(raw_by_index) | set(graph_by_index))
    checks = []
    mismatches = []
    for index in indexes:
        raw = raw_by_index.get(index)
        graph = graph_by_index.get(index)
        if raw is None or graph is None:
            mismatches.append({"record_index": index, "reason": "missing_record"})
            continue
        values = {
            "user_id": raw.user_id == graph.user_id,
            "label": raw.label == graph.label,
            "timestamp": raw.occurred_at == graph.occurred_at,
        }
        checks.extend(values.values())
        for field, ok in values.items():
            if not ok:
                mismatches.append(
                    {
                        "record_index": index,
                        "field": field,
                        "raw": getattr(raw, field if field != "timestamp" else "occurred_at"),
                        "graph": getattr(graph, field if field != "timestamp" else "occurred_at"),
                    }
                )
    duplicate_graph_records = len(graph_facts) - len({fact.record_index for fact in graph_facts})
    return {
        "record_index_recall": len(set(raw_by_index) & set(graph_by_index)) / len(raw_by_index) if raw_by_index else 1.0,
        "field_fidelity": sum(1 for item in checks if item) / len(checks) if checks else 1.0,
        "user_id_accuracy": _field_accuracy(raw_by_index, graph_by_index, "user_id"),
        "label_accuracy": _field_accuracy(raw_by_index, graph_by_index, "label"),
        "timestamp_accuracy": _timestamp_accuracy(raw_by_index, graph_by_index),
        "duplicate_graph_record_count": duplicate_graph_records,
        "mismatches": mismatches[:20],
    }


def _deduplication_amplification(record: BenchmarkRunRecord, graph) -> dict[str, Any]:
    source_records = max(len(graph.records), 1)
    docs_by_record = Counter()
    for document in graph.semantic_documents:
        for chunk_id in document.source_chunk_ids:
            if ":record:" in chunk_id:
                docs_by_record[chunk_id] += 1
    prompted_docs = _prompted_document_ids(record)
    prompted_source_records = set()
    for document in graph.semantic_documents:
        if document.semantic_document_id not in prompted_docs:
            continue
        prompted_source_records.update(chunk for chunk in document.source_chunk_ids if ":record:" in chunk)
    return {
        "source_record_count": len(graph.records),
        "semantic_document_count": len(graph.semantic_documents),
        "semantic_documents_per_source_record": len(graph.semantic_documents) / source_records,
        "max_documents_for_one_source_record": max(docs_by_record.values()) if docs_by_record else 0,
        "record_fact_document_count": sum(1 for document in graph.semantic_documents if document.owner_type == "record_fact"),
        "candidate_document_count": len(graph.semantic_documents),
        "candidate_documents_per_source_record": len(graph.semantic_documents) / source_records,
        "prompted_document_count": len(prompted_docs),
        "prompted_unique_source_record_count": len(prompted_source_records),
        "prompted_documents_per_unique_source_record": (
            len(prompted_docs) / len(prompted_source_records)
            if prompted_source_records
            else 0.0
        ),
    }


def _cost_split(record: BenchmarkRunRecord) -> dict[str, int]:
    worker_tokens = sum(int(trace.get("total_tokens") or 0) for trace in record.model_call_traces if trace.get("model_role") == "worker")
    root_tokens = sum(int(trace.get("total_tokens") or 0) for trace in record.model_call_traces if trace.get("model_role") == "root")
    return {
        "cold_total_tokens": record.total_tokens,
        "ingestion_tokens": worker_tokens,
        "embedding_tokens": 0,
        "query_planner_tokens": 0,
        "query_execution_tokens": root_tokens,
        "finalization_tokens": 0,
        "warm_query_tokens": root_tokens if root_tokens else 0,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mode_summary = {}
    for mode, mode_rows in _group_by_execution_mode(rows).items():
        mode_summary[mode] = {
            "runs": len(mode_rows),
            "score_sum": sum(float(row["active_score"] or 0.0) for row in mode_rows),
            "tokens": sum(row["cost"]["cold_total_tokens"] for row in mode_rows),
            "graph_query_runs": sum(1 for row in mode_rows if row["graph_use"]["graph_query_executed"]),
            "graph_contributed_runs": sum(1 for row in mode_rows if row["graph_contribution"]["graph_contributed"]),
        }
    return {
        "rows_total": len(rows),
        "graph_query_executed_count": sum(1 for row in rows if row["graph_use"]["graph_query_executed"]),
        "graph_contributed_count": sum(1 for row in rows if row["graph_contribution"]["graph_contributed"]),
        "local_projection_list_count": sum(1 for row in rows if row["graph_use"]["uses_local_projection_list"]),
        "execution_mode_counts": {mode: item["runs"] for mode, item in mode_summary.items()},
        "execution_mode_summary": mode_summary,
        "query_semantics_counts": dict(Counter(row["query_semantics"] for row in rows)),
        "graph_contribution_outcome_counts": dict(Counter(row["graph_contribution"]["graph_contribution_outcome"] for row in rows)),
        "field_fidelity_mean": _mean(row["projection_fidelity"]["field_fidelity"] for row in rows),
        "semantic_documents_per_record_mean": _mean(row["deduplication_amplification"]["semantic_documents_per_source_record"] for row in rows),
        "prompted_documents_per_unique_source_record_mean": _mean(row["deduplication_amplification"]["prompted_documents_per_unique_source_record"] for row in rows),
        "cold_total_tokens": sum(row["cost"]["cold_total_tokens"] for row in rows),
        "warm_query_total_tokens": sum(row["cost"]["warm_query_tokens"] for row in rows),
    }


def _group_by_execution_mode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["execution_mode"]].append(row)
    return dict(sorted(grouped.items()))


def _graph_query_run_details(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details = []
    for row in rows:
        if row["execution_mode"] != "graph_query_ops":
            continue
        graph_use = row["graph_use"]
        details.append(
            {
                "task_id": row["task_id"],
                "arm": row["arm"],
                "question": row["question"],
                "query_semantics": row["query_semantics"],
                "graph_query_reason": graph_use["graph_query_reason"],
                "graph_query_dsl": graph_use["cypher_or_dsl"],
                "matched_node_ids": graph_use["matched_node_ids"],
                "matched_edge_ids": graph_use["matched_edge_ids"],
                "matched_node_count": len(graph_use["matched_node_ids"]),
                "matched_edge_count": len(graph_use["matched_edge_ids"]),
                "path_length": graph_use["path_length"],
                "evidence_ids": graph_use["evidence_ids"],
                "evidence_id_count": len(graph_use["evidence_ids"]),
                "active_score": row["active_score"],
                "graph_contribution_outcome": row["graph_contribution"]["graph_contribution_outcome"],
                "graph_specific_success": row["graph_specific_success"],
                "total_tokens": row["cost"]["cold_total_tokens"],
                "warm_query_tokens": row["cost"]["warm_query_tokens"],
            }
        )
    return details


def _case_from_record(record: BenchmarkRunRecord) -> BenchmarkCase:
    source = dict(record.case_metadata.get("source_record") or record.case_metadata.get("native_fields") or {})
    source["dataset_origin"] = record.dataset_origin
    return oolong_case_from_record(source)


def _labels_from_record(record: BenchmarkRunRecord) -> dict[int, str]:
    for step in record.trace:
        operation = step.get("operation") or step.get("deterministic_operation")
        if isinstance(operation, dict):
            # Labels are not stored directly in operation traces; recover them from worker coverage when possible below.
            pass
    labels: dict[int, str] = {}
    native = record.case_metadata.get("native_fields", {})
    labelled_context = str(native.get("context_window_text_with_labels") or "")
    for index, line in enumerate(_oolong_record_lines(labelled_context)):
        label = _extract_field(line, "Label")
        if label:
            labels[index] = label.strip().lower()
    if labels:
        return labels
    for trace in record.model_call_traces:
        if trace.get("purpose") != "spam_label_extraction":
            continue
        # Worker traces store aggregate counts, not the response body. For old artifacts without labelled
        # source context this forensic cannot recover per-record labels.
    return labels


def _raw_input_fields(case: BenchmarkCase) -> list[dict[str, str | None]]:
    return [
        {
            "record_index": str(index),
            "date": _extract_date(line),
            "user": _extract_field(line, "User"),
            "instance_present": str(bool(_extract_field(line, "Instance"))).lower(),
        }
        for index, line in enumerate(_oolong_record_lines(case.context))
    ][:5]


def _field_accuracy(raw_by_index: dict[int, OOLONGRecordFact], graph_by_index: dict[int, OOLONGRecordFact], field: str) -> float:
    indexes = sorted(set(raw_by_index) & set(graph_by_index))
    if not indexes:
        return 1.0
    return sum(1 for index in indexes if getattr(raw_by_index[index], field) == getattr(graph_by_index[index], field)) / len(indexes)


def _timestamp_accuracy(raw_by_index: dict[int, OOLONGRecordFact], graph_by_index: dict[int, OOLONGRecordFact]) -> float:
    indexes = sorted(set(raw_by_index) & set(graph_by_index))
    if not indexes:
        return 1.0
    return sum(1 for index in indexes if raw_by_index[index].occurred_at == graph_by_index[index].occurred_at) / len(indexes)


def _prompted_document_ids(record: BenchmarkRunRecord) -> set[str]:
    ids = set()
    for step in record.trace:
        for value in step.values():
            ids.update(_collect_semantic_document_ids(value))
    return ids


def _collect_semantic_document_ids(value) -> set[str]:
    ids = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "semantic_document_id" and item:
                ids.add(str(item))
            else:
                ids.update(_collect_semantic_document_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.update(_collect_semantic_document_ids(item))
    return ids


def _nested_ids(value: dict, keys: list[str]) -> list[str]:
    found = []
    for key in keys:
        item = value.get(key)
        if isinstance(item, list):
            found.extend(str(part) for part in item)
        elif item:
            found.append(str(item))
    return found


def _mean(values) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0
