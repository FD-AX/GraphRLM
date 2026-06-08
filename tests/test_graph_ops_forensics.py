from pathlib import Path

import pytest

from app.benchmarks.metrics.graph_ops_forensics import build_graph_ops_forensics


def test_graph_ops_forensics_detects_projection_list_and_amplification() -> None:
    artifact_dirs = [
        Path("artifacts/research_matrix/oolong_synth_stratified5/graph_rlm_semantic_graph_ops_v2/repeat_0"),
        Path("artifacts/research_matrix/oolong_synth_stratified5/graph_rlm_semantic_graph_ops_v2/repeat_1"),
    ]
    if not all(path.exists() for path in artifact_dirs):
        pytest.skip("graph_rlm_semantic_graph_ops_v2 artifacts are not present in this checkout.")

    payload = build_graph_ops_forensics(artifact_dirs, encoder_backend="hashing")
    summary = payload["summary"]

    assert summary["rows_total"] == 10
    assert summary["graph_query_executed_count"] == 2
    assert summary["graph_contributed_count"] == 2
    assert summary["execution_mode_counts"] == {
        "graph_projection_ops": 8,
        "graph_query_ops": 2,
    }
    assert summary["query_semantics_counts"] == {
        "deterministic_operation": 8,
        "unsupported_analytic_operation": 2,
    }
    assert summary["graph_contribution_outcome_counts"]["insufficient"] == 2
    assert summary["local_projection_list_count"] == 8
    assert len(payload["graph_query_run_details"]) == 2
    assert all(detail["query_semantics"] == "unsupported_analytic_operation" for detail in payload["graph_query_run_details"])
    assert all(detail["graph_contribution_outcome"] == "insufficient" for detail in payload["graph_query_run_details"])
    assert all(detail["matched_node_count"] >= 0 for detail in payload["graph_query_run_details"])
    assert summary["field_fidelity_mean"] == 1.0
    assert summary["semantic_documents_per_record_mean"] > 8.0
    assert summary["cold_total_tokens"] == 36858
    assert summary["warm_query_total_tokens"] == 18692
