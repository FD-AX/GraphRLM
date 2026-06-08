from app.benchmarks.factlens import FactLensAuditArm, FactLensAuditScorer, make_factlens_audit_case
from app.benchmarks.factlens.loader import (
    FactLensOfficialRecord,
    compute_complexity_features,
    factlens_case_from_official_record,
    select_factlens_matrix_cases,
)
from app.benchmarks.metrics.postgres_projection import build_postgres_projection
from app.benchmarks.runner import run_benchmark_cases
from scripts.run_factlens_external_matrix import build_factlens_matrix_summary, build_protocol_freeze


def test_factlens_audit_modes_separate_flat_projection_and_graph_reasoning() -> None:
    case = make_factlens_audit_case()
    modes = [
        "flat_subclaim_verification",
        "graph_query_verification",
        "graph_shared_evidence",
    ]
    records = run_benchmark_cases(
        [case],
        [FactLensAuditArm(mode) for mode in modes],
        scorers=[FactLensAuditScorer()],
    )
    by_arm = {record.arm: record for record in records}

    flat = by_arm["flat_subclaim_verification"].trace[0]["factlens_audit"]
    assert flat["subclaims_total"] == 5
    assert flat["subclaims_with_evidence"] == 3
    assert flat["subclaim_recall"] == 0.6
    assert flat["complete_evidence_coverage"] is False
    assert flat["unsupported_subclaim_rate"] == 0.4
    assert flat["unsupported_verdict_rate"] == 1.0
    assert flat["graph_contributed"] is False
    assert flat["graph_contribution_outcome"] == "neutral"

    query = by_arm["graph_query_verification"].trace[0]["factlens_audit"]
    assert query["subclaims_with_evidence"] == 4
    assert query["subclaim_recall"] == 0.8
    assert query["complete_evidence_coverage"] is False
    assert query["cross_subclaim_edges"] == 3
    assert query["graph_contribution_outcome"] == "insufficient"

    shared = by_arm["graph_shared_evidence"].trace[0]["factlens_audit"]
    assert shared["subclaims_with_evidence"] == 5
    assert shared["subclaim_recall"] == 1.0
    assert shared["complete_evidence_coverage"] is True
    assert shared["cross_subclaim_edges"] == 4
    assert query["graph_contributed"] is True
    assert shared["graph_contribution_outcome"] == "useful"
    assert set(shared["evidence_span_ids"]) == {"e1", "e2", "e3", "e4", "e5"}

    assert all(record.model_calls == 0 for record in records)
    assert all(record.scores[0].score_value == 1.0 for record in records)
    assert by_arm["flat_subclaim_verification"].scores[2].score_value == 0.0
    assert by_arm["graph_shared_evidence"].scores[2].score_value == 1.0


def test_factlens_postgres_projection_contains_audit_metrics(tmp_path) -> None:
    case = make_factlens_audit_case()
    records = run_benchmark_cases(
        [case],
        [FactLensAuditArm("graph_query_verification")],
        scorers=[FactLensAuditScorer()],
    )
    artifact_dir = tmp_path / "factlens"
    artifact_dir.mkdir()
    (artifact_dir / "benchmark_records.jsonl").write_text(
        records[0].model_dump_json() + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "benchmark_manifest.json").write_text(
        '{"worker_model": null, "resolved_revision": "local_factlens_audit_v1"}',
        encoding="utf-8",
    )

    projection = build_postgres_projection([artifact_dir])

    assert len(projection["benchmark_factlens_audits"]) == 1
    row = projection["benchmark_factlens_audits"][0]
    assert row["mode"] == "graph_query_verification"
    assert row["subclaims_total"] == 5
    assert row["subclaim_recall"] == 0.8
    assert row["cross_subclaim_edges"] == 3
    assert row["graph_contribution_outcome"] == "insufficient"


def test_factlens_official_selector_freezes_ten_ten_ten_complexity_split() -> None:
    records = (
        [_official_record(index, ["A fact one.", "B fact two."]) for index in range(10)]
        + [
            _official_record(
                100 + index,
                [
                    "Mira founded Northstar Labs.",
                    "Northstar Labs acquired HelioAI.",
                    "HelioAI built search tools.",
                ],
            )
            for index in range(10)
        ]
        + [
            _official_record(
                200 + index,
                [
                    "A Von Baeyer represented Munich in the Geneva Rules.",
                    "J Gladstone represented London in the Geneva Rules.",
                    "W M Ramsay represented London in the Geneva Rules.",
                    "H Armstrong represented London in the Geneva Rules.",
                    "The Geneva Rules were established by the International Chemistry Committee.",
                    "The Geneva Rules were established in 1892.",
                ],
            )
            for index in range(10)
        ]
    )

    selected = select_factlens_matrix_cases(records, per_bucket=10)
    buckets = [compute_complexity_features(record.sub_claims)["complexity_bucket"] for record in selected]

    assert buckets.count("simple") == 10
    assert buckets.count("medium") == 10
    assert buckets.count("complex") == 10


def test_factlens_external_matrix_summary_reports_required_rates() -> None:
    source = _official_record(
        42,
        [
            "Mira founded Northstar Labs.",
            "Northstar Labs acquired HelioAI.",
            "HelioAI built search tools for Argo Health.",
            "Argo Health used HelioAI search.",
            "Mira is connected to Argo Health through HelioAI.",
        ],
    )
    case = factlens_case_from_official_record(
        source,
        dataset_revision="test_revision",
        source_path=__file__,
    )
    records = run_benchmark_cases(
        [case],
        [
            FactLensAuditArm("flat_subclaim_verification"),
            FactLensAuditArm("graph_query_verification"),
            FactLensAuditArm("graph_shared_evidence"),
        ],
        scorers=[FactLensAuditScorer()],
    )

    summary = build_factlens_matrix_summary(records)

    assert summary["arms"]["flat_subclaim_verification"]["complete_evidence_coverage_rate"] < 1.0
    assert summary["arms"]["graph_shared_evidence"]["complete_evidence_coverage_rate"] == 1.0
    assert summary["arms"]["flat_subclaim_verification"]["unsupported_verdict_rate"] == 1.0
    assert summary["arms"]["graph_shared_evidence"]["unsupported_verdict_rate"] == 0.0
    assert summary["arms"]["graph_shared_evidence"]["macro_subclaim_recall"] == 1.0
    assert "bootstrap_ci" in summary
    assert summary["bootstrap_ci"]["graph_shared_evidence"]["complete_evidence_coverage_rate"]["mean"] == 1.0
    assert "binomial_wilson_ci" in summary
    shared_wilson = summary["binomial_wilson_ci"]["graph_shared_evidence"]["complete_evidence_coverage_rate"]
    assert shared_wilson["mean"] == 1.0
    assert shared_wilson["low"] < 1.0
    assert shared_wilson["method"] == "wilson_score"


def test_factlens_protocol_freeze_captures_fixed_rules() -> None:
    protocol = build_protocol_freeze(
        {
            "dataset": "megagonlabs/factlens",
            "resolved_revision": "rev",
            "task_ids": ["factlens_official_1"],
            "split_contract": {"simple": 10, "medium": 10, "complex": 10},
            "arms": ["flat_subclaim_verification", "graph_shared_evidence"],
        }
    )

    assert protocol["frozen"] is True
    assert protocol["resolved_revision"] == "rev"
    assert "valid_edge_rule" in protocol["edge_construction_rules"]
    assert "masked_required_fact" in protocol["negative_controls"]
    assert "official corpus-level retrieval benchmark" in protocol["scope_note"]


def _official_record(index: int, subclaims: list[str]) -> FactLensOfficialRecord:
    return FactLensOfficialRecord(
        ind=str(index),
        claim=" ".join(subclaims),
        sub_claims=subclaims,
        labels=["true"] * len(subclaims),
        aggregated_label=True,
        native_row={},
    )
