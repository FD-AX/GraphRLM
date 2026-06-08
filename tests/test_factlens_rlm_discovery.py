from app.benchmarks.factlens.rlm_discovery import (
    FactLensRLMDiscoveryArm,
    FactLensRLMDiscoveryScorer,
    factlens_rlm_discovery_cases,
)
from app.benchmarks.factlens.loader import FactLensOfficialRecord
from app.benchmarks.runner import run_benchmark_cases


def test_factlens_rlm_discovery_recovers_hidden_evidence_and_records_prompt_versions() -> None:
    cases = factlens_rlm_discovery_cases([_record(0), _record(1), _record(2)], dataset_revision="rev", cases_total=3)
    records = run_benchmark_cases(
        cases,
        [FactLensRLMDiscoveryArm("model_graph_guided_rlm")],
        scorers=[FactLensRLMDiscoveryScorer()],
    )
    positive = records[0]
    result = positive.trace[0]

    assert result["coverage_after"] >= result["coverage_before"]
    assert result["model_call_count"] >= 3
    assert {"query_semantics_router_v1", "completeness_audit_v1", "rlm_evidence_discovery_v1"} <= {
        trace["prompt_id"] for trace in positive.model_call_traces
    }
    assert result["false_complete_coverage_rate"] == 0.0


def test_factlens_rlm_discovery_absent_fact_fails_closed() -> None:
    cases = factlens_rlm_discovery_cases([_record(0), _record(1), _record(2)], dataset_revision="rev", cases_total=3)
    absent_case = cases[2]
    record = run_benchmark_cases(
        [absent_case],
        [FactLensRLMDiscoveryArm("model_graph_guided_rlm")],
        scorers=[FactLensRLMDiscoveryScorer()],
    )[0]
    result = record.trace[0]

    assert result["complete_after"] is False
    assert result["unsupported_verdict_rate"] == 1.0
    assert result["false_complete_coverage_rate"] == 0.0
    assert result["stop_reason"] == "corpus_insufficient"


def test_factlens_rlm_discovery_gold_oracle_is_at_least_single_search() -> None:
    cases = factlens_rlm_discovery_cases([_record(0)], dataset_revision="rev", cases_total=1)
    records = run_benchmark_cases(
        cases,
        [
            FactLensRLMDiscoveryArm("scripted_single_search"),
            FactLensRLMDiscoveryArm("gold_search_goal_oracle"),
        ],
        scorers=[FactLensRLMDiscoveryScorer()],
    )
    by_arm = {record.arm: record.trace[0] for record in records}

    assert by_arm["gold_search_goal_oracle"]["coverage_after"] >= by_arm["scripted_single_search"]["coverage_after"]


def _record(index: int) -> FactLensOfficialRecord:
    subclaims = [
        "Mira founded Northstar Labs.",
        "Northstar Labs acquired HelioAI.",
        "HelioAI built search tools for Argo Health.",
        "Argo Health used HelioAI search in Berlin.",
        "Mira is connected to Argo Health through HelioAI.",
        "The Berlin workflow depended on HelioAI search.",
    ]
    return FactLensOfficialRecord(
        ind=str(index),
        claim=" ".join(subclaims),
        sub_claims=subclaims,
        labels=["true"] * len(subclaims),
        aggregated_label=True,
        native_row={},
    )
