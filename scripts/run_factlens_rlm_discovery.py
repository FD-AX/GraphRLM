from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.factlens import (
    FactLensRLMDiscoveryArm,
    FactLensRLMDiscoveryScorer,
    factlens_repo_revision,
    factlens_rlm_discovery_cases,
    load_factlens_official_csv,
)
from app.benchmarks.runner import run_benchmark_cases


DISCOVERY_ARMS = [
    "scripted_single_search",
    "scripted_iterative_search",
    "model_graph_guided_rlm",
    "gold_search_goal_oracle",
]

MODEL_CONTROLLER_ARMS = [
    "generic_model_prompt",
    "contract_model_prompts",
]


def main() -> None:
    args = _parse_args()
    source_csv = args.factlens_repo / "benchmark" / "fact_lens_benchmark.csv"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    revision = factlens_repo_revision(args.factlens_repo)
    official_records = load_factlens_official_csv(source_csv)
    cases = factlens_rlm_discovery_cases(
        official_records,
        dataset_revision=revision,
        cases_total=args.cases,
    )
    arm_names = list(DISCOVERY_ARMS)
    if args.include_model_controllers:
        arm_names.extend(MODEL_CONTROLLER_ARMS)
    arms = [
        FactLensRLMDiscoveryArm(
            arm,
            model_name=args.model_name if arm in MODEL_CONTROLLER_ARMS else None,
            reasoning_effort=args.reasoning_effort if arm in MODEL_CONTROLLER_ARMS else None,
            experiment_id=args.experiment_id,
        )
        for arm in arm_names
    ]
    records = []
    for repeat_index in range(args.repeat):
        run_records = run_benchmark_cases(cases, arms, scorers=[FactLensRLMDiscoveryScorer()])
        for record in run_records:
            record.case_metadata["repeat_index"] = repeat_index
        records.extend(run_records)
    manifest = _manifest(cases, revision, source_csv, arm_names=arm_names, args=args)
    summary = _summary(records)

    _write_jsonl(output_dir / "benchmark_records.jsonl", [record.model_dump(mode="json") for record in records])
    _write_jsonl(output_dir / "model_call_traces.jsonl", _model_call_rows(records))
    _write_jsonl(output_dir / "search_goal_traces.jsonl", _search_goal_rows(records))
    _write_jsonl(output_dir / "coverage_transitions.jsonl", _coverage_rows(records))
    (output_dir / "protocol_freeze.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "corpus_manifest.json").write_text(json.dumps(_corpus_manifest(cases), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "negative_controls.json").write_text(json.dumps(summary["negative_controls"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary_markdown(summary, manifest), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2))


def _summary(records) -> dict:
    by_arm = defaultdict(list)
    for record in records:
        by_arm[record.arm].append(record)
    arms = {arm: _arm_metrics(items) for arm, items in sorted(by_arm.items())}
    return {
        "arms": arms,
        "negative_controls": {
            arm: {
                "false_complete_coverage_rate": metrics["false_complete_coverage_rate"],
                "unsupported_verdict_rate": metrics["unsupported_verdict_rate"],
                "absent_fact_cases_completed": metrics["absent_fact_cases_completed"],
            }
            for arm, metrics in arms.items()
        },
        "prompt_versions": [
            "query_semantics_router_v1",
            "completeness_audit_v1",
            "rlm_evidence_discovery_v1",
            "graph_guided_rlm_search_v1",
        ],
    }


def _arm_metrics(records) -> dict:
    results = [_result(record) for record in records]
    absent = [result for result in results if result["hidden_required_evidence_in_corpus_ids"] != result["hidden_required_evidence_ids"]]
    return {
        "runs_total": len(records),
        "missing_evidence_recovery_rate": _mean(result["missing_evidence_recovery_rate"] for result in results),
        "all_hidden_required_evidence_recovered_rate": _mean(1.0 if result["all_hidden_required_evidence_recovered"] else 0.0 for result in results),
        "required_evidence_recall": _mean(result["required_evidence_recall"] for result in results),
        "accepted_evidence_precision": _mean(result["accepted_evidence_precision"] for result in results),
        "coverage_before_rlm": _mean(result["coverage_before"] for result in results),
        "coverage_after_rlm": _mean(result["coverage_after"] for result in results),
        "coverage_gain": _mean(result["coverage_gain"] for result in results),
        "claims_completed_after_rlm": _mean(1.0 if result["complete_after"] else 0.0 for result in results),
        "useful_search_rate": _mean(result["useful_search_rate"] for result in results),
        "redundant_search_rate": _mean(result["redundant_search_rate"] for result in results),
        "unsupported_verdict_rate": _mean(result["unsupported_verdict_rate"] for result in results),
        "false_complete_coverage_rate": _mean(result["false_complete_coverage_rate"] for result in results),
        "tokens_per_recovered_fact": _mean(
            result["tokens_per_recovered_fact"]
            for result in results
            if result["tokens_per_recovered_fact"] is not None
        ),
        "total_tokens": sum(record.total_tokens for record in records),
        "absent_fact_cases_completed": sum(1 for result in absent if result["complete_after"]),
        "rlm_calls_when_no_missing_slots": sum(
            1
            for result in results
            if result["coverage_before"] >= 1.0 and result["model_call_count"] > 2
        ),
    }


def _model_call_rows(records) -> list[dict]:
    rows = []
    for record in records:
        for index, trace in enumerate(record.model_call_traces):
            rows.append({"task_id": record.task_id, "arm": record.arm, "call_index": index, **trace})
    return rows


def _search_goal_rows(records) -> list[dict]:
    rows = []
    for record in records:
        for trace in _result(record)["search_traces"]:
            rows.append({"task_id": record.task_id, "arm": record.arm, **trace})
    return rows


def _coverage_rows(records) -> list[dict]:
    return [
        {
            "task_id": record.task_id,
            "arm": record.arm,
            "coverage_before": _result(record)["coverage_before"],
            "coverage_after": _result(record)["coverage_after"],
            "coverage_gain": _result(record)["coverage_gain"],
            "stop_reason": _result(record)["stop_reason"],
        }
        for record in records
    ]


def _manifest(cases, revision: str, source_csv: Path, *, arm_names: list[str], args) -> dict:
    return {
        "experiment_id": "factlens_rlm_discovery_v1",
        "benchmark": "factlens_rlm_discovery",
        "dataset": "megagonlabs/factlens",
        "dataset_file": str(source_csv),
        "resolved_revision": revision,
        "cases_total": len(cases),
        "task_ids": [case.task_id for case in cases],
        "arms": arm_names,
        "repeat": args.repeat,
        "model_name": args.model_name if args.include_model_controllers else None,
        "reasoning_effort": args.reasoning_effort if args.include_model_controllers else None,
        "prompt_versions": [
            "query_semantics_router_v1",
            "completeness_audit_v1",
            "rlm_evidence_discovery_v1",
            "graph_guided_rlm_search_v1",
        ],
        "invariants": {
            "no_rlm_when_missing_evidence_slots_empty": True,
            "frozen_completeness_evaluator_reused_after_graph_update": True,
            "false_complete_coverage_rate_target": 0.0,
        },
        "protocol_note": "Controlled local-corpus missing-evidence discovery benchmark; not internet retrieval and not official FactLens scoring.",
    }


def _corpus_manifest(cases) -> dict:
    rows = []
    for case in cases:
        payload = case.metadata["factlens_rlm_discovery"]
        rows.append(
            {
                "task_id": case.task_id,
                "required_evidence_count": len(payload["subclaims"]),
                "hidden_required_evidence_ids": payload["hidden_required_evidence_ids"],
                "initial_supported_subclaim_ids": payload["initial_supported_subclaim_ids"],
                "corpus_size": sum(1 for item in payload["corpus"].values() if item["in_corpus"]),
                "absent_required_fact_control": payload["absent_required_fact_control"],
                "dependency_case": payload["dependency_case"],
            }
        )
    return {"cases": rows}


def _summary_markdown(summary: dict, manifest: dict) -> str:
    lines = [
        "# FactLens RLM Discovery",
        "",
        f"- resolved_revision: `{manifest['resolved_revision']}`",
        f"- cases_total: `{manifest['cases_total']}`",
        f"- arms: `{', '.join(manifest['arms'])}`",
        "",
        "| Arm | Recovery | Complete after | Coverage gain | False complete | Tokens / recovered fact | Total tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, metrics in summary["arms"].items():
        lines.append(
            "| "
            f"{arm} | "
            f"{metrics['missing_evidence_recovery_rate']:.3f} | "
            f"{metrics['claims_completed_after_rlm']:.3f} | "
            f"{metrics['coverage_gain']:.3f} | "
            f"{metrics['false_complete_coverage_rate']:.3f} | "
            f"{metrics['tokens_per_recovered_fact']:.3f} | "
            f"{metrics['total_tokens']} |"
        )
    lines.extend(["", "## Safety", ""])
    lines.append("`false_complete_coverage_rate` must remain `0.0`.")
    return "\n".join(lines) + "\n"


def _result(record) -> dict:
    return json.loads(record.raw_response or "{}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _mean(values) -> float:
    collected = list(values)
    return round(mean(collected), 6) if collected else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factlens-repo", type=Path, default=Path("C:/tmp/factlens_official_repo"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/factlens_rlm_discovery_v1"))
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--include-model-controllers", action="store_true")
    parser.add_argument("--model-name", default="gpt-5-mini")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--experiment-id", default="factlens_rlm_discovery_v1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
