from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing required metrics artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def check_factlens_matrix(root: Path) -> None:
    summary = _load_json(root / "factlens_external_matrix" / "factlens_matrix_summary.json")
    arms = summary["arms"]
    _require(
        arms["graph_shared_evidence"]["complete_evidence_coverage_rate"] >= 1.0,
        "FactLens shared-evidence graph must keep complete coverage at 1.0",
    )
    _require(
        arms["graph_shared_evidence"]["unsupported_verdict_rate"] == 0.0,
        "FactLens shared-evidence graph must not emit unsupported verdicts",
    )
    _require(
        arms["graph_shared_evidence_masked_required_fact"]["complete_evidence_coverage_rate"] == 0.0,
        "Masked required fact control must fail closed",
    )


def check_rlm_discovery(root: Path) -> None:
    summary = _load_json(root / "factlens_rlm_discovery" / "summary.json")
    arms = summary["arms"]
    iterative = arms["scripted_iterative_search"]
    single = arms["scripted_single_search"]
    _require(
        iterative["missing_evidence_recovery_rate"] >= single["missing_evidence_recovery_rate"],
        "Iterative discovery must recover at least as much missing evidence as single search",
    )
    _require(
        iterative["false_complete_coverage_rate"] == 0.0,
        "Iterative discovery must keep false complete coverage at 0.0",
    )
    _require(
        iterative["rlm_calls_when_no_missing_slots"] == 0,
        "RLM discovery must not run when no missing slots exist",
    )
    if "contract_model_prompts" in arms and "generic_model_prompt" in arms:
        contract = arms["contract_model_prompts"]
        generic = arms["generic_model_prompt"]
        _require(
            contract["false_complete_coverage_rate"] == 0.0,
            "Contract model prompts must keep false complete coverage at 0.0",
        )
        _require(
            contract["missing_evidence_recovery_rate"] >= generic["missing_evidence_recovery_rate"],
            "Contract model prompts should not underperform generic prompts on recovery",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts/metrics_run")
    args = parser.parse_args()
    root = Path(args.artifact_root)
    check_factlens_matrix(root)
    check_rlm_discovery(root)
    print(json.dumps({"artifact_root": str(root), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
