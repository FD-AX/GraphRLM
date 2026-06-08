from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.factlens import (
    FactLensAuditArm,
    FactLensAuditScorer,
    factlens_case_from_official_record,
    factlens_repo_revision,
    load_factlens_official_csv,
    select_factlens_matrix_cases,
)
from app.benchmarks.runner import aggregate_records, run_benchmark_cases


MATRIX_MODES = [
    "flat_subclaim_verification",
    "graph_query_verification",
    "graph_shared_evidence",
    "graph_shared_evidence_without_cross_subclaim_edges",
    "graph_shared_evidence_shuffled_edges",
    "graph_shared_evidence_masked_required_fact",
]


def main() -> None:
    args = _parse_args()
    repo_path = args.factlens_repo
    source_csv = args.csv_path or repo_path / "benchmark" / "fact_lens_benchmark.csv"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    revision = factlens_repo_revision(repo_path)
    official_records = load_factlens_official_csv(source_csv)
    selected_records = select_factlens_matrix_cases(official_records, per_bucket=args.per_bucket)
    cases = [
        factlens_case_from_official_record(
            record,
            dataset_revision=revision,
            source_path=source_csv,
        )
        for record in selected_records
    ]
    arms = [
        FactLensAuditArm(mode, experiment_id=args.experiment_id)
        for mode in MATRIX_MODES
    ]
    records = run_benchmark_cases(cases, arms, scorers=[FactLensAuditScorer()])
    summary = build_factlens_matrix_summary(records)
    aggregate = aggregate_records(records, score_name="factlens_supported")

    (output_dir / "benchmark_records.jsonl").write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
    (output_dir / "benchmark_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": args.experiment_id,
        "benchmark": "factlens",
        "dataset": "megagonlabs/factlens",
        "dataset_file": str(source_csv),
        "dataset_origin": "official",
        "requested_revision": "local_clone",
        "resolved_revision": revision,
        "loader_version": "factlens_official_csv_loader_v1",
        "cases_total": len(cases),
        "task_ids": [case.task_id for case in cases],
        "split_contract": {
            "simple": args.per_bucket,
            "medium": args.per_bucket,
            "complex": args.per_bucket,
            "selected_before_arm_execution": True,
            "complexity_attributes": [
                "subclaim_count",
                "shared_entity_count",
                "shared_evidence_count",
                "cross_subclaim_dependency_count",
            ],
        },
        "arms": MATRIX_MODES,
        "score_backends": ["factlens_local_audit_v1"],
        "protocol_note": (
            "External FactLens audit matrix over official benchmark rows. "
            "The public CSV contains subclaims and labels, not retrieved evidence corpus; "
            "coverage metrics audit structured verification completeness, not official retrieval scoring."
        ),
    }
    protocol_freeze = build_protocol_freeze(manifest)
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "factlens_matrix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "factlens_input_parity.json").write_text(
        json.dumps(build_input_parity_report(cases), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "factlens_subclaim_coverage_table.json").write_text(
        json.dumps(build_subclaim_coverage_table(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "factlens_protocol_freeze.json").write_text(
        json.dumps(protocol_freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "factlens_protocol_freeze.md").write_text(
        render_protocol_freeze_markdown(protocol_freeze),
        encoding="utf-8",
    )
    (output_dir / "factlens_matrix_summary.md").write_text(
        render_summary_markdown(summary, manifest),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2))


def build_factlens_matrix_summary(records) -> dict:
    by_arm = _group(records, lambda record: record.arm)
    arm_rows = {
        arm: _metrics_for_records(items)
        for arm, items in sorted(by_arm.items())
    }
    return {
        "arms": arm_rows,
        "bootstrap_ci": {
            arm: bootstrap_metric_cis(items)
            for arm, items in sorted(by_arm.items())
        },
        "binomial_wilson_ci": {
            arm: wilson_metric_cis(items)
            for arm, items in sorted(by_arm.items())
        },
        "by_complexity": _breakdown(records, "complexity_bucket"),
        "by_subclaim_count": _breakdown(records, "subclaim_count_bucket"),
        "by_shared_evidence_requirement": _breakdown(records, "shared_evidence_required"),
        "by_cross_subclaim_dependency": _breakdown(records, "cross_dependency_bucket"),
        "deltas_vs_flat": _deltas(records, "flat_subclaim_verification", "graph_shared_evidence"),
        "negative_controls": _negative_controls(records),
    }


def build_protocol_freeze(manifest: dict) -> dict:
    return {
        "protocol_id": "factlens_external_structured_completeness_audit_v1",
        "frozen": True,
        "dataset": manifest["dataset"],
        "resolved_revision": manifest["resolved_revision"],
        "task_ids": manifest["task_ids"],
        "stratification": manifest["split_contract"],
        "arms": manifest["arms"],
        "edge_construction_rules": {
            "fact_extraction": "Normalize claim/subclaim tokens into deterministic term:<token> graph facts.",
            "edge_rule": "Create cross-subclaim edge when two subclaims share a graph_fact_id.",
            "valid_edge_rule": "An edge is usable only if graph_fact_id appears in both endpoint subclaims and both endpoints have evidence spans.",
            "shared_evidence_rule": "Shared graph mode starts from independently available graph-query subclaims and propagates support only over valid edges.",
        },
        "metric_definitions": {
            "complete_evidence_coverage": "All required subclaims for a claim are covered by the mode.",
            "unsupported_verdict_rate": "1 when the mode emits a claim verdict without complete evidence coverage, else 0.",
            "macro_subclaim_recall": "Mean of per-claim subclaim recall; each claim has equal weight.",
            "fully_verified_claims_per_10k_tokens": "Complete claims divided by token budget, normalized per 10k tokens.",
        },
        "negative_controls": {
            "without_cross_subclaim_edges": "Remove all cross-subclaim edges while keeping the same subclaims/evidence pool.",
            "shuffled_edges": "Rotate and invalidate edge targets/facts to test whether arbitrary edges preserve the effect.",
            "masked_required_fact": "Remove one required evidence span per claim; complete coverage must fail.",
        },
        "masking_policy": "Mask the final required subclaim evidence span for each claim without changing claim text or remaining subclaims.",
        "scope_note": (
            "This is a structured completeness audit over official FactLens rows. "
            "It is not an official corpus-level retrieval benchmark because the public CSV provides "
            "gold subclaims and labels, not an external evidence corpus."
        ),
    }


def build_input_parity_report(cases) -> dict:
    selected = []
    for bucket in ("medium", "complex"):
        case = next(
            item
            for item in cases
            if item.metadata.get("factlens_complexity", {}).get("complexity_bucket") == bucket
        )
        factlens = case.metadata.get("factlens", {})
        common = {
            "task_id": case.task_id,
            "complexity_bucket": bucket,
            "claim": case.question,
            "subclaims": [
                {
                    "subclaim_id": item.get("subclaim_id"),
                    "text": item.get("text"),
                    "label_visible_to_audit": item.get("label"),
                    "evidence_span_ids": item.get("evidence_span_ids", []),
                    "graph_fact_ids": item.get("graph_fact_ids", []),
                }
                for item in factlens.get("subclaims", [])
            ],
            "token_budget_proxy": case.context_tokens + len(case.question.split()) + 1,
            "evidence_candidates": case.gold_evidence_span_ids,
        }
        selected.append(
            {
                "bucket": bucket,
                "flat_input": {
                    **common,
                    "cross_subclaim_edges": [],
                },
                "graph_query_input": {
                    **common,
                    "cross_subclaim_edges": factlens.get("graph_edges", []),
                },
                "graph_shared_evidence_input": {
                    **common,
                    "cross_subclaim_edges": factlens.get("graph_edges", []),
                },
                "parity_assertions": {
                    "same_claim": True,
                    "same_subclaims": True,
                    "same_labels": True,
                    "same_evidence_candidates": True,
                    "same_token_budget_proxy": True,
                    "graph_only_extra": "cross_subclaim_edges",
                },
            }
        )
    return {"cases": selected}


def build_subclaim_coverage_table(records) -> list[dict]:
    rows = []
    by_task = _group(records, lambda record: record.task_id)
    for task_id, items in sorted(by_task.items()):
        by_arm = {record.arm: record for record in items}
        shared_audit = _audit(by_arm["graph_shared_evidence"])
        subclaim_ids = [item["subclaim_id"] for item in shared_audit.get("subclaim_coverage", [])]
        for subclaim_id in subclaim_ids:
            row = {"task_id": task_id, "subclaim_id": subclaim_id, "gold_required": True}
            for arm in MATRIX_MODES:
                audit = _audit(by_arm[arm])
                coverage = {
                    item["subclaim_id"]: item
                    for item in audit.get("subclaim_coverage", [])
                }
                item = coverage.get(subclaim_id, {})
                row[f"{arm}_found"] = bool(item.get("found"))
                row[f"{arm}_source_of_support"] = item.get("source_of_support")
            rows.append(row)
    return rows


def render_summary_markdown(summary: dict, manifest: dict) -> str:
    lines = [
        "# FactLens External Matrix",
        "",
        f"- dataset: `{manifest['dataset']}`",
        f"- resolved_revision: `{manifest['resolved_revision']}`",
        f"- cases_total: `{manifest['cases_total']}`",
        f"- arms: `{', '.join(manifest['arms'])}`",
        "",
        "## Arm Summary",
        "",
        "| Arm | Complete coverage | Unsupported verdict | Macro subclaim recall | Fully verified / 10k tokens | Total tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, metrics in summary["arms"].items():
        lines.append(
            "| "
            f"{arm} | "
            f"{metrics['complete_evidence_coverage_rate']:.3f} | "
            f"{metrics['unsupported_verdict_rate']:.3f} | "
            f"{metrics['macro_subclaim_recall']:.3f} | "
            f"{metrics['fully_verified_claims_per_10k_tokens']:.3f} | "
            f"{metrics['total_tokens']} |"
        )
    lines.extend(["", "## Bootstrap 95% CI", ""])
    lines.extend(
        [
            "| Arm | Complete coverage | Unsupported verdict | Macro subclaim recall |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm, metrics in summary["bootstrap_ci"].items():
        lines.append(
            "| "
            f"{arm} | "
            f"{_ci_text(metrics['complete_evidence_coverage_rate'])} | "
            f"{_ci_text(metrics['unsupported_verdict_rate'])} | "
            f"{_ci_text(metrics['macro_subclaim_recall'])} |"
        )
    lines.extend(["", "## Wilson 95% CI For Binary Rates", ""])
    lines.extend(
        [
            "| Arm | Complete coverage | Unsupported verdict |",
            "| --- | ---: | ---: |",
        ]
    )
    for arm, metrics in summary["binomial_wilson_ci"].items():
        lines.append(
            "| "
            f"{arm} | "
            f"{_ci_text(metrics['complete_evidence_coverage_rate'])} | "
            f"{_ci_text(metrics['unsupported_verdict_rate'])} |"
        )
    lines.extend(["", "## Gain By Complexity", ""])
    lines.extend(_render_delta_table(summary["deltas_vs_flat"]["by_complexity"], "Complexity"))
    lines.extend(["", "## Gain By Subclaim Count", ""])
    lines.extend(_render_delta_table(summary["deltas_vs_flat"]["by_subclaim_count"], "Subclaims"))
    lines.extend(["", "## Gain By Shared Evidence Requirement", ""])
    lines.extend(_render_delta_table(summary["deltas_vs_flat"]["by_shared_evidence_requirement"], "Shared evidence"))
    lines.extend(["", "## Gain By Cross-Subclaim Dependency", ""])
    lines.extend(_render_delta_table(summary["deltas_vs_flat"]["by_cross_subclaim_dependency"], "Dependency"))
    return "\n".join(lines) + "\n"


def render_protocol_freeze_markdown(protocol: dict) -> str:
    lines = [
        "# FactLens Protocol Freeze",
        "",
        f"- protocol_id: `{protocol['protocol_id']}`",
        f"- dataset: `{protocol['dataset']}`",
        f"- resolved_revision: `{protocol['resolved_revision']}`",
        f"- frozen: `{protocol['frozen']}`",
        f"- task_count: `{len(protocol['task_ids'])}`",
        "",
        "## Scope",
        "",
        protocol["scope_note"],
        "",
        "## Arms",
        "",
    ]
    lines.extend(f"- `{arm}`" for arm in protocol["arms"])
    lines.extend(["", "## Edge Construction Rules", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in protocol["edge_construction_rules"].items())
    lines.extend(["", "## Metric Definitions", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in protocol["metric_definitions"].items())
    lines.extend(["", "## Negative Controls", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in protocol["negative_controls"].items())
    lines.extend(["", "## Frozen Task IDs", ""])
    lines.extend(f"- `{task_id}`" for task_id in protocol["task_ids"])
    return "\n".join(lines) + "\n"


def _render_delta_table(rows: list[dict], label: str) -> list[str]:
    lines = [
        f"| {label} | Flat complete coverage | Shared graph coverage | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['value']} | "
            f"{row['flat_complete_evidence_coverage_rate']:.3f} | "
            f"{row['graph_shared_evidence_complete_evidence_coverage_rate']:.3f} | "
            f"{row['delta']:+.3f} |"
        )
    return lines


def _breakdown(records, dimension: str) -> dict:
    by_key = _group(records, lambda record: str(_complexity(record).get(dimension)))
    return {
        value: {
            arm: _metrics_for_records([record for record in items if record.arm == arm])
            for arm in MATRIX_MODES
        }
        for value, items in sorted(by_key.items())
    }


def _deltas(records, flat_arm: str, graph_arm: str) -> dict:
    dimensions = {
        "by_complexity": "complexity_bucket",
        "by_subclaim_count": "subclaim_count_bucket",
        "by_shared_evidence_requirement": "shared_evidence_required",
        "by_cross_subclaim_dependency": "cross_dependency_bucket",
    }
    result = {}
    for name, dimension in dimensions.items():
        rows = []
        by_value = _group(records, lambda record: str(_complexity(record).get(dimension)))
        for value, items in sorted(by_value.items()):
            flat = _metrics_for_records([record for record in items if record.arm == flat_arm])
            graph = _metrics_for_records([record for record in items if record.arm == graph_arm])
            rows.append(
                {
                    "value": value,
                    "flat_complete_evidence_coverage_rate": flat["complete_evidence_coverage_rate"],
                    "graph_shared_evidence_complete_evidence_coverage_rate": graph["complete_evidence_coverage_rate"],
                    "delta": graph["complete_evidence_coverage_rate"] - flat["complete_evidence_coverage_rate"],
                }
            )
        result[name] = rows
    return result


def _negative_controls(records) -> dict:
    controls = {}
    for arm in (
        "graph_shared_evidence",
        "graph_shared_evidence_without_cross_subclaim_edges",
        "graph_shared_evidence_shuffled_edges",
        "graph_shared_evidence_masked_required_fact",
    ):
        controls[arm] = _metrics_for_records([record for record in records if record.arm == arm])
    return controls


def bootstrap_metric_cis(records, *, samples: int = 2000, seed: int = 17) -> dict:
    if not records:
        return {
            "complete_evidence_coverage_rate": _ci(0.0, 0.0, 0.0),
            "unsupported_verdict_rate": _ci(0.0, 0.0, 0.0),
            "macro_subclaim_recall": _ci(0.0, 0.0, 0.0),
        }
    rng = random.Random(seed)
    observed = _metrics_for_records(records)
    boot = {
        "complete_evidence_coverage_rate": [],
        "unsupported_verdict_rate": [],
        "macro_subclaim_recall": [],
    }
    for _ in range(samples):
        sample = [records[rng.randrange(len(records))] for _ in records]
        metrics = _metrics_for_records(sample)
        for key in boot:
            boot[key].append(metrics[key])
    return {
        key: _ci(
            observed[key],
            _percentile(values, 0.025),
            _percentile(values, 0.975),
        )
        for key, values in boot.items()
    }


def wilson_metric_cis(records) -> dict:
    audits = [_audit(record) for record in records]
    total = len(audits)
    if not total:
        return {
            "complete_evidence_coverage_rate": _ci(0.0, 0.0, 0.0, method="wilson_score"),
            "unsupported_verdict_rate": _ci(0.0, 0.0, 0.0, method="wilson_score"),
        }
    complete_successes = sum(1 for audit in audits if audit.get("complete_evidence_coverage"))
    unsupported_successes = sum(1 for audit in audits if float(audit.get("unsupported_verdict_rate") or 0.0) >= 1.0)
    return {
        "complete_evidence_coverage_rate": _wilson_ci(complete_successes, total),
        "unsupported_verdict_rate": _wilson_ci(unsupported_successes, total),
    }


def _metrics_for_records(records) -> dict:
    audits = [_audit(record) for record in records]
    if not records:
        return {
            "runs_total": 0,
            "complete_evidence_coverage_rate": 0.0,
            "unsupported_verdict_rate": 0.0,
            "macro_subclaim_recall": 0.0,
            "fully_verified_claims_per_10k_tokens": 0.0,
            "total_tokens": 0,
        }
    return {
        "runs_total": len(records),
        "complete_evidence_coverage_rate": _mean(
            1.0 if audit.get("complete_evidence_coverage") else 0.0 for audit in audits
        ),
        "unsupported_verdict_rate": _mean(float(audit.get("unsupported_verdict_rate") or 0.0) for audit in audits),
        "macro_subclaim_recall": _mean(float(audit.get("subclaim_recall") or 0.0) for audit in audits),
        "fully_verified_claims_per_10k_tokens": _mean(
            float(audit.get("fully_verified_claims_per_10k_tokens") or 0.0) for audit in audits
        ),
        "total_tokens": sum(record.total_tokens for record in records),
    }


def _ci(mean_value: float, low: float, high: float, *, method: str = "nonparametric_bootstrap") -> dict:
    return {
        "mean": round(mean_value, 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "method": method,
        "confidence": 0.95,
    }


def _wilson_ci(successes: int, total: int) -> dict:
    if total <= 0:
        return _ci(0.0, 0.0, 0.0, method="wilson_score")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + (z * z / total)
    center = (p + (z * z) / (2.0 * total)) / denominator
    margin = (
        z
        * ((p * (1.0 - p) / total + (z * z) / (4.0 * total * total)) ** 0.5)
        / denominator
    )
    return _ci(
        p,
        max(0.0, center - margin),
        min(1.0, center + margin),
        method="wilson_score",
    )


def _ci_text(value: dict) -> str:
    return f"{value['mean']:.3f} [{value['low']:.3f}, {value['high']:.3f}]"


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(round((len(ordered) - 1) * quantile), 0), len(ordered) - 1)
    return ordered[index]


def _audit(record) -> dict:
    if record.trace:
        audit = record.trace[0].get("factlens_audit")
        if isinstance(audit, dict):
            return audit
    return {}


def _complexity(record) -> dict:
    return dict(record.case_metadata.get("factlens_complexity") or {})


def _group(records, key_fn) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[key_fn(record)].append(record)
    return grouped


def _mean(values) -> float:
    collected = list(values)
    return round(mean(collected), 6) if collected else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factlens-repo",
        type=Path,
        default=Path("C:/tmp/factlens_official_repo"),
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/factlens_external_matrix_v1"),
    )
    parser.add_argument("--per-bucket", type=int, default=10)
    parser.add_argument("--experiment-id", default="factlens_external_matrix_v1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
