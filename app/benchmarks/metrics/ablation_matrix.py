from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from app.benchmarks.metrics import build_evaluation_report


DEFAULT_ABLATION_DIRS = {
    "direct_model": [
        Path("artifacts/research_matrix/oolong_synth_stratified5/direct_gpt/repeat_0"),
        Path("artifacts/research_matrix/oolong_synth_stratified5/direct_gpt/repeat_1"),
    ],
    "raw_records_ops": [
        Path("artifacts/research_matrix/oolong_synth_stratified5/raw_records_ops_v2/repeat_0"),
        Path("artifacts/research_matrix/oolong_synth_stratified5/raw_records_ops_v2/repeat_1"),
    ],
    "semantic_graph_ops": [
        Path("artifacts/research_matrix/oolong_synth_stratified5/graph_rlm_semantic_graph_ops_v2/repeat_0"),
        Path("artifacts/research_matrix/oolong_synth_stratified5/graph_rlm_semantic_graph_ops_v2/repeat_1"),
    ],
    "semantic_graph_rlm": [
        Path("artifacts/research_matrix/oolong_synth_stratified5/graph_rlm_semantic_graph/repeat_0"),
        Path("artifacts/research_matrix/oolong_synth_stratified5/graph_rlm_semantic_graph/repeat_1"),
    ],
}


ARM_DESCRIPTIONS = {
    "direct_model": "whole raw context -> GPT-5",
    "raw_records_ops": "raw records -> planner -> executor",
    "semantic_graph_ops": "graph facts -> planner -> executor",
    "semantic_graph_rlm": "graph retrieval -> recursive reasoning",
}


def build_ablation_matrix(
    arm_dirs: dict[str, list[Path]] | None = None,
) -> dict[str, Any]:
    arm_dirs = arm_dirs or DEFAULT_ABLATION_DIRS
    paths = [path for paths in arm_dirs.values() for path in paths]
    report = build_evaluation_report(paths)
    rows_by_artifact = {
        str(Path(row["artifact_dir"])): row
        for row in report["runs"]
    }
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm_label, dirs in arm_dirs.items():
        selected = [
            row
            for row in report["runs"]
            if Path(row["artifact_dir"]) in dirs
        ]
        rows_by_arm[arm_label] = selected

    matrix_rows = [_summarize_arm(arm, rows) for arm, rows in rows_by_arm.items()]
    task_rows = _task_matrix(rows_by_arm)
    comparison = _pairwise_comparison(rows_by_arm)
    payload = {
        "arm_descriptions": ARM_DESCRIPTIONS,
        "matrix_rows": matrix_rows,
        "task_rows": task_rows,
        "pairwise_comparison": comparison,
        "graph_ops_forensics": _read_graph_ops_forensics(),
        "artifact_dirs": {
            arm: [str(path) for path in paths]
            for arm, paths in arm_dirs.items()
        },
        "raw_run_rows": rows_by_artifact,
    }
    return payload


def write_ablation_matrix(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation_matrix.json"
    markdown_path = output_dir / "ablation_matrix.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_ablation_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_ablation_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OOLONG Ablation Matrix",
        "",
        "## Arm Definitions",
        "",
        "| Arm | What it uses |",
        "|---|---|",
    ]
    for arm, description in payload["arm_descriptions"].items():
        lines.append(f"| `{arm}` | {description} |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Arm | Score | Runs | Total tokens | Root tokens | Worker tokens | Tokens / correct | Fallbacks | Invalid artifacts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["matrix_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['arm']}`",
                    f"{row['score_mean']:.3f}",
                    str(row["runs_total"]),
                    f"{row['total_tokens']:,}",
                    f"{row['root_tokens']:,}",
                    f"{row['worker_tokens']:,}",
                    "n/a" if row["tokens_per_correct"] is None else f"{row['tokens_per_correct']:,.0f}",
                    str(row["fallback_count"]),
                    str(row["invalid_artifact_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Comparison",
            "",
            "```json",
            json.dumps(payload["pairwise_comparison"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Graph Ops Forensics",
            "",
            "```json",
            json.dumps(payload.get("graph_ops_forensics") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Per Task",
            "",
            "| Task ID | direct_model | raw_records_ops | semantic_graph_ops | semantic_graph_rlm |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["task_rows"]:
        lines.append(
            f"| `{row['task_id']}` | {row.get('direct_model', '')} | {row.get('raw_records_ops', '')} | "
            f"{row.get('semantic_graph_ops', '')} | {row.get('semantic_graph_rlm', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _summarize_arm(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_sum = sum(float(row["active_score"] or 0) for row in rows)
    correct = sum(1 for row in rows if float(row["active_score"] or 0) >= 1.0)
    total_tokens = sum(int(row["total_tokens"] or 0) for row in rows)
    latencies = [int(row["latency_ms"] or 0) for row in rows]
    return {
        "arm": arm,
        "description": ARM_DESCRIPTIONS.get(arm, ""),
        "runs_total": len(rows),
        "score_sum": score_sum,
        "score_mean": score_sum / len(rows) if rows else 0.0,
        "correct_count": correct,
        "total_tokens": total_tokens,
        "root_tokens": sum(int(row["root_tokens"] or 0) for row in rows),
        "worker_tokens": sum(int(row["worker_tokens"] or 0) for row in rows),
        "tokens_per_correct": total_tokens / correct if correct else None,
        "latency_mean_ms": mean(latencies) if latencies else 0.0,
        "fallback_count": sum(1 for row in rows if row.get("fallback_used")),
        "invalid_artifact_count": sum(1 for row in rows if not row.get("artifact_valid")),
        "missing_response_id_count": sum(
            1
            for row in rows
            if row.get("provider") == "openai" and not row.get("response_id")
        ),
    }


def _task_matrix(rows_by_arm: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for arm, rows in rows_by_arm.items():
        for row in rows:
            grouped[str(row["task_id"])][arm].append(float(row["active_score"] or 0))
    task_rows = []
    for task_id, arm_scores in sorted(grouped.items()):
        task_rows.append(
            {
                "task_id": task_id,
                **{
                    arm: round(sum(scores) / len(scores), 3)
                    for arm, scores in arm_scores.items()
                },
            }
        )
    return task_rows


def _pairwise_comparison(rows_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    raw = _summarize_arm("raw_records_ops", rows_by_arm.get("raw_records_ops", []))
    graph_ops = _summarize_arm("semantic_graph_ops", rows_by_arm.get("semantic_graph_ops", []))
    graph_rlm = _summarize_arm("semantic_graph_rlm", rows_by_arm.get("semantic_graph_rlm", []))
    return {
        "raw_records_ops_vs_semantic_graph_ops": {
            "score_delta_graph_minus_raw": graph_ops["score_sum"] - raw["score_sum"],
            "token_delta_graph_minus_raw": graph_ops["total_tokens"] - raw["total_tokens"],
            "graph_token_multiplier": (
                graph_ops["total_tokens"] / raw["total_tokens"]
                if raw["total_tokens"]
                else None
            ),
            "interpretation": (
                "same_score_graph_more_expensive"
                if graph_ops["score_sum"] == raw["score_sum"] and graph_ops["total_tokens"] > raw["total_tokens"]
                else "different_result"
            ),
        },
        "semantic_graph_ops_vs_semantic_graph_rlm": {
            "score_delta_ops_minus_rlm": graph_ops["score_sum"] - graph_rlm["score_sum"],
            "token_reduction_ops_vs_rlm": (
                1 - graph_ops["total_tokens"] / graph_rlm["total_tokens"]
                if graph_rlm["total_tokens"]
                else None
            ),
        },
    }


def _read_graph_ops_forensics() -> dict[str, Any] | None:
    path = Path("artifacts/research_matrix/oolong_synth_stratified5/graph_ops_forensics_v1/graph_ops_forensics.json")
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("summary")
