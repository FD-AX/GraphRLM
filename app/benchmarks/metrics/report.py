from __future__ import annotations

import json
from pathlib import Path

from app.benchmarks.metrics.aggregation import aggregate_by_arm, score_vs_cost
from app.benchmarks.metrics.artifacts import read_manifest, read_records, read_score_revisions
from app.benchmarks.metrics.render import render_markdown
from app.benchmarks.metrics.run_metrics import build_run_metric_row


def build_evaluation_report(
    artifact_dirs: list[Path],
    *,
    score_name: str = "task_score",
) -> dict:
    rows = []
    for artifact_dir in artifact_dirs:
        manifest = read_manifest(artifact_dir)
        revisions = read_score_revisions(artifact_dir)
        for record in read_records(artifact_dir):
            rows.append(
                build_run_metric_row(
                    record,
                    artifact_dir=artifact_dir,
                    manifest=manifest,
                    revisions=revisions,
                    score_name=score_name,
                )
            )
    return {
        "report_version": "evaluation_report_v1",
        "score_name": score_name,
        "runs": rows,
        "by_arm": aggregate_by_arm(rows),
        "score_vs_cost": score_vs_cost(rows),
    }


def write_evaluation_report(report: dict, output_json: Path, output_markdown: Path | None = None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_markdown:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_markdown(report), encoding="utf-8")
