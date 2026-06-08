from __future__ import annotations


def render_markdown(report: dict) -> str:
    lines = [
        "# Evaluation Report V1",
        "",
        "| task_id | arm | original_score | active_score | original_scorer | active_scorer | total_tokens | root_tokens | worker_tokens | valid |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["runs"]:
        lines.append(
            "| {task_id} | {arm} | {original_score} | {active_score} | {original_scorer} | {active_scorer} | {total_tokens} | {root_tokens} | {worker_tokens} | {artifact_valid} |".format(
                **row
            )
        )
    lines.extend(["", "## By Arm", ""])
    lines.append("| arm | runs | original_score_mean | active_score_mean | tokens_total | context_amp_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["by_arm"]:
        lines.append(
            "| {arm} | {runs_total} | {original_score_mean} | {active_score_mean} | {tokens_total} | {context_amplification_mean} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"
