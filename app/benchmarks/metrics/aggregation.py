from __future__ import annotations

from collections import defaultdict
from statistics import mean


def aggregate_by_arm(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    output = []
    for arm, items in sorted(grouped.items()):
        output.append(
            {
                "arm": arm,
                "runs_total": len(items),
                "artifact_valid_count": sum(1 for item in items if item["artifact_valid"]),
                "active_score_mean": _mean(item["active_score"] for item in items),
                "original_score_mean": _mean(item["original_score"] for item in items),
                "tokens_total": sum(item["total_tokens"] for item in items),
                "tokens_mean": _mean(item["total_tokens"] for item in items),
                "root_tokens_total": sum(item["root_tokens"] for item in items),
                "worker_tokens_total": sum(item["worker_tokens"] for item in items),
                "latency_mean_ms": _mean(item["latency_ms"] for item in items),
                "context_amplification_mean": _mean(
                    item["context_amplification"]
                    for item in items
                    if item["context_amplification"] is not None
                ),
            }
        )
    return output


def score_vs_cost(rows: list[dict]) -> list[dict]:
    return [
        {
            "task_id": row["task_id"],
            "arm": row["arm"],
            "active_score": row["active_score"],
            "original_score": row["original_score"],
            "total_tokens": row["total_tokens"],
            "root_tokens": row["root_tokens"],
            "worker_tokens": row["worker_tokens"],
            "latency_ms": row["latency_ms"],
            "context_amplification": row["context_amplification"],
            "artifact_valid": row["artifact_valid"],
        }
        for row in rows
    ]


def _mean(values) -> float:
    collected = [value for value in values if value is not None]
    return round(mean(collected), 6) if collected else 0.0
