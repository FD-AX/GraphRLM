from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_rlm_rows(record_paths: list[Path]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in record_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["arm"] != "musique_graph_rlm":
                    continue
                if record["task_id"] in rows:
                    continue
                scores = {
                    score["score_name"]: score["score_value"]
                    for score in record["scores"]
                }
                rows[record["task_id"]] = {
                    "hops": int(record["task_id"].split("hop")[0]),
                    "evidence_recall": scores.get("evidence_recall"),
                    "complete_evidence_coverage": scores.get("complete_evidence_coverage"),
                    "retrieved_count": scores.get("retrieved_count"),
                    "answer_f1": scores.get("answer_f1"),
                    "exact_match": scores.get("exact_match"),
                    "model_calls": record["model_calls"],
                    "total_tokens": record["total_tokens"],
                    "stop_reason": record["stop_reason"],
                }
    return rows


def aggregate(rows: dict[str, dict]) -> dict:
    def mean(values):
        collected = [value for value in values if value is not None]
        return round(sum(collected) / len(collected), 4) if collected else None

    by_hops = defaultdict(list)
    for row in rows.values():
        by_hops[row["hops"]].append(row)
    return {
        "runs": len(rows),
        "evidence_recall_mean": mean(row["evidence_recall"] for row in rows.values()),
        "complete_coverage_mean": mean(
            row["complete_evidence_coverage"] for row in rows.values()
        ),
        "answer_f1_mean": mean(row["answer_f1"] for row in rows.values()),
        "exact_match_mean": mean(row["exact_match"] for row in rows.values()),
        "retrieved_count_mean": mean(row["retrieved_count"] for row in rows.values()),
        "model_calls_mean": mean(row["model_calls"] for row in rows.values()),
        "tokens_per_case_mean": mean(row["total_tokens"] for row in rows.values()),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows.values())),
        "by_hops": {
            str(hops): {
                "runs": len(items),
                "evidence_recall_mean": mean(row["evidence_recall"] for row in items),
                "complete_coverage_mean": mean(
                    row["complete_evidence_coverage"] for row in items
                ),
                "answer_f1_mean": mean(row["answer_f1"] for row in items),
                "tokens_per_case_mean": mean(row["total_tokens"] for row in items),
            }
            for hops, items in sorted(by_hops.items())
        },
    }


def paired(v1: dict[str, dict], v2: dict[str, dict]) -> dict:
    shared = sorted(set(v1) & set(v2))
    result = {}
    for metric in ["evidence_recall", "complete_evidence_coverage", "answer_f1"]:
        wins = losses = 0
        diff_sum = 0.0
        counted = 0
        for task_id in shared:
            left = v2[task_id][metric]
            right = v1[task_id][metric]
            if left is None or right is None:
                continue
            counted += 1
            diff_sum += left - right
            if left > right:
                wins += 1
            elif left < right:
                losses += 1
        result[metric] = {
            "v2_wins": wins,
            "v2_losses": losses,
            "ties": counted - wins - losses,
            "mean_diff_v2_minus_v1": round(diff_sum / counted, 4) if counted else None,
        }
    result["shared_cases"] = len(shared)
    return result


def main() -> None:
    args = parse_args()
    v1 = load_rlm_rows([Path(part) for part in args.v1_records])
    v2 = load_rlm_rows([Path(part) for part in args.v2_records])
    payload = {
        "comparison": "rlm_controller_budget_ablation",
        "v1": {
            "label": "budget: 10 calls, depth 5, expansions 6",
            **aggregate(v1),
        },
        "v2": {
            "label": "budget: 16 calls, depth 8, expansions 12, hop-chain prompt, min-2-evidence answer guard",
            **aggregate(v2),
        },
        "paired_v2_vs_v1": paired(v1, v2),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    v1_agg, v2_agg = payload["v1"], payload["v2"]
    print(f"{'metric':<28} {'v1 (10 calls)':>14} {'v2 (16 calls)':>14}")
    for key in [
        "runs",
        "evidence_recall_mean",
        "complete_coverage_mean",
        "answer_f1_mean",
        "exact_match_mean",
        "model_calls_mean",
        "tokens_per_case_mean",
    ]:
        print(f"{key:<28} {v1_agg[key]!s:>14} {v2_agg[key]!s:>14}")
    print()
    for hops in ["2", "3", "4"]:
        left = v1_agg["by_hops"].get(hops, {})
        right = v2_agg["by_hops"].get(hops, {})
        print(
            f"{hops}hop coverage: v1={left.get('complete_coverage_mean')} "
            f"v2={right.get('complete_coverage_mean')} | recall: v1={left.get('evidence_recall_mean')} "
            f"v2={right.get('evidence_recall_mean')}"
        )
    print()
    print("paired:", json.dumps(payload["paired_v2_vs_v1"], ensure_ascii=False))
    print(f"\nWrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1-records",
        nargs="+",
        default=[f"artifacts/musique_rlm_full/shard_{i}/records.jsonl" for i in range(5)],
    )
    parser.add_argument(
        "--v2-records",
        nargs="+",
        default=[f"artifacts/musique_rlm_v2/shard_{i}/records.jsonl" for i in range(5)],
    )
    parser.add_argument(
        "--output",
        default="docs/benchmarks/musique_completeness/rlm_budget_comparison.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
