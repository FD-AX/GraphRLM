from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.local_env import load_local_env


SCORE_NAMES = [
    "evidence_recall",
    "evidence_precision",
    "complete_evidence_coverage",
    "retrieved_count",
    "answer_f1",
    "exact_match",
]


def main() -> None:
    args = parse_args()
    load_local_env(PROJECT_ROOT)
    rows = collect_rows([Path(part) for part in args.records])
    print(f"Collected {len(rows)} unique (task, arm) rows.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_path = output_dir / "per_case_scores.json"
    per_case_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    summary = build_summary(rows, run_batch=args.run_batch)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {per_case_path} and {summary_path}")

    if args.postgres_dsn:
        sync_postgres(rows, dsn=args.postgres_dsn, run_batch=args.run_batch)
        print("Synced to PostgreSQL.")


def collect_rows(record_paths: list[Path]) -> list[dict]:
    rows: dict[tuple[str, str], dict] = {}
    for path in record_paths:
        if not path.exists():
            print(f"warning: {path} does not exist, skipping")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (record["task_id"], record["arm"])
                if key in rows:
                    continue
                scores = {
                    score["score_name"]: score["score_value"]
                    for score in record["scores"]
                }
                hops_label = (
                    record.get("case_metadata", {})
                    .get("native_fields", {})
                    .get("task_group", "")
                )
                rows[key] = {
                    "task_id": record["task_id"],
                    "arm": record["arm"],
                    "hops": int(hops_label[0]) if hops_label[:1].isdigit() else None,
                    **{name: scores.get(name) for name in SCORE_NAMES},
                    "model_calls": record["model_calls"],
                    "input_tokens": record["input_tokens"],
                    "output_tokens": record["output_tokens"],
                    "total_tokens": record["total_tokens"],
                    "latency_ms": record["latency_ms"],
                    "stop_reason": record["stop_reason"],
                    "error": record.get("error"),
                    "model_name": record.get("model_name"),
                }
    return sorted(rows.values(), key=lambda row: (row["arm"], row["task_id"]))


def build_summary(rows: list[dict], *, run_batch: str) -> dict:
    def mean(values: list) -> float | None:
        collected = [value for value in values if value is not None]
        return round(sum(collected) / len(collected), 4) if collected else None

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    arms = {}
    for arm, arm_rows in sorted(by_arm.items()):
        by_hops = defaultdict(list)
        for row in arm_rows:
            by_hops[row["hops"]].append(row)
        arms[arm] = {
            "runs": len(arm_rows),
            "errors": sum(1 for row in arm_rows if row["error"]),
            **{
                f"{name}_mean": mean([row[name] for row in arm_rows])
                for name in SCORE_NAMES
            },
            "latency_ms_mean": mean([row["latency_ms"] for row in arm_rows]),
            "total_tokens_sum": sum(row["total_tokens"] for row in arm_rows),
            "by_hops": {
                str(hops): {
                    "runs": len(hop_rows),
                    "evidence_recall_mean": mean(
                        [row["evidence_recall"] for row in hop_rows]
                    ),
                    "complete_evidence_coverage_mean": mean(
                        [row["complete_evidence_coverage"] for row in hop_rows]
                    ),
                    "answer_f1_mean": mean([row["answer_f1"] for row in hop_rows]),
                }
                for hops, hop_rows in sorted(by_hops.items(), key=lambda item: str(item[0]))
            },
        }

    paired = {}
    indexed: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        indexed[row["arm"]][row["task_id"]] = row
    comparisons = [
        ("musique_graph_navigator", "musique_dense_topk"),
        ("musique_graph_navigator_active", "musique_graph_navigator"),
        ("musique_graph_rlm", "musique_graph_navigator"),
        ("musique_graph_rlm", "musique_dense_topk"),
        ("musique_cross_encoder", "musique_dense_topk"),
        ("musique_graph_navigator", "musique_cross_encoder"),
        ("musique_graph_rlm", "musique_cross_encoder"),
        ("musique_mdr_iterative", "musique_dense_topk"),
        ("musique_graph_navigator", "musique_mdr_iterative"),
    ]
    for left, right in comparisons:
        if left not in indexed or right not in indexed:
            continue
        shared = set(indexed[left]) & set(indexed[right])
        entry = {}
        for metric in ["evidence_recall", "complete_evidence_coverage"]:
            wins = losses = 0
            diff_sum = 0.0
            for task_id in shared:
                left_value = indexed[left][task_id][metric]
                right_value = indexed[right][task_id][metric]
                if left_value is None or right_value is None:
                    continue
                diff_sum += left_value - right_value
                if left_value > right_value:
                    wins += 1
                elif left_value < right_value:
                    losses += 1
            entry[metric] = {
                "wins": wins,
                "losses": losses,
                "ties": len(shared) - wins - losses,
                "mean_diff": round(diff_sum / len(shared), 4) if shared else None,
            }
        paired[f"{left}__vs__{right}"] = entry

    return {
        "benchmark": "musique_completeness_v1",
        "run_batch": run_batch,
        "dataset": {
            "name": "dgslibisey/MuSiQue",
            "split": "validation",
            "selection": "stratified_by_hops_first_match",
        },
        "retrieval_budget_top_k": 5,
        "encoder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "rlm_controller": "gpt-5-mini (reasoning_effort=low)",
        "arms": arms,
        "paired_comparisons": paired,
    }


def sync_postgres(rows: list[dict], *, dsn: str, run_batch: str) -> None:
    import psycopg

    schema_sql = (PROJECT_ROOT / "infra/postgres/musique_completeness.sql").read_text(
        encoding="utf-8"
    )
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(schema_sql)
            for row in rows:
                cursor.execute(
                    """
                    INSERT INTO benchmark_metrics.musique_completeness (
                        task_id, arm_name, run_batch, hops,
                        evidence_recall, evidence_precision,
                        complete_evidence_coverage, retrieved_count,
                        answer_f1, exact_match,
                        model_calls, input_tokens, output_tokens, total_tokens,
                        latency_ms, stop_reason, error, model_name
                    ) VALUES (
                        %(task_id)s, %(arm)s, %(run_batch)s, %(hops)s,
                        %(evidence_recall)s, %(evidence_precision)s,
                        %(complete_evidence_coverage)s, %(retrieved_count)s,
                        %(answer_f1)s, %(exact_match)s,
                        %(model_calls)s, %(input_tokens)s, %(output_tokens)s,
                        %(total_tokens)s, %(latency_ms)s, %(stop_reason)s,
                        %(error)s, %(model_name)s
                    )
                    ON CONFLICT (task_id, arm_name, run_batch) DO UPDATE SET
                        evidence_recall = EXCLUDED.evidence_recall,
                        evidence_precision = EXCLUDED.evidence_precision,
                        complete_evidence_coverage = EXCLUDED.complete_evidence_coverage,
                        retrieved_count = EXCLUDED.retrieved_count,
                        answer_f1 = EXCLUDED.answer_f1,
                        exact_match = EXCLUDED.exact_match,
                        model_calls = EXCLUDED.model_calls,
                        input_tokens = EXCLUDED.input_tokens,
                        output_tokens = EXCLUDED.output_tokens,
                        total_tokens = EXCLUDED.total_tokens,
                        latency_ms = EXCLUDED.latency_ms,
                        stop_reason = EXCLUDED.stop_reason,
                        error = EXCLUDED.error,
                        model_name = EXCLUDED.model_name,
                        synced_at = now()
                    """,
                    {**row, "run_batch": run_batch},
                )
        connection.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        nargs="+",
        default=[
            "artifacts/musique_full/records.jsonl",
            "artifacts/musique_rlm_full/shard_0/records.jsonl",
            "artifacts/musique_rlm_full/shard_1/records.jsonl",
            "artifacts/musique_rlm_full/shard_2/records.jsonl",
            "artifacts/musique_rlm_full/shard_3/records.jsonl",
            "artifacts/musique_rlm_full/shard_4/records.jsonl",
        ],
    )
    parser.add_argument("--output-dir", default="docs/benchmarks/musique_completeness")
    parser.add_argument("--run-batch", default="musique_500_v1")
    parser.add_argument(
        "--postgres-dsn",
        default=None,
        help="e.g. postgresql://jmlc:jmlc_local@localhost:55432/jmlc",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
