from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.local_env import load_local_env
from app.benchmarks.musique.arms import (
    MuSiQueDenseTopKArm,
    MuSiQueGraphNavigatorArm,
    MuSiQueGraphRLMArm,
    MuSiQueKeywordArm,
)
from app.benchmarks.musique.loader import MuSiQueSource, load_musique_stratified_cases
from app.benchmarks.musique.scorer import MuSiQueCompletenessScorer
from app.benchmarks.runner import run_benchmark_cases
from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticEncoder,
    HashingSemanticEncoder,
    TransformerSemanticEncoder,
)


def main() -> None:
    args = parse_args()
    load_local_env(PROJECT_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    per_hop = {
        int(part.split(":")[0]): int(part.split(":")[1])
        for part in args.hop_mix.split(",")
    }
    print(f"Loading MuSiQue cases: {per_hop} ...", flush=True)
    cases = load_musique_stratified_cases(
        MuSiQueSource(split=args.split),
        per_hop_counts=per_hop,
    )
    print(f"Loaded {len(cases)} cases.", flush=True)
    cases_path = output_dir / "cases.jsonl"
    if not cases_path.exists():
        with cases_path.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(case.model_dump_json() + "\n")

    arms = build_arms(args)
    scorer = MuSiQueCompletenessScorer()

    completed = _completed_keys(records_path)
    print(
        f"Arms: {[arm.name for arm in arms]} | already completed: {len(completed)}",
        flush=True,
    )
    pending = [
        (case, arm)
        for case in cases
        for arm in arms
        if (case.task_id, arm.name) not in completed
    ]
    with records_path.open("a", encoding="utf-8") as handle:
        for position, (case, arm) in enumerate(pending, start=1):
            records = run_benchmark_cases([case], [arm], scorers=[scorer])
            for record in records:
                handle.write(record.model_dump_json() + "\n")
            handle.flush()
            if position % 25 == 0 or position == len(pending):
                print(f"progress: {position}/{len(pending)}", flush=True)

    summary = summarize(records_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["arms"], ensure_ascii=False, indent=2))
    print(f"Summary written to {summary_path}")


def build_arms(args: argparse.Namespace) -> list:
    base_config = dict(
        projection_dim=None if args.backend == "transformer" else 256,
        interaction_dim=64,
        local_frontier_cap=args.frontier_cap,
        max_graph_depth=args.max_depth,
        beam_width=args.beam_width,
    )
    backend = (
        TransformerSemanticEncoder(device=args.device)
        if args.backend == "transformer"
        else HashingSemanticEncoder(dimensions=256)
    )

    def encoder_with(active_weight: float) -> GraphSemanticEncoder:
        navigation_weight = 0.70 - active_weight
        return GraphSemanticEncoder(
            config=EncoderConfig(
                **base_config,
                navigation_similarity_weight=navigation_weight,
                active_profile_weight=active_weight,
            ),
            backend=backend,
        )

    arm_names = [name.strip() for name in args.arms.split(",") if name.strip()]
    arms = []
    for name in arm_names:
        if name == "keyword":
            arms.append(MuSiQueKeywordArm(top_k=args.top_k))
        elif name == "dense":
            arms.append(MuSiQueDenseTopKArm(encoder_with(0.0), top_k=args.top_k))
        elif name == "graph":
            arms.append(
                MuSiQueGraphNavigatorArm(
                    encoder_with(0.0),
                    name="musique_graph_navigator",
                    top_k=args.top_k,
                    seed_top_k=args.seed_top_k,
                    max_depth=args.max_depth,
                    beam_width=args.beam_width,
                )
            )
        elif name == "graph_active":
            arms.append(
                MuSiQueGraphNavigatorArm(
                    encoder_with(args.active_weight),
                    name="musique_graph_navigator_active",
                    top_k=args.top_k,
                    seed_top_k=args.seed_top_k,
                    max_depth=args.max_depth,
                    beam_width=args.beam_width,
                )
            )
        elif name == "rlm":
            arms.append(
                MuSiQueGraphRLMArm(
                    encoder_with(0.0),
                    model_name=args.rlm_model,
                    reasoning_effort=args.rlm_reasoning_effort,
                    experiment_id=args.experiment_id,
                )
            )
        else:
            raise ValueError(f"Unknown arm: {name!r}")
    return arms


def _completed_keys(records_path: Path) -> set[tuple[str, str]]:
    completed = set()
    if not records_path.exists():
        return completed
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            completed.add((record["task_id"], record["arm"]))
    return completed


def summarize(records_path: Path) -> dict:
    rows = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def score_of(row: dict, name: str) -> float | None:
        for score in row["scores"]:
            if score["score_name"] == name:
                return score["score_value"]
        return None

    arms: dict[str, dict] = {}
    for row in rows:
        bucket = arms.setdefault(
            row["arm"],
            {
                "runs": 0,
                "errors": 0,
                "evidence_recall": [],
                "evidence_precision": [],
                "complete_evidence_coverage": [],
                "retrieved_count": [],
                "answer_f1": [],
                "exact_match": [],
                "latency_ms": [],
                "total_tokens": [],
                "by_hops": {},
            },
        )
        bucket["runs"] += 1
        if row.get("error"):
            bucket["errors"] += 1
        for name in [
            "evidence_recall",
            "evidence_precision",
            "complete_evidence_coverage",
            "retrieved_count",
            "answer_f1",
            "exact_match",
        ]:
            value = score_of(row, name)
            if value is not None:
                bucket[name].append(value)
        bucket["latency_ms"].append(row["latency_ms"])
        bucket["total_tokens"].append(row["total_tokens"])
        hops = str(
            row.get("case_metadata", {}).get("native_fields", {}).get("task_group")
        )
        hop_bucket = bucket["by_hops"].setdefault(
            hops, {"recall": [], "coverage": []}
        )
        recall = score_of(row, "evidence_recall")
        coverage = score_of(row, "complete_evidence_coverage")
        if recall is not None:
            hop_bucket["recall"].append(recall)
        if coverage is not None:
            hop_bucket["coverage"].append(coverage)

    def mean(values: list) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    summary_arms = {}
    for arm, bucket in sorted(arms.items()):
        summary_arms[arm] = {
            "runs": bucket["runs"],
            "errors": bucket["errors"],
            "evidence_recall_mean": mean(bucket["evidence_recall"]),
            "evidence_precision_mean": mean(bucket["evidence_precision"]),
            "complete_coverage_rate": mean(bucket["complete_evidence_coverage"]),
            "retrieved_count_mean": mean(bucket["retrieved_count"]),
            "answer_f1_mean": mean(bucket["answer_f1"]),
            "exact_match_mean": mean(bucket["exact_match"]),
            "latency_ms_mean": mean(bucket["latency_ms"]),
            "tokens_total": sum(bucket["total_tokens"]),
            "by_hops": {
                hops: {
                    "evidence_recall_mean": mean(values["recall"]),
                    "complete_coverage_rate": mean(values["coverage"]),
                    "runs": len(values["recall"]),
                }
                for hops, values in sorted(bucket["by_hops"].items())
            },
        }
    return {"benchmark": "musique_completeness_v1", "arms": summary_arms}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--hop-mix", default="2:200,3:150,4:150")
    parser.add_argument(
        "--arms",
        default="keyword,dense,graph,graph_active",
        help="Comma list: keyword,dense,graph,graph_active,rlm",
    )
    parser.add_argument("--backend", choices=["transformer", "hashing"], default="transformer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-top-k", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--frontier-cap", type=int, default=20)
    parser.add_argument("--active-weight", type=float, default=0.20)
    parser.add_argument("--rlm-model", default="gpt-5-mini")
    parser.add_argument("--rlm-reasoning-effort", default="low")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output-dir", default="artifacts/musique_completeness")
    return parser.parse_args()


if __name__ == "__main__":
    main()
