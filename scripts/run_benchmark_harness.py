from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks import (
    BenchmarkRunRecord,
    DirectGPTArm,
    DirectHeuristicArm,
    FactLensAuditArm,
    FactLensAuditScorer,
    GoldAnswerArm,
    GraphRLMSemanticGraphOpsArm,
    GraphRLMSemanticGraphArm,
    GraphRLMBenchmarkArm,
    KeywordRAGArm,
    OOLONGLocalCompatibleScorer,
    OOLONGSynthSource,
    RawRecordsOpsArm,
    WrongAnswerArm,
    aggregate_records,
    load_jsonl_cases,
    load_oolong_synth_cases,
    load_oolong_synth_filtered_cases,
    load_oolong_synth_stratified_cases,
    make_s_niah_cases,
    run_benchmark_cases,
)
from app.core.local_env import load_local_env


def main() -> None:
    load_local_env(PROJECT_ROOT)
    args = parse_args()
    cases = load_cases(args)
    if args.require_case_count is not None and len(cases) != args.require_case_count:
        raise ValueError(f"Expected {args.require_case_count} cases, got {len(cases)}")
    arms = make_arms(args)
    scorers = make_scorers(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "benchmark_records.jsonl"
    aggregate_path = args.output_dir / "benchmark_aggregate.json"
    manifest_path = args.output_dir / "benchmark_manifest.json"
    if args.incremental:
        records = run_incremental(cases, arms, scorers, records_path)
    else:
        records = run_benchmark_cases(cases, arms, scorers=scorers)
        records_path.write_text(
            "\n".join(record.model_dump_json() for record in records) + "\n",
            encoding="utf-8",
        )
    aggregate = aggregate_records(records, score_name=aggregate_score_name(args))
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(build_manifest(args, cases, records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "arms": args.arms,
                "records_path": str(records_path),
                "aggregate_path": str(aggregate_path),
                "manifest_path": str(manifest_path),
                "aggregate": aggregate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--benchmark", choices=["s_niah", "oolong", "oolong_pairs", "factlens"])
    parser.add_argument("--synthetic-s-niah", action="store_true")
    parser.add_argument("--oolong-synth-official", action="store_true")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-split", default="validation")
    parser.add_argument("--hf-offset", type=int, default=0)
    parser.add_argument("--hf-length", type=int, default=1)
    parser.add_argument("--context-bucket", type=int)
    parser.add_argument("--dataset-subset")
    parser.add_argument("--max-filtered-cases", type=int)
    parser.add_argument("--require-case-count", type=int)
    parser.add_argument("--stratified-count", type=int)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    parser.add_argument("--worker-model")
    parser.add_argument("--worker-reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--prompt-id", default="oolong_direct_v1")
    parser.add_argument("--synthetic-count", type=int, default=10)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[
            "direct_heuristic",
            "flat_subclaim_verification",
            "graph_projection_verification",
            "graph_query_verification",
            "graph_shared_evidence",
            "graph_recursive_completeness",
            "keyword_rag",
            "raw_records_ops",
            "gold_fixture",
            "wrong_fixture",
            "direct_gpt",
            "graph_rlm",
            "graph_rlm_semantic_graph",
            "graph_rlm_semantic_graph_ops",
        ],
        default=["direct_heuristic", "keyword_rag"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/benchmark_harness"),
    )
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_s_niah and not args.oolong_synth_official and not args.input_jsonl:
        raise ValueError("Provide --input-jsonl, --synthetic-s-niah, or --oolong-synth-official")
    if args.input_jsonl and not args.benchmark:
        raise ValueError("--benchmark is required with --input-jsonl")
    return args


def load_cases(args: argparse.Namespace):
    if args.synthetic_s_niah:
        return make_s_niah_cases(args.synthetic_count, args.context_tokens)
    if args.oolong_synth_official:
        source = OOLONGSynthSource(
            split=args.hf_split,
            revision=args.hf_revision,
            offset=args.hf_offset,
            length=args.hf_length,
            context_bucket=args.context_bucket,
            dataset_subset=args.dataset_subset,
            max_rows=args.max_filtered_cases,
        )
        if args.context_bucket is not None or args.dataset_subset is not None:
            return load_oolong_synth_filtered_cases(source)
        if args.stratified_count:
            return load_oolong_synth_stratified_cases(source, args.stratified_count)
        return load_oolong_synth_cases(
            source
        )
    return load_jsonl_cases(args.input_jsonl, benchmark=args.benchmark)


def make_arms(args: argparse.Namespace):
    arms = []
    for name in args.arms:
        if name == "direct_heuristic":
            arms.append(DirectHeuristicArm())
        elif name in {
            "flat_subclaim_verification",
            "graph_projection_verification",
            "graph_query_verification",
            "graph_shared_evidence",
            "graph_recursive_completeness",
        }:
            arms.append(
                FactLensAuditArm(
                    name,
                    experiment_id=args.experiment_id or "factlens_audit_v1",
                    prompt_id=args.prompt_id
                    if args.prompt_id != "oolong_direct_v1"
                    else "factlens_audit_deterministic_v1",
                )
            )
        elif name == "keyword_rag":
            arms.append(KeywordRAGArm())
        elif name == "raw_records_ops":
            arms.append(
                RawRecordsOpsArm(
                    prompt_id=args.prompt_id
                    if args.prompt_id != "oolong_direct_v1"
                    else "oolong_raw_records_ops_v1",
                    experiment_id=args.experiment_id,
                    worker_model_name=args.worker_model,
                    worker_reasoning_effort=args.worker_reasoning_effort,
                )
            )
        elif name == "gold_fixture":
            arms.append(GoldAnswerArm())
        elif name == "wrong_fixture":
            arms.append(WrongAnswerArm())
        elif name == "direct_gpt":
            arms.append(
                DirectGPTArm(
                    model_name=args.model,
                    prompt_id=args.prompt_id,
                    reasoning_effort=args.reasoning_effort,
                    experiment_id=args.experiment_id,
                )
            )
        elif name == "graph_rlm":
            arms.append(
                GraphRLMBenchmarkArm(
                    model_name=args.model,
                    prompt_id=args.prompt_id,
                    reasoning_effort=args.reasoning_effort,
                    experiment_id=args.experiment_id,
                    worker_model_name=args.worker_model,
                    worker_reasoning_effort=args.worker_reasoning_effort,
                )
            )
        elif name == "graph_rlm_semantic_graph":
            arms.append(
                GraphRLMSemanticGraphArm(
                    model_name=args.model,
                    prompt_id=args.prompt_id,
                    reasoning_effort=args.reasoning_effort,
                    experiment_id=args.experiment_id,
                    worker_model_name=args.worker_model,
                    worker_reasoning_effort=args.worker_reasoning_effort,
                )
            )
        elif name == "graph_rlm_semantic_graph_ops":
            arms.append(
                GraphRLMSemanticGraphOpsArm(
                    model_name=args.model,
                    prompt_id=args.prompt_id
                    if args.prompt_id != "oolong_direct_v1"
                    else "oolong_graph_rlm_semantic_graph_ops_v1",
                    reasoning_effort=args.reasoning_effort,
                    experiment_id=args.experiment_id,
                    worker_model_name=args.worker_model,
                    worker_reasoning_effort=args.worker_reasoning_effort,
                )
            )
        else:
            raise ValueError(f"Unsupported arm={name!r}")
    return arms


def make_scorers(args: argparse.Namespace):
    if args.benchmark == "factlens":
        return [FactLensAuditScorer()]
    if args.oolong_synth_official or args.benchmark == "oolong":
        return [OOLONGLocalCompatibleScorer()]
    return None


def aggregate_score_name(args: argparse.Namespace) -> str:
    if args.benchmark == "factlens":
        return "factlens_supported"
    if args.oolong_synth_official or args.benchmark == "oolong":
        return "task_score"
    return "exact_match"


def build_manifest(args: argparse.Namespace, cases, records) -> dict:
    source = cases[0].metadata.get("source", {}) if cases else {}
    native_datasets = sorted(
        {
            str(case.metadata.get("native_fields", {}).get("dataset"))
            for case in cases
        }
    )
    context_buckets = sorted(
        {
            case.benchmark_context_len
            for case in cases
            if case.benchmark_context_len is not None
        }
    )
    return {
        "experiment_id": args.experiment_id,
        "benchmark": "oolong" if args.oolong_synth_official else args.benchmark,
        "dataset": source.get("dataset"),
        "dataset_subset_filter": args.dataset_subset,
        "native_datasets": native_datasets,
        "context_bucket_filter": args.context_bucket,
        "context_buckets": context_buckets,
        "split": source.get("split"),
        "requested_revision": source.get("revision") or args.hf_revision,
        "resolved_revision": source.get("resolved_revision"),
        "cases_total": len(cases),
        "task_ids": [case.task_id for case in cases],
        "arms": args.arms,
        "root_model": args.model,
        "root_reasoning_effort": args.reasoning_effort,
        "worker_model": args.worker_model,
        "worker_reasoning_effort": args.worker_reasoning_effort,
        "prompt_id": args.prompt_id,
        "score_backends": sorted(
            {
                score.score_backend
                for record in records
                for score in record.scores
            }
        ),
        "incremental": args.incremental,
        "require_case_count": args.require_case_count,
        "is_figure_1_oolong_8k_reproduction": (
            args.context_bucket == 8192 and args.dataset_subset == "trec_coarse"
        ),
        "protocol_note": (
            "Figure 1 OOLONG 8K only if dataset_subset_filter is trec_coarse "
            "and task IDs match the paper protocol. Otherwise this is a "
            "paper-model-aligned OOLONG-Synth pilot."
        ),
    }


def run_incremental(cases, arms, scorers, records_path: Path) -> list[BenchmarkRunRecord]:
    existing = _read_existing_records(records_path)
    records = list(existing)
    completed = {_record_key(record) for record in existing}
    for case in cases:
        for arm in arms:
            planned_key = (case.task_id, arm.name)
            if any(key[:2] == planned_key for key in completed):
                continue
            new_record = run_benchmark_cases([case], [arm], scorers=scorers)[0]
            records.append(new_record)
            completed.add(_record_key(new_record))
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(new_record.model_dump_json() + "\n")
    return records


def _read_existing_records(records_path: Path) -> list[BenchmarkRunRecord]:
    if not records_path.exists():
        return []
    records = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(BenchmarkRunRecord.model_validate_json(line))
    return records


def _record_key(record: BenchmarkRunRecord) -> tuple:
    return (
        record.task_id,
        record.arm,
        record.experiment_id,
        record.model_name,
        record.reasoning_effort,
        record.prompt_id,
    )


if __name__ == "__main__":
    main()
