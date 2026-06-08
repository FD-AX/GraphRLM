from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.models import BenchmarkArmResult, BenchmarkCase
from app.benchmarks.oolong.scorer import OOLONGLocalCompatibleScorer


def main() -> None:
    args = parse_args()
    scorer = OOLONGLocalCompatibleScorer()
    revisions = []
    for line in args.records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        case = _case_from_record(record)
        prediction = BenchmarkArmResult(
            prediction=str(record.get("prediction") or ""),
            raw_response=record.get("raw_response"),
            provider=record.get("provider"),
            model_name=record.get("model_name"),
            response_id=record.get("response_id"),
            total_tokens=int(record.get("total_tokens") or 0),
            fallback_used=bool(record.get("fallback_used")),
        )
        rescored = scorer.score(case, prediction)
        revisions.append(
            {
                "task_id": record.get("task_id"),
                "arm": record.get("arm"),
                "model_rerun": False,
                "original_scores": record.get("scores", []),
                "original_scorer_versions": sorted(
                    {
                        score.get("score_backend")
                        for score in record.get("scores", [])
                        if score.get("score_backend")
                    }
                ),
                "rescored_scores": [score.model_dump() for score in rescored],
                "rescored_scorer_version": scorer.score_backend,
                "rescore_reason": args.reason,
                "prediction": record.get("prediction"),
                "gold": record.get("gold"),
                "run_fingerprint": record.get("run_fingerprint"),
                "source_records_path": str(args.records_path),
            }
        )

    output = {
        "records_path": str(args.records_path),
        "model_rerun": False,
        "score_revision_count": len(revisions),
        "score_revisions": revisions,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--reason",
        default="OOLONG label normalization v2 strips trailing punctuation and rejects ambiguous label matches.",
    )
    return parser.parse_args()


def _case_from_record(record: dict) -> BenchmarkCase:
    case_metadata = dict(record.get("case_metadata") or {})
    native_fields = dict(case_metadata)
    metadata = {"native_fields": native_fields}
    source = case_metadata.get("source")
    if source:
        metadata["source"] = source
    return BenchmarkCase(
        benchmark=record.get("benchmark", "oolong"),
        benchmark_id=record.get("benchmark_id", "oolong"),
        dataset_origin=record.get("dataset_origin", "unknown"),
        task_id=str(record.get("task_id")),
        context=str(case_metadata.get("context_window_text") or ""),
        question=str(case_metadata.get("question") or ""),
        gold_answer=record.get("gold"),
        gold_evidence_span_ids=list(record.get("gold_evidence_span_ids") or []),
        context_tokens=int(record.get("context_tokens") or 0),
        benchmark_context_len=record.get("benchmark_context_len"),
        measured_context_tokens=int(record.get("measured_context_tokens") or 0),
        tokenizer_id=record.get("tokenizer_id") or "whitespace",
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
