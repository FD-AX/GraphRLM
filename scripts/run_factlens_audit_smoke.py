from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.factlens import FactLensAuditArm, FactLensAuditScorer, make_factlens_audit_case
from app.benchmarks.runner import aggregate_records, run_benchmark_cases


def main() -> None:
    output_dir = Path("artifacts/factlens_audit_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    case = make_factlens_audit_case()
    modes = [
        "flat_subclaim_verification",
        "graph_query_verification",
        "graph_shared_evidence",
    ]
    arms = [FactLensAuditArm(mode) for mode in modes]
    records = run_benchmark_cases([case], arms, scorers=[FactLensAuditScorer()])
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
        "experiment_id": "factlens_audit_v1",
        "benchmark": "factlens",
        "dataset": "factlens_local_audit",
        "dataset_origin": "local_generated",
        "requested_revision": "local",
        "resolved_revision": "local_factlens_audit_v1",
        "cases_total": 1,
        "task_ids": [case.task_id],
        "arms": modes,
        "worker_model": None,
        "score_backends": ["factlens_local_audit_v1"],
        "protocol_note": "Local FactLens audit smoke; not an official benchmark result.",
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "records": len(records),
                "aggregate": aggregate,
                "audit_modes": [
                    record.trace[0]["factlens_audit"]
                    for record in records
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
