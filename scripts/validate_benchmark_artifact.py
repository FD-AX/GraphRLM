from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.artifact_validator import validate_benchmark_record
from app.benchmarks.models import BenchmarkRunRecord


def main() -> None:
    args = parse_args()
    manifest = {}
    if args.manifest_path and args.manifest_path.exists():
        manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    rows = []
    for line in args.records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = BenchmarkRunRecord.model_validate_json(line)
        rows.append(
            {
                "task_id": record.task_id,
                "arm": record.arm,
                **validate_benchmark_record(record, manifest),
            }
        )
    payload = {
        "records_path": str(args.records_path),
        "manifest_path": str(args.manifest_path) if args.manifest_path else None,
        "records": rows,
        "all_valid": all(row["artifact_valid"] for row in rows),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/artifact_validation.json"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
