from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.metrics.postgres_sync import (
    BenchmarkMetricsSyncRequest,
    BenchmarkMetricsSyncService,
)


def main() -> None:
    _configure_asyncio_for_psycopg()
    result = asyncio.run(run_async())
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


async def run_async():
    args = parse_args()
    artifact_dirs = _artifact_dirs(args)
    dsn = args.dsn or os.getenv("JMLC_METRICS_DATABASE_URL") or os.getenv("DATABASE_URL")
    if args.upsert and not dsn:
        raise RuntimeError("--dsn or JMLC_METRICS_DATABASE_URL is required with --upsert")
    request = BenchmarkMetricsSyncRequest(
        artifact_dirs=artifact_dirs,
        registry_path=args.registry_path,
        score_name=args.score_name,
        dsn=dsn,
        schema_path=args.schema_path,
        output_json=args.output_json,
        create_schema=args.create_schema,
        upsert=args.upsert and not args.dry_run,
    )
    return await BenchmarkMetricsSyncService().sync(request)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, action="append", default=[])
    parser.add_argument("--scan-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--apply-score-revisions", action="store_true", default=True)
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--create-schema", action="store_true")
    parser.add_argument("--dsn")
    parser.add_argument("--score-name", default="task_score")
    parser.add_argument("--registry-path", type=Path, default=Path("artifacts/artifact_registry.json"))
    parser.add_argument("--schema-path", type=Path, default=Path("infra/postgres/benchmark_metrics.sql"))
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _artifact_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = list(args.artifact_dir)
    if args.scan_root:
        dirs.extend(
            path
            for path in args.scan_root.iterdir()
            if path.is_dir() and (path / "benchmark_records.jsonl").exists()
        )
    return sorted({path for path in dirs if path.is_dir()})


def _configure_asyncio_for_psycopg() -> None:
    if sys.platform != "win32":
        return
    policy_factory = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_factory is not None:
        asyncio.set_event_loop_policy(policy_factory())


if __name__ == "__main__":
    main()
