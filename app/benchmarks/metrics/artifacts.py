from __future__ import annotations

import json
from pathlib import Path

from app.benchmarks.models import BenchmarkRunRecord


def read_manifest(artifact_dir: Path) -> dict:
    path = artifact_dir / "benchmark_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(artifact_dir: Path) -> list[BenchmarkRunRecord]:
    path = artifact_dir / "benchmark_records.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(BenchmarkRunRecord.model_validate_json(line))
    return records


def read_score_revisions(artifact_dir: Path) -> list[dict]:
    revisions = []
    for path in sorted(artifact_dir.glob("score_revision_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for revision in payload.get("score_revisions", []):
            revision = dict(revision)
            revision["_revision_path"] = str(path)
            revisions.append(revision)
    return revisions
