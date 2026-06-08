from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.benchmarks.metrics.postgres_projection import build_postgres_projection
from app.benchmarks.metrics.postgres_repository import AsyncBenchmarkMetricsRepository


@dataclass(frozen=True)
class BenchmarkMetricsSyncRequest:
    artifact_dirs: list[Path]
    registry_path: Path
    score_name: str
    dsn: str | None = None
    schema_path: Path = Path("infra/postgres/benchmark_metrics.sql")
    output_json: Path | None = None
    create_schema: bool = False
    upsert: bool = False


@dataclass(frozen=True)
class BenchmarkMetricsSyncResult:
    summary: dict
    projection: dict


class BenchmarkMetricsSyncService:
    async def sync(self, request: BenchmarkMetricsSyncRequest) -> BenchmarkMetricsSyncResult:
        projection = build_postgres_projection(
            request.artifact_dirs,
            registry_path=request.registry_path,
            score_name=request.score_name,
        )
        summary = summarize_projection(projection)
        if request.output_json:
            request.output_json.parent.mkdir(parents=True, exist_ok=True)
            request.output_json.write_text(
                json.dumps({"summary": summary, "projection": projection}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if request.upsert:
            if not request.dsn:
                raise RuntimeError("PostgreSQL DSN is required when upsert=True")
            await self._save_projection(request, projection)
        return BenchmarkMetricsSyncResult(summary=summary, projection=projection)

    async def _save_projection(self, request: BenchmarkMetricsSyncRequest, projection: dict) -> None:
        async with AsyncBenchmarkMetricsRepository(request.dsn, schema_path=request.schema_path) as repository:
            if request.create_schema:
                await repository.initialize_schema()
            await repository.upsert_projection(projection)


def summarize_projection(projection: dict) -> dict:
    return {
        "runs": len(projection["benchmark_runs"]),
        "model_calls": len(projection["benchmark_model_calls"]),
        "score_revisions": len(projection["benchmark_score_revisions"]),
        "graph_metrics": len(projection["benchmark_graph_metrics"]),
        "graph_forensics": len(projection.get("benchmark_graph_forensics", [])),
        "factlens_audits": len(projection.get("benchmark_factlens_audits", [])),
        "run_ids": [row["run_id"] for row in projection["benchmark_runs"]],
        "active_scores": {
            f"{row['task_id']}:{row['arm_name']}": row["active_score"]
            for row in projection["benchmark_runs"]
        },
        "total_tokens": {
            f"{row['task_id']}:{row['arm_name']}": row["total_tokens"]
            for row in projection["benchmark_runs"]
        },
    }
