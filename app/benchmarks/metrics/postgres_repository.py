from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path("infra/postgres/benchmark_metrics.sql")


@dataclass(frozen=True)
class ProjectionTable:
    table_name: str
    projection_key: str
    conflict_column: str


PROJECTION_TABLES: tuple[ProjectionTable, ...] = (
    ProjectionTable("benchmark_metrics.benchmark_runs", "benchmark_runs", "run_id"),
    ProjectionTable("benchmark_metrics.benchmark_model_calls", "benchmark_model_calls", "call_id"),
    ProjectionTable(
        "benchmark_metrics.benchmark_score_revisions",
        "benchmark_score_revisions",
        "score_revision_id",
    ),
    ProjectionTable(
        "benchmark_metrics.benchmark_graph_metrics",
        "benchmark_graph_metrics",
        "run_id",
    ),
    ProjectionTable(
        "benchmark_metrics.benchmark_graph_forensics",
        "benchmark_graph_forensics",
        "run_id",
    ),
    ProjectionTable(
        "benchmark_metrics.benchmark_factlens_audits",
        "benchmark_factlens_audits",
        "run_id",
    ),
)


class AsyncBenchmarkMetricsRepository:
    def __init__(self, dsn: str, *, schema_path: Path = SCHEMA_PATH) -> None:
        self._dsn = dsn
        self._schema_path = schema_path
        self._conn = None

    async def __aenter__(self) -> "AsyncBenchmarkMetricsRepository":
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL metrics sync. Install psycopg[binary].") from exc

        self._conn = await psycopg.AsyncConnection.connect(self._dsn)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._conn is None:
            return
        if exc_type is None:
            await self._conn.commit()
        else:
            await self._conn.rollback()
        await self._conn.close()
        self._conn = None

    async def initialize_schema(self) -> None:
        await self._execute(self._schema_path.read_text(encoding="utf-8"))

    async def upsert_projection(self, projection: dict[str, list[dict[str, Any]]]) -> None:
        for table in PROJECTION_TABLES:
            await self.upsert_rows(
                table.table_name,
                projection.get(table.projection_key, []),
                conflict_column=table.conflict_column,
            )

    async def upsert_rows(
        self,
        table_name: str,
        rows: list[dict[str, Any]],
        *,
        conflict_column: str,
    ) -> None:
        for row in rows:
            await self._upsert_row(table_name, row, conflict_column=conflict_column)

    async def _upsert_row(self, table_name: str, row: dict[str, Any], *, conflict_column: str) -> None:
        columns = list(row.keys())
        placeholders = ", ".join(["%s"] * len(columns))
        column_sql = ", ".join(columns)
        updates = ", ".join(
            f"{column}=EXCLUDED.{column}"
            for column in columns
            if column != conflict_column
        )
        sql = (
            f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_column}) DO UPDATE SET {updates}"
        )
        await self._execute(sql, [_adapt_value(row[column]) for column in columns])

    async def _execute(self, sql: str, params: list[Any] | None = None) -> None:
        if self._conn is None:
            raise RuntimeError("AsyncBenchmarkMetricsRepository must be used as an async context manager.")
        await self._conn.execute(sql, params)


def _adapt_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        try:
            from psycopg.types.json import Jsonb
        except ImportError:
            return value
        return Jsonb(value)
    return value
