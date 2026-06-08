from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import urlopen

from app.benchmarks.adapters import oolong_case_from_record
from app.benchmarks.models import BenchmarkCase


HF_DATASETS_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class OOLONGSynthSource:
    dataset: str = "oolongbench/oolong-synth"
    config: str = "default"
    split: str = "validation"
    revision: str = "main"
    offset: int = 0
    length: int = 1
    context_bucket: int | None = None
    dataset_subset: str | None = None
    page_size: int = 100
    max_rows: int | None = None


def load_oolong_synth_cases(source: OOLONGSynthSource) -> list[BenchmarkCase]:
    resolved_revision = resolve_dataset_revision(source.dataset, source.revision)
    payload = fetch_oolong_synth_rows(source)
    cases = []
    for row in payload.get("rows", []):
        source_record = dict(row["row"])
        source_record["dataset_origin"] = "official"
        case = oolong_case_from_record(source_record, line_number=int(row["row_idx"]))
        case.metadata.update(
            {
                "source": {
                    "dataset": source.dataset,
                    "config": source.config,
                    "split": source.split,
                    "revision": source.revision,
                    "resolved_revision": resolved_revision,
                    "offset": source.offset,
                    "length": source.length,
                    "row_idx": row["row_idx"],
                    "source_row_id": source_record.get("id"),
                    "features": payload.get("features", []),
                    "num_rows_total": payload.get("num_rows_total"),
                    "partial": payload.get("partial"),
                },
                "source_record": source_record,
            }
        )
        cases.append(case)
    return cases


def load_oolong_synth_filtered_cases(source: OOLONGSynthSource) -> list[BenchmarkCase]:
    if source.context_bucket is None and source.dataset_subset is None:
        return load_oolong_synth_cases(source)

    resolved_revision = resolve_dataset_revision(source.dataset, source.revision)
    offset = source.offset
    collected: list[BenchmarkCase] = []
    scanned_rows = 0
    while True:
        page = OOLONGSynthSource(
            dataset=source.dataset,
            config=source.config,
            split=source.split,
            revision=source.revision,
            offset=offset,
            length=min(source.page_size, 100),
        )
        payload = fetch_oolong_synth_rows(page)
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            scanned_rows += 1
            source_record = dict(row["row"])
            if source.context_bucket is not None and int(source_record.get("context_len", -1)) != source.context_bucket:
                continue
            if source.dataset_subset is not None and str(source_record.get("dataset")) != source.dataset_subset:
                continue
            source_record["dataset_origin"] = "official"
            case = oolong_case_from_record(source_record, line_number=int(row["row_idx"]))
            case.metadata.update(
                {
                    "source": {
                        "dataset": source.dataset,
                        "config": source.config,
                        "split": source.split,
                        "revision": source.revision,
                        "resolved_revision": resolved_revision,
                        "offset": source.offset,
                        "length": source.length,
                        "context_bucket_filter": source.context_bucket,
                        "dataset_subset_filter": source.dataset_subset,
                        "row_idx": row["row_idx"],
                        "source_row_id": source_record.get("id"),
                        "features": payload.get("features", []),
                        "num_rows_total": payload.get("num_rows_total"),
                        "partial": payload.get("partial"),
                    },
                    "source_record": source_record,
                }
            )
            collected.append(case)
            if source.max_rows is not None and len(collected) >= source.max_rows:
                return collected
        offset += len(rows)
        total = payload.get("num_rows_total")
        if total is not None and offset >= int(total):
            break
        if source.length and scanned_rows >= source.length:
            break
    return collected


def load_oolong_synth_stratified_cases(
    source: OOLONGSynthSource,
    target_count: int,
) -> list[BenchmarkCase]:
    cases = load_oolong_synth_cases(source)
    return select_stratified_cases(cases, target_count)


def select_stratified_cases(
    cases: list[BenchmarkCase],
    target_count: int,
) -> list[BenchmarkCase]:
    selected: list[BenchmarkCase] = []
    seen_values: dict[str, set[str]] = {
        "task_group": set(),
        "task": set(),
        "answer_type": set(),
        "input_subset": set(),
        "num_labels": set(),
        "context_window_id": set(),
    }
    remaining = list(cases)
    while remaining and len(selected) < target_count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: _diversity_gain(remaining[index], seen_values),
        )
        case = remaining.pop(best_index)
        selected.append(case)
        native = case.metadata.get("native_fields", {})
        for key in seen_values:
            seen_values[key].add(str(native.get(key)))
    return selected


def _diversity_gain(case: BenchmarkCase, seen_values: dict[str, set[str]]) -> tuple[int, int]:
    native = case.metadata.get("native_fields", {})
    gain = sum(1 for key, seen in seen_values.items() if str(native.get(key)) not in seen)
    return gain, -int(case.task_id) if str(case.task_id).isdigit() else 0


def fetch_oolong_synth_rows(source: OOLONGSynthSource) -> dict:
    query = urlencode(
        {
            "dataset": source.dataset,
            "config": source.config,
            "split": source.split,
            "revision": source.revision,
            "offset": source.offset,
            "length": source.length,
        }
    )
    with urlopen(f"{HF_DATASETS_SERVER_ROWS_URL}?{query}", timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_dataset_revision(dataset: str, revision: str) -> str:
    with urlopen(
        f"https://huggingface.co/api/datasets/{dataset}/revision/{revision}",
        timeout=60,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["sha"])
