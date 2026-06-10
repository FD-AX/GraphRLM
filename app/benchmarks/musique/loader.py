from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from app.benchmarks.models import BenchmarkCase


HF_DATASETS_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class MuSiQueSource:
    dataset: str = "dgslibisey/MuSiQue"
    config: str = "default"
    split: str = "validation"
    offset: int = 0
    length: int = 100
    page_size: int = 100


def load_musique_cases(source: MuSiQueSource) -> list[BenchmarkCase]:
    cases = []
    remaining = source.length
    offset = source.offset
    while remaining > 0:
        page_length = min(source.page_size, remaining)
        payload = _fetch_rows(source, offset=offset, length=page_length)
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            cases.append(musique_case_from_record(row["row"], row_idx=row["row_idx"], source=source))
        offset += len(rows)
        remaining -= len(rows)
    return cases


def load_musique_stratified_cases(
    source: MuSiQueSource,
    *,
    per_hop_counts: dict[int, int],
    max_scan_rows: int = 2500,
) -> list[BenchmarkCase]:
    selected: dict[int, list[BenchmarkCase]] = {hops: [] for hops in per_hop_counts}
    offset = source.offset
    scanned = 0
    while scanned < max_scan_rows and any(
        len(selected[hops]) < target for hops, target in per_hop_counts.items()
    ):
        payload = _fetch_rows(source, offset=offset, length=source.page_size)
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            case = musique_case_from_record(row["row"], row_idx=row["row_idx"], source=source)
            hops = case.expected_hops or 0
            if hops in selected and len(selected[hops]) < per_hop_counts[hops]:
                selected[hops].append(case)
        offset += len(rows)
        scanned += len(rows)
    cases = [case for bucket in selected.values() for case in bucket]
    cases.sort(key=lambda case: case.task_id)
    return cases


def musique_case_from_record(
    record: dict,
    *,
    row_idx: int,
    source: MuSiQueSource,
) -> BenchmarkCase:
    paragraphs = record["paragraphs"]
    task_id = str(record["id"])
    context = "\n\n".join(
        f"[para_{paragraph['idx']}] {paragraph['title']}: {paragraph['paragraph_text']}"
        for paragraph in paragraphs
    )
    gold_evidence_ids = [
        f"para_{paragraph['idx']}" for paragraph in paragraphs if paragraph["is_supporting"]
    ]
    return BenchmarkCase(
        benchmark="musique",
        benchmark_id=f"musique_{source.split}",
        dataset_origin="official",
        task_id=task_id,
        context=context,
        question=record["question"],
        gold_answer=record["answer"],
        gold_evidence_span_ids=gold_evidence_ids,
        gold_entities=[
            paragraph["title"] for paragraph in paragraphs if paragraph["is_supporting"]
        ],
        expected_hops=_hops_from_id(task_id),
        answerable=bool(record.get("answerable", True)),
        context_tokens=len(context.split()),
        measured_context_tokens=len(context.split()),
        metadata={
            "source": {
                "dataset": source.dataset,
                "config": source.config,
                "split": source.split,
                "row_idx": row_idx,
            },
            "native_fields": {
                "answer": record["answer"],
                "answer_aliases": record.get("answer_aliases", []),
                "task_group": f"{_hops_from_id(task_id)}hop",
                "answer_type": "span",
            },
            "paragraphs": [
                {
                    "idx": paragraph["idx"],
                    "title": paragraph["title"],
                    "paragraph_text": paragraph["paragraph_text"],
                }
                for paragraph in paragraphs
            ],
        },
    )


def _hops_from_id(task_id: str) -> int | None:
    head = task_id.split("hop", 1)[0]
    return int(head) if head.isdigit() else None


def _fetch_rows(
    source: MuSiQueSource,
    *,
    offset: int,
    length: int,
    max_attempts: int = 5,
) -> dict:
    query = urlencode(
        {
            "dataset": source.dataset,
            "config": source.config,
            "split": source.split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{HF_DATASETS_SERVER_ROWS_URL}?{query}"
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(url) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError):
            if attempt == max_attempts:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("unreachable")
