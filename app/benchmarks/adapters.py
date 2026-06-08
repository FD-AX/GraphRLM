from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app.benchmarks.models import BenchmarkCase, BenchmarkName
from app.benchmarks.factlens.adapter import factlens_case_from_record


def load_jsonl_cases(path: Path, benchmark: BenchmarkName | None = None) -> list[BenchmarkCase]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        cases.append(case_from_record(record, benchmark=benchmark, line_number=line_number))
    return cases


def case_from_record(
    record: dict,
    benchmark: BenchmarkName | None = None,
    line_number: int = 0,
) -> BenchmarkCase:
    resolved_benchmark = benchmark or _benchmark_name(record)
    if resolved_benchmark == "s_niah":
        return s_niah_case_from_record(record, line_number=line_number)
    if resolved_benchmark == "oolong":
        return oolong_case_from_record(record, line_number=line_number)
    if resolved_benchmark == "oolong_pairs":
        return oolong_pairs_case_from_record(record, line_number=line_number)
    if resolved_benchmark == "factlens":
        return factlens_case_from_record(record, line_number=line_number)
    raise ValueError(f"Unsupported benchmark={resolved_benchmark!r}")


def s_niah_case_from_record(record: dict, line_number: int = 0) -> BenchmarkCase:
    context = _first(record, ["context", "haystack", "input", "prompt"], default="")
    question = _first(record, ["question", "query"], default="")
    answer = _first(record, ["gold_answer", "answer", "needle", "target"], default="")
    task_id = str(_first(record, ["task_id", "id"], default=f"s_niah_{line_number}"))
    evidence_ids = _list_field(record, ["gold_evidence_span_ids", "evidence_span_ids"])
    return BenchmarkCase(
        benchmark="s_niah",
        benchmark_id="sniah",
        dataset_origin=str(_first(record, ["dataset_origin"], default="unknown")),
        task_id=task_id,
        context=context,
        question=question,
        gold_answer=answer,
        gold_evidence_span_ids=evidence_ids,
        gold_entities=_list_field(record, ["gold_entities", "entities"]),
        expected_hops=_optional_int(_first(record, ["expected_hops"], default=None)),
        answerable=bool(_first(record, ["answerable"], default=True)),
        context_tokens=_context_tokens(record, context),
        benchmark_context_len=_optional_int(_first(record, ["context_len"], default=None)),
        measured_context_tokens=_measured_context_tokens(context),
        tokenizer_id="whitespace",
        metadata=_metadata(record),
    )


def oolong_case_from_record(record: dict, line_number: int = 0) -> BenchmarkCase:
    context = _oolong_context(record)
    question = _first(record, ["question", "query", "instruction"], default="")
    answer = _first(record, ["gold_answer", "answer", "answers", "label", "gold"], default="")
    task_id = str(_first(record, ["task_id", "id", "qid"], default=f"oolong_{line_number}"))
    return BenchmarkCase(
        benchmark="oolong",
        benchmark_id="oolong",
        dataset_origin=str(_first(record, ["dataset_origin"], default="unknown")),
        task_id=task_id,
        context=context,
        question=question,
        gold_answer=answer,
        gold_evidence_span_ids=_list_field(record, ["gold_evidence_span_ids", "evidence_span_ids"]),
        gold_entities=_list_field(record, ["gold_entities", "entities"]),
        expected_hops=_optional_int(_first(record, ["expected_hops"], default=None)),
        answerable=bool(_first(record, ["answerable"], default=True)),
        context_tokens=_context_tokens(record, context),
        benchmark_context_len=_optional_int(_first(record, ["context_len"], default=None)),
        measured_context_tokens=_measured_context_tokens(context),
        tokenizer_id="whitespace",
        metadata={
            **_metadata(record),
            "native_fields": _native_fields(
                record,
                [
                    "id",
                    "context_len",
                    "dataset",
                    "context_window_text",
                    "context_window_text_with_labels",
                    "question",
                    "task_group",
                    "task",
                    "answer",
                    "answer_type",
                    "input_subset",
                    "num_labels",
                    "context_window_id",
                ],
            ),
        },
    )


def oolong_pairs_case_from_record(record: dict, line_number: int = 0) -> BenchmarkCase:
    context = _oolong_context(record)
    question = _first(record, ["question", "query", "instruction"], default="")
    answer = _first(record, ["gold_answer", "answer", "answers", "label", "gold"], default="")
    task_id = str(_first(record, ["task_id", "id", "qid"], default=f"oolong_pairs_{line_number}"))
    return BenchmarkCase(
        benchmark="oolong_pairs",
        benchmark_id="oolong_pairs",
        dataset_origin=str(_first(record, ["dataset_origin"], default="unknown")),
        task_id=task_id,
        context=context,
        question=question,
        gold_answer=answer,
        gold_evidence_span_ids=_list_field(record, ["gold_evidence_span_ids", "evidence_span_ids"]),
        gold_entities=_list_field(record, ["gold_entities", "entities"]),
        expected_hops=_optional_int(_first(record, ["expected_hops"], default=None)),
        answerable=bool(_first(record, ["answerable"], default=True)),
        context_tokens=_context_tokens(record, context),
        benchmark_context_len=_optional_int(_first(record, ["context_len"], default=None)),
        measured_context_tokens=_measured_context_tokens(context),
        tokenizer_id="whitespace",
        metadata=_metadata(record),
    )


def make_s_niah_cases(
    count: int,
    context_tokens: int,
    *,
    prefix: str = "synthetic_s_niah",
) -> list[BenchmarkCase]:
    cases = []
    filler = " ".join(["hay"] * max(context_tokens - 20, 1))
    for index in range(count):
        answer = f"needle-{index:04d}"
        context = f"{filler} The secret code is {answer}. {filler}"
        cases.append(
            BenchmarkCase(
                benchmark="s_niah",
                benchmark_id="sniah_compatible_synthetic",
                dataset_origin="local_generated",
                task_id=f"{prefix}_{context_tokens}_{index}",
                context=context,
                question="What is the secret code?",
                gold_answer=answer,
                gold_evidence_span_ids=[f"span_{index}"],
                expected_hops=1,
                answerable=True,
                context_tokens=context_tokens,
                benchmark_context_len=context_tokens,
                measured_context_tokens=_measured_context_tokens(context),
                tokenizer_id="whitespace",
                metadata={"synthetic": True},
            )
        )
    return cases


def _benchmark_name(record: dict) -> BenchmarkName:
    value = record.get("benchmark")
    if value in {"s_niah", "oolong", "oolong_pairs", "factlens"}:
        return value
    raise ValueError("benchmark must be provided in record or loader argument")


def _oolong_context(record: dict) -> str:
    value = record.get("context_window_text")
    if isinstance(value, str):
        return value
    for key in ["context", "input", "prompt"]:
        value = record.get(key)
        if isinstance(value, str):
            return value
    records = record.get("records") or record.get("items") or record.get("documents")
    if isinstance(records, list):
        return "\n".join(_record_to_text(item) for item in records)
    return ""


def _record_to_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _first(record: dict, names: Iterable[str], default=None):
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _list_field(record: dict, names: Iterable[str]) -> list[str]:
    value = _first(record, names, default=[])
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _context_tokens(record: dict, context: str) -> int:
    value = _first(record, ["context_tokens", "tokens", "input_tokens"], default=None)
    if value is not None:
        return int(value)
    return len(context.split())


def _measured_context_tokens(context: str) -> int:
    return len(context.split())


def _metadata(record: dict) -> dict:
    excluded = {
        "benchmark",
        "task_id",
        "id",
        "qid",
        "context",
        "haystack",
        "input",
        "prompt",
        "records",
        "items",
        "documents",
        "question",
        "query",
        "instruction",
        "gold_answer",
        "answer",
        "answers",
        "needle",
        "target",
        "label",
        "gold",
        "gold_evidence_span_ids",
        "evidence_span_ids",
        "gold_entities",
        "entities",
        "expected_hops",
        "answerable",
        "context_tokens",
        "tokens",
        "input_tokens",
    }
    return {key: value for key, value in record.items() if key not in excluded}


def _native_fields(record: dict, names: list[str]) -> dict:
    return {name: record[name] for name in names if name in record}
