from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


LabelValue = Literal["ham", "spam"]
QuerySemantics = Literal[
    "deterministic_operation",
    "relational_lookup",
    "multi_hop_reasoning",
    "open_ended_semantic_search",
    "unsupported_analytic_operation",
]


class OOLONGRecordFact(BaseModel):
    record_id: str
    record_index: int
    user_id: str
    label: LabelValue | None = None
    occurred_at: datetime | None = None
    evidence_span_id: str
    source_chunk_id: str


class OOLONGQueryPlan(BaseModel):
    operation: Literal["count_distinct", "rank", "compare_group_rate", "label_mode"]
    target: Literal["source_record_id", "user_id", "label"] = "source_record_id"
    unit: Literal["distinct_record_id"] = "distinct_record_id"
    subject_filter: dict[str, str] = Field(default_factory=dict)
    predicate_filter: dict[str, str] = Field(default_factory=dict)
    group_by: str | None = None
    sort: Literal["count_desc", "count_asc"] | None = None
    rank: int | None = None
    partition: dict[str, str] = Field(default_factory=dict)
    numerator_filter: dict[str, str] = Field(default_factory=dict)
    denominator: str | None = None
    planner_source: Literal["question_text"] = "question_text"
    planner_version: str = "oolong_question_planner_v2"
    matched_patterns: list[str] = Field(default_factory=list)


class OOLONGOperationResult(BaseModel):
    status: Literal["complete", "unsupported", "insufficient_facts"]
    plan: OOLONGQueryPlan | None = None
    query_semantics: QuerySemantics = "open_ended_semantic_search"
    value: int | str | None = None
    answer_text: str | None = None
    evidence_span_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    matched_record_ids: list[str] = Field(default_factory=list)
    reason: str


def normalize_user_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    for prefix in ["user_", "user:", "User ", "user "]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    match = re.search(r"\d+", text)
    return match.group(0) if match else text or None


def normalize_label(value: str | None) -> LabelValue | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.removeprefix("label_").removeprefix("label:")
    text = text.strip(" .'\"[]()")
    if text in {"ham", "not spam"}:
        return "ham"
    if text == "spam":
        return "spam"
    return None


def parse_oolong_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    for fmt in ["%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"]:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def build_record_fact(
    *,
    record_id: str,
    record_index: int,
    user_id: str,
    label: str | None,
    date: str | None,
    evidence_span_id: str,
    source_chunk_id: str,
) -> OOLONGRecordFact:
    canonical_user_id = normalize_user_id(user_id) or user_id
    return OOLONGRecordFact(
        record_id=record_id,
        record_index=record_index,
        user_id=canonical_user_id,
        label=normalize_label(label),
        occurred_at=parse_oolong_date(date),
        evidence_span_id=evidence_span_id,
        source_chunk_id=source_chunk_id,
    )


def plan_oolong_operation(question: str) -> OOLONGQueryPlan | None:
    text = question.strip()
    lowered = text.lower()
    if "label" in lowered and "most common" in lowered:
        return OOLONGQueryPlan(
            operation="label_mode",
            target="label",
            group_by="label",
            sort="count_desc",
            matched_patterns=["label_most_common"],
        )
    if "label" in lowered and "least common" in lowered:
        return OOLONGQueryPlan(
            operation="label_mode",
            target="label",
            group_by="label",
            sort="count_asc",
            matched_patterns=["label_least_common"],
        )
    rank = _extract_rank(lowered)
    if rank is not None and "user" in lowered and any(term in lowered for term in ["most", "frequent", "often", "represented"]):
        return OOLONGQueryPlan(
            operation="rank",
            target="user_id",
            group_by="user_id",
            sort="count_desc",
            rank=rank,
            matched_patterns=["ordinal_user_frequency"],
        )

    temporal_label = normalize_label(_extract_label(text))
    boundary = _extract_date_literal(text)
    if temporal_label and boundary and "before" in lowered and "after" in lowered:
        return OOLONGQueryPlan(
            operation="compare_group_rate",
            target="source_record_id",
            numerator_filter={"label": temporal_label},
            partition={
                "field": "timestamp",
                "boundary": boundary,
                "left_operator": "lt",
                "right_operator": "gte",
            },
            denominator="all_matching_records",
            matched_patterns=["temporal_before_after_rate_candidate"],
        )

    if "how many" not in lowered:
        return None

    user_id = normalize_user_id(_extract_user_id(text))
    label = normalize_label(_extract_label(text))
    if not user_id or not label:
        return None
    return OOLONGQueryPlan(
        operation="count_distinct",
        target="source_record_id",
        subject_filter={"user_id": user_id},
        predicate_filter={"label": label},
        matched_patterns=["how_many_user_label"],
    )


def execute_oolong_operation(
    plan: OOLONGQueryPlan | None,
    facts: list[OOLONGRecordFact],
) -> OOLONGOperationResult:
    if plan is None:
        return OOLONGOperationResult(
            status="unsupported",
            query_semantics="open_ended_semantic_search",
            reason="No supported OOLONG operation matched.",
        )
    if plan.operation == "rank":
        return _execute_user_rank(plan, facts)
    if plan.operation == "label_mode":
        return _execute_label_mode(plan, facts)
    if plan.operation == "compare_group_rate":
        return OOLONGOperationResult(
            status="unsupported",
            plan=plan,
            query_semantics="unsupported_analytic_operation",
            reason="Temporal compare_group_rate planning is diagnostic-only until official OOLONG temporal boundary semantics are confirmed.",
        )
    if plan.operation != "count_distinct":
        return OOLONGOperationResult(
            status="unsupported",
            plan=plan,
            query_semantics="unsupported_analytic_operation",
            reason="Unsupported operation.",
        )
    if any(fact.label is None for fact in facts):
        return OOLONGOperationResult(
            status="insufficient_facts",
            plan=plan,
            query_semantics="deterministic_operation",
            reason="At least one record is missing a normalized label.",
        )

    user_id = plan.subject_filter.get("user_id")
    label = plan.predicate_filter.get("label")
    matched = [
        fact
        for fact in facts
        if fact.user_id == user_id and fact.label == label
    ]
    record_ids = list(dict.fromkeys(fact.record_id for fact in matched))
    return OOLONGOperationResult(
        status="complete",
        plan=plan,
        query_semantics="deterministic_operation",
        value=len(record_ids),
        answer_text=f"Answer: {len(record_ids)}",
        evidence_span_ids=list(dict.fromkeys(fact.evidence_span_id for fact in matched)),
        source_chunk_ids=list(dict.fromkeys(fact.source_chunk_id for fact in matched)),
        matched_record_ids=record_ids,
        reason="Executed deterministic count over normalized OOLONGRecordFact records.",
    )


def _execute_label_mode(
    plan: OOLONGQueryPlan,
    facts: list[OOLONGRecordFact],
) -> OOLONGOperationResult:
    if plan.group_by != "label" or plan.target != "label" or plan.sort not in {"count_desc", "count_asc"}:
        return OOLONGOperationResult(
            status="unsupported",
            plan=plan,
            query_semantics="unsupported_analytic_operation",
            reason="Unsupported label mode plan.",
        )
    labelled = [fact for fact in facts if fact.label is not None]
    if len(labelled) != len(facts):
        return OOLONGOperationResult(
            status="insufficient_facts",
            plan=plan,
            query_semantics="deterministic_operation",
            reason="At least one record is missing a normalized label.",
        )
    grouped: dict[str, list[OOLONGRecordFact]] = {}
    for fact in labelled:
        grouped.setdefault(fact.label, []).append(fact)
    if not grouped:
        return OOLONGOperationResult(
            status="insufficient_facts",
            plan=plan,
            query_semantics="deterministic_operation",
            reason="No labelled records available.",
        )
    reverse = plan.sort == "count_desc"
    label, matched = sorted(
        grouped.items(),
        key=lambda item: (len({fact.record_id for fact in item[1]}), item[0]),
        reverse=reverse,
    )[0]
    return OOLONGOperationResult(
        status="complete",
        plan=plan,
        query_semantics="deterministic_operation",
        value=label,
        answer_text=f"Label: {label}",
        evidence_span_ids=list(dict.fromkeys(fact.evidence_span_id for fact in matched)),
        source_chunk_ids=list(dict.fromkeys(fact.source_chunk_id for fact in matched)),
        matched_record_ids=list(dict.fromkeys(fact.record_id for fact in matched)),
        reason="Executed deterministic label mode over normalized OOLONGRecordFact records.",
    )


def _execute_user_rank(
    plan: OOLONGQueryPlan,
    facts: list[OOLONGRecordFact],
) -> OOLONGOperationResult:
    if plan.group_by != "user_id" or plan.target != "user_id" or plan.rank is None:
        return OOLONGOperationResult(
            status="unsupported",
            plan=plan,
            query_semantics="unsupported_analytic_operation",
            reason="Unsupported rank plan.",
        )
    user_records: dict[str, list[OOLONGRecordFact]] = {}
    for fact in facts:
        user_records.setdefault(fact.user_id, []).append(fact)
    ranked = sorted(
        user_records.items(),
        key=lambda item: (-len({fact.record_id for fact in item[1]}), item[0]),
    )
    if plan.rank < 1 or plan.rank > len(ranked):
        return OOLONGOperationResult(
            status="insufficient_facts",
            plan=plan,
            query_semantics="deterministic_operation",
            reason="Rank is outside grouped facts.",
        )
    user_id, matched = ranked[plan.rank - 1]
    return OOLONGOperationResult(
        status="complete",
        plan=plan,
        query_semantics="deterministic_operation",
        value=user_id,
        answer_text=str(user_id),
        evidence_span_ids=list(dict.fromkeys(fact.evidence_span_id for fact in matched)),
        source_chunk_ids=list(dict.fromkeys(fact.source_chunk_id for fact in matched)),
        matched_record_ids=list(dict.fromkeys(fact.record_id for fact in matched)),
        reason="Executed deterministic user rank over normalized OOLONGRecordFact records.",
    )


def _extract_user_id(text: str) -> str | None:
    patterns = [
        r"user ids?\s+([0-9]+)",
        r"user\s+([0-9]+)",
        r"belonging to user\s+([0-9]+)",
        r"associated with user ids?\s+([0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_label(text: str) -> str | None:
    patterns = [
        r"label\s*[:=]?\s*['\"]?(not spam|ham|spam)\b",
        r"classified as\s+['\"]?(not spam|ham|spam)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    for label in ["ham", "spam"]:
        if re.search(rf"\b{label}\b", text, flags=re.IGNORECASE):
            return label
    return None


def _extract_rank(lowered: str) -> int | None:
    ordinals = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
    }
    for token, value in ordinals.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            return value
    match = re.search(r"rank\s+([0-9]+)", lowered)
    return int(match.group(1)) if match else None


def _extract_date_literal(text: str) -> str | None:
    match = re.search(r"\b([0-9]{4}-[0-9]{2}-[0-9]{2})\b", text)
    return match.group(1) if match else None
