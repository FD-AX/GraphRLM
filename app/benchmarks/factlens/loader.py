from __future__ import annotations

import ast
import csv
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.benchmarks.factlens.adapter import factlens_case_from_record
from app.benchmarks.factlens.models import FactLensAuditMode
from app.benchmarks.models import BenchmarkCase


FACTLENS_LOADER_VERSION = "factlens_official_csv_loader_v1"


@dataclass(frozen=True)
class FactLensOfficialRecord:
    ind: str
    claim: str
    sub_claims: list[str]
    labels: list[str]
    aggregated_label: bool
    native_row: dict[str, str]


def load_factlens_official_csv(path: Path) -> list[FactLensOfficialRecord]:
    records: list[FactLensOfficialRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            subclaims = [str(value) for value in ast.literal_eval(row["sub_claims"])]
            labels = [_normalize_label(value) for value in ast.literal_eval(row["labels"])]
            records.append(
                FactLensOfficialRecord(
                    ind=str(row["ind"]),
                    claim=str(row["claim"]),
                    sub_claims=subclaims,
                    labels=labels,
                    aggregated_label=_normalize_bool(row["aggregated_label"]),
                    native_row=dict(row),
                )
            )
    return records


def select_factlens_matrix_cases(
    records: list[FactLensOfficialRecord],
    *,
    per_bucket: int = 10,
) -> list[FactLensOfficialRecord]:
    buckets = {"simple": [], "medium": [], "complex": []}
    for record in records:
        features = compute_complexity_features(record.sub_claims)
        buckets[features["complexity_bucket"]].append(record)
    selected: list[FactLensOfficialRecord] = []
    for bucket in ("simple", "medium", "complex"):
        if len(buckets[bucket]) < per_bucket:
            raise ValueError(
                f"FactLens bucket {bucket!r} has only {len(buckets[bucket])} records; "
                f"{per_bucket} required"
            )
        selected.extend(buckets[bucket][:per_bucket])
    return selected


def factlens_case_from_official_record(
    record: FactLensOfficialRecord,
    *,
    dataset_revision: str,
    source_path: Path,
) -> BenchmarkCase:
    features = compute_complexity_features(record.sub_claims)
    subclaims = []
    for index, (text, label) in enumerate(zip(record.sub_claims, record.labels), start=1):
        subclaim_id = f"s{index}"
        evidence_span_id = f"factlens:{record.ind}:subclaim:{index}"
        fact_ids = _graph_fact_ids(text)
        subclaims.append(
            {
                "subclaim_id": subclaim_id,
                "text": text,
                "label": label,
                "evidence_span_ids": [evidence_span_id],
                "graph_fact_ids": fact_ids,
                "verification_modes": _verification_modes_for_subclaim(
                    subclaim_index=index,
                    subclaim_count=len(record.sub_claims),
                    dependency_degree=_dependency_degree(fact_ids, record.sub_claims),
                ),
            }
        )
    graph_edges = _graph_edges(record.ind, record.sub_claims)
    answer = "supported" if record.aggregated_label else "unsupported"
    context = "\n".join(
        f"[{index}] {subclaim}"
        for index, subclaim in enumerate(record.sub_claims, start=1)
    )
    return factlens_case_from_record(
        {
            "id": f"factlens_official_{record.ind}",
            "claim_id": record.ind,
            "dataset_origin": "official",
            "context": context,
            "claim": record.claim,
            "answer": answer,
            "gold_evidence_span_ids": [
                f"factlens:{record.ind}:subclaim:{index}"
                for index in range(1, len(record.sub_claims) + 1)
            ],
            "expected_hops": max(1, features["cross_subclaim_dependency_count"]),
            "subclaims": subclaims,
            "graph_edges": graph_edges,
            "contradictions": [
                f"s{index}"
                for index, label in enumerate(record.labels, start=1)
                if label == "false"
            ],
            "aggregated_label": record.aggregated_label,
            "source": {
                "dataset": "megagonlabs/factlens",
                "file": str(source_path),
                "revision": dataset_revision,
                "resolved_revision": dataset_revision,
                "loader_version": FACTLENS_LOADER_VERSION,
            },
            "factlens_complexity": features,
            "native_fields": {
                "ind": record.ind,
                "claim": record.claim,
                "sub_claims": record.sub_claims,
                "labels": record.labels,
                "aggregated_label": record.aggregated_label,
                **features,
            },
        }
    )


def compute_complexity_features(subclaims: list[str]) -> dict:
    facts_by_subclaim = [_graph_fact_ids(subclaim) for subclaim in subclaims]
    fact_counts = Counter(fact for facts in facts_by_subclaim for fact in facts)
    shared_facts = {fact for fact, count in fact_counts.items() if count > 1}
    edge_count = 0
    for left_index in range(len(facts_by_subclaim)):
        for right_index in range(left_index + 1, len(facts_by_subclaim)):
            if set(facts_by_subclaim[left_index]) & set(facts_by_subclaim[right_index]):
                edge_count += 1
    subclaim_count = len(subclaims)
    if subclaim_count <= 2 and edge_count <= 1:
        bucket = "simple"
    elif subclaim_count <= 4 and edge_count <= 4:
        bucket = "medium"
    else:
        bucket = "complex"
    return {
        "subclaim_count": subclaim_count,
        "shared_entity_count": len(shared_facts),
        "shared_evidence_count": len(shared_facts),
        "cross_subclaim_dependency_count": edge_count,
        "complexity_bucket": bucket,
        "subclaim_count_bucket": _subclaim_count_bucket(subclaim_count),
        "shared_evidence_required": bool(shared_facts),
        "cross_dependency_bucket": _cross_dependency_bucket(edge_count),
    }


def factlens_repo_revision(repo_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _verification_modes_for_subclaim(
    *,
    subclaim_index: int,
    subclaim_count: int,
    dependency_degree: int,
) -> list[FactLensAuditMode]:
    modes: list[FactLensAuditMode] = ["graph_shared_evidence"]
    if subclaim_index == 1 or dependency_degree <= 1 or subclaim_count <= 2:
        modes.append("graph_query_verification")
    if dependency_degree == 0 or subclaim_count <= 2:
        modes.append("flat_subclaim_verification")
    return modes


def _dependency_degree(fact_ids: list[str], all_subclaims: list[str]) -> int:
    count = 0
    fact_set = set(fact_ids)
    for subclaim in all_subclaims:
        other = set(_graph_fact_ids(subclaim))
        if fact_set & other:
            count += 1
    return max(0, count - 1)


def _graph_edges(record_id: str, subclaims: list[str]) -> list[dict]:
    facts_by_subclaim = [_graph_fact_ids(subclaim) for subclaim in subclaims]
    edges = []
    for left_index in range(len(facts_by_subclaim)):
        for right_index in range(left_index + 1, len(facts_by_subclaim)):
            shared = sorted(set(facts_by_subclaim[left_index]) & set(facts_by_subclaim[right_index]))
            for fact in shared:
                edges.append(
                    {
                        "edge_id": f"factlens:{record_id}:edge:{left_index + 1}:{right_index + 1}:{fact}",
                        "source_subclaim_id": f"s{left_index + 1}",
                        "target_subclaim_id": f"s{right_index + 1}",
                        "relation": "shared_fact",
                        "graph_fact_id": fact,
                        "evidence_span_ids": [
                            f"factlens:{record_id}:subclaim:{left_index + 1}",
                            f"factlens:{record_id}:subclaim:{right_index + 1}",
                        ],
                    }
                )
    return edges


def _graph_fact_ids(text: str) -> list[str]:
    tokens = [
        _normalize_token(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'.-]*|[0-9]+(?:\.[0-9]+)?", text)
    ]
    return list(
        dict.fromkeys(
            f"term:{token}"
            for token in tokens
            if token and token not in _STOPWORDS and (len(token) > 3 or token.isdigit())
        )
    )


def _normalize_label(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "supported"}


def _normalize_token(token: str) -> str:
    return token.strip(".,;:!?()[]{}\"'").lower()


def _subclaim_count_bucket(count: int) -> str:
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    return "6+"


def _cross_dependency_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 2:
        return "1-2"
    return "3+"


_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "been",
    "being",
    "can",
    "from",
    "had",
    "has",
    "have",
    "held",
    "his",
    "into",
    "its",
    "more",
    "named",
    "than",
    "that",
    "the",
    "their",
    "there",
    "this",
    "was",
    "were",
    "which",
    "while",
    "with",
}
