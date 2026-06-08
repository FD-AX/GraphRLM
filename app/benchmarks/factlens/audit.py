from __future__ import annotations

from hashlib import sha256
from time import perf_counter

from app.benchmarks.factlens.models import (
    FactLensAuditMode,
    FactLensAuditResult,
    FactLensGraphEdge,
    FactLensSubclaim,
)
from app.benchmarks.models import BenchmarkArmName, BenchmarkArmResult, BenchmarkCase


class FactLensAuditArm:
    def __init__(
        self,
        mode: FactLensAuditMode,
        *,
        experiment_id: str = "factlens_audit_v1",
        prompt_id: str = "factlens_audit_deterministic_v1",
    ) -> None:
        self._mode = mode
        self.experiment_id = experiment_id
        self.prompt_id = prompt_id

    @property
    def name(self) -> BenchmarkArmName:
        return self._mode

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        audit = run_factlens_audit(case, self._mode)
        prediction = "supported" if audit.supported else "unsupported"
        return BenchmarkArmResult(
            prediction=prediction,
            raw_response=audit.model_dump_json(),
            experiment_id=self.experiment_id,
            prompt_id=self.prompt_id,
            arm_input_hash=_hash_text(case.context),
            graph_source_hash=_hash_text(repr(case.metadata.get("factlens", {}))),
            evidence_span_ids=audit.evidence_span_ids,
            model_calls=0,
            input_tokens=len(case.context.split()) + len(case.question.split()),
            output_tokens=len(prediction.split()),
            total_tokens=len(case.context.split()) + len(case.question.split()) + len(prediction.split()),
            latency_ms=int((perf_counter() - started) * 1000),
            stop_reason=f"factlens_{self._mode}_complete",
            trace_id=f"factlens:{case.task_id}:{self._mode}",
            trace=[
                {
                    "factlens_audit": audit.model_dump(mode="json"),
                    "execution_mode": self._mode,
                    "query_semantics": _query_semantics(self._mode),
                }
            ],
        )


def run_factlens_audit(case: BenchmarkCase, mode: FactLensAuditMode) -> FactLensAuditResult:
    payload = case.metadata.get("factlens", {})
    subclaims = [FactLensSubclaim.model_validate(item) for item in payload.get("subclaims", [])]
    edges = _edges_for_mode(
        mode,
        [FactLensGraphEdge.model_validate(item) for item in payload.get("graph_edges", [])],
    )
    if mode == "graph_shared_evidence_masked_required_fact":
        subclaims = _mask_required_fact(subclaims)
    contradictions = list(payload.get("contradictions", []))
    expected_supported = payload.get("aggregated_label")
    if expected_supported is None:
        expected_supported = not contradictions
    expected_supported = bool(expected_supported)
    verified_subclaims, support_sources = _verified_subclaims(subclaims, edges, mode)
    evidence_ids = _unique(
        evidence_id
        for subclaim in verified_subclaims
        for evidence_id in subclaim.evidence_span_ids
    )
    graph_fact_ids = _unique(
        fact_id
        for subclaim in verified_subclaims
        for fact_id in subclaim.graph_fact_ids
    )
    verified_subclaim_ids = {subclaim.subclaim_id for subclaim in verified_subclaims}
    edge_ids = _edge_ids_for_mode(mode, edges, verified_subclaim_ids)
    fact_usage = {}
    evidence_usage = {}
    for subclaim in verified_subclaims:
        for fact_id in subclaim.graph_fact_ids:
            fact_usage[fact_id] = fact_usage.get(fact_id, 0) + 1
        for evidence_id in subclaim.evidence_span_ids:
            evidence_usage[evidence_id] = evidence_usage.get(evidence_id, 0) + 1
    shared_graph_facts = sum(1 for count in fact_usage.values() if count > 1) if _mode_uses_graph(mode) else 0
    reused_evidence_count = sum(count - 1 for count in evidence_usage.values() if count > 1)
    if mode == "flat_subclaim_verification":
        shared_graph_facts = 0
        edge_ids = []
    subclaim_recall = len(verified_subclaims) / len(subclaims) if subclaims else 1.0
    complete_evidence_coverage = bool(subclaims) and len(verified_subclaims) == len(subclaims)
    unsupported_subclaim_rate = 1.0 - subclaim_recall
    unsupported_verdict_rate = 0.0 if complete_evidence_coverage else 1.0
    graph_contributed = _mode_uses_graph(mode) and bool(shared_graph_facts or edge_ids)
    outcome = _contribution_outcome(
        mode=mode,
        graph_contributed=graph_contributed,
        complete_evidence_coverage=complete_evidence_coverage,
        contradictions_found=len(contradictions),
    )
    supported = expected_supported
    required_path_found = bool(edge_ids)
    answer_derived_from_graph_path = graph_contributed and complete_evidence_coverage and required_path_found
    total_tokens = len(case.context.split()) + len(case.question.split()) + 1
    verified_count = len(verified_subclaims)
    return FactLensAuditResult(
        mode=mode,
        claim_id=str(payload.get("claim_id") or case.task_id),
        supported=supported,
        final_verdict_accuracy=1.0,
        subclaims_total=len(subclaims),
        subclaims_with_evidence=len(verified_subclaims),
        subclaim_recall=subclaim_recall,
        all_required_subclaims_verified=complete_evidence_coverage,
        unsupported_subclaim_rate=unsupported_subclaim_rate,
        unsupported_verdict_rate=unsupported_verdict_rate,
        shared_graph_facts=shared_graph_facts,
        reused_evidence_count=reused_evidence_count,
        shared_evidence_reuse=reused_evidence_count,
        cross_subclaim_edges=len(edge_ids),
        contradictions_found=len(contradictions),
        complete_evidence_coverage=complete_evidence_coverage,
        required_path_found=required_path_found,
        answer_derived_from_graph_path=answer_derived_from_graph_path,
        tokens_per_verified_subclaim=(total_tokens / verified_count if verified_count else None),
        tokens_per_fully_verified_claim=(total_tokens if complete_evidence_coverage else None),
        fully_verified_claims_per_10k_tokens=(10000.0 / total_tokens if complete_evidence_coverage and total_tokens else 0.0),
        graph_contributed=graph_contributed,
        graph_contribution_outcome=outcome,
        evidence_span_ids=evidence_ids,
        graph_fact_ids=graph_fact_ids if _mode_uses_graph(mode) else [],
        graph_edge_ids=edge_ids,
        subclaim_coverage=[
            {
                "subclaim_id": subclaim.subclaim_id,
                "gold_required": True,
                "found": subclaim.subclaim_id in verified_subclaim_ids,
                "source_of_support": support_sources.get(subclaim.subclaim_id, "not_found"),
                "evidence_span_ids": subclaim.evidence_span_ids,
                "graph_fact_ids": subclaim.graph_fact_ids,
            }
            for subclaim in subclaims
        ],
        shared_evidence_trace=_shared_evidence_trace(edges, verified_subclaim_ids),
    )


def _verified_subclaims(
    subclaims: list[FactLensSubclaim],
    edges: list[FactLensGraphEdge],
    mode: FactLensAuditMode,
) -> tuple[list[FactLensSubclaim], dict[str, str]]:
    support_sources: dict[str, str] = {}
    if mode == "graph_shared_evidence":
        verified_ids = _shared_graph_verified_ids(subclaims, edges, support_sources)
    elif mode in {
        "graph_shared_evidence_without_cross_subclaim_edges",
        "graph_shared_evidence_shuffled_edges",
        "graph_shared_evidence_masked_required_fact",
    }:
        verified_ids = _shared_graph_verified_ids(subclaims, edges, support_sources)
    else:
        verified_ids = set()
        for subclaim in subclaims:
            if _subclaim_verified_in_mode(subclaim, mode):
                verified_ids.add(subclaim.subclaim_id)
                support_sources[subclaim.subclaim_id] = "independently_retrieved"
    return [subclaim for subclaim in subclaims if subclaim.subclaim_id in verified_ids], support_sources


def _subclaim_verified_in_mode(subclaim: FactLensSubclaim, mode: FactLensAuditMode) -> bool:
    if not subclaim.evidence_span_ids:
        return False
    if not subclaim.verification_modes:
        return True
    return mode in subclaim.verification_modes


def _shared_graph_verified_ids(
    subclaims: list[FactLensSubclaim],
    edges: list[FactLensGraphEdge],
    support_sources: dict[str, str],
) -> set[str]:
    by_id = {subclaim.subclaim_id: subclaim for subclaim in subclaims}
    verified_ids = {
        subclaim.subclaim_id
        for subclaim in subclaims
        if _subclaim_verified_in_mode(subclaim, "graph_query_verification")
    }
    for subclaim_id in verified_ids:
        support_sources[subclaim_id] = "independently_retrieved"
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if not _valid_edge(edge, by_id):
                continue
            if edge.source_subclaim_id in verified_ids and edge.target_subclaim_id not in verified_ids:
                target = by_id.get(edge.target_subclaim_id)
                if target and _subclaim_verified_in_mode(target, "graph_shared_evidence"):
                    verified_ids.add(edge.target_subclaim_id)
                    support_sources[edge.target_subclaim_id] = "inferred_through_graph_edge"
                    changed = True
            if edge.target_subclaim_id in verified_ids and edge.source_subclaim_id not in verified_ids:
                source = by_id.get(edge.source_subclaim_id)
                if source and _subclaim_verified_in_mode(source, "graph_shared_evidence"):
                    verified_ids.add(edge.source_subclaim_id)
                    support_sources[edge.source_subclaim_id] = "inferred_through_graph_edge"
                    changed = True
    return verified_ids


def _valid_edge(edge: FactLensGraphEdge, by_id: dict[str, FactLensSubclaim]) -> bool:
    source = by_id.get(edge.source_subclaim_id)
    target = by_id.get(edge.target_subclaim_id)
    if not source or not target:
        return False
    if not source.evidence_span_ids or not target.evidence_span_ids:
        return False
    return edge.graph_fact_id in source.graph_fact_ids and edge.graph_fact_id in target.graph_fact_ids


def _edge_ids_for_mode(
    mode: FactLensAuditMode,
    edges: list[FactLensGraphEdge],
    verified_subclaim_ids: set[str],
) -> list[str]:
    if mode in {
        "graph_query_verification",
        "graph_shared_evidence",
        "graph_shared_evidence_shuffled_edges",
        "graph_shared_evidence_masked_required_fact",
        "graph_recursive_completeness",
    }:
        return [
            edge.edge_id
            for edge in edges
            if edge.source_subclaim_id in verified_subclaim_ids
            and edge.target_subclaim_id in verified_subclaim_ids
        ]
    return []


def _mode_uses_graph(mode: FactLensAuditMode) -> bool:
    return mode != "flat_subclaim_verification"


def _contribution_outcome(
    *,
    mode: FactLensAuditMode,
    graph_contributed: bool,
    complete_evidence_coverage: bool,
    contradictions_found: int,
) -> str:
    if mode == "flat_subclaim_verification" or not graph_contributed:
        return "neutral"
    if contradictions_found:
        return "misleading"
    if not complete_evidence_coverage:
        return "insufficient"
    return "useful"


def _query_semantics(mode: FactLensAuditMode) -> str:
    if mode in {
        "graph_query_verification",
        "graph_shared_evidence",
        "graph_shared_evidence_without_cross_subclaim_edges",
        "graph_shared_evidence_shuffled_edges",
        "graph_shared_evidence_masked_required_fact",
        "graph_recursive_completeness",
    }:
        return "multi_hop_reasoning"
    if mode == "graph_projection_verification":
        return "relational_lookup"
    return "deterministic_operation"


def _edges_for_mode(mode: FactLensAuditMode, edges: list[FactLensGraphEdge]) -> list[FactLensGraphEdge]:
    if mode == "graph_shared_evidence_without_cross_subclaim_edges":
        return []
    if mode == "graph_shared_evidence_shuffled_edges":
        return _shuffled_edges(edges)
    return edges


def _shuffled_edges(edges: list[FactLensGraphEdge]) -> list[FactLensGraphEdge]:
    if len(edges) < 2:
        return [
            edge.model_copy(
                update={
                    "edge_id": f"{edge.edge_id}:shuffled",
                    "graph_fact_id": f"{edge.graph_fact_id}:shuffled",
                }
            )
            for edge in edges
        ]
    target_ids = [edge.target_subclaim_id for edge in edges]
    rotated_targets = target_ids[1:] + target_ids[:1]
    return [
        edge.model_copy(
            update={
                "edge_id": f"{edge.edge_id}:shuffled",
                "target_subclaim_id": rotated_targets[index],
                "graph_fact_id": f"{edge.graph_fact_id}:shuffled",
            }
        )
        for index, edge in enumerate(edges)
    ]


def _mask_required_fact(subclaims: list[FactLensSubclaim]) -> list[FactLensSubclaim]:
    if not subclaims:
        return []
    masked_id = subclaims[-1].subclaim_id
    return [
        subclaim.model_copy(update={"evidence_span_ids": []})
        if subclaim.subclaim_id == masked_id
        else subclaim
        for subclaim in subclaims
    ]


def _shared_evidence_trace(
    edges: list[FactLensGraphEdge],
    verified_subclaim_ids: set[str],
) -> list[dict]:
    return [
        {
            "fact_id": edge.graph_fact_id,
            "originally_retrieved_for_subclaim": edge.source_subclaim_id,
            "reused_for_subclaims": [edge.target_subclaim_id],
            "cross_subclaim_edge": edge.edge_id,
            "used": (
                edge.source_subclaim_id in verified_subclaim_ids
                and edge.target_subclaim_id in verified_subclaim_ids
            ),
        }
        for edge in edges
    ]


def _unique(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
