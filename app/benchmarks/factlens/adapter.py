from __future__ import annotations

from app.benchmarks.models import BenchmarkCase


def factlens_case_from_record(record: dict, line_number: int = 0) -> BenchmarkCase:
    context = str(record.get("context") or "")
    task_id = str(record.get("task_id") or record.get("id") or f"factlens_{line_number}")
    metadata = {
        "native_fields": dict(record),
        "factlens": {
            "claim_id": record.get("claim_id") or task_id,
            "subclaims": list(record.get("subclaims") or []),
            "graph_edges": list(record.get("graph_edges") or []),
            "contradictions": list(record.get("contradictions") or []),
            "aggregated_label": record.get("aggregated_label"),
            "complexity": dict(record.get("factlens_complexity") or {}),
        },
    }
    if record.get("source"):
        metadata["source"] = dict(record["source"])
    if record.get("factlens_complexity"):
        metadata["factlens_complexity"] = dict(record["factlens_complexity"])
        metadata["native_fields"].update(record["factlens_complexity"])
    return BenchmarkCase(
        benchmark="factlens",
        benchmark_id="factlens_audit",
        dataset_origin=str(record.get("dataset_origin") or "local_generated"),
        task_id=task_id,
        context=context,
        question=str(record.get("claim") or record.get("question") or ""),
        gold_answer=[str(record.get("answer") or "supported")],
        gold_evidence_span_ids=[str(item) for item in record.get("gold_evidence_span_ids", [])],
        expected_hops=int(record.get("expected_hops") or 1),
        answerable=bool(record.get("answerable", True)),
        context_tokens=len(context.split()),
        benchmark_context_len=record.get("context_len"),
        measured_context_tokens=len(context.split()),
        tokenizer_id="whitespace",
        metadata=metadata,
    )


def make_factlens_audit_case() -> BenchmarkCase:
    return factlens_case_from_record(
        {
            "id": "factlens_local_chain_001",
            "dataset_origin": "local_generated",
            "context": (
                "[e1] Mira founded Northstar Labs in 2018.\n"
                "[e2] Northstar Labs acquired HelioAI in 2021.\n"
                "[e3] HelioAI built the search engine used by Argo Health.\n"
                "[e4] Argo Health launched its Berlin clinic in 2022.\n"
                "[e5] The Berlin clinic used HelioAI's triage search workflow."
            ),
            "claim": (
                "Mira is connected to Argo Health through Northstar Labs acquiring HelioAI, "
                "whose search workflow Argo Health used in its Berlin clinic."
            ),
            "answer": "supported",
            "gold_evidence_span_ids": ["e1", "e2", "e3", "e4", "e5"],
            "expected_hops": 5,
            "subclaims": [
                {
                    "subclaim_id": "s1",
                    "text": "Mira founded Northstar Labs.",
                    "evidence_span_ids": ["e1"],
                    "graph_fact_ids": ["entity:mira", "entity:northstar_labs"],
                    "verification_modes": [
                        "flat_subclaim_verification",
                        "graph_query_verification",
                        "graph_shared_evidence",
                        "graph_recursive_completeness",
                    ],
                },
                {
                    "subclaim_id": "s2",
                    "text": "Northstar Labs acquired HelioAI.",
                    "evidence_span_ids": ["e2"],
                    "graph_fact_ids": ["entity:northstar_labs", "entity:helioai"],
                    "verification_modes": [
                        "flat_subclaim_verification",
                        "graph_query_verification",
                        "graph_shared_evidence",
                        "graph_recursive_completeness",
                    ],
                },
                {
                    "subclaim_id": "s3",
                    "text": "HelioAI built the search engine used by Argo Health.",
                    "evidence_span_ids": ["e3"],
                    "graph_fact_ids": ["entity:helioai", "entity:argo_health"],
                    "verification_modes": [
                        "flat_subclaim_verification",
                        "graph_query_verification",
                        "graph_shared_evidence",
                        "graph_recursive_completeness",
                    ],
                },
                {
                    "subclaim_id": "s4",
                    "text": "Argo Health launched a Berlin clinic.",
                    "evidence_span_ids": ["e4"],
                    "graph_fact_ids": ["entity:argo_health", "location:berlin_clinic"],
                    "verification_modes": [
                        "graph_query_verification",
                        "graph_shared_evidence",
                        "graph_recursive_completeness",
                    ],
                },
                {
                    "subclaim_id": "s5",
                    "text": "The Berlin clinic used HelioAI's triage search workflow.",
                    "evidence_span_ids": ["e5"],
                    "graph_fact_ids": ["entity:helioai", "location:berlin_clinic"],
                    "verification_modes": [
                        "graph_shared_evidence",
                        "graph_recursive_completeness",
                    ],
                },
            ],
            "graph_edges": [
                {
                    "edge_id": "edge_s1_s2_northstar",
                    "source_subclaim_id": "s1",
                    "target_subclaim_id": "s2",
                    "relation": "shared_entity",
                    "graph_fact_id": "entity:northstar_labs",
                    "evidence_span_ids": ["e1", "e2"],
                },
                {
                    "edge_id": "edge_s2_s3_helioai",
                    "source_subclaim_id": "s2",
                    "target_subclaim_id": "s3",
                    "relation": "shared_entity",
                    "graph_fact_id": "entity:helioai",
                    "evidence_span_ids": ["e2", "e3"],
                },
                {
                    "edge_id": "edge_s3_s4_argo",
                    "source_subclaim_id": "s3",
                    "target_subclaim_id": "s4",
                    "relation": "shared_entity",
                    "graph_fact_id": "entity:argo_health",
                    "evidence_span_ids": ["e3", "e4"],
                },
                {
                    "edge_id": "edge_s4_s5_clinic",
                    "source_subclaim_id": "s4",
                    "target_subclaim_id": "s5",
                    "relation": "shared_location",
                    "graph_fact_id": "location:berlin_clinic",
                    "evidence_span_ids": ["e4", "e5"],
                },
            ],
            "contradictions": [],
        }
    )
