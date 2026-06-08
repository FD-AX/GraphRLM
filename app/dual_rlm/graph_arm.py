from __future__ import annotations

from app.dual_rlm.models import (
    DualRLMConfig,
    GraphViewRef,
    RetrievalArmResult,
)
from app.semantic_encoding.index import GraphSemanticIndex


class GraphRLMArm:
    def __init__(
        self,
        index: GraphSemanticIndex,
        graph_view: GraphViewRef,
        config: DualRLMConfig | None = None,
    ) -> None:
        self.index = index
        self.graph_view = graph_view
        self.config = config or DualRLMConfig()

    def run(self, query: str, run_id: str) -> RetrievalArmResult:
        query_embedding = self.index.encoder.encode_query(query)
        seed_results = self.index.search(query_embedding, top_k=self.config.graph_top_k)
        if not seed_results:
            return RetrievalArmResult(
                arm="graph_rlm",
                stop_reason="no_graph_seed",
                trace_id=f"{run_id}:graph",
            )

        evidence_span_ids = []
        consulted_ids = []
        trace = []
        summaries = []
        for result in seed_results:
            consulted_ids.append(result.semantic_document_id)
            consulted_ids.extend(result.source_entity_ids)
            evidence_span_ids.extend(result.evidence_span_ids)
            summaries.append(result.text)
            trace.append(
                {
                    "step": "seed_retrieval",
                    "semantic_document_id": result.semantic_document_id,
                    "owner_id": result.owner_id,
                    "score": result.score,
                    "evidence_span_ids": result.evidence_span_ids,
                }
            )

        confidence = max(0.0, min(1.0, seed_results[0].score))
        completeness = min(1.0, len(set(evidence_span_ids)) / max(self.config.graph_top_k, 1))
        return RetrievalArmResult(
            arm="graph_rlm",
            answer_candidate=_summarize(summaries),
            evidence_span_ids=list(dict.fromkeys(evidence_span_ids)),
            consulted_object_ids=list(dict.fromkeys(consulted_ids)),
            confidence=confidence,
            completeness=completeness,
            stop_reason="graph_seed_retrieved",
            trace_id=f"{run_id}:graph",
            trace=trace,
        )


def _summarize(texts: list[str]) -> str | None:
    if not texts:
        return None
    return " ".join(texts[:3])
