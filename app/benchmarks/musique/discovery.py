from __future__ import annotations

from app.dual_rlm.dynamic_graph_arm import DynamicGraphRLMArm
from app.dual_rlm.models import GraphRLMState


class MuSiQueDiscoveryArm(DynamicGraphRLMArm):
    """DynamicGraphRLMArm with two MuSiQue-contour extensions.

    1. reformulate performs a real re-search: the new subquery is searched
       against the whole index and the loop jumps to the best unvisited
       document. The base runtime only re-encodes the subquery without moving,
       which strands the loop when the seed lands in the wrong graph
       component.
    2. finalize synthesizes an answer from collected evidence when the
       traversal ends without an explicit answer decision, so answer metrics
       reflect the collected evidence instead of returning empty.
    """

    def _reformulate_query(self, state: GraphRLMState) -> GraphRLMState:
        query_embedding = self.index.encoder.encode_query(state["current_subquery"])
        results = self.index.search(
            query_embedding,
            top_k=self.config.graph_top_k,
            visited_document_ids=set(state["visited_document_ids"]),
        )
        if not results:
            state["path"].append(
                {
                    "step": "reformulate_query",
                    "current_subquery": state["current_subquery"],
                    "search_jump": None,
                    "reason": "no_unvisited_results",
                }
            )
            return state
        target = results[0]
        state["current_owner_id"] = target.owner_id
        state["current_document_ids"] = [target.semantic_document_id]
        state["visited_document_ids"] = list(
            dict.fromkeys(state["visited_document_ids"] + [target.semantic_document_id])
        )
        state["visited_owner_ids"] = list(
            dict.fromkeys(
                state["visited_owner_ids"] + [target.owner_id] + target.source_entity_ids
            )
        )
        state["collected_evidence_ids"] = list(
            dict.fromkeys(state["collected_evidence_ids"] + target.evidence_span_ids)
        )
        state["semantic_document_ids"] = list(state["visited_document_ids"])
        state["evidence_span_ids"] = list(state["collected_evidence_ids"])
        state["path"].append(
            {
                "step": "reformulate_query",
                "current_subquery": state["current_subquery"],
                "search_jump": target.semantic_document_id,
                "search_jump_owner": target.owner_id,
                "search_jump_score": round(target.score, 4),
            }
        )
        return state

    def _finalize_answer(self, state: GraphRLMState) -> GraphRLMState:
        state = super()._finalize_answer(state)
        return self._synthesize_if_missing(state)

    def _stop_insufficient(self, state: GraphRLMState) -> GraphRLMState:
        state = super()._stop_insufficient(state)
        return self._synthesize_if_missing(state)

    def _synthesize_if_missing(self, state: GraphRLMState) -> GraphRLMState:
        if state.get("final_answer"):
            return state
        evidence_texts = []
        for document_id in state["visited_document_ids"]:
            try:
                evidence_texts.append(self.index.document_for(document_id).text)
            except KeyError:
                continue
        if not evidence_texts:
            return state
        synthesized = self.gateway.synthesize_answer(
            state["original_query"],
            evidence_texts,
        )
        state["final_answer"] = synthesized.answer.strip() or None
        state["path"].append(
            {
                "step": "synthesize_answer",
                "final_answer": state["final_answer"],
                "confidence": synthesized.confidence,
            }
        )
        return state
