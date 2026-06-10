from __future__ import annotations

from app.dual_rlm.dynamic_graph_arm import (
    DynamicGraphRLMArm,
    _compact_frontier_evidence_refs,
    _transition,
)
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

    def _frontier_document_ids(self, state: GraphRLMState) -> list[str]:
        """Candidate generator for the controller frontier: typed graph links."""
        frontier_ids: list[str] = []
        for document_id in state["current_document_ids"]:
            current = self.index.document_for(document_id)
            frontier_ids.extend(
                sorted(
                    self.index.frontier_document_ids(
                        current,
                        cap=self.index.encoder.config.local_frontier_cap,
                    )
                )
            )
        return list(dict.fromkeys(frontier_ids))

    def _inspect_local(self, state: GraphRLMState) -> GraphRLMState:
        if state.get("stop_reason"):
            return state
        current = (
            self.index.document_for(state["current_document_ids"][0])
            if state["current_document_ids"]
            else None
        )
        frontier = []
        for target_document_id in self._frontier_document_ids(state):
            if target_document_id in state["visited_document_ids"]:
                continue
            target = self.index.document_for(target_document_id)
            frontier.append(self._shape_transition(_transition(current or target, target)))
        raw_frontier_count = len(frontier)
        frontier, omitted_evidence_ids, newly_presented = _compact_frontier_evidence_refs(
            frontier,
            previously_presented=set(state.get("presented_evidence_ids", [])),
        )
        state["presented_evidence_ids"] = list(
            dict.fromkeys(state.get("presented_evidence_ids", []) + newly_presented)
        )
        state["presented_frontier"] = frontier
        state["frontier_transition_ids"] = [item["transition_id"] for item in frontier]
        state["frontier_ids"] = list(state["frontier_transition_ids"])
        state["path"].append(
            {
                "step": "inspect_local",
                "current_document_ids": list(state["current_document_ids"]),
                "frontier_transition_ids": list(state["frontier_transition_ids"]),
                "raw_frontier_count": raw_frontier_count,
                "compacted_frontier_count": len(frontier),
                "omitted_repeated_evidence_span_ids": omitted_evidence_ids,
            }
        )
        return state

    def _shape_transition(self, transition: dict) -> dict:
        return transition

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


class TextLedgerDiscoveryArm(MuSiQueDiscoveryArm):
    """Budget-matched non-graph controller: same gateway, same actions, same
    budgets and completeness guards as the graph contour, but the frontier
    is generated by dense search against the current subquery instead of
    typed graph links, and typed-link metadata is stripped from presented
    transitions. Isolates the contribution of graph state itself.
    """

    def _frontier_document_ids(self, state: GraphRLMState) -> list[str]:
        query_embedding = self.index.encoder.encode_query(state["current_subquery"])
        results = self.index.search(
            query_embedding,
            top_k=self.index.encoder.config.local_frontier_cap,
            visited_document_ids=set(state["visited_document_ids"]),
        )
        return [result.semantic_document_id for result in results]

    def _shape_transition(self, transition: dict) -> dict:
        return {
            **transition,
            "shared_entity_ids": [],
            "shared_event_ids": [],
        }
