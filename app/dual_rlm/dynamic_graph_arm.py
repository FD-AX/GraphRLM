from __future__ import annotations

import hashlib
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.dual_rlm.models import (
    DualRLMConfig,
    GraphRLMDecision,
    GraphRLMState,
    GraphViewRef,
    RLMGateway,
    RetrievalArmResult,
)
from app.semantic_encoding.index import GraphSemanticIndex
from app.semantic_encoding.models import GraphSemanticDocument


GraphDestination = Literal[
    "inspect_local",
    "expand_frontier",
    "reformulate_query",
    "verify_evidence",
    "finalize_answer",
    "stop_insufficient",
]


class DynamicGraphRLMArm:
    def __init__(
        self,
        index: GraphSemanticIndex,
        graph_view: GraphViewRef,
        gateway: RLMGateway,
        config: DualRLMConfig | None = None,
    ) -> None:
        self.index = index
        self.graph_view = graph_view
        self.gateway = gateway
        self.config = config or DualRLMConfig()
        self._graph = self._build_graph()

    def run(self, query: str, run_id: str) -> RetrievalArmResult:
        initial_state = self._initial_state(query)
        final_state = self._graph.invoke(initial_state)
        evidence_ids = final_state.get("collected_evidence_ids", [])
        consulted_ids = list(
            dict.fromkeys(
                final_state.get("visited_document_ids", [])
                + final_state.get("visited_owner_ids", [])
                + final_state.get("visited_transition_ids", [])
            )
        )
        model_call_traces = [
            trace
            for trace in self.gateway.model_call_traces
            if trace.purpose == "graph_decision"
        ]
        return RetrievalArmResult(
            arm="graph_rlm",
            answer_candidate=final_state.get("final_answer"),
            evidence_span_ids=list(dict.fromkeys(evidence_ids)),
            consulted_object_ids=consulted_ids,
            confidence=_confidence(final_state),
            completeness=min(1.0, len(set(evidence_ids)) / max(self.config.graph_top_k, 1)),
            stop_reason=final_state.get("stop_reason") or "dynamic_graph_complete",
            trace_id=f"{run_id}:dynamic_graph",
            trace=final_state.get("path", []),
            model_call_traces=model_call_traces,
        )

    def _build_graph(self):
        workflow = StateGraph(GraphRLMState)
        workflow.add_node("locate_seed", self._locate_seed)
        workflow.add_node("inspect_local", self._inspect_local)
        workflow.add_node("graph_decide", self._graph_decide)
        workflow.add_node("expand_frontier", self._expand_frontier)
        workflow.add_node("reformulate_query", self._reformulate_query)
        workflow.add_node("verify_evidence", self._verify_evidence)
        workflow.add_node("finalize_answer", self._finalize_answer)
        workflow.add_node("stop_insufficient", self._stop_insufficient)
        workflow.add_edge(START, "locate_seed")
        workflow.add_edge("locate_seed", "inspect_local")
        workflow.add_edge("inspect_local", "graph_decide")
        workflow.add_edge("expand_frontier", "inspect_local")
        workflow.add_edge("reformulate_query", "inspect_local")
        workflow.add_edge("verify_evidence", "graph_decide")
        workflow.add_edge("finalize_answer", END)
        workflow.add_edge("stop_insufficient", END)
        return workflow.compile()

    def _initial_state(self, query: str) -> GraphRLMState:
        return {
            "original_query": query,
            "query": query,
            "graph_view": self.graph_view,
            "current_subquery": query,
            "current_owner_id": None,
            "current_document_ids": [],
            "frontier_transition_ids": [],
            "presented_frontier": [],
            "presented_evidence_ids": [],
            "visited_owner_ids": [],
            "visited_document_ids": [],
            "visited_transition_ids": [],
            "collected_evidence_ids": [],
            "path": [],
            "depth": 0,
            "max_depth": self.config.max_graph_depth,
            "remaining_model_calls": self.config.max_graph_model_calls,
            "remaining_expansions": self.config.max_graph_expansions,
            "last_decision": None,
            "model_call_traces": [],
            "final_answer": None,
            "stop_reason": None,
            "current_owner_ids": [],
            "semantic_document_ids": [],
            "frontier_ids": [],
            "graph_paths": [],
            "evidence_span_ids": [],
            "recursion_depth": 0,
            "remaining_calls": self.config.max_graph_model_calls,
        }

    def _locate_seed(self, state: GraphRLMState) -> GraphRLMState:
        query_embedding = self.index.encoder.encode_query(state["current_subquery"])
        results = self.index.search(query_embedding, top_k=self.config.graph_top_k)
        if not results:
            state["stop_reason"] = "no_seed"
            return state
        seed, seed_selection = self._select_seed_with_frontier(results)
        state["current_owner_id"] = seed.owner_id
        state["current_document_ids"] = [seed.semantic_document_id]
        state["visited_owner_ids"] = list(dict.fromkeys([seed.owner_id] + seed.source_entity_ids))
        state["visited_document_ids"] = [seed.semantic_document_id]
        state["collected_evidence_ids"] = list(seed.evidence_span_ids)
        state["current_owner_ids"] = [seed.owner_id]
        state["semantic_document_ids"] = [seed.semantic_document_id]
        state["evidence_span_ids"] = list(seed.evidence_span_ids)
        state["path"] = [
            {
                "step": "locate_seed",
                "document_id": seed.semantic_document_id,
                "owner_id": seed.owner_id,
                "score": seed.score,
                "seed_selection": seed_selection,
            }
        ]
        return state

    def _select_seed_with_frontier(self, results):
        scored = []
        for result in results:
            document = self.index.document_for(result.semantic_document_id)
            frontier_count = len(self.index.frontier_document_ids(
                document,
                cap=self.index.encoder.config.local_frontier_cap,
            ))
            diagnostic_structural_frontier_bonus = min(0.05, frontier_count * 0.01)
            structural_frontier_bonus = 0.0
            owner_type_bonus = 0.08 if result.owner_type == "entity" and frontier_count else 0.0
            final_seed_score = result.score + structural_frontier_bonus + owner_type_bonus
            scored.append(
                {
                    "result": result,
                    "semantic_document_id": result.semantic_document_id,
                    "owner_id": result.owner_id,
                    "owner_type": result.owner_type,
                    "semantic_seed_score": result.score,
                    "structural_frontier_count": frontier_count,
                    "structural_frontier_bonus": structural_frontier_bonus,
                    "diagnostic_structural_frontier_bonus": diagnostic_structural_frontier_bonus,
                    "owner_type_bonus": owner_type_bonus,
                    "final_seed_score": final_seed_score,
                }
            )
        selected = max(
            scored,
            key=lambda item: (
                item["final_seed_score"],
                item["semantic_seed_score"],
                item["semantic_document_id"],
            ),
        )
        return selected["result"], {
            "seed_candidate_count": len(scored),
            "selected_seed_id": selected["semantic_document_id"],
            "selection_reason": "semantic_score_plus_owner_type_bonus_structural_frontier_bonus_disabled",
            "candidates": [
                {key: value for key, value in item.items() if key != "result"}
                for item in scored
            ],
        }

    def _inspect_local(self, state: GraphRLMState) -> GraphRLMState:
        if state.get("stop_reason"):
            return state
        frontier = []
        for document_id in state["current_document_ids"]:
            current = self.index.document_for(document_id)
            frontier_ids = self.index.frontier_document_ids(
                current,
                cap=self.index.encoder.config.local_frontier_cap,
            )
            for target_document_id in sorted(frontier_ids):
                if target_document_id in state["visited_document_ids"]:
                    continue
                target = self.index.document_for(target_document_id)
                transition = _transition(current, target)
                frontier.append(transition)
        raw_frontier_count = len(frontier)
        frontier, omitted_evidence_ids, newly_presented_evidence_ids = _compact_frontier_evidence_refs(
            frontier,
            previously_presented=set(state.get("presented_evidence_ids", [])),
        )
        state["presented_evidence_ids"] = list(
            dict.fromkeys(state.get("presented_evidence_ids", []) + newly_presented_evidence_ids)
        )
        state["presented_frontier"] = frontier
        state["frontier_transition_ids"] = [item["transition_id"] for item in frontier]
        state["frontier_ids"] = list(state["frontier_transition_ids"])
        state["path"].append(
            {
                "step": "inspect_local",
                "current_document_ids": list(state["current_document_ids"]),
                "frontier_transition_ids": list(state["frontier_transition_ids"]),
                "presented_frontier": list(frontier),
                "raw_frontier_count": raw_frontier_count,
                "compacted_frontier_count": len(frontier),
                "omitted_repeated_evidence_span_ids": omitted_evidence_ids,
            }
        )
        return state

    def _graph_decide(self, state: GraphRLMState) -> Command[GraphDestination]:
        if state.get("stop_reason"):
            return Command(goto="stop_insufficient")
        if state["remaining_model_calls"] <= 0:
            state["stop_reason"] = "model_call_budget_exhausted"
            return Command(goto="stop_insufficient", update=state)
        if state["depth"] >= state["max_depth"]:
            state["stop_reason"] = "max_depth_reached"
            return Command(goto="finalize_answer", update=state)

        decision = self.gateway.decide_graph(state)
        self._validate_decision(decision, state)
        state["remaining_model_calls"] -= 1
        state["remaining_calls"] = state["remaining_model_calls"]
        state["last_decision"] = decision.model_dump()
        state["model_call_traces"] = [
            trace.model_dump() for trace in self.gateway.model_call_traces
        ]
        state["path"].append(
            {
                "step": "graph_decide",
                "decision": decision.model_dump(),
                "remaining_model_calls": state["remaining_model_calls"],
            }
        )

        if decision.action == "inspect":
            return Command(update=state, goto="inspect_local")
        if decision.action == "expand":
            return Command(update=state, goto="expand_frontier")
        if decision.action == "reformulate":
            if decision.subquery:
                state["current_subquery"] = decision.subquery
            return Command(update=state, goto="reformulate_query")
        if decision.action == "verify":
            return Command(update=state, goto="verify_evidence")
        if decision.action == "answer":
            return Command(update=state, goto="finalize_answer")
        state["stop_reason"] = "insufficient_evidence"
        return Command(update=state, goto="stop_insufficient")

    def _expand_frontier(self, state: GraphRLMState) -> GraphRLMState:
        decision = GraphRLMDecision.model_validate(state["last_decision"])
        selected_transition_id = decision.selected_transition_ids[0]
        transition = _frontier_by_id(state, selected_transition_id)
        target_document_id = transition["target_document_id"]
        target = self.index.document_for(target_document_id)
        state["current_owner_id"] = transition["target_owner_id"]
        state["current_document_ids"] = [target_document_id]
        state["visited_transition_ids"] = list(
            dict.fromkeys(state["visited_transition_ids"] + [selected_transition_id])
        )
        state["visited_document_ids"] = list(
            dict.fromkeys(state["visited_document_ids"] + [target_document_id])
        )
        state["visited_owner_ids"] = list(
            dict.fromkeys(state["visited_owner_ids"] + [target.owner_id] + target.source_entity_ids)
        )
        state["collected_evidence_ids"] = list(
            dict.fromkeys(state["collected_evidence_ids"] + target.evidence_span_ids)
        )
        state["depth"] += 1
        state["recursion_depth"] = state["depth"]
        state["remaining_expansions"] -= 1
        state["current_owner_ids"] = [state["current_owner_id"]] if state["current_owner_id"] else []
        state["semantic_document_ids"] = list(state["visited_document_ids"])
        state["evidence_span_ids"] = list(state["collected_evidence_ids"])
        state["path"].append(
            {
                "step": "expand_frontier",
                "transition_id": selected_transition_id,
                "source_owner_id": transition["source_owner_id"],
                "source_document_id": transition["source_document_id"],
                "target_owner_id": target.owner_id,
                "target_document_id": target_document_id,
                "depth": state["depth"],
            }
        )
        if state["remaining_expansions"] < 0:
            state["stop_reason"] = "expansion_budget_exhausted"
        return state

    def _reformulate_query(self, state: GraphRLMState) -> GraphRLMState:
        # Reformulation changes the semantic lens for the next decision, but it
        # does not grant permission to jump outside the current structural
        # frontier. Movement still has to happen through expand_frontier.
        self.index.encoder.encode_query(state["current_subquery"])
        state["path"].append(
            {
                "step": "reformulate_query",
                "current_subquery": state["current_subquery"],
                "current_document_ids": list(state["current_document_ids"]),
            }
        )
        return state

    def _verify_evidence(self, state: GraphRLMState) -> GraphRLMState:
        state["path"].append(
            {
                "step": "verify_evidence",
                "evidence_count": len(set(state["collected_evidence_ids"])),
                "depth": state["depth"],
            }
        )
        return state

    def _finalize_answer(self, state: GraphRLMState) -> GraphRLMState:
        decision = state.get("last_decision") or {}
        if isinstance(decision, dict) and decision.get("action") == "answer":
            state["final_answer"] = str(decision.get("decision_summary") or "").strip() or None
        else:
            state["final_answer"] = None
        state["stop_reason"] = state.get("stop_reason") or "answer"
        state["path"].append(
            {
                "step": "finalize_answer",
                "final_answer": state["final_answer"],
                "answer_materialized": state["final_answer"] is not None,
            }
        )
        return state

    def _stop_insufficient(self, state: GraphRLMState) -> GraphRLMState:
        state["stop_reason"] = state.get("stop_reason") or "insufficient_evidence"
        state["path"].append({"step": "stop_insufficient", "stop_reason": state["stop_reason"]})
        return state

    def _validate_decision(self, decision: GraphRLMDecision, state: GraphRLMState) -> None:
        if decision.action == "expand":
            if not decision.selected_transition_ids:
                raise ValueError("expand decision must include selected_transition_ids")
            allowed = set(state["frontier_transition_ids"])
            unknown = set(decision.selected_transition_ids) - allowed
            if unknown:
                raise ValueError(
                    "GraphRLM selected transitions outside presented frontier: "
                    f"{sorted(unknown)}"
                )
        if decision.action == "reformulate" and not decision.subquery:
            raise ValueError("reformulate decision must include subquery")


def _transition(source: GraphSemanticDocument, target: GraphSemanticDocument) -> dict:
    transition_id = _transition_id(source.semantic_document_id, target.semantic_document_id)
    return {
        "transition_id": transition_id,
        "source_document_id": source.semantic_document_id,
        "target_document_id": target.semantic_document_id,
        "source_owner_id": source.owner_id,
        "target_owner_id": target.owner_id,
        "shared_entity_ids": sorted(set(source.source_entity_ids) & set(target.source_entity_ids)),
        "shared_event_ids": sorted(set(source.event_ids) & set(target.event_ids)),
        "evidence_span_ids": list(target.evidence_span_ids),
        "text": target.text,
    }


def _transition_id(source_document_id: str, target_document_id: str) -> str:
    digest = hashlib.sha256(f"{source_document_id}->{target_document_id}".encode("utf-8")).hexdigest()[:12]
    return f"transition_{digest}"


def _compact_frontier_evidence_refs(
    frontier: list[dict],
    *,
    previously_presented: set[str] | None = None,
) -> tuple[list[dict], list[str], list[str]]:
    compacted = []
    seen_evidence_ids: set[str] = set(previously_presented or set())
    omitted_evidence_ids: set[str] = set()
    newly_presented_evidence_ids: list[str] = []
    for transition in frontier:
        transition = dict(transition)
        evidence_ids = set(transition.get("evidence_span_ids") or [])
        repeated = evidence_ids & seen_evidence_ids
        omitted_evidence_ids.update(repeated)
        visible_evidence_ids = [
            evidence_id
            for evidence_id in transition.get("evidence_span_ids", [])
            if evidence_id not in repeated
        ]
        transition["evidence_span_ids"] = visible_evidence_ids
        newly_presented_evidence_ids.extend(visible_evidence_ids)
        if repeated:
            transition["omitted_repeated_evidence_span_count"] = len(repeated)
        compacted.append(transition)
        seen_evidence_ids.update(evidence_ids)
    return compacted, sorted(omitted_evidence_ids), list(dict.fromkeys(newly_presented_evidence_ids))


def _frontier_by_id(state: GraphRLMState, transition_id: str) -> dict:
    for transition in state["presented_frontier"]:
        if transition["transition_id"] == transition_id:
            return transition
    raise ValueError(f"transition_id={transition_id} is not in presented_frontier")


def _confidence(state: GraphRLMState) -> float:
    decision = state.get("last_decision")
    if isinstance(decision, dict):
        return float(decision.get("confidence", 0.0))
    return 0.0
