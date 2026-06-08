from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.dual_rlm.arbiter import EvidenceArbiter
from app.dual_rlm.models import (
    AnswerArbitrationResult,
    GraphViewRef,
    RetrievalArm,
    RetrievalArmResult,
)


class DualRLMGraphState(TypedDict):
    run_id: str
    query: str
    graph_view: GraphViewRef
    graph_result: RetrievalArmResult | None
    text_result: RetrievalArmResult | None
    arbitration: AnswerArbitrationResult | None


def build_dual_rlm_graph(
    graph_arm: RetrievalArm,
    text_arm: RetrievalArm,
    arbiter: EvidenceArbiter | None = None,
):
    evidence_arbiter = arbiter or EvidenceArbiter()

    def run_graph_arm(state: DualRLMGraphState) -> DualRLMGraphState:
        state["graph_result"] = graph_arm.run(state["query"], state["run_id"])
        return state

    def run_text_arm(state: DualRLMGraphState) -> DualRLMGraphState:
        state["text_result"] = text_arm.run(state["query"], state["run_id"])
        return state

    def compare_results(state: DualRLMGraphState) -> DualRLMGraphState:
        graph_result = state["graph_result"]
        text_result = state["text_result"]
        if graph_result is None or text_result is None:
            raise ValueError("Both graph_result and text_result are required before arbitration.")
        state["arbitration"] = evidence_arbiter.arbitrate(
            state["query"],
            graph_result,
            text_result,
        )
        return state

    workflow = StateGraph(DualRLMGraphState)
    workflow.add_node("graph_rlm", run_graph_arm)
    workflow.add_node("text_rlm", run_text_arm)
    workflow.add_node("compare_results", compare_results)
    workflow.add_edge(START, "graph_rlm")
    workflow.add_edge("graph_rlm", "text_rlm")
    workflow.add_edge("text_rlm", "compare_results")
    workflow.add_edge("compare_results", END)
    return workflow.compile()
