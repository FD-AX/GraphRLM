from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agent_graph.chunk_builder import build_chunk_graph_workflow
from app.agent_graph.document_edges import should_continue
from app.agent_graph.document_nodes import (
    advance_chunk,
    run_chunk_graph_workflow,
    select_next_chunk,
    update_rlm_state_with_llm,
    write_graph_and_transition,
)
from app.agent_graph.document_state import DocumentGraphState


def build_document_graph_workflow(
    model_adapter,
    graph_writer,
    rlm_model_adapter=None,
    use_llm_rlm_update: bool = True,
):
    rlm_model_adapter = rlm_model_adapter or model_adapter
    chunk_graph = build_chunk_graph_workflow(
        model_adapter=model_adapter,
    )

    workflow = StateGraph(DocumentGraphState)

    workflow.add_node("select_next_chunk", select_next_chunk)
    workflow.add_node(
        "run_chunk_graph_workflow",
        partial(
            run_chunk_graph_workflow,
            chunk_graph=chunk_graph,
        ),
    )
    if use_llm_rlm_update:
        workflow.add_node(
            "update_rlm_state",
            partial(
                update_rlm_state_with_llm,
                rlm_model_adapter=rlm_model_adapter,
            ),
        )
    else:
        from app.agent_graph.document_nodes import update_rlm_state_from_patch

        workflow.add_node("update_rlm_state", update_rlm_state_from_patch)
    workflow.add_node(
        "write_graph_and_transition",
        partial(
            write_graph_and_transition,
            graph_writer=graph_writer,
        ),
    )
    workflow.add_node("advance_chunk", advance_chunk)

    workflow.add_edge(START, "select_next_chunk")
    workflow.add_conditional_edges(
        "select_next_chunk",
        should_continue,
        {
            "process_chunk": "run_chunk_graph_workflow",
            "end": END,
        },
    )
    workflow.add_edge("run_chunk_graph_workflow", "update_rlm_state")
    workflow.add_edge("update_rlm_state", "write_graph_and_transition")
    workflow.add_edge("write_graph_and_transition", "advance_chunk")
    workflow.add_edge("advance_chunk", "select_next_chunk")

    return workflow.compile()
