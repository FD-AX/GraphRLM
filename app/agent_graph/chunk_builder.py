from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agent_graph.chunk_nodes import (
    extract_graph_with_llm,
    normalize_graph_patch,
    validate_llm_extraction,
)
from app.agent_graph.chunk_state import ChunkGraphState


def build_chunk_graph_workflow(model_adapter):
    workflow = StateGraph(ChunkGraphState)

    workflow.add_node(
        "extract_graph_with_llm",
        partial(
            extract_graph_with_llm,
            model_adapter=model_adapter,
        ),
    )
    workflow.add_node("validate_llm_extraction", validate_llm_extraction)
    workflow.add_node("normalize_graph_patch", normalize_graph_patch)

    workflow.add_edge(START, "extract_graph_with_llm")
    workflow.add_edge("extract_graph_with_llm", "validate_llm_extraction")
    workflow.add_edge("validate_llm_extraction", "normalize_graph_patch")
    workflow.add_edge("normalize_graph_patch", END)

    return workflow.compile()
