from app.agent_graph.document_state import DocumentGraphState


def should_continue(state: DocumentGraphState) -> str:
    if state.get("done"):
        return "end"

    return "process_chunk"
