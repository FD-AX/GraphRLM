from app.agent_graph.document_state import DocumentGraphState
from app.llm.prompts import RLM_UPDATE_SYSTEM_PROMPT
from app.rlm.merge import apply_rlm_transition, build_rlm_transition_from_patch
from app.rlm.schemas import RLMTransitionExtraction
from app.rlm.state import EntityState, RLMTransition


def select_next_chunk(state: DocumentGraphState) -> dict:
    idx = state["current_chunk_index"]

    if idx >= len(state["chunks"]):
        return {
            "done": True,
        }

    return {
        "current_chunk": state["chunks"][idx],
        "done": False,
    }


async def run_chunk_graph_workflow(
    state: DocumentGraphState,
    chunk_graph,
) -> dict:
    result = await chunk_graph.ainvoke(
        {
            "chunk": state["current_chunk"],
            "rlm_state": state["rlm_state"],
            "errors": [],
        }
    )

    errors = state.get("errors", []) + result.get("errors", [])

    return {
        "graph_patch": result["graph_patch"],
        "errors": errors,
    }


async def update_rlm_state_with_llm(
    state: DocumentGraphState,
    rlm_model_adapter,
) -> dict:
    fallback_transition = build_rlm_transition_from_patch(
        state=state["rlm_state"],
        patch=state["graph_patch"],
    )

    try:
        extraction = await rlm_model_adapter.structured_call(
            system_prompt=RLM_UPDATE_SYSTEM_PROMPT,
            user_payload={
                "current_chunk": state["current_chunk"].model_dump(),
                "previous_rlm_state": state["rlm_state"].model_dump(),
                "graph_patch": state["graph_patch"].model_dump(),
                "fallback_transition": fallback_transition.model_dump(),
                "task": (
                    "Update the document-level RLM state using the graph patch. "
                    "Use the fallback_transition as a conservative baseline, "
                    "but improve continuity and evidence attribution when the "
                    "previous RLM state provides useful context."
                ),
            },
            output_schema=RLMTransitionExtraction,
        )

        added_entities = [
            EntityState(**entity.model_dump())
            for entity in extraction.added_entities
        ]
        updated_entities = [
            EntityState(**entity.model_dump())
            for entity in extraction.updated_entities
        ]

        if not added_entities and not updated_entities:
            added_entities = fallback_transition.added_entities
            updated_entities = fallback_transition.updated_entities
            extraction.notes.append(
                "Gemma RLM returned no entity state changes; used conservative entity continuity fallback."
            )

        transition = RLMTransition(
            transition_id=fallback_transition.transition_id,
            document_id=fallback_transition.document_id,
            from_chunk_id=fallback_transition.from_chunk_id,
            to_chunk_id=fallback_transition.to_chunk_id,
            from_chunk_index=fallback_transition.from_chunk_index,
            to_chunk_index=fallback_transition.to_chunk_index,
            added_entities=added_entities,
            updated_entities=updated_entities,
            added_relations=fallback_transition.added_relations,
            added_relation_candidates=fallback_transition.added_relation_candidates,
            added_evidence_spans=fallback_transition.added_evidence_spans,
            notes=["Gemma RLM update applied."] + extraction.notes,
        )
        state["rlm_state"].open_hypotheses = extraction.open_hypotheses
        state["rlm_state"].unresolved_references = extraction.unresolved_references
    except Exception as exc:
        transition = fallback_transition
        transition.notes.append(
            f"Gemma RLM update failed; deterministic fallback used: {exc}"
        )

    new_state = apply_rlm_transition(
        state=state["rlm_state"],
        transition=transition,
    )

    return {
        "rlm_transition": transition,
        "rlm_state": new_state,
    }


def update_rlm_state_from_patch(state: DocumentGraphState) -> dict:
    transition = build_rlm_transition_from_patch(
        state=state["rlm_state"],
        patch=state["graph_patch"],
    )

    new_state = apply_rlm_transition(
        state=state["rlm_state"],
        transition=transition,
    )

    return {
        "rlm_transition": transition,
        "rlm_state": new_state,
    }


def write_graph_and_transition(state: DocumentGraphState, graph_writer) -> dict:
    graph_writer.write_local_graph(state["graph_patch"])
    graph_writer.write_rlm_transition(state["rlm_transition"])

    return {}


def advance_chunk(state: DocumentGraphState) -> dict:
    return {
        "current_chunk_index": state["current_chunk_index"] + 1,
    }
