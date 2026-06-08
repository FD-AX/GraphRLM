import hashlib

from app.core.graph_models import LocalGraphPatch
from app.rlm.state import EntityState, RLMState, RLMTransition


def stable_id(prefix: str, raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def build_rlm_transition_from_patch(
    state: RLMState,
    patch: LocalGraphPatch,
) -> RLMTransition:
    chunk = patch.chunk

    added_entities: list[EntityState] = []
    updated_entities: list[EntityState] = []

    previous_index = state.current_chunk_index

    for entity in patch.entities:
        evidence_ref = chunk.chunk_id

        if entity.entity_id not in state.entities:
            entity_state = EntityState(
                entity_id=entity.entity_id,
                canonical_name=entity.canonical_name,
                attributes=[],
                hypotheses=[],
                evidence_refs=[evidence_ref],
                confidence=0.5,
            )

            if entity.description:
                entity_state.attributes.append(entity.description)

            added_entities.append(entity_state)
        else:
            old = state.entities[entity.entity_id]

            entity_state = EntityState(
                entity_id=old.entity_id,
                canonical_name=old.canonical_name,
                attributes=list(old.attributes),
                hypotheses=list(old.hypotheses),
                evidence_refs=list(old.evidence_refs),
                confidence=old.confidence,
            )

            if evidence_ref not in entity_state.evidence_refs:
                entity_state.evidence_refs.append(evidence_ref)

            if entity.description and entity.description not in entity_state.attributes:
                entity_state.attributes.append(entity.description)

            entity_state.confidence = max(entity_state.confidence, 0.6)
            updated_entities.append(entity_state)

    return RLMTransition(
        transition_id=stable_id(
            "rlm_transition",
            f"{state.document_id}:{previous_index}->{chunk.index}:{chunk.chunk_id}",
        ),
        document_id=state.document_id,
        from_chunk_id=None,
        to_chunk_id=chunk.chunk_id,
        from_chunk_index=previous_index if previous_index >= 0 else None,
        to_chunk_index=chunk.index,
        added_entities=added_entities,
        updated_entities=updated_entities,
        added_relations=patch.relations,
        added_relation_candidates=patch.relation_candidates,
        added_evidence_spans=patch.evidence_spans,
        notes=[
            f"Applied graph patch from chunk {chunk.index}",
        ],
    )


def apply_rlm_transition(
    state: RLMState,
    transition: RLMTransition,
) -> RLMState:
    for entity in transition.added_entities:
        state.entities[entity.entity_id] = entity

    for entity in transition.updated_entities:
        old = state.entities.get(entity.entity_id)

        if old is None:
            state.entities[entity.entity_id] = entity
            continue

        old.attributes = sorted(set(old.attributes + entity.attributes))
        old.hypotheses = sorted(set(old.hypotheses + entity.hypotheses))
        old.evidence_refs = sorted(set(old.evidence_refs + entity.evidence_refs))
        old.confidence = max(old.confidence, entity.confidence)

    state.current_chunk_index = transition.to_chunk_index
    state.recent_chunk_ids.append(transition.to_chunk_id)
    state.recent_chunk_ids = state.recent_chunk_ids[-5:]

    state.recent_evidence_spans.extend(transition.added_evidence_spans)
    state.recent_evidence_spans = state.recent_evidence_spans[-20:]

    for candidate in transition.added_relation_candidates:
        state.relation_candidates[candidate.relation_candidate_id] = candidate

    return state
