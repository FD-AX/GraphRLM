from app.retrieval import (
    EntityContext,
    GraphTraversalCandidate,
    LocateInspectExpandRuntime,
    Observation,
    PathState,
    SemanticGraphStore,
)


def test_locate_inspect_expand_retrieves_query_conditioned_subgraph() -> None:
    store = SemanticGraphStore(
        entities=[
            EntityContext(
                entity_id="semyon",
                canonical_name="Semyon",
                descriptions=[
                    "Semyon is tied to Anna through love, jealousy, and conflict.",
                ],
                evidence_span_ids=["span_entity_semyon"],
            ),
            EntityContext(
                entity_id="old_man",
                canonical_name="Old man",
                descriptions=["Old man appears in forest warnings and fear scenes."],
                evidence_span_ids=["span_entity_old_man"],
            ),
            EntityContext(
                entity_id="anna",
                canonical_name="Anna",
                descriptions=["Anna is part of Semyon's love relationship."],
                evidence_span_ids=["span_entity_anna"],
            ),
        ],
        observations=[
            Observation(
                observation_id="obs_love",
                entity_id="semyon",
                text="Semyon protected Anna and their love relationship grew tense.",
                evidence_span_ids=["span_love"],
            ),
            Observation(
                observation_id="obs_forest",
                entity_id="semyon",
                text="Semyon heard a warning from the old man in the forest.",
                evidence_span_ids=["span_forest"],
            ),
        ],
        transitions=[
            GraphTraversalCandidate(
                source_entity_id="semyon",
                target_entity_id="anna",
                relation_ids=["rel_love"],
                evidence_span_ids=["span_love"],
                local_context=(
                    "Semyon and Anna share love, jealousy, protection, and conflict."
                ),
                depth=1,
                structural_confidence=0.92,
            ),
            GraphTraversalCandidate(
                source_entity_id="semyon",
                target_entity_id="old_man",
                relation_ids=["rel_warning"],
                evidence_span_ids=["span_warning"],
                local_context="Semyon met the old man near the forest warning.",
                depth=1,
                structural_confidence=0.92,
            ),
        ],
    )

    runtime = LocateInspectExpandRuntime(store)
    result = runtime.retrieve(
        query="Semyon Anna love relationship conflict",
        seed_top_k=2,
        observation_top_k=2,
        beam_width=2,
        max_depth=1,
    )

    assert any(seed.item_id == "semyon" for seed in result.seed_entities)
    assert result.observations[0].item_id == "obs_love"
    assert result.paths[0].entity_ids == ["semyon", "anna"]
    assert "span_love" in result.evidence_span_ids


def test_transition_scoring_uses_path_context() -> None:
    store = SemanticGraphStore(
        entities=[],
        observations=[],
        transitions=[],
    )
    runtime = LocateInspectExpandRuntime(store)
    query_vector = runtime.encoder.encode_text(
        "love relationship caused conflict with Ivan"
    )

    relevant = GraphTraversalCandidate(
        source_entity_id="anna",
        target_entity_id="ivan",
        local_context="Ivan pursued Anna and this continued the conflict.",
        path_context="Semyon and Anna had a love relationship with jealousy.",
        depth=2,
        structural_confidence=0.9,
    )
    irrelevant = GraphTraversalCandidate(
        source_entity_id="anna",
        target_entity_id="forest",
        local_context="Anna walked through the forest at night.",
        path_context="Semyon and Anna had a love relationship with jealousy.",
        depth=2,
        structural_confidence=0.9,
    )

    relevant_score = runtime.score_transition(
        query="love relationship caused conflict with Ivan",
        query_vector=query_vector,
        candidate=relevant,
    )
    irrelevant_score = runtime.score_transition(
        query="love relationship caused conflict with Ivan",
        query_vector=query_vector,
        candidate=irrelevant,
    )

    assert relevant_score.score > irrelevant_score.score
    assert relevant_score.score_parts["path_continuity"] > 0


def test_transition_scoring_penalizes_repeated_evidence() -> None:
    store = SemanticGraphStore(entities=[], observations=[], transitions=[])
    runtime = LocateInspectExpandRuntime(store)
    query_vector = runtime.encoder.encode_text("love conflict evidence")

    base = GraphTraversalCandidate(
        source_entity_id="semyon",
        target_entity_id="anna",
        local_context="Semyon and Anna had love and conflict.",
        evidence_span_ids=["span_love"],
        depth=1,
    )
    base_state = runtime.extend_path_state(
        query_vector=query_vector,
        path_state=PathState(
            path_embedding=query_vector,
            covered_entities=["semyon", "anna"],
            accumulated_evidence=["span_love"],
        ),
        transition=base,
    )

    repeated = GraphTraversalCandidate(
        source_entity_id="anna",
        target_entity_id="ivan",
        local_context="Anna and Ivan continued the love conflict.",
        evidence_span_ids=["span_love"],
        depth=2,
    )
    novel = repeated.model_copy(update={"evidence_span_ids": ["span_ivan"]})

    repeated_score = runtime.score_transition(
        query="love conflict evidence",
        query_vector=query_vector,
        candidate=repeated,
        path_state=base_state,
    )
    novel_score = runtime.score_transition(
        query="love conflict evidence",
        query_vector=query_vector,
        candidate=novel,
        path_state=base_state,
    )

    assert novel_score.score > repeated_score.score
    assert repeated_score.score_parts["redundancy"] > 0
