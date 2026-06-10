from __future__ import annotations

from datetime import datetime

from app.projection.models import (
    EntityContextSnapshot,
    EntityEncodingInput,
    EntityPairContextSnapshot,
    EntityPairEncodingInput,
    PairEventLink,
    ProjectionEventObservation,
)
from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticEncoder,
    GraphSemanticIndex,
    LatentGraphNavigator,
    build_graph_semantic_documents,
    encode_graph_semantic_documents,
)


def make_snapshots():
    entity_snapshot = EntityContextSnapshot(
        snapshot_id="snap_semyon",
        entity_id="semyon",
        document_id="doc",
        canonical_name="Semyon",
        alias_surfaces=["Semyon", "hunter", "He"],
        event_observations=[
            ProjectionEventObservation(
                event_id="ev_identity",
                predicate="hunt",
                entity_role="agent",
                surface_text="hunter",
                evidence_span_ids=["span_hunter"],
                source_chunk_ids=["chunk_1"],
                event_resolution_status="complete",
                event_materialization_status="valid",
            ),
            ProjectionEventObservation(
                event_id="ev_hear",
                predicate="hear",
                entity_role="experiencer",
                surface_text="He",
                unresolved_counterparts=["steps", "door"],
                evidence_span_ids=["span_steps"],
                source_chunk_ids=["chunk_3"],
                event_resolution_status="partial",
                event_materialization_status="valid",
            ),
            ProjectionEventObservation(
                event_id="ev_warned",
                predicate="warned",
                entity_role="patient",
                surface_text="hunter",
                counterpart_entity_ids=["old_man"],
                evidence_span_ids=["span_warning"],
                source_chunk_ids=["chunk_2"],
                event_resolution_status="complete",
                event_materialization_status="valid",
            ),
        ],
        related_entity_ids=["old_man"],
        unresolved_counterparts=["steps", "door"],
        evidence_span_ids=["span_hunter", "span_steps", "span_warning"],
        source_chunk_ids=["chunk_1", "chunk_2", "chunk_3"],
        projection_version="projection_v1",
        extractor_versions=["gemma"],
        resolver_versions=["resolver"],
        encoding_input=EntityEncodingInput(
            entity_id="semyon",
            document_id="doc",
            projection_version="projection_v1",
        ),
        content_hash="hash_entity",
        created_at=datetime.utcnow(),
    )
    pair_snapshot = EntityPairContextSnapshot(
        pair_id="pair_semyon_old_man",
        snapshot_id="snap_pair",
        document_id="doc",
        source_entity_id="old_man",
        target_entity_id="semyon",
        target_to_source_events=[
            PairEventLink(
                event_id="ev_warned",
                predicate="warned",
                source_role="agent",
                target_role="patient",
                direction="source_to_target",
                evidence_span_ids=["span_warning"],
                source_chunk_ids=["chunk_2"],
                surface_texts=["old man", "hunter"],
            )
        ],
        relation_evidence_span_ids=["span_warning"],
        source_roles=["agent"],
        target_roles=["patient"],
        projection_version="projection_v1",
        encoding_input=EntityPairEncodingInput(
            pair_id="pair_semyon_old_man",
            document_id="doc",
            projection_version="projection_v1",
        ),
        content_hash="hash_pair",
        created_at=datetime.utcnow(),
    )
    return [entity_snapshot], [pair_snapshot]


def test_builds_local_graph_semantic_documents() -> None:
    entity_snapshots, pair_snapshots = make_snapshots()
    documents = build_graph_semantic_documents(entity_snapshots, pair_snapshots)

    owner_types = {document.owner_type for document in documents}
    assert "entity" in owner_types
    assert "entity_event" in owner_types
    assert "evidence" in owner_types
    assert "entity_pair" in owner_types
    assert "pair_interaction" in owner_types
    assert any("steps" in document.text and "door" in document.text for document in documents)


def test_encoder_outputs_256_space_and_mode_dependent_interaction_profile() -> None:
    documents = build_graph_semantic_documents(*make_snapshots())
    config = EncoderConfig(projection_dim=256, interaction_dim=64)
    encoder, embeddings, _ = encode_graph_semantic_documents(documents, config=config)
    hashed_config = EncoderConfig(
        projection_dim=256,
        interaction_dim=64,
        interaction_profile_mode="hashed_features",
    )
    hashed_encoder, _, _ = encode_graph_semantic_documents(documents, config=hashed_config)
    query = encoder.encode_query("Who heard steps behind the door?")
    profile = encoder.interaction_profile(query, embeddings[0].embedding)
    hashed_profile = hashed_encoder.interaction_profile(query, embeddings[0].embedding)

    assert len(query) == 256
    assert all(len(embedding.embedding) == 256 for embedding in embeddings)
    assert len(profile) == 256
    assert len(hashed_profile) == 64
    assert encoder.top_contribution_components(query, embeddings[0].embedding)


def test_search_finds_local_event_document_not_only_entity_profile() -> None:
    documents = build_graph_semantic_documents(*make_snapshots())
    config = EncoderConfig(projection_dim=256, interaction_dim=64)
    encoder = GraphSemanticEncoder(config)
    index = GraphSemanticIndex.build(documents, encoder)
    query = encoder.encode_query("heard steps behind the door")

    results = index.search(query, top_k=5)

    assert any(result.owner_type == "entity_event" for result in results)
    assert any("steps" in result.text for result in results)
    assert all(result.contribution_top_components for result in results)


def test_latent_navigation_trace_expands_structural_frontier() -> None:
    documents = build_graph_semantic_documents(*make_snapshots())
    config = EncoderConfig(
        projection_dim=256,
        interaction_dim=64,
        max_graph_depth=2,
        beam_width=2,
        local_frontier_cap=10,
    )
    encoder = GraphSemanticEncoder(config)
    index = GraphSemanticIndex.build(documents, encoder)
    trace = LatentGraphNavigator(index, encoder, config).traverse(
        "warning between old man and Semyon",
        seed_top_k=4,
        max_depth=2,
        beam_width=2,
    )

    assert trace.seed_results
    assert trace.steps
    assert trace.evidence_span_ids
    assert any(step.active_profile_similarity != 0 for step in trace.steps)
    assert any(step.contribution_top_components for step in trace.steps)
