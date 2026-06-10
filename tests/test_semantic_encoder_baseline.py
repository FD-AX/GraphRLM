from __future__ import annotations

from hashlib import sha256

from app.semantic_encoding.encoder import cosine_similarity

from app.semantic_encoding import (
    EncoderConfig,
    FrontierScoringWeights,
    GraphSemanticDocument,
    GraphSemanticEncoder,
    GraphSemanticIndex,
    HashingSemanticEncoder,
    LatentGraphNavigator,
    SemanticTraversalCase,
    evaluate_active_profile_clusters,
    evaluate_navigation,
    evaluate_seed_retrieval,
)


def make_document(document_id: str, owner_id: str, text: str) -> GraphSemanticDocument:
    return GraphSemanticDocument(
        document_id="doc",
        semantic_document_id=document_id,
        owner_type="entity_event",
        owner_id=owner_id,
        source_entity_ids=[owner_id],
        event_ids=[f"ev_{document_id}"],
        evidence_span_ids=[f"span_{document_id}"],
        source_chunk_ids=[f"chunk_{document_id}"],
        text=text,
        structural_features={"predicate": text.split()[1] if len(text.split()) > 1 else text},
        projection_version="projection_v1",
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
    )


def test_hashing_backend_is_explicit_fast_test_encoder() -> None:
    config = EncoderConfig(projection_dim=128, interaction_dim=32)
    encoder = GraphSemanticEncoder(
        config=config,
        backend=HashingSemanticEncoder(dimensions=128),
    )
    document = make_document("forest", "semyon", "Semyon entered the forest.")
    embedding = encoder.encode_document(document)
    query = encoder.encode_query("forest")
    profile = encoder.interaction_profile(query, embedding.embedding)

    assert embedding.encoder_name == "hashing-bert-like"
    assert embedding.embedding_dim == 128
    assert embedding.actual_embedding_dim == 128
    assert embedding.serializer_version == "graph_semantic_document_serializer_v1"
    assert embedding.query_prefix == "query:"
    assert embedding.document_prefix == "document:"
    assert len(query) == 128
    assert len(profile) == 128


def test_contribution_profile_keeps_native_dimensions_and_is_query_conditioned() -> None:
    backend = HashingSemanticEncoder(dimensions=128)
    contribution_encoder = GraphSemanticEncoder(
        config=EncoderConfig(
            projection_dim=128,
            interaction_dim=32,
            interaction_profile_mode="contribution",
        ),
        backend=backend,
    )
    hashed_encoder = GraphSemanticEncoder(
        config=EncoderConfig(
            projection_dim=128,
            interaction_dim=32,
            interaction_profile_mode="hashed_features",
        ),
        backend=backend,
    )
    document = make_document("forest", "semyon", "Semyon walked along the forest path")
    embedding = contribution_encoder.encode_document(document).embedding
    forest_query = contribution_encoder.encode_query("forest path")
    door_query = contribution_encoder.encode_query("door repair")

    forest_profile = contribution_encoder.interaction_profile(forest_query, embedding)
    door_profile = contribution_encoder.interaction_profile(door_query, embedding)
    hashed_profile = hashed_encoder.interaction_profile(forest_query, embedding)

    assert len(forest_profile) == 128
    assert len(hashed_profile) == 32
    assert abs(cosine_similarity(forest_profile, forest_profile) - 1.0) < 1e-9
    assert cosine_similarity(forest_profile, door_profile) < 0.99


def test_seed_retrieval_evaluation_reports_rank_metrics() -> None:
    docs = [
        make_document("forest", "semyon", "Semyon entered the forest."),
        make_document("door", "ivan", "Ivan repaired the door."),
    ]
    encoder = GraphSemanticEncoder(EncoderConfig(projection_dim=64))
    index = GraphSemanticIndex.build(docs, encoder)
    result = evaluate_seed_retrieval(
        index,
        encoder,
        [
            SemanticTraversalCase(
                query="forest",
                expected_document_ids=["forest"],
                expected_seed_owner_ids=["semyon"],
            )
        ],
        top_k=2,
    )

    assert result.case_results[0].top_results
    assert 0.0 <= result.mean_recall_at_k <= 1.0
    assert 0.0 <= result.mean_mrr <= 1.0
    assert 0.0 <= result.mean_ndcg_at_k <= 1.0


def test_active_profile_is_logged_but_not_weighted_by_default() -> None:
    docs = [
        make_document("forest", "semyon", "Semyon entered the forest."),
        make_document("warning", "semyon", "Semyon ignored the warning."),
    ]
    config = EncoderConfig(
        projection_dim=64,
        interaction_dim=16,
        max_graph_depth=1,
        beam_width=1,
        active_profile_weight=0.0,
    )
    encoder = GraphSemanticEncoder(config)
    trace = LatentGraphNavigator(
        GraphSemanticIndex.build(docs, encoder),
        encoder,
        config,
    ).traverse("warning", seed_top_k=1, max_depth=1, beam_width=1)

    scored_steps = [step for step in trace.steps if step.score_parts]
    assert scored_steps
    assert any("active_profile_similarity" in step.score_parts for step in scored_steps)
    assert all(
        step.score_parts.get("active_profile_weight") in (None, 0.0)
        for step in scored_steps
    )


def test_navigation_evaluation_scores_next_hop_and_clusters_profiles() -> None:
    docs = [
        make_document("seed", "semyon", "Semyon. Alias: hunter."),
        make_document("anna", "anna", "Semyon protected Anna."),
        make_document("forest", "forest", "Semyon entered the forest."),
    ]
    docs[0].owner_type = "entity"
    docs[1].source_entity_ids = ["semyon", "anna"]
    docs[2].source_entity_ids = ["semyon", "forest"]
    encoder = GraphSemanticEncoder(EncoderConfig(projection_dim=64, local_frontier_cap=10))
    index = GraphSemanticIndex.build(docs, encoder)
    cases = [
        (
            "care",
            SemanticTraversalCase(
                query="protected",
                current_owner_id="semyon",
                expected_document_ids=["anna"],
                relevant_next_owner_ids=["anna"],
                expected_evidence_span_ids=["span_anna"],
            ),
        ),
        (
            "place",
            SemanticTraversalCase(
                query="forest",
                current_owner_id="semyon",
                expected_document_ids=["forest"],
                relevant_next_owner_ids=["forest"],
                expected_evidence_span_ids=["span_forest"],
            ),
        ),
    ]

    result = evaluate_navigation(
        index,
        encoder,
        [case for _, case in cases],
        FrontierScoringWeights(name="query_only", query_similarity_weight=1.0),
        top_k=2,
    )
    clusters = evaluate_active_profile_clusters(index, encoder, cases)

    assert result.case_results
    assert 0.0 <= result.mean_next_hop_accuracy <= 1.0
    assert 0.0 <= result.mean_evidence_recall_at_k <= 1.0
    assert clusters.pairs
