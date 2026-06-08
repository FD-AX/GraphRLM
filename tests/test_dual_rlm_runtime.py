from __future__ import annotations

from hashlib import sha256

from app.dual_rlm import (
    DualRLMConfig,
    GraphRLMArm,
    GraphViewRef,
    ImmutableTextStore,
    IndependentDualRLMRuntime,
    SourceChunk,
    TextRLMArm,
    build_dual_rlm_graph,
)
from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticDocument,
    GraphSemanticEncoder,
    GraphSemanticIndex,
)


def make_document(document_id: str, owner_id: str, text: str) -> GraphSemanticDocument:
    return GraphSemanticDocument(
        document_id="doc",
        semantic_document_id=document_id,
        owner_type="pair_interaction",
        owner_id=owner_id,
        source_entity_ids=owner_id.split(":"),
        event_ids=[f"ev_{document_id}"],
        evidence_span_ids=[f"span_{document_id}"],
        source_chunk_ids=["chunk_1"],
        text=text,
        structural_features={"kind": "test"},
        projection_version="projection_v1",
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
    )


def make_runtime():
    encoder = GraphSemanticEncoder(EncoderConfig(projection_dim=64))
    index = GraphSemanticIndex.build(
        [
            make_document(
                "graph_anna",
                "entity_semyon:entity_anna",
                "Semyon protected Anna and Anna was jealous.",
            )
        ],
        encoder,
    )
    graph_view = GraphViewRef(
        document_id="doc",
        graph_version="graph_v1",
        projection_version="projection_v1",
        encoder_version=encoder.backend.encoder_version,
    )
    text_store = ImmutableTextStore(
        [
            SourceChunk(
                document_id="doc",
                chunk_id="chunk_1",
                text="Semyon protected Anna from danger.",
                start_char=0,
                end_char=35,
            ),
            SourceChunk(
                document_id="doc",
                chunk_id="chunk_2",
                text="Ivan confronted Semyon later.",
                start_char=36,
                end_char=66,
            ),
        ]
    )
    graph_arm = GraphRLMArm(index, graph_view, DualRLMConfig(graph_top_k=2))
    text_arm = TextRLMArm(text_store, DualRLMConfig(text_top_k=2))
    return graph_view, graph_arm, text_arm


def test_dual_runtime_returns_independent_arm_results_without_graph_mutation() -> None:
    graph_view, graph_arm, text_arm = make_runtime()
    result = IndependentDualRLMRuntime(graph_arm, text_arm).run(
        "Semyon protected Anna",
        graph_view=graph_view,
        run_id="run",
    )

    assert result.graph_result.arm == "graph_rlm"
    assert result.text_result.arm == "text_rlm"
    assert result.graph_result.consulted_object_ids
    assert result.text_result.consulted_object_ids
    assert result.graph_mutation_allowed is False
    assert result.arbitration.support_status in {
        "both_supported",
        "graph_supported_text_unconfirmed",
        "text_supported_graph_missing",
    }


def test_text_supported_graph_missing_does_not_change_graph_result() -> None:
    graph_view, graph_arm, text_arm = make_runtime()
    result = IndependentDualRLMRuntime(graph_arm, text_arm).run(
        "Ivan confronted Semyon later",
        graph_view=graph_view,
        run_id="run",
    )

    assert result.text_result.evidence_span_ids
    assert result.graph_mutation_allowed is False
    assert all(
        "chunk_" not in consulted_id
        for consulted_id in result.graph_result.consulted_object_ids
    )


def test_langgraph_dual_rlm_wrapper_runs_compare_step() -> None:
    graph_view, graph_arm, text_arm = make_runtime()
    app = build_dual_rlm_graph(graph_arm, text_arm)
    state = app.invoke(
        {
            "run_id": "run",
            "query": "Semyon protected Anna",
            "graph_view": graph_view,
            "graph_result": None,
            "text_result": None,
            "arbitration": None,
        }
    )

    assert state["graph_result"].arm == "graph_rlm"
    assert state["text_result"].arm == "text_rlm"
    assert state["arbitration"].selected_arm in {"hybrid", "graph_rlm", "text_rlm", "none"}
