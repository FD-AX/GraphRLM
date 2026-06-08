from __future__ import annotations

from hashlib import sha256

import pytest

from app.dual_rlm import (
    DualRLMConfig,
    DynamicGraphRLMArm,
    DynamicTextRLMArm,
    GraphRLMDecision,
    GraphViewRef,
    ImmutableTextStore,
    IndependentDualRLMRuntime,
    ScriptedRLMGateway,
    SourceChunk,
    TextRLMResult,
)
from app.dual_rlm.gateway import _graph_prompt_sections
from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticDocument,
    GraphSemanticEncoder,
    GraphSemanticIndex,
)


def make_doc(
    document_id: str,
    owner_id: str,
    source_entity_ids: list[str],
    text: str,
) -> GraphSemanticDocument:
    return GraphSemanticDocument(
        document_id="doc",
        semantic_document_id=document_id,
        owner_type="entity" if document_id == "doc_semyon" else "pair_interaction",
        owner_id=owner_id,
        source_entity_ids=source_entity_ids,
        event_ids=[f"ev_{document_id}"],
        evidence_span_ids=[f"span_{document_id}"],
        source_chunk_ids=["chunk_1"],
        text=text,
        structural_features={"kind": "test"},
        projection_version="projection_v1",
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
    )


def make_dynamic_arm(gateway: ScriptedRLMGateway) -> DynamicGraphRLMArm:
    encoder = GraphSemanticEncoder(EncoderConfig(projection_dim=64, local_frontier_cap=10))
    index = GraphSemanticIndex.build(
        [
            make_doc(
                "doc_semyon",
                "entity_semyon",
                ["entity_semyon"],
                "Semyon. Alias: hunter.",
            ),
            make_doc(
                "doc_anna",
                "entity_anna",
                ["entity_semyon", "entity_anna"],
                "Semyon protected Anna. Anna was jealous.",
            ),
            make_doc(
                "doc_ivan",
                "entity_ivan",
                ["entity_anna", "entity_ivan"],
                "Ivan approached Anna and later confronted Semyon.",
            ),
        ],
        encoder,
    )
    return DynamicGraphRLMArm(
        index=index,
        graph_view=GraphViewRef(
            document_id="doc",
            graph_version="graph_v1",
            projection_version="projection_v1",
            encoder_version=encoder.backend.encoder_version,
        ),
        gateway=gateway,
        config=DualRLMConfig(
            graph_top_k=3,
            max_graph_depth=3,
            max_graph_model_calls=8,
            max_graph_expansions=4,
        ),
    )


def test_dynamic_graph_rlm_uses_command_loop_for_two_validated_hops() -> None:
    gateway = ScriptedRLMGateway(
        graph_decisions=[
            GraphRLMDecision(
                action="inspect",
                confidence=0.4,
                decision_summary="Inspect Semyon's local graph evidence.",
            ),
            GraphRLMDecision(
                action="expand",
                selected_transition_ids=["__target_owner__:entity_anna"],
                confidence=0.7,
                decision_summary="Move from Semyon to Anna.",
            ),
            GraphRLMDecision(
                action="reformulate",
                subquery="Find interactions involving Anna that caused conflict.",
                confidence=0.7,
                decision_summary="Need Anna-mediated conflict context.",
            ),
            GraphRLMDecision(
                action="expand",
                selected_transition_ids=["__target_owner__:entity_ivan"],
                confidence=0.8,
                decision_summary="Move from Anna to Ivan.",
            ),
            GraphRLMDecision(
                action="answer",
                evidence_sufficient=True,
                confidence=0.9,
                decision_summary="Two-hop path has enough evidence.",
            ),
        ]
    )
    result = make_dynamic_arm(gateway).run(
        "How did Semyon's relationship with Anna lead to conflict with Ivan?",
        run_id="run",
    )

    expand_steps = [step for step in result.trace if step["step"] == "expand_frontier"]
    assert len(expand_steps) == 2
    assert expand_steps[0]["target_owner_id"] == "entity_anna"
    assert expand_steps[1]["target_owner_id"] == "entity_ivan"
    assert len(result.model_call_traces) >= 5
    assert all(trace.fallback_used for trace in result.model_call_traces)
    assert result.stop_reason == "answer"


def test_dynamic_graph_rlm_rejects_transition_outside_presented_frontier() -> None:
    gateway = ScriptedRLMGateway(
        graph_decisions=[
            GraphRLMDecision(
                action="expand",
                selected_transition_ids=["transition_not_presented"],
                confidence=0.1,
                decision_summary="Invalid jump.",
            )
        ]
    )

    with pytest.raises(ValueError, match="outside presented frontier"):
        make_dynamic_arm(gateway).run("Semyon and Anna", run_id="run")


def test_dynamic_graph_rlm_does_not_materialize_graph_text_on_max_depth() -> None:
    gateway = ScriptedRLMGateway(graph_decisions=[])
    arm = make_dynamic_arm(gateway)
    arm.config.max_graph_depth = 0

    result = arm.run("Semyon and Anna", run_id="run")

    assert result.stop_reason == "max_depth_reached"
    assert result.answer_candidate is None
    assert not any("Semyon. Alias: hunter." in str(step) for step in result.trace)
    finalize_steps = [step for step in result.trace if step["step"] == "finalize_answer"]
    assert finalize_steps[-1]["answer_materialized"] is False


def test_graph_prompt_sections_expose_budget_keys_without_raw_trace_growth() -> None:
    gateway = ScriptedRLMGateway(
        graph_decisions=[
            GraphRLMDecision(
                action="answer",
                evidence_sufficient=True,
                confidence=1.0,
                decision_summary="Done.",
            )
        ]
    )
    arm = make_dynamic_arm(gateway)
    state = arm._inspect_local(arm._locate_seed(arm._initial_state("Semyon and Anna")))

    sections = _graph_prompt_sections(state)

    assert {"system", "query", "state", "frontier", "visited_evidence", "history", "tool_schema"} <= set(sections)
    assert "model_call_traces" not in sections["history"]
    assert "presented_frontier_count" in sections["history"]
    assert "Semyon protected Anna" not in sections["history"]
    assert "transition_" in sections["frontier"]


def test_dynamic_text_rlm_uses_gateway_without_graph_state() -> None:
    gateway = ScriptedRLMGateway(
        graph_decisions=[],
        text_results=[
            TextRLMResult(
                status="evidence_found",
                answer_summary="Text evidence links Anna, Ivan, and Semyon.",
                consulted_chunk_ids=["chunk_1"],
                evidence_span_ids=["text_span_1"],
                entity_mentions=["Semyon", "Anna", "Ivan"],
                confidence=0.8,
            )
        ],
    )
    text_store = ImmutableTextStore(
        [
            SourceChunk(
                document_id="doc",
                chunk_id="chunk_1",
                text="Ivan approached Anna, and Semyon confronted him.",
            )
        ]
    )

    result = DynamicTextRLMArm(
        text_store=text_store,
        gateway=gateway,
        document_id="doc",
        config=DualRLMConfig(text_top_k=1, max_text_rounds=2),
    ).run("Anna Ivan Semyon", run_id="run")

    assert result.arm == "text_rlm"
    assert result.evidence_span_ids == ["text_span_1"]
    assert result.consulted_object_ids == ["chunk_1"]
    assert result.model_call_traces[0].purpose == "text_inspection"
    assert result.model_call_traces[0].fallback_used is True


def test_dual_runtime_accepts_dynamic_arms_without_static_graph_fallback() -> None:
    gateway = ScriptedRLMGateway(
        graph_decisions=[
            GraphRLMDecision(
                action="inspect",
                confidence=0.4,
                decision_summary="Inspect Semyon's local graph evidence.",
            ),
            GraphRLMDecision(
                action="expand",
                selected_transition_ids=["__target_owner__:entity_anna"],
                confidence=0.7,
                decision_summary="Move from Semyon to Anna.",
            ),
            GraphRLMDecision(
                action="expand",
                selected_transition_ids=["__target_owner__:entity_ivan"],
                confidence=0.8,
                decision_summary="Move from Anna to Ivan.",
            ),
            GraphRLMDecision(
                action="answer",
                evidence_sufficient=True,
                confidence=0.9,
                decision_summary="Dynamic graph traversal has enough evidence.",
            ),
        ],
        text_results=[
            TextRLMResult(
                status="evidence_found",
                answer_summary="Text evidence mentions Semyon, Anna, and Ivan.",
                consulted_chunk_ids=["chunk_1"],
                evidence_span_ids=["text_span_1"],
                entity_mentions=["Semyon", "Anna", "Ivan"],
                confidence=0.7,
            )
        ],
    )
    graph_arm = make_dynamic_arm(gateway)
    text_arm = DynamicTextRLMArm(
        text_store=ImmutableTextStore(
            [
                SourceChunk(
                    document_id="doc",
                    chunk_id="chunk_1",
                    text="Semyon protected Anna before Ivan confronted them.",
                )
            ]
        ),
        gateway=gateway,
        document_id="doc",
        config=DualRLMConfig(text_top_k=1, max_text_rounds=1),
    )

    result = IndependentDualRLMRuntime(graph_arm, text_arm).run(
        "How did Semyon's relationship with Anna lead to conflict with Ivan?",
        graph_view=graph_arm.graph_view,
        run_id="run",
    )

    expand_steps = [
        step for step in result.graph_result.trace if step["step"] == "expand_frontier"
    ]
    assert result.graph_result.arm == "graph_rlm"
    assert len(expand_steps) == 2
    assert expand_steps[0]["target_owner_id"] == "entity_anna"
    assert expand_steps[1]["target_owner_id"] == "entity_ivan"
    assert len(result.graph_result.model_call_traces) >= 4
    assert result.graph_mutation_allowed is False
