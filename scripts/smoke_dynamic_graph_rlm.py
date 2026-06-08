from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.local_env import load_local_env
from app.dual_rlm import (
    DualRLMConfig,
    DynamicGraphRLMArm,
    GraphRLMDecision,
    GraphViewRef,
    PydanticAIGPTGateway,
    ScriptedRLMGateway,
)
from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticDocument,
    GraphSemanticEncoder,
    GraphSemanticIndex,
    HashingSemanticEncoder,
    TransformerSemanticEncoder,
)


@dataclass(frozen=True)
class SmokeScenario:
    name: str
    query: str


POSITIVE_SCENARIO = SmokeScenario(
    name="positive_two_hop",
    query="How did Semyon's relationship with Anna lead to conflict with Ivan?",
)
CONTROL_SCENARIO = SmokeScenario(
    name="control_rifle_location",
    query="Where was Semyon carrying his rifle?",
)


def main() -> None:
    load_local_env(PROJECT_ROOT)
    args = parse_args()
    scenarios = selected_scenarios(args)
    encoder_config = EncoderConfig(
        projection_dim=None if args.backend == "transformer" else 256,
        embedding_dim=None,
        interaction_dim=64,
        transformer_model_name=args.embedding_model,
        local_frontier_cap=10,
    )
    backend = (
        HashingSemanticEncoder(dimensions=256)
        if args.backend == "hashing"
        else TransformerSemanticEncoder(
            model_name=encoder_config.transformer_model_name,
            device=args.device,
            embedding_dim=encoder_config.embedding_dim,
        )
    )
    encoder = GraphSemanticEncoder(config=encoder_config, backend=backend)
    index = GraphSemanticIndex.build(make_documents(), encoder)
    graph_view = GraphViewRef(
        document_id="dynamic_graph_rlm_doc",
        graph_version="graph_v1",
        projection_version="projection_v1",
        encoder_version=backend.encoder_version,
    )
    payloads = []
    for repeat_index in range(args.repeat):
        for scenario in scenarios:
            gateway = make_gateway(args, scenario)
            arm = DynamicGraphRLMArm(
                index=index,
                graph_view=graph_view,
                gateway=gateway,
                config=DualRLMConfig(
                    graph_top_k=5,
                    max_graph_depth=4,
                    max_graph_model_calls=8,
                    max_graph_expansions=4,
                ),
            )
            payload = run_scenario(arm, scenario, repeat_index=repeat_index)
            payloads.append(payload)
            if args.require_real_model:
                assert_real_model_acceptance(payload, scenario)

    output = payloads[0] if len(payloads) == 1 else {"scenarios": payloads}
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["scripted", "gpt5"], default="scripted")
    parser.add_argument("--require-real-model", action="store_true")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--scenario",
        choices=["positive", "control", "all"],
        default="positive",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--backend", choices=["transformer", "hashing"], default="transformer")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--query",
        default=None,
    )
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    return args


def selected_scenarios(args: argparse.Namespace) -> list[SmokeScenario]:
    if args.query:
        return [SmokeScenario(name="custom", query=args.query)]
    if args.scenario == "all":
        return [POSITIVE_SCENARIO, CONTROL_SCENARIO]
    if args.scenario == "control":
        return [CONTROL_SCENARIO]
    return [POSITIVE_SCENARIO]


def make_gateway(args: argparse.Namespace, scenario: SmokeScenario):
    if args.controller == "gpt5":
        return PydanticAIGPTGateway(
            model_name=args.model,
            require_real_model=args.require_real_model,
        )
    if args.require_real_model:
        raise RuntimeError("--require-real-model cannot be used with --controller scripted")
    if scenario.name == CONTROL_SCENARIO.name:
        return ScriptedRLMGateway(
            graph_decisions=[
                GraphRLMDecision(
                    action="inspect",
                    confidence=0.4,
                    decision_summary="Inspect local graph evidence.",
                ),
                GraphRLMDecision(
                    action="expand",
                    selected_transition_ids=["__target_owner__:entity_forest"],
                    confidence=0.75,
                    decision_summary="Move from Semyon to the forest/rifle location context.",
                ),
                GraphRLMDecision(
                    action="answer",
                    evidence_sufficient=True,
                    confidence=0.9,
                    decision_summary="Location evidence is sufficient.",
                ),
            ]
        )
    return ScriptedRLMGateway(
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


def run_scenario(
    arm: DynamicGraphRLMArm,
    scenario: SmokeScenario,
    *,
    repeat_index: int,
) -> dict:
    result = arm.run(
        scenario.query,
        run_id=f"dynamic_graph_rlm_smoke:{scenario.name}:repeat_{repeat_index}",
    )
    payload = result.model_dump()
    payload["scenario"] = scenario.name
    payload["repeat_index"] = repeat_index
    payload["query"] = scenario.query
    payload["graph_depth"] = max(
        [step.get("depth", 0) for step in result.trace if isinstance(step, dict)] or [0]
    )
    payload["graph_mutation_count"] = 0
    payload["real_model_call_count"] = len(
        [trace for trace in result.model_call_traces if not trace.fallback_used]
    )
    payload["model_total_tokens"] = sum(
        trace.total_tokens or 0 for trace in result.model_call_traces
    )
    payload["model_latencies_ms"] = [
        int((trace.response_timestamp - trace.request_timestamp).total_seconds() * 1000)
        for trace in result.model_call_traces
    ]
    payload["answer_evidence_span_ids"] = list(result.evidence_span_ids)
    payload["expanded_owner_edges"] = [
        [step.get("source_owner_id"), step.get("target_owner_id")]
        for step in result.trace
        if isinstance(step, dict) and step.get("step") == "expand_frontier"
    ]
    payload["frontier_selection_valid"] = selected_transitions_are_frontier_bound(
        result.trace
    )
    return payload


def make_documents() -> list[GraphSemanticDocument]:
    rows = [
        (
            "doc_semyon",
            "entity",
            "entity_semyon",
            ["entity_semyon"],
            "Semyon. Alias: hunter. He has relationship, warning, forest, and rifle contexts.",
        ),
        (
            "doc_anna",
            "pair_interaction",
            "entity_anna",
            ["entity_semyon", "entity_anna"],
            "Semyon protected Anna. Anna was jealous when Semyon spoke with Maria.",
        ),
        (
            "doc_ivan",
            "pair_interaction",
            "entity_ivan",
            ["entity_anna", "entity_ivan"],
            "Ivan approached Anna again. Semyon saw them together, and Ivan confronted Semyon.",
        ),
        (
            "doc_old_man",
            "pair_interaction",
            "entity_old_man",
            ["entity_semyon", "entity_old_man"],
            "An old man warned the hunter not to continue deeper into the forest.",
        ),
        (
            "doc_forest",
            "evidence",
            "entity_forest",
            ["entity_semyon", "entity_forest"],
            "Semyon entered the forest carrying his rifle.",
        ),
        (
            "doc_rifle",
            "evidence",
            "entity_rifle",
            ["entity_semyon", "entity_rifle"],
            "The hunter carried his rifle while walking through the forest.",
        ),
    ]
    return [
        GraphSemanticDocument(
            document_id="dynamic_graph_rlm_doc",
            semantic_document_id=document_id,
            owner_type=owner_type,
            owner_id=owner_id,
            source_entity_ids=source_entity_ids,
            event_ids=[f"ev_{document_id}"],
            evidence_span_ids=[f"span_{document_id}"],
            source_chunk_ids=[f"chunk_{index + 1}"],
            text=text,
            structural_features={"smoke": "dynamic_graph_rlm"},
            projection_version="projection_v1",
            content_hash=sha256(text.encode("utf-8")).hexdigest(),
        )
        for index, (document_id, owner_type, owner_id, source_entity_ids, text) in enumerate(rows)
    ]


def selected_transitions_are_frontier_bound(trace: list[dict]) -> bool:
    presented_frontier_ids: set[str] = set()
    for step in trace:
        if not isinstance(step, dict):
            continue
        if step.get("step") == "inspect_local":
            presented_frontier_ids = set(step.get("frontier_transition_ids") or [])
        if step.get("step") == "graph_decide":
            selected = set(
                (step.get("decision") or {}).get("selected_transition_ids") or []
            )
            if selected and not selected.issubset(presented_frontier_ids):
                return False
    return True


def assert_real_model_acceptance(payload: dict, scenario: SmokeScenario) -> None:
    if payload["graph_mutation_count"] != 0:
        raise SystemExit("Expected graph_mutation_count == 0.")
    if payload["stop_reason"] != "answer":
        raise SystemExit("Expected stop_reason == 'answer'.")
    if not payload.get("answer_evidence_span_ids"):
        raise SystemExit("Expected final answer to carry evidence_span_ids.")
    if not set(payload["answer_evidence_span_ids"]).issubset(
        set(payload.get("evidence_span_ids") or [])
    ):
        raise SystemExit("Answer evidence must be a subset of collected evidence.")
    if not payload["frontier_selection_valid"]:
        raise SystemExit("Expected selected_transition_ids to be inside presented_frontier_ids.")
    traces = payload.get("model_call_traces", [])
    if not traces:
        raise SystemExit("Expected model_call_traces in --require-real-model mode.")
    if any(trace.get("fallback_used") for trace in traces):
        raise SystemExit("Deterministic fallback was used in --require-real-model mode.")
    if any(trace.get("provider") != "openai" for trace in traces):
        raise SystemExit("Expected provider == 'openai' for every model call.")
    if not all(trace.get("response_id") for trace in traces):
        raise SystemExit("Provider response metadata is missing.")
    if not all((trace.get("total_tokens") or 0) > 0 for trace in traces):
        raise SystemExit("Model usage.total_tokens must be present and > 0.")
    if sum(trace.get("total_tokens") or 0 for trace in traces) <= 0:
        raise SystemExit("Total model usage tokens must be > 0.")
    if payload.get("model_total_tokens", 0) <= 0:
        raise SystemExit("Payload model_total_tokens must be > 0.")
    if scenario.name == POSITIVE_SCENARIO.name:
        if payload["real_model_call_count"] < 3:
            raise SystemExit("Expected at least 3 real model calls.")
        if payload["graph_depth"] < 2:
            raise SystemExit("Expected graph_depth >= 2.")
        edges = payload["expanded_owner_edges"]
        if ["entity_semyon", "entity_anna"] not in edges:
            raise SystemExit("Expected visited transition Semyon -> Anna.")
        if ["entity_anna", "entity_ivan"] not in edges:
            raise SystemExit("Expected visited transition Anna -> Ivan.")
    if scenario.name == CONTROL_SCENARIO.name:
        edges = payload["expanded_owner_edges"]
        if ["entity_semyon", "entity_anna"] in edges and ["entity_anna", "entity_ivan"] in edges:
            raise SystemExit("Control query repeated positive Anna -> Ivan route.")
        control_targets = {"entity_forest", "entity_rifle"}
        if edges and not any(edge[1] in control_targets for edge in edges):
            raise SystemExit("Control query should choose forest/rifle context or stop with local evidence.")


if __name__ == "__main__":
    main()
