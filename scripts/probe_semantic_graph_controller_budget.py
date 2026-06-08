from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.arms import GraphRLMSemanticGraphArm
from app.benchmarks.oolong.loader import OOLONGSynthSource, load_oolong_synth_cases
from app.core.local_env import load_local_env
from app.dual_rlm import GraphRLMDecision
from app.dual_rlm.gateway import _jsonable_state


class BudgetCaptureGateway:
    def __init__(self) -> None:
        self.captured_states: list[dict] = []

    @property
    def model_call_traces(self):
        return []

    def decide_graph(self, state):
        self.captured_states.append(dict(state))
        if len(self.captured_states) == 1 and state.get("presented_frontier"):
            selected = state["presented_frontier"][0]["transition_id"]
            return GraphRLMDecision(
                action="expand",
                selected_transition_ids=[selected],
                confidence=0.5,
                decision_summary="Budget capture expands the first valid presented transition.",
            )
        return GraphRLMDecision(
            action="answer",
            evidence_sufficient=True,
            confidence=1.0,
            decision_summary="Budget capture complete.",
        )


def main() -> None:
    load_local_env(PROJECT_ROOT)
    args = parse_args()
    case = load_oolong_synth_cases(
        OOLONGSynthSource(
            split=args.hf_split,
            revision=args.hf_revision,
            offset=args.hf_offset,
            length=1,
        )
    )[0].arm_view()
    gateway = BudgetCaptureGateway()
    arm = GraphRLMSemanticGraphArm(
        model_name=args.model,
        reasoning_effort=args.reasoning_effort,
        experiment_id=args.experiment_id,
        worker_model_name=args.worker_model,
        worker_reasoning_effort=args.worker_reasoning_effort,
        encoder_backend=args.encoder_backend,
        gateway_factory=lambda: gateway,
    )
    result = arm.run_case(case)
    states = gateway.captured_states
    payloads = [_budget_for_state(state, case.context) for state in states]
    repeated_evidence_ids = _repeated_evidence_across_payloads(payloads)
    report = {
        "task_id": case.task_id,
        "arm": result.trace[0]["runtime_provenance"]["arm"],
        "model_stack": {
            "root_model": args.model,
            "root_reasoning_effort": args.reasoning_effort,
            "worker_model": args.worker_model,
            "worker_reasoning_effort": args.worker_reasoning_effort,
            "experiment_id": args.experiment_id,
        },
        "runtime_provenance": result.trace[0]["runtime_provenance"],
        "payloads": payloads,
        "guards": {
            "initial_payload_under_20k_estimated_tokens": (
                not payloads or payloads[0]["controller_payload_estimated_tokens"] < 20000
            ),
            "no_labelled_context_leakage": result.trace[0]["runtime_provenance"]["validation"][
                "labelled_context_leakage"
            ]
            is False,
            "legacy_hash_frontier_used": result.trace[0]["runtime_provenance"][
                "legacy_hash_frontier_used"
            ],
            "duplicate_semantic_document_ids_absent": all(
                not payload["duplicate_semantic_document_ids"] for payload in payloads
            ),
            "duplicate_evidence_span_ids_absent": all(
                not payload["duplicate_evidence_span_ids"] for payload in payloads
            ),
            "repeated_evidence_span_ids_across_payloads_absent": not repeated_evidence_ids,
            "payloads_not_monotonically_growing_without_reason": _payload_growth_guard(payloads),
        },
        "repeated_evidence_span_ids_across_payloads": repeated_evidence_ids,
        "trace": result.trace,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:6000])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-split", default="validation")
    parser.add_argument("--hf-offset", type=int, default=0)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--worker-model", default=None)
    parser.add_argument("--worker-reasoning-effort", default=None)
    parser.add_argument("--experiment-id", default="paper_aligned_gpt5_v1")
    parser.add_argument("--encoder-backend", choices=["transformer", "hashing"], default="transformer")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/semantic_graph_controller_budget.json"),
    )
    return parser.parse_args()


def _budget_for_state(state: dict, original_context: str) -> dict:
    payload = json.dumps(_jsonable_state(state), ensure_ascii=False, sort_keys=True)
    presented_frontier = state.get("presented_frontier") or []
    current_docs = state.get("current_document_ids") or []
    semantic_ids = list(current_docs) + [
        item.get("target_document_id")
        for item in presented_frontier
        if item.get("target_document_id")
    ]
    evidence_ids = [
        evidence_id
        for item in presented_frontier
        for evidence_id in item.get("evidence_span_ids", [])
    ]
    texts = [str(item.get("text") or "") for item in presented_frontier]
    raw_context_chars_presented = sum(
        len(text) for text in texts if text and text in original_context
    )
    return {
        "controller_payload_chars": len(payload),
        "controller_payload_estimated_tokens": max(1, len(payload) // 4),
        "semantic_documents_presented": len(set(semantic_ids)),
        "frontier_transitions_presented": len(presented_frontier),
        "evidence_spans_presented": len(set(evidence_ids)),
        "evidence_span_ids_presented": sorted(set(evidence_ids)),
        "raw_context_chars_presented": raw_context_chars_presented,
        "duplicate_semantic_document_ids": sorted(
            {doc_id for doc_id in semantic_ids if semantic_ids.count(doc_id) > 1}
        ),
        "duplicate_evidence_span_ids": sorted(
            {
                evidence_id
                for evidence_id in evidence_ids
                if evidence_ids.count(evidence_id) > 1
            }
        ),
    }


def _repeated_evidence_across_payloads(payloads: list[dict]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for payload in payloads:
        evidence_ids = set(payload.get("evidence_span_ids_presented") or [])
        repeated.update(seen & evidence_ids)
        seen.update(evidence_ids)
    return sorted(repeated)


def _payload_growth_guard(payloads: list[dict]) -> bool:
    if len(payloads) < 2:
        return True
    initial = payloads[0]["controller_payload_estimated_tokens"]
    return all(
        payload["controller_payload_estimated_tokens"] <= int(initial * 1.25)
        for payload in payloads[1:]
    )


if __name__ == "__main__":
    main()
