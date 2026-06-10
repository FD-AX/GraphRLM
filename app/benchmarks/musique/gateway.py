from __future__ import annotations

import json

from pydantic import BaseModel

from app.dual_rlm.gateway import PydanticAIGPTGateway, _graph_prompt_sections, _jsonable_state
from app.dual_rlm.models import GraphRLMDecision, GraphRLMState


class SynthesizedAnswer(BaseModel):
    answer: str
    confidence: float = 0.0


_ACTION_GUIDE = (
    "You control a read-only traversal over a paragraph graph to collect ALL "
    "paragraphs needed to answer a multi-hop question. Actions:\n"
    "- expand: move to one transition from presented_frontier (REQUIRED: put its "
    "transition_id into selected_transition_ids). This is the only action that "
    "makes progress and collects new evidence.\n"
    "- inspect: re-list the frontier of the current paragraph. It does NOT move "
    "or collect anything. Never choose inspect twice in a row.\n"
    "- reformulate: set subquery to a refined search lens for the next hop.\n"
    "- verify: re-check collected evidence (no movement).\n"
    "- answer: finish. Put the short final answer (just the entity/span, no "
    "explanation) into decision_summary. Only answer when collected_evidence_ids "
    "cover every hop of the question.\n"
    "- stop: give up (evidence cannot be completed).\n"
    "Strategy: hop through bridge paragraphs. After expanding to a paragraph "
    "that mentions the next bridge entity, expand again toward it. Prefer "
    "expand over inspect whenever the frontier is non-empty. If the frontier "
    "has no transition toward the missing hop, choose reformulate with a "
    "subquery naming the missing entity or fact - the runtime will search the "
    "whole graph for it and jump to the best match."
)


class MuSiQueGraphGateway(PydanticAIGPTGateway):
    """Task-tuned controller gateway with local decision sanitization.

    Keeps the strict typed-output contract of the base gateway, but explains
    the action semantics for the paragraph-graph task and repairs the two
    failure modes observed in probes: expand without transition ids and
    inspect loops that burn the whole call budget.
    """

    def decide_graph(self, state: GraphRLMState) -> GraphRLMDecision:
        sections = _graph_prompt_sections(state)
        sections["instruction"] = _ACTION_GUIDE
        prompt = (
            f"{_ACTION_GUIDE}\n\n"
            f"State:\n{json.dumps(_jsonable_state(state), ensure_ascii=False, indent=2)}"
        )
        decision = self._call_model(
            purpose="graph_decision",
            prompt=prompt,
            output_type=GraphRLMDecision,
            prompt_sections=sections,
        )
        return self._sanitize(decision, state)

    def _sanitize(self, decision: GraphRLMDecision, state: GraphRLMState) -> GraphRLMDecision:
        frontier_ids = list(state.get("frontier_transition_ids", []))
        if decision.action == "expand":
            valid = [
                transition_id
                for transition_id in decision.selected_transition_ids
                if transition_id in frontier_ids
            ]
            if valid:
                return decision.model_copy(update={"selected_transition_ids": valid[:1]})
            if frontier_ids:
                return decision.model_copy(
                    update={
                        "selected_transition_ids": frontier_ids[:1],
                        "decision_summary": (
                            decision.decision_summary
                            + " [sanitized: expand without valid transition_id, took first frontier transition]"
                        ),
                    }
                )
            return decision.model_copy(
                update={
                    "action": "verify",
                    "decision_summary": (
                        decision.decision_summary
                        + " [sanitized: expand with empty frontier -> verify]"
                    ),
                }
            )
        if decision.action == "inspect" and _last_action(state) == "inspect" and frontier_ids:
            return decision.model_copy(
                update={
                    "action": "expand",
                    "selected_transition_ids": frontier_ids[:1],
                    "decision_summary": (
                        decision.decision_summary
                        + " [sanitized: repeated inspect -> expand first frontier transition]"
                    ),
                }
            )
        if decision.action == "reformulate" and not decision.subquery:
            return decision.model_copy(
                update={"subquery": state.get("current_subquery") or state.get("query")}
            )
        return decision


    def synthesize_answer(self, question: str, evidence_texts: list[str]) -> SynthesizedAnswer:
        sections = {
            "instruction": (
                "Answer the multi-hop question using ONLY the evidence paragraphs. "
                "Return the shortest exact answer span (an entity name, date, or "
                "short phrase). No explanation."
            ),
            "question": question,
            "evidence": json.dumps(evidence_texts, ensure_ascii=False),
        }
        prompt = (
            f"{sections['instruction']}\n\nQuestion: {question}\n\n"
            f"Evidence paragraphs:\n{sections['evidence']}"
        )
        return self._call_model(
            purpose="answer_synthesis",
            prompt=prompt,
            output_type=SynthesizedAnswer,
            prompt_sections=sections,
        )


def _last_action(state: GraphRLMState) -> str | None:
    last = state.get("last_decision")
    if isinstance(last, dict):
        return last.get("action")
    return None
