from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.dual_rlm.models import (
    GraphRLMDecision,
    GraphRLMState,
    ModelCallTrace,
    SourceChunk,
    TextRLMResult,
    TextRLMState,
)


class ScriptedRLMGateway:
    def __init__(
        self,
        graph_decisions: list[GraphRLMDecision],
        text_results: list[TextRLMResult] | None = None,
    ) -> None:
        self.graph_decisions = list(graph_decisions)
        self.text_results = list(text_results or [])
        self._model_call_traces: list[ModelCallTrace] = []

    @property
    def model_call_traces(self) -> list[ModelCallTrace]:
        return self._model_call_traces

    def decide_graph(self, state: GraphRLMState) -> GraphRLMDecision:
        if self.graph_decisions:
            decision = self.graph_decisions.pop(0)
        else:
            decision = GraphRLMDecision(
                action="stop",
                confidence=0.0,
                decision_summary="Scripted gateway exhausted decisions.",
            )
        decision = self._resolve_placeholders(decision, state)
        self._record("graph_decision", "GraphRLMDecision")
        return decision

    def inspect_text(self, state: TextRLMState, chunks: list[SourceChunk]) -> TextRLMResult:
        if self.text_results:
            result = self.text_results.pop(0)
        else:
            result = TextRLMResult(
                status="insufficient_evidence",
                answer_summary="Scripted gateway has no text result.",
                confidence=0.0,
                unresolved_questions=["No scripted Text-RLM result was provided."],
            )
        self._record("text_inspection", "TextRLMResult")
        return result

    def _record(self, purpose: str, output_type: str) -> None:
        now = datetime.now(timezone.utc)
        self._model_call_traces.append(
            ModelCallTrace(
                call_id=f"scripted_call_{uuid4().hex[:12]}",
                provider="scripted",
                model_name="scripted-gateway",
                purpose=purpose,
                request_timestamp=now,
                response_timestamp=now,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                response_id=None,
                validated_output_type=output_type,
                fallback_used=True,
            )
        )

    def _resolve_placeholders(
        self,
        decision: GraphRLMDecision,
        state: GraphRLMState,
    ) -> GraphRLMDecision:
        if decision.action != "expand" or not decision.selected_transition_ids:
            return decision
        resolved = []
        for transition_id in decision.selected_transition_ids:
            if transition_id == "__first__":
                if state["frontier_transition_ids"]:
                    resolved.append(state["frontier_transition_ids"][0])
                continue
            if transition_id.startswith("__target_owner__:"):
                target_owner = transition_id.split(":", 1)[1]
                for transition in state["presented_frontier"]:
                    if transition["target_owner_id"] == target_owner:
                        resolved.append(transition["transition_id"])
                        break
                continue
            resolved.append(transition_id)
        return decision.model_copy(update={"selected_transition_ids": resolved})
