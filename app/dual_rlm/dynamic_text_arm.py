from __future__ import annotations

from app.dual_rlm.models import (
    DualRLMConfig,
    RLMGateway,
    RetrievalArmResult,
    TextRLMState,
)
from app.dual_rlm.text_store import ImmutableTextStore


class DynamicTextRLMArm:
    def __init__(
        self,
        text_store: ImmutableTextStore,
        gateway: RLMGateway,
        document_id: str,
        config: DualRLMConfig | None = None,
    ) -> None:
        self.text_store = text_store
        self.gateway = gateway
        self.document_id = document_id
        self.config = config or DualRLMConfig()

    def run(self, query: str, run_id: str) -> RetrievalArmResult:
        state: TextRLMState = {
            "query": query,
            "document_id": self.document_id,
            "current_subquery": query,
            "retrieved_chunk_ids": [],
            "evidence_span_ids": [],
            "working_memory": [],
            "visited_chunk_ids": [],
            "recursion_depth": 0,
            "remaining_calls": self.config.max_text_rounds,
            "stop_reason": None,
        }
        trace = []
        answer = None
        confidence = 0.0
        status = "insufficient_evidence"

        for round_index in range(self.config.max_text_rounds):
            chunks = self.text_store.search_text(
                state["current_subquery"],
                top_k=self.config.text_top_k,
            )
            chunks = [
                chunk for chunk in chunks if chunk.chunk_id not in state["visited_chunk_ids"]
            ]
            if not chunks:
                state["stop_reason"] = "no_more_text_chunks"
                break

            state["retrieved_chunk_ids"].extend(chunk.chunk_id for chunk in chunks)
            state["visited_chunk_ids"].extend(chunk.chunk_id for chunk in chunks)
            result = self.gateway.inspect_text(state, chunks)
            status = result.status
            answer = result.answer_summary
            confidence = result.confidence
            state["evidence_span_ids"] = list(
                dict.fromkeys(state["evidence_span_ids"] + result.evidence_span_ids)
            )
            state["working_memory"].append(result.answer_summary)
            state["recursion_depth"] = round_index + 1
            state["remaining_calls"] = max(state["remaining_calls"] - 1, 0)
            trace.append(
                {
                    "round": round_index,
                    "action": "gateway.inspect_text",
                    "chunk_ids": [chunk.chunk_id for chunk in chunks],
                    "status": result.status,
                    "evidence_span_ids": result.evidence_span_ids,
                    "unresolved_questions": result.unresolved_questions,
                }
            )
            if result.status in {"evidence_found", "contradiction_found"}:
                state["stop_reason"] = result.status
                break
            if result.unresolved_questions:
                state["current_subquery"] = result.unresolved_questions[0]

        stop_reason = state["stop_reason"] or status
        model_call_traces = [
            trace for trace in self.gateway.model_call_traces if trace.purpose == "text_inspection"
        ]
        return RetrievalArmResult(
            arm="text_rlm",
            answer_candidate=answer,
            evidence_span_ids=list(dict.fromkeys(state["evidence_span_ids"])),
            consulted_object_ids=list(dict.fromkeys(state["visited_chunk_ids"])),
            confidence=confidence,
            completeness=min(
                1.0,
                len(set(state["evidence_span_ids"])) / max(self.config.text_top_k, 1),
            ),
            stop_reason=stop_reason,
            trace_id=f"{run_id}:dynamic_text",
            trace=trace,
            model_call_traces=model_call_traces,
        )
