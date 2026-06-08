from __future__ import annotations

from app.dual_rlm.models import DualRLMConfig, RetrievalArmResult
from app.dual_rlm.text_store import ImmutableTextStore


class TextRLMArm:
    def __init__(
        self,
        text_store: ImmutableTextStore,
        config: DualRLMConfig | None = None,
    ) -> None:
        self.text_store = text_store
        self.config = config or DualRLMConfig()

    def run(self, query: str, run_id: str) -> RetrievalArmResult:
        seed_chunks = self.text_store.search_text(query, top_k=self.config.text_top_k)
        if not seed_chunks:
            return RetrievalArmResult(
                arm="text_rlm",
                stop_reason="no_text_seed",
                trace_id=f"{run_id}:text",
            )

        consulted = []
        evidence = []
        trace = []
        working_texts = []
        visited = set()
        frontier = seed_chunks
        for round_index in range(self.config.max_text_rounds):
            next_frontier = []
            for chunk in frontier:
                if chunk.chunk_id in visited:
                    continue
                visited.add(chunk.chunk_id)
                consulted.append(chunk.chunk_id)
                span = self.text_store.evidence_for_chunk(chunk)
                evidence.append(span.evidence_span_id)
                working_texts.append(span.text)
                trace.append(
                    {
                        "round": round_index,
                        "action": "inspect_chunk",
                        "chunk_id": chunk.chunk_id,
                        "evidence_span_id": span.evidence_span_id,
                    }
                )
                if round_index + 1 < self.config.max_text_rounds:
                    for neighbor in self.text_store.read_window(
                        chunk.chunk_id,
                        before=self.config.text_window_radius,
                        after=self.config.text_window_radius,
                    ):
                        if neighbor.chunk_id not in visited:
                            next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        completeness = min(1.0, len(set(evidence)) / max(self.config.text_top_k, 1))
        confidence = 0.55 if evidence else 0.0
        return RetrievalArmResult(
            arm="text_rlm",
            answer_candidate=_summarize(working_texts),
            evidence_span_ids=list(dict.fromkeys(evidence)),
            consulted_object_ids=list(dict.fromkeys(consulted)),
            confidence=confidence,
            completeness=completeness,
            stop_reason="text_evidence_retrieved",
            trace_id=f"{run_id}:text",
            trace=trace,
        )


def _summarize(texts: list[str]) -> str | None:
    if not texts:
        return None
    return " ".join(texts[:3])
