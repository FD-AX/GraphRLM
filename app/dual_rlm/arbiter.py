from __future__ import annotations

from app.dual_rlm.models import AnswerArbitrationResult, RetrievalArmResult


class EvidenceArbiter:
    def arbitrate(
        self,
        query: str,
        graph_result: RetrievalArmResult,
        text_result: RetrievalArmResult,
    ) -> AnswerArbitrationResult:
        graph_has_evidence = bool(graph_result.evidence_span_ids)
        text_has_evidence = bool(text_result.evidence_span_ids)
        contradictions = graph_result.contradictions + text_result.contradictions
        if contradictions:
            return AnswerArbitrationResult(
                answer_candidate=None,
                support_status="contradiction",
                confidence=min(graph_result.confidence, text_result.confidence),
                evidence_span_ids=list(
                    dict.fromkeys(graph_result.evidence_span_ids + text_result.evidence_span_ids)
                ),
                selected_arm="none",
                rationale="Graph and text arms reported contradictions.",
            )

        if graph_has_evidence and text_has_evidence:
            return AnswerArbitrationResult(
                answer_candidate=_combine_answers(graph_result, text_result),
                support_status="both_supported",
                confidence=max(graph_result.confidence, text_result.confidence),
                evidence_span_ids=list(
                    dict.fromkeys(graph_result.evidence_span_ids + text_result.evidence_span_ids)
                ),
                selected_arm="hybrid",
                rationale="Both read-only retrieval arms returned supporting evidence.",
            )

        if graph_has_evidence:
            return AnswerArbitrationResult(
                answer_candidate=graph_result.answer_candidate,
                support_status="graph_supported_text_unconfirmed",
                confidence=graph_result.confidence * 0.85,
                evidence_span_ids=graph_result.evidence_span_ids,
                selected_arm="graph_rlm",
                rationale="Graph arm found evidence; text arm did not confirm within budget.",
            )

        if text_has_evidence:
            return AnswerArbitrationResult(
                answer_candidate=text_result.answer_candidate,
                support_status="text_supported_graph_missing",
                confidence=text_result.confidence * 0.85,
                evidence_span_ids=text_result.evidence_span_ids,
                selected_arm="text_rlm",
                rationale="Text arm found source evidence; graph view did not contain a matching route.",
            )

        return AnswerArbitrationResult(
            answer_candidate=None,
            support_status="insufficient_evidence",
            confidence=0.0,
            selected_arm="none",
            rationale=f"No read-only retrieval arm found evidence for query: {query}",
        )


def _combine_answers(
    graph_result: RetrievalArmResult,
    text_result: RetrievalArmResult,
) -> str | None:
    parts = [
        part
        for part in [graph_result.answer_candidate, text_result.answer_candidate]
        if part
    ]
    return "\n".join(parts) if parts else None
