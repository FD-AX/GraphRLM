from __future__ import annotations

import json

from app.benchmarks.models import BenchmarkArmResult, BenchmarkCase, BenchmarkScore
from app.benchmarks.proxy.scorers import answer_exact_match, evidence_scores


class FactLensAuditScorer:
    score_backend = "factlens_local_audit_v1"

    def score(self, case: BenchmarkCase, prediction: BenchmarkArmResult) -> list[BenchmarkScore]:
        audit = _audit_payload(prediction)
        precision, recall, f1 = evidence_scores(prediction.evidence_span_ids, case.gold_evidence_span_ids)
        supported_score = answer_exact_match(prediction.prediction, case.gold_answer)
        useful_graph = audit.get("graph_contribution_outcome") == "useful"
        return [
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="factlens_supported",
                score_value=supported_score,
                is_official_score=False,
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="factlens_evidence_recall",
                score_value=recall,
                is_official_score=False,
                metadata={"precision": precision, "f1": f1},
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="factlens_complete_evidence_coverage",
                score_value=1.0 if audit.get("complete_evidence_coverage") else 0.0,
                is_official_score=False,
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="factlens_subclaim_recall",
                score_value=float(audit.get("subclaim_recall") or 0.0),
                is_official_score=False,
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="factlens_all_required_subclaims_verified",
                score_value=1.0 if audit.get("all_required_subclaims_verified") else 0.0,
                is_official_score=False,
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="factlens_unsupported_subclaim_rate",
                score_value=float(audit.get("unsupported_subclaim_rate") or 0.0),
                is_official_score=False,
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="factlens_unsupported_verdict_rate",
                score_value=float(audit.get("unsupported_verdict_rate") or 0.0),
                is_official_score=False,
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="factlens_graph_useful",
                score_value=1.0 if useful_graph else 0.0,
                is_official_score=False,
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="factlens_fully_verified_claims_per_10k_tokens",
                score_value=float(audit.get("fully_verified_claims_per_10k_tokens") or 0.0),
                is_official_score=False,
            ),
        ]


def _audit_payload(prediction: BenchmarkArmResult) -> dict:
    if prediction.trace:
        audit = prediction.trace[0].get("factlens_audit")
        if isinstance(audit, dict):
            return audit
    if prediction.raw_response:
        try:
            value = json.loads(prediction.raw_response)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}
