from __future__ import annotations

import re
import string
from collections import Counter

from app.benchmarks.models import BenchmarkArmResult, BenchmarkCase, BenchmarkScore


class MuSiQueCompletenessScorer:
    """Evidence-completeness metrics plus standard span answer metrics."""

    score_backend = "musique_completeness_v1"

    def score(
        self,
        case: BenchmarkCase,
        prediction: BenchmarkArmResult,
    ) -> list[BenchmarkScore]:
        gold = set(case.gold_evidence_span_ids)
        retrieved = list(dict.fromkeys(prediction.evidence_span_ids))
        matched = gold & set(retrieved)
        scores = [
            self._score("evidence_recall", len(matched) / len(gold) if gold else None),
            self._score(
                "evidence_precision",
                len(matched) / len(retrieved) if retrieved else 0.0,
            ),
            self._score(
                "complete_evidence_coverage",
                1.0 if gold and matched == gold else 0.0,
            ),
            self._score("retrieved_count", float(len(retrieved))),
        ]
        answer_text = prediction.prediction or ""
        if answer_text.strip():
            references = [case.gold_answer] if isinstance(case.gold_answer, str) else list(case.gold_answer)
            references += list(
                case.metadata.get("native_fields", {}).get("answer_aliases", [])
            )
            scores.append(
                self._score(
                    "exact_match",
                    max(
                        1.0 if _normalize(answer_text) == _normalize(reference) else 0.0
                        for reference in references
                    ),
                )
            )
            scores.append(
                self._score(
                    "answer_f1",
                    max(_token_f1(answer_text, reference) for reference in references),
                )
            )
        return scores

    def _score(self, name: str, value: float | None) -> BenchmarkScore:
        return BenchmarkScore(
            score_backend=self.score_backend,
            score_name=name,
            score_value=value,
        )


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _normalize(prediction).split()
    reference_tokens = _normalize(reference).split()
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
