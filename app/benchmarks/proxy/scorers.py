from __future__ import annotations

import re
import string
from collections import Counter

from app.benchmarks.models import BenchmarkArmResult, BenchmarkCase, BenchmarkScore


class ProxyExactMatchScorer:
    score_backend = "proxy_exact_match"

    def score(self, case: BenchmarkCase, prediction: BenchmarkArmResult) -> list[BenchmarkScore]:
        return [
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="exact_match",
                score_value=answer_exact_match(prediction.prediction, case.gold_answer),
                is_official_score=False,
            )
        ]


class ProxyTokenF1Scorer:
    score_backend = "proxy_token_f1"

    def score(self, case: BenchmarkCase, prediction: BenchmarkArmResult) -> list[BenchmarkScore]:
        return [
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="token_f1",
                score_value=answer_token_f1(prediction.prediction, case.gold_answer),
                is_official_score=False,
            )
        ]


class ProxyEvidenceScorer:
    score_backend = "proxy_evidence"

    def score(self, case: BenchmarkCase, prediction: BenchmarkArmResult) -> list[BenchmarkScore]:
        precision, recall, f1 = evidence_scores(
            prediction.evidence_span_ids,
            case.gold_evidence_span_ids,
        )
        return [
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="evidence_precision",
                score_value=precision,
                is_official_score=False,
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="evidence_recall",
                score_value=recall,
                is_official_score=False,
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="evidence_f1",
                score_value=f1,
                is_official_score=False,
            ),
        ]


def default_proxy_scorers():
    return [ProxyExactMatchScorer(), ProxyTokenF1Scorer(), ProxyEvidenceScorer()]


def answer_exact_match(prediction: str, gold_answer: str | list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return 1.0 if any(
        normalized_prediction == normalize_answer(answer)
        for answer in _gold_answers(gold_answer)
    ) else 0.0


def answer_token_f1(prediction: str, gold_answer: str | list[str]) -> float:
    return max(
        (_token_f1(prediction, answer) for answer in _gold_answers(gold_answer)),
        default=0.0,
    )


def evidence_scores(
    predicted_ids: list[str],
    gold_ids: list[str],
) -> tuple[float, float, float]:
    predicted = set(predicted_ids)
    gold = set(gold_ids)
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    common_count = sum(common.values())
    if common_count == 0:
        return 0.0
    precision = common_count / len(prediction_tokens)
    recall = common_count / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _gold_answers(gold_answer: str | list[str]) -> list[str]:
    if isinstance(gold_answer, str):
        return [gold_answer]
    return [str(answer) for answer in gold_answer]
