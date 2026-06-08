from __future__ import annotations

from app.benchmarks.models import BenchmarkArmResult, BenchmarkCase, BenchmarkScore
from app.benchmarks.proxy.scorers import answer_exact_match


class SNIAHScorer:
    score_backend = "sniah_official_exact_match"

    def score(self, case: BenchmarkCase, prediction: BenchmarkArmResult) -> list[BenchmarkScore]:
        return [
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="accuracy",
                score_value=answer_exact_match(prediction.prediction, case.gold_answer),
                is_official_score=case.dataset_origin == "official",
                metadata={"dataset_origin": case.dataset_origin},
            )
        ]
