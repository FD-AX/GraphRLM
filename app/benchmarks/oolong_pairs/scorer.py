from __future__ import annotations

from app.benchmarks.models import BenchmarkArmResult, BenchmarkCase, BenchmarkScore
from app.benchmarks.proxy.scorers import answer_token_f1


class OOLONGPairsOfficialScorer:
    score_backend = "oolong_pairs_f1"

    def score(self, case: BenchmarkCase, prediction: BenchmarkArmResult) -> list[BenchmarkScore]:
        return [
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="pair_f1",
                score_value=answer_token_f1(prediction.prediction, case.gold_answer),
                is_official_score=case.dataset_origin == "official",
                metadata={"dataset_origin": case.dataset_origin},
            )
        ]
