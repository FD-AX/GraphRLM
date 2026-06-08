from __future__ import annotations

from app.benchmarks.models import BenchmarkRunRecord


def latest_revision_for_record(revisions: list[dict], record: BenchmarkRunRecord) -> dict | None:
    matching = [
        revision
        for revision in revisions
        if str(revision.get("task_id")) == str(record.task_id)
        and str(revision.get("arm")) == str(record.arm)
        and revision.get("model_rerun") is False
    ]
    return matching[-1] if matching else None


def score_from_record(record: BenchmarkRunRecord, score_name: str) -> dict:
    for score in record.scores:
        if score.score_name == score_name:
            return {
                "score_backend": score.score_backend,
                "score_value": score.score_value,
                "is_official_score": score.is_official_score,
            }
    return {"score_backend": None, "score_value": None, "is_official_score": False}


def active_score(original: dict, revision: dict | None, score_name: str) -> dict:
    if revision:
        for score in revision.get("rescored_scores", []):
            if score.get("score_name") == score_name:
                return {
                    "score_backend": score.get("score_backend"),
                    "score_value": score.get("score_value"),
                    "is_official_score": score.get("is_official_score", False),
                    "revision_reason": revision.get("rescore_reason"),
                    "revision_path": revision.get("_revision_path"),
                    "model_rerun": revision.get("model_rerun"),
                }
    return {
        "score_backend": original["score_backend"],
        "score_value": original["score_value"],
        "is_official_score": original["is_official_score"],
        "revision_reason": None,
        "revision_path": None,
        "model_rerun": False,
    }
