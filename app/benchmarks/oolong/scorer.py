from __future__ import annotations

import ast
import re

from app.benchmarks.models import BenchmarkArmResult, BenchmarkCase, BenchmarkScore
from app.benchmarks.proxy.scorers import answer_exact_match, answer_token_f1


class OOLONGLocalCompatibleScorer:
    """Local compatible scorer for OOLONG answer-field samples.

    The Hugging Face card defines the standard input as
    `context_window_text + "\n" + question` and output as `answer`.
    This adapter does not replace the upstream official evaluation harness.
    It intentionally reports `is_official_score=false` until a source
    repository, commit, entrypoint, and benchmark version are pinned.
    """

    score_backend = "oolong_local_compatible_v2"

    def score(self, case: BenchmarkCase, prediction: BenchmarkArmResult) -> list[BenchmarkScore]:
        exact = oolong_answer_match(case, prediction.prediction)
        token_f1 = answer_token_f1(prediction.prediction, _gold_values(case.gold_answer))
        return [
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="task_score",
                score_value=exact,
                is_official_score=False,
                metadata={"dataset_origin": case.dataset_origin},
            ),
            BenchmarkScore(
                score_backend=self.score_backend,
                score_name="answer_token_f1",
                score_value=token_f1,
                is_official_score=False,
                metadata={"dataset_origin": case.dataset_origin},
            ),
        ]


def oolong_answer_match(case: BenchmarkCase, prediction: str) -> float:
    expected_values = [_normalize_value(value) for value in _gold_values(case.gold_answer)]
    predicted_values = _predicted_values(case, prediction)
    return 1.0 if any(value in expected_values for value in predicted_values) else 0.0


def _gold_values(gold_answer: str | list[str]) -> list[str]:
    if isinstance(gold_answer, list):
        return [str(item) for item in gold_answer]
    value = str(gold_answer)
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _predicted_values(case: BenchmarkCase, prediction: str) -> list[str]:
    answer_type = str(case.metadata.get("native_fields", {}).get("answer_type", ""))
    text = prediction.strip()
    if "NUMERIC" in answer_type:
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        return [_normalize_value(match) for match in matches]
    if "USER" in answer_type:
        matches = re.findall(r"\b\d+\b", text)
        return [_normalize_value(match) for match in matches]
    if "LABEL" in answer_type:
        return _predicted_label_values(case, text)
    if "COMPARISON" in answer_type:
        options = [
            "more common than",
            "less common than",
            "same frequency as",
            "more common",
            "less common",
            "same frequency",
        ]
        lowered = text.lower()
        return [_normalize_value(option) for option in options if option in lowered]
    prefixed = re.search(r"\b(?:Answer|Label|User)\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if prefixed:
        return [_normalize_value(prefixed.group(1))]
    return [_normalize_value(text)]


def _literal_values(text: str) -> list[str]:
    try:
        parsed = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _normalize_value(value: str) -> str:
    return str(value).strip().strip("'\"").lower()


def _normalize_label_value(value: str) -> str:
    normalized = _normalize_value(value)
    return re.sub(r"[\s\.,;:!?]+$", "", normalized)


def _label_vocabulary(case: BenchmarkCase) -> list[str]:
    native = case.metadata.get("native_fields", {})
    labels = native.get("labels") or native.get("label_options") or native.get("answer_options")
    values: list[str] = []
    if isinstance(labels, list):
        values.extend(str(item) for item in labels)
    values.extend(_labels_from_question(case.question))
    values.extend(_gold_values(case.gold_answer))
    return sorted({_normalize_label_value(value) for value in values if str(value).strip()})


def _labels_from_question(question: str) -> list[str]:
    match = re.search(
        r"labels?\s*:\s*([A-Za-z0-9_\-,\s]+)",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    return [
        item.strip()
        for item in re.split(r",|\bor\b", match.group(1), flags=re.IGNORECASE)
        if item.strip()
    ]


def _predicted_label_values(case: BenchmarkCase, text: str) -> list[str]:
    literal_candidates = [_normalize_label_value(value) for value in _literal_values(text)]
    vocabulary = _label_vocabulary(case)
    if literal_candidates and any(candidate in vocabulary for candidate in literal_candidates):
        return [candidate for candidate in literal_candidates if candidate in vocabulary]

    lowered = text.lower()
    if vocabulary:
        matches = [
            label
            for label in vocabulary
            if re.search(rf"\b{re.escape(label)}\b", lowered, flags=re.IGNORECASE)
        ]
        unique_matches = sorted(set(matches))
        return unique_matches if len(unique_matches) == 1 else []

    prefixed = re.findall(
        r"\bLabel\s*:\s*([^\n\r\.;:!?]+)",
        text,
        flags=re.IGNORECASE,
    )
    unique_prefixed = sorted({_normalize_label_value(value) for value in prefixed})
    return unique_prefixed if len(unique_prefixed) == 1 else []
