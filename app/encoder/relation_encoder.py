from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


DEFAULT_RELATION_ENCODER_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_PROJECTION_SPACE_ID = "general_relation_space_v1"


@dataclass(frozen=True)
class EncodedRelation:
    vector: list[float]
    vector_ref: str
    projection_space_id: str
    encoder_model: str


class RelationEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_RELATION_ENCODER_MODEL,
        projection_space_id: str = DEFAULT_PROJECTION_SPACE_ID,
        device: str = "cpu",
        allow_hashing_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.projection_space_id = projection_space_id
        self.device = device
        self.allow_hashing_fallback = allow_hashing_fallback
        self._model = None

    def encode_relation(
        self,
        source_name: str,
        target_name: str,
        relation_span: str,
        evidence_text: str,
        relation_candidate_id: str,
    ) -> EncodedRelation:
        text = (
            f"source: {source_name}\n"
            f"target: {target_name}\n"
            f"relation: {relation_span}\n"
            f"evidence: {evidence_text}"
        )

        vector = self._encode_text(text)
        vector_ref = stable_id("vector", f"{self.projection_space_id}:{relation_candidate_id}")

        return EncodedRelation(
            vector=vector,
            vector_ref=vector_ref,
            projection_space_id=self.projection_space_id,
            encoder_model=self.model_name,
        )

    def _encode_text(self, text: str) -> list[float]:
        if self.allow_hashing_fallback and self.model_name == "hashing-fallback":
            return hashing_vector(text)

        try:
            model = self._get_model()
            vector = model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return [float(value) for value in vector.tolist()]
        except Exception:
            if not self.allow_hashing_fallback:
                raise

            return hashing_vector(text)

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for RelationEncoder. "
                    "Install requirements or pass allow_hashing_fallback=True "
                    "for smoke tests."
                ) from exc

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

        return self._model


def stable_id(prefix: str, raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def hashing_vector(text: str, dimensions: int = 384) -> list[float]:
    values = [0.0] * dimensions

    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values

    return [value / norm for value in values]
