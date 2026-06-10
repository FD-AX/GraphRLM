from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Protocol

from app.encoder.relation_encoder import (
    DEFAULT_RELATION_ENCODER_MODEL,
    hashing_vector,
)
from app.semantic_encoding.models import (
    EncoderConfig,
    GraphSemanticDocument,
    GraphSemanticEmbedding,
)


class SemanticEncoder(Protocol):
    encoder_name: str
    encoder_version: str

    def encode_query(self, text: str) -> list[float]:
        ...

    def encode_document(self, text: str) -> list[float]:
        ...

    def encode_transition(self, text: str) -> list[float]:
        ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def weighted_pool(vectors: list[tuple[list[float], float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = len(vectors[0][0])
    pooled = [0.0] * dimensions
    for vector, weight in vectors:
        for index, value in enumerate(vector):
            pooled[index] += value * weight
    return normalize(pooled)


class HashingSemanticEncoder:
    def __init__(
        self,
        dimensions: int = 256,
        encoder_name: str = "hashing-bert-like",
        encoder_version: str = "hashing_semantic_encoder_v1",
    ) -> None:
        self.dimensions = dimensions
        self.encoder_name = encoder_name
        self.encoder_version = encoder_version
        self.actual_embedding_dim = dimensions
        self.encoder_revision = encoder_version
        self.tokenizer_name = "whitespace_hashing"
        self.tokenizer_revision = "hashing_v1"
        self.pooling_strategy = "signed_token_hash_sum_l2_normalized"
        self.normalize_embeddings = True

    def encode_query(self, text: str) -> list[float]:
        return hashing_vector(f"query: {text}", dimensions=self.dimensions)

    def encode_document(self, text: str) -> list[float]:
        return hashing_vector(f"document: {text}", dimensions=self.dimensions)

    def encode_transition(self, text: str) -> list[float]:
        return hashing_vector(f"transition: {text}", dimensions=self.dimensions)


class TransformerSemanticEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_RELATION_ENCODER_MODEL,
        device: str = "cpu",
        embedding_dim: int | None = None,
        encoder_version: str = "transformer_semantic_encoder_v1",
        normalize_embeddings: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for TransformerSemanticEncoder. "
                "Install it or use HashingSemanticEncoder only for fast unit tests."
            ) from exc

        self.model_name = model_name
        self.device = device
        self.encoder_name = model_name
        self.encoder_version = encoder_version
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model_name, device=device)
        if hasattr(self._model, "get_embedding_dimension"):
            native_dim = self._model.get_embedding_dimension()
        else:
            native_dim = self._model.get_sentence_embedding_dimension()
        self.actual_embedding_dim = int(native_dim)
        tokenizer = getattr(self._model, "tokenizer", None)
        self.tokenizer_name = getattr(tokenizer, "name_or_path", model_name)
        self.tokenizer_revision = "not_persisted"
        self.encoder_revision = "not_persisted"
        self.pooling_strategy = _sentence_transformer_pooling_strategy(self._model)
        if embedding_dim is not None and embedding_dim != native_dim:
            raise ValueError(
                "TransformerSemanticEncoder uses the model native dimension by "
                "default. A different embedding_dim requires a trained projection "
                "head, which is not configured in this baseline."
            )

    def encode_query(self, text: str) -> list[float]:
        return self._encode(f"query: {text}")

    def encode_document(self, text: str) -> list[float]:
        return self._encode(f"document: {text}")

    def encode_transition(self, text: str) -> list[float]:
        return self._encode(f"transition: {text}")

    def _encode(self, text: str) -> list[float]:
        vector = self._model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return [float(value) for value in vector.tolist()]


class GraphSemanticEncoder:
    def __init__(
        self,
        config: EncoderConfig | None = None,
        backend: SemanticEncoder | None = None,
    ):
        self.config = config or EncoderConfig()
        self.backend = backend or HashingSemanticEncoder(
            dimensions=self._hashing_dimensions(),
            encoder_name=self.config.encoder_name,
            encoder_version=self.config.encoder_version,
        )

    def encode_query(self, text: str) -> list[float]:
        return self.backend.encode_query(text)

    def encode_document(self, document: GraphSemanticDocument) -> GraphSemanticEmbedding:
        text = (
            f"{document.owner_type}: {document.text}\n"
            f"features: {document.structural_features}"
        )
        vector = self.backend.encode_document(text)
        return GraphSemanticEmbedding(
            semantic_document_id=document.semantic_document_id,
            embedding=vector,
            encoder_name=self.backend.encoder_name,
            encoder_version=self.backend.encoder_version,
            embedding_dim=len(vector),
            content_hash=document.content_hash,
            actual_embedding_dim=getattr(self.backend, "actual_embedding_dim", len(vector)),
            encoder_revision=getattr(self.backend, "encoder_revision", self.backend.encoder_version),
            tokenizer_name=getattr(self.backend, "tokenizer_name", None),
            tokenizer_revision=getattr(self.backend, "tokenizer_revision", None),
            pooling_strategy=getattr(self.backend, "pooling_strategy", None),
            serializer_version="graph_semantic_document_serializer_v1",
            normalize_embeddings=getattr(self.backend, "normalize_embeddings", None),
            query_prefix="query:",
            document_prefix="document:",
        )

    def encode_transition(self, transition_text: str) -> list[float]:
        return self.backend.encode_transition(transition_text)

    def interaction_profile(
        self,
        query_embedding: list[float],
        document_embedding: list[float],
    ) -> list[float]:
        if self.config.interaction_profile_mode == "contribution":
            return normalize(
                [q * d for q, d in zip(query_embedding, document_embedding)]
            )
        features = []
        for q, d in zip(query_embedding, document_embedding):
            features.extend([q, d, q * d, abs(q - d)])
        interaction_dim = self._interaction_dimensions(query_embedding, document_embedding)
        projected = [0.0] * interaction_dim
        for index, value in enumerate(features):
            bucket = self._feature_bucket(index, interaction_dim)
            sign = 1.0 if index % 2 == 0 else -1.0
            projected[bucket] += sign * value
        return normalize(projected)

    def contribution_vector(
        self,
        query_embedding: list[float],
        document_embedding: list[float],
    ) -> list[float]:
        return [q * d for q, d in zip(query_embedding, document_embedding)]

    def top_contribution_components(
        self,
        query_embedding: list[float],
        document_embedding: list[float],
        top_k: int = 8,
    ) -> list[dict]:
        contributions = self.contribution_vector(query_embedding, document_embedding)
        ranked = sorted(
            enumerate(contributions),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_k]
        return [
            {
                "dimension": index,
                "contribution": value,
                "query_value": query_embedding[index],
                "document_value": document_embedding[index],
            }
            for index, value in ranked
        ]

    def update_navigation_embedding(
        self,
        original_query_embedding: list[float],
        current_navigation_embedding: list[float],
        current_document_embedding: list[float],
    ) -> list[float]:
        return weighted_pool(
            [
                (original_query_embedding, 0.45),
                (current_navigation_embedding, 0.25),
                (current_document_embedding, 0.30),
            ]
        )

    def _feature_bucket(self, feature_index: int, dimensions: int) -> int:
        return _feature_bucket_cached(feature_index, dimensions)

    def _hashing_dimensions(self) -> int:
        return self.config.embedding_dim or self.config.projection_dim or 256

    def _interaction_dimensions(
        self,
        query_embedding: list[float],
        document_embedding: list[float],
    ) -> int:
        if self.config.interaction_dim:
            return self.config.interaction_dim
        return min(64, len(query_embedding), len(document_embedding))


@lru_cache(maxsize=None)
def _feature_bucket_cached(feature_index: int, dimensions: int) -> int:
    digest = hashlib.sha256(str(feature_index).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dimensions


def _sentence_transformer_pooling_strategy(model) -> str:
    try:
        modules = list(model.modules())
    except Exception:
        return "sentence_transformer_pipeline"
    for module in modules:
        name = module.__class__.__name__.lower()
        if "pooling" not in name:
            continue
        modes = []
        for attr, label in [
            ("pooling_mode_cls_token", "cls"),
            ("pooling_mode_mean_tokens", "mean"),
            ("pooling_mode_max_tokens", "max"),
            ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len"),
            ("pooling_mode_weightedmean_tokens", "weighted_mean"),
            ("pooling_mode_lasttoken", "last_token"),
        ]:
            if getattr(module, attr, False):
                modes.append(label)
        return "+".join(modes) if modes else "sentence_transformer_pooling"
    return "sentence_transformer_pipeline"
