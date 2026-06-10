from __future__ import annotations

import hashlib
import re

from app.benchmarks.models import BenchmarkCase
from app.semantic_encoding.encoder import GraphSemanticEncoder
from app.semantic_encoding.index import GraphSemanticIndex
from app.semantic_encoding.models import GraphSemanticDocument, GraphSemanticEmbedding


_EMBEDDING_CACHE: dict[tuple[str, str], GraphSemanticEmbedding] = {}


def build_musique_semantic_documents(case: BenchmarkCase) -> list[GraphSemanticDocument]:
    paragraphs = case.metadata["paragraphs"]
    titles = {paragraph["idx"]: paragraph["title"] for paragraph in paragraphs}
    documents = []
    for paragraph in paragraphs:
        own_idx = paragraph["idx"]
        own_entity = _entity_id(titles[own_idx])
        linked_entities = {own_entity}
        text_lower = paragraph["paragraph_text"].lower()
        for other_idx, other_title in titles.items():
            if other_idx == own_idx or len(other_title) < 4:
                continue
            if other_title.lower() in text_lower:
                linked_entities.add(_entity_id(other_title))
        evidence_id = f"para_{own_idx}"
        text = f"{paragraph['title']}. {paragraph['paragraph_text']}"
        documents.append(
            GraphSemanticDocument(
                document_id=f"musique_{case.task_id}",
                semantic_document_id=f"{case.task_id}:{evidence_id}",
                owner_type="evidence",
                owner_id=evidence_id,
                source_entity_ids=sorted(linked_entities),
                evidence_span_ids=[evidence_id],
                source_chunk_ids=[evidence_id],
                text=text,
                structural_features={
                    "title": paragraph["title"],
                    "linked_title_count": len(linked_entities) - 1,
                },
                projection_version="musique_paragraph_graph_v1",
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:24],
            )
        )
    return documents


def build_musique_semantic_index(
    case: BenchmarkCase,
    encoder: GraphSemanticEncoder,
) -> GraphSemanticIndex:
    documents = build_musique_semantic_documents(case)
    embeddings = []
    for document in documents:
        cache_key = (encoder.backend.encoder_name, document.content_hash)
        cached = _EMBEDDING_CACHE.get(cache_key)
        if cached is None:
            cached = encoder.encode_document(document)
            _EMBEDDING_CACHE[cache_key] = cached
        embeddings.append(
            cached.model_copy(update={"semantic_document_id": document.semantic_document_id})
        )
    return GraphSemanticIndex(documents=documents, embeddings=embeddings, encoder=encoder)


def _entity_id(title: str) -> str:
    normalized = re.sub(r"\W+", "_", title.lower()).strip("_")
    return f"entity_{normalized}"
