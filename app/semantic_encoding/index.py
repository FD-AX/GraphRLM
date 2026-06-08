from __future__ import annotations

from app.semantic_encoding.encoder import GraphSemanticEncoder, cosine_similarity
from app.semantic_encoding.models import (
    GraphSemanticDocument,
    GraphSemanticEmbedding,
    SemanticSearchResult,
)


class GraphSemanticIndex:
    def __init__(
        self,
        documents: list[GraphSemanticDocument],
        embeddings: list[GraphSemanticEmbedding],
        encoder: GraphSemanticEncoder,
    ) -> None:
        self.documents = {document.semantic_document_id: document for document in documents}
        self.embeddings = {
            embedding.semantic_document_id: embedding for embedding in embeddings
        }
        self.encoder = encoder

    @classmethod
    def build(
        cls,
        documents: list[GraphSemanticDocument],
        encoder: GraphSemanticEncoder,
    ) -> "GraphSemanticIndex":
        embeddings = [encoder.encode_document(document) for document in documents]
        return cls(documents=documents, embeddings=embeddings, encoder=encoder)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        owner_types: set[str] | None = None,
        candidate_document_ids: set[str] | None = None,
        visited_document_ids: set[str] | None = None,
    ) -> list[SemanticSearchResult]:
        visited_document_ids = visited_document_ids or set()
        results = []
        for document_id, document in self.documents.items():
            if document_id in visited_document_ids:
                continue
            if candidate_document_ids is not None and document_id not in candidate_document_ids:
                continue
            if owner_types is not None and document.owner_type not in owner_types:
                continue
            embedding = self.embeddings.get(document_id)
            if embedding is None:
                continue
            score = cosine_similarity(query_embedding, embedding.embedding)
            results.append(
                SemanticSearchResult(
                    semantic_document_id=document.semantic_document_id,
                    owner_type=document.owner_type,
                    owner_id=document.owner_id,
                    score=score,
                    query_similarity=score,
                    contribution_top_components=self.encoder.top_contribution_components(
                        query_embedding,
                        embedding.embedding,
                    ),
                    source_entity_ids=document.source_entity_ids,
                    event_ids=document.event_ids,
                    evidence_span_ids=document.evidence_span_ids,
                    source_chunk_ids=document.source_chunk_ids,
                    text=document.text,
                    score_parts={"query_similarity": score},
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def embedding_for(self, semantic_document_id: str) -> list[float]:
        return self.embeddings[semantic_document_id].embedding

    def document_for(self, semantic_document_id: str) -> GraphSemanticDocument:
        return self.documents[semantic_document_id]

    def frontier_document_ids(
        self,
        current: GraphSemanticDocument,
        cap: int,
    ) -> set[str]:
        current_entities = set(current.source_entity_ids)
        current_events = set(current.event_ids)
        current_evidence = set(current.evidence_span_ids)
        candidates = []
        for document in self.documents.values():
            if document.semantic_document_id == current.semantic_document_id:
                continue
            shared_entities = current_entities & set(document.source_entity_ids)
            shared_events = current_events & set(document.event_ids)
            shared_evidence = current_evidence & set(document.evidence_span_ids)
            structural_score = (
                3 * len(shared_events)
                + 2 * len(shared_entities)
                + len(shared_evidence)
            )
            if structural_score > 0:
                candidates.append((structural_score, document.semantic_document_id))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return {document_id for _, document_id in candidates[:cap]}
