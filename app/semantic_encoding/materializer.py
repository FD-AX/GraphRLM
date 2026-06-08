from __future__ import annotations

import json

from app.graph.neo4j_client import Neo4jClient
from app.semantic_encoding.models import GraphSemanticDocument, GraphSemanticEmbedding


class GraphSemanticMaterializer:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def materialize_documents(
        self,
        documents: list[GraphSemanticDocument],
        embeddings: list[GraphSemanticEmbedding],
    ) -> None:
        embedding_by_document = {
            embedding.semantic_document_id: embedding for embedding in embeddings
        }
        for document in documents:
            self.materialize_document(document, embedding_by_document.get(document.semantic_document_id))

    def materialize_document(
        self,
        document: GraphSemanticDocument,
        embedding: GraphSemanticEmbedding | None,
    ) -> None:
        self.client.execute_write(
            """
            MERGE (d:GraphSemanticDocument {semantic_document_id: $semantic_document_id})
            SET d.document_id = $document_id,
                d.owner_type = $owner_type,
                d.owner_id = $owner_id,
                d.source_entity_ids = $source_entity_ids,
                d.event_ids = $event_ids,
                d.evidence_span_ids = $evidence_span_ids,
                d.source_chunk_ids = $source_chunk_ids,
                d.text = $text,
                d.structural_features_json = $structural_features_json,
                d.projection_version = $projection_version,
                d.content_hash = $content_hash,
                d.updated_at = datetime()
            """,
            {
                **document.model_dump(exclude={"structural_features"}),
                "structural_features_json": json.dumps(
                    document.structural_features,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        )
        for entity_id in document.source_entity_ids:
            self.client.execute_write(
                """
                MATCH (d:GraphSemanticDocument {semantic_document_id: $semantic_document_id})
                MATCH (e:Entity {entity_id: $entity_id})
                MERGE (d)-[:SOURCE_ENTITY]->(e)
                """,
                {"semantic_document_id": document.semantic_document_id, "entity_id": entity_id},
            )
        for event_id in document.event_ids:
            self.client.execute_write(
                """
                MATCH (d:GraphSemanticDocument {semantic_document_id: $semantic_document_id})
                MATCH (ev:EventFrame {event_frame_id: $event_id})
                MERGE (d)-[:SOURCE_EVENT]->(ev)
                """,
                {"semantic_document_id": document.semantic_document_id, "event_id": event_id},
            )
        for evidence_span_id in document.evidence_span_ids:
            self.client.execute_write(
                """
                MATCH (d:GraphSemanticDocument {semantic_document_id: $semantic_document_id})
                MATCH (span:EvidenceSpan {span_id: $evidence_span_id})
                MERGE (d)-[:SUPPORTED_BY]->(span)
                """,
                {
                    "semantic_document_id": document.semantic_document_id,
                    "evidence_span_id": evidence_span_id,
                },
            )
        if embedding is not None:
            self.materialize_embedding(embedding)

    def materialize_embedding(self, embedding: GraphSemanticEmbedding) -> None:
        self.client.execute_write(
            """
            MATCH (d:GraphSemanticDocument {semantic_document_id: $semantic_document_id})
            MERGE (e:GraphSemanticEmbedding {semantic_document_id: $semantic_document_id})
            SET e.embedding = $embedding,
                e.encoder_name = $encoder_name,
                e.encoder_version = $encoder_version,
                e.embedding_dim = $embedding_dim,
                e.content_hash = $content_hash,
                e.updated_at = datetime()
            MERGE (e)-[:EMBEDS]->(d)
            """,
            embedding.model_dump(),
        )
