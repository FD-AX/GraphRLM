from __future__ import annotations

import argparse
import json

from app.core.graph_models import LatentRelationEdge
from app.encoder.relation_encoder import RelationEncoder, stable_id
from app.graph.neo4j_client import Neo4jClient
from app.graph.writer import GraphWriter


def fetch_pending_relation_candidates(
    client: Neo4jClient,
    document_id: str | None,
    limit: int,
) -> list[dict]:
    document_filter = ""
    params: dict = {"limit": limit}

    if document_id:
        document_filter = """
        MATCH (span)-[:IN_CHUNK]->(c:Chunk {document_id: $document_id})
        """
        params["document_id"] = document_id

    return client.execute_read(
        f"""
        MATCH (source:Entity)-[:RELATION_SOURCE]->(rc:RelationCandidate)
        MATCH (rc)-[:RELATION_TARGET]->(target:Entity)
        MATCH (rc)-[:SUPPORTED_BY]->(span:EvidenceSpan)
        {document_filter}
        WHERE rc.encode_as_latent_edge = true
          AND NOT (rc)<-[:ENCODES]-(:LatentRelation)
        RETURN
            rc.relation_candidate_id AS relation_candidate_id,
            rc.source_entity_id AS source_entity_id,
            rc.target_entity_id AS target_entity_id,
            rc.relation_span AS relation_span,
            rc.evidence_span_id AS evidence_span_id,
            rc.direction_confidence AS direction_confidence,
            rc.symbolic_hint AS symbolic_hint,
            rc.confidence AS confidence,
            source.canonical_name AS source_name,
            target.canonical_name AS target_name,
            span.original_text AS evidence_text
        LIMIT $limit
        """,
        params,
    )


def encode_pending_relation_candidates(
    uri: str,
    username: str,
    password: str,
    document_id: str | None,
    limit: int,
    model_name: str,
    projection_space_id: str,
    device: str,
    allow_hashing_fallback: bool,
) -> dict:
    client = Neo4jClient(
        uri=uri,
        username=username,
        password=password,
        connection_timeout=2.0,
        max_transaction_retry_time=2.0,
    )
    writer = GraphWriter(client)
    encoder = RelationEncoder(
        model_name=model_name,
        projection_space_id=projection_space_id,
        device=device,
        allow_hashing_fallback=allow_hashing_fallback,
    )

    try:
        candidates = fetch_pending_relation_candidates(
            client=client,
            document_id=document_id,
            limit=limit,
        )

        encoded_count = 0
        for candidate in candidates:
            encoded = encoder.encode_relation(
                source_name=candidate["source_name"],
                target_name=candidate["target_name"],
                relation_span=candidate["relation_span"],
                evidence_text=candidate["evidence_text"],
                relation_candidate_id=candidate["relation_candidate_id"],
            )

            edge = LatentRelationEdge(
                edge_id=stable_id(
                    "latent_relation",
                    (
                        f"{candidate['relation_candidate_id']}:"
                        f"{encoded.vector_ref}"
                    ),
                ),
                source_entity_id=candidate["source_entity_id"],
                target_entity_id=candidate["target_entity_id"],
                relation_candidate_id=candidate["relation_candidate_id"],
                evidence_span_id=candidate["evidence_span_id"],
                vector_ref=encoded.vector_ref,
                projection_space_id=encoded.projection_space_id,
                direction="directed",
                confidence=candidate["confidence"] or 0.0,
                symbolic_hint=candidate["symbolic_hint"],
                vector=encoded.vector,
            )
            writer.write_latent_relation_edge(edge)
            encoded_count += 1

        return {
            "pending_seen": len(candidates),
            "encoded": encoded_count,
            "projection_space_id": projection_space_id,
            "model_name": model_name,
        }
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--projection-space-id", default="general_relation_space_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-hashing-fallback", action="store_true")
    args = parser.parse_args()

    summary = encode_pending_relation_candidates(
        uri=args.uri,
        username=args.username,
        password=args.password,
        document_id=args.document_id,
        limit=args.limit,
        model_name=args.model_name,
        projection_space_id=args.projection_space_id,
        device=args.device,
        allow_hashing_fallback=args.allow_hashing_fallback,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
