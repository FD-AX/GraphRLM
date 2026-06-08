from __future__ import annotations

from datetime import datetime
from typing import Any

from app.graph.neo4j_client import Neo4jClient
from app.projection.entity_context import _content_hash, _snapshot_id, _unique
from app.projection.models import (
    EncodingBlock,
    EntityPairContextSnapshot,
    EntityPairEncodingInput,
    PairEventLink,
)


AGENT_ROLES = {"agent", "subject", "source", "speaker", "actor"}
PATIENT_ROLES = {"patient", "object", "target", "recipient", "listener"}


class EntityPairContextProjectionBuilder:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def build_entity_pair_snapshot(
        self,
        document_id: str,
        source_entity_id: str,
        target_entity_id: str,
        projection_version: str,
    ) -> EntityPairContextSnapshot:
        self._assert_document_entity(document_id, source_entity_id)
        self._assert_document_entity(document_id, target_entity_id)
        if source_entity_id == target_entity_id:
            raise ValueError("Entity pair requires two different entity ids")

        pair_id = self._pair_id(document_id, source_entity_id, target_entity_id)
        events = self._get_pair_events(
            document_id=document_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
        )
        source_to_target = [
            event for event in events if event.direction == "source_to_target"
        ]
        target_to_source = [
            event for event in events if event.direction == "target_to_source"
        ]
        shared = [event for event in events if event.direction == "shared"]
        relation_evidence_span_ids = _unique(
            [span_id for event in events for span_id in event.evidence_span_ids]
        )
        source_roles = _unique([event.source_role for event in events])
        target_roles = _unique([event.target_role for event in events])
        shared_time_scopes = _unique([event.temporal_scope for event in events])
        shared_locations = self._get_shared_locations(
            document_id=document_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
        )
        indirect_paths = self._get_indirect_paths(
            document_id=document_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
        )
        encoding_input = self._build_encoding_input(
            pair_id=pair_id,
            document_id=document_id,
            projection_version=projection_version,
            direct_shared_events=shared,
            source_to_target_events=source_to_target,
            target_to_source_events=target_to_source,
            shared_locations=shared_locations,
            shared_time_scopes=shared_time_scopes,
            relation_evidence_span_ids=relation_evidence_span_ids,
        )
        hash_payload = {
            "pair_id": pair_id,
            "source_entity_id": source_entity_id,
            "target_entity_id": target_entity_id,
            "direct_shared_events": [event.model_dump(mode="json") for event in shared],
            "source_to_target_events": [
                event.model_dump(mode="json") for event in source_to_target
            ],
            "target_to_source_events": [
                event.model_dump(mode="json") for event in target_to_source
            ],
            "shared_locations": shared_locations,
            "shared_time_scopes": shared_time_scopes,
            "relation_evidence_span_ids": relation_evidence_span_ids,
            "source_roles": source_roles,
            "target_roles": target_roles,
            "indirect_graph_paths": indirect_paths,
            "projection_version": projection_version,
            "encoding": encoding_input.model_dump(mode="json"),
        }
        content_hash = _content_hash(hash_payload)
        return EntityPairContextSnapshot(
            pair_id=pair_id,
            snapshot_id=_snapshot_id(
                "entity_pair_context_snapshot",
                document_id,
                source_entity_id,
                target_entity_id,
                projection_version,
                content_hash,
                datetime.utcnow().isoformat(),
            ),
            document_id=document_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            direct_shared_events=shared,
            source_to_target_events=source_to_target,
            target_to_source_events=target_to_source,
            shared_locations=shared_locations,
            shared_time_scopes=shared_time_scopes,
            relation_evidence_span_ids=relation_evidence_span_ids,
            source_roles=source_roles,
            target_roles=target_roles,
            indirect_graph_paths=indirect_paths,
            projection_version=projection_version,
            encoding_input=encoding_input,
            content_hash=content_hash,
        )

    def _assert_document_entity(self, document_id: str, entity_id: str) -> None:
        rows = self.client.execute_read(
            """
            MATCH (e:Entity {entity_id: $entity_id})
            WHERE EXISTS {
                MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
                MATCH (c)-[:HAS_MENTION]->(:Mention)-[:REFERS_TO]->(e)
            } OR EXISTS {
                MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
                MATCH (arg:EventArgument)-[:RESOLVED_TO]->(e)
                MATCH (arg)-[:ARGUMENT_OF]->(:EventFrame {document_id: $document_id})
            }
            RETURN e.entity_id AS entity_id
            LIMIT 1
            """,
            {"document_id": document_id, "entity_id": entity_id},
        )
        if not rows:
            raise ValueError(
                f"Entity {entity_id!r} is not materialized in document {document_id!r}"
            )

    def _get_pair_events(
        self,
        document_id: str,
        source_entity_id: str,
        target_entity_id: str,
    ) -> list[PairEventLink]:
        rows = self.client.execute_read(
            """
            MATCH (source_arg:EventArgument)-[:RESOLVED_TO]->(:Entity {entity_id: $source_entity_id})
            MATCH (target_arg:EventArgument)-[:RESOLVED_TO]->(:Entity {entity_id: $target_entity_id})
            MATCH (source_arg)-[:ARGUMENT_OF]->(ev:EventFrame {document_id: $document_id})
            MATCH (target_arg)-[:ARGUMENT_OF]->(ev)
            MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)-[:HAS_EVENT]->(ev)
            OPTIONAL MATCH (ev)-[:SUPPORTED_BY]->(span:EvidenceSpan)
            RETURN ev.event_frame_id AS event_id,
                   coalesce(ev.normalized_predicate, ev.predicate) AS predicate,
                   source_arg.role AS source_role,
                   target_arg.role AS target_role,
                   source_arg.surface_text AS source_surface,
                   target_arg.surface_text AS target_surface,
                   CASE WHEN span.span_id IS NULL THEN [] ELSE [span.span_id] END AS evidence_span_ids,
                   [c.chunk_id] AS source_chunk_ids,
                   ev.temporal_scope AS temporal_scope
            ORDER BY c.index ASC, ev.event_frame_id ASC
            """,
            {
                "document_id": document_id,
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
            },
        )
        return [
            PairEventLink(
                event_id=row["event_id"],
                predicate=row["predicate"],
                source_role=row.get("source_role"),
                target_role=row.get("target_role"),
                direction=self._direction(row.get("source_role"), row.get("target_role")),
                evidence_span_ids=row.get("evidence_span_ids") or [],
                source_chunk_ids=row.get("source_chunk_ids") or [],
                temporal_scope=row.get("temporal_scope"),
                surface_texts=_unique(
                    [row.get("source_surface"), row.get("target_surface")]
                ),
            )
            for row in rows
        ]

    def _get_shared_locations(
        self,
        document_id: str,
        source_entity_id: str,
        target_entity_id: str,
    ) -> list[str]:
        rows = self.client.execute_read(
            """
            MATCH (source_arg:EventArgument)-[:RESOLVED_TO]->(:Entity {entity_id: $source_entity_id})
            MATCH (target_arg:EventArgument)-[:RESOLVED_TO]->(:Entity {entity_id: $target_entity_id})
            MATCH (source_arg)-[:ARGUMENT_OF]->(ev:EventFrame {document_id: $document_id})
            MATCH (target_arg)-[:ARGUMENT_OF]->(ev)
            OPTIONAL MATCH (loc_arg:EventArgument)-[:ARGUMENT_OF]->(ev)
            WHERE toLower(coalesce(loc_arg.role, '')) IN ['location', 'place']
            RETURN DISTINCT loc_arg.surface_text AS location
            ORDER BY location
            """,
            {
                "document_id": document_id,
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
            },
        )
        return _unique([row.get("location") for row in rows])

    def _get_indirect_paths(
        self,
        document_id: str,
        source_entity_id: str,
        target_entity_id: str,
    ) -> list[list[str]]:
        rows = self.client.execute_read(
            """
            MATCH (source:Entity {entity_id: $source_entity_id})
            MATCH (target:Entity {entity_id: $target_entity_id})
            MATCH path = shortestPath((source)-[*..4]-(target))
            WHERE ALL(node IN nodes(path) WHERE
                NOT node:Chunk OR node.document_id = $document_id
            )
            RETURN [node IN nodes(path) |
                coalesce(node.entity_id, node.event_frame_id, node.argument_id, node.chunk_id)
            ] AS path_ids
            LIMIT 5
            """,
            {
                "document_id": document_id,
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
            },
        )
        paths = []
        for row in rows:
            path = [item for item in (row.get("path_ids") or []) if item]
            if path:
                paths.append(path)
        return paths

    def _direction(self, source_role: str | None, target_role: str | None) -> str:
        source = (source_role or "").lower()
        target = (target_role or "").lower()
        if source in AGENT_ROLES and target in PATIENT_ROLES:
            return "source_to_target"
        if source in PATIENT_ROLES and target in AGENT_ROLES:
            return "target_to_source"
        return "shared"

    def _pair_id(
        self,
        document_id: str,
        source_entity_id: str,
        target_entity_id: str,
    ) -> str:
        return _snapshot_id("entity_pair", document_id, source_entity_id, target_entity_id)

    def _build_encoding_input(
        self,
        pair_id: str,
        document_id: str,
        projection_version: str,
        direct_shared_events: list[PairEventLink],
        source_to_target_events: list[PairEventLink],
        target_to_source_events: list[PairEventLink],
        shared_locations: list[str],
        shared_time_scopes: list[str],
        relation_evidence_span_ids: list[str],
    ) -> EntityPairEncodingInput:
        direct_blocks = [
            self._event_block(pair_id, "direct", idx, event)
            for idx, event in enumerate(direct_shared_events)
        ]
        directional_blocks = [
            self._event_block(pair_id, "source_to_target", idx, event)
            for idx, event in enumerate(source_to_target_events)
        ] + [
            self._event_block(pair_id, "target_to_source", idx, event)
            for idx, event in enumerate(target_to_source_events)
        ]
        shared_context_blocks = [
            EncodingBlock(
                block_id=f"{pair_id}:shared_location:{idx}",
                block_type="shared_context",
                text=f"shared location: {location}",
            )
            for idx, location in enumerate(shared_locations)
        ]
        temporal_blocks = [
            EncodingBlock(
                block_id=f"{pair_id}:temporal:{idx}",
                block_type="temporal",
                text=f"shared temporal scope: {scope}",
            )
            for idx, scope in enumerate(shared_time_scopes)
        ]
        evidence_blocks = [
            EncodingBlock(
                block_id=f"{pair_id}:evidence:{idx}",
                block_type="evidence",
                text=f"pair evidence_span={span_id}",
                evidence_span_ids=[span_id],
            )
            for idx, span_id in enumerate(relation_evidence_span_ids)
        ]
        return EntityPairEncodingInput(
            pair_id=pair_id,
            document_id=document_id,
            projection_version=projection_version,
            direct_interaction_blocks=direct_blocks,
            directional_blocks=directional_blocks,
            shared_context_blocks=shared_context_blocks,
            temporal_blocks=temporal_blocks,
            evidence_blocks=evidence_blocks,
        )

    def _event_block(
        self,
        pair_id: str,
        block_type: str,
        idx: int,
        event: PairEventLink,
    ) -> EncodingBlock:
        return EncodingBlock(
            block_id=f"{pair_id}:{block_type}:{event.event_id}:{idx}",
            block_type=block_type,
            text=(
                f"{event.direction}: predicate={event.predicate} "
                f"source_role={event.source_role} target_role={event.target_role} "
                f"surfaces={','.join(event.surface_texts)}"
            ),
            evidence_span_ids=event.evidence_span_ids,
            source_chunk_ids=event.source_chunk_ids,
            metadata={"event_id": event.event_id, "direction": event.direction},
        )
