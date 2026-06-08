from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from app.graph.neo4j_client import Neo4jClient
from app.projection.models import (
    EncodingBlock,
    EntityContextSnapshot,
    EntityEncodingInput,
    ProjectionEventObservation,
)


CANONICAL_HYPOTHESIS_STATUSES = {"confirmed", "likely"}
WEAK_UNRESOLVED_DECISIONS = {"link_to_existing", "create_entity"}


def _unique(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _snapshot_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class EntityContextProjectionBuilder:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def build_entity_snapshot(
        self,
        document_id: str,
        entity_id: str,
        projection_version: str,
    ) -> EntityContextSnapshot:
        entity = self._get_document_entity(document_id, entity_id)
        if entity is None:
            raise ValueError(
                f"Entity {entity_id!r} is not materialized in document {document_id!r}"
            )

        mentions = self._get_canonical_mentions(document_id, entity_id)
        observations = self._get_event_observations(document_id, entity_id)
        states = self._get_states(document_id, entity_id)
        attributes = self._get_attributes(entity, states)

        related_entity_ids = _unique(
            [
                counterpart
                for observation in observations
                for counterpart in observation.counterpart_entity_ids
            ]
        )
        unresolved_counterparts = _unique(
            [
                counterpart
                for observation in observations
                for counterpart in observation.unresolved_counterparts
            ]
        )
        evidence_span_ids = _unique(
            [
                evidence_span_id
                for observation in observations
                for evidence_span_id in observation.evidence_span_ids
            ]
        )
        source_chunk_ids = _unique(
            [
                chunk_id
                for observation in observations
                for chunk_id in observation.source_chunk_ids
            ]
            + [mention["chunk_id"] for mention in mentions]
        )
        extractor_versions = _unique(
            [mention.get("extractor_version") for mention in mentions]
            + [observation.extractor_version for observation in observations]
        )
        resolver_versions = _unique(
            [mention.get("resolver_version") for mention in mentions]
            + [observation.resolver_version for observation in observations]
        )
        alias_surfaces = self._alias_surfaces(entity, mentions)

        encoding_input = self._build_encoding_input(
            document_id=document_id,
            entity_id=entity_id,
            projection_version=projection_version,
            canonical_name=entity["canonical_name"],
            alias_surfaces=alias_surfaces,
            observations=observations,
            attributes=attributes,
            states=states,
        )
        hash_payload = {
            "entity_id": entity_id,
            "canonical_name": entity["canonical_name"],
            "aliases": alias_surfaces,
            "events": [observation.model_dump(mode="json") for observation in observations],
            "attributes": attributes,
            "states": states,
            "related_entity_ids": related_entity_ids,
            "unresolved_counterparts": unresolved_counterparts,
            "evidence_span_ids": evidence_span_ids,
            "source_chunk_ids": source_chunk_ids,
            "projection_version": projection_version,
            "encoding": encoding_input.model_dump(mode="json"),
        }
        content_hash = _content_hash(hash_payload)
        return EntityContextSnapshot(
            snapshot_id=_snapshot_id(
                "entity_context_snapshot",
                document_id,
                entity_id,
                projection_version,
                content_hash,
                datetime.utcnow().isoformat(),
            ),
            document_id=document_id,
            entity_id=entity_id,
            canonical_name=entity["canonical_name"],
            alias_surfaces=alias_surfaces,
            event_observations=observations,
            attributes=attributes,
            states=states,
            related_entity_ids=related_entity_ids,
            unresolved_counterparts=unresolved_counterparts,
            evidence_span_ids=evidence_span_ids,
            source_chunk_ids=source_chunk_ids,
            projection_version=projection_version,
            extractor_versions=extractor_versions,
            resolver_versions=resolver_versions,
            encoding_input=encoding_input,
            content_hash=content_hash,
        )

    def rebuild_document_snapshots(
        self,
        document_id: str,
        projection_version: str,
    ) -> list[EntityContextSnapshot]:
        entity_ids = self.client.execute_read(
            """
            MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (arg:EventArgument)-[:ARGUMENT_OF]->(:EventFrame {document_id: $document_id})
            MATCH (arg)-[:RESOLVED_TO]->(e:Entity)
            RETURN DISTINCT e.entity_id AS entity_id
            UNION
            MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[:HAS_MENTION]->(:Mention)-[:REFERS_TO]->(e:Entity)
            RETURN DISTINCT e.entity_id AS entity_id
            ORDER BY entity_id
            """,
            {"document_id": document_id},
        )
        return [
            self.build_entity_snapshot(
                document_id=document_id,
                entity_id=row["entity_id"],
                projection_version=projection_version,
            )
            for row in entity_ids
        ]

    def _get_document_entity(self, document_id: str, entity_id: str) -> dict | None:
        rows = self.client.execute_read(
            """
            MATCH (e:Entity {entity_id: $entity_id})
            WHERE EXISTS {
                MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
                MATCH (c)-[:HAS_MENTION]->(:Mention)-[:REFERS_TO]->(e)
            } OR EXISTS {
                MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(:Chunk)
                MATCH (:EventArgument)-[:RESOLVED_TO]->(e)
                MATCH (:EventArgument {entity_id: $entity_id})-[:ARGUMENT_OF]->(:EventFrame {document_id: $document_id})
            }
            RETURN e {
                .entity_id,
                .canonical_name,
                .entity_type,
                .aliases,
                .description
            } AS entity
            LIMIT 1
            """,
            {"document_id": document_id, "entity_id": entity_id},
        )
        return rows[0]["entity"] if rows else None

    def _get_canonical_mentions(self, document_id: str, entity_id: str) -> list[dict]:
        return self.client.execute_read(
            """
            MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[:HAS_MENTION]->(m:RawMention)
            MATCH (m)-[:HAS_RESOLUTION_HYPOTHESIS]->(h:ResolutionHypothesis)
            WHERE h.final_entity_id = $entity_id
              AND coalesce(h.status, '') IN $canonical_statuses
              AND coalesce(h.entity_creation_decision, '') <> 'rejected'
            RETURN m.mention_id AS mention_id,
                   m.text AS text,
                   m.normalized_text AS normalized_text,
                   m.mention_type AS mention_type,
                   m.mention_kind AS mention_kind,
                   m.extractor_version AS extractor_version,
                   h.resolver_version AS resolver_version,
                   h.evidence_span_id AS evidence_span_id,
                   c.chunk_id AS chunk_id,
                   c.index AS chunk_index
            ORDER BY c.index ASC, m.start_char ASC
            """,
            {
                "document_id": document_id,
                "entity_id": entity_id,
                "canonical_statuses": sorted(CANONICAL_HYPOTHESIS_STATUSES),
            },
        )

    def _get_event_observations(
        self,
        document_id: str,
        entity_id: str,
    ) -> list[ProjectionEventObservation]:
        rows = self.client.execute_read(
            """
            MATCH (self_arg:EventArgument)-[:RESOLVED_TO]->(:Entity {entity_id: $entity_id})
            MATCH (self_arg)-[:ARGUMENT_OF]->(ev:EventFrame {document_id: $document_id})
            MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)-[:HAS_EVENT]->(ev)
            OPTIONAL MATCH (ev)-[:SUPPORTED_BY]->(span:EvidenceSpan)
            OPTIONAL MATCH (other_arg:EventArgument)-[:ARGUMENT_OF]->(ev)
            WHERE other_arg.argument_id <> self_arg.argument_id
            OPTIONAL MATCH (other_arg)-[:RESOLVED_TO]->(other:Entity)
            WITH ev, c, span, self_arg,
                 collect(DISTINCT other.entity_id) AS counterpart_entity_ids,
                 collect(DISTINCT CASE
                   WHEN other IS NULL
                    AND other_arg.surface_text IS NOT NULL
                    AND coalesce(other_arg.resolution_status, '') <> 'resolved'
                   THEN other_arg.surface_text
                 END) AS unresolved_counterparts
            RETURN ev.event_frame_id AS event_id,
                   coalesce(ev.normalized_predicate, ev.predicate) AS predicate,
                   self_arg.role AS entity_role,
                   self_arg.surface_text AS surface_text,
                   counterpart_entity_ids AS counterpart_entity_ids,
                   unresolved_counterparts AS unresolved_counterparts,
                   ev.temporal_scope AS temporal_scope,
                   ev.modality AS modality,
                   ev.polarity AS polarity,
                   CASE WHEN span.span_id IS NULL THEN [] ELSE [span.span_id] END AS evidence_span_ids,
                   [c.chunk_id] AS source_chunk_ids,
                   ev.resolution_status AS event_resolution_status,
                   ev.materialization_status AS event_materialization_status,
                   coalesce(self_arg.extractor_version, ev.extractor_version) AS extractor_version,
                   coalesce(self_arg.resolver_version, ev.resolver_version) AS resolver_version
            ORDER BY c.index ASC, ev.event_frame_id ASC
            """,
            {"document_id": document_id, "entity_id": entity_id},
        )
        return [
            ProjectionEventObservation(
                event_id=row["event_id"],
                predicate=row["predicate"],
                entity_role=row["entity_role"],
                surface_text=row.get("surface_text"),
                counterpart_entity_ids=_unique(row.get("counterpart_entity_ids") or []),
                unresolved_counterparts=_unique(row.get("unresolved_counterparts") or []),
                temporal_scope=row.get("temporal_scope"),
                modality=row.get("modality"),
                polarity=row.get("polarity") or "positive",
                evidence_span_ids=row.get("evidence_span_ids") or [],
                source_chunk_ids=row.get("source_chunk_ids") or [],
                event_resolution_status=row.get("event_resolution_status"),
                event_materialization_status=row.get("event_materialization_status"),
                extractor_version=row.get("extractor_version"),
                resolver_version=row.get("resolver_version"),
            )
            for row in rows
        ]

    def _get_states(self, document_id: str, entity_id: str) -> list[str]:
        rows = self.client.execute_read(
            """
            MATCH (s:EntityState)
            WHERE s.entity_id = $entity_id OR EXISTS {
                MATCH (s)-[:STATE_OF]->(:Entity {entity_id: $entity_id})
            }
            OPTIONAL MATCH (t:RLMTransition {transition_id: s.transition_id})
            WHERE t.document_id = $document_id OR s.document_id = $document_id
            RETURN DISTINCT coalesce(s.state, s.description, s.canonical_name) AS state
            ORDER BY state
            """,
            {"document_id": document_id, "entity_id": entity_id},
        )
        return _unique([row.get("state") for row in rows])

    def _get_attributes(self, entity: dict, states: list[str]) -> list[str]:
        attributes = []
        if entity.get("entity_type"):
            attributes.append(f"type={entity['entity_type']}")
        if entity.get("description"):
            attributes.append(entity["description"])
        attributes.extend(states)
        return _unique(attributes)

    def _alias_surfaces(self, entity: dict, mentions: list[dict]) -> list[str]:
        aliases = [entity["canonical_name"]]
        aliases.extend(entity.get("aliases") or [])
        aliases.extend(mention["text"] for mention in mentions)
        return _unique(aliases)

    def _build_encoding_input(
        self,
        document_id: str,
        entity_id: str,
        projection_version: str,
        canonical_name: str,
        alias_surfaces: list[str],
        observations: list[ProjectionEventObservation],
        attributes: list[str],
        states: list[str],
    ) -> EntityEncodingInput:
        identity = EncodingBlock(
            block_id=f"{entity_id}:identity",
            block_type="identity_alias",
            text="entity: "
            + canonical_name
            + (" aliases: " + ", ".join(alias_surfaces) if alias_surfaces else ""),
        )
        event_blocks = [
            EncodingBlock(
                block_id=f"{entity_id}:event:{observation.event_id}:{idx}",
                block_type="event",
                text=(
                    f"{canonical_name} role={observation.entity_role} "
                    f"predicate={observation.predicate} "
                    f"surface={observation.surface_text or canonical_name} "
                    f"counterparts={','.join(observation.counterpart_entity_ids)} "
                    f"unresolved={','.join(observation.unresolved_counterparts)}"
                ),
                evidence_span_ids=observation.evidence_span_ids,
                source_chunk_ids=observation.source_chunk_ids,
                metadata={"event_id": observation.event_id},
            )
            for idx, observation in enumerate(observations)
        ]
        state_attribute_blocks = [
            EncodingBlock(
                block_id=f"{entity_id}:attribute:{idx}",
                block_type="state_attribute",
                text=f"{canonical_name} attribute/state: {value}",
            )
            for idx, value in enumerate(_unique(attributes + states))
        ]
        temporal_blocks = [
            EncodingBlock(
                block_id=f"{entity_id}:temporal:{idx}",
                block_type="temporal",
                text=f"{canonical_name} temporal={observation.temporal_scope}",
                evidence_span_ids=observation.evidence_span_ids,
                source_chunk_ids=observation.source_chunk_ids,
            )
            for idx, observation in enumerate(observations)
            if observation.temporal_scope
        ]
        evidence_blocks = [
            EncodingBlock(
                block_id=f"{entity_id}:evidence:{idx}",
                block_type="evidence",
                text=f"{canonical_name} evidence_span={span_id}",
                evidence_span_ids=[span_id],
            )
            for idx, span_id in enumerate(
                _unique(
                    [
                        span_id
                        for observation in observations
                        for span_id in observation.evidence_span_ids
                    ]
                )
            )
        ]
        return EntityEncodingInput(
            entity_id=entity_id,
            document_id=document_id,
            projection_version=projection_version,
            identity_blocks=[identity],
            event_blocks=event_blocks,
            state_attribute_blocks=state_attribute_blocks,
            temporal_blocks=temporal_blocks,
            evidence_blocks=evidence_blocks,
        )
