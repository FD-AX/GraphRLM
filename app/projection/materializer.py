from __future__ import annotations

import json
from typing import Any

from app.graph.neo4j_client import Neo4jClient
from app.projection.models import EntityContextSnapshot, EntityPairContextSnapshot


class ProjectionMaterializer:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def materialize_entity_snapshot(self, snapshot: EntityContextSnapshot) -> None:
        self.client.execute_write(
            """
            MATCH (old:EntityContextSnapshot {
                document_id: $document_id,
                entity_id: $entity_id,
                projection_version: $projection_version,
                status: 'active'
            })
            SET old.status = 'deprecated',
                old.deprecated_at = datetime()
            """,
            {
                "document_id": snapshot.document_id,
                "entity_id": snapshot.entity_id,
                "projection_version": snapshot.projection_version,
            },
        )
        params = self._entity_params(snapshot)
        self.client.execute_write(
            """
            CREATE (s:EntityContextSnapshot {
                snapshot_id: $snapshot_id,
                document_id: $document_id,
                entity_id: $entity_id,
                canonical_name: $canonical_name,
                alias_surfaces: $alias_surfaces,
                related_entity_ids: $related_entity_ids,
                unresolved_counterparts: $unresolved_counterparts,
                evidence_span_ids: $evidence_span_ids,
                source_chunk_ids: $source_chunk_ids,
                projection_version: $projection_version,
                extractor_versions: $extractor_versions,
                resolver_versions: $resolver_versions,
                content_hash: $content_hash,
                status: 'active',
                event_observations_json: $event_observations_json,
                attributes_json: $attributes_json,
                states_json: $states_json,
                encoding_input_json: $encoding_input_json,
                created_at: datetime($created_at)
            })
            WITH s
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (s)-[:FOR_ENTITY]->(e)
            """,
            params,
        )

    def materialize_entity_pair_snapshot(
        self,
        snapshot: EntityPairContextSnapshot,
    ) -> None:
        self.client.execute_write(
            """
            MATCH (old:EntityPairContextSnapshot {
                document_id: $document_id,
                source_entity_id: $source_entity_id,
                target_entity_id: $target_entity_id,
                projection_version: $projection_version,
                status: 'active'
            })
            SET old.status = 'deprecated',
                old.deprecated_at = datetime()
            """,
            {
                "document_id": snapshot.document_id,
                "source_entity_id": snapshot.source_entity_id,
                "target_entity_id": snapshot.target_entity_id,
                "projection_version": snapshot.projection_version,
            },
        )
        params = self._pair_params(snapshot)
        self.client.execute_write(
            """
            CREATE (s:EntityPairContextSnapshot {
                snapshot_id: $snapshot_id,
                pair_id: $pair_id,
                document_id: $document_id,
                source_entity_id: $source_entity_id,
                target_entity_id: $target_entity_id,
                relation_evidence_span_ids: $relation_evidence_span_ids,
                source_roles: $source_roles,
                target_roles: $target_roles,
                shared_locations: $shared_locations,
                shared_time_scopes: $shared_time_scopes,
                projection_version: $projection_version,
                content_hash: $content_hash,
                status: 'active',
                direct_shared_events_json: $direct_shared_events_json,
                source_to_target_events_json: $source_to_target_events_json,
                target_to_source_events_json: $target_to_source_events_json,
                indirect_graph_paths_json: $indirect_graph_paths_json,
                encoding_input_json: $encoding_input_json,
                created_at: datetime($created_at)
            })
            WITH s
            MATCH (source:Entity {entity_id: $source_entity_id})
            MATCH (target:Entity {entity_id: $target_entity_id})
            MERGE (s)-[:SOURCE_ENTITY]->(source)
            MERGE (s)-[:TARGET_ENTITY]->(target)
            """,
            params,
        )

    def read_active_entity_snapshot(
        self,
        document_id: str,
        entity_id: str,
        projection_version: str,
    ) -> dict | None:
        rows = self.client.execute_read(
            """
            MATCH (s:EntityContextSnapshot {
                document_id: $document_id,
                entity_id: $entity_id,
                projection_version: $projection_version,
                status: 'active'
            })-[:FOR_ENTITY]->(e:Entity {entity_id: $entity_id})
            RETURN s {
                .snapshot_id,
                .document_id,
                .entity_id,
                .canonical_name,
                .alias_surfaces,
                .related_entity_ids,
                .unresolved_counterparts,
                .evidence_span_ids,
                .source_chunk_ids,
                .projection_version,
                .extractor_versions,
                .resolver_versions,
                .content_hash,
                .status,
                .event_observations_json,
                .encoding_input_json,
                .created_at
            } AS snapshot
            ORDER BY s.created_at DESC
            LIMIT 1
            """,
            {
                "document_id": document_id,
                "entity_id": entity_id,
                "projection_version": projection_version,
            },
        )
        return self._decode_json_fields(rows[0]["snapshot"]) if rows else None

    def read_active_pair_snapshot(
        self,
        document_id: str,
        source_entity_id: str,
        target_entity_id: str,
        projection_version: str,
    ) -> dict | None:
        rows = self.client.execute_read(
            """
            MATCH (s:EntityPairContextSnapshot {
                document_id: $document_id,
                source_entity_id: $source_entity_id,
                target_entity_id: $target_entity_id,
                projection_version: $projection_version,
                status: 'active'
            })
            RETURN s {
                .snapshot_id,
                .pair_id,
                .document_id,
                .source_entity_id,
                .target_entity_id,
                .relation_evidence_span_ids,
                .source_roles,
                .target_roles,
                .shared_locations,
                .shared_time_scopes,
                .projection_version,
                .content_hash,
                .status,
                .direct_shared_events_json,
                .source_to_target_events_json,
                .target_to_source_events_json,
                .indirect_graph_paths_json,
                .encoding_input_json,
                .created_at
            } AS snapshot
            ORDER BY s.created_at DESC
            LIMIT 1
            """,
            {
                "document_id": document_id,
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "projection_version": projection_version,
            },
        )
        return self._decode_json_fields(rows[0]["snapshot"]) if rows else None

    def _entity_params(self, snapshot: EntityContextSnapshot) -> dict[str, Any]:
        return {
            "snapshot_id": snapshot.snapshot_id,
            "document_id": snapshot.document_id,
            "entity_id": snapshot.entity_id,
            "canonical_name": snapshot.canonical_name,
            "alias_surfaces": snapshot.alias_surfaces,
            "related_entity_ids": snapshot.related_entity_ids,
            "unresolved_counterparts": snapshot.unresolved_counterparts,
            "evidence_span_ids": snapshot.evidence_span_ids,
            "source_chunk_ids": snapshot.source_chunk_ids,
            "projection_version": snapshot.projection_version,
            "extractor_versions": snapshot.extractor_versions,
            "resolver_versions": snapshot.resolver_versions,
            "content_hash": snapshot.content_hash,
            "event_observations_json": json.dumps(
                [item.model_dump(mode="json") for item in snapshot.event_observations],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "attributes_json": json.dumps(snapshot.attributes, ensure_ascii=False),
            "states_json": json.dumps(snapshot.states, ensure_ascii=False),
            "encoding_input_json": json.dumps(
                snapshot.encoding_input.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "created_at": snapshot.created_at.isoformat(),
        }

    def _pair_params(self, snapshot: EntityPairContextSnapshot) -> dict[str, Any]:
        return {
            "snapshot_id": snapshot.snapshot_id,
            "pair_id": snapshot.pair_id,
            "document_id": snapshot.document_id,
            "source_entity_id": snapshot.source_entity_id,
            "target_entity_id": snapshot.target_entity_id,
            "relation_evidence_span_ids": snapshot.relation_evidence_span_ids,
            "source_roles": snapshot.source_roles,
            "target_roles": snapshot.target_roles,
            "shared_locations": snapshot.shared_locations,
            "shared_time_scopes": snapshot.shared_time_scopes,
            "projection_version": snapshot.projection_version,
            "content_hash": snapshot.content_hash,
            "direct_shared_events_json": json.dumps(
                [item.model_dump(mode="json") for item in snapshot.direct_shared_events],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "source_to_target_events_json": json.dumps(
                [
                    item.model_dump(mode="json")
                    for item in snapshot.source_to_target_events
                ],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "target_to_source_events_json": json.dumps(
                [
                    item.model_dump(mode="json")
                    for item in snapshot.target_to_source_events
                ],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "indirect_graph_paths_json": json.dumps(
                snapshot.indirect_graph_paths,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "encoding_input_json": json.dumps(
                snapshot.encoding_input.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "created_at": snapshot.created_at.isoformat(),
        }

    def _decode_json_fields(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        result = dict(snapshot)
        for key in list(result):
            if key.endswith("_json") and isinstance(result[key], str):
                result[key.removesuffix("_json")] = json.loads(result[key])
        return result
