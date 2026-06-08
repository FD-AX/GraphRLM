from __future__ import annotations

from itertools import combinations

from app.graph.neo4j_client import Neo4jClient
from app.projection.entity_context import EntityContextProjectionBuilder
from app.projection.entity_pair_context import EntityPairContextProjectionBuilder
from app.projection.materializer import ProjectionMaterializer
from app.projection.models import EntityContextSnapshot, EntityPairContextSnapshot


def build_entity_snapshot(
    client: Neo4jClient,
    document_id: str,
    entity_id: str,
    projection_version: str,
) -> EntityContextSnapshot:
    return EntityContextProjectionBuilder(client).build_entity_snapshot(
        document_id=document_id,
        entity_id=entity_id,
        projection_version=projection_version,
    )


def build_entity_pair_snapshot(
    client: Neo4jClient,
    document_id: str,
    source_entity_id: str,
    target_entity_id: str,
    projection_version: str,
) -> EntityPairContextSnapshot:
    return EntityPairContextProjectionBuilder(client).build_entity_pair_snapshot(
        document_id=document_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        projection_version=projection_version,
    )


def rebuild_document_snapshots(
    client: Neo4jClient,
    document_id: str,
    projection_version: str,
    materialize: bool = True,
) -> tuple[list[EntityContextSnapshot], list[EntityPairContextSnapshot]]:
    entity_builder = EntityContextProjectionBuilder(client)
    pair_builder = EntityPairContextProjectionBuilder(client)
    materializer = ProjectionMaterializer(client)

    entity_snapshots = entity_builder.rebuild_document_snapshots(
        document_id=document_id,
        projection_version=projection_version,
    )
    entity_ids = [snapshot.entity_id for snapshot in entity_snapshots]
    pair_snapshots = []
    for source_entity_id, target_entity_id in combinations(entity_ids, 2):
        pair_snapshot = pair_builder.build_entity_pair_snapshot(
            document_id=document_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            projection_version=projection_version,
        )
        if (
            pair_snapshot.direct_shared_events
            or pair_snapshot.source_to_target_events
            or pair_snapshot.target_to_source_events
        ):
            pair_snapshots.append(pair_snapshot)

    if materialize:
        for snapshot in entity_snapshots:
            materializer.materialize_entity_snapshot(snapshot)
        for snapshot in pair_snapshots:
            materializer.materialize_entity_pair_snapshot(snapshot)

    return entity_snapshots, pair_snapshots
