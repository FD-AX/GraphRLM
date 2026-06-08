from __future__ import annotations

import hashlib
import json

from app.projection.models import EntityContextSnapshot, EntityPairContextSnapshot
from app.semantic_encoding.models import GraphSemanticDocument


def _hash_payload(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class GraphSemanticDocumentBuilder:
    def build_from_entity_snapshot(
        self,
        snapshot: EntityContextSnapshot,
    ) -> list[GraphSemanticDocument]:
        documents = [self._entity_identity_document(snapshot)]
        for index, observation in enumerate(snapshot.event_observations):
            documents.append(
                GraphSemanticDocument(
                    document_id=snapshot.document_id,
                    semantic_document_id=_stable_id(
                        "semantic_doc",
                        snapshot.snapshot_id,
                        "entity_event",
                        observation.event_id,
                        index,
                    ),
                    owner_type="entity_event",
                    owner_id=f"{snapshot.entity_id}:{observation.event_id}",
                    source_entity_ids=[snapshot.entity_id, *observation.counterpart_entity_ids],
                    event_ids=[observation.event_id],
                    evidence_span_ids=observation.evidence_span_ids,
                    source_chunk_ids=observation.source_chunk_ids,
                    text=(
                        f"{snapshot.canonical_name} participated as "
                        f"{observation.entity_role} in event {observation.predicate}. "
                        f"Surface: {observation.surface_text}. "
                        f"Counterpart entities: {', '.join(observation.counterpart_entity_ids)}. "
                        f"Unresolved/concept counterparts: "
                        f"{', '.join(observation.unresolved_counterparts)}."
                    ),
                    structural_features={
                        "predicate": observation.predicate,
                        "entity_role": observation.entity_role,
                        "counterpart_count": len(observation.counterpart_entity_ids),
                        "unresolved_count": len(observation.unresolved_counterparts),
                        "event_resolution_status": observation.event_resolution_status,
                    },
                    projection_version=snapshot.projection_version,
                    content_hash=_hash_payload(observation.model_dump(mode="json")),
                )
            )
            if observation.unresolved_counterparts:
                documents.append(
                    GraphSemanticDocument(
                        document_id=snapshot.document_id,
                        semantic_document_id=_stable_id(
                            "semantic_doc",
                            snapshot.snapshot_id,
                            "unresolved_context",
                            observation.event_id,
                            index,
                        ),
                        owner_type="evidence",
                        owner_id=f"{snapshot.entity_id}:unresolved:{observation.event_id}",
                        source_entity_ids=[snapshot.entity_id],
                        event_ids=[observation.event_id],
                        evidence_span_ids=observation.evidence_span_ids,
                        source_chunk_ids=observation.source_chunk_ids,
                        text=(
                            f"{snapshot.canonical_name} event {observation.predicate} "
                            f"has unresolved or concept counterparts: "
                            f"{', '.join(observation.unresolved_counterparts)}."
                        ),
                        structural_features={
                            "predicate": observation.predicate,
                            "document_kind": "unresolved_context",
                        },
                        projection_version=snapshot.projection_version,
                        content_hash=_hash_payload(
                            {
                                "event_id": observation.event_id,
                                "unresolved": observation.unresolved_counterparts,
                            }
                        ),
                    )
                )
        for index, value in enumerate(snapshot.states):
            documents.append(
                GraphSemanticDocument(
                    document_id=snapshot.document_id,
                    semantic_document_id=_stable_id(
                        "semantic_doc",
                        snapshot.snapshot_id,
                        "entity_state",
                        index,
                    ),
                    owner_type="entity_state",
                    owner_id=f"{snapshot.entity_id}:state:{index}",
                    source_entity_ids=[snapshot.entity_id],
                    text=f"{snapshot.canonical_name} state: {value}",
                    structural_features={"document_kind": "entity_state"},
                    projection_version=snapshot.projection_version,
                    content_hash=_hash_payload({"state": value}),
                )
            )
        return documents

    def build_from_pair_snapshot(
        self,
        snapshot: EntityPairContextSnapshot,
    ) -> list[GraphSemanticDocument]:
        documents = [self._pair_aggregate_document(snapshot)]
        interactions = (
            snapshot.direct_shared_events
            + snapshot.source_to_target_events
            + snapshot.target_to_source_events
        )
        for index, interaction in enumerate(interactions):
            documents.append(
                GraphSemanticDocument(
                    document_id=snapshot.document_id,
                    semantic_document_id=_stable_id(
                        "semantic_doc",
                        snapshot.snapshot_id,
                        "pair_interaction",
                        interaction.event_id,
                        index,
                    ),
                    owner_type="pair_interaction",
                    owner_id=f"{snapshot.pair_id}:{interaction.event_id}",
                    source_entity_ids=[
                        snapshot.source_entity_id,
                        snapshot.target_entity_id,
                    ],
                    event_ids=[interaction.event_id],
                    evidence_span_ids=interaction.evidence_span_ids,
                    source_chunk_ids=interaction.source_chunk_ids,
                    text=(
                        f"Pair interaction {snapshot.source_entity_id} -> "
                        f"{snapshot.target_entity_id}. Predicate: {interaction.predicate}. "
                        f"Direction: {interaction.direction}. "
                        f"Roles: {interaction.source_role} -> {interaction.target_role}. "
                        f"Surfaces: {', '.join(interaction.surface_texts)}."
                    ),
                    structural_features={
                        "direction": interaction.direction,
                        "predicate": interaction.predicate,
                        "source_role": interaction.source_role,
                        "target_role": interaction.target_role,
                    },
                    projection_version=snapshot.projection_version,
                    content_hash=_hash_payload(interaction.model_dump(mode="json")),
                )
            )
        return documents

    def _entity_identity_document(
        self,
        snapshot: EntityContextSnapshot,
    ) -> GraphSemanticDocument:
        payload = {
            "entity_id": snapshot.entity_id,
            "canonical_name": snapshot.canonical_name,
            "aliases": snapshot.alias_surfaces,
            "attributes": snapshot.attributes,
        }
        return GraphSemanticDocument(
            document_id=snapshot.document_id,
            semantic_document_id=_stable_id(
                "semantic_doc",
                snapshot.snapshot_id,
                "entity_identity",
            ),
            owner_type="entity",
            owner_id=snapshot.entity_id,
            source_entity_ids=[snapshot.entity_id],
            event_ids=[item.event_id for item in snapshot.event_observations],
            evidence_span_ids=snapshot.evidence_span_ids,
            source_chunk_ids=snapshot.source_chunk_ids,
            text=(
                f"Entity profile {snapshot.canonical_name}. "
                f"Aliases: {', '.join(snapshot.alias_surfaces)}. "
                f"Attributes/states: {', '.join(snapshot.attributes + snapshot.states)}."
            ),
            structural_features={
                "document_kind": "entity_identity",
                "event_count": len(snapshot.event_observations),
                "alias_count": len(snapshot.alias_surfaces),
            },
            projection_version=snapshot.projection_version,
            content_hash=_hash_payload(payload),
        )

    def _pair_aggregate_document(
        self,
        snapshot: EntityPairContextSnapshot,
    ) -> GraphSemanticDocument:
        interactions = (
            snapshot.direct_shared_events
            + snapshot.source_to_target_events
            + snapshot.target_to_source_events
        )
        payload = {
            "pair_id": snapshot.pair_id,
            "interactions": [item.model_dump(mode="json") for item in interactions],
        }
        return GraphSemanticDocument(
            document_id=snapshot.document_id,
            semantic_document_id=_stable_id(
                "semantic_doc",
                snapshot.snapshot_id,
                "pair_aggregate",
            ),
            owner_type="entity_pair",
            owner_id=snapshot.pair_id,
            source_entity_ids=[
                snapshot.source_entity_id,
                snapshot.target_entity_id,
            ],
            event_ids=[item.event_id for item in interactions],
            evidence_span_ids=snapshot.relation_evidence_span_ids,
            source_chunk_ids=[
                chunk_id
                for item in interactions
                for chunk_id in item.source_chunk_ids
            ],
            text=(
                f"Pair aggregate {snapshot.source_entity_id} and "
                f"{snapshot.target_entity_id}. "
                + " ".join(
                    f"{item.direction} {item.predicate} roles "
                    f"{item.source_role}->{item.target_role}."
                    for item in interactions
                )
            ),
            structural_features={
                "document_kind": "pair_aggregate",
                "interaction_count": len(interactions),
                "source_roles": snapshot.source_roles,
                "target_roles": snapshot.target_roles,
            },
            projection_version=snapshot.projection_version,
            content_hash=_hash_payload(payload),
        )
