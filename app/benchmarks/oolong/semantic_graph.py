from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime

from app.benchmarks.models import BenchmarkCase
from app.benchmarks.oolong.operations import OOLONGRecordFact, build_record_fact, normalize_user_id
from app.projection.models import (
    EncodingBlock,
    EntityContextSnapshot,
    EntityEncodingInput,
    EntityPairContextSnapshot,
    EntityPairEncodingInput,
    PairEventLink,
    ProjectionEventObservation,
)
from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticDocument,
    GraphSemanticEncoder,
    GraphSemanticEmbedding,
    GraphSemanticIndex,
    HashingSemanticEncoder,
    LatentGraphNavigator,
    SemanticTraversalTrace,
    TransformerSemanticEncoder,
    build_graph_semantic_documents,
)


@dataclass(frozen=True)
class OOLONGRecordObservation:
    record_id: str
    record_index: int
    user_id: str
    canonical_user_id: str
    date: str
    instance_text: str
    evidence_span_id: str
    source_chunk_id: str
    predicted_label: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class OOLONGSemanticGraphBuild:
    document_id: str
    records: list[OOLONGRecordObservation]
    record_facts: list[OOLONGRecordFact]
    entity_snapshots: list[EntityContextSnapshot]
    pair_snapshots: list[EntityPairContextSnapshot]
    semantic_documents: list[GraphSemanticDocument]
    embeddings: list[GraphSemanticEmbedding]
    index: GraphSemanticIndex
    encoder: GraphSemanticEncoder
    traversal_trace: SemanticTraversalTrace
    validation: dict


def build_oolong_semantic_graph(
    case: BenchmarkCase,
    *,
    record_labels: dict[int, str] | None = None,
    encoder_backend: str = "transformer",
    projection_version: str = "oolong_semantic_projection_v1",
    encoder_config: EncoderConfig | None = None,
) -> OOLONGSemanticGraphBuild:
    record_labels = record_labels or {}
    document_id = f"oolong_semantic_graph_{case.task_id}"
    records = _extract_record_observations(case, record_labels)
    record_facts = _build_record_facts(records)
    entity_snapshots = _build_entity_snapshots(
        case,
        document_id=document_id,
        records=records,
        projection_version=projection_version,
    )
    pair_snapshots = _build_pair_snapshots(
        case,
        document_id=document_id,
        records=records,
        projection_version=projection_version,
    )
    semantic_documents = build_graph_semantic_documents(entity_snapshots, pair_snapshots)
    semantic_documents.extend(
        _record_fact_documents(
            document_id=document_id,
            record_facts=record_facts,
            projection_version=projection_version,
        )
    )
    encoder = _build_encoder(encoder_backend, encoder_config)
    embeddings = [encoder.encode_document(document) for document in semantic_documents]
    index = GraphSemanticIndex(
        documents=semantic_documents,
        embeddings=embeddings,
        encoder=encoder,
    )
    traversal_trace = LatentGraphNavigator(index, encoder, encoder.config).traverse(
        query=case.question,
        seed_top_k=5,
        max_depth=min(2, encoder.config.max_graph_depth),
        beam_width=min(2, encoder.config.beam_width),
    )
    validation = validate_oolong_semantic_graph(case, records, record_facts, semantic_documents, embeddings)
    return OOLONGSemanticGraphBuild(
        document_id=document_id,
        records=records,
        record_facts=record_facts,
        entity_snapshots=entity_snapshots,
        pair_snapshots=pair_snapshots,
        semantic_documents=semantic_documents,
        embeddings=embeddings,
        index=index,
        encoder=encoder,
        traversal_trace=traversal_trace,
        validation=validation,
    )


def validate_oolong_semantic_graph(
    case: BenchmarkCase,
    records: list[OOLONGRecordObservation],
    record_facts: list[OOLONGRecordFact],
    semantic_documents: list[GraphSemanticDocument],
    embeddings: list[GraphSemanticEmbedding],
) -> dict:
    native = case.metadata.get("native_fields", {})
    serialized_docs = "\n".join(document.text for document in semantic_documents)
    leaked_labelled_context = bool(native.get("context_window_text_with_labels")) and (
        str(native.get("context_window_text_with_labels")) in serialized_docs
    )
    aggregate_docs = [
        document for document in semantic_documents
        if document.structural_features.get("document_kind") == "entity_identity"
        and "aggregate" in document.owner_id
    ]
    return {
        "graph_node_count": len(semantic_documents),
        "embedding_count": len(embeddings),
        "record_count": len(records),
        "record_fact_count": len(record_facts),
        "canonical_user_id_count": len({fact.user_id for fact in record_facts}),
        "timestamp_normalized_count": sum(1 for fact in record_facts if fact.occurred_at is not None),
        "record_fact_label_count": sum(1 for fact in record_facts if fact.label is not None),
        "record_fact_candidate_count": sum(
            1
            for document in semantic_documents
            if document.owner_type == "record_fact"
        ),
        "stable_source_ids": all(record.record_id and record.evidence_span_id for record in records),
        "evidence_spans_reference_public_context": all(
            record.instance_text in case.context for record in records
        ),
        "labelled_context_leakage": leaked_labelled_context,
        "materialized_observations": sum(
            1
            for document in semantic_documents
            if document.owner_type in {"entity_event", "pair_interaction", "evidence"}
        ),
        "aggregate_projection_count": len(aggregate_docs),
        "encoder_name": embeddings[0].encoder_name if embeddings else None,
        "embedding_dim": embeddings[0].embedding_dim if embeddings else 0,
        "actual_embedding_dim": embeddings[0].actual_embedding_dim if embeddings else 0,
        "seed_strategy": "cosine_seed_structural_frontier",
        "frontier_strategy": "structural_overlap",
        "active_profile_weight": 0.0,
        "interaction_profile_enabled": False,
    }


def _build_encoder(
    encoder_backend: str,
    encoder_config: EncoderConfig | None,
) -> GraphSemanticEncoder:
    config = encoder_config or EncoderConfig(
        projection_dim=None if encoder_backend == "transformer" else 256,
        interaction_dim=64,
        local_frontier_cap=20,
    )
    if encoder_backend == "transformer":
        backend = TransformerSemanticEncoder(
            model_name=config.transformer_model_name,
            device=config.device,
        )
    elif encoder_backend == "hashing":
        backend = HashingSemanticEncoder(dimensions=config.projection_dim or 256)
    else:
        raise ValueError(f"Unsupported encoder_backend={encoder_backend!r}")
    return GraphSemanticEncoder(config=config, backend=backend)


def _record_fact_documents(
    *,
    document_id: str,
    record_facts: list[OOLONGRecordFact],
    projection_version: str,
) -> list[GraphSemanticDocument]:
    documents = []
    for fact in record_facts:
        date = fact.occurred_at.date().isoformat() if fact.occurred_at else "unknown"
        label = fact.label or "label_unresolved"
        text = (
            "TYPE=record "
            f"RECORD_ID={fact.record_id} "
            f"USER_ID={fact.user_id} "
            f"LABEL={label} "
            f"DATE={date}"
        )
        payload = {
            "candidate_type": "record_fact",
            "record_id": fact.record_id,
            "record_index": fact.record_index,
            "user_id": fact.user_id,
            "label": fact.label,
            "timestamp": fact.occurred_at.isoformat() if fact.occurred_at else None,
            "evidence_span_id": fact.evidence_span_id,
            "source_record_id": fact.source_chunk_id,
        }
        documents.append(
            GraphSemanticDocument(
                document_id=document_id,
                semantic_document_id=f"record_fact_{fact.record_id}",
                owner_type="record_fact",
                owner_id=fact.record_id,
                source_entity_ids=[
                    fact.record_id,
                    f"user_{fact.user_id}",
                    f"label_{label}",
                ],
                event_ids=[f"record_fact_event_{fact.record_id}"],
                evidence_span_ids=[fact.evidence_span_id],
                source_chunk_ids=[fact.source_chunk_id],
                text=text,
                structural_features=payload,
                projection_version=projection_version,
                content_hash=_hash_payload(payload),
            )
        )
    return documents


def _extract_record_observations(
    case: BenchmarkCase,
    record_labels: dict[int, str],
) -> list[OOLONGRecordObservation]:
    records = []
    for index, line in enumerate(_record_lines(case.context)):
        user_id = _extract_field(line, "User") or f"unknown_user_{index}"
        date = _extract_date(line) or "unknown_date"
        instance = _extract_field(line, "Instance") or line
        record_id = _stable_id("record", case.task_id, index, user_id, date, instance)
        records.append(
            OOLONGRecordObservation(
                record_id=record_id,
                record_index=index,
                user_id=f"user_{user_id}",
                canonical_user_id=normalize_user_id(user_id) or user_id,
                date=date,
                instance_text=instance,
                evidence_span_id=f"evidence_{record_id}",
                source_chunk_id=f"{case.task_id}:record:{index}",
                predicted_label=record_labels.get(index),
                confidence=1.0 if index in record_labels else None,
            )
        )
    return records


def _build_record_facts(records: list[OOLONGRecordObservation]) -> list[OOLONGRecordFact]:
    return [
        build_record_fact(
            record_id=record.record_id,
            record_index=record.record_index,
            user_id=record.user_id,
            label=record.predicted_label,
            date=record.date,
            evidence_span_id=record.evidence_span_id,
            source_chunk_id=record.source_chunk_id,
        )
        for record in records
    ]


def _build_entity_snapshots(
    case: BenchmarkCase,
    *,
    document_id: str,
    records: list[OOLONGRecordObservation],
    projection_version: str,
) -> list[EntityContextSnapshot]:
    snapshots = [
        _context_snapshot(case, document_id, records, projection_version),
        _aggregate_snapshot(case, document_id, records, projection_version),
    ]
    snapshots.extend(
        _record_snapshot(case, document_id, record, projection_version)
        for record in records
    )
    for label in sorted({record.predicted_label for record in records if record.predicted_label}):
        snapshots.append(_label_snapshot(case, document_id, label, records, projection_version))
    return snapshots


def _context_snapshot(
    case: BenchmarkCase,
    document_id: str,
    records: list[OOLONGRecordObservation],
    projection_version: str,
) -> EntityContextSnapshot:
    entity_id = f"context_window_{case.task_id}"
    observations = [
        ProjectionEventObservation(
            event_id=f"event_contains_{record.record_id}",
            predicate="contains_record",
            entity_role="container",
            surface_text=record.instance_text,
            counterpart_entity_ids=[record.record_id],
            evidence_span_ids=[record.evidence_span_id],
            source_chunk_ids=[record.source_chunk_id],
            event_resolution_status="complete",
            event_materialization_status="valid",
            extractor_version="oolong_record_observation_v1",
            resolver_version="deterministic_source_id_v1",
        )
        for record in records
    ]
    return _entity_snapshot(
        snapshot_id=f"snapshot_{entity_id}",
        entity_id=entity_id,
        document_id=document_id,
        canonical_name=f"OOLONG context window {case.task_id}",
        aliases=[case.task_id],
        observations=observations,
        related=[record.record_id for record in records],
        evidence=[record.evidence_span_id for record in records],
        chunks=[record.source_chunk_id for record in records],
        projection_version=projection_version,
    )


def _record_snapshot(
    case: BenchmarkCase,
    document_id: str,
    record: OOLONGRecordObservation,
    projection_version: str,
) -> EntityContextSnapshot:
    counterparts = [f"label_{record.predicted_label}"] if record.predicted_label else []
    unresolved = [] if record.predicted_label else ["label_unresolved"]
    observations = [
        ProjectionEventObservation(
            event_id=f"event_message_{record.record_id}",
            predicate="text_message_observed",
            entity_role="message",
            surface_text=record.instance_text,
            evidence_span_ids=[record.evidence_span_id],
            source_chunk_ids=[record.source_chunk_id],
            event_resolution_status="complete",
            event_materialization_status="valid",
            extractor_version="oolong_record_observation_v1",
            resolver_version="deterministic_source_id_v1",
        ),
        ProjectionEventObservation(
            event_id=f"event_label_{record.record_id}",
            predicate="has_predicted_label",
            entity_role="record",
            surface_text=record.predicted_label,
            counterpart_entity_ids=counterparts,
            unresolved_counterparts=unresolved,
            evidence_span_ids=[record.evidence_span_id],
            source_chunk_ids=[record.source_chunk_id],
            event_resolution_status="complete" if record.predicted_label else "unresolved",
            event_materialization_status="valid",
            extractor_version="oolong_label_worker_v1" if record.predicted_label else "none",
            resolver_version="label_projection_v1",
        ),
    ]
    return _entity_snapshot(
        snapshot_id=f"snapshot_{record.record_id}",
        entity_id=record.record_id,
        document_id=document_id,
        canonical_name=f"OOLONG record {record.record_index}",
        aliases=[record.record_id, record.user_id, record.date],
        observations=observations,
        related=counterparts,
        unresolved=unresolved,
        evidence=[record.evidence_span_id],
        chunks=[record.source_chunk_id],
        projection_version=projection_version,
        attributes=[
            f"user={record.user_id}",
            f"canonical_user_id={record.canonical_user_id}",
            f"date={record.date}",
        ],
    )


def _label_snapshot(
    case: BenchmarkCase,
    document_id: str,
    label: str,
    records: list[OOLONGRecordObservation],
    projection_version: str,
) -> EntityContextSnapshot:
    linked = [record for record in records if record.predicted_label == label]
    observations = [
        ProjectionEventObservation(
            event_id=f"event_label_observed_{record.record_id}",
            predicate="labels_record",
            entity_role="label",
            surface_text=label,
            counterpart_entity_ids=[record.record_id],
            evidence_span_ids=[record.evidence_span_id],
            source_chunk_ids=[record.source_chunk_id],
            event_resolution_status="complete",
            event_materialization_status="valid",
            extractor_version="oolong_label_worker_v1",
            resolver_version="label_projection_v1",
        )
        for record in linked
    ]
    return _entity_snapshot(
        snapshot_id=f"snapshot_label_{case.task_id}_{label}",
        entity_id=f"label_{label}",
        document_id=document_id,
        canonical_name=f"Label {label}",
        aliases=[label],
        observations=observations,
        related=[record.record_id for record in linked],
        evidence=[record.evidence_span_id for record in linked],
        chunks=[record.source_chunk_id for record in linked],
        projection_version=projection_version,
        attributes=[f"predicted_count={len(linked)}"],
    )


def _aggregate_snapshot(
    case: BenchmarkCase,
    document_id: str,
    records: list[OOLONGRecordObservation],
    projection_version: str,
) -> EntityContextSnapshot:
    counts = _label_counts(records)
    observations = [
        ProjectionEventObservation(
            event_id=f"event_aggregate_{case.task_id}_{label}",
            predicate="counts_predicted_label",
            entity_role="aggregate",
            surface_text=f"{label}={count}",
            counterpart_entity_ids=[f"label_{label}"],
            evidence_span_ids=[
                record.evidence_span_id
                for record in records
                if record.predicted_label == label
            ],
            source_chunk_ids=[
                record.source_chunk_id
                for record in records
                if record.predicted_label == label
            ],
            event_resolution_status="complete",
            event_materialization_status="valid",
            extractor_version="oolong_label_worker_v1",
            resolver_version="aggregate_projection_v1",
        )
        for label, count in counts.items()
    ]
    return _entity_snapshot(
        snapshot_id=f"snapshot_aggregate_{case.task_id}",
        entity_id=f"aggregate_{case.task_id}",
        document_id=document_id,
        canonical_name=f"OOLONG aggregate {case.task_id}",
        aliases=["label counts", "aggregate statistics"],
        observations=observations,
        related=[f"label_{label}" for label in counts],
        evidence=[record.evidence_span_id for record in records if record.predicted_label],
        chunks=[record.source_chunk_id for record in records if record.predicted_label],
        projection_version=projection_version,
        attributes=[f"{label}_count={count}" for label, count in counts.items()],
    )


def _build_pair_snapshots(
    case: BenchmarkCase,
    *,
    document_id: str,
    records: list[OOLONGRecordObservation],
    projection_version: str,
) -> list[EntityPairContextSnapshot]:
    pairs = []
    for record in records:
        if not record.predicted_label:
            continue
        pair_id = f"pair_{record.record_id}_label_{record.predicted_label}"
        link = PairEventLink(
            event_id=f"event_label_{record.record_id}",
            predicate="has_predicted_label",
            source_role="record",
            target_role="label",
            direction="source_to_target",
            evidence_span_ids=[record.evidence_span_id],
            source_chunk_ids=[record.source_chunk_id],
            surface_texts=[record.instance_text, record.predicted_label],
        )
        pairs.append(
            EntityPairContextSnapshot(
                pair_id=pair_id,
                snapshot_id=f"snapshot_{pair_id}",
                document_id=document_id,
                source_entity_id=record.record_id,
                target_entity_id=f"label_{record.predicted_label}",
                source_to_target_events=[link],
                relation_evidence_span_ids=[record.evidence_span_id],
                source_roles=["record"],
                target_roles=["label"],
                projection_version=projection_version,
                encoding_input=EntityPairEncodingInput(
                    pair_id=pair_id,
                    document_id=document_id,
                    projection_version=projection_version,
                    direct_interaction_blocks=[
                        EncodingBlock(
                            block_id=f"block_{pair_id}",
                            block_type="record_label",
                            text=f"{record.record_id} has predicted label {record.predicted_label}.",
                            evidence_span_ids=[record.evidence_span_id],
                            source_chunk_ids=[record.source_chunk_id],
                        )
                    ],
                ),
                content_hash=_hash_payload(link.model_dump(mode="json")),
                created_at=datetime.utcnow(),
            )
        )
    return pairs


def _entity_snapshot(
    *,
    snapshot_id: str,
    entity_id: str,
    document_id: str,
    canonical_name: str,
    aliases: list[str],
    observations: list[ProjectionEventObservation],
    related: list[str],
    evidence: list[str],
    chunks: list[str],
    projection_version: str,
    unresolved: list[str] | None = None,
    attributes: list[str] | None = None,
) -> EntityContextSnapshot:
    unresolved = unresolved or []
    attributes = attributes or []
    return EntityContextSnapshot(
        snapshot_id=snapshot_id,
        entity_id=entity_id,
        document_id=document_id,
        canonical_name=canonical_name,
        alias_surfaces=aliases,
        event_observations=observations,
        attributes=attributes,
        related_entity_ids=list(dict.fromkeys(related)),
        unresolved_counterparts=list(dict.fromkeys(unresolved)),
        evidence_span_ids=list(dict.fromkeys(evidence)),
        source_chunk_ids=list(dict.fromkeys(chunks)),
        projection_version=projection_version,
        extractor_versions=list(
            dict.fromkeys(
                item.extractor_version
                for item in observations
                if item.extractor_version
            )
        ),
        resolver_versions=list(
            dict.fromkeys(
                item.resolver_version
                for item in observations
                if item.resolver_version
            )
        ),
        encoding_input=EntityEncodingInput(
            entity_id=entity_id,
            document_id=document_id,
            projection_version=projection_version,
            identity_blocks=[
                EncodingBlock(
                    block_id=f"identity_{entity_id}",
                    block_type="identity",
                    text=f"{canonical_name}. Aliases: {', '.join(aliases)}.",
                    evidence_span_ids=evidence[:5],
                    source_chunk_ids=chunks[:5],
                )
            ],
            event_blocks=[
                EncodingBlock(
                    block_id=f"event_{item.event_id}",
                    block_type="event",
                    text=(
                        f"{canonical_name} role={item.entity_role} "
                        f"predicate={item.predicate} surface={item.surface_text}."
                    ),
                    evidence_span_ids=item.evidence_span_ids,
                    source_chunk_ids=item.source_chunk_ids,
                )
                for item in observations
            ],
        ),
        content_hash=_hash_payload(
            {
                "entity_id": entity_id,
                "canonical_name": canonical_name,
                "aliases": aliases,
                "observations": [item.model_dump(mode="json") for item in observations],
            }
        ),
        created_at=datetime.utcnow(),
    )


def _record_lines(context: str) -> list[str]:
    return [
        line.strip()
        for line in context.splitlines()
        if line.strip().startswith("Date:")
    ]


def _extract_field(line: str, field: str) -> str | None:
    match = re.search(rf"\|\|\s*{re.escape(field)}:\s*([^|]+)", line)
    return match.group(1).strip() if match else None


def _extract_date(line: str) -> str | None:
    match = re.search(r"Date:\s*([^|]+)", line)
    return match.group(1).strip() if match else None


def _label_counts(records: list[OOLONGRecordObservation]) -> dict[str, int]:
    labels = [record.predicted_label for record in records if record.predicted_label]
    return {
        label: labels.count(label)
        for label in sorted(set(labels))
    }


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _hash_payload(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
