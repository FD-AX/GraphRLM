import re

from app.graph.neo4j_client import Neo4jClient
from app.retrieval.models import EntityEventObservation


EVENTIVE_WORD_RE = re.compile(
    r"\b("
    r"saw|heard|said|went|came|opened|closed|took|gave|looked|walked|"
    r"[А-Яа-яЁё]+(?:л|ла|ли|ло|лся|лась|лись|ет|ют|ит|ат|ять|ить|ться|лся)"
    r")\b",
    re.IGNORECASE,
)


class GraphQueries:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def get_document_chunks(self, document_id: str) -> list[dict]:
        return self.client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            RETURN c
            ORDER BY c.index ASC
            """,
            {"document_id": document_id},
        )

    def get_entity_mentions(self, canonical_name: str) -> list[dict]:
        return self.client.execute_read(
            """
            MATCH (m:Mention)-[:REFERS_TO]->(e:Entity {canonical_name: $canonical_name})
            MATCH (c:Chunk)-[:HAS_MENTION]->(m)
            RETURN e, m, c
            ORDER BY c.index ASC
            """,
            {"canonical_name": canonical_name},
        )

    def get_raw_mentions_for_document(self, document_id: str) -> list[dict]:
        return self.client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[:HAS_MENTION]->(m:RawMention)
            RETURN c, m
            ORDER BY c.index ASC, m.start_char ASC
            """,
            {"document_id": document_id},
        )

    def get_resolution_audit_report(self, document_id: str) -> list[dict]:
        return self.client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[:HAS_MENTION]->(m:RawMention)
            MATCH (m)-[:HAS_RESOLUTION_HYPOTHESIS]->(h:ResolutionHypothesis)
            RETURN c.index AS chunk_index,
                   m.text AS raw_mention_text,
                   m.mention_type AS mention_type,
                   coalesce(h.mention_kind, m.mention_kind) AS mention_kind,
                   h.policy_version AS policy,
                   h.status AS status,
                   h.entity_creation_decision AS entity_creation_decision,
                   h.reason AS reason,
                   h.final_entity_id AS final_entity_id,
                   h.candidate_entity_ids AS candidate_entity_ids,
                   h.candidate_scores AS candidate_scores,
                   h.evidence_span_id AS evidence_span_id,
                   h.previous_decision AS previous_decision,
                   h.decision_stage AS decision_stage,
                   h.authority AS authority,
                   h.is_terminal AS is_terminal
            ORDER BY c.index ASC, m.start_char ASC, h.confidence DESC
            """,
            {"document_id": document_id},
        )

    def get_entity_states(self, canonical_name: str) -> list[dict]:
        return self.client.execute_read(
            """
            MATCH (s:EntityState)-[:STATE_OF]->(e:Entity {canonical_name: $canonical_name})
            RETURN e, s
            ORDER BY s.transition_id ASC
            """,
            {"canonical_name": canonical_name},
        )

    def get_claims_for_entity(self, canonical_name: str) -> list[dict]:
        return self.client.execute_read(
            """
            MATCH (e:Entity {canonical_name: $canonical_name})<-[:REFERS_TO]-(m:Mention)
            MATCH (c:Chunk)-[:HAS_MENTION]->(m)
            MATCH (cl:Claim)-[:SUPPORTED_BY]->(c)
            RETURN e, c, cl
            ORDER BY c.index ASC
            """,
            {"canonical_name": canonical_name},
        )

    def get_entity_context(
        self,
        canonical_name: str,
        document_id: str | None = None,
    ) -> dict | None:
        entity_rows = self.client.execute_read(
            """
            MATCH (e:Entity {canonical_name: $canonical_name})
            RETURN e {
                .entity_id,
                .canonical_name,
                .entity_type,
                .aliases,
                .description
            } AS entity
            LIMIT 1
            """,
            {"canonical_name": canonical_name},
        )
        if not entity_rows:
            return None

        entity = entity_rows[0]["entity"]
        entity_id = entity["entity_id"]
        params = {
            "entity_id": entity_id,
            "document_id": document_id,
        }

        mentions = self.client.execute_read(
            """
            MATCH (m:Mention)-[:REFERS_TO]->(:Entity {entity_id: $entity_id})
            MATCH (c:Chunk)-[:HAS_MENTION]->(m)
            WHERE $document_id IS NULL OR c.document_id = $document_id
            RETURN c.index AS chunk_index,
                   m.mention_id AS mention_id,
                   m.text AS text,
                   m.reference_type AS reference_type,
                   m.start_char AS start_char,
                   m.end_char AS end_char
            ORDER BY c.index ASC, m.start_char ASC
            """,
            params,
        )

        evidence_spans = self.client.execute_read(
            """
            MATCH (span:EvidenceSpan)-[r:RESOLVES_ENTITY]->(:Entity {entity_id: $entity_id})
            OPTIONAL MATCH (span)-[:IN_CHUNK]->(c:Chunk)
            WITH span, r, c
            WHERE $document_id IS NULL OR (
                span.document_id = $document_id AND c IS NOT NULL
            )
            RETURN c.index AS chunk_index,
                   span.span_id AS evidence_span_id,
                   span.original_text AS text,
                   span.start_char AS start_char,
                   span.end_char AS end_char,
                   r.role AS role,
                   r.surface_form AS surface_form,
                   r.reference_type AS reference_type,
                   r.confidence AS confidence
            ORDER BY c.index ASC, span.start_char ASC
            """,
            params,
        )

        claims = self.client.execute_read(
            """
            MATCH (cl:Claim)
            WHERE $entity_id IN coalesce(cl.subject_entity_ids, [])
            OPTIONAL MATCH (cl)-[:SUPPORTED_BY]->(c:Chunk)
            WITH cl, c
            WHERE $document_id IS NULL OR (
                c IS NOT NULL AND c.document_id = $document_id
            )
            RETURN c.index AS chunk_index,
                   cl.claim_id AS claim_id,
                   cl.text AS text,
                   cl.confidence AS confidence,
                   cl.needs_verification AS needs_verification,
                   cl.start_char AS start_char,
                   cl.end_char AS end_char
            ORDER BY c.index ASC, cl.start_char ASC
            """,
            params,
        )

        events = self.client.execute_read(
            """
            MATCH (ev:Event)
            WHERE $entity_id IN coalesce(ev.participants, [])
            OPTIONAL MATCH (ev)-[:SUPPORTED_BY]->(c:Chunk)
            WITH ev, c
            WHERE $document_id IS NULL OR (
                c IS NOT NULL AND c.document_id = $document_id
            )
            RETURN c.index AS chunk_index,
                   ev.event_id AS event_id,
                   ev.event_type AS event_type,
                   ev.description AS description,
                   ev.confidence AS confidence,
                   ev.start_char AS start_char,
                   ev.end_char AS end_char
            ORDER BY c.index ASC, ev.start_char ASC
            """,
            params,
        )

        event_frames = self.client.execute_read(
            """
            MATCH (arg:EventArgument)-[:RESOLVED_TO]->(:Entity {entity_id: $entity_id})
            MATCH (arg)-[:ARGUMENT_OF]->(ef:EventFrame)
            OPTIONAL MATCH (arg)-[:OBSERVED_AS]->(rm:RawMention)
            OPTIONAL MATCH (ef)-[:SUPPORTED_BY]->(span:EvidenceSpan)
            OPTIONAL MATCH (span)-[:IN_CHUNK]->(c:Chunk)
            WITH ef, arg, span, c
            WHERE $document_id IS NULL OR (
                span.document_id = $document_id AND c IS NOT NULL
            )
            RETURN c.index AS chunk_index,
                   ef.event_frame_id AS event_frame_id,
                   ef.predicate AS predicate,
                   ef.normalized_predicate AS normalized_predicate,
                   ef.event_type AS event_type,
                   arg.role AS role,
                   arg.entity_id AS participant_entity_id,
                   arg.surface_text AS surface_text,
                   arg.resolution_status AS participant_resolution_status,
                   arg.grounding_expectation AS grounding_expectation,
                   arg.argument_id AS argument_id,
                   ef.resolution_status AS event_resolution_status,
                   ef.materialization_status AS event_materialization_status,
                   span.span_id AS evidence_span_id,
                   span.original_text AS evidence_text,
                   ef.confidence AS confidence
            ORDER BY c.index ASC, span.start_char ASC
            """,
            params,
        )

        outgoing_relations = self.client.execute_read(
            """
            MATCH (:Entity {entity_id: $entity_id})<-[:FROM_ENTITY]-(r:Relation)-[:TO_ENTITY]->(target:Entity)
            OPTIONAL MATCH (c:Chunk {chunk_id: r.chunk_id})
            WITH r, target, c
            WHERE $document_id IS NULL OR (
                c IS NOT NULL AND c.document_id = $document_id
            )
            RETURN c.index AS chunk_index,
                   r.relation_id AS relation_id,
                   r.relation_type AS relation_type,
                   target.entity_id AS target_entity_id,
                   target.canonical_name AS target_name,
                   r.confidence AS confidence,
                   r.start_char AS start_char,
                   r.end_char AS end_char
            ORDER BY c.index ASC, r.start_char ASC
            """,
            params,
        )

        incoming_relations = self.client.execute_read(
            """
            MATCH (source:Entity)<-[:FROM_ENTITY]-(r:Relation)-[:TO_ENTITY]->(:Entity {entity_id: $entity_id})
            OPTIONAL MATCH (c:Chunk {chunk_id: r.chunk_id})
            WITH source, r, c
            WHERE $document_id IS NULL OR (
                c IS NOT NULL AND c.document_id = $document_id
            )
            RETURN c.index AS chunk_index,
                   r.relation_id AS relation_id,
                   r.relation_type AS relation_type,
                   source.entity_id AS source_entity_id,
                   source.canonical_name AS source_name,
                   r.confidence AS confidence,
                   r.start_char AS start_char,
                   r.end_char AS end_char
            ORDER BY c.index ASC, r.start_char ASC
            """,
            params,
        )

        relation_candidates = self.client.execute_read(
            """
            MATCH (source:Entity)-[:RELATION_SOURCE]->(rc:RelationCandidate)-[:RELATION_TARGET]->(target:Entity)
            WHERE source.entity_id = $entity_id OR target.entity_id = $entity_id
            OPTIONAL MATCH (rc)-[:SUPPORTED_BY]->(span:EvidenceSpan)
            OPTIONAL MATCH (span)-[:IN_CHUNK]->(c:Chunk)
            WITH source, rc, target, span, c
            WHERE $document_id IS NULL OR (
                span.document_id = $document_id AND c IS NOT NULL
            )
            RETURN c.index AS chunk_index,
                   rc.relation_candidate_id AS relation_candidate_id,
                   source.entity_id AS source_entity_id,
                   source.canonical_name AS source_name,
                   target.entity_id AS target_entity_id,
                   target.canonical_name AS target_name,
                   rc.relation_span AS relation_span,
                   rc.symbolic_hint AS symbolic_hint,
                   rc.direction_hint AS direction_hint,
                   rc.confidence AS confidence,
                   span.span_id AS evidence_span_id,
                   span.original_text AS evidence_text
            ORDER BY c.index ASC, span.start_char ASC
            """,
            params,
        )

        return {
            "entity": entity,
            "mentions": mentions,
            "evidence_spans": evidence_spans,
            "claims": claims,
            "events": events,
            "event_frames": event_frames,
            "outgoing_relations": outgoing_relations,
            "incoming_relations": incoming_relations,
            "relation_candidates": relation_candidates,
        }

    def get_event_coverage_audit(self, document_id: str) -> dict:
        chunk_rows = self.client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            RETURN c.chunk_id AS chunk_id,
                   c.index AS chunk_index,
                   c.text AS text
            ORDER BY c.index ASC
            """,
            {"document_id": document_id},
        )
        event_rows = self.client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(ef:EventFrame)
            OPTIONAL MATCH (arg:EventArgument)-[:ARGUMENT_OF]->(ef)
            RETURN c.chunk_id AS chunk_id,
                   c.index AS chunk_index,
                   count(DISTINCT ef) AS events_extracted,
                   count(arg) AS event_participants_count,
                   count(CASE WHEN arg IS NOT NULL AND arg.resolution_status = 'resolved' THEN 1 END)
                       AS participants_resolved,
                   count(CASE WHEN arg IS NOT NULL AND (arg.resolution_status <> 'resolved' OR arg.resolution_status IS NULL) THEN 1 END)
                       AS participants_unresolved,
                   count(DISTINCT CASE WHEN ef.evidence_span_id IS NULL THEN ef END)
                       AS events_without_evidence,
                   count(DISTINCT CASE WHEN ef.resolution_status <> 'complete' THEN ef END)
                       AS events_without_canonical_participants
            ORDER BY c.index ASC
            """,
            {"document_id": document_id},
        )
        event_by_chunk = {row["chunk_id"]: row for row in event_rows}

        chunks = []
        totals = {
            "sentences_count": 0,
            "eventive_sentences_count": 0,
            "events_extracted": 0,
            "event_participants_count": 0,
            "participants_resolved": 0,
            "participants_unresolved": 0,
            "events_without_evidence": 0,
            "events_without_canonical_participants": 0,
        }

        for chunk in chunk_rows:
            text = chunk["text"] or ""
            sentences = [
                sentence.strip()
                for sentence in re.split(r"[.!?…]+|\n+", text)
                if sentence.strip()
            ]
            eventive_sentences = [
                sentence
                for sentence in sentences
                if EVENTIVE_WORD_RE.search(sentence)
            ]
            event_stats = event_by_chunk.get(chunk["chunk_id"], {})
            row = {
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "sentences_count": len(sentences),
                "eventive_sentences_count": len(eventive_sentences),
                "events_extracted": event_stats.get("events_extracted", 0),
                "event_participants_count": event_stats.get(
                    "event_participants_count",
                    0,
                ),
                "participants_resolved": event_stats.get("participants_resolved", 0),
                "participants_unresolved": event_stats.get(
                    "participants_unresolved",
                    0,
                ),
                "events_without_evidence": event_stats.get(
                    "events_without_evidence",
                    0,
                ),
                "events_without_canonical_participants": event_stats.get(
                    "events_without_canonical_participants",
                    0,
                ),
            }
            chunks.append(row)
            for key in totals:
                totals[key] += row[key]

        coverage = (
            totals["events_extracted"] / totals["eventive_sentences_count"]
            if totals["eventive_sentences_count"]
            else 0.0
        )
        return {
            "document_id": document_id,
            "event_coverage": coverage,
            "totals": totals,
            "chunks": chunks,
        }

    def get_entity_event_observations(
        self,
        canonical_name: str,
        document_id: str | None = None,
    ) -> list[EntityEventObservation]:
        context = self.get_entity_context(canonical_name, document_id=document_id)
        if context is None:
            return []

        entity_id = context["entity"]["entity_id"]
        observations: list[EntityEventObservation] = []

        for frame in context["event_frames"]:
            event_id = frame["event_frame_id"]
            all_args = self.client.execute_read(
                """
                MATCH (arg:EventArgument)-[:ARGUMENT_OF]->(ef:EventFrame {event_frame_id: $event_id})
                OPTIONAL MATCH (arg)-[:RESOLVED_TO]->(e:Entity)
                RETURN arg.argument_id AS argument_id,
                       arg.role AS role,
                       arg.entity_id AS entity_id,
                       arg.surface_text AS surface_text,
                       arg.resolution_status AS resolution_status,
                       arg.grounding_expectation AS grounding_expectation,
                       e.entity_id AS resolved_entity_id
                ORDER BY arg.argument_index ASC
                """,
                {"event_id": event_id},
            )
            counterpart_entity_ids = []
            unresolved_counterparts = []
            for argument in all_args:
                argument_entity_id = (
                    argument["resolved_entity_id"] or argument["entity_id"]
                )
                if argument_entity_id == entity_id:
                    continue
                if argument_entity_id:
                    counterpart_entity_ids.append(argument_entity_id)
                elif argument["surface_text"]:
                    unresolved_counterparts.append(argument["surface_text"])

            observations.append(
                EntityEventObservation(
                    entity_id=entity_id,
                    event_id=event_id,
                    entity_role=frame["role"],
                    normalized_predicate=(
                        frame["normalized_predicate"] or frame["predicate"]
                    ),
                    counterpart_entity_ids=list(dict.fromkeys(counterpart_entity_ids)),
                    unresolved_counterparts=list(
                        dict.fromkeys(unresolved_counterparts)
                    ),
                    temporal_scope=None,
                    modality=None,
                    polarity="positive",
                    evidence_span_ids=[
                        frame["evidence_span_id"]
                    ]
                    if frame["evidence_span_id"]
                    else [],
                    event_resolution_status=frame["event_resolution_status"],
                    event_materialization_status=frame[
                        "event_materialization_status"
                    ],
                    extractor_version=None,
                    resolver_version=None,
                )
            )

        return observations
