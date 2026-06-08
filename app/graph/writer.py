import json

from app.core.graph_models import (
    ChunkNode,
    ClaimNode,
    DocumentNode,
    EntityNode,
    EventNode,
    EventFrame,
    EvidenceSpan,
    LatentRelationEdge,
    LocalGraph,
    MentionNode,
    RawMention,
    RawRelation,
    RelationCandidate,
    RelationEdge,
    ResolutionHypothesis,
    TerminalResolution,
)
from app.graph.neo4j_client import Neo4jClient
from app.rlm.state import EntityState, RLMTransition


class GraphWriter:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def write_document(self, document: DocumentNode) -> None:
        params = document.model_dump()
        params["metadata"] = json.dumps(document.metadata, ensure_ascii=False)

        self.client.execute_write(
            """
            MERGE (d:Document {document_id: $document_id})
            ON CREATE SET d.created_at = datetime()
            SET d.title = $title,
                d.source_path = $source_path,
                d.updated_at = datetime(),
                d.metadata = $metadata
            """,
            params,
        )

    def write_chunk(self, chunk: ChunkNode) -> None:
        self.client.execute_write(
            """
            MERGE (c:Chunk {chunk_id: $chunk_id})
            ON CREATE SET c.extraction_status = $extraction_status,
                c.extraction_attempts = $extraction_attempts,
                c.extraction_started_at = $extraction_started_at,
                c.extraction_finished_at = $extraction_finished_at,
                c.extraction_error = $extraction_error,
                c.created_at = datetime()
            SET c.document_id = $document_id,
                c.index = $index,
                c.text = $text,
                c.token_count = $token_count,
                c.start_char = $start_char,
                c.end_char = $end_char,
                c.content_hash = $content_hash,
                c.updated_at = datetime()

            WITH c
            MATCH (d:Document {document_id: $document_id})
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            chunk.model_dump(),
        )

    def get_chunk_by_index(self, document_id: str, index: int) -> dict | None:
        rows = self.client.execute_read(
            """
            MATCH (c:Chunk {document_id: $document_id, index: $index})
            RETURN c {
                .chunk_id,
                .document_id,
                .index,
                .content_hash,
                .extraction_status,
                .extraction_attempts,
                .extraction_error
            } AS chunk
            ORDER BY c.updated_at DESC
            LIMIT 1
            """,
            {
                "document_id": document_id,
                "index": index,
            },
        )

        if not rows:
            return None

        return rows[0]["chunk"]

    def mark_chunk_processing(self, chunk_id: str) -> None:
        self.client.execute_write(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            SET c.extraction_status = 'PROCESSING',
                c.extraction_started_at = datetime(),
                c.extraction_finished_at = null,
                c.extraction_error = null,
                c.extraction_attempts = coalesce(c.extraction_attempts, 0) + 1,
                c.updated_at = datetime()
            """,
            {"chunk_id": chunk_id},
        )

    def mark_chunk_done(self, chunk_id: str) -> None:
        self.client.execute_write(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            SET c.extraction_status = 'DONE',
                c.extraction_finished_at = datetime(),
                c.extraction_error = null,
                c.updated_at = datetime()
            """,
            {"chunk_id": chunk_id},
        )

    def mark_chunk_failed(self, chunk_id: str, error: str) -> None:
        self.client.execute_write(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            SET c.extraction_status = 'FAILED',
                c.extraction_finished_at = datetime(),
                c.extraction_error = $error,
                c.updated_at = datetime()
            """,
            {
                "chunk_id": chunk_id,
                "error": error[:4000],
            },
        )

    def write_raw_mention(self, mention: RawMention) -> None:
        params = mention.model_dump()
        params["start_char"] = mention.span.start_char
        params["end_char"] = mention.span.end_char
        params["semantic_payload"] = json.dumps(
            params["semantic_payload"],
            ensure_ascii=False,
        )
        params["repair_notes"] = json.dumps(
            params["repair_notes"],
            ensure_ascii=False,
        )

        self.client.execute_write(
            """
            MERGE (m:RawMention {mention_id: $mention_id})
            SET m.chunk_id = $chunk_id,
                m.text = $text,
                m.normalized_text = $normalized_text,
                m.mention_type = $mention_type,
                m.mention_kind = $mention_kind,
                m.start_char = $start_char,
                m.end_char = $end_char,
                m.source = $source,
                m.extractor_source = $extractor_source,
                m.extractor_version = $extractor_version,
                m.repaired_by = $repaired_by,
                m.repair_notes = $repair_notes,
                m.confidence = $confidence,
                m.semantic_payload = $semantic_payload

            WITH m
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (c)-[:HAS_MENTION]->(m)
            """,
            params,
        )

    def write_raw_relation(self, relation: RawRelation) -> None:
        self.client.execute_write(
            """
            MERGE (r:RawRelation {raw_relation_id: $raw_relation_id})
            SET r.chunk_id = $chunk_id,
                r.relation_span = $relation_span,
                r.evidence_span_id = $evidence_span_id,
                r.relation_type = $relation_type,
                r.confidence = $confidence

            WITH r
            MATCH (source:RawMention {mention_id: $source_mention_id})
            MERGE (source)-[:RAW_RELATION_SOURCE]->(r)

            WITH r
            MATCH (target:RawMention {mention_id: $target_mention_id})
            MERGE (r)-[:RAW_RELATION_TARGET]->(target)

            WITH r
            MATCH (span:EvidenceSpan {span_id: $evidence_span_id})
            MERGE (r)-[:SUPPORTED_BY]->(span)
            """,
            relation.model_dump(),
        )

    def write_event_frame(self, event_frame: EventFrame) -> None:
        params = event_frame.model_dump(exclude={"arguments"})

        self.client.execute_write(
            """
            MERGE (ev:EventFrame {event_frame_id: $event_frame_id})
            SET ev.chunk_id = $chunk_id,
                ev.document_id = $document_id,
                ev.predicate = $predicate,
                ev.normalized_predicate = $normalized_predicate,
                ev.event_type = $event_type,
                ev.evidence_span_id = $evidence_span_id,
                ev.temporal_scope = $temporal_scope,
                ev.modality = $modality,
                ev.polarity = $polarity,
                ev.resolution_status = $resolution_status,
                ev.materialization_status = $materialization_status,
                ev.source = $source,
                ev.extractor_version = $extractor_version,
                ev.resolver_version = $resolver_version,
                ev.confidence = $confidence

            WITH ev
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (c)-[:HAS_EVENT]->(ev)

            WITH ev
            MATCH (span:EvidenceSpan {span_id: $evidence_span_id})
            MERGE (ev)-[:SUPPORTED_BY]->(span)
            """,
            params,
        )

        for argument in event_frame.arguments:
            self.client.execute_write(
                """
                MERGE (arg:EventArgument {argument_id: $argument_id})
                SET arg.event_frame_id = $event_frame_id,
                    arg.role = $role,
                    arg.normalized_role = $role,
                    arg.mention_id = $mention_id,
                    arg.entity_id = $entity_id,
                    arg.surface_text = $surface_text,
                    arg.evidence_span_id = $evidence_span_id,
                    arg.resolution_status = $resolution_status,
                    arg.grounding_expectation = $grounding_expectation,
                    arg.argument_index = $argument_index,
                    arg.extractor_version = $extractor_version,
                    arg.resolver_version = $resolver_version,
                    arg.confidence = $confidence

                WITH arg
                MATCH (ev:EventFrame {event_frame_id: $event_frame_id})
                MERGE (arg)-[:ARGUMENT_OF]->(ev)

                WITH arg
                MATCH (m:RawMention {mention_id: $mention_id})
                MERGE (arg)-[:OBSERVED_AS]->(m)
                """,
                {
                    "argument_id": argument.argument_id,
                    "event_frame_id": event_frame.event_frame_id,
                    "role": argument.role,
                    "mention_id": argument.mention_id,
                    "entity_id": argument.entity_id,
                    "surface_text": argument.surface_text,
                    "evidence_span_id": argument.evidence_span_id,
                    "resolution_status": argument.resolution_status,
                    "grounding_expectation": argument.grounding_expectation,
                    "argument_index": argument.argument_index,
                    "extractor_version": argument.extractor_version,
                    "resolver_version": argument.resolver_version,
                    "confidence": argument.confidence,
                },
            )
            if argument.entity_id:
                self.client.execute_write(
                    """
                    MATCH (arg:EventArgument {argument_id: $argument_id})
                    MATCH (e:Entity {entity_id: $entity_id})
                    MERGE (arg)-[:RESOLVED_TO]->(e)
                    """,
                    {
                        "argument_id": argument.argument_id,
                        "entity_id": argument.entity_id,
                    },
                )

            self.client.execute_write(
                """
                MATCH (m:RawMention {mention_id: $mention_id})
                MATCH (ev:EventFrame {event_frame_id: $event_frame_id})
                MERGE (m)-[r:ARGUMENT_OF]->(ev)
                SET r.argument_id = $argument_id,
                    r.role = $role,
                    r.normalized_role = $role,
                    r.mention_id = $mention_id,
                    r.entity_id = $entity_id,
                    r.surface_text = $surface_text,
                    r.evidence_span_id = $evidence_span_id,
                    r.resolution_status = $resolution_status,
                    r.grounding_expectation = $grounding_expectation,
                    r.argument_index = $argument_index,
                    r.extractor_version = $extractor_version,
                    r.resolver_version = $resolver_version,
                    r.confidence = $confidence
                """,
                {
                    "event_frame_id": event_frame.event_frame_id,
                    "argument_id": argument.argument_id,
                    "mention_id": argument.mention_id,
                    "role": argument.role,
                    "entity_id": argument.entity_id,
                    "surface_text": argument.surface_text,
                    "evidence_span_id": argument.evidence_span_id,
                    "resolution_status": argument.resolution_status,
                    "grounding_expectation": argument.grounding_expectation,
                    "argument_index": argument.argument_index,
                    "extractor_version": argument.extractor_version,
                    "resolver_version": argument.resolver_version,
                    "confidence": argument.confidence,
                },
            )

    def write_resolution_hypothesis(
        self,
        hypothesis: ResolutionHypothesis,
    ) -> None:
        params = hypothesis.model_dump(exclude={"features"})
        params["positive_evidence"] = json.dumps(
            params["positive_evidence"],
            ensure_ascii=False,
        )
        params["negative_evidence"] = json.dumps(
            params["negative_evidence"],
            ensure_ascii=False,
        )
        params["features"] = json.dumps(
            [feature.model_dump() for feature in hypothesis.features],
            ensure_ascii=False,
        )
        params["candidate_scores"] = json.dumps(
            params["candidate_scores"],
            ensure_ascii=False,
        )

        self.client.execute_write(
            """
            MERGE (h:ResolutionHypothesis {hypothesis_id: $hypothesis_id})
            SET h.mention_id = $mention_id,
                h.hypothesis_type = $hypothesis_type,
                h.candidate_entity_id = $candidate_entity_id,
                h.candidate_entity_name = $candidate_entity_name,
                h.confidence = $confidence,
                h.status = $status,
                h.mention_kind = $mention_kind,
                h.entity_creation_decision = $entity_creation_decision,
                h.final_entity_id = $final_entity_id,
                h.candidate_entity_ids = $candidate_entity_ids,
                h.candidate_scores = $candidate_scores,
                h.evidence_span_id = $evidence_span_id,
                h.previous_decision = $previous_decision,
                h.decision_stage = $decision_stage,
                h.authority = $authority,
                h.is_terminal = $is_terminal,
                h.reason = $reason,
                h.positive_evidence = $positive_evidence,
                h.negative_evidence = $negative_evidence,
                h.features = $features,
                h.resolution_run_id = $resolution_run_id,
                h.resolver_version = $resolver_version,
                h.policy_version = $policy_version,
                h.extractor_version = $extractor_version,
                h.model_name = $model_name,
                h.chunking_version = $chunking_version

            WITH h
            MATCH (m:RawMention {mention_id: $mention_id})
            MERGE (m)-[:HAS_RESOLUTION_HYPOTHESIS]->(h)
            """,
            params,
        )

        if hypothesis.candidate_entity_id:
            self.client.execute_write(
                """
                MATCH (h:ResolutionHypothesis {hypothesis_id: $hypothesis_id})
                MATCH (e:Entity {entity_id: $candidate_entity_id})
                MERGE (h)-[:CANDIDATE_ENTITY]->(e)
                """,
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "candidate_entity_id": hypothesis.candidate_entity_id,
                },
            )

    def write_terminal_resolution(self, resolution: TerminalResolution) -> None:
        self.client.execute_write(
            """
            MERGE (tr:TerminalResolution {
                mention_id: $mention_id,
                created_at_stage: $created_at_stage,
                policy_version: $policy_version
            })
            SET tr.decision = $decision,
                tr.authority = $authority,
                tr.confidence = $confidence,
                tr.final_entity_id = $final_entity_id,
                tr.revisable_by_higher_authority = $revisable_by_higher_authority

            WITH tr
            MATCH (m:RawMention {mention_id: $mention_id})
            MERGE (m)-[:HAS_TERMINAL_RESOLUTION]->(tr)
            """,
            resolution.model_dump(),
        )

    def delete_chunk_subtree(self, chunk_id: str) -> None:
        self.client.execute_write(
            """
            MATCH (t:RLMTransition {to_chunk_id: $chunk_id})
            OPTIONAL MATCH (s:EntityState {transition_id: t.transition_id})
            DETACH DELETE s, t
            """,
            {"chunk_id": chunk_id},
        )

        self.client.execute_write(
            """
            MATCH (span:EvidenceSpan {chunk_id: $chunk_id})
            OPTIONAL MATCH (rc:RelationCandidate)-[:SUPPORTED_BY]->(span)
            OPTIONAL MATCH (lr:LatentRelation)-[:SUPPORTED_BY]->(span)
            OPTIONAL MATCH (rr:RawRelation)-[:SUPPORTED_BY]->(span)
            OPTIONAL MATCH (ef:EventFrame)-[:SUPPORTED_BY]->(span)
            OPTIONAL MATCH (arg:EventArgument)-[:ARGUMENT_OF]->(ef)
            DETACH DELETE arg, lr, rc, rr, ef, span
            """,
            {"chunk_id": chunk_id},
        )

        self.client.execute_write(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            OPTIONAL MATCH (c)-[:HAS_MENTION]->(rm:RawMention)
            OPTIONAL MATCH (rm)-[:HAS_RESOLUTION_HYPOTHESIS]->(rh:ResolutionHypothesis)
            OPTIONAL MATCH (rm)-[:HAS_TERMINAL_RESOLUTION]->(tr:TerminalResolution)
            OPTIONAL MATCH (c)-[:HAS_MENTION]->(m:Mention)
            OPTIONAL MATCH (cl:Claim {chunk_id: $chunk_id})
            OPTIONAL MATCH (ev:Event {chunk_id: $chunk_id})
            OPTIONAL MATCH (r:Relation {chunk_id: $chunk_id})
            DETACH DELETE tr, rh, rm, r, ev, cl, m, c
            """,
            {"chunk_id": chunk_id},
        )

    def write_entity(self, entity: EntityNode) -> None:
        self.client.execute_write(
            """
            MERGE (e:Entity {entity_id: $entity_id})
            SET e.canonical_name = $canonical_name,
                e.entity_type = $entity_type,
                e.aliases = $aliases,
                e.description = $description
            """,
            entity.model_dump(),
        )

    def write_mention(self, mention: MentionNode) -> None:
        params = mention.model_dump()
        params["start_char"] = mention.span.start_char
        params["end_char"] = mention.span.end_char

        self.client.execute_write(
            """
            MERGE (m:Mention {mention_id: $mention_id})
            SET m.chunk_id = $chunk_id,
                m.text = $text,
                m.start_char = $start_char,
                m.end_char = $end_char

            WITH m
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (c)-[:HAS_MENTION]->(m)

            WITH m
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (m)-[:REFERS_TO]->(e)
            """,
            params,
        )

    def write_claim(self, claim: ClaimNode) -> None:
        params = claim.model_dump()
        params["start_char"] = (
            claim.evidence_span.start_char if claim.evidence_span else None
        )
        params["end_char"] = (
            claim.evidence_span.end_char if claim.evidence_span else None
        )

        self.client.execute_write(
            """
            MERGE (cl:Claim {claim_id: $claim_id})
            SET cl.chunk_id = $chunk_id,
                cl.text = $text,
                cl.confidence = $confidence,
                cl.needs_verification = $needs_verification,
                cl.start_char = $start_char,
                cl.end_char = $end_char,
                cl.subject_entity_ids = $subject_entity_ids

            WITH cl
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (cl)-[:SUPPORTED_BY]->(c)
            """,
            params,
        )

    def write_evidence_span(self, evidence_span: EvidenceSpan) -> None:
        params = evidence_span.model_dump()
        params["resolved_entities"] = json.dumps(
            params["resolved_entities"],
            ensure_ascii=False,
        )

        self.client.execute_write(
            """
            MERGE (span:EvidenceSpan {span_id: $span_id})
            SET span.original_text = $original_text,
                span.normalized_text = $normalized_text,
                span.start_char = $start_char,
                span.end_char = $end_char,
                span.chunk_id = $chunk_id,
                span.document_id = $document_id,
                span.resolved_entities = $resolved_entities

            WITH span
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (span)-[:IN_CHUNK]->(c)
            """,
            params,
        )

        for ref in evidence_span.resolved_entities:
            self.client.execute_write(
                """
                MATCH (span:EvidenceSpan {span_id: $span_id})
                MATCH (e:Entity {entity_id: $entity_id})
                MERGE (span)-[r:RESOLVES_ENTITY]->(e)
                SET r.surface_form = $surface_form,
                    r.reference_type = $reference_type,
                    r.role = $role,
                    r.confidence = $confidence
                """,
                {
                    "span_id": evidence_span.span_id,
                    "entity_id": ref.entity_id,
                    "surface_form": ref.surface_form,
                    "reference_type": ref.reference_type,
                    "role": ref.role,
                    "confidence": ref.confidence,
                },
            )

    def write_relation_candidate(self, candidate: RelationCandidate) -> None:
        self.client.execute_write(
            """
            MERGE (rc:RelationCandidate {
                relation_candidate_id: $relation_candidate_id
            })
            SET rc.source_entity_id = $source_entity_id,
                rc.target_entity_id = $target_entity_id,
                rc.relation_span = $relation_span,
                rc.evidence_span_id = $evidence_span_id,
                rc.direction_hint = $direction_hint,
                rc.direction_confidence = $direction_confidence,
                rc.encode_as_latent_edge = $encode_as_latent_edge,
                rc.symbolic_hint = $symbolic_hint,
                rc.confidence = $confidence

            WITH rc
            MATCH (source:Entity {entity_id: $source_entity_id})
            MERGE (source)-[:RELATION_SOURCE]->(rc)

            WITH rc
            MATCH (target:Entity {entity_id: $target_entity_id})
            MERGE (rc)-[:RELATION_TARGET]->(target)

            WITH rc
            MATCH (span:EvidenceSpan {span_id: $evidence_span_id})
            MERGE (rc)-[:SUPPORTED_BY]->(span)
            """,
            candidate.model_dump(),
        )

    def write_latent_relation_edge(self, edge: LatentRelationEdge) -> None:
        self.client.execute_write(
            """
            MERGE (lr:LatentRelation {edge_id: $edge_id})
            SET lr.source_entity_id = $source_entity_id,
                lr.target_entity_id = $target_entity_id,
                lr.relation_candidate_id = $relation_candidate_id,
                lr.evidence_span_id = $evidence_span_id,
                lr.vector_ref = $vector_ref,
                lr.projection_space_id = $projection_space_id,
                lr.direction = $direction,
                lr.confidence = $confidence,
                lr.symbolic_hint = $symbolic_hint,
                lr.vector = $vector

            WITH lr
            MATCH (source:Entity {entity_id: $source_entity_id})
            MERGE (source)-[:LATENT_RELATION_SOURCE]->(lr)

            WITH lr
            MATCH (target:Entity {entity_id: $target_entity_id})
            MERGE (lr)-[:LATENT_RELATION_TARGET]->(target)

            WITH lr
            MATCH (span:EvidenceSpan {span_id: $evidence_span_id})
            MERGE (lr)-[:SUPPORTED_BY]->(span)

            WITH lr
            MATCH (rc:RelationCandidate {
                relation_candidate_id: $relation_candidate_id
            })
            MERGE (lr)-[:ENCODES]->(rc)
            """,
            edge.model_dump(),
        )

    def write_event(self, event: EventNode) -> None:
        params = event.model_dump()
        params["start_char"] = (
            event.evidence_span.start_char if event.evidence_span else None
        )
        params["end_char"] = (
            event.evidence_span.end_char if event.evidence_span else None
        )

        self.client.execute_write(
            """
            MERGE (ev:Event {event_id: $event_id})
            SET ev.chunk_id = $chunk_id,
                ev.event_type = $event_type,
                ev.description = $description,
                ev.participants = $participants,
                ev.confidence = $confidence,
                ev.start_char = $start_char,
                ev.end_char = $end_char

            WITH ev
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (ev)-[:SUPPORTED_BY]->(c)
            """,
            params,
        )

    def write_relation(self, relation: RelationEdge) -> None:
        params = relation.model_dump()
        params["start_char"] = (
            relation.evidence_span.start_char if relation.evidence_span else None
        )
        params["end_char"] = (
            relation.evidence_span.end_char if relation.evidence_span else None
        )

        self.client.execute_write(
            """
            MERGE (r:Relation {relation_id: $relation_id})
            SET r.source_id = $source_id,
                r.target_id = $target_id,
                r.relation_type = $relation_type,
                r.chunk_id = $chunk_id,
                r.confidence = $confidence,
                r.start_char = $start_char,
                r.end_char = $end_char

            WITH r
            MATCH (source:Entity {entity_id: $source_id})
            MERGE (r)-[:FROM_ENTITY]->(source)

            WITH r
            MATCH (target:Entity {entity_id: $target_id})
            MERGE (r)-[:TO_ENTITY]->(target)

            WITH r
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (r)-[:SUPPORTED_BY]->(c)
            """,
            params,
        )

    def write_local_graph(self, local_graph: LocalGraph) -> None:
        self.write_chunk(local_graph.chunk)

        for raw_mention in local_graph.raw_mentions:
            self.write_raw_mention(raw_mention)

        for entity in local_graph.entities:
            self.write_entity(entity)

        for mention in local_graph.mentions:
            self.write_mention(mention)

        for evidence_span in local_graph.evidence_spans:
            self.write_evidence_span(evidence_span)

        for event_frame in local_graph.event_frames:
            self.write_event_frame(event_frame)

        for raw_relation in local_graph.raw_relations:
            self.write_raw_relation(raw_relation)

        for claim in local_graph.claims:
            self.write_claim(claim)

        for event in local_graph.events:
            self.write_event(event)

        for relation in local_graph.relations:
            self.write_relation(relation)

        for candidate in local_graph.relation_candidates:
            self.write_relation_candidate(candidate)

        for edge in local_graph.latent_relation_edges:
            self.write_latent_relation_edge(edge)

        for hypothesis in local_graph.resolution_hypotheses:
            self.write_resolution_hypothesis(hypothesis)

        for resolution in local_graph.terminal_resolutions:
            self.write_terminal_resolution(resolution)

    def write_entity_state(self, entity_state: EntityState, transition_id: str) -> None:
        params = entity_state.model_dump()
        params["transition_id"] = transition_id

        self.client.execute_write(
            """
            MERGE (s:EntityState {
                transition_id: $transition_id,
                entity_id: $entity_id
            })
            SET s.canonical_name = $canonical_name,
                s.attributes = $attributes,
                s.hypotheses = $hypotheses,
                s.evidence_refs = $evidence_refs,
                s.confidence = $confidence

            WITH s
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (s)-[:STATE_OF]->(e)
            """,
            params,
        )

    def write_rlm_transition(self, transition: RLMTransition) -> None:
        params = transition.model_dump(
            exclude={
                "added_entities",
                "updated_entities",
                "added_relations",
                "added_relation_candidates",
                "added_evidence_spans",
            }
        )

        self.client.execute_write(
            """
            MERGE (t:RLMTransition {transition_id: $transition_id})
            SET t.document_id = $document_id,
                t.from_chunk_id = $from_chunk_id,
                t.to_chunk_id = $to_chunk_id,
                t.from_chunk_index = $from_chunk_index,
                t.to_chunk_index = $to_chunk_index,
                t.notes = $notes

            WITH t
            MATCH (to:Chunk {chunk_id: $to_chunk_id})
            MERGE (t)-[:TO_CHUNK]->(to)
            """,
            params,
        )

        for entity_state in transition.added_entities:
            self.write_entity_state(entity_state, transition.transition_id)

        for entity_state in transition.updated_entities:
            self.write_entity_state(entity_state, transition.transition_id)
