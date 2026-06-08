from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from app.agent_graph.chunk_schemas import (
    ExtractedEventArgument,
    ExtractedEventFrame,
    ExtractedRawMention,
    LLMGraphExtraction,
)


pytest.importorskip("neo4j")

from app.graph.neo4j_client import Neo4jClient
from app.graph.queries import GraphQueries
from app.graph.writer import GraphWriter
from app.runtime.ingest_pipeline import IngestPipeline


class FakeGraphModelAdapter:
    async def structured_call(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        output_schema: type[LLMGraphExtraction],
    ) -> LLMGraphExtraction:
        text = user_payload["chunk"]["text"]
        semyon_start = text.index("Semyon")
        old_man_start = text.index("old man")
        old_man_second_start = text.index("Old man")

        return output_schema(
            raw_mentions=[
                ExtractedRawMention(
                    text="Semyon",
                    start_char_in_chunk=semyon_start,
                    end_char_in_chunk=semyon_start + len("Semyon"),
                    mention_type="named",
                ),
                ExtractedRawMention(
                    text="old man",
                    start_char_in_chunk=old_man_start,
                    end_char_in_chunk=old_man_start + len("old man"),
                    mention_type="descriptor",
                    semantic_payload={"age_hint": "old", "gender_hint": "male"},
                ),
                ExtractedRawMention(
                    text="Old man",
                    start_char_in_chunk=old_man_second_start,
                    end_char_in_chunk=old_man_second_start + len("Old man"),
                    mention_type="descriptor",
                    semantic_payload={"age_hint": "old", "gender_hint": "male"},
                ),
            ],
            event_frames=[
                ExtractedEventFrame(
                    predicate="saw",
                    event_type="perception",
                    evidence_start_char_in_chunk=0,
                    evidence_end_char_in_chunk=len("Semyon saw old man."),
                    arguments=[
                        ExtractedEventArgument(
                            role="subject",
                            mention_text="Semyon",
                            mention_start_char_in_chunk=semyon_start,
                            mention_end_char_in_chunk=semyon_start
                            + len("Semyon"),
                        ),
                        ExtractedEventArgument(
                            role="object",
                            mention_text="old man",
                            mention_start_char_in_chunk=old_man_start,
                            mention_end_char_in_chunk=old_man_start
                            + len("old man"),
                        ),
                    ],
                )
            ],
        )


def _skip_if_neo4j_port_is_closed() -> None:
    try:
        with socket.create_connection(("127.0.0.1", 7687), timeout=1.0):
            return
    except OSError as exc:
        pytest.skip(
            "Neo4j Bolt is not available on localhost:7687. "
            "Run: kubectl -n jmlc port-forward svc/neo4j 7687:7687 7474:7474. "
            f"Original error: {exc}"
        )


def _cleanup_demo_document(client: Neo4jClient, document_id: str) -> None:
    client.execute_write(
        """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)-[:HAS_MENTION]->(rm:RawMention)
        OPTIONAL MATCH (rm)-[:HAS_RESOLUTION_HYPOTHESIS]->(rh:ResolutionHypothesis)
        OPTIONAL MATCH (rm)-[:HAS_TERMINAL_RESOLUTION]->(tr:TerminalResolution)
        OPTIONAL MATCH (c)-[:HAS_MENTION]->(m:Mention)
        OPTIONAL MATCH (c)-[:HAS_EVENT]->(ef:EventFrame)
        OPTIONAL MATCH (arg:EventArgument)-[:ARGUMENT_OF]->(ef)
        OPTIONAL MATCH (cl:Claim)-[:SUPPORTED_BY]->(c)
        OPTIONAL MATCH (ev:Event)-[:SUPPORTED_BY]->(c)
        OPTIONAL MATCH (r:Relation {chunk_id: c.chunk_id})
        OPTIONAL MATCH (span:EvidenceSpan)-[:IN_CHUNK]->(c)
        OPTIONAL MATCH (rr:RawRelation)-[:SUPPORTED_BY]->(span)
        OPTIONAL MATCH (rc:RelationCandidate)-[:SUPPORTED_BY]->(span)
        OPTIONAL MATCH (lr:LatentRelation)-[:SUPPORTED_BY]->(span)
        OPTIONAL MATCH (t:RLMTransition {document_id: $document_id})
        OPTIONAL MATCH (s:EntityState)
        WHERE s.transition_id = t.transition_id
        DETACH DELETE s, t, lr, rc, rr, span, tr, rh, arg, ef, r, ev, cl, rm, m, c, d
        """,
        {"document_id": document_id},
    )


def test_graph_ingest_vertical_slice() -> None:
    _skip_if_neo4j_port_is_closed()

    client = Neo4jClient(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        connection_timeout=1.0,
        max_transaction_retry_time=1.0,
    )

    try:
        try:
            client.execute_read("RETURN 1 AS ok")
        except Exception as exc:
            pytest.skip(f"Neo4j is not available at bolt://localhost:7687: {exc}")

        writer = GraphWriter(client)
        queries = GraphQueries(client)
        pipeline = IngestPipeline(
            model_adapter=FakeGraphModelAdapter(),
            graph_writer=writer,
            max_chunk_tokens=100,
            use_llm_rlm_update=False,
        )

        document_id = "neo4j_observation_demo_001"
        _cleanup_demo_document(client, document_id)

        result = asyncio.run(
            pipeline.run(
                document_id=document_id,
                text="Semyon saw old man. Old man held rifle.",
                title="Neo4j observation smoke",
            )
        )

        chunks = queries.get_document_chunks(document_id)
        assert len(chunks) == 1

        skeleton_counts = client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (c)-[:HAS_MENTION]->(rm:RawMention)
            OPTIONAL MATCH (c)-[:HAS_MENTION]->(m:Mention)
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(ef:EventFrame)
            OPTIONAL MATCH (span:EvidenceSpan)-[:IN_CHUNK]->(c)
            OPTIONAL MATCH (rm)-[:HAS_RESOLUTION_HYPOTHESIS]->(rh:ResolutionHypothesis)
            OPTIONAL MATCH (r:Relation {chunk_id: c.chunk_id})
            OPTIONAL MATCH (rc:RelationCandidate)-[:SUPPORTED_BY]->(span)
            RETURN count(DISTINCT rm) AS raw_mentions,
                   count(DISTINCT m) AS canonical_mentions,
                   count(DISTINCT ef) AS event_frames,
                   count(DISTINCT rh) AS resolution_hypotheses,
                   count(DISTINCT r) AS canonical_relations,
                   count(DISTINCT span) AS evidence_spans,
                   count(DISTINCT rc) AS relation_candidates
            """,
            {"document_id": document_id},
        )[0]
        assert skeleton_counts["raw_mentions"] == 3
        assert skeleton_counts["canonical_mentions"] == 3
        assert skeleton_counts["event_frames"] == 1
        assert skeleton_counts["resolution_hypotheses"] == 4
        assert skeleton_counts["canonical_relations"] == 1
        assert skeleton_counts["evidence_spans"] >= 1
        assert skeleton_counts["relation_candidates"] >= 1

        hypotheses = client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[:HAS_MENTION]->(rm:RawMention)
            MATCH (rm)-[:HAS_RESOLUTION_HYPOTHESIS]->(rh:ResolutionHypothesis)
            RETURN rm.text AS mention,
                   rh.hypothesis_type AS hypothesis_type,
                   rh.status AS status,
                   rh.candidate_entity_name AS candidate,
                   rh.reason AS reason
            ORDER BY rm.start_char ASC, rh.confidence DESC
            """,
            {"document_id": document_id},
        )
        assert any(
            row["mention"] == "old man"
            and row["status"] == "rejected"
            and row["candidate"] == "Semyon"
            and "event role conflict" in row["reason"]
            for row in hypotheses
        )
        assert any(
            row["mention"] == "old man"
            and row["status"] == "likely"
            and row["candidate"] == "unknown_old_man_1"
            for row in hypotheses
        )

        audit_rows = queries.get_resolution_audit_report(document_id)
        assert any(
            row["raw_mention_text"] == "old man"
            and row["mention_kind"] == "role_anchor"
            and row["entity_creation_decision"] == "create_entity"
            and row["is_terminal"] is True
            and row["final_entity_id"] is not None
            for row in audit_rows
        )

        event_args = client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[:HAS_EVENT]->(ef:EventFrame)
            MATCH (rm:RawMention)-[arg:ARGUMENT_OF]->(ef)
            RETURN ef.predicate AS predicate,
                   rm.text AS mention,
                   arg.role AS role,
                   arg.grounding_expectation AS grounding_expectation,
                   arg.resolution_status AS resolution_status
            ORDER BY role ASC
            """,
            {"document_id": document_id},
        )
        assert event_args == [
            {
                "predicate": "saw",
                "mention": "Semyon",
                "role": "agent",
                "grounding_expectation": "entity_expected",
                "resolution_status": "resolved",
            },
            {
                "predicate": "saw",
                "mention": "old man",
                "role": "patient",
                "grounding_expectation": "entity_expected",
                "resolution_status": "resolved",
            },
        ]
        event_argument_nodes = client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[:HAS_EVENT]->(ef:EventFrame)
            MATCH (arg:EventArgument)-[:ARGUMENT_OF]->(ef)
            OPTIONAL MATCH (arg)-[:OBSERVED_AS]->(rm:RawMention)
            OPTIONAL MATCH (arg)-[:RESOLVED_TO]->(e:Entity)
            RETURN arg.role AS role,
                   arg.argument_id AS argument_id,
                   rm.text AS observed_as,
                   e.canonical_name AS resolved_to
            ORDER BY role ASC
            """,
            {"document_id": document_id},
        )
        assert all(row["argument_id"] for row in event_argument_nodes)
        assert event_argument_nodes == [
            {
                "role": "agent",
                "argument_id": event_argument_nodes[0]["argument_id"],
                "observed_as": "Semyon",
                "resolved_to": "Semyon",
            },
            {
                "role": "patient",
                "argument_id": event_argument_nodes[1]["argument_id"],
                "observed_as": "old man",
                "resolved_to": "unknown_old_man_1",
            },
        ]

        canonical_relations = client.execute_read(
            """
            MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (r:Relation {chunk_id: c.chunk_id})
            MATCH (r)-[:FROM_ENTITY]->(source:Entity)
            MATCH (r)-[:TO_ENTITY]->(target:Entity)
            RETURN source.canonical_name AS source,
                   r.relation_type AS relation_type,
                   target.canonical_name AS target
            """,
            {"document_id": document_id},
        )
        assert canonical_relations == [
            {
                "source": "Semyon",
                "relation_type": "PERCEPTION",
                "target": "unknown_old_man_1",
            }
        ]

        semyon_context = queries.get_entity_context("Semyon", document_id=document_id)
        assert semyon_context is not None
        assert semyon_context["entity"]["canonical_name"] == "Semyon"
        assert any(
            mention["text"] == "Semyon"
            for mention in semyon_context["mentions"]
        )
        assert any(
            evidence["role"] == "source"
            and evidence["surface_form"] == "Semyon"
            and "Semyon saw old man" in evidence["text"]
            for evidence in semyon_context["evidence_spans"]
        )
        assert any(
            frame["predicate"] == "saw"
            and frame["role"] == "agent"
            and frame["participant_resolution_status"] == "resolved"
            and frame["grounding_expectation"] == "entity_expected"
            and frame["event_resolution_status"] == "complete"
            and frame["event_materialization_status"] == "valid"
            and "Semyon saw old man" in frame["evidence_text"]
            for frame in semyon_context["event_frames"]
        )
        assert any(
            relation["relation_type"] == "PERCEPTION"
            and relation["target_name"] == "unknown_old_man_1"
            for relation in semyon_context["outgoing_relations"]
        )
        assert any(
            candidate["symbolic_hint"] == "PERCEPTION"
            and candidate["target_name"] == "unknown_old_man_1"
            and "Semyon saw old man" in candidate["evidence_text"]
            for candidate in semyon_context["relation_candidates"]
        )

        event_audit = queries.get_event_coverage_audit(document_id)
        assert event_audit["totals"]["events_extracted"] == 1
        assert event_audit["totals"]["event_participants_count"] == 2
        assert event_audit["totals"]["participants_resolved"] == 2
        assert event_audit["totals"]["events_without_canonical_participants"] == 0

        event_observations = queries.get_entity_event_observations(
            "Semyon",
            document_id=document_id,
        )
        assert len(event_observations) == 1
        assert event_observations[0].entity_role == "agent"
        assert event_observations[0].normalized_predicate == "saw"
        assert event_observations[0].counterpart_entity_ids
        assert event_observations[0].unresolved_counterparts == []

        assert len(result["rlm_state"].entities) == 2
    finally:
        client.close()
