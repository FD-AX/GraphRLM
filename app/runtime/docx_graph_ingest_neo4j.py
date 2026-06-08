from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.agent_graph.chunk_builder import build_chunk_graph_workflow
from app.agent_graph.document_nodes import (
    update_rlm_state_from_patch,
    update_rlm_state_with_llm,
)
from app.core.chunking import chunk_text
from app.core.graph_models import ChunkNode, DocumentNode
from app.graph.neo4j_client import Neo4jClient
from app.graph.writer import GraphWriter
from app.llm.model_adapter import OpenAICompatibleModelAdapter
from app.rlm.state import EntityState, RLMState
from app.runtime.docx_reader import read_docx_text


def cleanup_document(client: Neo4jClient, document_id: str) -> None:
    chunk_rows = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        RETURN c.chunk_id AS chunk_id
        """,
        {"document_id": document_id},
    )

    writer = GraphWriter(client)
    for row in chunk_rows:
        writer.delete_chunk_subtree(row["chunk_id"])

    client.execute_write(
        """
        MATCH (d:Document {document_id: $document_id})
        DETACH DELETE d
        """,
        {"document_id": document_id},
    )


def graph_summary(client: Neo4jClient, document_id: str) -> dict:
    counts = client.execute_read(
        """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, collect(DISTINCT c) AS chunks
        OPTIONAL MATCH (m:Mention)-[:REFERS_TO]->(e:Entity)
        WHERE m.chunk_id IN [chunk IN chunks | chunk.chunk_id]
        OPTIONAL MATCH (cl:Claim)
        WHERE cl.chunk_id IN [chunk IN chunks | chunk.chunk_id]
        RETURN
            size(chunks) AS chunks,
            count(DISTINCT e) AS entities,
            count(DISTINCT m) AS mentions,
            count(DISTINCT cl) AS claims,
            size([chunk IN chunks WHERE chunk.extraction_status = 'DONE']) AS done_chunks,
            size([chunk IN chunks WHERE chunk.extraction_status = 'FAILED']) AS failed_chunks,
            size([chunk IN chunks WHERE chunk.extraction_status = 'PROCESSING']) AS processing_chunks,
            size([chunk IN chunks WHERE chunk.extraction_status = 'PENDING']) AS pending_chunks
        """,
        {"document_id": document_id},
    )[0]

    top_entities = client.execute_read(
        """
        MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(m:Mention)-[:REFERS_TO]->(e:Entity)
        RETURN e.canonical_name AS name, count(m) AS mentions
        ORDER BY mentions DESC, name ASC
        LIMIT 20
        """,
        {"document_id": document_id},
    )

    return {
        "document_id": document_id,
        "chunks": counts["chunks"],
        "entities": counts["entities"],
        "mentions": counts["mentions"],
        "claims": counts["claims"],
        "done_chunks": counts["done_chunks"],
        "failed_chunks": counts["failed_chunks"],
        "processing_chunks": counts["processing_chunks"],
        "pending_chunks": counts["pending_chunks"],
        "top_entities": top_entities,
    }


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_done_chunk_context(
    client: Neo4jClient,
    rlm_state: RLMState,
    chunk: ChunkNode,
) -> RLMState:
    rows = client.execute_read(
        """
        MATCH (t:RLMTransition {to_chunk_id: $chunk_id})
        MATCH (s:EntityState {transition_id: t.transition_id})
        RETURN s {
            .entity_id,
            .canonical_name,
            .attributes,
            .hypotheses,
            .evidence_refs,
            .confidence
        } AS entity_state
        """,
        {"chunk_id": chunk.chunk_id},
    )

    for row in rows:
        entity_state = EntityState(**row["entity_state"])
        rlm_state.entities[entity_state.entity_id] = entity_state

    rlm_state.current_chunk_index = max(rlm_state.current_chunk_index, chunk.index)
    if chunk.chunk_id not in rlm_state.recent_chunk_ids:
        rlm_state.recent_chunk_ids.append(chunk.chunk_id)
    rlm_state.recent_chunk_ids = rlm_state.recent_chunk_ids[-5:]

    return rlm_state


async def process_chunk(
    chunk: ChunkNode,
    rlm_state: RLMState,
    chunk_graph,
    writer: GraphWriter,
    rlm_model_adapter: OpenAICompatibleModelAdapter,
    use_llm_rlm_update: bool,
) -> RLMState:
    result = await chunk_graph.ainvoke(
        {
            "chunk": chunk,
            "rlm_state": rlm_state,
            "errors": [],
        }
    )

    state = {
        "current_chunk": chunk,
        "rlm_state": rlm_state,
        "graph_patch": result["graph_patch"],
        "errors": result.get("errors", []),
    }

    if use_llm_rlm_update:
        transition_result = await update_rlm_state_with_llm(
            state,
            rlm_model_adapter=rlm_model_adapter,
        )
    else:
        transition_result = update_rlm_state_from_patch(state)

    state.update(transition_result)

    writer.write_local_graph(state["graph_patch"])
    writer.write_rlm_transition(state["rlm_transition"])

    return state["rlm_state"]


async def ingest_docx_to_neo4j(
    docx_path: Path,
    uri: str,
    username: str,
    password: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    max_chunk_tokens: int,
    max_chunks: int | None,
    use_llm_rlm_update: bool,
    write_mode: str,
    chunk_timeout_seconds: int,
    continue_on_error: bool,
    document_id: str | None,
) -> dict:
    client = Neo4jClient(
        uri=uri,
        username=username,
        password=password,
        connection_timeout=2.0,
        max_transaction_retry_time=2.0,
    )

    try:
        resolved_document_id = document_id or docx_path.stem

        if write_mode == "overwrite":
            cleanup_document(client, resolved_document_id)

        writer = GraphWriter(client)
        model_adapter = OpenAICompatibleModelAdapter(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        chunk_graph = build_chunk_graph_workflow(model_adapter)

        document = DocumentNode(
            document_id=resolved_document_id,
            title=docx_path.stem,
            source_path=str(docx_path),
            metadata={},
        )
        writer.write_document(document)

        text_chunks = chunk_text(
            text=read_docx_text(docx_path),
            source_document_id=resolved_document_id,
            max_chunk_tokens=max_chunk_tokens,
        )
        if max_chunks is not None:
            text_chunks = text_chunks[:max_chunks]

        chunks = [
            ChunkNode(
                chunk_id=chunk.chunk_id,
                document_id=resolved_document_id,
                index=chunk.index,
                text=chunk.text,
                start_char=chunk.start_char or 0,
                end_char=chunk.end_char or 0,
                token_count=chunk.token_count,
                content_hash=content_hash(chunk.text),
            )
            for chunk in text_chunks
        ]

        rlm_state = RLMState(document_id=resolved_document_id)
        processed = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        for chunk in chunks:
            existing = writer.get_chunk_by_index(
                document_id=resolved_document_id,
                index=chunk.index,
            )

            if (
                write_mode != "append"
                and existing
                and existing.get("content_hash") != chunk.content_hash
            ):
                writer.delete_chunk_subtree(existing["chunk_id"])
                existing = None

            if (
                write_mode == "resume"
                and existing
                and existing.get("content_hash") == chunk.content_hash
                and existing.get("extraction_status") == "DONE"
            ):
                skipped += 1
                rlm_state = load_done_chunk_context(client, rlm_state, chunk)
                continue

            writer.write_chunk(chunk)
            writer.mark_chunk_processing(chunk.chunk_id)

            try:
                rlm_state = await asyncio.wait_for(
                    process_chunk(
                        chunk=chunk,
                        rlm_state=rlm_state,
                        chunk_graph=chunk_graph,
                        writer=writer,
                        rlm_model_adapter=model_adapter,
                        use_llm_rlm_update=use_llm_rlm_update,
                    ),
                    timeout=chunk_timeout_seconds,
                )
                writer.mark_chunk_done(chunk.chunk_id)
                processed += 1
            except Exception as exc:
                failed += 1
                error = f"chunk {chunk.index} failed: {type(exc).__name__}: {exc}"
                errors.append(error)
                writer.mark_chunk_failed(chunk.chunk_id, error)

                if not continue_on_error:
                    break

        summary = graph_summary(client, resolved_document_id)
        summary["processed"] = processed
        summary["skipped"] = skipped
        summary["failed"] = failed
        summary["errors"] = errors

        return summary
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", nargs="?", default=None)
    parser.add_argument("--path", dest="docx_path", default=None)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="gemma-3-4b-it")
    parser.add_argument("--llm-api-key", default="local")
    parser.add_argument("--max-chunk-tokens", type=int, default=1000)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--document-id", default=None)
    parser.add_argument(
        "--rlm-mode",
        choices=["llm", "deterministic"],
        default="llm",
    )
    parser.add_argument(
        "--write-mode",
        choices=["append", "resume", "overwrite"],
        default="resume",
    )
    parser.add_argument("--chunk-timeout-seconds", type=int, default=180)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    raw_docx_path = args.docx_path or args.docx
    docx_path = Path(raw_docx_path) if raw_docx_path else next(Path(".").glob("*.docx"))

    summary = asyncio.run(
        ingest_docx_to_neo4j(
            docx_path=docx_path,
            uri=args.uri,
            username=args.username,
            password=args.password,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            max_chunk_tokens=args.max_chunk_tokens,
            max_chunks=args.max_chunks,
            use_llm_rlm_update=args.rlm_mode == "llm",
            write_mode=args.write_mode,
            chunk_timeout_seconds=args.chunk_timeout_seconds,
            continue_on_error=args.continue_on_error,
            document_id=args.document_id,
        )
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
