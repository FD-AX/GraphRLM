import hashlib

from app.agent_graph.document_builder import build_document_graph_workflow
from app.core.chunking import chunk_text
from app.core.graph_models import ChunkNode, DocumentNode
from app.graph.writer import GraphWriter
from app.llm.model_adapter import ModelAdapter
from app.rlm.state import RLMState


class IngestPipeline:
    def __init__(
        self,
        model_adapter: ModelAdapter,
        graph_writer: GraphWriter,
        max_chunk_tokens: int | None = None,
        rlm_model_adapter: ModelAdapter | None = None,
        max_chunks: int | None = None,
        use_llm_rlm_update: bool = True,
    ):
        self.graph_writer = graph_writer
        self.max_chunk_tokens = max_chunk_tokens
        self.max_chunks = max_chunks
        self.graph = build_document_graph_workflow(
            model_adapter=model_adapter,
            rlm_model_adapter=rlm_model_adapter or model_adapter,
            graph_writer=graph_writer,
            use_llm_rlm_update=use_llm_rlm_update,
        )

    async def run(
        self,
        document_id: str,
        text: str,
        title: str | None = None,
        source_path: str | None = None,
    ) -> dict:
        document = DocumentNode(
            document_id=document_id,
            title=title,
            source_path=source_path,
            metadata={},
        )

        self.graph_writer.write_document(document)

        text_chunks = chunk_text(
            text=text,
            source_document_id=document_id,
            max_chunk_tokens=self.max_chunk_tokens,
        )
        if self.max_chunks is not None:
            text_chunks = text_chunks[: self.max_chunks]

        chunks = [
            ChunkNode(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                index=chunk.index,
                text=chunk.text,
                start_char=chunk.start_char or 0,
                end_char=chunk.end_char or 0,
                token_count=chunk.token_count,
                content_hash=hashlib.sha256(
                    chunk.text.encode("utf-8")
                ).hexdigest(),
            )
            for chunk in text_chunks
        ]

        initial_state = {
            "document": document,
            "chunks": chunks,
            "current_chunk_index": 0,
            "rlm_state": RLMState(document_id=document_id),
            "errors": [],
            "done": False,
        }

        return await self.graph.ainvoke(initial_state)
