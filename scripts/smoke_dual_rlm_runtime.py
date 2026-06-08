from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dual_rlm import (
    DualRLMConfig,
    GraphRLMArm,
    GraphViewRef,
    ImmutableTextStore,
    IndependentDualRLMRuntime,
    SourceChunk,
    TextRLMArm,
)
from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticDocument,
    GraphSemanticEncoder,
    GraphSemanticIndex,
    HashingSemanticEncoder,
    TransformerSemanticEncoder,
)


def main() -> None:
    args = parse_args()
    config = EncoderConfig(
        projection_dim=None if args.backend == "transformer" else 256,
        embedding_dim=None,
        interaction_dim=64,
        transformer_model_name=args.model,
    )
    backend = (
        HashingSemanticEncoder(dimensions=256)
        if args.backend == "hashing"
        else TransformerSemanticEncoder(
            model_name=config.transformer_model_name,
            device=args.device,
            embedding_dim=config.embedding_dim,
        )
    )
    encoder = GraphSemanticEncoder(config=config, backend=backend)
    graph_index = GraphSemanticIndex.build(make_graph_documents(), encoder)
    graph_view = GraphViewRef(
        document_id="dual_rlm_doc",
        graph_version="graph_v1",
        projection_version="projection_v1",
        encoder_version=backend.encoder_version,
    )
    runtime = IndependentDualRLMRuntime(
        graph_arm=GraphRLMArm(
            index=graph_index,
            graph_view=graph_view,
            config=DualRLMConfig(graph_top_k=3),
        ),
        text_arm=TextRLMArm(
            text_store=ImmutableTextStore(make_chunks()),
            config=DualRLMConfig(text_top_k=2, max_text_rounds=2, text_window_radius=1),
        ),
    )
    result = runtime.run(
        query=args.query,
        graph_view=graph_view,
        run_id="dual_rlm_smoke",
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["transformer", "hashing"],
        default="transformer",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--query",
        default="How did Semyon and Anna's relationship lead to conflict?",
    )
    return parser.parse_args()


def make_graph_documents() -> list[GraphSemanticDocument]:
    rows = [
        (
            "doc_graph_semyon_anna",
            "pair_interaction",
            "pair:semyon:anna",
            ["entity_semyon", "entity_anna"],
            ["ev_protect_anna", "ev_anna_jealous"],
            ["span_graph_anna"],
            "Semyon protected Anna. Anna was jealous of Semyon.",
            {"line": "romance"},
        ),
        (
            "doc_graph_semyon_ivan",
            "pair_interaction",
            "pair:semyon:ivan",
            ["entity_semyon", "entity_ivan"],
            ["ev_ivan_conflict"],
            ["span_graph_ivan"],
            "Ivan confronted Semyon after a dispute.",
            {"line": "conflict"},
        ),
    ]
    return [
        GraphSemanticDocument(
            document_id="dual_rlm_doc",
            semantic_document_id=document_id,
            owner_type=owner_type,
            owner_id=owner_id,
            source_entity_ids=source_entity_ids,
            event_ids=event_ids,
            evidence_span_ids=evidence_span_ids,
            source_chunk_ids=[f"chunk_{index + 1}"],
            text=text,
            structural_features=features,
            projection_version="projection_v1",
            content_hash=sha256(text.encode("utf-8")).hexdigest(),
        )
        for index, (
            document_id,
            owner_type,
            owner_id,
            source_entity_ids,
            event_ids,
            evidence_span_ids,
            text,
            features,
        ) in enumerate(rows)
    ]


def make_chunks() -> list[SourceChunk]:
    texts = [
        "Semyon protected Anna when a stranger threatened her.",
        "Anna became jealous when Semyon spoke with Maria.",
        "Ivan approached Anna again, and Semyon saw them together.",
        "After that, Ivan confronted Semyon and the quarrel became a fight.",
    ]
    chunks = []
    cursor = 0
    for index, text in enumerate(texts, start=1):
        chunks.append(
            SourceChunk(
                document_id="dual_rlm_doc",
                chunk_id=f"chunk_{index}",
                text=text,
                start_char=cursor,
                end_char=cursor + len(text),
            )
        )
        cursor += len(text) + 1
    return chunks


if __name__ == "__main__":
    main()
