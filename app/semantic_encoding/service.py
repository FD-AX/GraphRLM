from __future__ import annotations

from app.semantic_encoding.builder import GraphSemanticDocumentBuilder
from app.semantic_encoding.encoder import GraphSemanticEncoder
from app.semantic_encoding.index import GraphSemanticIndex
from app.semantic_encoding.materializer import GraphSemanticMaterializer
from app.semantic_encoding.models import (
    EncoderConfig,
    GraphSemanticDocument,
    GraphSemanticEmbedding,
    SemanticTraversalTrace,
)
from app.semantic_encoding.navigation import LatentGraphNavigator


def build_graph_semantic_documents(
    entity_snapshots,
    pair_snapshots,
) -> list[GraphSemanticDocument]:
    builder = GraphSemanticDocumentBuilder()
    documents: list[GraphSemanticDocument] = []
    for snapshot in entity_snapshots:
        documents.extend(builder.build_from_entity_snapshot(snapshot))
    for snapshot in pair_snapshots:
        documents.extend(builder.build_from_pair_snapshot(snapshot))
    return documents


def encode_graph_semantic_documents(
    documents: list[GraphSemanticDocument],
    config: EncoderConfig | None = None,
) -> tuple[GraphSemanticEncoder, list[GraphSemanticEmbedding], GraphSemanticIndex]:
    encoder = GraphSemanticEncoder(config=config)
    embeddings = [encoder.encode_document(document) for document in documents]
    index = GraphSemanticIndex(
        documents=documents,
        embeddings=embeddings,
        encoder=encoder,
    )
    return encoder, embeddings, index


def materialize_graph_semantic_documents(
    client,
    documents: list[GraphSemanticDocument],
    embeddings: list[GraphSemanticEmbedding],
) -> None:
    GraphSemanticMaterializer(client).materialize_documents(documents, embeddings)


def run_latent_graph_navigation(
    query: str,
    documents: list[GraphSemanticDocument],
    config: EncoderConfig | None = None,
    seed_top_k: int = 5,
) -> SemanticTraversalTrace:
    encoder, _, index = encode_graph_semantic_documents(documents, config=config)
    navigator = LatentGraphNavigator(
        index=index,
        encoder=encoder,
        config=config or encoder.config,
    )
    return navigator.traverse(query=query, seed_top_k=seed_top_k)
