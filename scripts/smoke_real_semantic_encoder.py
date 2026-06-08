from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticDocument,
    GraphSemanticEncoder,
    GraphSemanticIndex,
    HashingSemanticEncoder,
    SemanticTraversalCase,
    TransformerSemanticEncoder,
    evaluate_seed_retrieval,
)


def main() -> None:
    args = parse_args()
    config = EncoderConfig(
        projection_dim=None if args.backend == "transformer" else 256,
        embedding_dim=None,
        interaction_dim=64,
        transformer_model_name=args.model,
        active_profile_weight=0.0,
    )
    backend = make_backend(args, config)
    encoder = GraphSemanticEncoder(config=config, backend=backend)
    documents = make_documents()
    index = GraphSemanticIndex.build(documents, encoder)
    cases = make_cases()
    evaluation = evaluate_seed_retrieval(index, encoder, cases, top_k=args.top_k)

    print(
        json.dumps(
            {
                "backend": backend.encoder_name,
                "encoder_version": backend.encoder_version,
                "embedding_dim": len(index.embedding_for(documents[0].semantic_document_id)),
                "active_profile_weight": config.active_profile_weight,
                "metrics": {
                    "top_k": evaluation.top_k,
                    "mean_recall_at_k": evaluation.mean_recall_at_k,
                    "mean_mrr": evaluation.mean_mrr,
                    "mean_ndcg_at_k": evaluation.mean_ndcg_at_k,
                },
                "cases": [
                    {
                        "query": case.query,
                        "recall_at_k": result.recall_at_k,
                        "mrr": result.mrr,
                        "ndcg_at_k": result.ndcg_at_k,
                        "top_results": [
                            {
                                "semantic_document_id": item.semantic_document_id,
                                "owner_id": item.owner_id,
                                "owner_type": item.owner_type,
                                "score": item.score,
                                "evidence_span_ids": item.evidence_span_ids,
                                "text": item.text,
                                "top_contribution_components": item.contribution_top_components[:5],
                            }
                            for item in result.top_results
                        ],
                    }
                    for case, result in zip(cases, evaluation.case_results)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.assert_min_recall is not None and evaluation.mean_recall_at_k < args.assert_min_recall:
        raise SystemExit(
            f"mean_recall_at_k={evaluation.mean_recall_at_k:.3f} "
            f"is below required {args.assert_min_recall:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["transformer", "hashing"],
        default="transformer",
        help="Transformer is the real baseline. Hashing is only for fast mechanics tests.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--assert-min-recall", type=float, default=None)
    return parser.parse_args()


def make_backend(args: argparse.Namespace, config: EncoderConfig):
    if args.backend == "hashing":
        return HashingSemanticEncoder(dimensions=256)
    return TransformerSemanticEncoder(
        model_name=config.transformer_model_name,
        device=args.device,
        embedding_dim=config.embedding_dim,
    )


def make_documents() -> list[GraphSemanticDocument]:
    rows = [
        (
            "doc_love_protect",
            "pair_interaction",
            "pair:semyon:anna",
            ["semyon", "anna"],
            ["ev_protect"],
            ["span_protect"],
            "Semyon protected Anna from a stranger.",
            {"predicate": "protected", "source_role": "protector", "target_role": "protected"},
        ),
        (
            "doc_love_jealous",
            "pair_interaction",
            "pair:anna:semyon",
            ["anna", "semyon", "maria"],
            ["ev_jealous"],
            ["span_jealous"],
            "Anna was jealous of Semyon toward Maria.",
            {"predicate": "jealous", "emotion": "jealousy"},
        ),
        (
            "doc_love_avoid",
            "pair_interaction",
            "pair:semyon:anna",
            ["semyon", "anna"],
            ["ev_avoid"],
            ["span_avoid"],
            "After a quarrel, Semyon avoided Anna.",
            {"predicate": "avoided", "emotion": "conflict"},
        ),
        (
            "doc_forest_enter",
            "entity_event",
            "semyon",
            ["semyon"],
            ["ev_enter_forest"],
            ["span_forest"],
            "Semyon entered the forest.",
            {"predicate": "entered", "location": "forest"},
        ),
        (
            "doc_river_old_man",
            "entity_event",
            "old_man",
            ["old_man"],
            ["ev_live_river"],
            ["span_river"],
            "The old man lived near the river.",
            {"predicate": "lived", "location": "river"},
        ),
        (
            "doc_door_ivan",
            "entity_event",
            "ivan",
            ["ivan"],
            ["ev_repair_door"],
            ["span_door"],
            "Ivan repaired the door.",
            {"predicate": "repaired", "object": "door"},
        ),
        (
            "doc_warning_old_man",
            "pair_interaction",
            "pair:old_man:semyon",
            ["old_man", "semyon"],
            ["ev_warn"],
            ["span_warn"],
            "The old man warned Semyon.",
            {"predicate": "warned", "source_role": "speaker", "target_role": "addressee"},
        ),
        (
            "doc_warning_ignored",
            "entity_event",
            "semyon",
            ["semyon"],
            ["ev_ignore_warning"],
            ["span_ignore"],
            "Semyon ignored the warning.",
            {"predicate": "ignored", "object": "warning"},
        ),
    ]
    return [
        GraphSemanticDocument(
            document_id="semantic_encoder_eval_doc",
            semantic_document_id=document_id,
            owner_type=owner_type,
            owner_id=owner_id,
            source_entity_ids=source_entity_ids,
            event_ids=event_ids,
            evidence_span_ids=evidence_span_ids,
            source_chunk_ids=[f"chunk_{index + 1}"],
            text=text,
            structural_features=features,
            projection_version="semantic_encoder_eval_v1",
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


def make_cases() -> list[SemanticTraversalCase]:
    return [
        SemanticTraversalCase(
            query="Как развивались любовные отношения Семёна?",
            expected_document_ids=[
                "doc_love_protect",
                "doc_love_jealous",
                "doc_love_avoid",
            ],
            expected_seed_owner_ids=["pair:semyon:anna", "pair:anna:semyon"],
            expected_evidence_span_ids=["span_protect", "span_jealous", "span_avoid"],
        ),
        SemanticTraversalCase(
            query="Кто предупредил Семёна и что произошло потом?",
            expected_document_ids=["doc_warning_old_man", "doc_warning_ignored"],
            expected_seed_owner_ids=["pair:old_man:semyon", "semyon"],
            expected_evidence_span_ids=["span_warn", "span_ignore"],
        ),
        SemanticTraversalCase(
            query="Кто находился в лесу?",
            expected_document_ids=["doc_forest_enter"],
            expected_seed_owner_ids=["semyon"],
            expected_evidence_span_ids=["span_forest"],
        ),
    ]


if __name__ == "__main__":
    main()
