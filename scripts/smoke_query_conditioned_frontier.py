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
    FrontierScoringWeights,
    GraphSemanticDocument,
    GraphSemanticEncoder,
    GraphSemanticIndex,
    HashingSemanticEncoder,
    SemanticTraversalCase,
    TransformerSemanticEncoder,
    evaluate_active_profile_clusters,
    evaluate_navigation,
    evaluate_seed_retrieval,
)


def main() -> None:
    args = parse_args()
    config = EncoderConfig(
        projection_dim=None if args.backend == "transformer" else 256,
        embedding_dim=None,
        interaction_dim=64,
        transformer_model_name=args.model,
        local_frontier_cap=100,
    )
    backend = make_backend(args, config)
    encoder = GraphSemanticEncoder(config=config, backend=backend)
    documents = make_documents()
    index = GraphSemanticIndex.build(documents, encoder)
    labeled_cases = make_labeled_cases()
    cases = [case for _, case in labeled_cases]

    seed_eval = evaluate_seed_retrieval(index, encoder, cases, top_k=args.top_k)
    navigation_results = [
        evaluate_navigation(index, encoder, cases, weights, top_k=args.top_k)
        for weights in make_modes()
    ]
    cluster_eval = evaluate_active_profile_clusters(index, encoder, labeled_cases)

    payload = {
        "benchmark": "query_conditioned_frontier_v1",
        "backend": backend.encoder_name,
        "encoder_version": backend.encoder_version,
        "embedding_dim": len(index.embedding_for("doc_entity_semyon")),
        "case_count": len(cases),
        "seed_retrieval": {
            "mean_recall_at_k": seed_eval.mean_recall_at_k,
            "mean_mrr": seed_eval.mean_mrr,
            "mean_ndcg_at_k": seed_eval.mean_ndcg_at_k,
        },
        "navigation_modes": [
            {
                "mode": result.mode,
                "mean_next_hop_accuracy": result.mean_next_hop_accuracy,
                "mean_next_hop_mrr": result.mean_next_hop_mrr,
                "mean_next_hop_ndcg_at_k": result.mean_next_hop_ndcg_at_k,
                "mean_path_recall_at_k": result.mean_path_recall_at_k,
                "mean_evidence_recall_at_k": result.mean_evidence_recall_at_k,
                "mean_local_observation_recall_at_k": result.mean_local_observation_recall_at_k,
                "errors": [
                    {
                        "query": item.query,
                        "selected_owner_id": item.selected_owner_id,
                        "selected_document_id": item.selected_document_id,
                        "top_candidates": [
                            {
                                "owner_id": candidate.owner_id,
                                "semantic_document_id": candidate.semantic_document_id,
                                "score": candidate.score,
                                "query_similarity": candidate.query_similarity,
                                "active_profile_similarity": candidate.active_profile_similarity,
                                "structural_confidence": candidate.structural_confidence,
                                "evidence_quality": candidate.evidence_quality,
                                "novelty": candidate.novelty,
                                "text": candidate.text,
                            }
                            for candidate in item.ranked_candidates[:3]
                        ],
                    }
                    for item in result.case_results
                    if item.next_hop_accuracy == 0.0
                ],
            }
            for result in navigation_results
        ],
        "active_profile_clusters": {
            "same_label_mean_similarity": cluster_eval.same_label_mean_similarity,
            "different_label_mean_similarity": cluster_eval.different_label_mean_similarity,
            "separation_margin": cluster_eval.separation_margin,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    best_accuracy = max(
        result.mean_next_hop_accuracy for result in navigation_results
    )
    if args.assert_min_next_hop_accuracy is not None:
        if best_accuracy < args.assert_min_next_hop_accuracy:
            raise SystemExit(
                f"best next-hop accuracy={best_accuracy:.3f} is below "
                f"{args.assert_min_next_hop_accuracy:.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["transformer", "hashing"],
        default="transformer",
        help="Transformer is the real semantic benchmark. Hashing is mechanics only.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--assert-min-next-hop-accuracy", type=float, default=None)
    return parser.parse_args()


def make_backend(args: argparse.Namespace, config: EncoderConfig):
    if args.backend == "hashing":
        return HashingSemanticEncoder(dimensions=256)
    return TransformerSemanticEncoder(
        model_name=config.transformer_model_name,
        device=args.device,
        embedding_dim=config.embedding_dim,
    )


def make_modes() -> list[FrontierScoringWeights]:
    modes = [
        FrontierScoringWeights(
            name="query_cosine",
            query_similarity_weight=1.0,
        ),
        FrontierScoringWeights(
            name="query_structure",
            query_similarity_weight=0.70,
            structural_confidence_weight=0.15,
            evidence_quality_weight=0.10,
            novelty_weight=0.05,
        ),
    ]
    for active_weight in [0.0, 0.05, 0.10, 0.20, 0.30]:
        modes.append(
            FrontierScoringWeights(
                name=f"query_structure_active_{active_weight:.2f}",
                query_similarity_weight=0.70,
                structural_confidence_weight=0.15,
                evidence_quality_weight=0.10,
                novelty_weight=0.05,
                active_profile_weight=active_weight,
            )
        )
    return modes


def make_documents() -> list[GraphSemanticDocument]:
    rows = [
        (
            "doc_entity_semyon",
            "entity",
            "entity_semyon",
            ["entity_semyon"],
            [],
            [],
            "Semyon. Aliases: hunter. Roles: hunter, traveler, armed man.",
            {"kind": "identity"},
        ),
        (
            "doc_anna_protect",
            "pair_interaction",
            "entity_anna",
            ["entity_semyon", "entity_anna"],
            ["ev_protect_anna"],
            ["span_anna_protect"],
            "Semyon protected Anna from a stranger and cared about her safety.",
            {"line": "romance", "predicate": "protected"},
        ),
        (
            "doc_anna_jealous",
            "pair_interaction",
            "entity_anna",
            ["entity_semyon", "entity_anna"],
            ["ev_anna_jealous"],
            ["span_anna_jealous"],
            "Anna was jealous of Semyon and Maria, revealing emotional attachment.",
            {"line": "romance", "predicate": "jealous"},
        ),
        (
            "doc_anna_quarrel",
            "pair_interaction",
            "entity_anna",
            ["entity_semyon", "entity_anna"],
            ["ev_anna_quarrel"],
            ["span_anna_quarrel"],
            "After a quarrel, Semyon avoided Anna but kept thinking about her.",
            {"line": "romance", "predicate": "avoided"},
        ),
        (
            "doc_old_man_warn",
            "pair_interaction",
            "entity_old_man",
            ["entity_semyon", "entity_old_man"],
            ["ev_old_man_warn"],
            ["span_old_man_warn"],
            "The old man warned Semyon not to continue toward the lights.",
            {"line": "warning", "predicate": "warned"},
        ),
        (
            "doc_old_man_ignored",
            "pair_interaction",
            "entity_old_man",
            ["entity_semyon", "entity_old_man"],
            ["ev_warning_ignored"],
            ["span_warning_ignored"],
            "Semyon ignored the old man's warning and walked on.",
            {"line": "warning", "predicate": "ignored"},
        ),
        (
            "doc_old_man_cautioned_hunter",
            "pair_interaction",
            "entity_old_man",
            ["entity_semyon", "entity_old_man"],
            ["ev_caution_hunter"],
            ["span_caution_hunter"],
            "The elder cautioned the hunter about danger ahead.",
            {"line": "warning", "predicate": "cautioned"},
        ),
        (
            "doc_forest_enter",
            "entity_event",
            "context_forest",
            ["entity_semyon", "context_forest"],
            ["ev_enter_forest"],
            ["span_enter_forest"],
            "Semyon entered the forest and followed a narrow path.",
            {"line": "location", "predicate": "entered", "location": "forest"},
        ),
        (
            "doc_forest_lights",
            "entity_event",
            "context_forest",
            ["entity_semyon", "context_forest"],
            ["ev_lights_forest"],
            ["span_lights_forest"],
            "The hunter walked through the dark forest toward distant lights.",
            {"line": "location", "predicate": "walked", "location": "forest"},
        ),
        (
            "doc_forest_river",
            "entity_event",
            "context_forest",
            ["entity_semyon", "context_forest"],
            ["ev_river_path"],
            ["span_river_path"],
            "Near the river path, Semyon remained inside the forest.",
            {"line": "location", "predicate": "remained", "location": "river path"},
        ),
        (
            "doc_ivan_rival",
            "pair_interaction",
            "entity_ivan",
            ["entity_semyon", "entity_ivan"],
            ["ev_ivan_rival"],
            ["span_ivan_rival"],
            "Ivan rivaled Semyon and competed with him over Anna.",
            {"line": "conflict", "predicate": "rivaled"},
        ),
        (
            "doc_ivan_confront",
            "pair_interaction",
            "entity_ivan",
            ["entity_semyon", "entity_ivan"],
            ["ev_ivan_confront"],
            ["span_ivan_confront"],
            "Ivan confronted Semyon after the dispute became open conflict.",
            {"line": "conflict", "predicate": "confronted"},
        ),
        (
            "doc_ivan_jealous",
            "pair_interaction",
            "entity_ivan",
            ["entity_semyon", "entity_ivan"],
            ["ev_ivan_jealous"],
            ["span_ivan_jealous"],
            "Ivan was jealous of Semyon and challenged him.",
            {"line": "conflict", "predicate": "challenged"},
        ),
        (
            "doc_rifle_carry",
            "entity_event",
            "object_rifle",
            ["entity_semyon", "object_rifle"],
            ["ev_rifle_carry"],
            ["span_rifle_carry"],
            "Semyon carried his rifle while entering the forest.",
            {"line": "instrument", "predicate": "carried", "object": "rifle"},
        ),
        (
            "doc_rifle_weapon",
            "entity_event",
            "object_rifle",
            ["entity_semyon", "object_rifle"],
            ["ev_rifle_weapon"],
            ["span_rifle_weapon"],
            "The rifle was Semyon's hunting weapon.",
            {"line": "instrument", "predicate": "armed_with", "object": "rifle"},
        ),
        (
            "doc_rifle_protection",
            "entity_event",
            "object_rifle",
            ["entity_semyon", "object_rifle"],
            ["ev_rifle_protection"],
            ["span_rifle_protection"],
            "The hunter used the weapon for protection.",
            {"line": "instrument", "predicate": "used", "object": "weapon"},
        ),
    ]
    return [
        GraphSemanticDocument(
            document_id="query_conditioned_frontier_doc",
            semantic_document_id=document_id,
            owner_type=owner_type,
            owner_id=owner_id,
            source_entity_ids=source_entity_ids,
            event_ids=event_ids,
            evidence_span_ids=evidence_span_ids,
            source_chunk_ids=[f"chunk_{index + 1}"],
            text=text,
            structural_features=features,
            projection_version="query_conditioned_frontier_v1",
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


def make_labeled_cases() -> list[tuple[str, SemanticTraversalCase]]:
    cases = []
    for label, next_owner, expected_docs, evidence_ids, queries in [
        (
            "love",
            "entity_anna",
            ["doc_anna_protect", "doc_anna_jealous", "doc_anna_quarrel"],
            ["span_anna_protect", "span_anna_jealous", "span_anna_quarrel"],
            [
                "How did Semyon's romantic relationship develop?",
                "Which relationship involved jealousy and care around Semyon?",
                "Who was emotionally attached to Semyon?",
                "Whose feelings caused conflict around Semyon?",
                "Who did Semyon protect as someone close?",
            ],
        ),
        (
            "warning",
            "entity_old_man",
            ["doc_old_man_warn", "doc_old_man_ignored", "doc_old_man_cautioned_hunter"],
            ["span_old_man_warn", "span_warning_ignored", "span_caution_hunter"],
            [
                "Who warned Semyon?",
                "Who told Semyon not to continue?",
                "What elder tried to stop Semyon?",
                "Who gave Semyon a warning that he ignored?",
                "Who cautioned the hunter?",
            ],
        ),
        (
            "location",
            "context_forest",
            ["doc_forest_enter", "doc_forest_lights", "doc_forest_river"],
            ["span_enter_forest", "span_lights_forest", "span_river_path"],
            [
                "Where was Semyon located?",
                "Where did Semyon go with the rifle?",
                "Which place surrounded Semyon?",
                "Where did the hunter walk?",
                "What location was connected to Semyon's path?",
            ],
        ),
        (
            "conflict",
            "entity_ivan",
            ["doc_ivan_rival", "doc_ivan_confront", "doc_ivan_jealous"],
            ["span_ivan_rival", "span_ivan_confront", "span_ivan_jealous"],
            [
                "Who rivaled Semyon?",
                "Who fought with Semyon over Anna?",
                "Which man was jealous of Semyon?",
                "Who confronted Semyon?",
                "Who was in conflict with Semyon?",
            ],
        ),
        (
            "instrument",
            "object_rifle",
            ["doc_rifle_carry", "doc_rifle_weapon", "doc_rifle_protection"],
            ["span_rifle_carry", "span_rifle_weapon", "span_rifle_protection"],
            [
                "What was Semyon armed with?",
                "What object did Semyon carry?",
                "Which weapon belonged to Semyon?",
                "What did the hunter use to protect himself?",
                "What item connected Semyon to hunting?",
            ],
        ),
    ]:
        for query in queries:
            cases.append(
                (
                    label,
                    SemanticTraversalCase(
                        query=query,
                        expected_seed_owner_ids=["entity_semyon"],
                        expected_document_ids=expected_docs,
                        current_owner_id="entity_semyon",
                        relevant_next_owner_ids=[next_owner],
                        irrelevant_next_owner_ids=[
                            owner
                            for owner in [
                                "entity_anna",
                                "entity_old_man",
                                "context_forest",
                                "entity_ivan",
                                "object_rifle",
                            ]
                            if owner != next_owner
                        ],
                        expected_evidence_span_ids=evidence_ids,
                    ),
                )
            )
    return cases


if __name__ == "__main__":
    main()
