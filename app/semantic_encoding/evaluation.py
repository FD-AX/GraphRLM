from __future__ import annotations

import math

from pydantic import BaseModel, Field

from app.semantic_encoding.encoder import GraphSemanticEncoder
from app.semantic_encoding.encoder import cosine_similarity
from app.semantic_encoding.index import GraphSemanticIndex
from app.semantic_encoding.models import GraphSemanticDocument, SemanticSearchResult


class SemanticTraversalCase(BaseModel):
    query: str
    expected_seed_owner_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    current_owner_id: str | None = None
    relevant_next_owner_ids: list[str] = Field(default_factory=list)
    irrelevant_next_owner_ids: list[str] = Field(default_factory=list)
    expected_evidence_span_ids: list[str] = Field(default_factory=list)


class SeedRetrievalCaseResult(BaseModel):
    query: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    top_results: list[SemanticSearchResult]


class SeedRetrievalEvaluationResult(BaseModel):
    top_k: int
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    case_results: list[SeedRetrievalCaseResult]


class FrontierScoringWeights(BaseModel):
    name: str
    query_similarity_weight: float = 1.0
    structural_confidence_weight: float = 0.0
    evidence_quality_weight: float = 0.0
    novelty_weight: float = 0.0
    active_profile_weight: float = 0.0


class NextHopCandidateScore(BaseModel):
    semantic_document_id: str
    owner_id: str
    score: float
    query_similarity: float
    active_profile_similarity: float
    structural_confidence: float
    evidence_quality: float
    novelty: float
    evidence_span_ids: list[str] = Field(default_factory=list)
    text: str


class NavigationCaseResult(BaseModel):
    query: str
    mode: str
    active_profile_weight: float
    next_hop_accuracy: float
    next_hop_mrr: float
    next_hop_ndcg_at_k: float
    path_recall_at_k: float
    evidence_recall_at_k: float
    local_observation_recall_at_k: float
    selected_owner_id: str | None
    selected_document_id: str | None
    ranked_candidates: list[NextHopCandidateScore] = Field(default_factory=list)
    active_profile_cluster: str | None = None


class NavigationEvaluationResult(BaseModel):
    mode: str
    top_k: int
    mean_next_hop_accuracy: float
    mean_next_hop_mrr: float
    mean_next_hop_ndcg_at_k: float
    mean_path_recall_at_k: float
    mean_evidence_recall_at_k: float
    mean_local_observation_recall_at_k: float
    case_results: list[NavigationCaseResult]


class ActiveProfileClusterResult(BaseModel):
    same_label_mean_similarity: float
    different_label_mean_similarity: float
    separation_margin: float
    pairs: list[dict] = Field(default_factory=list)


def evaluate_seed_retrieval(
    index: GraphSemanticIndex,
    encoder: GraphSemanticEncoder,
    cases: list[SemanticTraversalCase],
    top_k: int = 5,
) -> SeedRetrievalEvaluationResult:
    case_results = []
    for case in cases:
        query_embedding = encoder.encode_query(case.query)
        results = index.search(query_embedding, top_k=top_k)
        relevant_ids = set(case.expected_document_ids)
        relevant_owner_ids = set(case.expected_seed_owner_ids)
        case_results.append(
            SeedRetrievalCaseResult(
                query=case.query,
                recall_at_k=recall_at_k(results, relevant_ids, relevant_owner_ids),
                mrr=mean_reciprocal_rank(results, relevant_ids, relevant_owner_ids),
                ndcg_at_k=ndcg_at_k(results, relevant_ids, relevant_owner_ids, top_k),
                top_results=results,
            )
        )

    count = len(case_results) or 1
    return SeedRetrievalEvaluationResult(
        top_k=top_k,
        mean_recall_at_k=sum(item.recall_at_k for item in case_results) / count,
        mean_mrr=sum(item.mrr for item in case_results) / count,
        mean_ndcg_at_k=sum(item.ndcg_at_k for item in case_results) / count,
        case_results=case_results,
    )


def evaluate_navigation(
    index: GraphSemanticIndex,
    encoder: GraphSemanticEncoder,
    cases: list[SemanticTraversalCase],
    weights: FrontierScoringWeights,
    top_k: int = 5,
) -> NavigationEvaluationResult:
    case_results = []
    for case in cases:
        ranked = score_frontier_case(index, encoder, case, weights)
        top_ranked = ranked[:top_k]
        relevant_owner_ids = set(case.relevant_next_owner_ids)
        relevant_document_ids = set(case.expected_document_ids)
        selected = top_ranked[0] if top_ranked else None
        case_results.append(
            NavigationCaseResult(
                query=case.query,
                mode=weights.name,
                active_profile_weight=weights.active_profile_weight,
                next_hop_accuracy=(
                    1.0 if selected and selected.owner_id in relevant_owner_ids else 0.0
                ),
                next_hop_mrr=_candidate_mrr(top_ranked, relevant_owner_ids),
                next_hop_ndcg_at_k=_candidate_ndcg(top_ranked, relevant_owner_ids, top_k),
                path_recall_at_k=_candidate_owner_recall(top_ranked, relevant_owner_ids),
                evidence_recall_at_k=_candidate_evidence_recall(
                    top_ranked,
                    set(case.expected_evidence_span_ids),
                ),
                local_observation_recall_at_k=_candidate_document_recall(
                    top_ranked,
                    relevant_document_ids,
                ),
                selected_owner_id=selected.owner_id if selected else None,
                selected_document_id=selected.semantic_document_id if selected else None,
                ranked_candidates=top_ranked,
            )
        )

    count = len(case_results) or 1
    return NavigationEvaluationResult(
        mode=weights.name,
        top_k=top_k,
        mean_next_hop_accuracy=sum(item.next_hop_accuracy for item in case_results) / count,
        mean_next_hop_mrr=sum(item.next_hop_mrr for item in case_results) / count,
        mean_next_hop_ndcg_at_k=sum(item.next_hop_ndcg_at_k for item in case_results) / count,
        mean_path_recall_at_k=sum(item.path_recall_at_k for item in case_results) / count,
        mean_evidence_recall_at_k=sum(item.evidence_recall_at_k for item in case_results) / count,
        mean_local_observation_recall_at_k=sum(
            item.local_observation_recall_at_k for item in case_results
        )
        / count,
        case_results=case_results,
    )


def score_frontier_case(
    index: GraphSemanticIndex,
    encoder: GraphSemanticEncoder,
    case: SemanticTraversalCase,
    weights: FrontierScoringWeights,
) -> list[NextHopCandidateScore]:
    seed_document = _seed_document_for_owner(index, case.current_owner_id)
    if seed_document is None:
        return []

    query_embedding = encoder.encode_query(case.query)
    seed_embedding = index.embedding_for(seed_document.semantic_document_id)
    seed_profile = encoder.interaction_profile(query_embedding, seed_embedding)
    frontier_ids = index.frontier_document_ids(
        seed_document,
        cap=index.encoder.config.local_frontier_cap,
    )
    candidates = []
    for document_id in frontier_ids:
        document = index.document_for(document_id)
        if document.owner_id == seed_document.owner_id:
            continue
        embedding = index.embedding_for(document_id)
        candidate_profile = encoder.interaction_profile(query_embedding, embedding)
        query_similarity = cosine_similarity(query_embedding, embedding)
        active_profile_similarity = cosine_similarity(seed_profile, candidate_profile)
        structural_confidence = _structural_confidence(seed_document, document)
        evidence_quality = 1.0 if document.evidence_span_ids else 0.4
        novelty = _novelty(seed_document, document)
        score = _weighted_score(
            weights,
            query_similarity=query_similarity,
            active_profile_similarity=active_profile_similarity,
            structural_confidence=structural_confidence,
            evidence_quality=evidence_quality,
            novelty=novelty,
        )
        candidates.append(
            NextHopCandidateScore(
                semantic_document_id=document.semantic_document_id,
                owner_id=document.owner_id,
                score=score,
                query_similarity=query_similarity,
                active_profile_similarity=active_profile_similarity,
                structural_confidence=structural_confidence,
                evidence_quality=evidence_quality,
                novelty=novelty,
                evidence_span_ids=document.evidence_span_ids,
                text=document.text,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def evaluate_active_profile_clusters(
    index: GraphSemanticIndex,
    encoder: GraphSemanticEncoder,
    labeled_cases: list[tuple[str, SemanticTraversalCase]],
) -> ActiveProfileClusterResult:
    profiles = []
    for label, case in labeled_cases:
        seed_document = _seed_document_for_owner(index, case.current_owner_id)
        if seed_document is None:
            continue
        query_embedding = encoder.encode_query(case.query)
        seed_embedding = index.embedding_for(seed_document.semantic_document_id)
        profiles.append(
            (
                label,
                case.query,
                encoder.interaction_profile(query_embedding, seed_embedding),
            )
        )

    same = []
    different = []
    pairs = []
    for left_index, (left_label, left_query, left_profile) in enumerate(profiles):
        for right_label, right_query, right_profile in profiles[left_index + 1 :]:
            similarity = cosine_similarity(left_profile, right_profile)
            pair = {
                "left_label": left_label,
                "right_label": right_label,
                "left_query": left_query,
                "right_query": right_query,
                "similarity": similarity,
            }
            pairs.append(pair)
            if left_label == right_label:
                same.append(similarity)
            else:
                different.append(similarity)

    same_mean = sum(same) / len(same) if same else 0.0
    different_mean = sum(different) / len(different) if different else 0.0
    return ActiveProfileClusterResult(
        same_label_mean_similarity=same_mean,
        different_label_mean_similarity=different_mean,
        separation_margin=same_mean - different_mean,
        pairs=pairs,
    )


def recall_at_k(
    results: list[SemanticSearchResult],
    relevant_document_ids: set[str],
    relevant_owner_ids: set[str],
) -> float:
    expected_count = len(relevant_document_ids | relevant_owner_ids)
    if expected_count == 0:
        return 1.0
    matched = _matched_keys(results, relevant_document_ids, relevant_owner_ids)
    return min(1.0, len(matched) / expected_count)


def mean_reciprocal_rank(
    results: list[SemanticSearchResult],
    relevant_document_ids: set[str],
    relevant_owner_ids: set[str],
) -> float:
    for index, result in enumerate(results, start=1):
        if _is_relevant(result, relevant_document_ids, relevant_owner_ids):
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    results: list[SemanticSearchResult],
    relevant_document_ids: set[str],
    relevant_owner_ids: set[str],
    top_k: int,
) -> float:
    gains = [
        1.0 if _is_relevant(result, relevant_document_ids, relevant_owner_ids) else 0.0
        for result in results[:top_k]
    ]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_relevant = min(top_k, len(relevant_document_ids | relevant_owner_ids))
    if ideal_relevant == 0:
        return 1.0
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_relevant))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _seed_document_for_owner(
    index: GraphSemanticIndex,
    owner_id: str | None,
) -> GraphSemanticDocument | None:
    if owner_id is None:
        return None
    matches = [
        document
        for document in index.documents.values()
        if document.owner_id == owner_id and document.owner_type == "entity"
    ]
    if not matches:
        matches = [
            document
            for document in index.documents.values()
            if document.owner_id == owner_id
        ]
    return sorted(matches, key=lambda item: item.semantic_document_id)[0] if matches else None


def _weighted_score(
    weights: FrontierScoringWeights,
    *,
    query_similarity: float,
    active_profile_similarity: float,
    structural_confidence: float,
    evidence_quality: float,
    novelty: float,
) -> float:
    weighted_sum = (
        weights.query_similarity_weight * ((query_similarity + 1.0) / 2.0)
        + weights.active_profile_weight * ((active_profile_similarity + 1.0) / 2.0)
        + weights.structural_confidence_weight * structural_confidence
        + weights.evidence_quality_weight * evidence_quality
        + weights.novelty_weight * novelty
    )
    total_weight = (
        weights.query_similarity_weight
        + weights.active_profile_weight
        + weights.structural_confidence_weight
        + weights.evidence_quality_weight
        + weights.novelty_weight
    )
    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight


def _structural_confidence(
    seed_document: GraphSemanticDocument,
    candidate: GraphSemanticDocument,
) -> float:
    shared_entities = set(seed_document.source_entity_ids) & set(candidate.source_entity_ids)
    shared_events = set(seed_document.event_ids) & set(candidate.event_ids)
    shared_evidence = set(seed_document.evidence_span_ids) & set(candidate.evidence_span_ids)
    return min(
        1.0,
        0.2
        + 0.3 * len(shared_entities)
        + 0.3 * len(shared_events)
        + 0.2 * len(shared_evidence),
    )


def _novelty(seed_document: GraphSemanticDocument, candidate: GraphSemanticDocument) -> float:
    new_entities = set(candidate.source_entity_ids) - set(seed_document.source_entity_ids)
    new_events = set(candidate.event_ids) - set(seed_document.event_ids)
    new_evidence = set(candidate.evidence_span_ids) - set(seed_document.evidence_span_ids)
    return min(1.0, 0.4 * len(new_entities) + 0.3 * len(new_events) + 0.3 * len(new_evidence))


def _candidate_mrr(
    candidates: list[NextHopCandidateScore],
    relevant_owner_ids: set[str],
) -> float:
    for index, candidate in enumerate(candidates, start=1):
        if candidate.owner_id in relevant_owner_ids:
            return 1.0 / index
    return 0.0


def _candidate_ndcg(
    candidates: list[NextHopCandidateScore],
    relevant_owner_ids: set[str],
    top_k: int,
) -> float:
    seen_relevant_owners = set()
    gains = []
    for candidate in candidates[:top_k]:
        if (
            candidate.owner_id in relevant_owner_ids
            and candidate.owner_id not in seen_relevant_owners
        ):
            gains.append(1.0)
            seen_relevant_owners.add(candidate.owner_id)
        else:
            gains.append(0.0)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_count = min(top_k, len(relevant_owner_ids))
    if ideal_count == 0:
        return 1.0
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _candidate_owner_recall(
    candidates: list[NextHopCandidateScore],
    relevant_owner_ids: set[str],
) -> float:
    if not relevant_owner_ids:
        return 1.0
    matched = {candidate.owner_id for candidate in candidates if candidate.owner_id in relevant_owner_ids}
    return len(matched) / len(relevant_owner_ids)


def _candidate_document_recall(
    candidates: list[NextHopCandidateScore],
    relevant_document_ids: set[str],
) -> float:
    if not relevant_document_ids:
        return 1.0
    matched = {
        candidate.semantic_document_id
        for candidate in candidates
        if candidate.semantic_document_id in relevant_document_ids
    }
    return len(matched) / len(relevant_document_ids)


def _candidate_evidence_recall(
    candidates: list[NextHopCandidateScore],
    relevant_evidence_ids: set[str],
) -> float:
    if not relevant_evidence_ids:
        return 1.0
    found = set()
    for candidate in candidates:
        found.update(candidate.evidence_span_ids)
    return len(found & relevant_evidence_ids) / len(relevant_evidence_ids)


def _is_relevant(
    result: SemanticSearchResult,
    relevant_document_ids: set[str],
    relevant_owner_ids: set[str],
) -> bool:
    return (
        result.semantic_document_id in relevant_document_ids
        or result.owner_id in relevant_owner_ids
    )


def _matched_keys(
    results: list[SemanticSearchResult],
    relevant_document_ids: set[str],
    relevant_owner_ids: set[str],
) -> set[str]:
    matched = set()
    for result in results:
        if result.semantic_document_id in relevant_document_ids:
            matched.add(result.semantic_document_id)
        if result.owner_id in relevant_owner_ids:
            matched.add(result.owner_id)
    return matched
