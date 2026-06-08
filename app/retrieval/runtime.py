from __future__ import annotations

import math
from dataclasses import dataclass

from app.encoder.relation_encoder import hashing_vector
from app.retrieval.models import (
    EntityContext,
    GraphTraversalCandidate,
    Observation,
    PathState,
    RankedItem,
    RetrievalResult,
    TraversalPath,
)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


def normalized_similarity(left: list[float], right: list[float]) -> float:
    return (cosine_similarity(left, right) + 1.0) / 2.0


def weighted_pool(vectors: list[tuple[list[float], float]]) -> list[float]:
    if not vectors:
        return []

    dimensions = len(vectors[0][0])
    pooled = [0.0] * dimensions
    for vector, weight in vectors:
        for index, value in enumerate(vector):
            pooled[index] += value * weight

    norm = math.sqrt(sum(value * value for value in pooled))
    if norm == 0.0:
        return pooled

    return [value / norm for value in pooled]


@dataclass(frozen=True)
class ScoreConfig:
    query_weight: float = 0.45
    path_weight: float = 0.20
    structure_weight: float = 0.15
    novelty_weight: float = 0.15
    redundancy_weight: float = 0.10
    depth_weight: float = 0.04


class HashingQueryEncoder:
    def encode_text(self, text: str) -> list[float]:
        return hashing_vector(text)


class SemanticGraphStore:
    def __init__(
        self,
        entities: list[EntityContext],
        observations: list[Observation],
        transitions: list[GraphTraversalCandidate],
    ) -> None:
        self.entities = {entity.entity_id: entity for entity in entities}
        self.observations = observations
        self.transitions = transitions

    def observations_for_entity(self, entity_id: str) -> list[Observation]:
        return [
            observation
            for observation in self.observations
            if observation.entity_id == entity_id
        ]

    def transitions_from(self, entity_id: str) -> list[GraphTraversalCandidate]:
        return [
            transition
            for transition in self.transitions
            if transition.source_entity_id == entity_id
        ]


class LocateInspectExpandRuntime:
    def __init__(
        self,
        store: SemanticGraphStore,
        encoder: HashingQueryEncoder | None = None,
        score_config: ScoreConfig | None = None,
    ) -> None:
        self.store = store
        self.encoder = encoder or HashingQueryEncoder()
        self.score_config = score_config or ScoreConfig()

    def retrieve(
        self,
        query: str,
        seed_top_k: int = 3,
        observation_top_k: int = 6,
        beam_width: int = 2,
        max_depth: int = 2,
        candidate_budget: int = 12,
        evidence_budget: int = 8,
    ) -> RetrievalResult:
        query_vector = self.encoder.encode_text(query)
        seed_entities = self.locate_entities(query_vector, seed_top_k)
        observations = self.inspect_entities(
            query_vector=query_vector,
            seed_entities=seed_entities,
            top_k=observation_top_k,
        )
        paths = self.expand(
            query=query,
            query_vector=query_vector,
            seed_entities=seed_entities,
            beam_width=beam_width,
            max_depth=max_depth,
            candidate_budget=candidate_budget,
        )

        evidence_span_ids: list[str] = []
        for ranked in observations:
            observation = ranked.item
            if isinstance(observation, Observation):
                evidence_span_ids.extend(observation.evidence_span_ids)
        for path in paths:
            for transition in path.transitions:
                evidence_span_ids.extend(transition.evidence_span_ids)

        deduped_evidence = list(dict.fromkeys(evidence_span_ids))[:evidence_budget]

        return RetrievalResult(
            query=query,
            seed_entities=seed_entities,
            observations=observations,
            paths=paths,
            evidence_span_ids=deduped_evidence,
        )

    def locate_entities(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[RankedItem]:
        ranked = []
        for entity in self.store.entities.values():
            entity_vector = entity.vector or self.encoder.encode_text(entity.as_text())
            score = cosine_similarity(query_vector, entity_vector)
            ranked.append(
                RankedItem(
                    item_id=entity.entity_id,
                    score=score,
                    item=entity,
                    score_parts={"query_entity_similarity": score},
                )
            )

        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]

    def inspect_entities(
        self,
        query_vector: list[float],
        seed_entities: list[RankedItem],
        top_k: int,
    ) -> list[RankedItem]:
        ranked = []
        for seed in seed_entities:
            entity = seed.item
            if not isinstance(entity, EntityContext):
                continue

            for observation in self.store.observations_for_entity(entity.entity_id):
                observation_vector = observation.vector or self.encoder.encode_text(
                    observation.text
                )
                observation_score = cosine_similarity(query_vector, observation_vector)
                final_score = 0.75 * observation_score + 0.25 * seed.score
                ranked.append(
                    RankedItem(
                        item_id=observation.observation_id,
                        score=final_score,
                        item=observation,
                        score_parts={
                            "query_observation_similarity": observation_score,
                            "seed_entity_score": seed.score,
                        },
                    )
                )

        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]

    def expand(
        self,
        query: str,
        query_vector: list[float],
        seed_entities: list[RankedItem],
        beam_width: int,
        max_depth: int,
        candidate_budget: int,
    ) -> list[TraversalPath]:
        active_paths = [
            TraversalPath(
                entity_ids=[seed.item.entity_id],
                state=PathState(
                    path_embedding=query_vector,
                    covered_entities=[seed.item.entity_id],
                ),
                score=seed.score,
            )
            for seed in seed_entities
            if isinstance(seed.item, EntityContext)
        ][:beam_width]
        completed_paths: list[TraversalPath] = []

        for depth in range(1, max_depth + 1):
            frontier_by_signature: dict[tuple, TraversalPath] = {}
            for path in active_paths:
                for transition in self.store.transitions_from(path.last_entity_id):
                    transition_key = self.transition_key(transition)
                    if any(
                        self.transition_key(existing) == transition_key
                        for existing in path.transitions
                    ):
                        continue
                    candidate = transition.model_copy(
                        update={
                            "path_context": path.as_text(),
                            "depth": depth,
                        }
                    )
                    transition_score = self.score_transition(
                        query=query,
                        query_vector=query_vector,
                        candidate=candidate,
                        path_state=path.state,
                    )
                    next_state = self.extend_path_state(
                        query_vector=query_vector,
                        path_state=path.state,
                        transition=candidate,
                    )
                    next_path = TraversalPath(
                        entity_ids=path.entity_ids + [candidate.target_entity_id],
                        transitions=path.transitions + [candidate],
                        state=next_state,
                        score=0.65 * path.score + 0.35 * transition_score.score,
                    )
                    signature = next_path.signature()
                    existing = frontier_by_signature.get(signature)
                    if existing is None or next_path.score > existing.score:
                        frontier_by_signature[signature] = next_path

            if not frontier_by_signature:
                break

            active_paths = sorted(
                frontier_by_signature.values(),
                key=lambda path: path.score,
                reverse=True,
            )[:beam_width]
            completed_paths.extend(active_paths)

            if len(completed_paths) >= candidate_budget:
                break

        return sorted(
            completed_paths,
            key=lambda path: path.score,
            reverse=True,
        )[:candidate_budget]

    def score_transition(
        self,
        query: str,
        query_vector: list[float],
        candidate: GraphTraversalCandidate,
        path_state: PathState | None = None,
    ) -> RankedItem:
        transition_vector = candidate.vector or self.encoder.encode_text(
            candidate.as_text()
        )
        path_vector = (
            path_state.path_embedding
            if path_state and path_state.path_embedding
            else query_vector
        )
        query_similarity = normalized_similarity(query_vector, transition_vector)
        path_continuity = normalized_similarity(path_vector, transition_vector)
        novelty = self.novelty_score(path_state, candidate)
        redundancy = self.redundancy_penalty(path_state, candidate)
        depth_penalty = max(candidate.depth - 1, 0)
        config = self.score_config
        final_score = max(
            0.0,
            min(
                1.0,
                config.query_weight * query_similarity
                + config.path_weight * path_continuity
                + config.structure_weight * candidate.structural_confidence
                + config.novelty_weight * novelty
                - config.redundancy_weight * redundancy
                - config.depth_weight * depth_penalty,
            ),
        )

        return RankedItem(
            item_id=f"{candidate.source_entity_id}->{candidate.target_entity_id}",
            score=final_score,
            item=candidate,
            score_parts={
                "query_transition_similarity": query_similarity,
                "query_transition_score": query_similarity,
                "path_continuity": path_continuity,
                "structural_confidence": candidate.structural_confidence,
                "novelty": novelty,
                "redundancy": redundancy,
                "depth_penalty": depth_penalty,
            },
        )

    def extend_path_state(
        self,
        query_vector: list[float],
        path_state: PathState,
        transition: GraphTraversalCandidate,
    ) -> PathState:
        transition_vector = transition.vector or self.encoder.encode_text(
            transition.as_text()
        )
        previous_vector = path_state.path_embedding or query_vector
        path_embedding = weighted_pool(
            [
                (query_vector, 0.20),
                (previous_vector, 0.45),
                (transition_vector, 0.35),
            ]
        )
        return PathState(
            path_embedding=path_embedding,
            covered_entities=list(
                dict.fromkeys(
                    path_state.covered_entities
                    + [transition.source_entity_id, transition.target_entity_id]
                )
            ),
            covered_events=list(
                dict.fromkeys(path_state.covered_events + transition.event_ids)
            ),
            accumulated_evidence=list(
                dict.fromkeys(
                    path_state.accumulated_evidence + transition.evidence_span_ids
                )
            ),
            unresolved_query_facets=path_state.unresolved_query_facets,
        )

    def novelty_score(
        self,
        path_state: PathState | None,
        candidate: GraphTraversalCandidate,
    ) -> float:
        if path_state is None:
            return 1.0

        new_entities = set([candidate.target_entity_id]) - set(
            path_state.covered_entities
        )
        new_events = set(candidate.event_ids) - set(path_state.covered_events)
        new_evidence = set(candidate.evidence_span_ids) - set(
            path_state.accumulated_evidence
        )
        return min(
            1.0,
            0.40 * bool(new_entities)
            + 0.25 * bool(new_events)
            + 0.35 * bool(new_evidence),
        )

    def redundancy_penalty(
        self,
        path_state: PathState | None,
        candidate: GraphTraversalCandidate,
    ) -> float:
        if path_state is None:
            return 0.0

        repeated_evidence = set(candidate.evidence_span_ids) & set(
            path_state.accumulated_evidence
        )
        if not candidate.evidence_span_ids:
            return 0.2

        return len(repeated_evidence) / len(candidate.evidence_span_ids)

    def transition_key(self, transition: GraphTraversalCandidate) -> tuple:
        return (
            transition.source_entity_id,
            tuple(sorted(transition.relation_ids)),
            tuple(sorted(transition.event_ids)),
            transition.target_entity_id,
        )
