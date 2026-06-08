from __future__ import annotations

from app.semantic_encoding.encoder import (
    GraphSemanticEncoder,
    cosine_similarity,
)
from app.semantic_encoding.index import GraphSemanticIndex
from app.semantic_encoding.models import (
    EncoderConfig,
    LatentTraversalState,
    SemanticTraversalTrace,
    SemanticTraversalTraceStep,
)


class LatentGraphNavigator:
    def __init__(
        self,
        index: GraphSemanticIndex,
        encoder: GraphSemanticEncoder,
        config: EncoderConfig | None = None,
    ) -> None:
        self.index = index
        self.encoder = encoder
        self.config = config or encoder.config

    def traverse(
        self,
        query: str,
        seed_top_k: int = 5,
        max_depth: int | None = None,
        beam_width: int | None = None,
    ) -> SemanticTraversalTrace:
        max_depth = max_depth if max_depth is not None else self.config.max_graph_depth
        beam_width = beam_width if beam_width is not None else self.config.beam_width
        query_embedding = self.encoder.encode_query(query)
        seeds = self.index.search(query_embedding, top_k=seed_top_k)
        if not seeds:
            return SemanticTraversalTrace(query=query, stop_reason="no_seed_documents")

        seed = seeds[0]
        seed_embedding = self.index.embedding_for(seed.semantic_document_id)
        active_profile = self.encoder.interaction_profile(query_embedding, seed_embedding)
        state = LatentTraversalState(
            query_text=query,
            original_query_embedding=query_embedding,
            navigation_embedding=query_embedding,
            active_interaction_embedding=active_profile,
            current_document_id=seed.semantic_document_id,
            current_owner_id=seed.owner_id,
            visited_document_ids=[seed.semantic_document_id],
            visited_entity_ids=seed.source_entity_ids,
            visited_event_ids=seed.event_ids,
            evidence_span_ids=seed.evidence_span_ids,
            depth=0,
            remaining_budget=max_depth,
        )
        steps = [
            SemanticTraversalTraceStep(
                depth=0,
                current_document_id=seed.semantic_document_id,
                current_owner_id=seed.owner_id,
                query_similarity=seed.query_similarity,
                active_profile_similarity=1.0,
                final_score=seed.score,
                score_parts=seed.score_parts,
                contribution_top_components=seed.contribution_top_components,
                evidence_span_ids=seed.evidence_span_ids,
            )
        ]

        active_states = [(state, seed.score)]
        for depth in range(1, max_depth + 1):
            candidates = []
            for current_state, path_score in active_states:
                current_document = self.index.document_for(
                    current_state.current_document_id
                )
                frontier_ids = self.index.frontier_document_ids(
                    current_document,
                    cap=self.config.local_frontier_cap,
                )
                frontier_ids -= set(current_state.visited_document_ids)
                scored = self._score_frontier(
                    state=current_state,
                    candidate_document_ids=frontier_ids,
                    depth=depth,
                    path_score=path_score,
                )
                candidates.extend(scored)
            if not candidates:
                steps.append(
                    SemanticTraversalTraceStep(
                        depth=depth,
                        current_document_id=active_states[0][0].current_document_id,
                        current_owner_id=active_states[0][0].current_owner_id,
                        query_similarity=0.0,
                        active_profile_similarity=0.0,
                        final_score=0.0,
                        stop_reason="frontier_exhausted",
                    )
                )
                return self._trace(query, seeds, steps, active_states, "frontier_exhausted")

            candidates.sort(key=lambda item: item[1], reverse=True)
            selected = candidates[:beam_width]
            active_states = [(item[0], item[1]) for item in selected]
            for selected_state, final_score, score_parts, components in selected:
                steps.append(
                    SemanticTraversalTraceStep(
                        depth=depth,
                        current_document_id=selected_state.current_document_id,
                        current_owner_id=selected_state.current_owner_id,
                        selected_document_id=selected_state.current_document_id,
                        query_similarity=score_parts["navigation_similarity"],
                        active_profile_similarity=score_parts[
                            "active_profile_similarity"
                        ],
                        final_score=final_score,
                        score_parts=score_parts,
                        contribution_top_components=components,
                        evidence_span_ids=selected_state.evidence_span_ids,
                    )
                )

        return self._trace(query, seeds, steps, active_states, "max_depth_reached")

    def _score_frontier(
        self,
        state: LatentTraversalState,
        candidate_document_ids: set[str],
        depth: int,
        path_score: float,
    ) -> list[tuple[LatentTraversalState, float, dict, list[dict]]]:
        scored = []
        for document_id in candidate_document_ids:
            document = self.index.document_for(document_id)
            embedding = self.index.embedding_for(document_id)
            candidate_profile = self.encoder.interaction_profile(
                state.original_query_embedding,
                embedding,
            )
            navigation_similarity = cosine_similarity(
                state.navigation_embedding,
                embedding,
            )
            active_profile_similarity = cosine_similarity(
                state.active_interaction_embedding,
                candidate_profile,
            )
            evidence_confidence = 1.0 if document.evidence_span_ids else 0.4
            structural_confidence = self._structural_confidence(state, document)
            novelty = self._novelty(state, document)
            depth_penalty = max(depth - 1, 0) * 0.04
            navigation_score = (navigation_similarity + 1.0) / 2.0
            active_profile_score = (active_profile_similarity + 1.0) / 2.0
            final_score = max(
                0.0,
                min(
                    1.0,
                    self.config.navigation_similarity_weight * navigation_score
                    + self.config.active_profile_weight * active_profile_score
                    + self.config.structural_confidence_weight * structural_confidence
                    + self.config.evidence_quality_weight * evidence_confidence
                    + self.config.novelty_weight * novelty
                    - depth_penalty,
                ),
            )
            updated_navigation = self.encoder.update_navigation_embedding(
                state.original_query_embedding,
                state.navigation_embedding,
                embedding,
            )
            updated_state = state.model_copy(
                update={
                    "navigation_embedding": updated_navigation,
                    "active_interaction_embedding": candidate_profile,
                    "current_document_id": document.semantic_document_id,
                    "current_owner_id": document.owner_id,
                    "visited_document_ids": list(
                        dict.fromkeys(
                            state.visited_document_ids + [document.semantic_document_id]
                        )
                    ),
                    "visited_entity_ids": list(
                        dict.fromkeys(
                            state.visited_entity_ids + document.source_entity_ids
                        )
                    ),
                    "visited_event_ids": list(
                        dict.fromkeys(state.visited_event_ids + document.event_ids)
                    ),
                    "evidence_span_ids": list(
                        dict.fromkeys(
                            state.evidence_span_ids + document.evidence_span_ids
                        )
                    ),
                    "depth": depth,
                    "remaining_budget": max(state.remaining_budget - 1, 0),
                }
            )
            components = self.encoder.top_contribution_components(
                state.original_query_embedding,
                embedding,
            )
            scored.append(
                (
                    updated_state,
                    0.65 * path_score + 0.35 * final_score,
                    {
                        "navigation_similarity": navigation_similarity,
                        "active_profile_similarity": active_profile_similarity,
                        "active_profile_weight": self.config.active_profile_weight,
                        "structural_confidence": structural_confidence,
                        "evidence_confidence": evidence_confidence,
                        "novelty": novelty,
                        "depth_penalty": depth_penalty,
                        "local_score": final_score,
                    },
                    components,
                )
            )
        return scored

    def _structural_confidence(self, state: LatentTraversalState, document) -> float:
        shared_entities = set(state.visited_entity_ids) & set(document.source_entity_ids)
        shared_events = set(state.visited_event_ids) & set(document.event_ids)
        shared_evidence = set(state.evidence_span_ids) & set(document.evidence_span_ids)
        return min(
            1.0,
            0.2
            + 0.3 * len(shared_entities)
            + 0.3 * len(shared_events)
            + 0.2 * len(shared_evidence),
        )

    def _novelty(self, state: LatentTraversalState, document) -> float:
        new_entities = set(document.source_entity_ids) - set(state.visited_entity_ids)
        new_events = set(document.event_ids) - set(state.visited_event_ids)
        new_evidence = set(document.evidence_span_ids) - set(state.evidence_span_ids)
        return min(1.0, 0.4 * len(new_entities) + 0.3 * len(new_events) + 0.3 * len(new_evidence))

    def _trace(
        self,
        query: str,
        seeds,
        steps,
        active_states,
        stop_reason: str,
    ) -> SemanticTraversalTrace:
        visited_document_ids = []
        visited_entity_ids = []
        evidence_span_ids = []
        for state, _ in active_states:
            visited_document_ids.extend(state.visited_document_ids)
            visited_entity_ids.extend(state.visited_entity_ids)
            evidence_span_ids.extend(state.evidence_span_ids)
        return SemanticTraversalTrace(
            query=query,
            seed_results=seeds,
            steps=steps,
            visited_document_ids=list(dict.fromkeys(visited_document_ids)),
            visited_entity_ids=list(dict.fromkeys(visited_entity_ids)),
            evidence_span_ids=list(dict.fromkeys(evidence_span_ids)),
            stop_reason=stop_reason,
        )
