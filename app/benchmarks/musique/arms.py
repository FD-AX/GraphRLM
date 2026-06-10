from __future__ import annotations

import hashlib
import re
from time import perf_counter

from app.benchmarks.models import BenchmarkArmName, BenchmarkArmResult, BenchmarkCase
from app.benchmarks.musique.graph import build_musique_semantic_index
from app.dual_rlm.dynamic_graph_arm import DynamicGraphRLMArm
from app.dual_rlm.models import DualRLMConfig, GraphViewRef
from app.semantic_encoding.encoder import GraphSemanticEncoder, cosine_similarity


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"\w+", text.lower()) if len(term) > 2}


class MuSiQueKeywordArm:
    """Term-overlap paragraph retrieval. External lexical baseline."""

    name: BenchmarkArmName = "musique_keyword"

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        query_terms = _terms(case.question)
        ranked = sorted(
            case.metadata["paragraphs"],
            key=lambda paragraph: (
                -len(query_terms & _terms(f"{paragraph['title']} {paragraph['paragraph_text']}")),
                paragraph["idx"],
            ),
        )[: self.top_k]
        evidence_ids = [f"para_{paragraph['idx']}" for paragraph in ranked]
        return BenchmarkArmResult(
            prediction="",
            evidence_span_ids=evidence_ids,
            latency_ms=int((perf_counter() - started) * 1000),
            stop_reason="retrieval_complete",
            trace_id=f"musique_keyword:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
            trace=[{"top_k": self.top_k, "retrieved": evidence_ids}],
        )


class MuSiQueDenseTopKArm:
    """Plain dense cosine retrieval over paragraphs. No graph structure."""

    name: BenchmarkArmName = "musique_dense_topk"

    def __init__(self, encoder: GraphSemanticEncoder, top_k: int = 5) -> None:
        self.encoder = encoder
        self.top_k = top_k

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        index = build_musique_semantic_index(case, self.encoder)
        query_embedding = self.encoder.encode_query(case.question)
        results = index.search(query_embedding, top_k=self.top_k)
        evidence_ids = [result.owner_id for result in results]
        return BenchmarkArmResult(
            prediction="",
            evidence_span_ids=evidence_ids,
            latency_ms=int((perf_counter() - started) * 1000),
            stop_reason="retrieval_complete",
            trace_id=f"musique_dense_topk:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
            graph_source_hash=_hash_text(
                "".join(document.content_hash for document in index.documents.values())
            ),
            trace=[
                {
                    "top_k": self.top_k,
                    "retrieved": evidence_ids,
                    "scores": [round(result.score, 4) for result in results],
                }
            ],
        )


class MuSiQueCrossEncoderArm:
    """Exhaustive cross-encoder matching over all candidate paragraphs.

    Scores every (question, paragraph) pair with a joint BERT pass - the
    upper bound of pure pairwise matching without graph structure. A custom
    score_fn can be injected for tests.
    """

    name: BenchmarkArmName = "musique_cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        device: str = "cpu",
        score_fn=None,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self._score_fn = score_fn

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._score_fn is not None:
            return self._score_fn(pairs)
        if not hasattr(self, "_model"):
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return [float(value) for value in self._model.predict(pairs)]

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        paragraphs = case.metadata["paragraphs"]
        pairs = [
            (case.question, f"{paragraph['title']}. {paragraph['paragraph_text']}")
            for paragraph in paragraphs
        ]
        scores = self._score(pairs)
        ranked = sorted(
            zip(paragraphs, scores),
            key=lambda item: (-item[1], item[0]["idx"]),
        )[: self.top_k]
        evidence_ids = [f"para_{paragraph['idx']}" for paragraph, _ in ranked]
        return BenchmarkArmResult(
            prediction="",
            evidence_span_ids=evidence_ids,
            latency_ms=int((perf_counter() - started) * 1000),
            stop_reason="retrieval_complete",
            trace_id=f"musique_cross_encoder:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
            trace=[
                {
                    "model": self.model_name,
                    "top_k": self.top_k,
                    "retrieved": evidence_ids,
                    "scores": [round(score, 4) for _, score in ranked],
                }
            ],
        )


class MuSiQueMDRIterativeArm:
    """MDR-style iterative dense retrieval (Xiong et al., ICLR 2021 shape).

    Hop t re-encodes the query as (question + previously selected passage)
    and retrieves the next-best unvisited paragraph by cosine. No graph
    structure - isolates what iterative latent re-querying alone achieves
    with the same encoder and the same top-k evidence budget.
    """

    name: BenchmarkArmName = "musique_mdr_iterative"

    def __init__(self, encoder: GraphSemanticEncoder, top_k: int = 5) -> None:
        self.encoder = encoder
        self.top_k = top_k

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        index = build_musique_semantic_index(case, self.encoder)
        selected: list[str] = []
        hops = []
        query_text = case.question
        for _ in range(self.top_k):
            query_embedding = self.encoder.encode_query(query_text)
            results = index.search(
                query_embedding,
                top_k=1,
                visited_document_ids=set(selected),
            )
            if not results:
                break
            best = results[0]
            selected.append(best.semantic_document_id)
            hops.append({"owner_id": best.owner_id, "score": round(best.score, 4)})
            passage = index.document_for(best.semantic_document_id).text
            query_text = f"{case.question} {passage}"
        evidence_ids = [
            index.document_for(document_id).owner_id for document_id in selected
        ]
        return BenchmarkArmResult(
            prediction="",
            evidence_span_ids=evidence_ids,
            latency_ms=int((perf_counter() - started) * 1000),
            stop_reason="retrieval_complete",
            trace_id=f"musique_mdr_iterative:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
            trace=[{"top_k": self.top_k, "hops": hops}],
        )


class MuSiQueGraphNavigatorArm:
    """Seeded dense search + structural frontier traversal with re-ranked output.

    Collects candidate paragraphs along the beam traversal, then ranks the
    visited pool against the query and emits the top_k as evidence. This keeps
    the retrieval budget identical to the dense baseline so completeness
    deltas come from graph connectivity, not from a larger evidence set.
    """

    def __init__(
        self,
        encoder: GraphSemanticEncoder,
        *,
        name: BenchmarkArmName = "musique_graph_navigator",
        top_k: int = 5,
        seed_top_k: int = 3,
        max_depth: int = 3,
        beam_width: int = 3,
    ) -> None:
        self.name = name
        self.encoder = encoder
        self.top_k = top_k
        self.seed_top_k = seed_top_k
        self.max_depth = max_depth
        self.beam_width = beam_width

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        from app.semantic_encoding.navigation import LatentGraphNavigator

        started = perf_counter()
        index = build_musique_semantic_index(case, self.encoder)
        navigator = LatentGraphNavigator(index, self.encoder, self.encoder.config)
        trace = navigator.traverse(
            case.question,
            seed_top_k=self.seed_top_k,
            max_depth=self.max_depth,
            beam_width=self.beam_width,
        )
        candidate_ids = list(
            dict.fromkeys(
                [result.semantic_document_id for result in trace.seed_results]
                + trace.visited_document_ids
            )
        )
        query_embedding = self.encoder.encode_query(case.question)
        ranked = sorted(
            candidate_ids,
            key=lambda document_id: -cosine_similarity(
                query_embedding,
                index.embedding_for(document_id),
            ),
        )[: self.top_k]
        if len(ranked) < self.top_k:
            backfill = index.search(
                query_embedding,
                top_k=self.top_k,
                visited_document_ids=set(ranked),
            )
            ranked += [result.semantic_document_id for result in backfill][
                : self.top_k - len(ranked)
            ]
        evidence_ids = [index.document_for(document_id).owner_id for document_id in ranked]
        return BenchmarkArmResult(
            prediction="",
            evidence_span_ids=evidence_ids,
            latency_ms=int((perf_counter() - started) * 1000),
            stop_reason=trace.stop_reason,
            trace_id=f"{self.name}:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
            graph_source_hash=_hash_text(
                "".join(document.content_hash for document in index.documents.values())
            ),
            trace=[
                {
                    "top_k": self.top_k,
                    "seed_top_k": self.seed_top_k,
                    "max_depth": self.max_depth,
                    "beam_width": self.beam_width,
                    "active_profile_weight": self.encoder.config.active_profile_weight,
                    "interaction_profile_mode": self.encoder.config.interaction_profile_mode,
                    "visited_count": len(candidate_ids),
                    "retrieved": evidence_ids,
                    "traversal_stop_reason": trace.stop_reason,
                }
            ],
        )


class MuSiQueGraphRLMArm:
    """Full contour: graph index + model-driven discovery loop (GPT controller)."""

    name: BenchmarkArmName = "musique_graph_rlm"

    def __init__(
        self,
        encoder: GraphSemanticEncoder,
        *,
        model_name: str = "gpt-5-mini",
        reasoning_effort: str | None = "low",
        max_graph_model_calls: int = 10,
        max_graph_depth: int = 5,
        max_graph_expansions: int = 6,
        graph_top_k: int = 5,
        experiment_id: str | None = None,
    ) -> None:
        self.encoder = encoder
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.config = DualRLMConfig(
            graph_top_k=graph_top_k,
            max_graph_depth=max_graph_depth,
            max_graph_model_calls=max_graph_model_calls,
            max_graph_expansions=max_graph_expansions,
        )
        self.experiment_id = experiment_id

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        from app.benchmarks.musique.discovery import MuSiQueDiscoveryArm
        from app.benchmarks.musique.gateway import MuSiQueGraphGateway

        started = perf_counter()
        index = build_musique_semantic_index(case, self.encoder)
        gateway = MuSiQueGraphGateway(
            model_name=self.model_name,
            require_real_model=True,
            reasoning_effort=self.reasoning_effort,
            model_role="root",
        )
        arm = MuSiQueDiscoveryArm(
            index=index,
            graph_view=GraphViewRef(
                document_id=f"musique_{case.task_id}",
                graph_version="musique_paragraph_graph_v1",
                projection_version="musique_paragraph_graph_v1",
                encoder_version=self.encoder.backend.encoder_version,
            ),
            gateway=gateway,
            config=self.config,
        )
        result = None
        error = None
        try:
            result = arm.run(case.question, run_id=f"musique_graph_rlm:{case.task_id}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        trace_dicts = [trace.model_dump() for trace in gateway.model_call_traces]
        input_tokens = sum(trace.get("input_tokens") or 0 for trace in trace_dicts)
        output_tokens = sum(trace.get("output_tokens") or 0 for trace in trace_dicts)
        return BenchmarkArmResult(
            prediction=(result.answer_candidate or "") if result else "",
            raw_response=(result.answer_candidate or "") if result else None,
            provider="openai",
            model_name=self.model_name,
            model_role="root",
            reasoning_effort=self.reasoning_effort,
            experiment_id=self.experiment_id,
            evidence_span_ids=(
                [
                    evidence_id
                    for evidence_id in result.evidence_span_ids
                ]
                if result
                else []
            ),
            model_calls=len(trace_dicts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=sum(trace.get("total_tokens") or 0 for trace in trace_dicts),
            latency_ms=int((perf_counter() - started) * 1000),
            error=error,
            stop_reason=result.stop_reason if result else "error",
            trace_id=f"musique_graph_rlm:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
            trace=(result.trace if result else []),
            model_call_traces=trace_dicts,
        )
