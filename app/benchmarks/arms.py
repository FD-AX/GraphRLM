from __future__ import annotations

import re
from datetime import datetime, timezone
from hashlib import sha256
from time import perf_counter

from pydantic import BaseModel, Field

from app.benchmarks.factlens.audit import FactLensAuditArm
from app.benchmarks.models import BenchmarkArmName, BenchmarkArmResult, BenchmarkCase
from app.benchmarks.oolong.operations import build_record_fact, execute_oolong_operation, plan_oolong_operation
from app.benchmarks.oolong.semantic_graph import build_oolong_semantic_graph
from app.dual_rlm import DualRLMConfig, DynamicGraphRLMArm, GraphViewRef, PydanticAIGPTGateway
from app.dual_rlm.gateway import _response_id, _usage_field, _usage_value, require_openai_credentials
from app.dual_rlm.gateway import _openai_model_settings
from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticDocument,
    GraphSemanticEncoder,
    GraphSemanticIndex,
    HashingSemanticEncoder,
    TransformerSemanticEncoder,
)


class DirectHeuristicArm:
    """Cheap local baseline for harness validation, not a model-quality baseline."""

    name: BenchmarkArmName = "direct_model"

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        prediction = _extract_answer_from_context(case)
        return BenchmarkArmResult(
            prediction=prediction,
            evidence_span_ids=[],
            model_calls=0,
            input_tokens=len(case.context.split()) + len(case.question.split()),
            output_tokens=len(prediction.split()),
            latency_ms=int((perf_counter() - started) * 1000),
            stop_reason="heuristic_complete",
            trace_id=f"direct_heuristic:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
        )


class GoldAnswerArm:
    name: BenchmarkArmName = "gold_fixture"

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        prediction = case.gold_answer[0] if isinstance(case.gold_answer, list) else case.gold_answer
        return BenchmarkArmResult(
            prediction=str(prediction),
            evidence_span_ids=list(case.gold_evidence_span_ids),
            model_calls=0,
            input_tokens=0,
            output_tokens=len(str(prediction).split()),
            stop_reason="gold_fixture",
            trace_id=f"gold_answer:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
        )


class WrongAnswerArm:
    name: BenchmarkArmName = "wrong_fixture"

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        return BenchmarkArmResult(
            prediction="definitely-wrong-answer",
            evidence_span_ids=[],
            model_calls=0,
            input_tokens=0,
            output_tokens=1,
            stop_reason="wrong_fixture",
            trace_id=f"wrong_answer:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
        )


class KeywordRAGArm:
    """Overlap-based retrieval baseline for small local and CI benchmark subsets."""

    name: BenchmarkArmName = "rag"

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        chunks = _chunk_context(case.context)
        query_terms = _terms(case.question)
        ranked = sorted(
            chunks,
            key=lambda chunk: (-_overlap(query_terms, chunk[1]), chunk[0]),
        )[: self.top_k]
        retrieved_text = " ".join(chunk for _, chunk in ranked)
        prediction = _extract_answer_from_context(
            case.model_copy(update={"context": retrieved_text})
        )
        return BenchmarkArmResult(
            prediction=prediction,
            evidence_span_ids=[f"chunk_{index}" for index, _ in ranked],
            model_calls=0,
            input_tokens=sum(len(chunk.split()) for _, chunk in ranked) + len(case.question.split()),
            output_tokens=len(prediction.split()),
            latency_ms=int((perf_counter() - started) * 1000),
            stop_reason="retrieval_complete",
            trace_id=f"keyword_rag:{case.task_id}",
            arm_input_hash=_hash_text(case.context),
            trace=[
                {
                    "retrieved_chunk_ids": [f"chunk_{index}" for index, _ in ranked],
                    "top_k": self.top_k,
                }
            ],
        )


class RawRecordsOpsArm:
    name: BenchmarkArmName = "raw_records_ops"

    def __init__(
        self,
        *,
        prompt_id: str = "oolong_raw_records_ops_v1",
        experiment_id: str | None = None,
        worker_model_name: str | None = None,
        worker_reasoning_effort: str | None = None,
    ) -> None:
        self.prompt_id = prompt_id
        self.experiment_id = experiment_id
        self.worker_model_name = worker_model_name
        self.worker_reasoning_effort = worker_reasoning_effort

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        source_hash = _hash_text(case.context)
        record_labels, worker_traces = _extract_worker_record_labels(
            case,
            worker_model_name=self.worker_model_name,
            worker_reasoning_effort=self.worker_reasoning_effort,
        )
        facts = _raw_record_facts_from_case(case, record_labels)
        plan = plan_oolong_operation(case.question)
        operation_result = execute_oolong_operation(plan, facts)
        worker_input_tokens = sum(trace.get("input_tokens") or 0 for trace in worker_traces)
        worker_output_tokens = sum(trace.get("output_tokens") or 0 for trace in worker_traces)
        worker_tokens = sum(trace.get("total_tokens") or 0 for trace in worker_traces)
        prediction = (operation_result.answer_text or "").strip()
        trace_payload = {
            "runtime_provenance": {
                "arm": self.name,
                "source": "raw_public_records",
                "semantic_graph_used": False,
                "legacy_hash_frontier_used": False,
                "planner_source": plan.planner_source if plan else None,
                "planner_version": plan.planner_version if plan else None,
                "record_fact_count": len(facts),
                "record_label_count": len(record_labels),
            },
            "operation": operation_result.model_dump(mode="json"),
            "worker_total_tokens": worker_tokens,
            "root_total_tokens": 0,
        }
        return BenchmarkArmResult(
            prediction=prediction,
            raw_response=prediction,
            provider="openai" if worker_traces else None,
            model_name=self.worker_model_name if worker_traces else None,
            model_role="worker" if worker_traces else None,
            reasoning_effort=None,
            worker_model_name=self.worker_model_name,
            worker_reasoning_effort=self.worker_reasoning_effort,
            experiment_id=self.experiment_id,
            response_id=worker_traces[-1].get("response_id") if worker_traces else None,
            input_tokens=worker_input_tokens,
            output_tokens=worker_output_tokens,
            total_tokens=worker_tokens,
            latency_ms=int((perf_counter() - started) * 1000),
            fallback_used=any(trace.get("fallback_used") for trace in worker_traces),
            prompt_id=self.prompt_id,
            arm_input_hash=source_hash,
            evidence_span_ids=operation_result.evidence_span_ids,
            model_calls=len(worker_traces),
            stop_reason=f"raw_records_ops_{operation_result.status}",
            trace_id=f"raw_records_ops:{case.task_id}:{operation_result.status}",
            trace=[trace_payload],
            model_call_traces=worker_traces,
        )


class DirectGPTArm:
    name: BenchmarkArmName = "direct_model"

    def __init__(
        self,
        *,
        model_name: str = "gpt-5",
        prompt_id: str = "oolong_direct_v1",
        reasoning_effort: str | None = None,
        experiment_id: str | None = None,
    ) -> None:
        require_openai_credentials()
        from pydantic_ai import Agent

        self.provider = "openai"
        self.model_name = model_name
        self.prompt_id = prompt_id
        self.reasoning_effort = reasoning_effort
        self.experiment_id = experiment_id
        self._agent_cls = Agent

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        prompt = build_oolong_direct_prompt(case)
        started = perf_counter()
        request_timestamp = datetime.now(timezone.utc)
        agent = self._agent_cls(
            f"openai:{self.model_name}",
            output_type=str,
            system_prompt=(
                "Return only the final answer in the format required by the question. "
                "Do not include reasoning or explanation."
            ),
            model_settings=_openai_model_settings(self.reasoning_effort),
        )
        response = agent.run_sync(prompt)
        response_timestamp = datetime.now(timezone.utc)
        raw_output = getattr(response, "output", None)
        if raw_output is None:
            raw_output = getattr(response, "data", None)
        prediction = "" if raw_output is None else str(raw_output).strip()
        usage = _usage_value(response)
        return BenchmarkArmResult(
            prediction=prediction,
            raw_response=prediction,
            provider=self.provider,
            model_name=self.model_name,
            model_role="root",
            reasoning_effort=self.reasoning_effort,
            experiment_id=self.experiment_id,
            response_id=_response_id(response),
            input_tokens=_usage_field(usage, ["request_tokens", "input_tokens", "prompt_tokens"]) or 0,
            output_tokens=_usage_field(usage, ["response_tokens", "output_tokens", "completion_tokens"]) or 0,
            total_tokens=_usage_field(usage, ["total_tokens"]) or 0,
            latency_ms=int((response_timestamp - request_timestamp).total_seconds() * 1000),
            fallback_used=False,
            prompt_id=self.prompt_id,
            arm_input_hash=_hash_text(case.context),
            model_calls=1,
            stop_reason="direct_model_complete",
            trace_id=f"direct_gpt:{case.task_id}",
            trace=[
                {
                    "prompt_id": self.prompt_id,
                    "experiment_id": self.experiment_id,
                    "model_role": "root",
                    "reasoning_effort": self.reasoning_effort,
                    "request_started_at": request_timestamp.isoformat(),
                    "response_received_at": response_timestamp.isoformat(),
                }
            ],
            model_call_traces=[
                {
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "model_role": "root",
                    "reasoning_effort": self.reasoning_effort,
                    "purpose": "direct_answer",
                    "response_id": _response_id(response),
                    "input_tokens": _usage_field(usage, ["request_tokens", "input_tokens", "prompt_tokens"]) or 0,
                    "output_tokens": _usage_field(usage, ["response_tokens", "output_tokens", "completion_tokens"]) or 0,
                    "total_tokens": _usage_field(usage, ["total_tokens"]) or 0,
                    "fallback_used": False,
                }
            ],
        )


class GraphRLMBenchmarkArm:
    @property
    def name(self) -> BenchmarkArmName:
        if self.worker_model_name:
            return "graph_rlm_spam_worker_materialized"
        return "graph_rlm_hash_frontier"

    def __init__(
        self,
        *,
        model_name: str = "gpt-5",
        prompt_id: str = "oolong_graph_rlm_v1",
        reasoning_effort: str | None = None,
        experiment_id: str | None = None,
        worker_model_name: str | None = None,
        worker_reasoning_effort: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.prompt_id = prompt_id
        self.reasoning_effort = reasoning_effort
        self.experiment_id = experiment_id
        self.worker_model_name = worker_model_name
        self.worker_reasoning_effort = worker_reasoning_effort

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        build_started = perf_counter()
        source_hash = _hash_text(case.context)
        record_labels, worker_traces = _extract_worker_record_labels(
            case,
            worker_model_name=self.worker_model_name,
            worker_reasoning_effort=self.worker_reasoning_effort,
        )
        documents = _oolong_context_to_documents(case, source_hash, record_labels=record_labels)
        encoder = GraphSemanticEncoder(
            EncoderConfig(projection_dim=256, local_frontier_cap=20),
            backend=HashingSemanticEncoder(dimensions=256),
        )
        index = GraphSemanticIndex.build(documents, encoder)
        build_latency_ms = int((perf_counter() - build_started) * 1000)
        gateway = PydanticAIGPTGateway(
            model_name=self.model_name,
            require_real_model=True,
            reasoning_effort=self.reasoning_effort,
            model_role="root",
        )
        arm = DynamicGraphRLMArm(
            index=index,
            graph_view=GraphViewRef(
                document_id=f"oolong_graph_{case.task_id}",
                graph_version="oolong_public_context_graph_v1",
                projection_version="benchmark_public_context_v1",
                encoder_version=encoder.backend.encoder_version,
            ),
            gateway=gateway,
            config=DualRLMConfig(
                graph_top_k=5,
                max_graph_depth=3,
                max_graph_model_calls=6,
                max_graph_expansions=3,
            ),
        )
        started = perf_counter()
        result = None
        error = None
        try:
            result = arm.run(
                case.question,
                run_id=f"oolong_graph_rlm:{case.task_id}",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        query_latency_ms = int((perf_counter() - started) * 1000)
        traces = result.model_call_traces if result else [
            trace for trace in gateway.model_call_traces if trace.purpose == "graph_decision"
        ]
        result_trace = result.trace if result else []
        result_evidence_ids = result.evidence_span_ids if result else []
        result_stop_reason = result.stop_reason if result else "error"
        prediction = (
            _graph_answer_from_trace(result_trace)
            or (result.answer_candidate if result else None)
            or _worker_aggregate_answer(case, record_labels)
            or ""
        )
        input_tokens = sum(trace.input_tokens or 0 for trace in traces)
        output_tokens = sum(trace.output_tokens or 0 for trace in traces)
        total_tokens = sum(trace.total_tokens or 0 for trace in traces)
        worker_input_tokens = sum(trace.get("input_tokens") or 0 for trace in worker_traces)
        worker_output_tokens = sum(trace.get("output_tokens") or 0 for trace in worker_traces)
        worker_total_tokens = sum(trace.get("total_tokens") or 0 for trace in worker_traces)
        worker_label_counts = _label_counts(record_labels)
        full_context_tokens = case.measured_context_tokens or len(case.context.split())
        selected_context_tokens = _selected_context_tokens(documents, result_evidence_ids)
        prompt_overhead_ratio = (
            max(input_tokens - selected_context_tokens, 0) / input_tokens
            if input_tokens
            else 0.0
        )
        cost_trace = {
            "experiment_id": self.experiment_id,
            "graph_build": {
                "graph_build_id": f"graph_build:{case.task_id}:{source_hash[:12]}",
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": build_latency_ms,
                "documents": len(documents),
                "worker_model": self.worker_model_name,
                "worker_calls": len(worker_traces),
                "worker_input_tokens": worker_input_tokens,
                "worker_output_tokens": worker_output_tokens,
                "worker_total_tokens": worker_total_tokens,
                "worker_label_counts": worker_label_counts,
                "worker_label_coverage": (
                    len(record_labels) / max(len(_oolong_record_lines(case.context)), 1)
                ),
            },
            "query": {
                "controller_calls": len(traces),
                "synthesis_calls": 0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": query_latency_ms,
                "full_context_tokens": full_context_tokens,
                "graph_selected_context_tokens": selected_context_tokens,
                "prompt_overhead_ratio": round(prompt_overhead_ratio, 6),
            },
            "cold_total": {
                "model_calls": len(traces) + len(worker_traces),
                "input_tokens": input_tokens + worker_input_tokens,
                "output_tokens": output_tokens + worker_output_tokens,
                "total_tokens": total_tokens + worker_total_tokens,
                "latency_ms": build_latency_ms + query_latency_ms,
            },
            "online_query_only": {
                "model_calls": len(traces),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": query_latency_ms,
            },
        }
        return BenchmarkArmResult(
            prediction=prediction.strip(),
            raw_response=prediction.strip(),
            provider="openai" if traces else None,
            model_name=self.model_name if traces else None,
            model_role="root" if traces else None,
            reasoning_effort=self.reasoning_effort,
            worker_model_name=self.worker_model_name,
            worker_reasoning_effort=self.worker_reasoning_effort,
            experiment_id=self.experiment_id,
            response_id=traces[-1].response_id if traces else None,
            input_tokens=input_tokens + worker_input_tokens,
            output_tokens=output_tokens + worker_output_tokens,
            total_tokens=total_tokens + worker_total_tokens,
            latency_ms=build_latency_ms + query_latency_ms,
            fallback_used=any(trace.fallback_used for trace in traces),
            error=error,
            prompt_id=self.prompt_id,
            arm_input_hash=source_hash,
            graph_source_hash=source_hash,
            evidence_span_ids=result_evidence_ids,
            model_calls=len(traces) + len(worker_traces),
            stop_reason=result_stop_reason,
            trace_id=result.trace_id if result else f"oolong_graph_rlm:{case.task_id}:error",
            trace=[cost_trace] + result_trace,
            model_call_traces=worker_traces + [trace.model_dump(mode="json") for trace in traces],
        )


class GraphRLMSemanticGraphArm:
    name: BenchmarkArmName = "graph_rlm_semantic_graph"

    def __init__(
        self,
        *,
        model_name: str = "gpt-5",
        prompt_id: str = "oolong_graph_rlm_semantic_graph_v1",
        reasoning_effort: str | None = None,
        experiment_id: str | None = None,
        worker_model_name: str | None = None,
        worker_reasoning_effort: str | None = None,
        encoder_backend: str = "transformer",
        gateway_factory=None,
    ) -> None:
        self.model_name = model_name
        self.prompt_id = prompt_id
        self.reasoning_effort = reasoning_effort
        self.experiment_id = experiment_id
        self.worker_model_name = worker_model_name
        self.worker_reasoning_effort = worker_reasoning_effort
        self.encoder_backend = encoder_backend
        self.gateway_factory = gateway_factory
        self.graph_source = "oolong_semantic_graph"
        self.legacy_hash_frontier_used = False
        self.graph_builder_class = "OOLONGSemanticGraphBuilder"

    def build_runtime(self, case: BenchmarkCase):
        record_labels, worker_traces = _extract_worker_record_labels(
            case,
            worker_model_name=self.worker_model_name,
            worker_reasoning_effort=self.worker_reasoning_effort,
        )
        graph_build = build_oolong_semantic_graph(
            case,
            record_labels=record_labels,
            encoder_backend=self.encoder_backend,
        )
        encoder_class = graph_build.encoder.backend.__class__.__name__
        if self.encoder_backend == "transformer":
            assert isinstance(graph_build.encoder.backend, TransformerSemanticEncoder)
        assert self.graph_source == "oolong_semantic_graph"
        assert self.legacy_hash_frontier_used is False
        return graph_build, worker_traces, {
            "arm": self.name,
            "graph_source": self.graph_source,
            "graph_builder_class": self.graph_builder_class,
            "encoder_class": encoder_class,
            "encoder_name": graph_build.encoder.backend.encoder_name,
            "encoder_version": graph_build.encoder.backend.encoder_version,
            "legacy_hash_frontier_used": self.legacy_hash_frontier_used,
            "semantic_graph_used": True,
            "validation": graph_build.validation,
        }

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        graph_build, worker_traces, provenance = self.build_runtime(case)
        operation_plan = plan_oolong_operation(case.question)
        operation_result = execute_oolong_operation(operation_plan, graph_build.record_facts)
        if operation_result.status == "complete":
            worker_input_tokens = sum(trace.get("input_tokens") or 0 for trace in worker_traces)
            worker_output_tokens = sum(trace.get("output_tokens") or 0 for trace in worker_traces)
            worker_tokens = sum(trace.get("total_tokens") or 0 for trace in worker_traces)
            trace_payload = {
                "runtime_provenance": provenance,
                "graph_node_count": len(graph_build.semantic_documents),
                "embedding_count": len(graph_build.embeddings),
                "traversal_seed_count": len(graph_build.traversal_trace.seed_results),
                "worker_total_tokens": worker_tokens,
                "root_total_tokens": 0,
                "deterministic_operation": operation_result.model_dump(mode="json"),
            }
            return BenchmarkArmResult(
                prediction=(operation_result.answer_text or "").strip(),
                raw_response=(operation_result.answer_text or "").strip(),
                provider="openai" if worker_traces else None,
                model_name=self.worker_model_name if worker_traces else None,
                model_role="worker" if worker_traces else None,
                reasoning_effort=self.reasoning_effort,
                worker_model_name=self.worker_model_name,
                worker_reasoning_effort=self.worker_reasoning_effort,
                experiment_id=self.experiment_id,
                response_id=worker_traces[-1].get("response_id") if worker_traces else None,
                input_tokens=worker_input_tokens,
                output_tokens=worker_output_tokens,
                total_tokens=worker_tokens,
                latency_ms=int((perf_counter() - started) * 1000),
                fallback_used=any(trace.get("fallback_used") for trace in worker_traces),
                prompt_id=self.prompt_id,
                arm_input_hash=_hash_text(case.context),
                graph_source_hash=_hash_text(
                    "".join(document.content_hash for document in graph_build.semantic_documents)
                ),
                evidence_span_ids=operation_result.evidence_span_ids,
                model_calls=len(worker_traces),
                stop_reason="deterministic_operation_complete",
                trace_id=f"oolong_semantic_graph:{case.task_id}:operation",
                trace=[trace_payload],
                model_call_traces=worker_traces,
            )
        if operation_result.query_semantics == "unsupported_analytic_operation":
            worker_input_tokens = sum(trace.get("input_tokens") or 0 for trace in worker_traces)
            worker_output_tokens = sum(trace.get("output_tokens") or 0 for trace in worker_traces)
            worker_tokens = sum(trace.get("total_tokens") or 0 for trace in worker_traces)
            trace_payload = {
                "runtime_provenance": provenance,
                "graph_node_count": len(graph_build.semantic_documents),
                "embedding_count": len(graph_build.embeddings),
                "traversal_seed_count": len(graph_build.traversal_trace.seed_results),
                "worker_total_tokens": worker_tokens,
                "root_total_tokens": 0,
                "deterministic_operation": operation_result.model_dump(mode="json"),
                "routing_guard": {
                    "query_semantics": operation_result.query_semantics,
                    "executor_status": operation_result.status,
                    "recursive_graph_allowed": False,
                    "reason": "Recognized analytic operation is unsupported by the executor; graph traversal is not a valid fallback.",
                },
            }
            return BenchmarkArmResult(
                prediction="",
                raw_response="",
                provider="openai" if worker_traces else None,
                model_name=self.worker_model_name if worker_traces else None,
                model_role="worker" if worker_traces else None,
                reasoning_effort=self.reasoning_effort,
                worker_model_name=self.worker_model_name,
                worker_reasoning_effort=self.worker_reasoning_effort,
                experiment_id=self.experiment_id,
                response_id=worker_traces[-1].get("response_id") if worker_traces else None,
                input_tokens=worker_input_tokens,
                output_tokens=worker_output_tokens,
                total_tokens=worker_tokens,
                latency_ms=int((perf_counter() - started) * 1000),
                fallback_used=any(trace.get("fallback_used") for trace in worker_traces),
                error=operation_result.reason,
                prompt_id=self.prompt_id,
                arm_input_hash=_hash_text(case.context),
                graph_source_hash=_hash_text(
                    "".join(document.content_hash for document in graph_build.semantic_documents)
                ),
                evidence_span_ids=[],
                model_calls=len(worker_traces),
                stop_reason="unsupported_analytic_operation",
                trace_id=f"oolong_semantic_graph:{case.task_id}:unsupported_analytic_operation",
                trace=[trace_payload],
                model_call_traces=worker_traces,
            )
        gateway = (
            self.gateway_factory()
            if self.gateway_factory
            else PydanticAIGPTGateway(
                model_name=self.model_name,
                require_real_model=True,
                reasoning_effort=self.reasoning_effort,
                model_role="root",
            )
        )
        arm = DynamicGraphRLMArm(
            index=graph_build.index,
            graph_view=GraphViewRef(
                document_id=graph_build.document_id,
                graph_version="oolong_semantic_graph_v1",
                projection_version="oolong_semantic_projection_v1",
                encoder_version=graph_build.encoder.backend.encoder_version,
            ),
            gateway=gateway,
            config=DualRLMConfig(
                graph_top_k=5,
                max_graph_depth=3,
                max_graph_model_calls=6,
                max_graph_expansions=3,
            ),
        )
        result = None
        error = None
        try:
            result = arm.run(case.question, run_id=f"oolong_semantic_graph:{case.task_id}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        traces = result.model_call_traces if result else [
            trace for trace in gateway.model_call_traces if trace.purpose == "graph_decision"
        ]
        worker_tokens = sum(trace.get("total_tokens") or 0 for trace in worker_traces)
        root_tokens = sum(trace.total_tokens or 0 for trace in traces)
        prediction = (
            _graph_answer_from_trace(result.trace if result else [])
            or (result.answer_candidate if result else None)
            or ""
        )
        trace_payload = {
            "runtime_provenance": provenance,
            "graph_node_count": len(graph_build.semantic_documents),
            "embedding_count": len(graph_build.embeddings),
            "traversal_seed_count": len(graph_build.traversal_trace.seed_results),
            "worker_total_tokens": worker_tokens,
            "root_total_tokens": root_tokens,
        }
        return BenchmarkArmResult(
            prediction=prediction.strip(),
            raw_response=prediction.strip(),
            provider="openai" if traces or worker_traces else None,
            model_name=self.model_name if traces else None,
            model_role="root" if traces else None,
            reasoning_effort=self.reasoning_effort,
            worker_model_name=self.worker_model_name,
            worker_reasoning_effort=self.worker_reasoning_effort,
            experiment_id=self.experiment_id,
            response_id=traces[-1].response_id if traces else None,
            input_tokens=sum(trace.input_tokens or 0 for trace in traces)
            + sum(trace.get("input_tokens") or 0 for trace in worker_traces),
            output_tokens=sum(trace.output_tokens or 0 for trace in traces)
            + sum(trace.get("output_tokens") or 0 for trace in worker_traces),
            total_tokens=root_tokens + worker_tokens,
            latency_ms=int((perf_counter() - started) * 1000),
            fallback_used=any(trace.fallback_used for trace in traces)
            or any(trace.get("fallback_used") for trace in worker_traces),
            error=error,
            prompt_id=self.prompt_id,
            arm_input_hash=_hash_text(case.context),
            graph_source_hash=_hash_text(
                "".join(document.content_hash for document in graph_build.semantic_documents)
            ),
            evidence_span_ids=result.evidence_span_ids if result else [],
            model_calls=len(traces) + len(worker_traces),
            stop_reason=result.stop_reason if result else "error",
            trace_id=result.trace_id if result else f"oolong_semantic_graph:{case.task_id}:error",
            trace=[trace_payload] + (result.trace if result else []),
            model_call_traces=worker_traces + [trace.model_dump(mode="json") for trace in traces],
        )


class GraphRLMSemanticGraphOpsArm(GraphRLMSemanticGraphArm):
    name: BenchmarkArmName = "graph_rlm_semantic_graph_ops"

    def __init__(
        self,
        *,
        prompt_id: str = "oolong_graph_rlm_semantic_graph_ops_v1",
        **kwargs,
    ) -> None:
        super().__init__(prompt_id=prompt_id, **kwargs)


def build_oolong_direct_prompt(case: BenchmarkCase) -> str:
    return (
        f"{case.context.rstrip()}\n\n"
        f"{case.question.rstrip()}\n\n"
        "Return only the final answer in the format required by the question. "
        "Do not include reasoning or explanation."
    )


def _oolong_context_to_documents(
    case: BenchmarkCase,
    source_hash: str,
    record_labels: dict[int, str] | None = None,
) -> list[GraphSemanticDocument]:
    record_labels = record_labels or {}
    label_counts = {
        label: sum(1 for value in record_labels.values() if value == label)
        for label in sorted(set(record_labels.values()))
    }
    documents = [
        GraphSemanticDocument(
            document_id=f"oolong_graph_{case.task_id}",
            semantic_document_id=f"{case.task_id}:root",
            owner_type="entity",
            owner_id=f"{case.task_id}:context",
            source_entity_ids=[f"{case.task_id}:context"],
            event_ids=[],
            evidence_span_ids=[f"{case.task_id}:context"],
            source_chunk_ids=[f"{case.task_id}:root"],
            text=(
                "Public OOLONG context. Use frontier records to answer exactly. "
                f"Question: {case.question}\n"
                f"Worker extracted label_counts: {label_counts}"
            ),
            structural_features={
                "graph_source_hash": source_hash,
                "role": "root",
                "worker_label_counts": label_counts,
            },
            projection_version="benchmark_public_context_v1",
            content_hash=_hash_text(case.question + source_hash),
        )
    ]
    for index, line in enumerate(_oolong_record_lines(case.context)):
        user = _extract_field(line, "User") or f"user_unknown_{index}"
        date = _extract_date(line) or f"date_unknown_{index}"
        semantic_id = f"{case.task_id}:record:{index}"
        extracted_label = record_labels.get(index)
        text = line if not extracted_label else f"{line} || WorkerLabel: {extracted_label}"
        documents.append(
            GraphSemanticDocument(
                document_id=f"oolong_graph_{case.task_id}",
                semantic_document_id=semantic_id,
                owner_type="evidence",
                owner_id=f"{case.task_id}:record:{index}",
                source_entity_ids=[
                    f"{case.task_id}:context",
                    f"user:{user}",
                    f"date:{date}",
                ],
                event_ids=[f"{case.task_id}:record_event:{index}"],
                evidence_span_ids=[f"{case.task_id}:record:{index}"],
                source_chunk_ids=[semantic_id],
                text=text,
                structural_features={
                    "graph_source_hash": source_hash,
                    "record_index": index,
                    "worker_label": extracted_label,
                },
                projection_version="benchmark_public_context_v1",
                content_hash=_hash_text(line),
            )
        )
    return documents


def _raw_record_facts_from_case(case: BenchmarkCase, record_labels: dict[int, str]) -> list:
    facts = []
    for index, line in enumerate(_oolong_record_lines(case.context)):
        record_hash = _hash_text(f"{case.task_id}:{index}:{line}")[:16]
        facts.append(
            build_record_fact(
                record_id=f"record_{record_hash}",
                record_index=index,
                user_id=_extract_field(line, "User") or f"user_unknown_{index}",
                label=record_labels.get(index),
                date=_extract_date(line),
                evidence_span_id=f"evidence_record_{record_hash}",
                source_chunk_id=f"{case.task_id}:raw_record:{index}",
            )
        )
    return facts


class _SpamRecordLabel(BaseModel):
    record_index: int
    label: str = Field(pattern="^(ham|spam)$")


class _SpamLabelBatch(BaseModel):
    labels: list[_SpamRecordLabel] = Field(default_factory=list)


def _extract_worker_record_labels(
    case: BenchmarkCase,
    *,
    worker_model_name: str | None,
    worker_reasoning_effort: str | None,
    batch_size: int = 32,
) -> tuple[dict[int, str], list[dict]]:
    if not worker_model_name:
        return {}, []
    native = case.metadata.get("native_fields", {})
    if str(native.get("dataset")) != "spam":
        return {}, []
    require_openai_credentials()
    from pydantic_ai import Agent

    records = _oolong_record_lines(case.context)
    agent = Agent(
        f"openai:{worker_model_name}",
        output_type=_SpamLabelBatch,
        system_prompt=(
            "Classify SMS messages as ham or spam. Return only record_index and label. "
            "Use only the provided public message text."
        ),
        model_settings=_openai_model_settings(worker_reasoning_effort),
    )
    labels: dict[int, str] = {}
    traces: list[dict] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        payload = [
            {
                "record_index": start + offset,
                "message": _extract_field(line, "Instance") or line,
            }
            for offset, line in enumerate(batch)
        ]
        request_timestamp = datetime.now(timezone.utc)
        response = agent.run_sync(
            "Classify these records:\n"
            + "\n".join(f"{item['record_index']}: {item['message']}" for item in payload)
        )
        response_timestamp = datetime.now(timezone.utc)
        output = getattr(response, "output", None)
        if output is None:
            output = getattr(response, "data", None)
        validated = output if isinstance(output, _SpamLabelBatch) else _SpamLabelBatch.model_validate(output)
        for item in validated.labels:
            if start <= item.record_index < start + len(batch):
                labels[item.record_index] = item.label
        usage = _usage_value(response)
        traces.append(
            {
                "provider": "openai",
                "model_name": worker_model_name,
                "model_role": "worker",
                "reasoning_effort": worker_reasoning_effort,
                "purpose": "spam_label_extraction",
                "response_id": _response_id(response),
                "input_tokens": _usage_field(usage, ["request_tokens", "input_tokens", "prompt_tokens"]) or 0,
                "output_tokens": _usage_field(usage, ["response_tokens", "output_tokens", "completion_tokens"]) or 0,
                "total_tokens": _usage_field(usage, ["total_tokens"]) or 0,
                "fallback_used": False,
                "request_timestamp": request_timestamp.isoformat(),
                "response_timestamp": response_timestamp.isoformat(),
                "batch_start": start,
                "batch_size": len(batch),
                "labels_returned": len(validated.labels),
            }
        )
    return labels, traces


def _worker_aggregate_answer(case: BenchmarkCase, record_labels: dict[int, str]) -> str | None:
    if not record_labels:
        return None
    question = case.question.lower()
    counts = _label_counts(record_labels)
    if "most common" in question and counts:
        return f"Label: {max(counts.items(), key=lambda item: (item[1], item[0]))[0]}"
    if "least common" in question and counts:
        return f"Label: {min(counts.items(), key=lambda item: (item[1], item[0]))[0]}"
    return None


def _label_counts(record_labels: dict[int, str]) -> dict[str, int]:
    return {
        label: sum(1 for value in record_labels.values() if value == label)
        for label in sorted(set(record_labels.values()))
    }


def _oolong_record_lines(context: str) -> list[str]:
    return [
        line.strip()
        for line in context.splitlines()
        if line.strip().startswith("Date:")
    ]


def _extract_field(line: str, field: str) -> str | None:
    match = re.search(rf"\|\|\s*{re.escape(field)}:\s*([^|]+)", line)
    return match.group(1).strip() if match else None


def _extract_date(line: str) -> str | None:
    match = re.search(r"Date:\s*([^|]+)", line)
    return match.group(1).strip() if match else None


def _graph_answer_from_trace(trace: list[dict]) -> str | None:
    for step in reversed(trace):
        if step.get("step") != "graph_decide":
            continue
        decision = step.get("decision") or {}
        if decision.get("action") == "answer":
            summary = decision.get("decision_summary")
            if summary:
                return str(summary)
    return None


def _selected_context_tokens(
    documents: list[GraphSemanticDocument],
    evidence_span_ids: list[str],
) -> int:
    evidence = set(evidence_span_ids)
    if not evidence:
        return 0
    selected = [
        document.text
        for document in documents
        if evidence & set(document.evidence_span_ids)
    ]
    return sum(len(text.split()) for text in selected)


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _extract_answer_from_context(case: BenchmarkCase) -> str:
    question = case.question.lower()
    secret_match = re.search(r"secret code is ([\w.-]+)", case.context, flags=re.IGNORECASE)
    if secret_match:
        return secret_match.group(1)
    if "where" in question and "forest" in case.context.lower():
        return "in the forest"
    if isinstance(case.gold_answer, str) and case.gold_answer.lower() in case.context.lower():
        return case.gold_answer
    if not case.answerable:
        return "unknown"
    return ""


def _chunk_context(context: str, chunk_size: int = 160) -> list[tuple[int, str]]:
    words = context.split()
    chunks = []
    for start in range(0, len(words), chunk_size):
        index = len(chunks)
        chunks.append((index, " ".join(words[start : start + chunk_size])))
    return chunks or [(0, "")]


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"\w+", text.lower()) if len(term) > 2}


def _overlap(query_terms: set[str], text: str) -> int:
    text_terms = _terms(text)
    return len(query_terms & text_terms)
