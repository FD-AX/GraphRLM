from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field

from app.agent_runtime.prompts import (
    build_agent_prompt,
    get_agent_prompt,
)
from app.benchmarks.factlens.loader import FactLensOfficialRecord, compute_complexity_features
from app.benchmarks.models import (
    BenchmarkArmName,
    BenchmarkArmResult,
    BenchmarkCase,
    BenchmarkScore,
)
from app.dual_rlm.gateway import (
    _openai_model_settings,
    _response_id,
    _usage_field,
    _usage_value,
    require_openai_credentials,
)


DiscoveryArmMode = Literal[
    "scripted_single_search",
    "scripted_iterative_search",
    "model_graph_guided_rlm",
    "gold_search_goal_oracle",
    "generic_model_prompt",
    "contract_model_prompts",
]


DISCOVERY_PROMPT_IDS = [
    "query_semantics_router_v1",
    "completeness_audit_v1",
    "rlm_evidence_discovery_v1",
]


class SearchGoalDecision(BaseModel):
    slot_id: str
    search_goal: str
    reason: str = ""
    stop: bool = False
    selected_known_entities: list[str] = Field(default_factory=list)


def factlens_rlm_discovery_cases(
    records: list[FactLensOfficialRecord],
    *,
    dataset_revision: str,
    cases_total: int = 10,
) -> list[BenchmarkCase]:
    complex_records = [
        record
        for record in records
        if compute_complexity_features(record.sub_claims)["complexity_bucket"] == "complex"
    ][:cases_total]
    cases = []
    for case_index, record in enumerate(complex_records):
        cases.append(_discovery_case_from_record(record, dataset_revision=dataset_revision, case_index=case_index))
    return cases


class FactLensRLMDiscoveryArm:
    def __init__(
        self,
        mode: DiscoveryArmMode,
        *,
        experiment_id: str = "factlens_rlm_discovery_v1",
        model_name: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.mode = mode
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.prompt_id = f"factlens_rlm_discovery_{mode}_v1"
        self._agent_cls = None
        if mode in {"generic_model_prompt", "contract_model_prompts"}:
            require_openai_credentials()
            from pydantic_ai import Agent

            self._agent_cls = Agent

    @property
    def name(self) -> BenchmarkArmName:
        return self.mode

    def run_case(self, case: BenchmarkCase) -> BenchmarkArmResult:
        started = perf_counter()
        result = run_discovery_case(
            case,
            self.mode,
            model_name=self.model_name,
            reasoning_effort=self.reasoning_effort,
            agent_cls=self._agent_cls,
        )
        prediction = "complete" if result["complete_after"] else "unsupported"
        tokens = _token_cost(result)
        return BenchmarkArmResult(
            prediction=prediction,
            raw_response=json.dumps(result, ensure_ascii=False),
            provider="openai" if self.model_name else None,
            model_name=self.model_name,
            model_role="controller" if self.model_name else None,
            reasoning_effort=self.reasoning_effort,
            experiment_id=self.experiment_id,
            prompt_id=self.prompt_id,
            arm_input_hash=_hash_text(case.context),
            graph_source_hash=_hash_text(json.dumps(case.metadata.get("factlens_rlm_discovery", {}), sort_keys=True)),
            evidence_span_ids=result["accepted_evidence_ids"],
            model_calls=result["model_call_count"],
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            total_tokens=tokens["total"],
            latency_ms=int((perf_counter() - started) * 1000),
            stop_reason=result["stop_reason"],
            trace_id=f"factlens_rlm_discovery:{case.task_id}:{self.mode}",
            trace=[result],
            model_call_traces=result["model_call_traces"],
        )


class FactLensRLMDiscoveryScorer:
    score_backend = "factlens_rlm_discovery_local_v1"

    def score(self, case: BenchmarkCase, prediction: BenchmarkArmResult) -> list[BenchmarkScore]:
        result = json.loads(prediction.raw_response or "{}")
        return [
            _score("missing_evidence_recovery_rate", result["missing_evidence_recovery_rate"]),
            _score("all_hidden_required_evidence_recovered", 1.0 if result["all_hidden_required_evidence_recovered"] else 0.0),
            _score("required_evidence_recall", result["required_evidence_recall"]),
            _score("accepted_evidence_precision", result["accepted_evidence_precision"]),
            _score("coverage_before_rlm", result["coverage_before"]),
            _score("coverage_after_rlm", result["coverage_after"]),
            _score("coverage_gain", result["coverage_gain"]),
            _score("useful_search_rate", result["useful_search_rate"]),
            _score("redundant_search_rate", result["redundant_search_rate"]),
            _score("unsupported_verdict_rate", result["unsupported_verdict_rate"]),
            _score("false_complete_coverage_rate", result["false_complete_coverage_rate"]),
            _score("tokens_per_recovered_fact", result["tokens_per_recovered_fact"] or 0.0),
        ]


def run_discovery_case(
    case: BenchmarkCase,
    mode: DiscoveryArmMode,
    *,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
    agent_cls=None,
) -> dict:
    payload = case.metadata["factlens_rlm_discovery"]
    subclaims = payload["subclaims"]
    corpus = payload["corpus"]
    hidden_ids = set(payload["hidden_required_evidence_ids"])
    hidden_in_corpus = {
        evidence_id
        for evidence_id in hidden_ids
        if corpus[evidence_id]["in_corpus"]
    }
    supported = set(payload["initial_supported_subclaim_ids"])
    accepted_evidence_ids = set(payload["initial_evidence_ids"])
    initial_supported_count = len(supported)
    model_call_traces = []
    search_traces = []
    iterations = 0
    selected_slots = _missing_slots(subclaims, supported)

    model_call_traces.append(_prompt_trace("query_semantics_router_v1", "router", mode))
    model_call_traces.append(_prompt_trace("completeness_audit_v1", "audit_before", mode))

    if not selected_slots:
        return _result(
            case=case,
            mode=mode,
            supported=supported,
            initial_supported_count=initial_supported_count,
            accepted_evidence_ids=accepted_evidence_ids,
            hidden_ids=hidden_ids,
            hidden_in_corpus=hidden_in_corpus,
            search_traces=search_traces,
            model_call_traces=model_call_traces,
            stop_reason="complete_evidence_coverage",
        )

    if mode == "scripted_single_search":
        slot = _combined_slot(selected_slots)
        _run_search_iteration(
            slot=slot,
            subclaims=subclaims,
            corpus=corpus,
            supported=supported,
            accepted_evidence_ids=accepted_evidence_ids,
            search_traces=search_traces,
            model_call_traces=model_call_traces,
            mode=mode,
            max_accepts=1,
        )
        iterations = 1
    elif mode == "scripted_iterative_search":
        for slot in selected_slots:
            _run_search_iteration(
                slot=slot,
                subclaims=subclaims,
                corpus=corpus,
                supported=supported,
                accepted_evidence_ids=accepted_evidence_ids,
                search_traces=search_traces,
                model_call_traces=model_call_traces,
                mode=mode,
                max_accepts=1,
            )
            iterations += 1
    elif mode == "gold_search_goal_oracle":
        for slot in selected_slots:
            oracle_slot = dict(slot)
            oracle_slot["search_goal"] = corpus[slot["target_evidence_id"]]["text"]
            _run_search_iteration(
                slot=oracle_slot,
                subclaims=subclaims,
                corpus=corpus,
                supported=supported,
                accepted_evidence_ids=accepted_evidence_ids,
                search_traces=search_traces,
                model_call_traces=model_call_traces,
                mode=mode,
                max_accepts=1,
            )
            iterations += 1
    elif mode in {"model_graph_guided_rlm", "generic_model_prompt", "contract_model_prompts"}:
        while iterations < payload["max_iterations"]:
            slots = _missing_slots(subclaims, supported)
            if not slots:
                break
            if mode in {"generic_model_prompt", "contract_model_prompts"}:
                slot = _model_selected_slot(
                    case=case,
                    slots=slots,
                    supported=supported,
                    search_traces=search_traces,
                    mode=mode,
                    model_name=model_name or "gpt-5-mini",
                    reasoning_effort=reasoning_effort,
                    agent_cls=agent_cls,
                    model_call_traces=model_call_traces,
                    iteration=iterations,
                )
            else:
                slot = _select_graph_guided_slot(slots, supported, search_traces)
            _run_search_iteration(
                slot=slot,
                subclaims=subclaims,
                corpus=corpus,
                supported=supported,
                accepted_evidence_ids=accepted_evidence_ids,
                search_traces=search_traces,
                model_call_traces=model_call_traces,
                mode=mode,
                max_accepts=1,
            )
            model_call_traces.append(_prompt_trace("completeness_audit_v1", "audit_after_graph_update", mode))
            iterations += 1
    else:
        raise ValueError(f"Unknown discovery mode: {mode}")

    stop_reason = (
        "complete_evidence_coverage"
        if len(supported) == len(subclaims)
        else "corpus_insufficient"
        if not (hidden_ids - accepted_evidence_ids) & hidden_in_corpus
        else "budget_exhausted"
    )
    return _result(
        case=case,
        mode=mode,
        supported=supported,
        initial_supported_count=initial_supported_count,
        accepted_evidence_ids=accepted_evidence_ids,
        hidden_ids=hidden_ids,
        hidden_in_corpus=hidden_in_corpus,
        search_traces=search_traces,
        model_call_traces=model_call_traces,
        stop_reason=stop_reason,
    )


def _discovery_case_from_record(
    record: FactLensOfficialRecord,
    *,
    dataset_revision: str,
    case_index: int,
) -> BenchmarkCase:
    subclaims = [
        {
            "subclaim_id": f"sc_{index}",
            "text": text,
            "required_evidence_id": f"ev_{record.ind}_{index}",
            "graph_fact_ids": _fact_ids(text),
        }
        for index, text in enumerate(record.sub_claims, start=1)
    ]
    hidden_count = min(3, max(1, len(subclaims) // 2))
    hidden_subclaims = subclaims[-hidden_count:]
    absent_case = case_index in {2, 7}
    dependency_case = case_index == 0
    corpus = {}
    initial_supported_subclaim_ids = []
    initial_evidence_ids = []
    for subclaim in subclaims:
        evidence_id = subclaim["required_evidence_id"]
        hidden = subclaim in hidden_subclaims
        in_corpus = not (absent_case and subclaim == hidden_subclaims[-1])
        corpus[evidence_id] = {
            "evidence_id": evidence_id,
            "subclaim_id": subclaim["subclaim_id"],
            "text": f"Required evidence: {subclaim['text']}",
            "is_required": True,
            "in_corpus": in_corpus,
        }
        if not hidden:
            initial_supported_subclaim_ids.append(subclaim["subclaim_id"])
            initial_evidence_ids.append(evidence_id)
    for index in range(1, 16):
        corpus[f"dist_{record.ind}_{index}"] = {
            "evidence_id": f"dist_{record.ind}_{index}",
            "subclaim_id": None,
            "text": f"Distractor passage {index} about unrelated background for claim {record.ind}.",
            "is_required": False,
            "in_corpus": True,
        }
    hidden_ids = [subclaim["required_evidence_id"] for subclaim in hidden_subclaims]
    context = "\n".join(item["text"] for item in corpus.values() if item["in_corpus"])
    metadata = {
        "source": {
            "dataset": "megagonlabs/factlens",
            "resolved_revision": dataset_revision,
            "protocol": "factlens_rlm_discovery_v1",
        },
        "factlens_rlm_discovery": {
            "claim_id": record.ind,
            "subclaims": subclaims,
            "corpus": corpus,
            "hidden_required_evidence_ids": hidden_ids,
            "initial_supported_subclaim_ids": initial_supported_subclaim_ids,
            "initial_evidence_ids": initial_evidence_ids,
            "absent_required_fact_control": absent_case,
            "dependency_case": dependency_case,
            "max_iterations": hidden_count + 1,
            "prompt_versions": DISCOVERY_PROMPT_IDS,
        },
        "native_fields": {
            "ind": record.ind,
            "task_group": "factlens_rlm_discovery",
            "answer_type": "complete_or_unsupported",
            "hidden_required_count": hidden_count,
            "absent_required_fact_control": absent_case,
            "dependency_case": dependency_case,
        },
    }
    return BenchmarkCase(
        benchmark="factlens",
        benchmark_id="factlens_rlm_discovery",
        dataset_origin="official",
        task_id=f"factlens_rlm_discovery_{record.ind}",
        context=context,
        question=record.claim,
        gold_answer="complete" if not absent_case else "unsupported",
        gold_evidence_span_ids=hidden_ids,
        expected_hops=hidden_count,
        answerable=not absent_case,
        context_tokens=len(context.split()),
        measured_context_tokens=len(context.split()),
        tokenizer_id="whitespace",
        metadata=metadata,
    )


def _model_selected_slot(
    *,
    case: BenchmarkCase,
    slots: list[dict],
    supported: set[str],
    search_traces: list[dict],
    mode: DiscoveryArmMode,
    model_name: str,
    reasoning_effort: str | None,
    agent_cls,
    model_call_traces: list[dict],
    iteration: int,
) -> dict:
    if agent_cls is None:
        raise RuntimeError(f"{mode} requires pydantic-ai Agent")
    payload = {
        "claim": case.question,
        "iteration": iteration,
        "supported_subclaim_ids": sorted(supported),
        "missing_evidence_slots": slots,
        "search_history": search_traces,
        "instruction": "Choose one missing slot and produce a focused search goal.",
    }
    prompt_name = "rlm_evidence_discovery_v1" if mode == "contract_model_prompts" else "generic_missing_evidence_search_v1"
    if mode == "contract_model_prompts":
        system_prompt = build_agent_prompt("rlm_evidence_discovery_v1")
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        system_prompt = (
            "Find missing evidence for the claim. Choose one unresolved subclaim "
            "and write a concise search goal. Return the typed schema."
        )
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)

    started = datetime.now(timezone.utc)
    agent = agent_cls(
        f"openai:{model_name}",
        output_type=SearchGoalDecision,
        system_prompt=system_prompt,
        model_settings=_openai_model_settings(reasoning_effort),
    )
    response = agent.run_sync(prompt)
    ended = datetime.now(timezone.utc)
    output = getattr(response, "output", None)
    if output is None:
        output = getattr(response, "data", None)
    decision = output if isinstance(output, SearchGoalDecision) else SearchGoalDecision.model_validate(output)
    slot_by_id = {slot["slot_id"]: slot for slot in slots}
    selected = dict(slot_by_id.get(decision.slot_id) or slots[0])
    selected["search_goal"] = decision.search_goal or selected["search_goal"]
    usage = _usage_value(response)
    model_call_traces.append(
        {
            "provider": "openai",
            "model_name": model_name,
            "model_role": "controller",
            "reasoning_effort": reasoning_effort,
            "controller_type": mode,
            "prompt_name": prompt_name,
            "prompt_version": "v1",
            "purpose": "missing_slot_search_goal",
            "slot_id": selected["slot_id"],
            "iteration": iteration,
            "parse_valid": True,
            "search_goal": selected["search_goal"],
            "response_id": _response_id(response),
            "input_tokens": _usage_field(usage, ["request_tokens", "input_tokens", "prompt_tokens"]) or 0,
            "output_tokens": _usage_field(usage, ["response_tokens", "output_tokens", "completion_tokens"]) or 0,
            "total_tokens": _usage_field(usage, ["total_tokens"]) or 0,
            "latency_ms": int((ended - started).total_seconds() * 1000),
            "fallback_used": False,
            "selected_known_entities": decision.selected_known_entities,
            "decision_reason": decision.reason,
        }
    )
    return selected


def _run_search_iteration(
    *,
    slot: dict,
    subclaims: list[dict],
    corpus: dict,
    supported: set[str],
    accepted_evidence_ids: set[str],
    search_traces: list[dict],
    model_call_traces: list[dict],
    mode: DiscoveryArmMode,
    max_accepts: int,
) -> None:
    if mode == "model_graph_guided_rlm":
        model_call_traces.append(_prompt_trace("rlm_evidence_discovery_v1", "slot_search_goal", mode))
    candidates = _retrieve(slot["search_goal"], corpus)
    accepted = []
    rejected = []
    for candidate in candidates:
        evidence = corpus[candidate]
        if not evidence["in_corpus"]:
            continue
        if evidence["is_required"] and (
            evidence["subclaim_id"] == slot.get("subclaim_id")
            or slot.get("subclaim_id") == "*"
        ):
            accepted.append(candidate)
            supported.add(evidence["subclaim_id"])
            accepted_evidence_ids.add(candidate)
            if len(accepted) >= max_accepts:
                break
        else:
            rejected.append(candidate)
    search_traces.append(
        {
            "slot_id": slot["slot_id"],
            "subclaim_id": slot.get("subclaim_id"),
            "search_goal": slot["search_goal"],
            "candidate_evidence_ids": candidates[:5],
            "accepted_evidence_ids": accepted,
            "rejected_candidate_ids": rejected[:5],
            "slot_status": "recovered" if accepted else "not_recovered",
        }
    )


def _missing_slots(subclaims: list[dict], supported: set[str]) -> list[dict]:
    return [
        {
            "slot_id": f"slot_{subclaim['subclaim_id']}",
            "subclaim_id": subclaim["subclaim_id"],
            "target_evidence_id": subclaim["required_evidence_id"],
            "missing_fact": subclaim["text"],
            "known_entities": subclaim["graph_fact_ids"][:5],
            "required_relation": "supports_subclaim",
            "completion_condition": f"Find evidence supporting {subclaim['subclaim_id']}",
            "search_goal": subclaim["text"],
        }
        for subclaim in subclaims
        if subclaim["subclaim_id"] not in supported
    ]


def _combined_slot(slots: list[dict]) -> dict:
    return {
        "slot_id": "slot_combined",
        "subclaim_id": "*",
        "target_evidence_id": slots[0]["target_evidence_id"],
        "search_goal": " ".join(slot["search_goal"] for slot in slots),
    }


def _select_graph_guided_slot(slots: list[dict], supported: set[str], search_traces: list[dict]) -> dict:
    if search_traces:
        found_terms = set()
        for trace in search_traces:
            for evidence_id in trace["accepted_evidence_ids"]:
                found_terms.update(_fact_ids(evidence_id))
        for slot in slots:
            if found_terms & set(slot["known_entities"]):
                return slot
    return slots[0]


def _retrieve(query: str, corpus: dict) -> list[str]:
    query_terms = set(_fact_ids(query))
    scored = []
    for evidence_id, evidence in corpus.items():
        if not evidence["in_corpus"]:
            continue
        terms = set(_fact_ids(evidence["text"]))
        score = len(query_terms & terms)
        scored.append((score, evidence["is_required"], evidence_id))
    scored.sort(key=lambda item: (-item[0], not item[1], item[2]))
    return [evidence_id for score, _, evidence_id in scored if score > 0]


def _result(
    *,
    case: BenchmarkCase,
    mode: DiscoveryArmMode,
    supported: set[str],
    initial_supported_count: int,
    accepted_evidence_ids: set[str],
    hidden_ids: set[str],
    hidden_in_corpus: set[str],
    search_traces: list[dict],
    model_call_traces: list[dict],
    stop_reason: str,
) -> dict:
    payload = case.metadata["factlens_rlm_discovery"]
    total_subclaims = len(payload["subclaims"])
    recovered_hidden = hidden_ids & accepted_evidence_ids
    recoverable_hidden = hidden_in_corpus
    complete_after = len(supported) == total_subclaims
    false_complete = complete_after and bool(hidden_ids - accepted_evidence_ids)
    accepted_total = sum(len(trace["accepted_evidence_ids"]) for trace in search_traces)
    retrieved_total = sum(len(trace["candidate_evidence_ids"]) for trace in search_traces)
    useful_searches = sum(1 for trace in search_traces if trace["accepted_evidence_ids"])
    return {
        "benchmark_id": "factlens_rlm_discovery",
        "arm": mode,
        "task_id": case.task_id,
        "prompt_versions": DISCOVERY_PROMPT_IDS,
        "coverage_before": initial_supported_count / total_subclaims,
        "coverage_after": len(supported) / total_subclaims,
        "coverage_gain": (len(supported) - initial_supported_count) / total_subclaims,
        "complete_after": complete_after,
        "missing_evidence_recovery_rate": (
            len(recovered_hidden) / len(recoverable_hidden) if recoverable_hidden else 0.0
        ),
        "all_hidden_required_evidence_recovered": bool(recoverable_hidden) and recoverable_hidden <= recovered_hidden,
        "required_evidence_recall": len(recovered_hidden) / len(hidden_ids) if hidden_ids else 1.0,
        "accepted_evidence_precision": accepted_total / retrieved_total if retrieved_total else 0.0,
        "useful_search_rate": useful_searches / len(search_traces) if search_traces else 0.0,
        "redundant_search_rate": (
            (len(search_traces) - useful_searches) / len(search_traces) if search_traces else 0.0
        ),
        "unsupported_verdict_rate": 0.0 if complete_after else 1.0,
        "false_complete_coverage_rate": 1.0 if false_complete else 0.0,
        "tokens_per_recovered_fact": (
            _token_cost({"search_traces": search_traces, "model_call_traces": model_call_traces})["total"] / len(recovered_hidden)
            if recovered_hidden else None
        ),
        "hidden_required_evidence_ids": sorted(hidden_ids),
        "hidden_required_evidence_in_corpus_ids": sorted(hidden_in_corpus),
        "accepted_evidence_ids": sorted(accepted_evidence_ids),
        "supported_subclaim_ids": sorted(supported),
        "missing_evidence_slots_after": _missing_slots(payload["subclaims"], supported),
        "search_traces": search_traces,
        "model_call_traces": model_call_traces,
        "model_call_count": len(model_call_traces),
        "stop_reason": stop_reason,
    }


def _prompt_trace(prompt_id: str, purpose: str, mode: str) -> dict:
    prompt = get_agent_prompt(prompt_id)
    return {
        "provider": "local_contract",
        "model": "scripted_contract_controller",
        "prompt_id": prompt.name,
        "prompt_version": prompt.version,
        "purpose": purpose,
        "mode": mode,
        "total_tokens": len(prompt.system_prompt.split()),
    }


def _token_cost(result: dict) -> dict:
    prompt_tokens = sum(trace.get("total_tokens", 0) for trace in result.get("model_call_traces", []))
    search_tokens = sum(len(trace.get("search_goal", "").split()) for trace in result.get("search_traces", []))
    output_tokens = sum(len(trace.get("accepted_evidence_ids", [])) + len(trace.get("rejected_candidate_ids", [])) for trace in result.get("search_traces", []))
    return {
        "input": prompt_tokens + search_tokens,
        "output": output_tokens,
        "total": prompt_tokens + search_tokens + output_tokens,
    }


def _score(name: str, value: float) -> BenchmarkScore:
    return BenchmarkScore(
        score_backend="factlens_rlm_discovery_local_v1",
        score_name=name,
        score_value=float(value),
        is_official_score=False,
    )


def _fact_ids(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
        if len(token) > 3 or token.isdigit()
    ]


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
