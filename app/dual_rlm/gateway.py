from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel

from app.dual_rlm.models import (
    GraphRLMDecision,
    GraphRLMState,
    ModelCallTrace,
    SourceChunk,
    TextRLMResult,
    TextRLMState,
)


TModel = TypeVar("TModel", bound=BaseModel)


def require_openai_credentials() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --controller gpt5")


class PydanticAIGPTGateway:
    """OpenAI GPT gateway with Pydantic-validated typed outputs.

    The class is intentionally strict: it never falls back to deterministic
    decisions, and it records provider response metadata for acceptance tests.
    """

    def __init__(
        self,
        model_name: str = "gpt-5",
        provider: str = "openai",
        require_real_model: bool = True,
        reasoning_effort: str | None = None,
        model_role: str = "root",
    ) -> None:
        if require_real_model:
            require_openai_credentials()
        try:
            from pydantic_ai import Agent
        except ImportError as exc:
            raise RuntimeError("pydantic-ai package is required for PydanticAIGPTGateway") from exc

        self.provider = provider
        self.model_name = model_name
        self.require_real_model = require_real_model
        self.reasoning_effort = reasoning_effort
        self.model_role = model_role
        self._agent_cls = Agent
        self._model_call_traces: list[ModelCallTrace] = []

    @property
    def model_call_traces(self) -> list[ModelCallTrace]:
        return self._model_call_traces

    def decide_graph(self, state: GraphRLMState) -> GraphRLMDecision:
        sections = _graph_prompt_sections(state)
        prompt = (
            "You control a read-only Graph-RLM traversal. "
            "Choose exactly one next action. You may only select transition IDs "
            "from presented_frontier. Do not invent owners, documents, or writes.\n\n"
            f"State:\n{json.dumps(_jsonable_state(state), ensure_ascii=False, indent=2)}"
        )
        return self._call_model(
            purpose="graph_decision",
            prompt=prompt,
            output_type=GraphRLMDecision,
            prompt_sections=sections,
        )

    def inspect_text(self, state: TextRLMState, chunks: list[SourceChunk]) -> TextRLMResult:
        sections = {
            "instruction": (
                "You control a read-only Text-RLM recursive inspection. "
                "Use only the provided chunks. Return evidence IDs only from the "
                "chunk IDs shown in the request. Do not write graph facts."
            ),
            "state": json.dumps(_jsonable_state(state), ensure_ascii=False, indent=2),
            "chunks": json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False, indent=2),
        }
        prompt = f"{sections['instruction']}\n\nState:\n{sections['state']}\n\nChunks:\n{sections['chunks']}"
        return self._call_model(
            purpose="text_inspection",
            prompt=prompt,
            output_type=TextRLMResult,
            prompt_sections=sections,
        )

    def _call_model(
        self,
        *,
        purpose: str,
        prompt: str,
        output_type: type[TModel],
        prompt_sections: dict[str, str] | None = None,
    ) -> TModel:
        call_id = f"model_call_{uuid4().hex[:12]}"
        request_timestamp = datetime.now(timezone.utc)
        agent = self._agent_cls(
            f"openai:{self.model_name}",
            output_type=output_type,
            system_prompt="Return a typed decision. Do not invent graph IDs or write actions.",
            model_settings=_openai_model_settings(self.reasoning_effort),
        )
        response = agent.run_sync(prompt)
        response_timestamp = datetime.now(timezone.utc)
        output = getattr(response, "output", None)
        if output is None:
            output = getattr(response, "data", None)
        if output is None:
            raise RuntimeError(f"{purpose}: PydanticAI returned no typed output")
        validated = output if isinstance(output, output_type) else output_type.model_validate(output)
        usage = _usage_value(response)
        trace = ModelCallTrace(
            call_id=call_id,
            provider=self.provider,
            model_name=self.model_name,
            model_role=self.model_role,
            reasoning_effort=self.reasoning_effort,
            purpose=purpose,
            request_timestamp=request_timestamp,
            response_timestamp=response_timestamp,
            input_tokens=_usage_field(usage, ["request_tokens", "input_tokens", "prompt_tokens"]),
            output_tokens=_usage_field(usage, ["response_tokens", "output_tokens", "completion_tokens"]),
            total_tokens=_usage_field(usage, ["total_tokens"]),
            response_id=_response_id(response),
            validated_output_type=output_type.__name__,
            fallback_used=False,
            prompt_section_estimates=_prompt_section_estimates(
                prompt_sections or {"prompt": prompt},
                output_type=output_type,
            ),
            prompt_section_hashes=_prompt_section_hashes(prompt_sections or {"prompt": prompt}),
        )
        self._model_call_traces.append(trace)
        return validated


def _jsonable_state(state) -> dict:
    serialized = {key: _jsonable_value(value) for key, value in dict(state).items()}
    if isinstance(serialized.get("path"), list):
        serialized["path"] = [_compact_path_step(step) for step in serialized["path"]]
    if isinstance(serialized.get("model_call_traces"), list):
        serialized["model_call_trace_count"] = len(serialized["model_call_traces"])
        serialized["model_call_traces"] = []
    return serialized


def _graph_prompt_sections(state: GraphRLMState) -> dict[str, str]:
    serialized = _jsonable_state(state)
    query_payload = {
        "original_query": serialized.get("original_query"),
        "query": serialized.get("query"),
        "current_subquery": serialized.get("current_subquery"),
    }
    state_payload = {
        key: serialized.get(key)
        for key in [
            "graph_view",
            "current_owner_id",
            "current_document_ids",
            "frontier_transition_ids",
            "depth",
            "max_depth",
            "remaining_model_calls",
            "remaining_expansions",
            "last_decision",
        ]
    }
    evidence_payload = {
        "presented_evidence_ids": serialized.get("presented_evidence_ids"),
        "collected_evidence_ids": serialized.get("collected_evidence_ids"),
    }
    history_payload = {
        "visited_owner_ids": serialized.get("visited_owner_ids"),
        "visited_document_ids": serialized.get("visited_document_ids"),
        "visited_transition_ids": serialized.get("visited_transition_ids"),
        "path": serialized.get("path"),
        "model_call_trace_count": serialized.get("model_call_trace_count"),
    }
    return {
        "system": "Return a typed decision. Do not invent graph IDs or write actions.",
        "instruction": (
            "You control a read-only Graph-RLM traversal. "
            "Choose exactly one next action. You may only select transition IDs "
            "from presented_frontier. Do not invent owners, documents, or writes."
        ),
        "query": json.dumps(query_payload, ensure_ascii=False, indent=2),
        "state": json.dumps(state_payload, ensure_ascii=False, indent=2),
        "frontier": json.dumps(serialized.get("presented_frontier", []), ensure_ascii=False, indent=2),
        "visited_evidence": json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        "history": json.dumps(history_payload, ensure_ascii=False, indent=2),
        "tool_schema": json.dumps(GraphRLMDecision.model_json_schema(), ensure_ascii=False, sort_keys=True),
    }


def _prompt_section_estimates(
    sections: dict[str, str],
    *,
    output_type: type[BaseModel],
) -> dict[str, dict[str, int]]:
    estimates = {
        key: {
            "chars": len(value),
            "estimated_tokens": _estimate_tokens(value),
        }
        for key, value in sections.items()
    }
    if "tool_schema" not in estimates:
        schema = json.dumps(output_type.model_json_schema(), ensure_ascii=False, sort_keys=True)
        estimates["tool_schema"] = {
            "chars": len(schema),
            "estimated_tokens": _estimate_tokens(schema),
        }
    return estimates


def _prompt_section_hashes(sections: dict[str, str]) -> dict[str, str]:
    return {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in sections.items()
    }


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _compact_path_step(step):
    if not isinstance(step, dict):
        return step
    compacted = dict(step)
    presented_frontier = compacted.pop("presented_frontier", None)
    if isinstance(presented_frontier, list):
        compacted["presented_frontier_count"] = len(presented_frontier)
    return compacted


def _jsonable_value(value):
    if hasattr(value, "model_dump"):
        return _jsonable_value(value.model_dump())
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    return value


def _usage_value(response):
    usage = getattr(response, "usage", None)
    if _has_usage_fields(usage):
        return usage
    if callable(usage):
        return usage()
    return usage


def _has_usage_fields(usage) -> bool:
    if usage is None or isinstance(usage, dict):
        return False
    return any(
        hasattr(usage, name)
        for name in [
            "request_tokens",
            "input_tokens",
            "prompt_tokens",
            "response_tokens",
            "output_tokens",
            "completion_tokens",
            "total_tokens",
        ]
    )


def _usage_field(usage, names: list[str]) -> int | None:
    if usage is None:
        return None
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    if isinstance(usage, dict):
        for name in names:
            value = usage.get(name)
            if value is not None:
                return int(value)
    return None


def _response_id(response) -> str | None:
    for attr in ["run_id", "id", "conversation_id"]:
        value = getattr(response, attr, None)
        if value:
            return str(value)
    model_response = getattr(response, "response", None)
    for attr in ["id", "provider_response_id"]:
        value = getattr(model_response, attr, None)
        if value:
            return str(value)
    return None


def _openai_model_settings(reasoning_effort: str | None) -> dict | None:
    if not reasoning_effort:
        return None
    return {"openai_reasoning_effort": reasoning_effort}
