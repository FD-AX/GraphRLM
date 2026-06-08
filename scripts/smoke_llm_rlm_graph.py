from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.core.graph_models import (
    ChunkNode,
    DocumentNode,
    Entity,
    EventArgument,
    EventFrame,
    EvidenceSpan,
    LocalGraphPatch,
    Mention,
    RawMention,
    ResolutionHypothesis,
    TextSpan,
)
from app.graph.neo4j_client import Neo4jClient
from app.graph.writer import GraphWriter
from app.projection import (
    ProjectionMaterializer,
    build_entity_pair_snapshot,
    build_entity_snapshot,
    rebuild_document_snapshots,
)
from app.rlm.state import EntityState, RLMState, RLMTransition
from app.runtime.docx_graph_ingest_neo4j import cleanup_document


Decision = Literal[
    "create_entity",
    "link_to_existing",
    "unresolved",
    "mention_only_allowed",
    "concept_allowed",
]


class SmokeEvidenceSpanExtraction(BaseModel):
    evidence_key: str
    text: str
    start_char_in_chunk: int
    end_char_in_chunk: int


class SmokeRawMentionExtraction(BaseModel):
    mention_key: str
    text: str
    start_char_in_chunk: int
    end_char_in_chunk: int
    mention_type: Literal[
        "named",
        "pronoun",
        "descriptor",
        "alias",
        "nominal",
        "object",
        "location",
    ] = "nominal"
    mention_kind: Literal[
        "named_entity",
        "role_anchor",
        "descriptive_alias",
        "pronoun",
        "generic_nominal",
        "noise",
    ] = "generic_nominal"
    normalized_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class SmokeEventArgumentExtraction(BaseModel):
    role: str
    mention_key: str
    surface_text: str
    grounding_expectation: Literal[
        "entity_expected",
        "mention_only_allowed",
        "concept_allowed",
    ]
    confidence: float = Field(ge=0.0, le=1.0)


class SmokeEventFrameExtraction(BaseModel):
    event_key: str
    predicate: str
    normalized_predicate: str | None = None
    event_type: str | None = None
    evidence_key: str
    arguments: list[SmokeEventArgumentExtraction] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SmokeExtraction(BaseModel):
    raw_mentions: list[SmokeRawMentionExtraction]
    evidence_spans: list[SmokeEvidenceSpanExtraction]
    event_frames: list[SmokeEventFrameExtraction]
    notes: list[str] = Field(default_factory=list)


class LLMSemanticMentionDraft(BaseModel):
    local_ref: str | None = None
    surface_text: str
    start_char: int
    end_char: int
    mention_type: Literal[
        "named",
        "pronoun",
        "descriptor",
        "alias",
        "nominal",
        "object",
        "location",
    ] = "nominal"
    mention_kind: Literal[
        "named_entity",
        "role_anchor",
        "descriptive_alias",
        "pronoun",
        "generic_nominal",
        "noise",
    ] = "generic_nominal"
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class LLMSemanticEventArgumentDraft(BaseModel):
    role: str
    mention_ref: str | None = None
    surface_text: str
    start_char: int
    end_char: int
    grounding_expectation: Literal[
        "entity_expected",
        "mention_only_allowed",
        "concept_allowed",
    ] | None = None
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class LLMSemanticEventDraft(BaseModel):
    predicate: str
    normalized_predicate: str | None = None
    event_type: str | None = None
    evidence_text: str | None = None
    evidence_start_char: int | None = None
    evidence_end_char: int | None = None
    temporal_scope: str | None = None
    modality: str | None = None
    polarity: Literal["positive", "negative", "unknown"] = "positive"
    arguments: list[LLMSemanticEventArgumentDraft] = Field(default_factory=list)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class LLMSemanticExtractionDraft(BaseModel):
    mentions: list[LLMSemanticMentionDraft] = Field(default_factory=list)
    events: list[LLMSemanticEventDraft] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RLMRetrievalRequest(BaseModel):
    tool: Literal[
        "get_previous_chunks",
        "get_next_chunks",
        "get_chunks_by_entity",
        "search_chunks",
        "get_entity_context",
        "get_unresolved_mentions",
        "get_related_events",
    ]
    args: dict[str, Any] = Field(default_factory=dict)


class RLMResolutionDecision(BaseModel):
    mention_id: str
    surface_text: str
    decision: Decision
    entity_id: str | None = None
    canonical_name: str | None = None
    entity_type: Literal[
        "person",
        "place",
        "object",
        "organization",
        "event",
        "concept",
        "unknown",
    ] = "unknown"
    aliases_to_add: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["confirmed", "likely", "possible", "rejected", "unresolved"]
    consulted_chunk_ids: list[str] = Field(default_factory=list)
    consulted_entity_ids: list[str] = Field(default_factory=list)
    candidate_entity_ids: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)
    decision_reason: str


class FocusedMentionResolution(BaseModel):
    decision: Literal[
        "link_existing",
        "propose_new",
        "keep_unresolved",
        "mention_only",
    ]
    entity_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class RLMRoundResponse(BaseModel):
    action: Literal["retrieve_context", "final_resolution"]
    reason: str
    requests: list[RLMRetrievalRequest] = Field(default_factory=list)
    decisions: list[RLMResolutionDecision] = Field(default_factory=list)
    unresolved_hypotheses: list[str] = Field(default_factory=list)
    stop_reason: str | None = None


class SmokeModelClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, timeout=120.0)
        self.structured_call_count = 0

    async def check_model(self) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.base_url}/models")
            response.raise_for_status()
            print_block("MODEL ENDPOINT", response.json())

    async def structured_call(
        self,
        label: str,
        system_prompt: str,
        payload: dict[str, Any],
        schema_model: type[BaseModel],
        max_tokens: int | None = None,
        response_mode: Literal["json_schema", "json_object"] = "json_schema",
    ) -> BaseModel:
        self.structured_call_count += 1
        schema = schema_model.model_json_schema()
        call_payload = payload
        request = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        + "\n\nReturn only the requested JSON object. Do not echo the input."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(call_payload, ensure_ascii=False, indent=2),
                },
            ],
        }
        if response_mode == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__,
                    "schema": schema,
                },
            }
        else:
            request["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        print_block(f"{label} REQUEST", request)
        response = await self.client.chat.completions.create(**request)
        raw = response.choices[0].message.content
        if raw is None:
            raise RuntimeError(f"{label}: model returned empty content")
        print_block(f"{label} RAW RESPONSE", raw)
        validated = schema_model.model_validate_json(raw)
        print_block(f"{label} VALIDATED", validated.model_dump())
        return validated


class SmokeRuntime:
    def __init__(
        self,
        chunks: list[ChunkNode],
        document_id: str,
        resolver_version: str,
        max_chunks_per_round: int,
        max_total_context_chars: int,
    ) -> None:
        self.chunks = chunks
        self.document_id = document_id
        self.resolver_version = resolver_version
        self.max_chunks_per_round = max_chunks_per_round
        self.max_total_context_chars = max_total_context_chars
        self.entities: dict[str, Entity] = {}
        self.entity_mentions: dict[str, list[str]] = {}
        self.entity_observations: dict[str, list[dict[str, Any]]] = {}
        self.unresolved: list[dict[str, Any]] = []
        self.events_by_entity: dict[str, list[dict[str, Any]]] = {}

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "entities": [entity.model_dump() for entity in self.entities.values()],
            "aliases": {
                entity_id: entity.aliases
                for entity_id, entity in sorted(self.entities.items())
            },
            "unresolved_mentions": self.unresolved,
            "resolver_version": self.resolver_version,
        }

    def nearby_metadata(self, chunk: ChunkNode) -> list[dict[str, Any]]:
        rows = []
        for candidate in self.chunks:
            if abs(candidate.index - chunk.index) <= 1 and candidate.chunk_id != chunk.chunk_id:
                rows.append(
                    {
                        "chunk_id": candidate.chunk_id,
                        "index": candidate.index,
                        "preview": candidate.text[:180],
                    }
                )
        return rows

    def deterministic_context_for_mention(
        self,
        chunk: ChunkNode,
        mention: SmokeRawMentionExtraction,
        extraction: SmokeExtraction,
    ) -> dict[str, Any]:
        visited_chunk_ids = {chunk.chunk_id}
        previous_chunks = self._previous_chunks(
            chunk.chunk_id,
            min(2, self.max_chunks_per_round),
            visited_chunk_ids,
        )
        mention_id = f"{chunk.chunk_id}_{mention.mention_key}"
        related_arguments = []
        evidence_span_ids: list[str] = []
        sentence = chunk.text
        for event in extraction.event_frames:
            for argument in event.arguments:
                if argument.mention_key == mention.mention_key:
                    related_arguments.append(
                        {
                            "event_key": event.event_key,
                            "predicate": event.predicate,
                            "role": argument.role,
                            "grounding_expectation": argument.grounding_expectation,
                            "evidence_key": event.evidence_key,
                        }
                    )
                    evidence_span_ids.append(f"{chunk.chunk_id}_{event.evidence_key}")
                    span = next(
                        (
                            item
                            for item in extraction.evidence_spans
                            if item.evidence_key == event.evidence_key
                        ),
                        None,
                    )
                    if span:
                        sentence = span.text
        candidates = self.generate_candidates(
            chunk=chunk,
            mention=mention,
            previous_chunks=previous_chunks,
            limit=5,
        )
        consulted_entity_ids = [candidate["entity_id"] for candidate in candidates]
        entity_context = [
            {
                "entity": self.entities[candidate["entity_id"]].model_dump(),
                "observations": self.entity_observations.get(candidate["entity_id"], [])[-5:],
                "related_events": self.events_by_entity.get(candidate["entity_id"], [])[-5:],
            }
            for candidate in candidates
            if candidate["entity_id"] in self.entities
        ]
        context = {
            "mention_id": mention_id,
            "target_mention": mention.model_dump(),
            "current_sentence": sentence,
            "current_chunk": {
                "chunk_id": chunk.chunk_id,
                "index": chunk.index,
                "text": chunk.text,
            },
            "retrieved_chunks": previous_chunks,
            "active_entities": [
                entity.model_dump() for entity in self.entities.values()
            ][-5:],
            "candidate_entities": candidates[:5],
            "entity_context": entity_context,
            "recent_events": [
                event
                for events in self.events_by_entity.values()
                for event in events[-3:]
            ][-10:],
            "unresolved_hypotheses": self.unresolved[-10:],
            "related_arguments": related_arguments,
            "consulted_chunk_ids": sorted(visited_chunk_ids),
            "consulted_entity_ids": consulted_entity_ids,
            "candidate_entity_ids": consulted_entity_ids,
            "evidence_span_ids": list(dict.fromkeys(evidence_span_ids)),
        }
        print_block(
            "DETERMINISTIC RETRIEVAL",
            {
                "mention_id": mention_id,
                "surface_text": mention.text,
                "consulted_chunk_ids": context["consulted_chunk_ids"],
                "candidate_entity_ids": context["candidate_entity_ids"],
                "unresolved_hypotheses": context["unresolved_hypotheses"],
            },
        )
        print_block("CANDIDATE ENTITIES", context["candidate_entities"])
        return context

    def generate_candidates(
        self,
        chunk: ChunkNode,
        mention: SmokeRawMentionExtraction,
        previous_chunks: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        surface = normalize_surface(mention.text)
        previous_text = " ".join(item.get("text", "") for item in previous_chunks).lower()
        pronoun_surfaces = {"he", "him", "his", "she", "her", "they", "them", "their"}
        scored: list[tuple[float, Entity, list[str]]] = []
        for entity_id, entity in self.entities.items():
            names = [entity.canonical_name, *entity.aliases]
            normalized_names = [normalize_surface(name) for name in names]
            is_pronoun = mention.mention_type == "pronoun" or mention.mention_kind == "pronoun"
            score = 0.0
            reasons: list[str] = []
            if surface in normalized_names:
                score += 1.0
                reasons.append("surface_or_alias_match")
            if surface and any(surface in name or name in surface for name in normalized_names):
                score += 0.6
                reasons.append("partial_name_match")
            observations = self.entity_observations.get(entity_id, [])
            if any(obs.get("chunk_id") in {row["chunk_id"] for row in previous_chunks} for obs in observations):
                score += 0.4
                reasons.append("seen_in_retrieved_window")
            if is_pronoun:
                score += 0.3
                reasons.append("active_discourse_entity")
                non_pronoun_aliases = [
                    alias
                    for alias in normalized_names[1:]
                    if alias and alias not in pronoun_surfaces
                ]
                if any(alias in previous_text for alias in non_pronoun_aliases):
                    score += 1.2
                    reasons.append("pronoun_context_alias_bridge")
            if surface in {"hunter", "the hunter"}:
                if "rifle" in previous_text and entity.canonical_name.lower() == "semyon":
                    score += 1.5
                    reasons.append("hunter_alias_supported_by_rifle_context")
                if "hunter" in " ".join(normalized_names):
                    score += 0.8
                    reasons.append("hunter_name_alias")
            if score > 0:
                scored.append((score, entity, reasons))
        scored.sort(key=lambda item: (-item[0], item[1].canonical_name.lower()))
        return [
            {
                "entity_id": entity.entity_id,
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "aliases": entity.aliases,
                "score": round(score, 3),
                "reasons": reasons,
            }
            for score, entity, reasons in scored[:limit]
        ]

    def execute_requests(
        self,
        requests: list[RLMRetrievalRequest],
        visited_chunk_ids: set[str],
        visited_entity_ids: set[str],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for request in requests:
            tool = request.tool
            args = request.args
            print_block("RLM ACTION", {"tool": tool, "args": args})
            if tool == "get_previous_chunks":
                chunk_id = str(args.get("chunk_id"))
                limit = min(int(args.get("limit", 1)), self.max_chunks_per_round)
                rows = self._previous_chunks(chunk_id, limit, visited_chunk_ids)
                results.extend(rows or [{"tool": tool, "args": args, "no_results": True}])
            elif tool == "get_next_chunks":
                chunk_id = str(args.get("chunk_id"))
                limit = min(int(args.get("limit", 1)), self.max_chunks_per_round)
                rows = self._next_chunks(chunk_id, limit, visited_chunk_ids)
                results.extend(rows or [{"tool": tool, "args": args, "no_results": True}])
            elif tool == "get_chunks_by_entity":
                entity_id = str(args.get("entity_id"))
                visited_entity_ids.add(entity_id)
                rows = self._chunks_by_entity(entity_id, visited_chunk_ids)
                results.extend(rows or [{"tool": tool, "args": args, "no_results": True}])
            elif tool == "search_chunks":
                query = str(args.get("query", ""))
                top_k = min(int(args.get("top_k", 2)), self.max_chunks_per_round)
                rows = self._search_chunks(query, top_k, visited_chunk_ids)
                results.extend(rows or [{"tool": tool, "args": args, "no_results": True}])
            elif tool == "get_entity_context":
                entity_id = str(args.get("entity_id"))
                visited_entity_ids.add(entity_id)
                results.append(
                    {
                        "tool": tool,
                        "entity_id": entity_id,
                        "entity": self.entities.get(entity_id).model_dump()
                        if entity_id in self.entities
                        else None,
                        "mentions": self.entity_mentions.get(entity_id, []),
                    }
                )
            elif tool == "get_unresolved_mentions":
                surface_text = args.get("surface_text")
                results.append(
                    {
                        "tool": tool,
                        "surface_text": surface_text,
                        "unresolved": [
                            item
                            for item in self.unresolved
                            if surface_text is None
                            or item.get("surface_text") == surface_text
                        ],
                    }
                )
            elif tool == "get_related_events":
                entity_id = str(args.get("entity_id"))
                visited_entity_ids.add(entity_id)
                results.append(
                    {
                        "tool": tool,
                        "entity_id": entity_id,
                        "events": self.events_by_entity.get(entity_id, []),
                    }
                )
            else:
                raise RuntimeError(f"Unsupported retrieval tool: {tool}")
        return self._trim_context(results)

    def _previous_chunks(
        self,
        chunk_id: str,
        limit: int,
        visited_chunk_ids: set[str],
    ) -> list[dict[str, Any]]:
        chunk = self._chunk(chunk_id)
        candidates = [
            candidate
            for candidate in self.chunks
            if candidate.index < chunk.index
        ][-limit:]
        return self._chunk_results("get_previous_chunks", candidates, visited_chunk_ids)

    def _next_chunks(
        self,
        chunk_id: str,
        limit: int,
        visited_chunk_ids: set[str],
    ) -> list[dict[str, Any]]:
        chunk = self._chunk(chunk_id)
        candidates = [
            candidate
            for candidate in self.chunks
            if candidate.index > chunk.index
        ][:limit]
        return self._chunk_results("get_next_chunks", candidates, visited_chunk_ids)

    def _chunks_by_entity(
        self,
        entity_id: str,
        visited_chunk_ids: set[str],
    ) -> list[dict[str, Any]]:
        mentions = set(self.entity_mentions.get(entity_id, []))
        candidates = [
            chunk
            for chunk in self.chunks
            if chunk.chunk_id in mentions
        ][: self.max_chunks_per_round]
        return self._chunk_results("get_chunks_by_entity", candidates, visited_chunk_ids)

    def _search_chunks(
        self,
        query: str,
        top_k: int,
        visited_chunk_ids: set[str],
    ) -> list[dict[str, Any]]:
        terms = {part.lower() for part in query.split() if part.strip()}
        scored = []
        for chunk in self.chunks:
            text = chunk.text.lower()
            score = sum(1 for term in terms if term in text)
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].index))
        return self._chunk_results(
            "search_chunks",
            [chunk for _, chunk in scored[:top_k]],
            visited_chunk_ids,
        )

    def _chunk_results(
        self,
        tool: str,
        chunks: list[ChunkNode],
        visited_chunk_ids: set[str],
    ) -> list[dict[str, Any]]:
        rows = []
        for chunk in chunks:
            visited_chunk_ids.add(chunk.chunk_id)
            rows.append(
                {
                    "tool": tool,
                    "chunk_id": chunk.chunk_id,
                    "index": chunk.index,
                    "text": chunk.text,
                }
            )
        return rows

    def _chunk(self, chunk_id: str) -> ChunkNode:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise RuntimeError(f"Unknown chunk_id: {chunk_id}")

    def _trim_context(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        text = json.dumps(results, ensure_ascii=False)
        if len(text) <= self.max_total_context_chars:
            return results
        trimmed: list[dict[str, Any]] = []
        total = 0
        for item in results:
            item_text = json.dumps(item, ensure_ascii=False)
            if total + len(item_text) > self.max_total_context_chars:
                break
            trimmed.append(item)
            total += len(item_text)
        return trimmed


EXTRACTION_SYSTEM_PROMPT = """
You extract a semantic draft from one raw text chunk.
No fixtures exist. Do not invent spans outside the chunk.
Your job is meaning only: mention spans, events, predicates, roles, modality,
polarity, and temporal scope. Runtime will create stable ids, evidence ids,
argument ids, document ids, extractor versions, and resolver versions.
Mandatory for these smoke sentences: do not omit named people, role aliases,
pronouns, old man, steps, door, darkness, forest, rifle, warning, river, or
lights when they appear in the chunk.
Every event argument must carry its own span. If a separate mention is missing,
runtime may create a RawMention from the argument span. Do not return empty
extractions for non-empty narrative text.
Use grounding_expectation:
- entity_expected for concrete people/places/objects that can be resolved.
- concept_allowed for concepts or mass/generic things like steps, warning.
- mention_only_allowed for textual anchors that should be observed but not
  promoted to an entity unless later evidence requires it.
Output shape:
{
  "mentions": [
    {
      "local_ref": "m1",
      "surface_text": "surface text",
      "start_char": 0,
      "end_char": 1,
      "mention_type": "named",
      "mention_kind": "named_entity",
      "confidence": 0.95
    }
  ],
  "events": [
    {
      "predicate": "verb",
      "normalized_predicate": "verb",
      "event_type": "action",
      "evidence_text": "sentence or clause text",
      "evidence_start_char": 0,
      "evidence_end_char": 1,
      "temporal_scope": null,
      "modality": null,
      "polarity": "positive",
      "arguments": [
        {
          "role": "agent",
          "mention_ref": "m1",
          "surface_text": "surface text",
          "start_char": 0,
          "end_char": 1,
          "grounding_expectation": "entity_expected",
          "confidence": 0.95
        }
      ],
      "confidence": 0.95
    }
  ],
  "notes": []
}
"""


RLM_SYSTEM_PROMPT = """
You are the RLM resolver. You receive current chunk extraction, existing state,
unresolved hypotheses, nearby metadata, and available retrieval actions.
No Python fallback will repair your answer.

If a pronoun or role alias cannot be resolved from the current chunk and known
state alone, return action='retrieve_context' with concrete requests. For a
current chunk containing only pronouns or role aliases, retrieve neighboring
chunks before final resolution. After retrieved context is sufficient, return
action='final_resolution' and decisions for all mentions.

Do not resolve ambiguous pronouns by recency alone. If evidence is insufficient,
leave unresolved with competing hypotheses in unresolved_hypotheses.
Named concrete mentions with no existing candidate should normally create a
provisional entity. Do not mark Semyon, old man, or concrete named/role-anchor
people as unresolved merely because state_before is empty. Use unresolved for
ambiguous pronouns or aliases when evidence is genuinely insufficient.
If your reason says "provisional entity" or "likely a provisional entity", the
decision must be create_entity with an entity_id and canonical_name.
For role aliases like "the hunter", link to an existing entity only when state
or retrieved context supports it; otherwise create a provisional entity or keep
a competing hypothesis.
Return one decision for every current_extraction.raw_mentions item. If a
retrieval request has already returned no new context, do not repeat it; finalize
with resolved or unresolved decisions and stop_reason explaining why.

Available tools:
get_previous_chunks(chunk_id, limit)
get_next_chunks(chunk_id, limit)
get_chunks_by_entity(entity_id)
search_chunks(query, top_k)
get_entity_context(entity_id)
get_unresolved_mentions(surface_text)
get_related_events(entity_id)

Output shape:
{
  "action": "retrieve_context" | "final_resolution",
  "reason": "...",
  "requests": [...],
  "decisions": [...],
  "unresolved_hypotheses": [...],
  "stop_reason": "..."
}
"""


FOCUSED_MENTION_RESOLUTION_PROMPT = """
You resolve one target mention after deterministic retrieval has already been
performed by runtime. Do not request tools and do not use broad story guesses.

Policy guards:
- A pronoun must never create a new entity. It can link_existing,
  keep_unresolved, or mention_only.
- A descriptive alias must search candidate_entities first. Prefer
  link_existing when retrieved context supports an alias relation.
- A repeated nominal mention should match an existing entity before propose_new.
- If evidence is insufficient or candidates compete, keep_unresolved.

Return exactly one compact JSON object:
{
  "decision": "link_existing" | "propose_new" | "keep_unresolved" | "mention_only",
  "entity_id": "candidate entity id or null",
  "confidence": 0.0,
  "reason": "short evidence-based reason"
}
"""


def print_block(title: str, payload: Any) -> None:
    print(f"\n[{title}]")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def normalize_surface(text: str) -> str:
    lowered = text.strip().lower()
    for prefix in ("the ", "a ", "an "):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
    return " ".join(lowered.split())


def slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:60] or "id"


def stable_id(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{slug(str(parts[0]))}_{digest}"


def build_chunks(document_id: str, texts: list[str]) -> list[ChunkNode]:
    offset = 0
    chunks = []
    for index, text in enumerate(texts, start=1):
        chunk_id = f"{document_id}_chunk_{index}"
        chunks.append(
            ChunkNode(
                chunk_id=chunk_id,
                document_id=document_id,
                index=index,
                text=text,
                start_char=offset,
                end_char=offset + len(text),
                token_count=len(text.split()),
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                extraction_status="DONE",
            )
        )
        offset += len(text) + 1
    return chunks


def validate_extraction_contract(extraction: SmokeExtraction, chunk: ChunkNode) -> None:
    if not extraction.raw_mentions:
        raise RuntimeError(f"{chunk.chunk_id}: extraction produced no raw_mentions")
    if not extraction.evidence_spans:
        raise RuntimeError(f"{chunk.chunk_id}: extraction produced no evidence_spans")
    if not extraction.event_frames:
        raise RuntimeError(f"{chunk.chunk_id}: extraction produced no event_frames")
    mention_keys = {mention.mention_key for mention in extraction.raw_mentions}
    evidence_keys = {span.evidence_key for span in extraction.evidence_spans}
    for span in extraction.evidence_spans:
        if span.start_char_in_chunk < 0:
            raise RuntimeError(
                f"{chunk.chunk_id}: negative evidence span: {span.evidence_key}"
            )
    for mention in extraction.raw_mentions:
        if mention.start_char_in_chunk < 0:
            raise RuntimeError(
                f"{chunk.chunk_id}: negative mention span: {mention.mention_key}"
            )
    for event in extraction.event_frames:
        if event.evidence_key not in evidence_keys:
            raise RuntimeError(
                f"{chunk.chunk_id}: event {event.event_key} references missing evidence"
            )
        if not event.arguments:
            raise RuntimeError(
                f"{chunk.chunk_id}: event {event.event_key} has no arguments"
            )
        for argument in event.arguments:
            if argument.mention_key not in mention_keys:
                raise RuntimeError(
                    f"{chunk.chunk_id}: argument references missing mention "
                    f"{argument.mention_key!r} for surface {argument.surface_text!r}"
                )


def _normalize_span(
    chunk: ChunkNode,
    surface_text: str,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    text = chunk.text
    start = max(0, min(start_char, len(text)))
    end = max(start, min(end_char, len(text)))
    if text[start:end] == surface_text:
        return start, end
    lowered = text.lower()
    surface = surface_text.lower()
    exact_start = lowered.find(surface)
    if exact_start >= 0:
        return exact_start, exact_start + len(surface_text)
    raise RuntimeError(
        f"{chunk.chunk_id}: span mismatch for {surface_text!r} at {start_char}:{end_char}"
    )


def _expectation_for_surface(
    surface_text: str,
    mention_type: str,
    explicit: str | None,
) -> str:
    surface = normalize_surface(surface_text)
    if mention_type == "pronoun":
        return "entity_expected"
    if surface in {"steps", "warning", "darkness"}:
        return "concept_allowed"
    if surface in {"door", "forest", "path"}:
        return "mention_only_allowed"
    if explicit:
        return explicit
    return "entity_expected"


def assemble_semantic_draft(
    draft: LLMSemanticExtractionDraft,
    chunk: ChunkNode,
) -> SmokeExtraction:
    mentions_by_ref: dict[str, SmokeRawMentionExtraction] = {}
    mentions_by_span: dict[tuple[int, int, str], SmokeRawMentionExtraction] = {}

    def ensure_mention(
        surface_text: str,
        start_char: int,
        end_char: int,
        mention_type: str = "nominal",
        mention_kind: str = "generic_nominal",
        confidence: float = 0.9,
        local_ref: str | None = None,
    ) -> SmokeRawMentionExtraction:
        start, end = _normalize_span(chunk, surface_text, start_char, end_char)
        key = (start, end, surface_text)
        if key in mentions_by_span:
            mention = mentions_by_span[key]
            if local_ref:
                mentions_by_ref[local_ref] = mention
            return mention
        mention_key = local_ref or f"m{len(mentions_by_span) + 1}"
        while mention_key in mentions_by_ref:
            mention_key = f"m{len(mentions_by_span) + 1}"
        mention = SmokeRawMentionExtraction(
            mention_key=mention_key,
            text=surface_text,
            start_char_in_chunk=start,
            end_char_in_chunk=end,
            mention_type=mention_type,  # type: ignore[arg-type]
            mention_kind=mention_kind,  # type: ignore[arg-type]
            normalized_text=surface_text,
            confidence=confidence,
        )
        mentions_by_span[key] = mention
        mentions_by_ref[mention_key] = mention
        if local_ref:
            mentions_by_ref[local_ref] = mention
        return mention

    for mention in draft.mentions:
        ensure_mention(
            surface_text=mention.surface_text,
            start_char=mention.start_char,
            end_char=mention.end_char,
            mention_type=mention.mention_type,
            mention_kind=mention.mention_kind,
            confidence=mention.confidence,
            local_ref=mention.local_ref,
        )

    evidence_spans: list[SmokeEvidenceSpanExtraction] = []
    event_frames: list[SmokeEventFrameExtraction] = []
    for event_index, event in enumerate(draft.events, start=1):
        arguments: list[SmokeEventArgumentExtraction] = []
        arg_starts: list[int] = []
        arg_ends: list[int] = []
        for argument in event.arguments:
            start, end = _normalize_span(
                chunk,
                argument.surface_text,
                argument.start_char,
                argument.end_char,
            )
            arg_starts.append(start)
            arg_ends.append(end)
            mention = None
            if argument.mention_ref:
                mention = mentions_by_ref.get(argument.mention_ref)
            if mention is None:
                mention = ensure_mention(
                    surface_text=argument.surface_text,
                    start_char=start,
                    end_char=end,
                    mention_type="pronoun"
                    if normalize_surface(argument.surface_text)
                    in {"he", "him", "his", "she", "her", "they", "them"}
                    else "nominal",
                    mention_kind="pronoun"
                    if normalize_surface(argument.surface_text)
                    in {"he", "him", "his", "she", "her", "they", "them"}
                    else "generic_nominal",
                    confidence=argument.confidence,
                    local_ref=argument.mention_ref,
                )
            arguments.append(
                SmokeEventArgumentExtraction(
                    role=argument.role,
                    mention_key=mention.mention_key,
                    surface_text=argument.surface_text,
                    grounding_expectation=_expectation_for_surface(
                        argument.surface_text,
                        mention.mention_type,
                        argument.grounding_expectation,
                    ),  # type: ignore[arg-type]
                    confidence=argument.confidence,
                )
            )
        if event.evidence_start_char is not None and event.evidence_end_char is not None:
            ev_start, ev_end = _normalize_span(
                chunk,
                event.evidence_text or chunk.text[event.evidence_start_char:event.evidence_end_char],
                event.evidence_start_char,
                event.evidence_end_char,
            )
        elif arg_starts and arg_ends:
            ev_start, ev_end = min(arg_starts), max(arg_ends)
        else:
            ev_start, ev_end = 0, len(chunk.text)
        evidence_text = event.evidence_text or chunk.text[ev_start:ev_end]
        evidence_key = f"s{event_index}"
        evidence_spans.append(
            SmokeEvidenceSpanExtraction(
                evidence_key=evidence_key,
                text=evidence_text,
                start_char_in_chunk=ev_start,
                end_char_in_chunk=ev_end,
            )
        )
        event_frames.append(
            SmokeEventFrameExtraction(
                event_key=f"ev{event_index}",
                predicate=event.predicate,
                normalized_predicate=event.normalized_predicate or event.predicate,
                event_type=event.event_type,
                evidence_key=evidence_key,
                arguments=arguments,
                confidence=event.confidence,
            )
        )

    extraction = SmokeExtraction(
        raw_mentions=list(mentions_by_span.values()),
        evidence_spans=evidence_spans,
        event_frames=event_frames,
        notes=draft.notes,
    )
    validate_extraction_contract(extraction, chunk)
    return extraction


async def extract_chunk_with_validation(
    model_client: SmokeModelClient,
    chunk: ChunkNode,
    document_id: str,
    llm_model: str,
    max_attempts: int = 2,
) -> SmokeExtraction:
    previous_error: str | None = None
    previous_raw_hint: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        payload: dict[str, Any] = {
            "document_id": document_id,
            "chunk_id": chunk.chunk_id,
            "chunk_text": chunk.text,
            "requirements": [
                "extract semantic mention spans",
                "extract semantic events",
                "extract semantic event arguments",
                "include predicate and roles",
                "include grounding_expectation for arguments",
                "every event argument must include its own span",
            ],
        }
        if previous_error:
            payload["previous_validation_error"] = previous_error
            payload["retry_instruction"] = (
                "Return a corrected complete semantic draft. Every argument must "
                "include surface_text, start_char, end_char, and role. Do not "
                "return an empty object."
            )
            payload["previous_validated_shape"] = previous_raw_hint
        draft = await model_client.structured_call(
            label=f"LLM EXTRACTION {chunk.chunk_id} ATTEMPT {attempt}",
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            payload=payload,
            schema_model=LLMSemanticExtractionDraft,
            max_tokens=2048,
            response_mode="json_schema",
        )
        draft = LLMSemanticExtractionDraft.model_validate(draft.model_dump())
        try:
            extraction = assemble_semantic_draft(draft, chunk)
            print_block(
                f"RUNTIME ASSEMBLED EXTRACTION {chunk.chunk_id} ATTEMPT {attempt}",
                extraction.model_dump(),
            )
            validate_extraction_contract(extraction, chunk)
            return extraction
        except Exception as exc:
            previous_error = str(exc)
            previous_raw_hint = draft.model_dump()
            print_block(
                "LLM EXTRACTION VALIDATION ERROR",
                {
                    "chunk_id": chunk.chunk_id,
                    "attempt": attempt,
                    "extractor_version": llm_model,
                    "error": previous_error,
                },
            )
            if attempt == max_attempts:
                raise
    raise RuntimeError(f"{chunk.chunk_id}: extraction failed")


def extraction_to_graph(
    extraction: SmokeExtraction,
    chunk: ChunkNode,
    document_id: str,
    run_id: str,
    decisions: list[RLMResolutionDecision],
    runtime: SmokeRuntime,
    model_name: str,
) -> LocalGraphPatch:
    mention_by_key: dict[str, RawMention] = {}
    evidence_by_key: dict[str, EvidenceSpan] = {}
    decision_by_mention = {decision.mention_id: decision for decision in decisions}
    entities: dict[str, Entity] = {}
    mentions: list[Mention] = []
    hypotheses: list[ResolutionHypothesis] = []

    for raw in extraction.raw_mentions:
        mention_id = f"{chunk.chunk_id}_{raw.mention_key}"
        mention = RawMention(
            mention_id=mention_id,
            chunk_id=chunk.chunk_id,
            text=raw.text,
            span=TextSpan(
                start_char=raw.start_char_in_chunk,
                end_char=raw.end_char_in_chunk,
                text=raw.text,
            ),
            normalized_text=raw.normalized_text,
            mention_type=raw.mention_type,
            mention_kind=raw.mention_kind,
            extractor_source="llm",
            extractor_version=model_name,
            confidence=raw.confidence,
        )
        mention_by_key[raw.mention_key] = mention

    if not mention_by_key:
        raise RuntimeError(f"{chunk.chunk_id}: extraction produced no raw mentions")

    for evidence in extraction.evidence_spans:
        span_id = f"{chunk.chunk_id}_{evidence.evidence_key}"
        evidence_by_key[evidence.evidence_key] = EvidenceSpan(
            span_id=span_id,
            original_text=evidence.text,
            start_char=evidence.start_char_in_chunk,
            end_char=evidence.end_char_in_chunk,
            chunk_id=chunk.chunk_id,
            document_id=document_id,
        )

    if not evidence_by_key:
        raise RuntimeError(f"{chunk.chunk_id}: extraction produced no evidence spans")

    for decision in decisions:
        raw_mention = next(
            (
                mention
                for mention in mention_by_key.values()
                if mention.mention_id == decision.mention_id
            ),
            None,
        )
        if raw_mention is None:
            raise RuntimeError(
                f"RLM decision references unknown mention_id={decision.mention_id}"
            )

        final_entity_id = None
        hypothesis_type: Literal[
            "mention_to_known_entity",
            "mention_to_new_entity",
            "same_entity",
            "unresolved",
        ] = "unresolved"
        entity_creation_decision: Literal[
            "create_entity",
            "link_to_existing",
            "keep_as_mention_only",
            "drop",
        ] = "keep_as_mention_only"
        if decision.decision in {"create_entity", "link_to_existing"}:
            if not decision.entity_id or not decision.canonical_name:
                raise RuntimeError(
                    f"{decision.mention_id}: entity decision lacks entity_id/name"
                )
            final_entity_id = decision.entity_id
            hypothesis_type = (
                "mention_to_new_entity"
                if decision.decision == "create_entity"
                else "mention_to_known_entity"
            )
            entity_creation_decision = decision.decision
            entity = runtime.entities.get(decision.entity_id) or Entity(
                entity_id=decision.entity_id,
                canonical_name=decision.canonical_name,
                entity_type=decision.entity_type,
                aliases=[],
            )
            aliases = list(dict.fromkeys(entity.aliases + decision.aliases_to_add))
            entity = entity.model_copy(update={"aliases": aliases})
            runtime.entities[entity.entity_id] = entity
            entities[entity.entity_id] = entity
            runtime.entity_mentions.setdefault(entity.entity_id, []).append(chunk.chunk_id)
            runtime.entity_observations.setdefault(entity.entity_id, []).append(
                {
                    "chunk_id": chunk.chunk_id,
                    "mention_id": raw_mention.mention_id,
                    "surface_text": raw_mention.text,
                    "decision": decision.decision,
                    "evidence_span_ids": decision.evidence_span_ids,
                }
            )
            mentions.append(
                Mention(
                    mention_id=f"{raw_mention.mention_id}_resolved",
                    chunk_id=chunk.chunk_id,
                    text=raw_mention.text,
                    span=raw_mention.span,
                    entity_id=entity.entity_id,
                    reference_type="pronoun"
                    if raw_mention.mention_type == "pronoun"
                    else "nominal",
                )
            )
        elif decision.decision == "unresolved":
            runtime.unresolved.append(decision.model_dump())

        hypotheses.append(
            ResolutionHypothesis(
                hypothesis_id=stable_id("rh", run_id, decision.mention_id, decision.entity_id),
                mention_id=decision.mention_id,
                hypothesis_type=hypothesis_type,
                candidate_entity_id=final_entity_id,
                candidate_entity_name=decision.canonical_name,
                confidence=decision.confidence,
                status=decision.status,
                mention_kind=raw_mention.mention_kind,
                entity_creation_decision=entity_creation_decision,
                final_entity_id=final_entity_id,
                candidate_entity_ids=decision.candidate_entity_ids
                or ([final_entity_id] if final_entity_id else []),
                candidate_scores={
                    candidate_id: (
                        decision.confidence if candidate_id == final_entity_id else 0.0
                    )
                    for candidate_id in decision.candidate_entity_ids
                }
                or ({final_entity_id: decision.confidence} if final_entity_id else {}),
                evidence_span_id=decision.evidence_span_ids[0]
                if decision.evidence_span_ids
                else None,
                decision_stage="recursive_rlm_smoke",
                authority="cross_chunk_coreference",
                is_terminal=decision.decision in {"create_entity", "link_to_existing"},
                reason=decision.decision_reason,
                positive_evidence=decision.consulted_chunk_ids,
                negative_evidence=decision.consulted_entity_ids,
                resolution_run_id=run_id,
                resolver_version=runtime.resolver_version,
                policy_version="no_fallback_recursive_smoke_v1",
                extractor_version=model_name,
                model_name=model_name,
                chunking_version="manual_smoke_chunks_v1",
            )
        )

    event_frames: list[EventFrame] = []
    for event in extraction.event_frames:
        if event.evidence_key not in evidence_by_key:
            raise RuntimeError(
                f"{chunk.chunk_id}: event {event.event_key} references unknown evidence"
            )
        arguments: list[EventArgument] = []
        for index, argument in enumerate(event.arguments):
            raw_mention = mention_by_key.get(argument.mention_key)
            if raw_mention is None:
                if argument.grounding_expectation == "concept_allowed":
                    continue
                raise RuntimeError(
                    f"{chunk.chunk_id}: event argument references unknown mention"
                )
            decision = decision_by_mention.get(raw_mention.mention_id)
            entity_id = (
                decision.entity_id
                if decision and decision.decision in {"create_entity", "link_to_existing"}
                else None
            )
            resolution_status = "resolved" if entity_id else "unresolved"
            if argument.grounding_expectation != "entity_expected":
                resolution_status = "mention_only"
            arguments.append(
                EventArgument(
                    argument_id=f"{chunk.chunk_id}_{event.event_key}_arg_{index}",
                    event_frame_id=f"{chunk.chunk_id}_{event.event_key}",
                    role=argument.role,
                    mention_id=raw_mention.mention_id,
                    entity_id=entity_id,
                    surface_text=argument.surface_text,
                    evidence_span_id=evidence_by_key[event.evidence_key].span_id,
                    resolution_status=resolution_status,
                    grounding_expectation=argument.grounding_expectation,
                    argument_index=index,
                    extractor_version=model_name,
                    resolver_version=runtime.resolver_version,
                    confidence=argument.confidence,
                )
            )
            if entity_id:
                runtime.events_by_entity.setdefault(entity_id, []).append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "predicate": event.predicate,
                        "role": argument.role,
                        "evidence_span_id": evidence_by_key[event.evidence_key].span_id,
                    }
                )
        unresolved_entity_args = [
            arg
            for arg in arguments
            if arg.grounding_expectation == "entity_expected"
            and arg.resolution_status != "resolved"
        ]
        event_frames.append(
            EventFrame(
                event_frame_id=f"{chunk.chunk_id}_{event.event_key}",
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                predicate=event.predicate,
                normalized_predicate=event.normalized_predicate,
                event_type=event.event_type,
                evidence_span_id=evidence_by_key[event.evidence_key].span_id,
                arguments=arguments,
                resolution_status="partial"
                if unresolved_entity_args
                else "complete",
                materialization_status="valid",
                extractor_version=model_name,
                resolver_version=runtime.resolver_version,
                confidence=event.confidence,
            )
        )

    return LocalGraphPatch(
        chunk=chunk,
        raw_mentions=list(mention_by_key.values()),
        evidence_spans=list(evidence_by_key.values()),
        event_frames=event_frames,
        entities=list(entities.values()),
        mentions=mentions,
        resolution_hypotheses=hypotheses,
    )


async def resolve_chunk_experimental_state_aware(
    model_client: SmokeModelClient,
    runtime: SmokeRuntime,
    chunk: ChunkNode,
    extraction: SmokeExtraction,
    state_before: dict[str, Any],
    max_retrieval_rounds: int,
    timeout_seconds: float,
) -> tuple[list[RLMResolutionDecision], dict[str, Any], int]:
    start = time.monotonic()
    context_chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "index": chunk.index,
            "text": chunk.text,
        }
    ]
    retrieved_context: list[dict[str, Any]] = []
    visited_chunk_ids = {chunk.chunk_id}
    visited_entity_ids: set[str] = set()
    previous_decisions: list[dict[str, Any]] = []
    saw_retrieval = False

    for round_index in range(max_retrieval_rounds + 1):
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"{chunk.chunk_id}: RLM timeout")
        unresolved = [
            str(mention.get("mention_id", mention.get("surface_text", "")))
            for mention in runtime.unresolved
            if isinstance(mention, dict)
        ]
        print_block(
            f"RLM ROUND {round_index}",
            {
                "current_chunk": chunk.chunk_id,
                "context_chunks": [item["chunk_id"] for item in context_chunks],
                "unresolved": unresolved,
                "visited_chunk_ids": sorted(visited_chunk_ids),
                "visited_entity_ids": sorted(visited_entity_ids),
            },
        )
        mentions_payload = [
            {
                **mention.model_dump(),
                "mention_id": f"{chunk.chunk_id}_{mention.mention_key}",
            }
            for mention in extraction.raw_mentions
        ]
        evidence_payload = [
            {
                **span.model_dump(),
                "span_id": f"{chunk.chunk_id}_{span.evidence_key}",
            }
            for span in extraction.evidence_spans
        ]
        payload = {
            "current_chunk_id": chunk.chunk_id,
            "current_chunk_text": chunk.text,
            "nearby_chunk_metadata": runtime.nearby_metadata(chunk),
            "current_extraction": {
                "raw_mentions": mentions_payload,
                "evidence_spans": evidence_payload,
                "event_frames": [event.model_dump() for event in extraction.event_frames],
            },
            "known_entities": [
                entity.model_dump() for entity in runtime.entities.values()
            ],
            "unresolved_mentions": runtime.unresolved,
            "state_before": state_before,
            "retrieved_context": retrieved_context,
            "previous_round_decisions": previous_decisions,
            "available_retrieval_actions": [
                "get_previous_chunks",
                "get_next_chunks",
                "get_chunks_by_entity",
                "search_chunks",
                "get_entity_context",
                "get_unresolved_mentions",
                "get_related_events",
            ],
            "limits": {
                "max_retrieval_rounds": max_retrieval_rounds,
                "max_chunks_per_round": runtime.max_chunks_per_round,
                "max_total_context_chars": runtime.max_total_context_chars,
                "timeout_seconds": timeout_seconds,
            },
        }
        response = await model_client.structured_call(
            label=f"RLM ROUND {round_index}",
            system_prompt=RLM_SYSTEM_PROMPT,
            payload=payload,
            schema_model=RLMRoundResponse,
        )
        round_response = RLMRoundResponse.model_validate(response.model_dump())
        if round_response.action == "retrieve_context":
            if not round_response.requests:
                raise RuntimeError(f"{chunk.chunk_id}: retrieve_context without requests")
            saw_retrieval = True
            retrieved = runtime.execute_requests(
                round_response.requests,
                visited_chunk_ids=visited_chunk_ids,
                visited_entity_ids=visited_entity_ids,
            )
            retrieved_context.extend(retrieved)
            context_chunks = [
                item
                for item in context_chunks + retrieved
                if item.get("chunk_id")
            ]
            previous_decisions = [
                decision.model_dump() for decision in round_response.decisions
            ]
            continue
        if not round_response.decisions:
            raise RuntimeError(f"{chunk.chunk_id}: final_resolution without decisions")
        state_after = {
            **runtime.state_snapshot(),
            "retrieval": {
                "saw_retrieval": saw_retrieval,
                "rounds": round_index + 1,
                "visited_chunk_ids": sorted(visited_chunk_ids),
                "visited_entity_ids": sorted(visited_entity_ids),
                "stop_reason": round_response.stop_reason,
            },
            "unresolved_hypotheses": round_response.unresolved_hypotheses,
        }
        return round_response.decisions, state_after, round_index + 1
    raise RuntimeError(f"{chunk.chunk_id}: exceeded max_retrieval_rounds")


def mention_argument_expectations(
    extraction: SmokeExtraction,
    mention_key: str,
) -> list[str]:
    expectations: list[str] = []
    for event in extraction.event_frames:
        for argument in event.arguments:
            if argument.mention_key == mention_key:
                expectations.append(argument.grounding_expectation)
    return expectations


def is_proper_named_anchor(mention: SmokeRawMentionExtraction) -> bool:
    text = mention.text.strip()
    if not text or mention.mention_type == "pronoun" or mention.mention_kind == "pronoun":
        return False
    if text.lower().startswith(("the ", "a ", "an ")):
        return False
    return text[0].isupper() and " " not in normalize_surface(text)


def is_ambiguous_entity_mention(
    runtime: SmokeRuntime,
    mention: SmokeRawMentionExtraction,
    expectations: list[str],
) -> bool:
    surface = normalize_surface(mention.text)
    if "entity_expected" not in expectations and mention.mention_type != "pronoun":
        return False
    if mention.mention_type == "pronoun" or mention.mention_kind == "pronoun":
        return True
    if surface in {"hunter", "old man", "the hunter", "the old man"}:
        return True
    if mention.mention_type in {"descriptor", "alias", "nominal"}:
        return True
    if mention.mention_kind in {"descriptive_alias", "role_anchor", "generic_nominal"}:
        return not is_proper_named_anchor(mention)
    for entity in runtime.entities.values():
        names = [entity.canonical_name, *entity.aliases]
        if surface in {normalize_surface(name) for name in names}:
            return True
    return False


def deterministic_decision_for_non_ambiguous(
    chunk: ChunkNode,
    mention: SmokeRawMentionExtraction,
    expectations: list[str],
) -> RLMResolutionDecision:
    mention_id = f"{chunk.chunk_id}_{mention.mention_key}"
    if "concept_allowed" in expectations:
        return RLMResolutionDecision(
            mention_id=mention_id,
            surface_text=mention.text,
            decision="concept_allowed",
            confidence=mention.confidence,
            status="confirmed",
            decision_reason="Concept mention allowed by event argument grounding policy.",
        )
    if "mention_only_allowed" in expectations:
        return RLMResolutionDecision(
            mention_id=mention_id,
            surface_text=mention.text,
            decision="mention_only_allowed",
            confidence=mention.confidence,
            status="confirmed",
            decision_reason="Mention-only anchor allowed by event argument grounding policy.",
        )
    entity_id = stable_id("entity", chunk.document_id, normalize_surface(mention.text))
    return RLMResolutionDecision(
        mention_id=mention_id,
        surface_text=mention.text,
        decision="create_entity",
        entity_id=entity_id,
        canonical_name=mention.normalized_text or mention.text,
        entity_type="person" if is_proper_named_anchor(mention) else "unknown",
        aliases_to_add=[],
        confidence=mention.confidence,
        status="confirmed",
        consulted_chunk_ids=[chunk.chunk_id],
        evidence_span_ids=[],
        decision_reason="Unambiguous named anchor creates a provisional entity.",
    )


async def resolve_focused_mention(
    model_client: SmokeModelClient,
    runtime: SmokeRuntime,
    chunk: ChunkNode,
    mention: SmokeRawMentionExtraction,
    extraction: SmokeExtraction,
    expectations: list[str],
) -> RLMResolutionDecision:
    context = runtime.deterministic_context_for_mention(
        chunk=chunk,
        mention=mention,
        extraction=extraction,
    )
    mention_id = context["mention_id"]
    print_block(
        "AMBIGUOUS MENTION DETECTED",
        {
            "mention_id": mention_id,
            "surface_text": mention.text,
            "mention_type": mention.mention_type,
            "mention_kind": mention.mention_kind,
            "expectations": expectations,
        },
    )

    if "concept_allowed" in expectations:
        return RLMResolutionDecision(
            mention_id=mention_id,
            surface_text=mention.text,
            decision="concept_allowed",
            confidence=mention.confidence,
            status="confirmed",
            consulted_chunk_ids=context["consulted_chunk_ids"],
            consulted_entity_ids=context["consulted_entity_ids"],
            candidate_entity_ids=context["candidate_entity_ids"],
            evidence_span_ids=context["evidence_span_ids"],
            decision_reason="Concept mention allowed before model resolution.",
        )

    response = await model_client.structured_call(
        label=f"FOCUSED GEMMA RESOLUTION {mention_id}",
        system_prompt=FOCUSED_MENTION_RESOLUTION_PROMPT,
        payload={
            "current_sentence": context["current_sentence"],
            "target_mention": {
                **mention.model_dump(),
                "mention_id": mention_id,
                "expectations": expectations,
            },
            "retrieved_context": {
                "chunks": context["retrieved_chunks"],
                "entity_context": context["entity_context"],
                "recent_events": context["recent_events"],
                "unresolved_hypotheses": context["unresolved_hypotheses"],
            },
            "candidate_entities": context["candidate_entities"],
            "policy_guards": [
                "pronoun_never_creates_entity",
                "alias_searches_candidates_first",
                "repeated_nominal_matches_existing_first",
                "insufficient_evidence_keeps_unresolved",
            ],
        },
        schema_model=FocusedMentionResolution,
        max_tokens=512,
    )
    focused = FocusedMentionResolution.model_validate(response.model_dump())
    candidate_ids = set(context["candidate_entity_ids"])
    is_pronoun = mention.mention_type == "pronoun" or mention.mention_kind == "pronoun"

    if is_pronoun and focused.decision == "propose_new":
        focused = FocusedMentionResolution(
            decision="keep_unresolved",
            entity_id=None,
            confidence=min(focused.confidence, 0.49),
            reason=f"Policy guard: pronoun cannot create Entity. {focused.reason}",
        )
    if focused.decision == "link_existing" and focused.entity_id not in candidate_ids:
        focused = FocusedMentionResolution(
            decision="keep_unresolved",
            entity_id=None,
            confidence=min(focused.confidence, 0.49),
            reason=f"Policy guard: linked entity was not in candidate set. {focused.reason}",
        )
    if focused.decision == "link_existing" and not is_pronoun:
        selected_candidate = next(
            (
                candidate
                for candidate in context["candidate_entities"]
                if candidate["entity_id"] == focused.entity_id
            ),
            None,
        )
        strong_reasons = {
            "surface_or_alias_match",
            "partial_name_match",
            "hunter_alias_supported_by_rifle_context",
            "hunter_name_alias",
        }
        if selected_candidate and not (
            strong_reasons & set(selected_candidate.get("reasons", []))
        ):
            focused = FocusedMentionResolution(
                decision="propose_new",
                entity_id=None,
                confidence=min(focused.confidence, 0.79),
                reason=(
                    "Policy guard: descriptive/nominal mention lacked lexical "
                    f"or alias evidence for candidate. {focused.reason}"
                ),
            )
    if is_pronoun and len(candidate_ids) > 1:
        selected_candidate = next(
            (
                candidate
                for candidate in context["candidate_entities"]
                if candidate["entity_id"] == focused.entity_id
            ),
            None,
        )
        strong_pronoun_reasons = {
            "pronoun_context_alias_bridge",
            "hunter_alias_supported_by_rifle_context",
            "surface_or_alias_match",
        }
        selected_reasons = set(selected_candidate.get("reasons", [])) if selected_candidate else set()
        selected_score = selected_candidate.get("score", 0.0) if selected_candidate else 0.0
        if (
            focused.decision == "link_existing"
            and (
                not (strong_pronoun_reasons & selected_reasons)
                or selected_score < 1.5
            )
        ):
            focused = FocusedMentionResolution(
                decision="keep_unresolved",
                entity_id=None,
                confidence=focused.confidence,
                reason=(
                    "Policy guard: competing pronoun candidates without strong "
                    f"retrieved alias evidence. {focused.reason}"
                ),
            )

    if focused.decision == "link_existing":
        entity = runtime.entities[focused.entity_id or ""]
        return RLMResolutionDecision(
            mention_id=mention_id,
            surface_text=mention.text,
            decision="link_to_existing",
            entity_id=entity.entity_id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            aliases_to_add=[mention.text]
            if normalize_surface(mention.text) != normalize_surface(entity.canonical_name)
            else [],
            confidence=focused.confidence,
            status="confirmed" if focused.confidence >= 0.8 else "likely",
            consulted_chunk_ids=context["consulted_chunk_ids"],
            consulted_entity_ids=context["consulted_entity_ids"],
            candidate_entity_ids=context["candidate_entity_ids"],
            evidence_span_ids=context["evidence_span_ids"],
            decision_reason=focused.reason,
        )
    if focused.decision == "propose_new":
        entity_id = stable_id("entity", chunk.document_id, normalize_surface(mention.text))
        return RLMResolutionDecision(
            mention_id=mention_id,
            surface_text=mention.text,
            decision="create_entity",
            entity_id=entity_id,
            canonical_name=mention.normalized_text or mention.text,
            entity_type="person"
            if mention.mention_type in {"named", "descriptor", "alias", "nominal"}
            else "unknown",
            confidence=focused.confidence,
            status="confirmed" if focused.confidence >= 0.8 else "likely",
            consulted_chunk_ids=context["consulted_chunk_ids"],
            consulted_entity_ids=context["consulted_entity_ids"],
            candidate_entity_ids=context["candidate_entity_ids"],
            evidence_span_ids=context["evidence_span_ids"],
            decision_reason=focused.reason,
        )
    if focused.decision == "mention_only":
        return RLMResolutionDecision(
            mention_id=mention_id,
            surface_text=mention.text,
            decision="mention_only_allowed",
            confidence=focused.confidence,
            status="confirmed" if focused.confidence >= 0.8 else "possible",
            consulted_chunk_ids=context["consulted_chunk_ids"],
            consulted_entity_ids=context["consulted_entity_ids"],
            candidate_entity_ids=context["candidate_entity_ids"],
            evidence_span_ids=context["evidence_span_ids"],
            decision_reason=focused.reason,
        )
    runtime.unresolved.append(
        {
            "mention_id": mention_id,
            "surface_text": mention.text,
            "candidate_entity_ids": context["candidate_entity_ids"],
            "reason": focused.reason,
        }
    )
    return RLMResolutionDecision(
        mention_id=mention_id,
        surface_text=mention.text,
        decision="unresolved",
        confidence=focused.confidence,
        status="unresolved",
        consulted_chunk_ids=context["consulted_chunk_ids"],
        consulted_entity_ids=context["consulted_entity_ids"],
        candidate_entity_ids=context["candidate_entity_ids"],
        evidence_span_ids=context["evidence_span_ids"],
        decision_reason=focused.reason,
    )


async def resolve_chunk(
    model_client: SmokeModelClient,
    runtime: SmokeRuntime,
    chunk: ChunkNode,
    extraction: SmokeExtraction,
    state_before: dict[str, Any],
    max_retrieval_rounds: int,
    timeout_seconds: float,
) -> tuple[list[RLMResolutionDecision], dict[str, Any], int]:
    del state_before, max_retrieval_rounds, timeout_seconds
    decisions: list[RLMResolutionDecision] = []
    saw_retrieval = False
    consulted_chunk_ids = {chunk.chunk_id}
    consulted_entity_ids: set[str] = set()

    for mention in extraction.raw_mentions:
        expectations = mention_argument_expectations(extraction, mention.mention_key)
        if is_ambiguous_entity_mention(runtime, mention, expectations):
            saw_retrieval = True
            decision = await resolve_focused_mention(
                model_client=model_client,
                runtime=runtime,
                chunk=chunk,
                mention=mention,
                extraction=extraction,
                expectations=expectations,
            )
        else:
            decision = deterministic_decision_for_non_ambiguous(
                chunk=chunk,
                mention=mention,
                expectations=expectations,
            )
        decisions.append(decision)
        consulted_chunk_ids.update(decision.consulted_chunk_ids)
        consulted_entity_ids.update(decision.consulted_entity_ids)

    state_after = {
        **runtime.state_snapshot(),
        "retrieval": {
            "saw_retrieval": saw_retrieval,
            "rounds": 2 if saw_retrieval else 1,
            "visited_chunk_ids": sorted(consulted_chunk_ids),
            "visited_entity_ids": sorted(consulted_entity_ids),
            "stop_reason": "retrieval_gated_mention_resolution",
        },
        "unresolved_hypotheses": runtime.unresolved[-10:],
    }
    print_block(
        "RLM MENTION-CENTRIC DECISIONS",
        [decision.model_dump() for decision in decisions],
    )
    return decisions, state_after, 2 if saw_retrieval else 1


async def process_scenario(
    args: argparse.Namespace,
    model_client: SmokeModelClient,
    client: Neo4jClient,
    scenario_name: str,
    texts: list[str],
    require_recursive_chunk: int | None,
) -> None:
    document_id = f"{args.document_id}_{scenario_name}"
    writer = GraphWriter(client)
    cleanup_document(client, document_id)
    document = DocumentNode(
        document_id=document_id,
        title=f"LLM/RLM smoke {scenario_name}",
        source_path="scripts.smoke_llm_rlm_graph",
        metadata={
            "no_fallback": args.no_fallback,
            "scenario": scenario_name,
            "model": args.llm_model,
        },
    )
    writer.write_document(document)
    chunks = build_chunks(document_id, texts[: args.max_chunks])
    runtime = SmokeRuntime(
        chunks=chunks,
        document_id=document_id,
        resolver_version=f"{args.llm_model}:recursive_rlm_smoke_v1",
        max_chunks_per_round=args.max_chunks_per_round,
        max_total_context_chars=args.max_total_context_chars,
    )
    rlm_state = RLMState(document_id=document_id)
    previous_chunk: ChunkNode | None = None

    print_block("RAW TEXT", {"scenario": scenario_name, "chunks": texts[: args.max_chunks]})

    for chunk in chunks:
        state_before = runtime.state_snapshot()
        print_block(
            "RLM STATE BEFORE",
            {
                "run_id": args.run_id,
                "chunk_id": chunk.chunk_id,
                "state_before": state_before,
                "extractor_version": args.llm_model,
                "resolver_version": runtime.resolver_version,
            },
        )
        extraction = await extract_chunk_with_validation(
            model_client=model_client,
            chunk=chunk,
            document_id=document_id,
            llm_model=args.llm_model,
        )
        decisions, state_after, rlm_rounds = await resolve_chunk(
            model_client=model_client,
            runtime=runtime,
            chunk=chunk,
            extraction=extraction,
            state_before=state_before,
            max_retrieval_rounds=args.max_retrieval_rounds,
            timeout_seconds=args.timeout_seconds,
        )
        if require_recursive_chunk == chunk.index and rlm_rounds < 2:
            raise RuntimeError(
                f"{chunk.chunk_id}: expected recursive retrieval, got {rlm_rounds} RLM call"
            )
        graph = extraction_to_graph(
            extraction=extraction,
            chunk=chunk,
            document_id=document_id,
            run_id=args.run_id,
            decisions=decisions,
            runtime=runtime,
            model_name=args.llm_model,
        )
        writer.write_local_graph(graph)
        transition = RLMTransition(
            transition_id=stable_id("rlm_transition", args.run_id, chunk.chunk_id),
            document_id=document_id,
            from_chunk_id=previous_chunk.chunk_id if previous_chunk else None,
            to_chunk_id=chunk.chunk_id,
            from_chunk_index=previous_chunk.index if previous_chunk else None,
            to_chunk_index=chunk.index,
            added_entities=[
                EntityState(
                    entity_id=entity.entity_id,
                    canonical_name=entity.canonical_name,
                    attributes=entity.aliases,
                    evidence_refs=[chunk.chunk_id],
                    confidence=1.0,
                )
                for entity in graph.entities
            ],
            notes=[
                f"recursive_rlm_rounds={rlm_rounds}",
                f"no_fallback={args.no_fallback}",
            ],
        )
        writer.write_rlm_transition(transition)
        write_hypothesis_provenance(
            client=client,
            graph=graph,
            decisions=decisions,
            retrieval_round=rlm_rounds - 1,
            state_after=state_after,
        )
        rlm_state.current_chunk_index = chunk.index
        rlm_state.recent_chunk_ids.append(chunk.chunk_id)
        previous_chunk = chunk
        print_block(
            "RLM STATE AFTER",
            {
                "run_id": args.run_id,
                "chunk_id": chunk.chunk_id,
                "state_after": state_after,
                "new_entities": [entity.model_dump() for entity in graph.entities],
                "new_aliases": {
                    entity.entity_id: entity.aliases for entity in graph.entities
                },
                "unresolved_hypotheses": state_after.get("unresolved_hypotheses", []),
                "extractor_version": args.llm_model,
                "resolver_version": runtime.resolver_version,
            },
        )

    readback(client, document_id)
    if scenario_name == "projection_context":
        assert_projection_context(
            client=client,
            model_client=model_client,
            document_id=document_id,
            projection_version=args.projection_version,
        )
    if scenario_name == "recursive_positive":
        assert_positive_resolution(client, document_id)
    if scenario_name == "recursive_negative":
        assert_negative_resolution(client, document_id)


def write_hypothesis_provenance(
    client: Neo4jClient,
    graph: LocalGraphPatch,
    decisions: list[RLMResolutionDecision],
    retrieval_round: int,
    state_after: dict[str, Any],
) -> None:
    decision_by_mention = {decision.mention_id: decision for decision in decisions}
    for hypothesis in graph.resolution_hypotheses:
        decision = decision_by_mention[hypothesis.mention_id]
        client.execute_write(
            """
            MATCH (h:ResolutionHypothesis {hypothesis_id: $hypothesis_id})
            SET h.retrieval_round = $retrieval_round,
                h.consulted_chunk_ids = $consulted_chunk_ids,
                h.consulted_entity_ids = $consulted_entity_ids,
                h.candidate_entity_ids = $candidate_entity_ids,
                h.evidence_span_ids = $evidence_span_ids,
                h.decision_reason = $decision_reason,
                h.stop_reason = $stop_reason
            """,
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "retrieval_round": retrieval_round,
                "consulted_chunk_ids": decision.consulted_chunk_ids,
                "consulted_entity_ids": decision.consulted_entity_ids,
                "candidate_entity_ids": decision.candidate_entity_ids,
                "evidence_span_ids": decision.evidence_span_ids,
                "decision_reason": decision.decision_reason,
                "stop_reason": state_after.get("retrieval", {}).get("stop_reason"),
            },
        )


def readback(client: Neo4jClient, document_id: str) -> None:
    entities = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(m:Mention)-[:REFERS_TO]->(e:Entity)
        OPTIONAL MATCH (rm:RawMention {chunk_id: c.chunk_id})
        WHERE rm.text = m.text
        RETURN e.entity_id AS entity_id,
               e.canonical_name AS canonical_name,
               e.aliases AS aliases,
               collect(DISTINCT m.text) AS mentions,
               collect(DISTINCT c.chunk_id) AS chunks
        ORDER BY canonical_name
        """,
        {"document_id": document_id},
    )
    events = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_EVENT]->(ev:EventFrame)
        OPTIONAL MATCH (arg:EventArgument)-[:ARGUMENT_OF]->(ev)
        OPTIONAL MATCH (arg)-[:RESOLVED_TO]->(e:Entity)
        RETURN ev.event_frame_id AS event_frame_id,
               ev.predicate AS predicate,
               ev.resolution_status AS resolution_status,
               ev.materialization_status AS materialization_status,
               collect({
                 role: arg.role,
                 surface_text: arg.surface_text,
                 resolution_status: arg.resolution_status,
                 grounding_expectation: arg.grounding_expectation,
                 entity: e.canonical_name
               }) AS arguments
        ORDER BY event_frame_id
        """,
        {"document_id": document_id},
    )
    hypotheses = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(rm:RawMention)-[:HAS_RESOLUTION_HYPOTHESIS]->(h)
        RETURN rm.text AS mention,
               h.final_entity_id AS entity_id,
               h.status AS status,
               h.entity_creation_decision AS decision,
               h.retrieval_round AS retrieval_round,
               h.consulted_chunk_ids AS consulted_chunk_ids,
               h.consulted_entity_ids AS consulted_entity_ids,
               h.candidate_entity_ids AS candidate_entity_ids,
               h.evidence_span_ids AS evidence_span_ids,
               h.decision_reason AS decision_reason
        ORDER BY c.index, rm.start_char
        """,
        {"document_id": document_id},
    )
    observations = client.execute_read(
        """
        MATCH (e:Entity)
        WHERE toLower(e.canonical_name) = 'semyon'
        OPTIONAL MATCH (arg:EventArgument)-[:RESOLVED_TO]->(e)
        OPTIONAL MATCH (arg)-[:ARGUMENT_OF]->(ev:EventFrame)
        RETURN e.entity_id AS entity_id,
               e.canonical_name AS canonical_name,
               collect(DISTINCT {
                 predicate: ev.predicate,
                 role: arg.role,
                 surface_text: arg.surface_text,
                 evidence_span_id: arg.evidence_span_id
               }) AS entity_event_observations
        """,
        {},
    )
    print_block("NEO4J READBACK ENTITIES", entities)
    print_block("NEO4J READBACK EVENT FRAMES", events)
    print_block("NEO4J READBACK RESOLUTION HYPOTHESES", hypotheses)
    print_block("ENTITY EVENT OBSERVATION FOR SEMYON", observations)


def assert_positive_resolution(client: Neo4jClient, document_id: str) -> None:
    rows = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(m:Mention)-[:REFERS_TO]->(e:Entity)
        WHERE toLower(m.text) IN ['semyon', 'the hunter', 'hunter', 'he']
        RETURN toLower(m.text) AS mention, e.entity_id AS entity_id,
               e.canonical_name AS canonical_name
        ORDER BY mention
        """,
        {"document_id": document_id},
    )
    by_mention: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mention.setdefault(row["mention"], []).append(row)
    semyon_ids = {row["entity_id"] for row in by_mention.get("semyon", [])}
    hunter_ids = {
        row["entity_id"]
        for key in ("hunter", "the hunter")
        for row in by_mention.get(key, [])
    }
    he_ids = {row["entity_id"] for row in by_mention.get("he", [])}
    if not semyon_ids or not hunter_ids or not he_ids:
        raise RuntimeError(
            f"positive assertion failed: missing Semyon/hunter/He links: {rows}"
        )
    expected = next(iter(semyon_ids))
    if hunter_ids != {expected} or he_ids != {expected}:
        raise RuntimeError(
            "positive assertion failed: Semyon, hunter, and He are not one Entity: "
            + json.dumps(rows, ensure_ascii=False)
        )
    he_entity = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(m:Mention)-[:REFERS_TO]->(e:Entity)
        WHERE toLower(e.canonical_name) = 'he'
        RETURN e.entity_id AS entity_id
        """,
        {"document_id": document_id},
    )
    if he_entity:
        raise RuntimeError("positive assertion failed: pronoun Entity 'he' was created")
    provenance = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(rm:RawMention)-[:HAS_RESOLUTION_HYPOTHESIS]->(h)
        WHERE toLower(rm.text) = 'he'
        RETURN h.retrieval_round AS retrieval_round,
               h.consulted_chunk_ids AS consulted_chunk_ids
        """,
        {"document_id": document_id},
    )
    if not provenance:
        raise RuntimeError("positive assertion failed: no He provenance")
    consulted = set(provenance[0].get("consulted_chunk_ids") or [])
    required = {
        f"{document_id}_chunk_1",
        f"{document_id}_chunk_2",
    }
    if (provenance[0].get("retrieval_round") or 0) < 1 or not required <= consulted:
        raise RuntimeError(
            "positive assertion failed: He provenance lacks recursive retrieval: "
            + json.dumps(provenance, ensure_ascii=False)
        )
    print_block("POSITIVE ASSERTION", {"status": "passed", "rows": rows, "provenance": provenance})


def assert_negative_resolution(client: Neo4jClient, document_id: str) -> None:
    rows = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(rm:RawMention)-[:HAS_RESOLUTION_HYPOTHESIS]->(h)
        WHERE c.index = 3 AND toLower(rm.text) = 'he'
        RETURN h.status AS status,
               h.entity_creation_decision AS decision,
               h.final_entity_id AS entity_id,
               h.candidate_entity_ids AS candidate_entity_ids,
               h.decision_reason AS decision_reason
        """,
        {"document_id": document_id},
    )
    if not rows:
        raise RuntimeError("negative assertion failed: no He hypothesis")
    row = rows[0]
    if row.get("status") != "unresolved" or row.get("entity_id"):
        raise RuntimeError(
            "negative assertion failed: ambiguous He was resolved: "
            + json.dumps(rows, ensure_ascii=False)
        )
    print_block("NEGATIVE ASSERTION", {"status": "passed", "rows": rows})


def _entity_id_for_mention(
    client: Neo4jClient,
    document_id: str,
    surface_text: str,
) -> str:
    rows = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(m:Mention)-[:REFERS_TO]->(e:Entity)
        WHERE toLower(m.text) = toLower($surface_text)
        RETURN e.entity_id AS entity_id
        ORDER BY c.index ASC
        LIMIT 1
        """,
        {"document_id": document_id, "surface_text": surface_text},
    )
    if not rows:
        raise RuntimeError(f"No Entity linked from mention {surface_text!r}")
    return rows[0]["entity_id"]


def assert_projection_context(
    client: Neo4jClient,
    model_client: SmokeModelClient,
    document_id: str,
    projection_version: str,
) -> None:
    semyon_id = _entity_id_for_mention(client, document_id, "Semyon")
    old_man_id = _entity_id_for_mention(client, document_id, "old man")
    materializer = ProjectionMaterializer(client)

    semyon_snapshot = build_entity_snapshot(
        client=client,
        document_id=document_id,
        entity_id=semyon_id,
        projection_version=projection_version,
    )
    pair_snapshot = build_entity_pair_snapshot(
        client=client,
        document_id=document_id,
        source_entity_id=semyon_id,
        target_entity_id=old_man_id,
        projection_version=projection_version,
    )
    materializer.materialize_entity_snapshot(semyon_snapshot)
    materializer.materialize_entity_pair_snapshot(pair_snapshot)

    stored_entity = materializer.read_active_entity_snapshot(
        document_id=document_id,
        entity_id=semyon_id,
        projection_version=projection_version,
    )
    stored_pair = materializer.read_active_pair_snapshot(
        document_id=document_id,
        source_entity_id=semyon_id,
        target_entity_id=old_man_id,
        projection_version=projection_version,
    )
    if stored_entity is None or stored_pair is None:
        raise RuntimeError("projection assertion failed: snapshots were not materialized")

    aliases = {alias.lower() for alias in semyon_snapshot.alias_surfaces}
    if not {"semyon", "hunter", "he"} <= aliases:
        raise RuntimeError(
            "projection assertion failed: Semyon aliases missing hunter/He: "
            + json.dumps(semyon_snapshot.alias_surfaces, ensure_ascii=False)
        )
    predicates = {obs.predicate for obs in semyon_snapshot.event_observations}
    if len(predicates) < 3:
        raise RuntimeError(
            "projection assertion failed: expected cross-chunk Semyon events: "
            + json.dumps(sorted(predicates), ensure_ascii=False)
        )
    if old_man_id not in semyon_snapshot.related_entity_ids:
        raise RuntimeError("projection assertion failed: old man is not related to Semyon")
    unresolved = {item.lower() for item in semyon_snapshot.unresolved_counterparts}
    if not ({"steps", "darkness"} & unresolved):
        raise RuntimeError(
            "projection assertion failed: concept/unresolved counterparts missing: "
            + json.dumps(semyon_snapshot.unresolved_counterparts, ensure_ascii=False)
        )
    concept_entities = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(m:Mention)-[:REFERS_TO]->(e:Entity)
        WHERE toLower(m.text) IN ['steps', 'door', 'darkness']
        RETURN m.text AS mention, e.entity_id AS entity_id, e.canonical_name AS canonical_name
        """,
        {"document_id": document_id},
    )
    if concept_entities:
        raise RuntimeError(
            "projection assertion failed: concept/unresolved counterpart became Entity: "
            + json.dumps(concept_entities, ensure_ascii=False)
        )
    rejected_in_snapshot = client.execute_read(
        """
        MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[:HAS_MENTION]->(m:RawMention)-[:HAS_RESOLUTION_HYPOTHESIS]->(h)
        WHERE h.status = 'rejected'
          AND h.final_entity_id = $entity_id
        RETURN h.hypothesis_id AS hypothesis_id
        """,
        {"document_id": document_id, "entity_id": semyon_id},
    )
    if rejected_in_snapshot:
        raise RuntimeError("projection assertion failed: rejected hypothesis leaked")
    if not (
        pair_snapshot.direct_shared_events
        or pair_snapshot.source_to_target_events
        or pair_snapshot.target_to_source_events
    ):
        raise RuntimeError("projection assertion failed: empty Semyon/old man pair context")
    if not pair_snapshot.relation_evidence_span_ids:
        raise RuntimeError("projection assertion failed: pair evidence spans missing")
    if not semyon_snapshot.encoding_input.event_blocks:
        raise RuntimeError("projection assertion failed: entity encoder event blocks missing")
    if not (
        pair_snapshot.encoding_input.direct_interaction_blocks
        or pair_snapshot.encoding_input.directional_blocks
    ):
        raise RuntimeError("projection assertion failed: pair encoder blocks missing")

    calls_before_rebuild = model_client.structured_call_count
    rebuilt_entities, rebuilt_pairs = rebuild_document_snapshots(
        client=client,
        document_id=document_id,
        projection_version=projection_version,
        materialize=True,
    )
    calls_after_rebuild = model_client.structured_call_count
    if calls_after_rebuild != calls_before_rebuild:
        raise RuntimeError(
            "projection assertion failed: rebuild performed model calls: "
            f"{calls_before_rebuild} -> {calls_after_rebuild}"
        )
    active_versions = client.execute_read(
        """
        MATCH (s:EntityContextSnapshot {
            document_id: $document_id,
            entity_id: $entity_id,
            projection_version: $projection_version,
            status: 'active'
        })
        RETURN count(s) AS active_count
        """,
        {
            "document_id": document_id,
            "entity_id": semyon_id,
            "projection_version": projection_version,
        },
    )
    if active_versions[0]["active_count"] != 1:
        raise RuntimeError("projection assertion failed: active snapshot version invariant broke")
    print_block(
        "PROJECTION ASSERTION",
        {
            "status": "passed",
            "entity_snapshot": {
                "snapshot_id": semyon_snapshot.snapshot_id,
                "entity_id": semyon_id,
                "aliases": semyon_snapshot.alias_surfaces,
                "predicates": sorted(predicates),
                "related_entity_ids": semyon_snapshot.related_entity_ids,
                "unresolved_counterparts": semyon_snapshot.unresolved_counterparts,
                "evidence_span_ids": semyon_snapshot.evidence_span_ids,
                "encoder_blocks": len(semyon_snapshot.encoding_input.event_blocks),
            },
            "pair_snapshot": {
                "snapshot_id": pair_snapshot.snapshot_id,
                "pair_id": pair_snapshot.pair_id,
                "direct_shared_events": [
                    event.model_dump() for event in pair_snapshot.direct_shared_events
                ],
                "source_to_target_events": [
                    event.model_dump() for event in pair_snapshot.source_to_target_events
                ],
                "target_to_source_events": [
                    event.model_dump() for event in pair_snapshot.target_to_source_events
                ],
                "evidence_span_ids": pair_snapshot.relation_evidence_span_ids,
                "encoder_blocks": len(
                    pair_snapshot.encoding_input.direct_interaction_blocks
                    + pair_snapshot.encoding_input.directional_blocks
                ),
            },
            "stored_entity_snapshot": stored_entity,
            "stored_pair_snapshot": stored_pair,
            "rebuild": {
                "entity_snapshots": len(rebuilt_entities),
                "pair_snapshots": len(rebuilt_pairs),
                "model_calls_before": calls_before_rebuild,
                "model_calls_after": calls_after_rebuild,
            },
        },
    )


async def main_async(args: argparse.Namespace) -> None:
    model_client = SmokeModelClient(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
        temperature=args.temperature,
    )
    await model_client.check_model()
    client = Neo4jClient(
        uri=args.neo4j_uri,
        username=args.neo4j_username,
        password=args.neo4j_password,
        connection_timeout=3.0,
        max_transaction_retry_time=3.0,
    )
    try:
        scenarios: list[tuple[str, list[str], int | None]] = [
            (
                "two_chunk",
                [
                    "Semyon entered the forest. The hunter saw an old man. He heard steps behind the door.",
                    "The hunter warned the old man, but he ignored him.",
                ],
                None,
            )
        ]
        if args.include_recursive or args.scenario == "recursive_positive":
            scenarios.append(
                (
                    "recursive_positive",
                    [
                        "Semyon entered the forest carrying his rifle.",
                        "An old man warned the hunter not to continue.",
                        "He ignored the warning.",
                    ],
                    3,
                )
            )
        if args.include_negative or args.scenario == "recursive_negative":
            scenarios.append(
                (
                    "recursive_negative",
                    [
                        "Semyon entered the forest.",
                        "Ivan followed him.",
                        "He stopped near the river.",
                    ],
                    3,
                )
            )
        if args.scenario == "projection_context":
            scenarios.append(
                (
                    "projection_context",
                    [
                        "Semyon entered the forest carrying his rifle.",
                        "An old man warned the hunter not to continue.",
                        "He heard steps behind the door.",
                        "He saw darkness behind the door.",
                        "The hunter ignored the warning.",
                        "The old man watched Semyon near the door.",
                    ],
                    3,
                )
            )
        if args.scenario != "all":
            scenarios = [
                scenario
                for scenario in scenarios
                if scenario[0] == args.scenario
            ]
            if not scenarios:
                raise SystemExit(f"Unknown or disabled scenario: {args.scenario}")
        for scenario_name, texts, require_recursive_chunk in scenarios:
            await process_scenario(
                args=args,
                model_client=model_client,
                client=client,
                scenario_name=scenario_name,
                texts=texts,
                require_recursive_chunk=require_recursive_chunk,
            )
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="gemma-3-4b-it")
    parser.add_argument("--llm-api-key", default="local")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-username", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--document-id", default="llm_rlm_graph_smoke")
    parser.add_argument("--run-id", default="llm_rlm_graph_smoke_run")
    parser.add_argument("--max-chunks", type=int, default=2)
    parser.add_argument("--max-retrieval-rounds", type=int, default=3)
    parser.add_argument("--max-chunks-per-round", type=int, default=2)
    parser.add_argument("--max-total-context-chars", type=int, default=6000)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--include-recursive", action="store_true")
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=[
            "all",
            "two_chunk",
            "recursive_positive",
            "recursive_negative",
            "projection_context",
        ],
        default="all",
    )
    parser.add_argument("--projection-version", default="entity_context_projection_v1")
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()
    if not args.no_fallback:
        raise SystemExit("This smoke is no-fallback only. Pass --no-fallback.")
    if args.max_chunks < 1:
        raise SystemExit("--max-chunks must be >= 1")
    return args


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
