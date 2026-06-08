from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class ModelAdapter(Protocol):
    async def structured_call(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        output_schema: type[T],
    ) -> T:
        ...


class OpenAICompatibleModelAdapter:
    """
    Adapter for OpenAI-compatible HTTP endpoints.

    This is intended for local vLLM OpenAI server, for example:
        base_url=http://127.0.0.1:8000/v1
        model=gemma-3-4b-it
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "local",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    async def structured_call(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        output_schema: type[T],
    ) -> T:
        if output_schema.__name__ == "LLMGraphExtraction":
            return await self._graph_extraction_call(
                system_prompt=system_prompt,
                user_payload=user_payload,
                output_schema=output_schema,
            )

        schema = output_schema.model_json_schema()
        user_content = json.dumps(
            user_payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n"
                        "Return only valid JSON matching the provided JSON Schema."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": schema,
                },
            },
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty structured response.")

        return output_schema.model_validate_json(content)

    async def _graph_extraction_call(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        output_schema: type[T],
    ) -> T:
        compact_result = await self._json_object_call(
            system_prompt=system_prompt,
            user_payload=user_payload,
        )

        repaired = _repair_graph_extraction_payload(
            raw=compact_result,
            chunk_text=user_payload.get("chunk", {}).get("text", ""),
        )

        return output_schema.model_validate(repaired)

    async def _json_schema_call(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        output_schema: type[T],
    ) -> T:
        schema = output_schema.model_json_schema()
        user_content = json.dumps(
            user_payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n"
                        "Return only valid JSON matching the provided JSON Schema."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": schema,
                },
            },
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty structured response.")

        return output_schema.model_validate_json(content)

    async def _json_object_call(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> dict[str, Any]:
        chunk_text = user_payload.get("chunk", {}).get("text", "")
        compact_payload = {
            "chunk_text": chunk_text,
            "previous_entities": list(
                user_payload.get("previous_rlm_state", {})
                .get("entities", {})
                .keys()
            )[:50],
            "task": (
                "Extract an observation-first graph from chunk_text. Return JSON "
                "object with keys: raw_mentions, event_frames, claims, relations, "
                "notes. raw_mentions are text spans, not entities. event_frames "
                "contain predicate plus arguments with roles subject/object/actor/"
                "target/location/instrument. Keep Russian text in Cyrillic. Do "
                "not decide whether two mentions are the same entity."
            ),
        }

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n"
                        "Return ONLY a valid JSON object. No markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        compact_payload,
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty JSON object response.")

        return json.loads(content)


def _is_empty_graph_extraction(result: BaseModel) -> bool:
    return (
        not getattr(result, "entities", [])
        and not getattr(result, "mentions", [])
        and not getattr(result, "claims", [])
        and not getattr(result, "events", [])
        and not getattr(result, "relations", [])
    )


def _graph_extraction_spans_ok(
    result: BaseModel,
    user_payload: dict[str, Any],
) -> bool:
    chunk_text = user_payload.get("chunk", {}).get("text", "")

    for mention in getattr(result, "mentions", []):
        start = mention.start_char_in_chunk
        end = mention.end_char_in_chunk
        if not (0 <= start < end <= len(chunk_text)):
            return False
        if chunk_text[start:end].strip() != mention.text.strip():
            return False

    for claim in getattr(result, "claims", []):
        start = claim.evidence_start_char_in_chunk
        end = claim.evidence_end_char_in_chunk
        if not (0 <= start < end <= len(chunk_text)):
            return False

    for event in getattr(result, "events", []):
        start = event.evidence_start_char_in_chunk
        end = event.evidence_end_char_in_chunk
        if not (0 <= start < end <= len(chunk_text)):
            return False

    for relation in getattr(result, "relations", []):
        start = relation.evidence_start_char_in_chunk
        end = relation.evidence_end_char_in_chunk
        if not (0 <= start < end <= len(chunk_text)):
            return False

    return True


def _repair_graph_extraction_payload(
    raw: dict[str, Any],
    chunk_text: str,
) -> dict[str, Any]:
    raw_mentions = _repair_raw_mentions(raw, chunk_text)
    event_frames = _repair_event_frames(raw, raw_mentions, chunk_text)
    entity_names = _extract_entity_names(raw.get("entities", []))
    mention_names = _extract_mention_names(raw.get("mentions", []))
    claim_texts = _extract_claim_texts(raw.get("claims", []))

    for name in mention_names:
        if name not in entity_names:
            entity_names.append(name)

    if not entity_names:
        entity_names = _fallback_capitalized_entities(chunk_text)

    entities = [
        {
            "canonical_name": name,
            "entity_type": _guess_entity_type(name),
            "aliases": [name],
            "description": None,
        }
        for name in entity_names
        if name
    ]

    mentions = []
    for name in entity_names:
        for match in re.finditer(re.escape(name), chunk_text):
            mentions.append(
                {
                    "text": match.group(0),
                    "start_char_in_chunk": match.start(),
                    "end_char_in_chunk": match.end(),
                    "canonical_entity_name": name,
                    "entity_type": _guess_entity_type(name),
                    "reference_type": "named",
                }
            )

    if not claim_texts and chunk_text.strip():
        claim_texts = [_first_sentence_or_text(chunk_text)]

    claims = []
    for claim_text in claim_texts[:5]:
        start = chunk_text.find(claim_text)
        if start < 0:
            start = 0
            evidence_text = _first_sentence_or_text(chunk_text)
            end = len(evidence_text)
        else:
            evidence_text = claim_text
            end = start + len(claim_text)

        subject_names = [
            name
            for name in entity_names
            if name in evidence_text or name in claim_text
        ] or entity_names[:2]

        claims.append(
            {
                "text": claim_text,
                "subject_entity_names": subject_names,
                "evidence_start_char_in_chunk": start,
                "evidence_end_char_in_chunk": end,
                "confidence": 0.55,
                "needs_verification": False,
            }
        )

    return {
        "raw_mentions": raw_mentions,
        "event_frames": event_frames,
        "entities": [],
        "mentions": [],
        "claims": claims,
        "events": [],
        "relations": _repair_relations(raw.get("relations", []), entity_names, chunk_text),
        "notes": ["Gemma json_object output repaired into LLMGraphExtraction."],
    }


def _repair_raw_mentions(raw: dict[str, Any], chunk_text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int, str]] = set()

    def append_mention(
        text: str,
        start: int,
        end: int,
        mention_type: str | None = None,
        confidence: float | None = None,
        semantic_payload: dict[str, Any] | None = None,
        extractor_source: str = "llm",
        extractor_version: str | None = "openai_compatible_json_object_v0.1",
        repaired_by: str | None = None,
        repair_notes: list[str] | None = None,
    ) -> None:
        if not text or not (0 <= start < end <= len(chunk_text)):
            return
        actual = chunk_text[start:end]
        if actual.strip() != text.strip():
            return

        key = (start, end, actual)
        if key in seen_spans:
            return
        seen_spans.add(key)

        mentions.append(
            {
                "text": actual,
                "start_char_in_chunk": start,
                "end_char_in_chunk": end,
                "mention_type": mention_type or _guess_mention_type(actual),
                "normalized_text": actual.strip().lower(),
                "extractor_source": extractor_source,
                "extractor_version": extractor_version,
                "repaired_by": repaired_by,
                "repair_notes": repair_notes or [],
                "confidence": confidence,
                "semantic_payload": semantic_payload or {},
            }
        )

    for item in raw.get("raw_mentions", []) or []:
        if isinstance(item, str):
            for match in re.finditer(re.escape(item.strip()), chunk_text):
                append_mention(item.strip(), match.start(), match.end())
            continue
        if not isinstance(item, dict):
            continue

        text = str(item.get("text") or item.get("surface") or "").strip()
        start = item.get("start_char_in_chunk")
        end = item.get("end_char_in_chunk")
        if isinstance(start, int) and isinstance(end, int):
            append_mention(
                text=text,
                start=start,
                end=end,
                mention_type=item.get("mention_type"),
                confidence=item.get("confidence"),
                semantic_payload=item.get("semantic_payload") or {},
                extractor_source=item.get("extractor_source") or "llm",
                extractor_version=item.get("extractor_version")
                or "openai_compatible_json_object_v0.1",
                repaired_by=item.get("repaired_by"),
                repair_notes=item.get("repair_notes") or [],
            )
        elif text:
            for match in re.finditer(re.escape(text), chunk_text):
                append_mention(
                    text=text,
                    start=match.start(),
                    end=match.end(),
                    mention_type=item.get("mention_type"),
                    confidence=item.get("confidence"),
                    semantic_payload=item.get("semantic_payload") or {},
                    extractor_source=item.get("extractor_source") or "repair",
                    extractor_version=item.get("extractor_version")
                    or "openai_compatible_json_object_v0.1",
                    repaired_by=item.get("repaired_by") or "adapter_postprocess",
                    repair_notes=item.get("repair_notes")
                    or ["Located mention text after LLM omitted offsets."],
                )

    legacy_names = _extract_mention_names(raw.get("mentions", []))
    legacy_names.extend(_extract_entity_names(raw.get("entities", [])))
    for name in legacy_names:
        for match in re.finditer(re.escape(name), chunk_text):
            append_mention(
                name,
                match.start(),
                match.end(),
                extractor_source="repair",
                repaired_by="adapter_postprocess",
                repair_notes=["Converted legacy entities/mentions into raw mentions."],
            )

    if not mentions:
        for name in _fallback_observation_mentions(chunk_text):
            for match in re.finditer(re.escape(name), chunk_text):
                append_mention(
                    name,
                    match.start(),
                    match.end(),
                    extractor_source="fallback",
                    repaired_by="adapter_postprocess",
                    repair_notes=["Fallback regex mention candidate."],
                )

    return mentions


def _repair_event_frames(
    raw: dict[str, Any],
    raw_mentions: list[dict[str, Any]],
    chunk_text: str,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []

    for item in raw.get("event_frames", []) or []:
        if not isinstance(item, dict):
            continue

        start = item.get("evidence_start_char_in_chunk")
        end = item.get("evidence_end_char_in_chunk")
        if not isinstance(start, int) or not isinstance(end, int):
            evidence = str(item.get("evidence") or item.get("text") or "").strip()
            start = chunk_text.find(evidence) if evidence else 0
            end = start + len(evidence) if start >= 0 and evidence else len(_first_sentence_or_text(chunk_text))

        if not (0 <= start < end <= len(chunk_text)):
            continue

        arguments = []
        for argument in item.get("arguments", []) or []:
            if not isinstance(argument, dict):
                continue
            mention_text = str(
                argument.get("mention_text")
                or argument.get("text")
                or argument.get("mention")
                or ""
            ).strip()
            arg_start = argument.get("mention_start_char_in_chunk")
            arg_end = argument.get("mention_end_char_in_chunk")
            if not isinstance(arg_start, int) or not isinstance(arg_end, int):
                match = _find_mention_row(raw_mentions, mention_text)
                if match is None:
                    continue
                arg_start = match["start_char_in_chunk"]
                arg_end = match["end_char_in_chunk"]
                mention_text = match["text"]

            arguments.append(
                {
                    "role": str(argument.get("role") or "participant"),
                    "mention_text": mention_text,
                    "mention_start_char_in_chunk": arg_start,
                    "mention_end_char_in_chunk": arg_end,
                    "confidence": argument.get("confidence"),
                }
            )

        frames.append(
            {
                "predicate": str(item.get("predicate") or item.get("verb") or "event"),
                "event_type": item.get("event_type"),
                "arguments": arguments,
                "evidence_start_char_in_chunk": start,
                "evidence_end_char_in_chunk": end,
                "confidence": item.get("confidence"),
            }
        )

    if frames:
        return frames

    relation_frames = _event_frames_from_relations(
        raw.get("relations", []),
        raw_mentions,
        chunk_text,
    )
    if relation_frames:
        return relation_frames

    return _fallback_event_frames(raw_mentions, chunk_text)


def _event_frames_from_relations(
    raw_relations: Any,
    raw_mentions: list[dict[str, Any]],
    chunk_text: str,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    if not isinstance(raw_relations, list):
        return frames

    for item in raw_relations:
        if not isinstance(item, dict):
            continue

        source = str(
            item.get("source_entity_name")
            or item.get("source")
            or item.get("subject")
            or ""
        ).strip()
        target = str(
            item.get("target_entity_name")
            or item.get("target")
            or item.get("object")
            or ""
        ).strip()
        source_row = _find_mention_row(raw_mentions, source)
        target_row = _find_mention_row(raw_mentions, target)
        if source_row is None or target_row is None:
            continue

        evidence = str(
            item.get("relation_span")
            or item.get("evidence")
            or _first_sentence_or_text(chunk_text)
        ).strip()
        start = chunk_text.find(evidence)
        if start < 0:
            start = min(
                source_row["start_char_in_chunk"],
                target_row["start_char_in_chunk"],
            )
            end = max(
                source_row["end_char_in_chunk"],
                target_row["end_char_in_chunk"],
            )
        else:
            end = start + len(evidence)

        frames.append(
            {
                "predicate": str(
                    item.get("relation_type")
                    or item.get("predicate")
                    or "related"
                ),
                "event_type": item.get("relation_type"),
                "arguments": [
                    {
                        "role": "subject",
                        "mention_text": source_row["text"],
                        "mention_start_char_in_chunk": source_row[
                            "start_char_in_chunk"
                        ],
                        "mention_end_char_in_chunk": source_row[
                            "end_char_in_chunk"
                        ],
                        "confidence": item.get("confidence"),
                    },
                    {
                        "role": "object",
                        "mention_text": target_row["text"],
                        "mention_start_char_in_chunk": target_row[
                            "start_char_in_chunk"
                        ],
                        "mention_end_char_in_chunk": target_row[
                            "end_char_in_chunk"
                        ],
                        "confidence": item.get("confidence"),
                    },
                ],
                "evidence_start_char_in_chunk": start,
                "evidence_end_char_in_chunk": end,
                "confidence": item.get("confidence"),
            }
        )

    return frames


def _fallback_event_frames(
    raw_mentions: list[dict[str, Any]],
    chunk_text: str,
) -> list[dict[str, Any]]:
    if len(raw_mentions) < 2:
        return []

    first_sentence = _first_sentence_or_text(chunk_text)
    sentence_end = len(first_sentence)
    mentions_in_sentence = [
        mention
        for mention in raw_mentions
        if mention["start_char_in_chunk"] < sentence_end
    ]
    if len(mentions_in_sentence) < 2:
        return []

    source = mentions_in_sentence[0]
    target = mentions_in_sentence[1]
    predicate = _guess_predicate_between(
        chunk_text[
            source["end_char_in_chunk"]:
            target["start_char_in_chunk"]
        ]
    )

    return [
        {
            "predicate": predicate,
            "event_type": None,
            "arguments": [
                {
                    "role": "subject",
                    "mention_text": source["text"],
                    "mention_start_char_in_chunk": source["start_char_in_chunk"],
                    "mention_end_char_in_chunk": source["end_char_in_chunk"],
                    "confidence": 0.4,
                },
                {
                    "role": "object",
                    "mention_text": target["text"],
                    "mention_start_char_in_chunk": target["start_char_in_chunk"],
                    "mention_end_char_in_chunk": target["end_char_in_chunk"],
                    "confidence": 0.4,
                },
            ],
            "evidence_start_char_in_chunk": 0,
            "evidence_end_char_in_chunk": sentence_end,
            "confidence": 0.4,
        }
    ]


def _find_mention_row(
    raw_mentions: list[dict[str, Any]],
    mention_text: str,
) -> dict[str, Any] | None:
    normalized = mention_text.strip().lower()
    for mention in raw_mentions:
        if mention["text"].strip().lower() == normalized:
            return mention
    return None


def _extract_entity_names(raw_entities: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(raw_entities, list):
        return names

    for item in raw_entities:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(
                item.get("canonical_name")
                or item.get("name")
                or item.get("text")
                or ""
            ).strip()
        else:
            continue

        if name and not _is_pronoun(name) and name not in names:
            names.append(name)

    return names


def _extract_mention_names(raw_mentions: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(raw_mentions, list):
        return names

    for item in raw_mentions:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(
                item.get("canonical_entity_name")
                or item.get("canonical_name")
                or item.get("text")
                or ""
            ).strip()
        else:
            continue

        if name and not _is_pronoun(name) and name not in names:
            names.append(name)

    return names


def _extract_claim_texts(raw_claims: Any) -> list[str]:
    claims: list[str] = []
    if not isinstance(raw_claims, list):
        return claims

    for item in raw_claims:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("claim") or "").strip()
        else:
            continue

        if text:
            claims.append(text)

    return claims


def _repair_relations(
    raw_relations: Any,
    entity_names: list[str],
    chunk_text: str,
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    if isinstance(raw_relations, list):
        for item in raw_relations:
            if not isinstance(item, dict):
                continue

            source = str(
                item.get("source_entity_name")
                or item.get("source")
                or ""
            ).strip()
            target = str(
                item.get("target_entity_name")
                or item.get("target")
                or ""
            ).strip()

            if source not in entity_names or target not in entity_names or source == target:
                continue

            evidence = str(
                item.get("relation_span")
                or item.get("evidence")
                or _first_sentence_or_text(chunk_text)
            ).strip()
            start = chunk_text.find(evidence)
            if start < 0:
                start = 0
                end = len(_first_sentence_or_text(chunk_text))
            else:
                end = start + len(evidence)

            relations.append(
                {
                    "source_entity_name": source,
                    "target_entity_name": target,
                    "relation_type": item.get("relation_type"),
                    "relation_span": evidence,
                    "evidence_start_char_in_chunk": start,
                    "evidence_end_char_in_chunk": end,
                    "confidence": float(item.get("confidence") or 0.5),
                    "direction_confidence": float(
                        item.get("direction_confidence") or 0.0
                    ),
                }
            )

    if not relations and len(entity_names) >= 2:
        evidence = _first_sentence_or_text(chunk_text)
        for source, target in zip(entity_names, entity_names[1:]):
            relations.append(
                {
                    "source_entity_name": source,
                    "target_entity_name": target,
                    "relation_type": None,
                    "relation_span": evidence,
                    "evidence_start_char_in_chunk": 0,
                    "evidence_end_char_in_chunk": len(evidence),
                    "confidence": 0.4,
                    "direction_confidence": 0.0,
                }
            )

    return relations


def _fallback_capitalized_entities(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b", text):
        name = match.group(0).strip()
        if not _is_pronoun(name) and name not in names:
            names.append(name)
    return names[:20]


def _first_sentence_or_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    match = re.search(r"^(.+?[.!?…])(?:\s|$)", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped[:500]


def _is_pronoun(value: str) -> bool:
    return value.lower() in {
        "он",
        "она",
        "они",
        "оно",
        "его",
        "ее",
        "её",
        "их",
        "ему",
        "ей",
        "им",
        "he",
        "she",
        "they",
        "it",
    }


def _guess_entity_type(name: str) -> str:
    lowered = name.lower()
    if lowered in {"лес", "леса", "край леса"}:
        return "place"
    if lowered in {"папоротник"}:
        return "object"
    return "person"


def _fallback_capitalized_entities(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b", text):
        name = match.group(0).strip()
        if not _is_pronoun(name) and name not in names:
            names.append(name)
    return names[:20]


def _fallback_observation_mentions(text: str) -> list[str]:
    values: list[str] = []
    patterns = [
        r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b",
        r"\b(?:я|мы|он|она|они|оно|мне|меня|его|её|ее|их|ему|ей|им)\b",
        (
            r"\b(?:старик|старика|старику|стариком|охотник|охотника|"
            r"мужик|мужика|девушка|девушку|незнакомец|незнакомца)\b"
        ),
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(0).strip()
            if value and value not in values:
                values.append(value)

    return values[:30]


def _first_sentence_or_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    match = re.search(r"^(.+?[.!?…])(?:\s|$)", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped[:500]


def _is_pronoun(value: str) -> bool:
    return value.lower() in {
        "я",
        "мы",
        "он",
        "она",
        "они",
        "оно",
        "его",
        "ее",
        "её",
        "их",
        "ему",
        "ей",
        "им",
        "мне",
        "меня",
        "he",
        "she",
        "they",
        "it",
    }


def _guess_entity_type(name: str) -> str:
    lowered = name.lower()
    if lowered in {"лес", "леса", "край леса", "марушкино", "палата", "больница"}:
        return "place"
    if lowered in {"папоротник", "ружье", "ружьё", "трубка", "яд"}:
        return "object"
    return "person"


def _guess_mention_type(value: str) -> str:
    lowered = value.strip().lower()
    if _is_pronoun(value):
        return "pronoun"
    if lowered in {
        "старик",
        "старика",
        "старику",
        "стариком",
        "охотник",
        "охотника",
        "мужик",
        "мужика",
        "девушка",
        "девушку",
        "незнакомец",
        "незнакомца",
    }:
        return "descriptor"
    if lowered in {"лес", "леса", "марушкино", "палата", "больница"}:
        return "location"
    if lowered in {"ружье", "ружьё", "трубка", "яд", "папоротник"}:
        return "object"
    if value[:1].isupper():
        return "named"
    return "nominal"


def _guess_predicate_between(text: str) -> str:
    words = re.findall(r"[А-Яа-яЁёA-Za-z-]+", text)
    return words[0].lower() if words else "related"
