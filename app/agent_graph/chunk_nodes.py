# app/agent_graph/chunk_nodes.py

from __future__ import annotations

import hashlib

from app.agent_graph.chunk_schemas import ExtractedRawMention, LLMGraphExtraction
from app.agent_graph.chunk_state import ChunkGraphState
from app.core.graph_models import (
    ClaimNode,
    EntityNode,
    EventArgument,
    EventFrame,
    EventNode,
    EvidenceSpan,
    LocalGraphPatch,
    MentionNode,
    RawMention,
    RawRelation,
    RelationEdge,
    RelationCandidate,
    ResolutionHypothesis,
    ResolvedEntityRef,
    TerminalResolution,
    TextSpan,
)
from app.llm.prompts import GRAPH_EXTRACTION_SYSTEM_PROMPT


CORE_EVENT_ROLES = {
    "actor",
    "agent",
    "patient",
    "subject",
    "object",
    "target",
    "experiencer",
    "stimulus",
    "speaker",
    "listener",
}

ANCHOR_NOISE_TERMS = {
    "вокруг",
    "передо мной",
    "настоящим текстом",
    "публикации",
    "чуть позже",
    "абсолютно",
    "более",
    "больше",
    "ведь",
    "видимо",
    "вместе",
    "вдоль",
}

ANCHOR_NOISE_PREFIXES = {
    "в ",
    "во ",
    "на ",
    "перед ",
    "после ",
    "чуть ",
    "более ",
    "вокруг ",
}

DISCOURSE_MARKERS = {
    "итак",
    "наконец",
    "однако",
    "например",
    "следовательно",
    "настоящим",
}

EVENT_ROLE_MAP = {
    "subject": "agent",
    "actor": "agent",
    "agent": "agent",
    "object": "patient",
    "target": "patient",
    "patient": "patient",
    "listener": "addressee",
    "addressee": "addressee",
    "speaker": "speaker",
    "recipient": "recipient",
    "experiencer": "experiencer",
    "stimulus": "stimulus",
    "instrument": "instrument",
    "location": "location",
    "source": "source",
    "destination": "destination",
}

TERMINAL_RESOLUTION_STATUSES = {"confirmed", "rejected"}

TERMINAL_DECISIONS = {
    "create_entity",
    "link_to_existing",
    "drop",
}


def stable_id(prefix: str, raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _span_text(chunk_text: str, start: int, end: int) -> str:
    if 0 <= start < end <= len(chunk_text):
        return chunk_text[start:end]
    return ""


def _legacy_reference_to_mention_type(reference_type: str) -> str:
    if reference_type == "implicit":
        return "nominal"
    return reference_type


def _normalize_event_role(role: str) -> str:
    normalized = role.strip().lower()
    return EVENT_ROLE_MAP.get(normalized, normalized or "participant")


def _grounding_expectation_for_event_argument(
    mention: RawMention,
    role: str,
) -> str:
    if mention.mention_type in {"named", "pronoun", "descriptor", "alias"}:
        return "entity_expected"
    if role in {"agent", "speaker", "addressee", "recipient", "experiencer"}:
        return "entity_expected"
    if mention.mention_type in {"object", "location"}:
        return "concept_allowed"
    return "mention_only_allowed"


def _raw_mentions_inside_evidence(
    raw_mentions: list[RawMention],
    evidence_span: EvidenceSpan,
) -> list[RawMention]:
    return [
        mention
        for mention in raw_mentions
        if mention.span.start_char >= evidence_span.start_char
        and mention.span.end_char <= evidence_span.end_char
    ]


def _build_event_argument(
    chunk: ChunkNode,
    event_frame_id: str,
    predicate: str,
    raw_mention: RawMention,
    role: str,
    argument_index: int,
    evidence_span_id: str,
    confidence: float | None = None,
    extractor_version: str = "event_argument_repair_v0.1",
) -> EventArgument:
    return EventArgument(
        argument_id=stable_id(
            "eventarg",
            (
                f"{chunk.chunk_id}:{event_frame_id}:"
                f"{raw_mention.mention_id}:{role}:{argument_index}"
            ),
        ),
        event_frame_id=event_frame_id,
        role=role,
        mention_id=raw_mention.mention_id,
        surface_text=raw_mention.text,
        evidence_span_id=evidence_span_id,
        resolution_status="unresolved",
        grounding_expectation=_grounding_expectation_for_event_argument(
            raw_mention,
            role,
        ),
        argument_index=argument_index,
        extractor_version=extractor_version,
        resolver_version="pending",
        confidence=confidence,
    )


def _repair_event_frame_arguments(
    chunk: ChunkNode,
    event_frame: EventFrame,
    evidence_span: EvidenceSpan,
    raw_mentions: list[RawMention],
) -> list[EventArgument]:
    candidates = _raw_mentions_inside_evidence(raw_mentions, evidence_span)
    if not candidates:
        return []

    local_evidence_start = evidence_span.start_char - chunk.start_char
    local_evidence_end = evidence_span.end_char - chunk.start_char
    predicate_start = chunk.text.find(
        event_frame.predicate,
        max(0, local_evidence_start),
        max(0, local_evidence_end),
    )
    predicate_global_start = (
        chunk.start_char + predicate_start
        if predicate_start >= 0
        else evidence_span.start_char
    )
    predicate_global_end = predicate_global_start + len(event_frame.predicate)

    before = [
        mention
        for mention in candidates
        if mention.span.end_char <= predicate_global_start
    ]
    after = [
        mention
        for mention in candidates
        if mention.span.start_char >= predicate_global_end
    ]

    repaired: list[EventArgument] = []
    if before:
        agent = max(before, key=lambda mention: mention.span.end_char)
        repaired.append(
            _build_event_argument(
                chunk=chunk,
                event_frame_id=event_frame.event_frame_id,
                predicate=event_frame.predicate,
                raw_mention=agent,
                role="agent",
                argument_index=0,
                evidence_span_id=evidence_span.span_id,
                confidence=0.45,
            )
        )

    if after:
        patient = min(after, key=lambda mention: mention.span.start_char)
        if not repaired or patient.mention_id != repaired[0].mention_id:
            repaired.append(
                _build_event_argument(
                    chunk=chunk,
                    event_frame_id=event_frame.event_frame_id,
                    predicate=event_frame.predicate,
                    raw_mention=patient,
                    role="patient",
                    argument_index=len(repaired),
                    evidence_span_id=evidence_span.span_id,
                    confidence=0.35,
                )
            )

    if not repaired:
        nearest = min(
            candidates,
            key=lambda mention: abs(mention.span.start_char - predicate_global_start),
        )
        repaired.append(
            _build_event_argument(
                chunk=chunk,
                event_frame_id=event_frame.event_frame_id,
                predicate=event_frame.predicate,
                raw_mention=nearest,
                role="participant",
                argument_index=0,
                evidence_span_id=evidence_span.span_id,
                confidence=0.30,
            )
        )

    return repaired


def _mention_kind_for_raw_mention(mention: RawMention) -> str:
    if mention.mention_type == "named":
        return "named_entity"
    if mention.mention_type == "pronoun":
        return "pronoun"
    if mention.mention_type == "alias":
        return "descriptive_alias"
    if mention.mention_type == "descriptor":
        if any(
            key in mention.semantic_payload
            for key in ("gender_hint", "age_hint", "person_hint")
        ):
            return "role_anchor"
        return "descriptive_alias"
    return "generic_nominal"


def _mark_terminal(
    terminal_mention_ids: set[str],
    hypothesis: ResolutionHypothesis,
) -> None:
    if hypothesis.is_terminal:
        terminal_mention_ids.add(hypothesis.mention_id)


def _reference_type_for_raw_mention(mention: RawMention) -> str:
    if mention.mention_type == "pronoun":
        return "pronoun"
    if mention.mention_type == "named":
        return "named"
    return "nominal"


def _entity_type_for_raw_mention(mention: RawMention) -> str:
    if mention.mention_type == "object":
        return "object"
    if mention.mention_type == "location":
        return "place"
    if any(
        key in mention.semantic_payload
        for key in ("gender_hint", "age_hint", "person_hint")
    ):
        return "person"
    if mention.mention_type in {"descriptor", "alias", "pronoun"}:
        return "person"
    return "unknown"


def _unknown_entity_name(mention: RawMention) -> str:
    normalized = mention.normalized_text or mention.text.strip().lower()
    safe = "_".join(part for part in normalized.split() if part)
    return f"unknown_{safe or 'entity'}_1"


def _is_sentence_start(mention: RawMention, chunk: ChunkNode) -> bool:
    local_start = max(0, mention.span.start_char - chunk.start_char)
    before = chunk.text[:local_start].rstrip()
    return not before or before.endswith((".", "!", "?", "…", "\n"))


def _has_anchor_evidence(mention: RawMention, event_frames: list[EventFrame]) -> bool:
    if mention.semantic_payload.get("person_hint"):
        return True
    if mention.semantic_payload.get("location_hint"):
        return True
    if mention.semantic_payload.get("organization_hint"):
        return True

    for event_frame in event_frames:
        for argument in event_frame.arguments:
            if (
                argument.mention_id == mention.mention_id
                and argument.role in CORE_EVENT_ROLES
            ):
                return True

    text = mention.text.strip()
    parts = text.split()
    return len(parts) >= 2 and all(part[:1].isupper() for part in parts)


def _anchor_policy_rejection_reason(
    mention: RawMention,
    chunk: ChunkNode,
    event_frames: list[EventFrame],
) -> str | None:
    text = mention.text.strip()
    normalized = (mention.normalized_text or text.lower()).strip()

    if mention.mention_type != "named":
        return "not_named_candidate"
    if not text:
        return "empty_anchor_candidate"
    if mention.mention_type == "pronoun":
        return "pronoun_not_anchor"
    if normalized in ANCHOR_NOISE_TERMS:
        return "sentence_start_adverbial_noise"
    if any(normalized.startswith(prefix) for prefix in ANCHOR_NOISE_PREFIXES):
        return "prepositional_or_adverbial_phrase_noise"
    if normalized.split()[0] in DISCOURSE_MARKERS:
        return "discourse_marker_noise"
    if _is_sentence_start(mention, chunk) and len(text.split()) == 1 and not _has_anchor_evidence(mention, event_frames):
        return "generic_capitalized_sentence_start"
    if not _has_anchor_evidence(mention, event_frames):
        return "missing_anchor_evidence"

    return None


async def extract_graph_with_llm(
    state: ChunkGraphState,
    model_adapter,
) -> dict:
    """
    LangGraph node.

    Input:
        chunk + previous RLM state

    Output:
        raw LLM structured extraction

    Important:
        LLM does NOT write to Neo4j.
        LLM does NOT create final IDs.
        LLM only returns semantic extraction with names and offsets.
    """

    chunk = state["chunk"]
    rlm_state = state["rlm_state"]

    extraction = await model_adapter.structured_call(
        system_prompt=GRAPH_EXTRACTION_SYSTEM_PROMPT,
        user_payload={
            "chunk": chunk.model_dump(),
            "previous_rlm_state": rlm_state.model_dump(),
            "task": (
                "Extract raw mentions, evidence spans, event frames, claims, "
                "and raw relation candidates from the current chunk. Use "
                "previous RLM state only as salience context, not for identity "
                "resolution. Do not invent facts that are not supported by "
                "the current chunk. Do not return an empty observation graph "
                "when the chunk contains named characters, places, objects, "
                "or explicit story facts."
            ),
        },
        output_schema=LLMGraphExtraction,
    )

    return {
        "llm_extraction": extraction,
    }


def validate_llm_extraction(state: ChunkGraphState) -> dict:
    """
    LangGraph node.

    Validates LLM output before normalization.

    Current validation:
        - character spans are inside chunk boundaries
        - mention span text approximately matches chunk text
        - confidence is already validated by Pydantic schema

    Later:
        - add repair node
        - add retry on invalid spans
        - add stricter evidence checks
    """

    chunk = state["chunk"]
    extraction = state["llm_extraction"]

    errors: list[str] = []

    def valid_span(start: int, end: int) -> bool:
        return 0 <= start < end <= len(chunk.text)

    def span_text(start: int, end: int) -> str:
        if not valid_span(start, end):
            return ""
        return chunk.text[start:end]

    for mention in extraction.raw_mentions:
        start = mention.start_char_in_chunk
        end = mention.end_char_in_chunk

        if not valid_span(start, end):
            errors.append(
                f"Invalid raw mention span: text={mention.text!r}, "
                f"start={start}, end={end}, chunk_len={len(chunk.text)}"
            )
            continue

        actual = span_text(start, end)

        if actual.strip() != mention.text.strip():
            errors.append(
                f"Raw mention span mismatch: expected={mention.text!r}, "
                f"actual={actual!r}, start={start}, end={end}"
            )

    for mention in extraction.mentions:
        start = mention.start_char_in_chunk
        end = mention.end_char_in_chunk

        if not valid_span(start, end):
            errors.append(
                f"Invalid mention span: text={mention.text!r}, "
                f"start={start}, end={end}, chunk_len={len(chunk.text)}"
            )
            continue

        actual = span_text(start, end)

        if actual.strip() != mention.text.strip():
            errors.append(
                f"Mention span mismatch: expected={mention.text!r}, "
                f"actual={actual!r}, start={start}, end={end}"
            )

    for event_frame in extraction.event_frames:
        start = event_frame.evidence_start_char_in_chunk
        end = event_frame.evidence_end_char_in_chunk

        if not valid_span(start, end):
            errors.append(
                f"Invalid event frame evidence span: predicate={event_frame.predicate!r}, "
                f"start={start}, end={end}, chunk_len={len(chunk.text)}"
            )

        for argument in event_frame.arguments:
            arg_start = argument.mention_start_char_in_chunk
            arg_end = argument.mention_end_char_in_chunk

            if not valid_span(arg_start, arg_end):
                errors.append(
                    f"Invalid event frame argument span: text={argument.mention_text!r}, "
                    f"start={arg_start}, end={arg_end}, chunk_len={len(chunk.text)}"
                )
                continue

            actual = span_text(arg_start, arg_end)
            if actual.strip() != argument.mention_text.strip():
                errors.append(
                    f"Event frame argument span mismatch: expected={argument.mention_text!r}, "
                    f"actual={actual!r}, start={arg_start}, end={arg_end}"
                )

    for claim in extraction.claims:
        start = claim.evidence_start_char_in_chunk
        end = claim.evidence_end_char_in_chunk

        if not valid_span(start, end):
            errors.append(
                f"Invalid claim evidence span: text={claim.text!r}, "
                f"start={start}, end={end}, chunk_len={len(chunk.text)}"
            )

    for event in extraction.events:
        start = event.evidence_start_char_in_chunk
        end = event.evidence_end_char_in_chunk

        if not valid_span(start, end):
            errors.append(
                f"Invalid event evidence span: description={event.description!r}, "
                f"start={start}, end={end}, chunk_len={len(chunk.text)}"
            )

    for relation in extraction.relations:
        start = relation.evidence_start_char_in_chunk
        end = relation.evidence_end_char_in_chunk

        if not valid_span(start, end):
            errors.append(
                f"Invalid relation evidence span: "
                f"{relation.source_entity_name!r} -[{relation.relation_type}]-> "
                f"{relation.target_entity_name!r}, "
                f"start={start}, end={end}, chunk_len={len(chunk.text)}"
            )

    return {
        "errors": state.get("errors", []) + errors,
    }


def normalize_graph_patch(state: ChunkGraphState) -> dict:
    """
    LangGraph node.

    Converts raw LLMGraphExtraction into deterministic LocalGraphPatch.

    Responsibilities:
        - generate stable IDs
        - convert chunk-local offsets to document-global offsets
        - preserve raw mentions before identity resolution
        - project named anchors through explicit resolution hypotheses
        - link claims/events/relations to evidence spans
    """

    chunk = state["chunk"]
    extraction = state["llm_extraction"]

    entity_by_name: dict[str, EntityNode] = {}
    raw_mentions: list[RawMention] = []
    raw_mention_by_span: dict[tuple[int, int, str], RawMention] = {}

    raw_mention_inputs = list(extraction.raw_mentions)
    for legacy_mention in extraction.mentions:
        if any(
            candidate.start_char_in_chunk == legacy_mention.start_char_in_chunk
            and candidate.end_char_in_chunk == legacy_mention.end_char_in_chunk
            for candidate in raw_mention_inputs
        ):
            continue

        raw_mention_inputs.append(
            ExtractedRawMention(
                text=legacy_mention.text,
                start_char_in_chunk=legacy_mention.start_char_in_chunk,
                end_char_in_chunk=legacy_mention.end_char_in_chunk,
                mention_type=_legacy_reference_to_mention_type(
                    legacy_mention.reference_type
                ),
                normalized_text=legacy_mention.text.strip().lower(),
                extractor_source="legacy",
                extractor_version="legacy_entities_mentions_v0.1",
                repaired_by="legacy_to_raw_mention",
                repair_notes=[
                    "Converted legacy ExtractedMention into RawMention candidate."
                ],
                confidence=0.8,
                semantic_payload={
                    "legacy_canonical_entity_name": legacy_mention.canonical_entity_name,
                    "legacy_entity_type": legacy_mention.entity_type,
                },
            )
        )

    for raw_input in raw_mention_inputs:
        text = raw_input.text.strip()
        if not text:
            continue

        mention_id = stable_id(
            "rawmention",
            (
                f"{chunk.chunk_id}:"
                f"{raw_input.start_char_in_chunk}:"
                f"{raw_input.end_char_in_chunk}:"
                f"{raw_input.text}"
            ),
        )
        raw_mention = RawMention(
            mention_id=mention_id,
            chunk_id=chunk.chunk_id,
            text=raw_input.text,
            span=TextSpan(
                start_char=chunk.start_char + raw_input.start_char_in_chunk,
                end_char=chunk.start_char + raw_input.end_char_in_chunk,
                text=raw_input.text,
            ),
            normalized_text=raw_input.normalized_text or text.lower(),
            mention_type=raw_input.mention_type,
            source=raw_input.extractor_source,
            extractor_source=raw_input.extractor_source,
            extractor_version=raw_input.extractor_version,
            repaired_by=raw_input.repaired_by,
            repair_notes=raw_input.repair_notes,
            confidence=raw_input.confidence,
            semantic_payload=raw_input.semantic_payload,
        )
        raw_mention.mention_kind = _mention_kind_for_raw_mention(raw_mention)
        raw_mentions.append(raw_mention)
        raw_mention_by_span[
            (
                raw_input.start_char_in_chunk,
                raw_input.end_char_in_chunk,
                raw_input.text,
            )
        ] = raw_mention

    mentions: list[MentionNode] = []
    resolution_hypotheses: list[ResolutionHypothesis] = []
    terminal_resolutions: list[TerminalResolution] = []
    resolved_mention_entity_ids: dict[str, str] = {}
    terminal_mention_ids: set[str] = set()
    resolution_run_id = stable_id(
        "resolution_run",
        f"{chunk.document_id}:{chunk.chunk_id}:deterministic_anchor_v0.1",
    )

    evidence_spans: dict[str, EvidenceSpan] = {}

    def add_resolution_hypothesis(hypothesis: ResolutionHypothesis) -> None:
        resolution_hypotheses.append(hypothesis)
        _mark_terminal(terminal_mention_ids, hypothesis)
        if not hypothesis.is_terminal:
            return

        terminal_resolutions.append(
            TerminalResolution(
                mention_id=hypothesis.mention_id,
                decision=hypothesis.entity_creation_decision,
                authority=hypothesis.authority,
                confidence=hypothesis.confidence,
                policy_version=hypothesis.policy_version,
                created_at_stage=hypothesis.decision_stage or "unknown",
                final_entity_id=hypothesis.final_entity_id,
                revisable_by_higher_authority=(
                    hypothesis.authority != "human_override"
                ),
            )
        )

    def build_evidence_span(
        prefix: str,
        local_start: int,
        local_end: int,
        refs: list[ResolvedEntityRef],
        normalized_text: str | None = None,
    ) -> EvidenceSpan:
        original_text = _span_text(chunk.text, local_start, local_end)
        span_id = stable_id(
            prefix,
            f"{chunk.chunk_id}:{local_start}:{local_end}:{original_text}",
        )

        span = EvidenceSpan(
            span_id=span_id,
            original_text=original_text,
            normalized_text=normalized_text,
            start_char=chunk.start_char + local_start,
            end_char=chunk.start_char + local_end,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            resolved_entities=refs,
        )
        evidence_spans[span_id] = span
        return span

    event_frames: list[EventFrame] = []

    for raw_event_frame in extraction.event_frames:
        argument_refs: list[EventArgument] = []

        for argument_index, argument in enumerate(raw_event_frame.arguments):
            raw_mention = raw_mention_by_span.get(
                (
                    argument.mention_start_char_in_chunk,
                    argument.mention_end_char_in_chunk,
                    argument.mention_text,
                )
            )
            if raw_mention is None:
                continue
            normalized_role = _normalize_event_role(argument.role)

            argument_refs.append(
                EventArgument(
                    argument_id=stable_id(
                        "eventarg",
                        (
                            f"{chunk.chunk_id}:{raw_event_frame.predicate}:"
                            f"{raw_mention.mention_id}:{normalized_role}:{argument_index}"
                        ),
                    ),
                    role=normalized_role,
                    mention_id=raw_mention.mention_id,
                    surface_text=raw_mention.text,
                    evidence_span_id=None,
                    resolution_status="unresolved",
                    grounding_expectation=_grounding_expectation_for_event_argument(
                        raw_mention,
                        normalized_role,
                    ),
                    argument_index=argument_index,
                    extractor_version="llm_graph_extraction_v0.2",
                    resolver_version="pending",
                    confidence=argument.confidence,
                )
            )

        evidence_span = build_evidence_span(
            prefix="evidence",
            local_start=raw_event_frame.evidence_start_char_in_chunk,
            local_end=raw_event_frame.evidence_end_char_in_chunk,
            refs=[],
        )
        event_frame_id = stable_id(
            "eventframe",
            (
                f"{chunk.chunk_id}:{raw_event_frame.predicate}:"
                f"{raw_event_frame.evidence_start_char_in_chunk}:"
                f"{raw_event_frame.evidence_end_char_in_chunk}"
            ),
        )
        event_frames.append(
            EventFrame(
                event_frame_id=event_frame_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                predicate=raw_event_frame.predicate,
                normalized_predicate=raw_event_frame.predicate.strip().lower(),
                event_type=raw_event_frame.event_type,
                arguments=argument_refs,
                evidence_span_id=evidence_span.span_id,
                resolution_status="unresolved",
                materialization_status="valid",
                confidence=raw_event_frame.confidence,
                extractor_version="llm_graph_extraction_v0.2",
                resolver_version="pending",
            )
        )
        if not event_frames[-1].arguments:
            event_frames[-1].arguments = _repair_event_frame_arguments(
                chunk=chunk,
                event_frame=event_frames[-1],
                evidence_span=evidence_span,
                raw_mentions=raw_mentions,
            )
        for argument in event_frames[-1].arguments:
            argument.event_frame_id = event_frame_id
            argument.evidence_span_id = evidence_span.span_id

    # Named mentions become canonical anchors only through a confirmed
    # resolution hypothesis. Descriptors and pronouns stay raw here.
    for raw_mention in raw_mentions:
        if raw_mention.mention_type != "named":
            continue

        rejection_reason = _anchor_policy_rejection_reason(
            raw_mention,
            chunk,
            event_frames,
        )
        if rejection_reason is not None:
            raw_mention.mention_kind = "noise"
            add_resolution_hypothesis(
                ResolutionHypothesis(
                    hypothesis_id=stable_id(
                        "reshyp",
                        (
                            f"{resolution_run_id}:{raw_mention.mention_id}:"
                            f"anchor_rejected"
                        ),
                    ),
                    mention_id=raw_mention.mention_id,
                    hypothesis_type="unresolved",
                    confidence=0.0,
                    status="rejected",
                    mention_kind=raw_mention.mention_kind,
                    entity_creation_decision="drop",
                    final_entity_id=None,
                    candidate_entity_ids=[],
                    candidate_scores={},
                    decision_stage="anchor_policy",
                    authority="entity_anchor_policy",
                    is_terminal=True,
                    reason=rejection_reason,
                    negative_evidence=[
                        "EntityAnchorPolicy rejected named candidate",
                        rejection_reason,
                    ],
                    resolution_run_id=resolution_run_id,
                    resolver_version="entity_anchor_policy_v0.1",
                    policy_version="entity_anchor_policy_v0.1",
                    extractor_version=raw_mention.extractor_version,
                    chunking_version="chunker_v0.2",
                )
            )
            continue

        canonical_name = raw_mention.text.strip()
        entity_id = stable_id("entity", canonical_name.lower())
        entity = entity_by_name.get(canonical_name)
        if entity is None:
            legacy_payload = raw_mention.semantic_payload
            entity = EntityNode(
                entity_id=entity_id,
                canonical_name=canonical_name,
                entity_type=legacy_payload.get("legacy_entity_type", "person"),
                aliases=[canonical_name],
                description=None,
            )
            entity_by_name[canonical_name] = entity

        mention_id = stable_id("mention", raw_mention.mention_id)
        mentions.append(
            MentionNode(
                mention_id=mention_id,
                chunk_id=chunk.chunk_id,
                text=raw_mention.text,
                span=raw_mention.span,
                entity_id=entity.entity_id,
                reference_type="named",
            )
        )
        resolved_mention_entity_ids[raw_mention.mention_id] = entity.entity_id
        add_resolution_hypothesis(
            ResolutionHypothesis(
                hypothesis_id=stable_id(
                    "reshyp",
                    f"{resolution_run_id}:{raw_mention.mention_id}:{entity.entity_id}",
                ),
                mention_id=raw_mention.mention_id,
                hypothesis_type="mention_to_new_entity",
                candidate_entity_id=entity.entity_id,
                candidate_entity_name=entity.canonical_name,
                confidence=0.95,
                status="confirmed",
                mention_kind=raw_mention.mention_kind,
                entity_creation_decision="create_entity",
                final_entity_id=entity.entity_id,
                candidate_entity_ids=[entity.entity_id],
                candidate_scores={entity.entity_id: 0.95},
                decision_stage="anchor_policy",
                authority="entity_anchor_policy",
                is_terminal=True,
                reason="Named mention passed EntityAnchorPolicy.",
                positive_evidence=[
                    "named mention passed anchor policy",
                ],
                resolution_run_id=resolution_run_id,
                resolver_version="entity_anchor_policy_v0.1",
                policy_version="entity_anchor_policy_v0.1",
                extractor_version=raw_mention.extractor_version,
                chunking_version="chunker_v0.2",
            )
        )

    def refs_for_entity_ids(
        entity_ids: list[str],
        role: str = "participant",
    ) -> list[ResolvedEntityRef]:
        refs: list[ResolvedEntityRef] = []
        for entity_id in entity_ids:
            entity = next(
                (
                    candidate
                    for candidate in entity_by_name.values()
                    if candidate.entity_id == entity_id
                ),
                None,
            )
            refs.append(
                ResolvedEntityRef(
                    entity_id=entity_id,
                    surface_form=entity.canonical_name if entity else entity_id,
                    reference_type="named",
                    role=role,
                    confidence=0.8,
                )
            )
        return refs

    raw_relations: list[RawRelation] = []

    raw_mentions_by_id = {
        raw_mention.mention_id: raw_mention
        for raw_mention in raw_mentions
    }
    anchor_mentions = [
        raw_mention
        for raw_mention in raw_mentions
        if raw_mention.mention_id in resolved_mention_entity_ids
    ]

    def event_role_conflict(
        anchor: RawMention,
        candidate: RawMention,
    ) -> tuple[bool, str | None]:
        for event_frame in event_frames:
            roles_by_mention = {
                argument.mention_id: argument.role
                for argument in event_frame.arguments
            }
            anchor_role = roles_by_mention.get(anchor.mention_id)
            candidate_role = roles_by_mention.get(candidate.mention_id)

            if (
                anchor_role in CORE_EVENT_ROLES
                and candidate_role in CORE_EVENT_ROLES
                and anchor_role != candidate_role
            ):
                return (
                    True,
                    (
                        f"event role conflict: {anchor.text} is {anchor_role}, "
                        f"{candidate.text} is {candidate_role}"
                    ),
                )

        return False, None

    for raw_mention in raw_mentions:
        if raw_mention.mention_id in resolved_mention_entity_ids:
            continue
        if raw_mention.mention_id in terminal_mention_ids:
            continue

        if not anchor_mentions:
            add_resolution_hypothesis(
                ResolutionHypothesis(
                    hypothesis_id=stable_id(
                        "reshyp",
                        f"{resolution_run_id}:{raw_mention.mention_id}:unresolved",
                    ),
                    mention_id=raw_mention.mention_id,
                    hypothesis_type="unresolved",
                    confidence=0.0,
                    status="unresolved",
                    mention_kind=raw_mention.mention_kind,
                    entity_creation_decision="keep_as_mention_only",
                    final_entity_id=None,
                    decision_stage="local_grounding",
                    authority="heuristic_fallback",
                    is_terminal=False,
                    reason="No named anchor is available in the local context.",
                    resolution_run_id=resolution_run_id,
                    resolver_version="entity_grounding_rlm_v0.1",
                    policy_version="entity_resolution_policy_v0.1",
                    extractor_version="gliner_relex_v1",
                    chunking_version="chunker_v0.2",
                )
            )
            continue

        previous_same_surface = next(
            (
                previous
                for previous in raw_mentions
                if previous.mention_id != raw_mention.mention_id
                and previous.span.start_char < raw_mention.span.start_char
                and previous.normalized_text == raw_mention.normalized_text
                and previous.mention_id in resolved_mention_entity_ids
            ),
            None,
        )
        if previous_same_surface is not None:
            entity_id = resolved_mention_entity_ids[previous_same_surface.mention_id]
            entity = next(
                entity
                for entity in entity_by_name.values()
                if entity.entity_id == entity_id
            )
            resolved_mention_entity_ids[raw_mention.mention_id] = entity.entity_id
            mentions.append(
                MentionNode(
                    mention_id=stable_id("mention", raw_mention.mention_id),
                    chunk_id=chunk.chunk_id,
                    text=raw_mention.text,
                    span=raw_mention.span,
                    entity_id=entity.entity_id,
                    reference_type=_reference_type_for_raw_mention(raw_mention),
                )
            )
            add_resolution_hypothesis(
                ResolutionHypothesis(
                    hypothesis_id=stable_id(
                        "reshyp",
                        (
                            f"{resolution_run_id}:{raw_mention.mention_id}:"
                            f"{entity.entity_id}:same_surface"
                        ),
                    ),
                    mention_id=raw_mention.mention_id,
                    hypothesis_type="same_entity",
                    candidate_entity_id=entity.entity_id,
                    candidate_entity_name=entity.canonical_name,
                    confidence=0.82,
                    status="likely",
                    mention_kind=raw_mention.mention_kind,
                    entity_creation_decision="link_to_existing",
                    final_entity_id=entity.entity_id,
                    candidate_entity_ids=[entity.entity_id],
                    candidate_scores={entity.entity_id: 0.82},
                    decision_stage="local_grounding",
                    authority="heuristic_fallback",
                    is_terminal=True,
                    reason=(
                        "A previous raw mention with the same normalized text "
                        "was already resolved in this chunk."
                    ),
                    positive_evidence=[
                        "same normalized descriptor surface",
                    ],
                    resolution_run_id=resolution_run_id,
                    resolver_version="entity_grounding_rlm_v0.1",
                    policy_version="entity_resolution_policy_v0.1",
                    extractor_version="gliner_relex_v1",
                    chunking_version="chunker_v0.2",
                )
            )
            continue

        rejected_anchor_ids: set[str] = set()
        for anchor in anchor_mentions:
            anchor_entity_id = resolved_mention_entity_ids[anchor.mention_id]
            anchor_entity = next(
                entity
                for entity in entity_by_name.values()
                if entity.entity_id == anchor_entity_id
            )
            has_conflict, conflict_reason = event_role_conflict(anchor, raw_mention)
            if not has_conflict:
                continue

            rejected_anchor_ids.add(anchor_entity_id)
            add_resolution_hypothesis(
                ResolutionHypothesis(
                    hypothesis_id=stable_id(
                        "reshyp",
                        (
                            f"{resolution_run_id}:{raw_mention.mention_id}:"
                            f"{anchor_entity.entity_id}:rejected"
                        ),
                    ),
                    mention_id=raw_mention.mention_id,
                    hypothesis_type="mention_to_known_entity",
                    candidate_entity_id=anchor_entity.entity_id,
                    candidate_entity_name=anchor_entity.canonical_name,
                    confidence=0.12,
                    status="rejected",
                    mention_kind=raw_mention.mention_kind,
                    entity_creation_decision="keep_as_mention_only",
                    final_entity_id=None,
                    candidate_entity_ids=[anchor_entity.entity_id],
                    candidate_scores={anchor_entity.entity_id: 0.12},
                    decision_stage="local_grounding",
                    authority="heuristic_fallback",
                    is_terminal=False,
                    reason=conflict_reason or "Event role conflict.",
                    negative_evidence=[conflict_reason or "event role conflict"],
                    resolution_run_id=resolution_run_id,
                    resolver_version="entity_grounding_rlm_v0.1",
                    policy_version="entity_resolution_policy_v0.1",
                    extractor_version="gliner_relex_v1",
                    chunking_version="chunker_v0.2",
                )
            )

        if rejected_anchor_ids:
            candidate_name = _unknown_entity_name(raw_mention)
            candidate_entity_id = stable_id(
                "entity",
                f"{chunk.document_id}:{candidate_name}",
            )
            entity = EntityNode(
                entity_id=candidate_entity_id,
                canonical_name=candidate_name,
                entity_type=_entity_type_for_raw_mention(raw_mention),
                aliases=[raw_mention.text],
                description=None,
            )
            entity_by_name[candidate_name] = entity
            resolved_mention_entity_ids[raw_mention.mention_id] = entity.entity_id
            mentions.append(
                MentionNode(
                    mention_id=stable_id("mention", raw_mention.mention_id),
                    chunk_id=chunk.chunk_id,
                    text=raw_mention.text,
                    span=raw_mention.span,
                    entity_id=entity.entity_id,
                    reference_type=_reference_type_for_raw_mention(raw_mention),
                )
            )
            add_resolution_hypothesis(
                ResolutionHypothesis(
                    hypothesis_id=stable_id(
                        "reshyp",
                        (
                            f"{resolution_run_id}:{raw_mention.mention_id}:"
                            f"{entity.entity_id}:new"
                        ),
                    ),
                    mention_id=raw_mention.mention_id,
                    hypothesis_type="mention_to_new_entity",
                    candidate_entity_id=entity.entity_id,
                    candidate_entity_name=entity.canonical_name,
                    confidence=0.86,
                    status="likely",
                    mention_kind=raw_mention.mention_kind,
                    entity_creation_decision="create_entity",
                    final_entity_id=entity.entity_id,
                    candidate_entity_ids=[entity.entity_id],
                    candidate_scores={entity.entity_id: 0.86},
                    decision_stage="local_grounding",
                    authority="heuristic_fallback",
                    is_terminal=True,
                    reason=(
                        "Descriptor occupies a conflicting event role relative "
                        "to a named anchor, so a new entity is more likely."
                    ),
                    positive_evidence=[
                        "descriptor appears as distinct event participant",
                    ],
                    negative_evidence=[
                        "same-entity hypothesis rejected by event role conflict",
                    ],
                    resolution_run_id=resolution_run_id,
                    resolver_version="entity_grounding_rlm_v0.1",
                    policy_version="entity_resolution_policy_v0.1",
                    extractor_version="gliner_relex_v1",
                    chunking_version="chunker_v0.2",
                )
            )
            continue

        if len(anchor_mentions) == 1 and raw_mention.mention_type in {
            "descriptor",
            "alias",
            "pronoun",
            "nominal",
        }:
            anchor = anchor_mentions[0]
            anchor_entity_id = resolved_mention_entity_ids[anchor.mention_id]
            anchor_entity = next(
                entity
                for entity in entity_by_name.values()
                if entity.entity_id == anchor_entity_id
            )
            resolved_mention_entity_ids[raw_mention.mention_id] = anchor_entity.entity_id
            mentions.append(
                MentionNode(
                    mention_id=stable_id("mention", raw_mention.mention_id),
                    chunk_id=chunk.chunk_id,
                    text=raw_mention.text,
                    span=raw_mention.span,
                    entity_id=anchor_entity.entity_id,
                    reference_type=_reference_type_for_raw_mention(raw_mention),
                )
            )
            add_resolution_hypothesis(
                ResolutionHypothesis(
                    hypothesis_id=stable_id(
                        "reshyp",
                        (
                            f"{resolution_run_id}:{raw_mention.mention_id}:"
                            f"{anchor_entity.entity_id}:known"
                        ),
                    ),
                    mention_id=raw_mention.mention_id,
                    hypothesis_type="mention_to_known_entity",
                    candidate_entity_id=anchor_entity.entity_id,
                    candidate_entity_name=anchor_entity.canonical_name,
                    confidence=0.70,
                    status="likely",
                    mention_kind=raw_mention.mention_kind,
                    entity_creation_decision="link_to_existing",
                    final_entity_id=anchor_entity.entity_id,
                    candidate_entity_ids=[anchor_entity.entity_id],
                    candidate_scores={anchor_entity.entity_id: 0.70},
                    decision_stage="local_grounding",
                    authority="heuristic_fallback",
                    is_terminal=True,
                    reason=(
                        "Single nearby named anchor and no event-role conflict "
                        "in local event frames."
                    ),
                    positive_evidence=[
                        "single local anchor",
                        "no event role conflict",
                    ],
                    resolution_run_id=resolution_run_id,
                    resolver_version="entity_grounding_rlm_v0.1",
                    policy_version="entity_resolution_policy_v0.1",
                    extractor_version="gliner_relex_v1",
                    chunking_version="chunker_v0.2",
                )
            )
            continue

        add_resolution_hypothesis(
            ResolutionHypothesis(
                hypothesis_id=stable_id(
                    "reshyp",
                    f"{resolution_run_id}:{raw_mention.mention_id}:unresolved",
                ),
                mention_id=raw_mention.mention_id,
                hypothesis_type="unresolved",
                confidence=0.0,
                status="unresolved",
                mention_kind=raw_mention.mention_kind,
                entity_creation_decision="keep_as_mention_only",
                final_entity_id=None,
                decision_stage="local_grounding",
                authority="heuristic_fallback",
                is_terminal=False,
                reason="Grounding candidates are ambiguous or below policy threshold.",
                resolution_run_id=resolution_run_id,
                resolver_version="entity_grounding_rlm_v0.1",
                policy_version="entity_resolution_policy_v0.1",
                extractor_version="gliner_relex_v1",
                chunking_version="chunker_v0.2",
            )
        )

    for event_frame in event_frames:
        resolved_count = 0
        expected_count = 0
        bound_arguments: list[EventArgument] = []
        for argument in event_frame.arguments:
            entity_id = resolved_mention_entity_ids.get(argument.mention_id)
            expects_entity = argument.grounding_expectation == "entity_expected"
            if expects_entity:
                expected_count += 1
            if entity_id:
                resolution_status = "resolved"
            elif expects_entity:
                resolution_status = "unresolved"
            else:
                resolution_status = "mention_only"
            if entity_id:
                resolved_count += 1
            bound_arguments.append(
                argument.model_copy(
                    update={
                        "entity_id": entity_id,
                        "resolution_status": resolution_status,
                        "resolver_version": "entity_resolution_policy_v0.1",
                    }
                )
            )
        event_frame.arguments = bound_arguments
        if not bound_arguments:
            event_frame.resolution_status = "unresolved"
            event_frame.materialization_status = "rejected"
        elif expected_count == 0:
            event_frame.resolution_status = "complete"
            event_frame.materialization_status = "valid"
        elif resolved_count == 0:
            event_frame.resolution_status = "unresolved"
            event_frame.materialization_status = "degraded"
        elif resolved_count >= expected_count:
            event_frame.resolution_status = "complete"
            event_frame.materialization_status = "valid"
        else:
            event_frame.resolution_status = "partial"
            event_frame.materialization_status = "degraded"
        event_frame.resolver_version = "entity_resolution_policy_v0.1"

    # 4. Normalize claims and evidence spans.
    claims: list[ClaimNode] = []

    for raw_claim in extraction.claims:
        subject_entity_ids = []

        for name in raw_claim.subject_entity_names:
            entity = entity_by_name.get(name.strip())
            if entity is not None:
                subject_entity_ids.append(entity.entity_id)

        evidence_text = chunk.text[
            raw_claim.evidence_start_char_in_chunk:
            raw_claim.evidence_end_char_in_chunk
        ]
        evidence_span = build_evidence_span(
            prefix="evidence",
            local_start=raw_claim.evidence_start_char_in_chunk,
            local_end=raw_claim.evidence_end_char_in_chunk,
            refs=refs_for_entity_ids(subject_entity_ids, role="subject"),
        )

        claim_id = stable_id(
            "claim",
            f"{chunk.chunk_id}:{raw_claim.text}",
        )

        claims.append(
            ClaimNode(
                claim_id=claim_id,
                chunk_id=chunk.chunk_id,
                text=raw_claim.text,
                subject_entity_ids=subject_entity_ids,
                evidence_span=TextSpan(
                    start_char=evidence_span.start_char,
                    end_char=evidence_span.end_char,
                    text=evidence_text,
                ),
                confidence=raw_claim.confidence,
                needs_verification=raw_claim.needs_verification,
            )
        )

    # 4. Normalize events.
    events: list[EventNode] = []

    for raw_event in extraction.events:
        participant_ids = []

        for name in raw_event.participant_entity_names:
            entity = entity_by_name.get(name.strip())
            if entity is not None:
                participant_ids.append(entity.entity_id)

        evidence_text = chunk.text[
            raw_event.evidence_start_char_in_chunk:
            raw_event.evidence_end_char_in_chunk
        ]
        evidence_span = build_evidence_span(
            prefix="evidence",
            local_start=raw_event.evidence_start_char_in_chunk,
            local_end=raw_event.evidence_end_char_in_chunk,
            refs=refs_for_entity_ids(participant_ids, role="participant"),
        )

        event_id = stable_id(
            "event",
            f"{chunk.chunk_id}:{raw_event.event_type}:{raw_event.description}",
        )

        events.append(
            EventNode(
                event_id=event_id,
                chunk_id=chunk.chunk_id,
                event_type=raw_event.event_type,
                description=raw_event.description,
                participants=participant_ids,
                evidence_span=TextSpan(
                    start_char=evidence_span.start_char,
                    end_char=evidence_span.end_char,
                    text=evidence_text,
                ),
                confidence=raw_event.confidence,
            )
        )

    # 5. Normalize explicit relations and latent relation candidates.
    relations: list[RelationEdge] = []
    relation_candidates: list[RelationCandidate] = []

    for event_frame in event_frames:
        role_mentions = {
            argument.role: argument.mention_id
            for argument in event_frame.arguments
        }
        source_mention_id = role_mentions.get("agent")
        target_mention_id = role_mentions.get("patient")

        if source_mention_id is None or target_mention_id is None:
            continue

        source_entity_id = resolved_mention_entity_ids.get(source_mention_id)
        target_entity_id = resolved_mention_entity_ids.get(target_mention_id)

        if source_entity_id is None or target_entity_id is None:
            continue

        if source_entity_id == target_entity_id:
            continue

        source_mention = raw_mentions_by_id[source_mention_id]
        target_mention = raw_mentions_by_id[target_mention_id]
        source_entity = next(
            entity
            for entity in entity_by_name.values()
            if entity.entity_id == source_entity_id
        )
        target_entity = next(
            entity
            for entity in entity_by_name.values()
            if entity.entity_id == target_entity_id
        )
        evidence_span = evidence_spans[event_frame.evidence_span_id]
        evidence_span.resolved_entities = [
            ResolvedEntityRef(
                entity_id=source_entity.entity_id,
                surface_form=source_mention.text,
                reference_type=_reference_type_for_raw_mention(source_mention),
                role="source",
                confidence=0.86,
            ),
            ResolvedEntityRef(
                entity_id=target_entity.entity_id,
                surface_form=target_mention.text,
                reference_type=_reference_type_for_raw_mention(target_mention),
                role="target",
                confidence=0.86,
            ),
        ]

        relation_span = event_frame.predicate
        symbolic_hint = (event_frame.event_type or event_frame.predicate).upper()
        relation_id = stable_id(
            "relation",
            (
                f"{chunk.chunk_id}:{event_frame.event_frame_id}:"
                f"{source_entity.entity_id}:{target_entity.entity_id}"
            ),
        )
        relation_candidate_id = stable_id(
            "relcand",
            (
                f"{chunk.chunk_id}:{event_frame.event_frame_id}:"
                f"{source_entity.entity_id}:{target_entity.entity_id}:projected"
            ),
        )
        relation_candidates.append(
            RelationCandidate(
                relation_candidate_id=relation_candidate_id,
                source_entity_id=source_entity.entity_id,
                target_entity_id=target_entity.entity_id,
                relation_span=relation_span,
                evidence_span_id=evidence_span.span_id,
                direction_hint=f"{source_entity.entity_id} -> {target_entity.entity_id}",
                direction_confidence=0.86,
                encode_as_latent_edge=True,
                symbolic_hint=symbolic_hint,
                confidence=event_frame.confidence or 0.86,
            )
        )
        relations.append(
            RelationEdge(
                relation_id=relation_id,
                source_id=source_entity.entity_id,
                target_id=target_entity.entity_id,
                relation_type=symbolic_hint,
                chunk_id=chunk.chunk_id,
                evidence_span=TextSpan(
                    start_char=evidence_span.start_char,
                    end_char=evidence_span.end_char,
                    text=evidence_span.original_text,
                ),
                confidence=event_frame.confidence or 0.86,
            )
        )

    for raw_relation in extraction.relations:
        source_name = raw_relation.source_entity_name.strip()
        target_name = raw_relation.target_entity_name.strip()

        relation_evidence_span = build_evidence_span(
            prefix="evidence",
            local_start=raw_relation.evidence_start_char_in_chunk,
            local_end=raw_relation.evidence_end_char_in_chunk,
            refs=[],
        )
        source_raw_mention = next(
            (
                mention
                for mention in raw_mentions
                if mention.text.strip() == source_name
            ),
            None,
        )
        target_raw_mention = next(
            (
                mention
                for mention in raw_mentions
                if mention.text.strip() == target_name
            ),
            None,
        )
        if source_raw_mention is not None and target_raw_mention is not None:
            raw_relations.append(
                RawRelation(
                    raw_relation_id=stable_id(
                        "rawrel",
                        (
                            f"{chunk.chunk_id}:{source_raw_mention.mention_id}:"
                            f"{target_raw_mention.mention_id}:"
                            f"{raw_relation.relation_span or ''}"
                        ),
                    ),
                    chunk_id=chunk.chunk_id,
                    source_mention_id=source_raw_mention.mention_id,
                    target_mention_id=target_raw_mention.mention_id,
                    relation_span=raw_relation.relation_span
                    or relation_evidence_span.original_text,
                    evidence_span_id=relation_evidence_span.span_id,
                    relation_type=raw_relation.relation_type,
                    confidence=raw_relation.confidence,
                )
            )

        source = entity_by_name.get(source_name)
        target = entity_by_name.get(target_name)

        # For the first version we skip relations whose entities were not extracted.
        # Canonical projection must wait until resolution has accepted entities.
        if source is None or target is None:
            continue

        evidence_text = chunk.text[
            raw_relation.evidence_start_char_in_chunk:
            raw_relation.evidence_end_char_in_chunk
        ]
        relation_evidence_span.resolved_entities = [
            ResolvedEntityRef(
                entity_id=source.entity_id,
                surface_form=source.canonical_name,
                reference_type="named",
                role="source",
                confidence=0.8,
            ),
            ResolvedEntityRef(
                entity_id=target.entity_id,
                surface_form=target.canonical_name,
                reference_type="named",
                role="target",
                confidence=0.8,
            ),
        ]
        evidence_span = build_evidence_span(
            prefix="evidence",
            local_start=raw_relation.evidence_start_char_in_chunk,
            local_end=raw_relation.evidence_end_char_in_chunk,
            refs=[
                ResolvedEntityRef(
                    entity_id=source.entity_id,
                    surface_form=source.canonical_name,
                    reference_type="named",
                    role="source",
                    confidence=0.8,
                ),
                ResolvedEntityRef(
                    entity_id=target.entity_id,
                    surface_form=target.canonical_name,
                    reference_type="named",
                    role="target",
                    confidence=0.8,
                ),
            ],
        )

        relation_span = raw_relation.relation_span or evidence_text
        symbolic_hint = raw_relation.relation_type

        relation_id = stable_id(
            "relation",
            (
                f"{chunk.chunk_id}:"
                f"{source.entity_id}:"
                f"{symbolic_hint or relation_span}:"
                f"{target.entity_id}"
            ),
        )
        relation_candidate_id = stable_id(
            "relcand",
            (
                f"{chunk.chunk_id}:{source.entity_id}:"
                f"{target.entity_id}:{relation_span}:{evidence_span.span_id}"
            ),
        )
        relation_candidates.append(
            RelationCandidate(
                relation_candidate_id=relation_candidate_id,
                source_entity_id=source.entity_id,
                target_entity_id=target.entity_id,
                relation_span=relation_span,
                evidence_span_id=evidence_span.span_id,
                direction_hint=f"{source.entity_id} -> {target.entity_id}",
                direction_confidence=raw_relation.direction_confidence,
                encode_as_latent_edge=True,
                symbolic_hint=symbolic_hint,
                confidence=raw_relation.confidence,
            )
        )

        relations.append(
            RelationEdge(
                relation_id=relation_id,
                source_id=source.entity_id,
                target_id=target.entity_id,
                relation_type=symbolic_hint or "latent_candidate",
                chunk_id=chunk.chunk_id,
                evidence_span=TextSpan(
                    start_char=evidence_span.start_char,
                    end_char=evidence_span.end_char,
                    text=evidence_text,
                ),
                confidence=raw_relation.confidence,
            )
        )

    # 6. If a claim has two or more entities, create entity-first relation candidates.
    for claim in claims:
        if claim.evidence_span is None or len(claim.subject_entity_ids) < 2:
            continue

        evidence_span = build_evidence_span(
            prefix="evidence",
            local_start=claim.evidence_span.start_char - chunk.start_char,
            local_end=claim.evidence_span.end_char - chunk.start_char,
            refs=refs_for_entity_ids(claim.subject_entity_ids, role="participant"),
        )

        for source_id in claim.subject_entity_ids:
            for target_id in claim.subject_entity_ids:
                if source_id == target_id:
                    continue

                relation_candidate_id = stable_id(
                    "relcand",
                    f"{chunk.chunk_id}:{source_id}:{target_id}:{claim.claim_id}",
                )
                if any(
                    candidate.relation_candidate_id == relation_candidate_id
                    for candidate in relation_candidates
                ):
                    continue

                relation_candidates.append(
                    RelationCandidate(
                        relation_candidate_id=relation_candidate_id,
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relation_span=claim.text,
                        evidence_span_id=evidence_span.span_id,
                        direction_hint=None,
                        direction_confidence=0.0,
                        encode_as_latent_edge=True,
                        symbolic_hint=None,
                        confidence=claim.confidence,
                    )
                )

    graph_patch = LocalGraphPatch(
        chunk=chunk,
        raw_mentions=raw_mentions,
        raw_relations=raw_relations,
        event_frames=event_frames,
        resolution_hypotheses=resolution_hypotheses,
        terminal_resolutions=terminal_resolutions,
        entities=list(entity_by_name.values()),
        mentions=mentions,
        claims=claims,
        events=events,
        relations=relations,
        evidence_spans=list(evidence_spans.values()),
        relation_candidates=relation_candidates,
    )

    return {
        "graph_patch": graph_patch,
    }
