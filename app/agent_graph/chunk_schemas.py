from typing import Literal
from pydantic import BaseModel, Field


EntityType = Literal[
    "person",
    "place",
    "object",
    "organization",
    "event",
    "concept",
    "unknown",
]

MentionType = Literal[
    "named",
    "pronoun",
    "descriptor",
    "alias",
    "nominal",
    "object",
    "location",
]


class ExtractedEntity(BaseModel):
    canonical_name: str
    entity_type: EntityType = "unknown"
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class ExtractedMention(BaseModel):
    text: str
    start_char_in_chunk: int
    end_char_in_chunk: int
    canonical_entity_name: str
    entity_type: EntityType = "unknown"
    reference_type: Literal["named", "pronoun", "nominal", "implicit"] = "named"


class ExtractedRawMention(BaseModel):
    text: str
    start_char_in_chunk: int
    end_char_in_chunk: int
    mention_type: MentionType = "nominal"
    normalized_text: str | None = None
    extractor_source: Literal["llm", "gliner_relex", "repair", "fallback", "legacy"] = "llm"
    extractor_version: str | None = None
    repaired_by: str | None = None
    repair_notes: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_payload: dict = Field(default_factory=dict)


class ExtractedEventArgument(BaseModel):
    role: str
    mention_text: str
    mention_start_char_in_chunk: int
    mention_end_char_in_chunk: int
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ExtractedEventFrame(BaseModel):
    predicate: str
    event_type: str | None = None
    arguments: list[ExtractedEventArgument] = Field(default_factory=list)
    evidence_start_char_in_chunk: int
    evidence_end_char_in_chunk: int
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ExtractedClaim(BaseModel):
    text: str
    subject_entity_names: list[str] = Field(default_factory=list)
    evidence_start_char_in_chunk: int
    evidence_end_char_in_chunk: int
    confidence: float = Field(ge=0.0, le=1.0)
    needs_verification: bool = False


class ExtractedEvent(BaseModel):
    event_type: str
    description: str
    participant_entity_names: list[str] = Field(default_factory=list)
    evidence_start_char_in_chunk: int
    evidence_end_char_in_chunk: int
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    source_entity_name: str
    target_entity_name: str
    relation_type: str | None = None
    relation_span: str | None = None
    evidence_start_char_in_chunk: int
    evidence_end_char_in_chunk: int
    confidence: float = Field(ge=0.0, le=1.0)
    direction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMGraphExtraction(BaseModel):
    raw_mentions: list[ExtractedRawMention] = Field(default_factory=list)
    event_frames: list[ExtractedEventFrame] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    mentions: list[ExtractedMention] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    events: list[ExtractedEvent] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
