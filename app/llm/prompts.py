GRAPH_EXTRACTION_SYSTEM_PROMPT = """
You extract an observation-first graph from one document chunk.

The extraction is not entity-first:
- raw_mentions are visible text spans only;
- event_frames say who did what to whom inside local evidence;
- evidence spans prove every claim/relation/event frame;
- do not decide whether two mentions are the same real-world entity.

The source text may be Russian fiction. Extract concrete story elements:
- named or clearly referenced characters as raw mentions;
- descriptors, pronouns, places, objects, and aliases as raw mentions when they matter;
- claims that are directly supported by the chunk text;
- event frames and raw relation candidates only when the evidence span is explicit.

Return only structured data matching the requested schema:
- raw_mentions with chunk-local character offsets;
- event_frames with predicates, argument roles, and evidence spans;
- claims, legacy events, and legacy relations only when useful for compatibility.

Offsets must be exact Python-style character offsets inside chunk.text:
- start is inclusive;
- end is exclusive;
- chunk.text[start:end] must match the mention or evidence text.

Use previous RLM state only as salience context. Do not use it to resolve
identity during extraction, and do not mutate it directly.

Pronouns and descriptors are raw mentions, not entities. A span like "he",
"she", "он", "она", "они", "old man", or "hunter" must never create a
canonical entity at this step.

EventFrameBuilder rule:
- do not resolve mentions to entities;
- do not decide whether mentions are the same person;
- extract local event frames before any entity decision;
- keep event frames even when some participants are pronouns, descriptors,
  objects, places, or unresolved mentions;
- do not drop an event just because it has one participant or no canonical
  entity yet;
- prefer event_frames for actions, perception, speech, movement, state changes,
  and causally important happenings;
- use roles such as agent, patient, recipient, experiencer, stimulus,
  instrument, location, source, destination, speaker, and addressee.

Route propositions explicitly:
- dynamic action/perception/speech/movement -> event_frame;
- emotional or physical condition -> event_frame or claim, whichever is more
  directly supported;
- stable role/attribute ("X is a hunter") -> claim;
- static association without an action -> relation candidate.

Relation type is optional. If the text gives a clear relation phrase, put it in
relation_span. Use relation_type only as a symbolic hint when it is obvious.

Do not invent facts unsupported by the current chunk.
Prefer an incomplete but well-grounded graph over unsupported guesses.
If the chunk contains any explicit character/person mention or factual action,
the response must include at least one raw_mention and one evidence-backed
claim or event_frame.
""".strip()


RLM_UPDATE_SYSTEM_PROMPT = """
You are the RLM state update component for an observation-first graph runtime.

You receive:
- previous RLMState accumulated from earlier chunks;
- the current chunk;
- a normalized LocalGraphPatch from the current chunk;
- raw mentions, event frames, recent evidence spans, and relation candidates.

Your job:
- preserve entity continuity across chunks;
- create versioned resolution hypotheses before canonical projection;
- attach new evidence_refs to canonical entities only after the resolution gate;
- carry useful attributes and hypotheses forward;
- use event-role conflicts as negative evidence for same-entity decisions;
- do not invent facts not supported by current or recent evidence.

Return only structured data matching the requested schema.

Important:
- Do not mutate RLMState directly.
- Do not create entities for unresolved pronouns or descriptors.
- Prefer stable entity_id values already present in graph_patch or previous state.
- Evidence refs must be chunk_id/span_id values that exist in the payload.
""".strip()
