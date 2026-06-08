from app.agent_graph.chunk_nodes import normalize_graph_patch
from app.agent_graph.chunk_schemas import (
    ExtractedEventArgument,
    ExtractedEventFrame,
    ExtractedRawMention,
    LLMGraphExtraction,
)
from app.core.graph_models import ChunkNode
from app.rlm.state import RLMState


def _chunk(text: str) -> ChunkNode:
    return ChunkNode(
        chunk_id="chunk_obs_1",
        document_id="doc_obs",
        index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
    )


def test_observation_first_extraction_keeps_descriptor_raw() -> None:
    text = "Semyon saw old man. Old man held rifle."
    chunk = _chunk(text)

    semyon_start = text.index("Semyon")
    old_man_start = text.index("old man")
    old_man_second_start = text.index("Old man")

    extraction = LLMGraphExtraction(
        raw_mentions=[
            ExtractedRawMention(
                text="Semyon",
                start_char_in_chunk=semyon_start,
                end_char_in_chunk=semyon_start + len("Semyon"),
                mention_type="named",
            ),
            ExtractedRawMention(
                text="old man",
                start_char_in_chunk=old_man_start,
                end_char_in_chunk=old_man_start + len("old man"),
                mention_type="descriptor",
                semantic_payload={"age_hint": "old", "gender_hint": "male"},
            ),
            ExtractedRawMention(
                text="Old man",
                start_char_in_chunk=old_man_second_start,
                end_char_in_chunk=old_man_second_start + len("Old man"),
                mention_type="descriptor",
                semantic_payload={"age_hint": "old", "gender_hint": "male"},
            ),
        ],
        event_frames=[
            ExtractedEventFrame(
                predicate="saw",
                event_type="perception",
                evidence_start_char_in_chunk=0,
                evidence_end_char_in_chunk=len("Semyon saw old man."),
                arguments=[
                    ExtractedEventArgument(
                        role="subject",
                        mention_text="Semyon",
                        mention_start_char_in_chunk=semyon_start,
                        mention_end_char_in_chunk=semyon_start + len("Semyon"),
                    ),
                    ExtractedEventArgument(
                        role="object",
                        mention_text="old man",
                        mention_start_char_in_chunk=old_man_start,
                        mention_end_char_in_chunk=old_man_start + len("old man"),
                    ),
                ],
            )
        ],
    )

    result = normalize_graph_patch(
        {
            "chunk": chunk,
            "llm_extraction": extraction,
            "rlm_state": RLMState(document_id="doc_obs"),
            "errors": [],
        }
    )
    patch = result["graph_patch"]

    assert [mention.text for mention in patch.raw_mentions] == [
        "Semyon",
        "old man",
        "Old man",
    ]
    assert [entity.canonical_name for entity in patch.entities] == [
        "Semyon",
        "unknown_old_man_1",
    ]
    assert [mention.text for mention in patch.mentions] == [
        "Semyon",
        "old man",
        "Old man",
    ]

    rejected_hypotheses = [
        hypothesis
        for hypothesis in patch.resolution_hypotheses
        if hypothesis.status == "rejected"
    ]
    assert len(rejected_hypotheses) == 1
    assert all(
        "event role conflict" in hypothesis.reason
        for hypothesis in rejected_hypotheses
    )
    assert any(
        hypothesis.candidate_entity_name == "unknown_old_man_1"
        and hypothesis.status == "likely"
        for hypothesis in patch.resolution_hypotheses
    )

    event_frame = patch.event_frames[0]
    roles = {argument.role for argument in event_frame.arguments}
    assert roles == {"agent", "patient"}
    assert event_frame.resolution_status == "complete"
    assert len(patch.relations) == 1
    assert patch.relations[0].source_id != patch.relations[0].target_id


def test_descriptor_without_role_conflict_can_ground_to_anchor() -> None:
    text = "Semyon sat near fire. Old man took rifle."
    chunk = _chunk(text)

    semyon_start = text.index("Semyon")
    old_man_start = text.index("Old man")

    extraction = LLMGraphExtraction(
        raw_mentions=[
            ExtractedRawMention(
                text="Semyon",
                start_char_in_chunk=semyon_start,
                end_char_in_chunk=semyon_start + len("Semyon"),
                mention_type="named",
            ),
            ExtractedRawMention(
                text="Old man",
                start_char_in_chunk=old_man_start,
                end_char_in_chunk=old_man_start + len("Old man"),
                mention_type="descriptor",
                semantic_payload={"age_hint": "old", "gender_hint": "male"},
            ),
        ],
        event_frames=[
            ExtractedEventFrame(
                predicate="sat",
                event_type="posture",
                evidence_start_char_in_chunk=0,
                evidence_end_char_in_chunk=len("Semyon sat near fire."),
                arguments=[
                    ExtractedEventArgument(
                        role="subject",
                        mention_text="Semyon",
                        mention_start_char_in_chunk=semyon_start,
                        mention_end_char_in_chunk=semyon_start + len("Semyon"),
                    ),
                ],
            ),
            ExtractedEventFrame(
                predicate="took",
                event_type="possession",
                evidence_start_char_in_chunk=old_man_start,
                evidence_end_char_in_chunk=len(text),
                arguments=[
                    ExtractedEventArgument(
                        role="subject",
                        mention_text="Old man",
                        mention_start_char_in_chunk=old_man_start,
                        mention_end_char_in_chunk=old_man_start + len("Old man"),
                    ),
                ],
            ),
        ],
    )

    result = normalize_graph_patch(
        {
            "chunk": chunk,
            "llm_extraction": extraction,
            "rlm_state": RLMState(document_id="doc_obs"),
            "errors": [],
        }
    )
    patch = result["graph_patch"]

    assert [entity.canonical_name for entity in patch.entities] == ["Semyon"]
    assert [mention.text for mention in patch.mentions] == ["Semyon", "Old man"]
    assert patch.mentions[0].entity_id == patch.mentions[1].entity_id
    assert any(
        hypothesis.hypothesis_type == "mention_to_known_entity"
        and hypothesis.status == "likely"
        and hypothesis.confidence == 0.70
        for hypothesis in patch.resolution_hypotheses
    )


def test_sentence_start_noise_named_candidate_does_not_create_entity() -> None:
    text = "Вокруг была абсолютная темнота. Я висел в полном вакууме."
    chunk = _chunk(text)

    vokrug_start = text.index("Вокруг")
    ya_start = text.index("Я")

    extraction = LLMGraphExtraction(
        raw_mentions=[
            ExtractedRawMention(
                text="Вокруг",
                start_char_in_chunk=vokrug_start,
                end_char_in_chunk=vokrug_start + len("Вокруг"),
                mention_type="named",
                extractor_source="llm",
                extractor_version="test_extractor_v0",
            ),
            ExtractedRawMention(
                text="Я",
                start_char_in_chunk=ya_start,
                end_char_in_chunk=ya_start + len("Я"),
                mention_type="pronoun",
                extractor_source="llm",
                extractor_version="test_extractor_v0",
            ),
        ]
    )

    result = normalize_graph_patch(
        {
            "chunk": chunk,
            "llm_extraction": extraction,
            "rlm_state": RLMState(document_id="doc_obs"),
            "errors": [],
        }
    )
    patch = result["graph_patch"]

    assert [mention.text for mention in patch.raw_mentions] == ["Вокруг", "Я"]
    assert patch.entities == []
    assert patch.mentions == []
    assert any(
        hypothesis.mention_id == patch.raw_mentions[0].mention_id
        and hypothesis.status == "rejected"
        and hypothesis.mention_kind == "noise"
        and hypothesis.entity_creation_decision == "drop"
        and hypothesis.is_terminal is True
        and hypothesis.authority == "entity_anchor_policy"
        and hypothesis.decision_stage == "anchor_policy"
        and hypothesis.final_entity_id is None
        and hypothesis.reason == "sentence_start_adverbial_noise"
        and hypothesis.resolver_version == "entity_anchor_policy_v0.1"
        for hypothesis in patch.resolution_hypotheses
    )
    assert patch.raw_mentions[0].mention_kind == "noise"
    assert len(patch.terminal_resolutions) == 1
    assert patch.terminal_resolutions[0].decision == "drop"


def test_unresolved_descriptor_does_not_auto_create_entity() -> None:
    text = "A stranger waited near the gate."
    chunk = _chunk(text)

    stranger_start = text.index("stranger")

    extraction = LLMGraphExtraction(
        raw_mentions=[
            ExtractedRawMention(
                text="stranger",
                start_char_in_chunk=stranger_start,
                end_char_in_chunk=stranger_start + len("stranger"),
                mention_type="descriptor",
            ),
        ]
    )

    result = normalize_graph_patch(
        {
            "chunk": chunk,
            "llm_extraction": extraction,
            "rlm_state": RLMState(document_id="doc_obs"),
            "errors": [],
        }
    )
    patch = result["graph_patch"]

    assert [mention.text for mention in patch.raw_mentions] == ["stranger"]
    assert patch.entities == []
    assert patch.mentions == []
    assert any(
        hypothesis.mention_id == patch.raw_mentions[0].mention_id
        and hypothesis.status == "unresolved"
        and hypothesis.entity_creation_decision == "keep_as_mention_only"
        and hypothesis.is_terminal is False
        and hypothesis.final_entity_id is None
        for hypothesis in patch.resolution_hypotheses
    )
    assert patch.terminal_resolutions == []


def test_event_frame_survives_unresolved_participants() -> None:
    text = "He heard steps behind the door."
    chunk = _chunk(text)

    he_start = text.index("He")
    steps_start = text.index("steps")
    door_start = text.index("door")

    extraction = LLMGraphExtraction(
        raw_mentions=[
            ExtractedRawMention(
                text="He",
                start_char_in_chunk=he_start,
                end_char_in_chunk=he_start + len("He"),
                mention_type="pronoun",
            ),
            ExtractedRawMention(
                text="steps",
                start_char_in_chunk=steps_start,
                end_char_in_chunk=steps_start + len("steps"),
                mention_type="object",
            ),
            ExtractedRawMention(
                text="door",
                start_char_in_chunk=door_start,
                end_char_in_chunk=door_start + len("door"),
                mention_type="location",
            ),
        ],
        event_frames=[
            ExtractedEventFrame(
                predicate="heard",
                event_type="perception",
                evidence_start_char_in_chunk=0,
                evidence_end_char_in_chunk=len(text),
                arguments=[
                    ExtractedEventArgument(
                        role="experiencer",
                        mention_text="He",
                        mention_start_char_in_chunk=he_start,
                        mention_end_char_in_chunk=he_start + len("He"),
                    ),
                    ExtractedEventArgument(
                        role="stimulus",
                        mention_text="steps",
                        mention_start_char_in_chunk=steps_start,
                        mention_end_char_in_chunk=steps_start + len("steps"),
                    ),
                    ExtractedEventArgument(
                        role="location",
                        mention_text="door",
                        mention_start_char_in_chunk=door_start,
                        mention_end_char_in_chunk=door_start + len("door"),
                    ),
                ],
            )
        ],
    )

    result = normalize_graph_patch(
        {
            "chunk": chunk,
            "llm_extraction": extraction,
            "rlm_state": RLMState(document_id="doc_obs"),
            "errors": [],
        }
    )
    patch = result["graph_patch"]

    assert patch.entities == []
    assert patch.mentions == []
    assert len(patch.event_frames) == 1
    event_frame = patch.event_frames[0]
    assert event_frame.predicate == "heard"
    assert event_frame.resolution_status == "unresolved"
    assert event_frame.materialization_status == "degraded"
    assert {argument.role for argument in event_frame.arguments} == {
        "experiencer",
        "stimulus",
        "location",
    }
    assert all(argument.entity_id is None for argument in event_frame.arguments)
    by_role = {argument.role: argument for argument in event_frame.arguments}
    assert by_role["experiencer"].grounding_expectation == "entity_expected"
    assert by_role["experiencer"].resolution_status == "unresolved"
    assert by_role["stimulus"].grounding_expectation == "concept_allowed"
    assert by_role["stimulus"].resolution_status == "mention_only"
    assert by_role["location"].grounding_expectation == "concept_allowed"
    assert by_role["location"].resolution_status == "mention_only"


def test_event_frame_arguments_are_repaired_from_evidence_mentions() -> None:
    text = "People fought with swords."
    chunk = _chunk(text)

    people_start = text.index("People")
    swords_start = text.index("swords")

    extraction = LLMGraphExtraction(
        raw_mentions=[
            ExtractedRawMention(
                text="People",
                start_char_in_chunk=people_start,
                end_char_in_chunk=people_start + len("People"),
                mention_type="nominal",
            ),
            ExtractedRawMention(
                text="swords",
                start_char_in_chunk=swords_start,
                end_char_in_chunk=swords_start + len("swords"),
                mention_type="object",
            ),
        ],
        event_frames=[
            ExtractedEventFrame(
                predicate="fought",
                event_type="conflict",
                evidence_start_char_in_chunk=0,
                evidence_end_char_in_chunk=len(text),
                arguments=[],
            )
        ],
    )

    result = normalize_graph_patch(
        {
            "chunk": chunk,
            "llm_extraction": extraction,
            "rlm_state": RLMState(document_id="doc_obs"),
            "errors": [],
        }
    )
    event_frame = result["graph_patch"].event_frames[0]

    assert event_frame.materialization_status == "degraded"
    assert [(argument.role, argument.surface_text) for argument in event_frame.arguments] == [
        ("agent", "People"),
        ("patient", "swords"),
    ]
    assert all(argument.argument_id for argument in event_frame.arguments)
    assert all(
        argument.extractor_version == "event_argument_repair_v0.1"
        for argument in event_frame.arguments
    )
