from __future__ import annotations

from app.benchmarks.models import BenchmarkCase
from app.benchmarks.musique.arms import (
    MuSiQueCrossEncoderArm,
    MuSiQueDenseTopKArm,
    MuSiQueGraphNavigatorArm,
    MuSiQueKeywordArm,
    MuSiQueMDRIterativeArm,
)
from app.benchmarks.musique.graph import build_musique_semantic_documents
from app.benchmarks.musique.loader import musique_case_from_record, MuSiQueSource
from app.benchmarks.musique.scorer import MuSiQueCompletenessScorer
from app.benchmarks.runner import run_benchmark_cases
from app.semantic_encoding import (
    EncoderConfig,
    GraphSemanticEncoder,
    HashingSemanticEncoder,
)


def make_record() -> dict:
    return {
        "id": "2hop__test_1",
        "question": "Who founded the company that built the bridge tower?",
        "answer": "Anna Petrova",
        "answer_aliases": ["A. Petrova"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "Bridge Tower",
                "paragraph_text": "The bridge tower was built by Northstar Construction in 1998.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Northstar Construction",
                "paragraph_text": "Northstar Construction was founded by Anna Petrova.",
                "is_supporting": True,
            },
            {
                "idx": 2,
                "title": "River Ferry",
                "paragraph_text": "The river ferry crosses the bay twice a day.",
                "is_supporting": False,
            },
            {
                "idx": 3,
                "title": "City Museum",
                "paragraph_text": "The museum keeps artifacts from the old harbor.",
                "is_supporting": False,
            },
        ],
    }


def make_case() -> BenchmarkCase:
    return musique_case_from_record(make_record(), row_idx=0, source=MuSiQueSource())


def make_encoder() -> GraphSemanticEncoder:
    return GraphSemanticEncoder(
        config=EncoderConfig(projection_dim=256, local_frontier_cap=10),
        backend=HashingSemanticEncoder(dimensions=256),
    )


def test_loader_maps_supporting_paragraphs_to_gold_evidence() -> None:
    case = make_case()
    assert case.benchmark == "musique"
    assert case.gold_evidence_span_ids == ["para_0", "para_1"]
    assert case.expected_hops == 2
    assert "para_3" in case.context


def test_graph_builder_links_paragraphs_via_title_comention() -> None:
    documents = build_musique_semantic_documents(make_case())
    by_owner = {document.owner_id: document for document in documents}
    # para_0 mentions "Northstar Construction" -> shares that entity with para_1
    shared = set(by_owner["para_0"].source_entity_ids) & set(
        by_owner["para_1"].source_entity_ids
    )
    assert shared
    assert not (
        set(by_owner["para_2"].source_entity_ids)
        & set(by_owner["para_1"].source_entity_ids)
    )


def test_arms_emit_evidence_and_scorer_reports_completeness() -> None:
    case = make_case()
    encoder = make_encoder()
    arms = [
        MuSiQueKeywordArm(top_k=2),
        MuSiQueDenseTopKArm(encoder, top_k=2),
        MuSiQueGraphNavigatorArm(encoder, top_k=2, seed_top_k=2, max_depth=2, beam_width=2),
    ]
    records = run_benchmark_cases([case], arms, scorers=[MuSiQueCompletenessScorer()])

    assert len(records) == 3
    for record in records:
        assert record.evidence_span_ids
        names = {score.score_name for score in record.scores}
        assert {"evidence_recall", "complete_evidence_coverage", "evidence_precision"} <= names
        retrieved_count = next(
            score.score_value for score in record.scores if score.score_name == "retrieved_count"
        )
        assert retrieved_count <= 2


def test_mdr_iterative_arm_collects_distinct_paragraphs_per_hop() -> None:
    case = make_case()
    arm = MuSiQueMDRIterativeArm(make_encoder(), top_k=3)
    result = arm.run_case(case)

    assert len(result.evidence_span_ids) == 3
    assert len(set(result.evidence_span_ids)) == 3
    assert len(result.trace[0]["hops"]) == 3


def test_cross_encoder_arm_ranks_by_injected_scores() -> None:
    case = make_case()

    def score_fn(pairs):
        return [
            2.0 if "Northstar" in passage or "bridge tower" in passage.lower() else -1.0
            for _, passage in pairs
        ]

    arm = MuSiQueCrossEncoderArm(top_k=2, score_fn=score_fn)
    result = arm.run_case(case)

    assert set(result.evidence_span_ids) == {"para_0", "para_1"}
    assert result.trace[0]["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_scorer_answer_metrics_use_aliases() -> None:
    from app.benchmarks.models import BenchmarkArmResult

    case = make_case()
    scorer = MuSiQueCompletenessScorer()
    scores = scorer.score(
        case,
        BenchmarkArmResult(prediction="A. Petrova", evidence_span_ids=["para_0", "para_1"]),
    )
    by_name = {score.score_name: score.score_value for score in scores}
    assert by_name["exact_match"] == 1.0
    assert by_name["complete_evidence_coverage"] == 1.0
    assert by_name["evidence_recall"] == 1.0
