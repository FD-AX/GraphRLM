from app.encoder.relation_encoder import RelationEncoder


def test_relation_encoder_hashing_fallback_smoke() -> None:
    encoder = RelationEncoder(
        model_name="hashing-fallback",
        projection_space_id="test_relation_space",
        allow_hashing_fallback=True,
    )

    encoded = encoder.encode_relation(
        source_name="Олег",
        target_name="Андрей",
        relation_span="был разочарован в способностях",
        evidence_text="Олег был разочарован в способностях Андрея",
        relation_candidate_id="relcand_test",
    )

    assert encoded.projection_space_id == "test_relation_space"
    assert encoded.vector_ref.startswith("vector_")
    assert len(encoded.vector) == 384
    assert any(value != 0 for value in encoded.vector)
