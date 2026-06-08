from app.agent_runtime.prompts import (
    AGENT_PROMPT_REGISTRY,
    build_agent_prompt,
    get_agent_prompt,
)


def test_agent_prompt_registry_contains_runtime_modes() -> None:
    assert set(AGENT_PROMPT_REGISTRY) == {
        "query_semantics_router_v1",
        "completeness_audit_v1",
        "rlm_evidence_discovery_v1",
        "graph_guided_rlm_search_v1",
    }


def test_router_prompt_makes_rlm_an_escalation_not_default() -> None:
    prompt = get_agent_prompt("query_semantics_router_v1").system_prompt

    assert "Recursive RLM is not the default" in prompt
    assert "Verify completeness before searching" in prompt
    assert "unsupported_operation" in prompt
    assert "allow_rlm_escalation" in prompt


def test_completeness_prompt_fails_closed_when_evidence_is_missing() -> None:
    prompt = get_agent_prompt("completeness_audit_v1").system_prompt

    assert "Your job is not to search" in prompt
    assert "Graph structure without evidence is not sufficient" in prompt
    assert "If complete_evidence_coverage is false, do not emit a supported verdict" in prompt
    assert "missing_evidence_slots" in prompt


def test_rlm_prompt_only_finds_missing_evidence_and_returns_to_audit() -> None:
    prompt = get_agent_prompt("rlm_evidence_discovery_v1").system_prompt

    assert "only after the completeness auditor reports missing_evidence_slots" in prompt
    assert "Do not search for facts that are already covered" in prompt
    assert "Ask the completeness auditor to recheck coverage" in prompt
    assert "corpus_insufficient" in prompt


def test_build_agent_prompt_appends_task_payload() -> None:
    prompt = build_agent_prompt(
        "graph_guided_rlm_search_v1",
        task_payload='{"claim_id":"claim_1"}',
    )

    assert "Task payload:" in prompt
    assert '"claim_id":"claim_1"' in prompt
