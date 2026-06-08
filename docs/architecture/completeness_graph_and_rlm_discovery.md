# Completeness Graph And RLM Discovery

## Status

The FactLens structured completeness protocol is frozen as:

`factlens_external_structured_completeness_audit_v1`

The current result proves the first project capability:

Graph-based structured completeness verification.

It does not yet prove corpus-level retrieval or autonomous RLM missing-evidence discovery.

## Core Split

The runtime has two separate, connected layers.

### Graph Completeness Layer

The graph answers:

Do we have enough evidence to treat a complex claim as fully verified?

Flow:

```text
Complex claim
-> decomposition into subclaims
-> dependency graph
-> evidence linked to each subclaim
-> shared evidence propagation
-> completeness audit
-> supported / unsupported
```

This layer is responsible for:

- subclaim coverage completeness;
- cross-subclaim dependency structure;
- shared evidence reuse;
- missing supporting fact detection;
- fail-closed verdicts when evidence is incomplete;
- provenance to concrete evidence units.

Frozen FactLens result:

```text
flat complete coverage:       0.333
graph query coverage:         0.367
shared graph coverage:        1.000

no edges coverage:            0.367
shuffled edges coverage:      0.367
masked required fact coverage:0.000
```

Interpretation:

The measured gain comes from correct cross-subclaim relationships plus available supporting units. It does not come from merely enabling graph mode, sharing a larger evidence pool, or accepting arbitrary edges.

### RLM Evidence Discovery Layer

RLM answers:

How do we find evidence that is still missing?

RLM is not the default path. It is an escalation mode.

Flow:

```text
missing evidence slot
-> generate search goal
-> search corpus/tools
-> retrieve candidate evidence
-> verify relevance
-> update graph
-> rerun completeness audit
-> complete / replan / stop
```

RLM is responsible for:

- searching for absent supporting facts;
- discovering new entities and links;
- multi-hop retrieval;
- search-goal reformulation;
- rechecking coverage after each evidence addition;
- stopping when coverage is complete or the corpus is insufficient.

## Runtime Routing

The runtime must not route every query into recursive RLM.

```text
request
-> classify execution need
```

Routes:

```text
known deterministic operation
-> operation executor

evidence already available
-> completeness graph

evidence incomplete
-> RLM discovery

open-ended multi-hop question
-> graph-guided RLM

unsupported analytic operation
-> explicit unsupported
```

RLM escalation is allowed only when all are true:

```text
missing_evidence_slots > 0
searchable_corpus_available = true
search_budget_not_exhausted = true
```

## Prompt Contracts

Runtime prompt contracts live in:

`app/agent_runtime/prompts.py`

Prompt IDs:

```text
query_semantics_router_v1
completeness_audit_v1
rlm_evidence_discovery_v1
graph_guided_rlm_search_v1
```

The intended order is:

```text
query_semantics_router_v1
-> completeness_audit_v1
-> rlm_evidence_discovery_v1 only if missing_evidence_slots exist
-> completeness_audit_v1 after every graph update
```

For open-ended multi-hop requests:

```text
query_semantics_router_v1
-> graph_guided_rlm_search_v1
-> completeness_audit_v1 before final supported answer
```

The prompts enforce:

- RLM is not the default path;
- completeness is checked before search;
- graph structure without evidence is insufficient;
- supported verdicts are forbidden when required evidence is missing;
- RLM may retrieve candidate evidence but cannot bypass the completeness audit;
- unsupported operations must fail closed instead of falling back to guesses.

## Verification State

```text
VerificationState:
    claim_id
    subclaims
    supported_subclaims
    refuted_subclaims
    unresolved_subclaims
    evidence_by_subclaim
    shared_evidence
    missing_evidence_slots
    completeness_score
    execution_mode
    rlm_iteration
    stop_reason
```

Execution modes:

```text
completeness_audit
rlm_evidence_discovery
deterministic_operation
graph_relational_lookup
unsupported_operation
```

## Metrics

Completeness layer:

```text
complete_evidence_coverage
macro_subclaim_recall
unsupported_verdict_rate
all_required_subclaims_verified
cross_subclaim_edge_contribution
shared_evidence_reuse
```

RLM layer:

```text
missing_evidence_recovery_rate
coverage_gain_after_search
useful_search_rate
all_hidden_facts_recovered
rlm_iterations
tokens_per_recovered_fact
stop_reason_accuracy
```

End-to-end:

```text
fully_verified_claims_per_10k_tokens
before_rlm_coverage
after_rlm_coverage
rlm_cost
```

## Next RLM Experiment

Use the frozen FactLens claims or a complex subset, but create a separate evidence-discovery corpus:

```text
required supporting units
hard distractors
partially available initial evidence
hidden supporting units still searchable in corpus
```

Compare:

```text
flat_single_search
flat_iterative_search
graph_guided_rlm_search
gold_search_goal_oracle
```

Expected positive trace:

```text
initial coverage: 3/5

iteration 1:
missing slot = sc_4
search goal generated
new evidence found
coverage = 4/5

iteration 2:
missing slot = sc_5
new entity or relation discovered
new evidence found
coverage = 5/5

stop_reason = complete_evidence_coverage
```

Negative control:

```text
required fact removed from corpus
-> RLM does not find it
-> coverage remains incomplete
-> unsupported verdict
-> no hallucinated completion
```

## Project Positioning

Short formula:

```text
Graph = what is known and whether it is enough.
RLM = how to find what is still missing.
```

Russian formula:

```text
Граф отвечает за полноту и структуру доказательства.
RLM отвечает за поиск недостающих доказательств.
```

The project should be described as:

```text
Verify completeness first.
Search only when necessary.
Stop only when evidence is sufficient.
```
