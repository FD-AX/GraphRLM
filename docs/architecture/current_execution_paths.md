# Current Execution Paths

This file keeps durable execution-path classification. Long forensic notes live in artifacts and tests; this document is only for stable rules that should not drift.

## Research-Eligible

### `graph_rlm_semantic_graph`

Status: research-eligible.

Required evidence:

- `semantic_graph_used = true`
- `legacy_hash_frontier_used = false`
- `graph_builder_class = OOLONGSemanticGraphBuilder`
- `encoder_class = TransformerSemanticEncoder`
- model calls have response IDs
- trace token totals match record totals
- score revisions preserve `model_rerun = false` when rescoring an existing output

Validated artifact:

- `artifacts/oolong_synth_semantic_graph_real_113010009`

## Surrogate / Engineering-Only

### `graph_rlm_hash_frontier`

Status: surrogate.

Reason: builds a retrieval graph over text records with hashing-based similarity. It is useful for command-loop and frontier validation, but it is not the production semantic graph path.

### `graph_rlm_spam_worker_materialized`

Status: surrogate.

Reason: materializes task-specific spam/ham labels and aggregates. Useful for ablation, but not a general semantic graph with entity/event/evidence projections.

## Invalid Artifacts

### `oolong_synth_spam_8k_graph_rlm_gpt5mini_worker_task113010009_v2`

Status: invalid.

Reason: `manifest_runtime_worker_mismatch`.

Use only as a negative artifact-validation fixture.

## Registry

Machine-readable status lives in:

```text
artifacts/artifact_registry.json
```
