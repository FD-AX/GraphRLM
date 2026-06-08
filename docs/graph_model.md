# Multidimensional Latent Relation Graph

JMLC/RLM uses a graph-first approach:

```text
text
  -> unique entities
  -> evidence spans around entities
  -> relation candidates between entities
  -> BERT-like relation encoding
  -> latent multidimensional graph
```

The primary retrieval unit is not only a chunk. It is an evidence-grounded
relation between entities.

## Core Roles

```text
RLM builds graph skeleton.
BERT-like models build relation geometry.
Neo4j stores topology and evidence links.
Decoder returns evidence.
```

## Stateful RLM Pass

`DocumentRLMWorkflow` is sequential and stateful.

RLM does not process chunks in isolation. Every chunk is interpreted with the
context accumulated from previous chunks:

```text
chunk_0 + empty RLMState -> RLMState_1
chunk_1 + RLMState_1    -> RLMState_2
chunk_2 + RLMState_2    -> RLMState_3
```

Each chunk receives:

```text
current_chunk
previous RLMState
accumulated entities
recent evidence spans
recent relation candidates
unresolved references
```

This is required for:

```text
pronoun resolution
entity continuity
alias merging
relation disambiguation across neighboring chunks
evidence accumulation
```

## Gemma-backed RLM Update

The RLM update node is LLM-backed.

Current flow:

```text
previous RLMState
  + current Chunk
  + LocalGraphPatch
  + recent evidence context
  -> Gemma 3 4B via OpenAI-compatible vLLM endpoint
  -> RLMTransitionExtraction
  -> RLMTransition
  -> apply_rlm_transition()
```

Implementation:

```text
app/agent_graph/document_nodes.py:update_rlm_state_with_llm
app/rlm/schemas.py
app/llm/model_adapter.py:OpenAICompatibleModelAdapter
```

The deterministic merge remains only as a conservative fallback:

```text
Gemma error or empty entity-state update -> deterministic continuity fallback
```

The fallback must not be treated as the main RLM path. It exists so a bad or
empty model response does not break document ingestion.

## Invariants

```text
1. Entity nodes represent unique canonical entities.
2. Pronouns are not Entity nodes.
3. Pronouns are resolution artifacts.
4. Resolved pronoun content attaches to canonical Entity.
5. EvidenceSpan always stores original_text.
6. normalized_text may be used by encoders.
7. RelationCandidate is not a final ontology relation.
8. relation_type is optional symbolic_hint.
9. Latent vectors are not evidence.
10. EvidenceSpan is evidence.
```

## Current MVP Models

Implemented in:

```text
app/core/graph_models.py
```

Main graph skeleton models:

```text
Entity
Mention
EvidenceSpan
ResolvedEntityRef
RelationCandidate
LatentRelationEdge
ProjectionSpace
EntityPair
```

`LocalGraphPatch` now carries:

```text
chunk
entities
mentions
claims
events
relations
evidence_spans
relation_candidates
latent_relation_edges
```

## Neo4j Topology

The MVP materializes:

```text
(:Document)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:HAS_MENTION]->(:Mention)
(:Mention)-[:REFERS_TO]->(:Entity)

(:EvidenceSpan)-[:IN_CHUNK]->(:Chunk)
(:EvidenceSpan)-[:RESOLVES_ENTITY]->(:Entity)

(:Entity)-[:RELATION_SOURCE]->(:RelationCandidate)
(:RelationCandidate)-[:RELATION_TARGET]->(:Entity)
(:RelationCandidate)-[:SUPPORTED_BY]->(:EvidenceSpan)

(:LatentRelation)-[:ENCODES]->(:RelationCandidate)
(:LatentRelation)-[:SUPPORTED_BY]->(:EvidenceSpan)
(:Entity)-[:LATENT_RELATION_SOURCE]->(:LatentRelation)
(:LatentRelation)-[:LATENT_RELATION_TARGET]->(:Entity)
```

## CPU Relation Encoder

Implemented in:

```text
app/encoder/relation_encoder.py
app/runtime/encode_relation_candidates_neo4j.py
```

Default small CPU model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Encoding operation:

```text
RelationCandidate + EvidenceSpan -> LatentRelationEdge
```

Run against Neo4j:

```powershell
python -m app.runtime.encode_relation_candidates_neo4j `
  --document-id demo_doc_001 `
  --limit 100 `
  --device cpu
```

For smoke tests without downloading the model:

```powershell
python -m app.runtime.encode_relation_candidates_neo4j `
  --document-id demo_doc_001 `
  --limit 5 `
  --model-name hashing-fallback `
  --allow-hashing-fallback
```

## Query Direction

Entity-pair relation retrieval should follow:

```text
query
  -> detect entities
  -> find EntityPair / relation candidates
  -> encode query relation intent
  -> compare with latent relation vectors
  -> rerank
  -> return EvidenceSpan
  -> answer only from evidence
```
