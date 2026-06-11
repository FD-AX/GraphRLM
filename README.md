# GraphRLM

GraphRLM is a typed research runtime for evidence-completeness auditing and
targeted recursive evidence discovery.

LLMs can produce a plausible final verdict while the supporting evidence is
still incomplete. GraphRLM separates those two questions:

> Graph verifies completeness. RLM closes evidence gaps.

The graph layer decomposes a claim into required subclaims, shared evidence,
cross-subclaim dependencies, and missing evidence slots. The RLM discovery loop
is activated only when the completeness audit reports a real gap.

## Key Results

### MuSiQue evidence completeness (500 validation cases, stratified 2/3/4-hop)

**Central finding: without an LLM controller, graph connectivity is the
only training-free mechanism that breaks the evidence-completeness
plateau; with a budget-rich LLM controller in a 20-candidate pool, the
controller itself breaks the plateau, and typed graph state buys ~1.8x
token efficiency and better answer precision rather than additional
completeness.** Every graph-free retrieval family we tested - lexical
overlap, single-shot dense, exhaustive cross-encoder matching, and
MDR-style iterative re-querying (including an anchor-weight sweep over its
whole untrained family) - lands on ~0.21 complete coverage at a
five-paragraph budget; the same encoder navigating typed graph links
reaches 0.30 with zero model calls. The gap-driven discovery contour
reaches 0.70 on the full 500 cases. For context, the reported supervised
[Beam Retrieval](https://arxiv.org/abs/2308.08973) result is 0.774 under
its published protocol; that external comparison is directional only,
because the evaluation protocols are not identical.

**Evaluation target.** Each MuSiQue case contains a question, 20 candidate
paragraphs, and a gold multi-hop evidence chain. *Evidence recall* is the
fraction of gold evidence paragraphs retrieved. *Complete coverage* is 1
only when every gold evidence paragraph of the case is present in the
retrieved evidence set, 0 otherwise. *Answer F1 / EM* are computed from the
final generated answer against the gold answer and its aliases. Latent
graph links and query encodings come from the same BERT-family encoder
(multilingual MiniLM). Full per-case data, by-hop breakdowns, and paired
significance tests live in
[docs/benchmarks/musique_completeness/](docs/benchmarks/musique_completeness/).

**Local retrieval comparison** - every arm returns at most five evidence
paragraphs:

| Arm | Evidence recall | Complete coverage |
| --- | ---: | ---: |
| Keyword (lexical baseline) | 0.375 | 0.068 |
| Cross-encoder, exhaustive matching | 0.600 | 0.206 |
| MDR-style re-querying (text concat) | 0.538 | 0.208 |
| MDR-style re-querying (latent + beam) | 0.538 | 0.220 |
| Dense top-5 (no graph) | 0.586 | 0.210 |
| Graph navigator | 0.623 | 0.298 |
| Graph + interaction profile | 0.626 | 0.298 |

The interaction profile augments graph traversal scoring with
training-free per-dimension interaction features between the query and
candidate evidence nodes; its null effect here is itself a finding (the
completeness gain comes from connectivity, not from richer pairwise
scoring - see the frontier ablation in
[docs/benchmarks/frontier_ablation/](docs/benchmarks/frontier_ablation/)).

**System-level comparison** - graph-RLM starts from the same retrieval
state but may take additional gap-directed discovery actions within its
controller budget, so its final evidence set is larger (v1: mean 7.4
paragraphs, v2: mean 11.1 of 20). Budget-matched dense rows control for
set size ([dense_budget_curve.json](docs/benchmarks/musique_completeness/dense_budget_curve.json)):

| System | Evidence set size | Evidence recall | Complete coverage | Answer F1 | EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense top-7 (budget control) | 7 | 0.683 | 0.332 | — | — |
| Graph-RLM v1 (10 calls) | ~7.4 | 0.771 | 0.494 | 0.573 | 0.470 |
| Dense top-11 (budget control) | 11 | 0.819 | 0.534 | — | — |
| Graph-RLM v2 (16 calls) | ~11.1 | 0.872 | 0.700 | 0.639 | 0.522 |

At matched evidence-set size the discovery contour stays +16-17 pp complete
coverage above dense retrieval in both budget regimes: the gain comes from
gap-directed discovery, not from returning more paragraphs. All arms cover
the full 500 cases with zero errors; graph-RLM costs ~$0.013 (v1) / ~$0.025
(v2) per case with gpt-5-mini.

**Discovery budget ablation (v1 -> v2).** Identical graph, encoder,
controller model and cases; v2 raises the controller budget (10 -> 16
calls, depth 5 -> 8, expansions 6 -> 12) and adds two completeness guards
(hop-chain decomposition before answering; answers backed by fewer than two
paragraphs are redirected toward the stated evidence gap). Full data:
[rlm_budget_comparison.json](docs/benchmarks/musique_completeness/rlm_budget_comparison.json).

| Complete coverage | v1 | v2 | Evidence recall | v1 | v2 |
| --- | ---: | ---: | --- | ---: | ---: |
| 2-hop | 0.740 | 0.885 | 2-hop | 0.868 | 0.940 |
| 3-hop | 0.280 | 0.460 | 3-hop | 0.664 | 0.771 |
| 4-hop | 0.380 | 0.693 | 4-hop | 0.748 | 0.882 |
| all | 0.494 | 0.700 | all | 0.771 | 0.872 |

Key paired comparisons (sign tests on shared cases, p << 0.001):

- graph navigator vs dense: **+8.8 pp** complete coverage;
- graph-RLM v1 vs graph navigator: **+19.6 pp**;
- graph-RLM v2 vs v1: **+20.6 pp**.

**Controller frontier ablation (graph state itself).** A budget-matched
non-graph control (`musique_text_rlm`: same controller, actions, budgets
and guards; frontier from dense search with typed-link metadata stripped)
was run on a stratified 60-case subset. It reaches *higher* completeness
than graph-RLM (0.850 vs 0.733 coverage paired on shared cases, 11L/4W for
graph, marginal at n=60) at **1.8x the token cost** (121k vs 68k per case,
the dense frontier re-presents the whole candidate pool every step), while
graph-RLM answers better (+5 pp F1/EM). Honest reading: in a 20-candidate
distractor pool an LLM controller does not need typed graph state for
completeness - the graph contributes efficiency and answer precision here.
Exhaustive frontier re-presentation does not exist at open-corpus scale,
where the frontier must be structural; the distractor setting therefore
systematically understates the graph's value, and an open-corpus replication
is the next experiment
([text_rlm_control.json](docs/benchmarks/musique_completeness/text_rlm_control.json)).

All remaining pairwise tests, W/L counts, by-hop matrices and the MDR
anchor-weight sweep are reported in
[docs/benchmarks/README.md](docs/benchmarks/README.md).

Scope note: this is the MuSiQue distractor setting (20 candidate paragraphs
per question), not open-corpus retrieval. Within-matrix comparisons are
rigorous; external leaderboard comparisons are directional only.

### FactLens structured completeness

Controlled FactLens experiments currently show:

| Metric | Baseline | GraphRLM contour |
| --- | ---: | ---: |
| FactLens complete evidence coverage | 33.3% | 100% |
| Missing-evidence recovery | 51.7% | 100% |
| Complete after discovery | 0% | 80% |
| False completion | 0% | 0% |

Important scope note: FactLens structured completeness and controlled
RLM-discovery are separate experiments. These numbers are not an official
corpus-level FactLens retrieval leaderboard score.

The repository keeps source code, tests, infrastructure definitions, and small
entrypoint scripts under version control. Generated benchmark artifacts, model
caches, logs, local secrets, and source documents stay local.

## Reproduce

Install:

```bash
python -m pip install -r requirements.txt
```

Test:

```bash
python -m pytest -q
```

Run the benchmark metrics and invariant checks:

PowerShell:

```powershell
.\scripts\run_metrics.ps1 -FactLensRepo C:\tmp\factlens_official_repo
```

Bash:

```bash
FACTLENS_REPO=/tmp/factlens_official_repo ./scripts/run_metrics.sh
```

For real model controller runs, copy `.env.example` to `.env.local` and set
`OPENAI_API_KEY`. Do not commit `.env.local`.

PowerShell with real GPT controller arms:

```powershell
.\scripts\run_metrics.ps1 -FactLensRepo C:\tmp\factlens_official_repo -IncludeModelControllers
```

Bash with real GPT controller arms:

```bash
INCLUDE_MODEL_CONTROLLERS=1 OPENAI_API_KEY=... ./scripts/run_metrics.sh
```

Both wrappers run:

1. FactLens structured completeness matrix.
2. FactLens RLM discovery.
3. Metric invariant checks via `python -m scripts.check_metrics`.

## MuSiQue completeness benchmark

Published runs are reproduced from versioned configs; every run writes a
`run_manifest.json` (commit SHA, config hash, dataset selection, model,
token usage, output paths) next to its records.

Local arms (no API key, deterministic, ~20 min on CPU):

```bash
python scripts/run_musique_completeness.py --config configs/musique/local_arms_published.yaml
```

Graph-RLM rows (require `OPENAI_API_KEY` in `.env.local`; the controller is
nondeterministic, expect small variation; ~$6 / ~$12 and several hours,
shardable via `--shard i/n`):

```bash
python scripts/run_musique_completeness.py --config configs/musique/rlm_v1_published.yaml
python scripts/run_musique_completeness.py --config configs/musique/rlm_v2_published.yaml
```

Export versioned results and feed the Grafana dashboard:

```bash
python scripts/export_musique_results.py \
  --postgres-dsn postgresql://jmlc:jmlc_local@localhost:55432/jmlc
```

## Metrics dashboards

```bash
cd infra/metrics
docker compose up -d
```

Grafana at http://localhost:3300 (admin/admin) provisions the
"MuSiQue Evidence Completeness" and benchmark research dashboards from
[infra/metrics/grafana/dashboards/](infra/metrics/grafana/dashboards/).
