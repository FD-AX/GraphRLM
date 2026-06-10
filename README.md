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

Equal retrieval budget (top-5 paragraphs) for all non-RLM arms. Latent graph
links and query encodings come from the same BERT-family encoder
(multilingual MiniLM); the graph-RLM arm adds a gpt-5-mini discovery
controller. Full per-case data, by-hop breakdowns, and paired significance
tests live in [docs/benchmarks/musique_completeness/](docs/benchmarks/musique_completeness/).

| Arm | Evidence recall | Complete coverage | Answer F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| Keyword (lexical baseline) | 0.375 | 0.068 | — | — |
| Cross-encoder, exhaustive matching | 0.600 | 0.206 | — | — |
| MDR-style re-querying (text concat) | 0.538 | 0.208 | — | — |
| MDR-style re-querying (latent + beam) | 0.538 | 0.220 | — | — |
| Dense top-5 (no graph) | 0.586 | 0.210 | — | — |
| Graph navigator | 0.623 | 0.298 | — | — |
| Graph + interaction profile | 0.626 | 0.298 | — | — |
| Graph-RLM discovery (v1 budget) | 0.771 | 0.494 | 0.573 | 0.470 |
| Graph-RLM discovery (v2 budget) | 0.872 | 0.700 | 0.639 | 0.522 |

All arms cover the full 500 cases with zero errors. The v1 graph-RLM arm
spends ~32k tokens (~$0.013, gpt-5-mini) per case (~$6 per 500-case run);
v2 spends ~68k (~$0.025, ~$12 per run).

### Discovery budget ablation (v1 vs v2)

The only differences between the two graph-RLM rows are the controller
budget and two completeness guards; the graph, encoder, controller model
and cases are identical. v1: 10 model calls, depth 5, 6 expansions. v2: 16
calls, depth 8, 12 expansions, hop-chain decomposition required before
answering, and answer decisions backed by fewer than two evidence
paragraphs are redirected toward the stated evidence gap. Full data:
[docs/benchmarks/musique_completeness/rlm_budget_comparison.json](docs/benchmarks/musique_completeness/rlm_budget_comparison.json).

| Complete coverage | v1 | v2 | Evidence recall | v1 | v2 |
| --- | ---: | ---: | --- | ---: | ---: |
| 2-hop | 0.740 | 0.885 | 2-hop | 0.868 | 0.940 |
| 3-hop | 0.280 | 0.460 | 3-hop | 0.664 | 0.771 |
| 4-hop | 0.380 | 0.693 | 4-hop | 0.748 | 0.882 |
| all | 0.494 | 0.700 | all | 0.771 | 0.872 |

Paired on the same 500 cases: +20.6 pp complete coverage (131W/28L),
+10.1 pp recall (160W/41L), +6.6 pp answer F1 (111W/65L). The gain
concentrates exactly where v1 diagnostics located the losses: 208 v1 cases
were cut mid-collection by the call budget and 87 3-hop cases answered
prematurely. Deeper discovery converts budget into completeness at roughly
+1 pp complete coverage per ~1.7k tokens per case.

Three graph-free retrieval families hit the same completeness plateau
(~0.21-0.22 complete coverage): single-shot dense retrieval, exhaustive
cross-encoder matching (`cross-encoder/ms-marco-MiniLM-L-6-v2`, every
(question, paragraph) pair scored jointly - the upper bound of pairwise
matching), and MDR-style iterative re-querying (the Multi-hop Dense
Retrieval shape, untrained, same encoder as ours) in two variants: naive
text concatenation, and a steelmanned version with latent query pooling
plus beam search that avoids encoder-window truncation. Cross-encoder vs
dense: 46W/48L paired, a wash. Steelmanned MDR-style vs dense: 29W/24L on
coverage, also a wash. Graph connectivity breaks the plateau: the navigator
beats exhaustive cross-encoder matching +9.2 pp complete coverage
(81W/35L), steelmanned MDR-style re-querying +7.8 pp (63W/24L), and
graph-RLM beats the cross-encoder +28.8 pp (179W/35L).

Paired per-case comparisons (sign tests, all p << 0.001 except the
interaction-profile delta): graph vs dense +8.8 pp complete coverage
(52W/8L); graph-RLM vs graph navigator +19.6 pp (129W/31L); graph-RLM vs
dense +28.4 pp (163W/21L). The gap-closing effect grows with depth: on 4-hop
cases the local graph reaches 0.06 complete coverage while the discovery
loop reaches 0.38 (recall 0.49 -> 0.75).

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

Local arms (no API key needed):

```bash
python scripts/run_musique_completeness.py --arms keyword,dense,graph,graph_active
```

Full matrix with the GPT discovery controller (requires `OPENAI_API_KEY` in
`.env.local`):

```bash
python scripts/run_musique_completeness.py --arms rlm --rlm-model gpt-5-mini
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
