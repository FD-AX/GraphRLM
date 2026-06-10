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
| Dense top-5 (no graph) | 0.586 | 0.210 | — | — |
| Graph navigator | 0.623 | 0.298 | — | — |
| Graph + interaction profile | 0.626 | 0.298 | — | — |
| Graph-RLM discovery* | 0.816 | 0.628 | 0.615 | 0.516 |

\* Graph-RLM snapshot covers 277/500 cases (all 2-hop done, 3/4-hop tail in
progress) at ~28k tokens (~$0.012) per case; the table will be refreshed when
the run completes.

Paired per-case comparisons (sign tests): graph vs dense +8.8 pp complete
coverage (52W/8L, p << 0.001); graph-RLM vs graph navigator +17.5 pp
(65W/19L). The gap-closing effect grows with depth: on 4-hop cases the local
graph reaches 0.06 complete coverage while the discovery loop reaches ~0.47.

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
