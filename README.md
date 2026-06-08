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
