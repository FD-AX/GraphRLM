# GraphRLM

GraphRLM is a typed research runtime for evidence-completeness auditing and
targeted recursive evidence discovery.

The repository keeps source code, tests, infrastructure definitions, and small
entrypoint scripts under version control. Generated benchmark artifacts, model
caches, logs, local secrets, and source documents stay local.

## Setup

```bash
python -m pip install -r requirements.txt
```

For real model controller runs, copy `.env.example` to `.env.local` and set
`OPENAI_API_KEY`. Do not commit `.env.local`.

## Run Metrics

PowerShell:

```powershell
.\scripts\run_metrics.ps1 -FactLensRepo C:\tmp\factlens_official_repo
```

Bash:

```bash
FACTLENS_REPO=/tmp/factlens_official_repo ./scripts/run_metrics.sh
```

To include real GPT controller arms:

```powershell
.\scripts\run_metrics.ps1 -FactLensRepo C:\tmp\factlens_official_repo -IncludeModelControllers
```

```bash
INCLUDE_MODEL_CONTROLLERS=1 OPENAI_API_KEY=... ./scripts/run_metrics.sh
```

Both wrappers run:

1. FactLens structured completeness matrix.
2. FactLens RLM discovery.
3. Metric invariant checks via `python -m scripts.check_metrics`.

## Tests

```bash
python -m pytest -q
```
