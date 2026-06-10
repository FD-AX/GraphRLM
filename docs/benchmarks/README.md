# Benchmark results

Versioned benchmark outputs. Raw run artifacts (full traces, model call
logs) stay local under `artifacts/`; this directory keeps the compact,
reproducible result sets.

## musique_completeness/

MuSiQue (validation, stratified 2/3/4-hop) evidence-completeness
comparison under an equal retrieval budget (top-5):

- `summary.json` — per-arm aggregates, by-hop breakdown, paired
  comparisons, run configuration.
- `per_case_scores.json` — per (task, arm) metric rows used to build
  the summary and the Grafana dashboard.

Reproduce: `python scripts/run_musique_completeness.py` (see flags),
then `python scripts/export_musique_results.py`.

## frontier_ablation/

Query-conditioned frontier ablation for the interaction-profile modes
(`hashed_features` vs `contribution`) on the synthetic narrative graph:
`scripts/smoke_query_conditioned_frontier.py`.

## Grafana

```
cd infra/metrics
docker compose up -d
python ../../scripts/export_musique_results.py --postgres-dsn postgresql://jmlc:jmlc_local@localhost:55432/jmlc
```

Grafana: http://localhost:3300 (admin/admin), dashboard
"MuSiQue Evidence Completeness" in the JMLC folder.
