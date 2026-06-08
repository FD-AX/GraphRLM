#!/usr/bin/env bash
set -euo pipefail

FACTLENS_REPO="${FACTLENS_REPO:-/tmp/factlens_official_repo}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/metrics_run}"
DISCOVERY_CASES="${DISCOVERY_CASES:-10}"
DISCOVERY_REPEAT="${DISCOVERY_REPEAT:-1}"
MODEL_NAME="${MODEL_NAME:-gpt-5-mini}"
INCLUDE_MODEL_CONTROLLERS="${INCLUDE_MODEL_CONTROLLERS:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="python.exe"
elif command -v py.exe >/dev/null 2>&1; then
  PYTHON_BIN="py.exe -3"
else
  echo "Python was not found in this Bash environment. Install python3 or run scripts/run_metrics.ps1 from PowerShell." >&2
  exit 1
fi

echo
echo "== FactLens structured completeness matrix =="
$PYTHON_BIN -m scripts.run_factlens_external_matrix \
  --factlens-repo "$FACTLENS_REPO" \
  --output-dir "$OUTPUT_DIR/factlens_external_matrix"

discovery_args=(
  -m scripts.run_factlens_rlm_discovery
  --factlens-repo "$FACTLENS_REPO"
  --output-dir "$OUTPUT_DIR/factlens_rlm_discovery"
  --cases "$DISCOVERY_CASES"
  --repeat "$DISCOVERY_REPEAT"
)

if [[ "$INCLUDE_MODEL_CONTROLLERS" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required when INCLUDE_MODEL_CONTROLLERS=1" >&2
    exit 1
  fi
  discovery_args+=(--include-model-controllers --model-name "$MODEL_NAME")
fi

echo
echo "== FactLens RLM discovery =="
$PYTHON_BIN "${discovery_args[@]}"

echo
echo "== Metric invariant checks =="
$PYTHON_BIN -m scripts.check_metrics --artifact-root "$OUTPUT_DIR"
