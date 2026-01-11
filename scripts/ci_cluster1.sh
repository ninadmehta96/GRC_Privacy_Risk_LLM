#!/usr/bin/env bash
set -euo pipefail

# Public-safe CI wrapper:
# - If adapters exist locally, run full gate steps (smoke + inference evals)
# - Otherwise, fall back to CPU-only public gate (lint/tests/offline eval)

PYTHON_BIN="${PYTHON_BIN:-python}"
REPORTS_DIR="${REPORTS_DIR:-reports}"

POLICY_LORA="${POLICY_LORA:-adapters/mistral7b-cluster1-policy-refactor-lora-v2-strict}"
RISK_LORA="${RISK_LORA:-adapters/mistral7b-cluster1-risk-narrative-lora-v2-strict}"
SMOKE_DATA="${SMOKE_DATA:-data/smoke/cluster1_smoke_inputs.jsonl}"

if [[ -d "$POLICY_LORA" && -d "$RISK_LORA" ]]; then
  echo "[ci_cluster1] Adapters found. Running full local checks..."
  mkdir -p "$REPORTS_DIR"

  # Smoke gate (inference)
  $PYTHON_BIN -m unit_testing.smoke_gate --smoke-data "$SMOKE_DATA"     --policy-lora "$POLICY_LORA" --risk-lora "$RISK_LORA"     --report "$REPORTS_DIR/smoke_report.jsonl"

  echo "[ci_cluster1] Full local checks complete."
else
  echo "[ci_cluster1] Adapters not found (expected in public repo)."
  echo "[ci_cluster1] Falling back to public CPU-only checks: make ci-lite"
  make ci-lite
fi
