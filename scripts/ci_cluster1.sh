#!/usr/bin/env bash
set -euo pipefail

# Cluster1 CI Gate: strict-eval must be green (0 fails) before you commit/retrain.
#
# Defaults assume you're running from repo root: ~/llm-local/Cluster1
#
# You can override via env vars:
#   POLICY_LORA=... RISK_LORA=... POLICY_DATA=... RISK_DATA=... SPLIT=test REPORTS_DIR=reports
#
# Or pass flags (flags override env vars):
#   ./scripts/ci_cluster1.sh --policy-lora ... --risk-lora ... --policy-data ... --risk-data ... --split test
#
# Optional:
#   RUN_NO_AUDIT=1  -> also run strict-eval with --no-audit and gate on that too (off by default).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Always execute from repo root so relative paths resolve consistently
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT_DEFAULT="${REPO_ROOT}/unit_testing/eval_cluster1_strict.py"

POLICY_LORA_DEFAULT="adapters/mistral7b-cluster1-policy-refactor-lora-v2-strict"
RISK_LORA_DEFAULT="adapters/mistral7b-cluster1-risk-narrative-lora-v2-strict"
POLICY_DATA_DEFAULT="data/training_data/v1/grc_policy_refactor_v1_1.jsonl"
RISK_DATA_DEFAULT="data/training_data/v1/grc_risk_narrative_v1.jsonl"
SPLIT_DEFAULT="test"
REPORTS_DIR_DEFAULT="reports"

POLICY_LORA="${POLICY_LORA:-$POLICY_LORA_DEFAULT}"
RISK_LORA="${RISK_LORA:-$RISK_LORA_DEFAULT}"
POLICY_DATA="${POLICY_DATA:-$POLICY_DATA_DEFAULT}"
RISK_DATA="${RISK_DATA:-$RISK_DATA_DEFAULT}"
SPLIT="${SPLIT:-$SPLIT_DEFAULT}"
REPORTS_DIR="${REPORTS_DIR:-$REPORTS_DIR_DEFAULT}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$EVAL_SCRIPT_DEFAULT}"

RUN_NO_AUDIT="${RUN_NO_AUDIT:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --policy-lora PATH    Policy refactor LoRA directory (default: $POLICY_LORA_DEFAULT)
  --risk-lora PATH      Risk narrative LoRA directory (default: $RISK_LORA_DEFAULT)
  --policy-data PATH    Policy dataset JSONL (default: $POLICY_DATA_DEFAULT)
  --risk-data PATH      Risk dataset JSONL (default: $RISK_DATA_DEFAULT)
  --split NAME          Dataset split (default: $SPLIT_DEFAULT)
  --reports-dir DIR     Reports directory (default: $REPORTS_DIR_DEFAULT)
  --eval-script PATH    Path to eval_cluster1_strict.py
  -h, --help            Show help

Environment variables:
  POLICY_LORA, RISK_LORA, POLICY_DATA, RISK_DATA, SPLIT, REPORTS_DIR, EVAL_SCRIPT, PYTHON_BIN
  RUN_NO_AUDIT=1        Also gate on --no-audit runs

Examples:
  bash scripts/ci_cluster1.sh
  POLICY_LORA=adapters/... RISK_LORA=adapters/... bash scripts/ci_cluster1.sh
  RUN_NO_AUDIT=1 bash scripts/ci_cluster1.sh
EOF
}

# Simple flag parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-lora) POLICY_LORA="$2"; shift 2;;
    --risk-lora) RISK_LORA="$2"; shift 2;;
    --policy-data) POLICY_DATA="$2"; shift 2;;
    --risk-data) RISK_DATA="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --reports-dir) REPORTS_DIR="$2"; shift 2;;
    --eval-script) EVAL_SCRIPT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

cd "$REPO_ROOT"

# Guardrails: fail fast if paths look wrong.
[[ -f "$EVAL_SCRIPT" ]] || { echo "ERROR: eval script not found: $EVAL_SCRIPT"; exit 2; }
[[ -d "$POLICY_LORA" ]] || { echo "ERROR: policy LoRA dir not found: $POLICY_LORA"; exit 2; }
[[ -d "$RISK_LORA" ]] || { echo "ERROR: risk LoRA dir not found: $RISK_LORA"; exit 2; }
[[ -f "$POLICY_DATA" ]] || { echo "ERROR: policy dataset not found: $POLICY_DATA"; exit 2; }
[[ -f "$RISK_DATA" ]] || { echo "ERROR: risk dataset not found: $RISK_DATA"; exit 2; }

mkdir -p "$REPORTS_DIR"

ts="$(date +"%Y%m%d_%H%M%S")"
policy_report="${REPORTS_DIR}/strict_policy_refactor_${SPLIT}_${ts}.jsonl"
risk_report="${REPORTS_DIR}/strict_risk_narrative_${SPLIT}_${ts}.jsonl"
policy_report_no_audit="${REPORTS_DIR}/strict_policy_refactor_${SPLIT}_${ts}_no_audit.jsonl"
risk_report_no_audit="${REPORTS_DIR}/strict_risk_narrative_${SPLIT}_${ts}_no_audit.jsonl"

echo "=== Cluster1 CI Gate ==="
echo "Repo:          $REPO_ROOT"
echo "Eval script:   $EVAL_SCRIPT"
echo "Split:         $SPLIT"
echo "Reports dir:   $REPORTS_DIR"
echo "Policy LoRA:   $POLICY_LORA"
echo "Risk LoRA:     $RISK_LORA"
echo "Policy data:   $POLICY_DATA"
echo "Risk data:     $RISK_DATA"
echo "RUN_NO_AUDIT:  $RUN_NO_AUDIT"
echo

run_eval() {
  local task="$1"
  local data="$2"
  local lora="$3"
  local report="$4"
  local extra_flag="${5:-}"

  echo "--- Running strict eval: $task ${extra_flag:-} ---"
  set +e
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --task "$task" \
    --data "$data" \
    --split "$SPLIT" \
    --lora-dir "$lora" \
    --report "$report" \
    ${extra_flag:-}
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "ERROR: evaluator exited non-zero for $task (rc=$rc)"
    exit $rc
  fi
  echo "Wrote: $report"
  echo
}

count_fails_py='
import json, sys
fn=sys.argv[1]
fails=0
samples=0
by_cat={"numbers":0,"roles":0,"data_classes":0,"cadence":0,"legal":0}
with open(fn,"r",encoding="utf-8") as f:
  for line in f:
    if not line.strip():
      continue
    r=json.loads(line)
    flags=r.get("flags",{}) or {}
    samples += 1
    nums = flags.get("numbers_not_in_input", []) or []
    roles = flags.get("roles_not_in_input", []) or []
    dcls = flags.get("data_classes_not_in_input", []) or []
    cad  = flags.get("cadence_not_in_input", []) or []
    legal= flags.get("legal_claims_not_in_input", []) or []
    bad = bool(nums or roles or dcls or cad or legal)
    if bad:
      fails += 1
      if nums:  by_cat["numbers"] += 1
      if roles: by_cat["roles"] += 1
      if dcls:  by_cat["data_classes"] += 1
      if cad:   by_cat["cadence"] += 1
      if legal: by_cat["legal"] += 1
print(f"Samples: {samples}")
print(f"Failing samples: {fails}")
print("By category (count of samples triggering):")
for k in ["numbers","roles","data_classes","cadence","legal"]:
  print(f"  - {k}: {by_cat[k]}")
sys.exit(0 if fails==0 else 1)
'

gate_report() {
  local label="$1"
  local report="$2"

  echo "--- Gate check: $label ---"
  set +e
  "$PYTHON_BIN" -c "$count_fails_py" "$report"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "❌ CI GATE FAILED for: $label"
    echo "See report: $report"
    echo
    return 1
  fi
  echo "✅ Gate passed for: $label"
  echo
  return 0
}

# 1) Default strict (audit on)
run_eval "policy_refactor" "$POLICY_DATA" "$POLICY_LORA" "$policy_report"
run_eval "risk_narrative"  "$RISK_DATA"   "$RISK_LORA"   "$risk_report"

fails=0
gate_report "policy_refactor (audit)" "$policy_report" || fails=1
gate_report "risk_narrative (audit)" "$risk_report"   || fails=1

# 2) Optional: strict without audit pass
if [[ "$RUN_NO_AUDIT" == "1" ]]; then
  run_eval "policy_refactor" "$POLICY_DATA" "$POLICY_LORA" "$policy_report_no_audit" "--no-audit"
  run_eval "risk_narrative"  "$RISK_DATA"   "$RISK_LORA"   "$risk_report_no_audit"   "--no-audit"
  gate_report "policy_refactor (no-audit)" "$policy_report_no_audit" || fails=1
  gate_report "risk_narrative (no-audit)"  "$risk_report_no_audit"   || fails=1
fi

if [[ $fails -ne 0 ]]; then
  echo "=== RESULT: FAIL ==="
  exit 1
fi

echo "=== RESULT: PASS ==="
echo "All strict gates are green."
