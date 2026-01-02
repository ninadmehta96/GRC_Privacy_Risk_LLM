# Cluster1 Makefile (local developer workflow)
# Usage:
#   make ci
#   make ci-no-audit
#   make eval-policy
#   make eval-risk
#
# Override paths:
#   make ci POLICY_LORA=... RISK_LORA=... SPLIT=test

PYTHON_BIN ?= python

POLICY_LORA ?= adapters/mistral7b-cluster1-policy-refactor-lora-v2-strict
RISK_LORA   ?= adapters/mistral7b-cluster1-risk-narrative-lora-v2-strict

POLICY_DATA ?= data/training_data/v1/grc_policy_refactor_v1_1.jsonl
RISK_DATA   ?= data/training_data/v1/grc_risk_narrative_v1.jsonl

SPLIT ?= test
REPORTS_DIR ?= reports
EVAL_SCRIPT ?= unit_testing/eval_cluster1_strict.py

.PHONY: ci ci-no-audit eval-policy eval-risk clean-reports

ci:
	PYTHON_BIN="$(PYTHON_BIN)" POLICY_LORA="$(POLICY_LORA)" RISK_LORA="$(RISK_LORA)" \
	POLICY_DATA="$(POLICY_DATA)" RISK_DATA="$(RISK_DATA)" SPLIT="$(SPLIT)" \
	REPORTS_DIR="$(REPORTS_DIR)" EVAL_SCRIPT="$(EVAL_SCRIPT)" \
	bash scripts/ci_cluster1.sh

ci-no-audit:
	RUN_NO_AUDIT=1 $(MAKE) ci

eval-policy:
	$(PYTHON_BIN) $(EVAL_SCRIPT) --task policy_refactor --data $(POLICY_DATA) --split $(SPLIT) --lora-dir $(POLICY_LORA) \
	  --report $(REPORTS_DIR)/manual_policy_$(SPLIT).jsonl

eval-risk:
	$(PYTHON_BIN) $(EVAL_SCRIPT) --task risk_narrative --data $(RISK_DATA) --split $(SPLIT) --lora-dir $(RISK_LORA) \
	  --report $(REPORTS_DIR)/manual_risk_$(SPLIT).jsonl

clean-reports:
	rm -rf $(REPORTS_DIR)
