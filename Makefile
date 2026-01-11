# Cluster1_PUBLIC Makefile
#
# Public (no GPU / no adapters):
#   make ci
#
# Full local (requires adapters + ML deps):
#   make ci-full
#   make smoke

PYTHON_BIN ?= python
REPORTS_DIR ?= reports

# Public sample datasets (included in this public repo)
PUBLIC_POLICY_DEVTEST ?= data/public_samples/policy_refactor_v2_devtest.jsonl
PUBLIC_RISK_DEVTEST   ?= data/public_samples/risk_narrative_v2_devtest.jsonl
PUBLIC_CTRL_DEVTEST   ?= data/public_samples/control_narrative_v1_devtest.jsonl
SPLIT ?= test

# Full local defaults (NOT included in this public repo)
POLICY_LORA ?= adapters/mistral7b-cluster1-policy-refactor-lora-v2-strict
RISK_LORA   ?= adapters/mistral7b-cluster1-risk-narrative-lora-v2-strict
SMOKE_DATA  ?= data/smoke/cluster1_smoke_inputs.jsonl

.PHONY: help lint test public-eval ci-lite ci ci-full smoke eval-policy eval-risk eval-control

help:
	@echo "Targets:"
	@echo "  make ci       - Public CI: lint + tests + offline strict audit (no adapters)"
	@echo "  make ci-full  - Full local gate (requires adapters + ML deps)"
	@echo "  make smoke    - GPU smoke gate (requires adapters + ML deps)"
	@echo ""
	@echo "Notes:"
	@echo "  Public CI uses data/public_samples/* and does not run model inference."

lint:
	@ruff --version
	@ruff check . --select F,E9

test:
	@pytest -q

public-eval:
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON_BIN) -m unit_testing.eval_offline_public \
	  --data $(PUBLIC_POLICY_DEVTEST) --split $(SPLIT) \
	  --report $(REPORTS_DIR)/public_policy_refactor_$(SPLIT).jsonl
	$(PYTHON_BIN) -m unit_testing.eval_offline_public \
	  --data $(PUBLIC_RISK_DEVTEST) --split $(SPLIT) \
	  --report $(REPORTS_DIR)/public_risk_narrative_$(SPLIT).jsonl
	$(PYTHON_BIN) -m unit_testing.eval_offline_public \
	  --data $(PUBLIC_CTRL_DEVTEST) --split $(SPLIT) \
	  --report $(REPORTS_DIR)/public_control_narrative_$(SPLIT).jsonl

ci-lite: lint test public-eval

# Public default
ci: ci-lite

# Full local (script will fall back to ci-lite if adapters are missing)
ci-full:
	PYTHON_BIN="$(PYTHON_BIN)" POLICY_LORA="$(POLICY_LORA)" RISK_LORA="$(RISK_LORA)" \
	REPORTS_DIR="$(REPORTS_DIR)" SMOKE_DATA="$(SMOKE_DATA)" \
	bash scripts/ci_cluster1.sh

# Convenience targets (for full local mode)
smoke:
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON_BIN) -m unit_testing.smoke_gate --smoke-data $(SMOKE_DATA) \
	  --policy-lora $(POLICY_LORA) --risk-lora $(RISK_LORA) \
	  --report $(REPORTS_DIR)/smoke_report.jsonl

eval-policy:
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON_BIN) -m unit_testing.eval_cluster1_strict \
	  --task-type policy_refactor --data $(PUBLIC_POLICY_DEVTEST) --split $(SPLIT) \
	  --lora-dir $(POLICY_LORA) --report $(REPORTS_DIR)/eval_policy_$(SPLIT).jsonl

eval-risk:
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON_BIN) -m unit_testing.eval_cluster1_strict \
	  --task-type risk_narrative --data $(PUBLIC_RISK_DEVTEST) --split $(SPLIT) \
	  --lora-dir $(RISK_LORA) --report $(REPORTS_DIR)/eval_risk_$(SPLIT).jsonl

eval-control:
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON_BIN) -m unit_testing.eval_cluster1_strict \
	  --task-type control_narrative --data $(PUBLIC_CTRL_DEVTEST) --split $(SPLIT) \
	  --lora-dir $(RISK_LORA) --report $(REPORTS_DIR)/eval_control_$(SPLIT).jsonl
