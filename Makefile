# Public Makefile (CPU-only CI)
# Usage:
#   make ci        # ruff + pytest + offline strict audit on synthetic samples
#   make lint
#   make test
#   make public-eval
PYTHON_BIN ?= python

RUFF ?= ruff

REPORTS_DIR ?= reports

.PHONY: ci lint test public-eval clean-reports

ci: lint test public-eval

lint:
	$(RUFF) --version
	$(RUFF) check .

test:
	$(PYTHON_BIN) -m pytest -q

public-eval:
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON_BIN) -m unit_testing.eval_offline_public \
	  --data data/public_samples/policy_refactor_v2_devtest.jsonl --split test \
	  --report $(REPORTS_DIR)/public_policy_refactor_test.jsonl
	$(PYTHON_BIN) -m unit_testing.eval_offline_public \
	  --data data/public_samples/risk_narrative_v2_devtest.jsonl --split test \
	  --report $(REPORTS_DIR)/public_risk_narrative_test.jsonl

clean-reports:
	rm -rf $(REPORTS_DIR)
