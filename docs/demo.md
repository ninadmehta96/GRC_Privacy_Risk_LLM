# Demo Guide (3–7 minutes)

This repo demonstrates **reliable, guardrailed LLM behavior** for GRC/Privacy use-cases, with an emphasis on **evaluation that does not require running a model**.

The public demo is CPU-only and runs in GitHub Actions.

---

## 0) 15-second setup (optional)

```bash
git clone git@github.com:ninadmehta96/GRC_Privacy_Risk_LLM.git
cd GRC_Privacy_Risk_LLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

---

## 1) 60-second demo (the fastest “proof”)

```bash
make ci
```

What happens:
1. `ruff` runs static checks
2. `pytest` runs unit tests
3. `eval_offline_public.py` performs an **offline strict audit** against `data/public_samples/*`
4. JSONL reports are written to `reports/`

Open a report:
```bash
ls -lah reports/
sed -n '1,40p' reports/public_risk_narrative_test.jsonl
```

---

## 2) The core idea (show these files)

### A) The strict contract (what the model is NOT allowed to invent)
Open:
- `unit_testing/strict_rules.py`

Look for rules that flag:
- roles/teams/vendors
- numbers and timelines
- cadence (“monthly”, “quarterly”)
- legal/regulatory claims
- data classes (“credentials”, “SSN”, “personal data”, etc.)

### B) The offline evaluator (CPU-only, deterministic)
Open:
- `unit_testing/eval_offline_public.py`

Key point:
- It evaluates **reference outputs stored in the dataset** (not model-generated outputs).
- This keeps CI fast and reproducible while still enforcing the behavior contract.

### C) The public synthetic datasets
Open:
- `data/public_samples/risk_narrative_v2_devtest.jsonl`

Each row contains:
- an input “evidence” block
- a reference assistant output
- a split label (dev/test)
- metadata (id, etc.)

---

## 3) A clean 3-minute talk track

**Problem:** In GRC/Privacy, hallucinations are worse than silence.  
**Approach:** Define a strict contract + enforce it with tests and evaluators.  
**Engineering:** Separate lightweight evaluation (public CI) from heavyweight inference (private GPU workflows).  
**Outcome:** A repo that proves reliability practices without shipping model weights.

---

## 4) Optional: show GitHub Actions artifact (30 seconds)

On GitHub:
- Actions → **CI (Public)** → latest run
- Download artifact: `offline-public-reports`

This proves the workflow is truly CPU-only and reproducible.

---

## 5) Optional “deep dive” extensions (if they ask)

- Expand `strict_rules.py` as policy evolves
- Add more categories (e.g., vendor names, retention periods, data residency)
- Add property-based tests to generate adversarial examples
- Add a second evaluator that runs inference (private GPU CI)
