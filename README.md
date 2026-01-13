[![CI (Public)](https://github.com/ninadmehta96/GRC_Privacy_Risk_LLM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ninadmehta96/GRC_Privacy_Risk_LLM/actions/workflows/ci.yml)

# Cluster1 — GRC LoRAs (Policy Refactor + Risk Narrative)

This repository is a **public slice** of my local InfoSec LLM work.
This repository contains **Cluster 1** of an InfoSec LLM system: two LoRA adapters fine-tuned on
**Mistral-7B-Instruct-v0.3** for:

- **policy_refactor** — refactor messy policy text into a clear, structured policy section
- **risk_narrative** — turn raw notes into an executive-ready narrative

## Style contract (product behavior)

### `--style strict` (auditor-safe default)
Strict mode is designed to be safe for audits and controlled environments:

- **No invention**: do **not** add roles/teams, numbers, timelines, cadences, data categories, or legal/regulatory claims
  unless explicitly present in the input.
- **Unknowns are explicit**: use `TBD` / `Not yet determined`.
- **Deterministic decoding**: strict uses non-sampling generation and can optionally run an **audit pass**.

### `--style advisor` (non-binding recommendations)
Advisor mode may propose best practices, but must:

- Separate **Facts** vs **Unknowns** vs **Recommendations**
- Keep recommendations **non-binding**
- Avoid contradictions (e.g., don’t say something “was not mentioned” when it appears in the notes)

---

## Repo structure

Typical layout:

```
Cluster1/
  cluster1_cli.py
  Training/
    train_cluster1_policy_refactor_lora_v1.py
    train_cluster1_risk_narrative_lora_v1.py
  Testing/
    test_cluster1_policy_refactor.py
    test_cluster1_risk_narrative.py
  unit_testing/
    eval_cluster1_strict.py
    eval_cluster1_advisor.py          # optional lint (if present)
  data/
    training_data/
      v1/
        grc_policy_refactor_v1_1.jsonl
        grc_risk_narrative_v1.jsonl
      v2/
        grc_policy_refactor_v2_strict.jsonl
        grc_risk_narrative_v2_strict.jsonl
  adapters/                           # local-only LoRA outputs (ignored by git unless using LFS)
```

> **Recommendation:** keep `adapters/` out of normal Git. Store adapters locally, or use Git LFS if you must version them.

---

## Setup

### Python environment
Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install dependencies (adjust for your CUDA/PyTorch setup):

```bash
pip install torch transformers peft trl datasets accelerate bitsandbytes
```

### Hugging Face auth (if required)
If your environment needs authentication to download the base model:

```bash
huggingface-cli login
```

---

## Quickstart (CLI)

### Policy refactor (strict default)
```bash
python cluster1_cli.py policy_refactor \
  --style strict \
  --text "The Company shall endeavor at all times to delete..."
```

### Risk narrative (strict)
```bash
python cluster1_cli.py risk_narrative \
  --style strict \
  --text "If we see a privacy issue, we tell the DPO and sometimes notify the regulator if it seems serious."
```

### Advisor mode (non-binding recommendations)
```bash
python cluster1_cli.py risk_narrative \
  --style advisor \
  --text "If we see a privacy issue, we tell the DPO..."
```

### Debug prompt (inspect rendered chat template)
```bash
python cluster1_cli.py risk_narrative \
  --style strict \
  --debug-prompt \
  --text "..."
```

### JSON output (for integrations)
```bash
python cluster1_cli.py policy_refactor \
  --style strict \
  --json \
  --text "..."
```

---

## Evaluation

### Strict evaluation (main gate)
Runs the strict harness over a dataset split and reports any “introduced” items.

Policy refactor:
```bash
python unit_testing/eval_cluster1_strict.py \
  --task policy_refactor \
  --data data/training_data/v1/grc_policy_refactor_v1_1.jsonl \
  --split test \
  --lora-dir adapters/mistral7b-cluster1-policy-refactor-lora-v2-strict
```

Risk narrative:
```bash
python unit_testing/eval_cluster1_strict.py \
  --task risk_narrative \
  --data data/training_data/v1/grc_risk_narrative_v1.jsonl \
  --split test \
  --lora-dir adapters/mistral7b-cluster1-risk-narrative-lora-v2-strict
```

Outputs:
- `eval_report_strict*.jsonl` (per-sample records)
- CLI summary shows fail counts by category

### Advisor evaluation (optional lint)
If `unit_testing/eval_cluster1_advisor.py` exists, it can lint advisor output format and contradictions:

```bash
python unit_testing/eval_cluster1_advisor.py \
  --task risk_narrative \
  --data data/training_data/v1/grc_risk_narrative_v1.jsonl \
  --split test \
  --lora-dir adapters/mistral7b-cluster1-risk-narrative-lora-v2-strict
```

---

## Training

### Train policy_refactor LoRA (v2 strict dataset)
```bash
python Training/train_cluster1_policy_refactor_lora_v1.py \
  --data-path data/training_data/v2/grc_policy_refactor_v2_strict.jsonl \
  --output-dir adapters/mistral7b-cluster1-policy-refactor-lora-v2-strict \
  --learning-rate 5e-5 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --save-steps 100 \
  --eval-steps 50 \
  --logging-steps 10 \
  --max-length 2048 \
  --seed 7
```

### Train risk_narrative LoRA (v2 strict dataset)
```bash
python Training/train_cluster1_risk_narrative_lora_v1.py \
  --data-path data/training_data/v2/grc_risk_narrative_v2_strict.jsonl \
  --output-dir adapters/mistral7b-cluster1-risk-narrative-lora-v2-strict \
  --learning-rate 5e-5 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --save-steps 100 \
  --eval-steps 50 \
  --logging-steps 10 \
  --max-length 2048 \
  --seed 7
```

After training, validate with strict eval. Treat strict eval as a **CI-style gate**.

---

## Git hygiene (private repo recommended)

### Recommended: keep adapters out of git
Adapters are large artifacts and can exceed GitHub limits.

Add to `.gitignore` (example):
- `adapters/`
- `**/*.safetensors`
- `**/checkpoint-*`
- `eval_report*.jsonl`

If you want to version adapters, use **Git LFS**.

---

## Disclaimer
This project assists in drafting and structuring GRC/security content. Outputs must be reviewed by qualified personnel.
This repository does not provide legal advice.
