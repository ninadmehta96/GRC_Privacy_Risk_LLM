[![CI (Public)](https://github.com/ninadmehta96/GRC_Privacy_Risk_LLM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ninadmehta96/GRC_Privacy_Risk_LLM/actions/workflows/ci.yml)

# GRC Privacy Risk LLM — Cluster 1  
### Policy Refactor & Risk Narrative (Public Evaluation Slice)

This repository is a **public, recruiter‑friendly slice** of my local InfoSec / GRC LLM work.

It focuses on something that matters in regulated domains but is often under‑engineered:
**reliability, correctness, and non‑hallucination guarantees**, not just fluent text generation.

> **What this repo is**
> - CPU‑only
> - Fully reproducible
> - Focused on evaluation, guardrails, and behavior contracts  
>
> **What this repo is NOT**
> - It does **not** ship LoRA weights or checkpoints  
> - It does **not** require GPUs  
> - It is **not** a demo of “prompt cleverness”

---

## What Cluster 1 does

Cluster 1 covers two GRC‑oriented LLM behaviors:

- **Policy Refactor**  
  Convert messy or inconsistent policy text into a clear, structured, audit‑friendly policy section without inventing obligations.

- **Risk Narrative**  
  Turn raw findings or notes into an executive‑ready risk narrative that cleanly separates facts, unknowns, and implications.

Both behaviors support a **Strict** mode intended for audit‑safe usage.

---

## 60‑second demo (recommended)

```bash
make ci
```

This runs, entirely on CPU:

1. `ruff` (static checks)
2. `pytest` (unit tests)
3. **Offline strict audit** against synthetic public datasets

Reports are written to:

```bash
reports/
```

No model inference is performed in public CI.

---

## Why offline evaluation?

In real systems:

- Inference is **expensive** and environment‑dependent  
- Evaluation must be **cheap, deterministic, and always‑on**

This repo demonstrates that separation:

- Heavy inference + LoRA adapters exist **privately**
- Evaluation logic runs **independently** in CI
- Behavior regressions are caught without GPUs

---

## Style contract (product behavior)

### Strict mode (auditor‑safe default)

Strict mode enforces a **negative contract** — what the model is *not allowed* to invent.

Rules include:

- ❌ No invented roles, teams, vendors, or departments  
- ❌ No invented numbers, timelines, or cadences  
- ❌ No invented legal or regulatory claims  
- ❌ No invented data classes (e.g., credentials, personal data)  
- ✅ Unknowns must be explicit (`TBD`, `Not specified`)

Rules are encoded in:

```
unit_testing/strict_rules.py
```

---

## Offline evaluator

The CPU‑only evaluator lives in:

```
unit_testing/eval_offline_public.py
```

It evaluates **reference outputs stored in the dataset**, flags violations, and produces deterministic JSONL reports.

---

## Public datasets (100% synthetic)

```
data/public_samples/
```

All data in this repository is synthetic and safe to share.
See `README_PUBLIC_DATA.md` for details.

---

## What to review first

- `unit_testing/strict_rules.py`
- `unit_testing/eval_offline_public.py`
- `.github/workflows/ci.yml`
- `docs/demo.md`

---

## Full local / GPU mode (intentionally excluded)

The full system includes LoRA adapters and GPU inference.
Those artifacts are intentionally **not published** here.

A reference private workflow exists at:

```
docs/private-gpu-ci.example.yml
```

---

## Disclaimer

This project assists with drafting and structuring GRC / security content.
Outputs must be reviewed by qualified professionals.
This repository does **not** provide legal advice.
