# GRC Privacy Risk LLM — Cluster 1 (Policy Refactor & Risk Narrative)

This repository is a **public slice** of my local InfoSec LLM work.

It demonstrates how to design **reliable, guardrailed LLM behavior** for Governance, Risk, and Privacy use‑cases — with a strong emphasis on:
- correctness over creativity
- explicit handling of unknowns
- measurable evaluation (not just demos)
- production‑style separation of evaluation vs inference

This public repo intentionally **does NOT ship model weights, LoRA adapters, or checkpoints**.
Those are heavy, environment‑specific, and not required to demonstrate system design or evaluation rigor.

---

## What this project does

**Cluster 1 (GRC / Privacy)** implements two core LLM behaviors:

### 1) Policy Refactor
- Takes raw policy text
- Produces a structured, audit‑friendly rewrite
- Preserves meaning without inventing obligations, roles, or timelines

### 2) Risk Narrative
- Converts findings or gaps into a clear risk statement
- Separates facts from assumptions
- Uses conservative language suitable for governance and audit contexts

Both behaviors support a **Strict** mode designed for audit‑safe output.

---

## Demo in 60 seconds (no GPU, no adapters)

This is the primary public demo path and is what runs in GitHub Actions.

```bash
pip install -r requirements-dev.txt
make ci
```

What `make ci` runs:
- static linting (`ruff`)
- unit tests (`pytest`)
- **offline strict audit** against public synthetic datasets

The offline audit validates that reference outputs:
- do NOT invent roles, teams, vendors, or departments
- do NOT invent numbers, timelines, or cadences
- do NOT invent legal or regulatory claims
- only use information present in the input evidence
- explicitly mark unknowns (e.g., `TBD`, `Not specified`)

No model inference is performed in public CI.

---

## Why offline evaluation?

In real production systems, **evaluation and safety gates must be cheap, deterministic, and fast**.
They should not require GPUs or large model inference.

This repo demonstrates that pattern:
- inference‑heavy paths exist (privately)
- evaluation logic is **decoupled** and runs on CPU‑only CI
- behavior contracts are enforced even without running a model

---

## Data layout (all synthetic)

All published data in this repository is **100% synthetic**.

```
data/
├── smoke/
│   └── cluster1_smoke_inputs.jsonl
└── public_samples/
    ├── policy_refactor_v2_devtest.jsonl
    ├── policy_refactor_v2_tinytrain.jsonl
    ├── risk_narrative_v2_devtest.jsonl
    ├── risk_narrative_v2_tinytrain.jsonl
    ├── control_narrative_v1_devtest.jsonl
    └── control_narrative_v1_tinytrain.jsonl
```

- **smoke/**  
  Messy, unstructured snippets used to simulate real‑world inputs.

- **public_samples/**  
  Curated dev/test and tiny‑train splits used for:
  - offline evaluation
  - CI verification
  - demonstration of expected behavior

See `README_PUBLIC_DATA.md` for dataset details.

---

## Style contract (Strict mode)

Strict mode is designed for regulated or audit‑sensitive environments.

Rules enforced by evaluation:
- No invented roles, teams, vendors, or departments
- No invented numbers, timelines, or cadences
- No invented legal, regulatory, or compliance claims
- Explicit marking of unknowns

If required information is missing, the output must say so.

The rules are implemented in:
```
unit_testing/strict_rules.py
```

---

## What to look at (for reviewers)

If you’re skimming, these files capture the core ideas:

- `unit_testing/strict_rules.py`  
  Guardrails that define what the model is *not allowed* to invent.

- `unit_testing/eval_offline_public.py`  
  CPU‑only evaluator used in CI to enforce the strict contract.

- `data/public_samples/*`  
  Synthetic datasets with reference outputs used for evaluation.

- `.github/workflows/ci.yml`  
  Public‑safe GitHub Actions workflow (no self‑hosted runners).

---

## Full local mode (optional, not required for public demo)

If you have your own **private adapters** and ML dependencies installed, you can run
the full inference‑based gates locally:

```bash
make ci-full
make smoke
```

These paths require:
- LoRA adapters (not included)
- torch / transformers / peft
- a suitable GPU

A reference workflow for private GPU CI is provided at:

```
docs/private-gpu-ci.example.yml
```

This file is intentionally **not active** in the public repo.

---

## Why this structure matters

This project mirrors how production LLM systems should be built:

- evaluation logic is lightweight, deterministic, and always on
- inference is isolated, heavy, and optional
- safety guarantees do not depend on GPU availability

This separation allows fast iteration while maintaining reliability.

---

## Disclaimer

This project assists with drafting and structuring GRC and security content.
Outputs must be reviewed by qualified personnel.
This repository does **not** provide legal advice.
