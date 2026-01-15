# Cluster1 — Reliable LLMs for GRC, Privacy, and Risk

This repository demonstrates a practical pattern for building **reliable LLM systems** in high‑risk domains such as security, privacy, and compliance.

The focus is not on prompt tricks or model hype, but on **treating an LLM as an unreliable component** inside a controlled, auditable system.

---

## What this project shows

Modern LLMs are fluent, but fluency is not correctness.

In GRC and security contexts, the most dangerous failure mode is not refusal or silence — it is **confident invention**:
numbers, roles, timelines, cadences, or legal claims that were never present in the input.

This project shows how to:

- constrain model behavior with explicit contracts
- detect invented content deterministically
- surface failures clearly instead of hiding them
- gate regressions the same way production systems are gated

---

## End‑to‑end demo (recommended starting point)

The fastest way to understand this repo is to run:

```bash
make demo-all
```

This executes a full pipeline:

```
input → inference → strict audit → PASS / FAIL → report artifact
```

You will see:

- **PASS** cases where unknowns are explicitly marked as `TBD`
- **FAIL** cases where the output intentionally invents details and is flagged

Failures are intentional and are part of the demonstration.

Reports are written to `reports/demo_*.json`.

No GPU, model downloads, or external APIs are required.

---

## Core design principles

### 1) Inference is a boundary, not the system
Inference is treated as a swappable dependency:

- **offline (default)** — deterministic outputs for demos and CI
- **hf (optional)** — real Transformers inference for local experimentation

The system logic does not depend on any single model.

### 2) Behavior is defined by contract
Outputs are not allowed to introduce:

- roles or teams
- numbers or timelines
- cadences or frequencies
- data classifications
- legal or regulatory claims

Unless those elements appear in the input.

Unknowns must remain explicit.

### 3) Deterministic verification
Every run produces:

- category‑level violation flags
- a clear verdict (`PASS` or `FAIL`)
- a structured JSON artifact suitable for diffing or review

This makes model behavior observable and testable.

### 4) Honest failure handling
The demo includes trap inputs designed to fail.
These failures are visible by design — hiding them is how unsafe systems ship.

---

## Repository structure

```
GRC_Privacy_Risk_LLM/
├── src/
│   ├── inference.py        # inference boundary (offline / hf)
│   └── demo_runner.py      # orchestrates demo execution
├── unit_testing/
│   ├── strict_rules.py     # canonical behavior contract
│   └── smoke_gate.py       # gating logic
├── demo/
│   ├── offline_outputs.json
│   └── inputs/             # good + trap cases
├── docs/
│   ├── architecture.md
│   └── why_this_matters.md
└── Makefile
```

---

## Optional: real inference

For local experimentation only:

```bash
python -m src.demo_runner --task policy_refactor --case good --mode hf
```

A local LoRA adapter can be supplied if available.
Public CI does not run in this mode.

---

## Scope and limitations

This repository intentionally does **not** include:

- model weights or LoRA adapters
- GPU training pipelines
- proprietary or customer data

Those elements are part of a separate private implementation.
This public repo exists to demonstrate the **reliability pattern**, not proprietary assets.

---

## Disclaimer

This project assists in drafting and structuring GRC and security‑related content.
All outputs must be reviewed by qualified personnel.
This repository does not provide legal advice.
