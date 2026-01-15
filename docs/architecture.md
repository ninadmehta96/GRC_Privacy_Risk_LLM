# Architecture: Reliable LLM “Clusters” for Security & GRC

This public repo is a **reference implementation** of a pattern: how to build **reliable** LLM systems for high-risk domains (security, privacy, compliance) where “sounds right” is not good enough.

The key idea is to treat an LLM like production infrastructure:

- **Scope**: what it *is allowed* to do for a specific task
- **Contracts**: what it *must not* do (e.g., invent facts not present in input)
- **Evaluation**: deterministic checks + reports that make failures visible
- **Gates**: CI prevents regressions from landing

This repo exposes the **public slice** of that pattern for Cluster 1.

---

## What is a “cluster”?

A **cluster** is a domain-specific capability unit with:

- a clear task boundary (inputs/outputs)
- a strict behavior contract (“failure rules”)
- a test/eval harness
- (optionally) a model adapter behind it (private repo)

Think of clusters as “microservices for AI behavior”.

Instead of one giant assistant doing everything, you get a **portfolio of small, auditable capabilities** that can be tested and evolved independently.

---

## The reliability pattern demonstrated here

### 1) Inference as a swappable boundary

Public repos should be runnable and deterministic in CI.

So this repo introduces an explicit inference boundary with two modes:

- **offline**: deterministic canned outputs (CI-safe)
- **hf**: optional real Transformers inference (local use only)

This lets the repo demonstrate the full pipeline (infer → audit → verdict) without requiring GPU or model downloads in CI.

### 2) Contracts: “Don’t invent what isn’t in input”

LLMs frequently hallucinate:

- **roles** (“Security Team must…” when no team is mentioned)
- **data classes** (“PII/PHI” when the input never says it)
- **legal claims** (“GDPR requires…” without source)
- **numbers/cadence** (“30 days / quarterly” without evidence)

This repo encodes strict rules that detect those cases by comparing output to the raw input.

### 3) Deterministic evaluation + gating

Public CI runs:

- lint
- unit tests
- offline strict audits over synthetic samples

Failures show up as explicit flags and reports, and CI blocks regressions.

---

## Repository modules (public)

- `unit_testing/strict_rules.py`  
  Encodes the **behavior contract**: what is forbidden to invent.

- `src/inference.py`  
  Defines the inference boundary (offline default; optional HF mode).

- `src/demo_runner.py`  
  Orchestrates: input → inference → strict audit → PASS/FAIL + report.

- `data/public_samples/*.jsonl`  
  Synthetic sample datasets for offline evaluation and CI.

---

## How to run the public demo

```bash
pip install -r requirements-dev.txt
make demo-all
```

You’ll see:

- one PASS case (bounded output with TBDs)
- one FAIL case (intentional invented specifics flagged by strict rules)

Reports are written to `reports/demo_*.json`.

---

## What’s intentionally NOT in this public repo

- model weights / LoRA adapters
- GPU training pipelines
- customer data, proprietary datasets, or sensitive artifacts

Those belong in the private repo. The public repo is about the **pattern**, not the proprietary payload.
