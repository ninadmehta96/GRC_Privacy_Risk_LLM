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

## Cluster Map (Roadmap)

This is the intended long-term structure. Only Cluster 1 is demonstrated publicly.

### Cluster 1 — GRC / Privacy / Risk Narratives (public demo slice)
- **policy_refactor**: convert messy policy text into structured, bounded policy language
- **risk_narrative**: convert a finding into a structured risk statement (risk/impact/likelihood/mitigations)
- **control_narrative**: map controls to narrative justification (why this control reduces risk)

### Cluster 2 — DevSecOps / Security Architecture
- threat model summaries (what could go wrong)
- architecture risk tradeoffs (where the blast radius is)
- secure-by-design control mapping

### Cluster 3 — SOC / IR / Detection Engineering
- detection logic explanations + tuning proposals
- triage narratives and containment plans
- post-incident learning → prevention controls

### Cluster 4 — Training / Human Risk / Policy Ops
- security training content aligned to incidents
- policy exceptions workflow narratives

### Cluster 5 — Offensive / Research
- recon summaries and exploit-path narratives
- “what attacker would do next” reasoning (bounded and safe)

**Why this structure matters:** it scales AI behavior like real systems: add capability without turning everything into an untestable blob.

---

## The “Reliability Pattern” (what we actually demonstrate here)

### 1) Task contracts: “Don’t invent what isn’t in input”
LLMs frequently hallucinate:
- **roles** (“Security Team must…” when no team is mentioned)
- **data classes** (“PII/PHI” when the input never says it)
- **legal claims** (“GDPR requires…” when the input doesn’t cite it)
- **numbers/cadence** (“30 days / quarterly” without evidence)

This repo encodes strict rules that detect those cases by comparing output to the raw input.

### 2) Deterministic, CPU-only evaluation
Public CI must be:
- cheap
- deterministic
- runnable without GPU access

So the public demo uses **synthetic** samples and a CPU-only offline audit.

### 3) CI as a behavior gate
Every commit must pass:
- lint
- unit tests
- offline strict audit

This prevents “demo drift”: README says one thing, repo does another.

---

## Repository Modules (Public)

- `unit_testing/strict_rules.py`  
  Encodes the **behavior contract**: what is forbidden to invent.

- `unit_testing/eval_offline_public.py`  
  CPU-only audit runner over public sample JSONL files.

- `data/public_samples/*.jsonl`  
  Synthetic sample datasets used for demo and CI.

- `.github/workflows/ci.yml`  
  Public CI job running the above gates.

---

## How to run the demo

```bash
pip install -r requirements-dev.txt
make ci
```

What you’ll see:
- strict audit summaries
- reports written to `reports/*.jsonl` (ignored from git)

---

## What’s intentionally NOT in this public repo

- model weights / LoRA adapters
- GPU training pipelines
- customer data, proprietary datasets, or sensitive artifacts

Those belong in the private repo. The public repo is about the **pattern**, not the proprietary payload.

---

## How to extend (even publicly)

If you add a new cluster capability publicly, you must add:

1. A public synthetic dataset in `data/public_samples/`
2. A contract rule (or reuse existing strict_rules)
3. A deterministic evaluator invocation wired into `make ci`

This keeps the public repo coherent and credible.
