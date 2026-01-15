# Why this matters: Making LLMs safe for high-risk work

Non-technical summary:

LLMs are great at generating convincing text — and that’s the problem.
In security, privacy, and compliance, **convincing but wrong** can cause real damage.

This repo demonstrates a reliability approach that prevents a common failure:
**the model invents specifics that were never provided**.

Importantly, the public demo shows the full system shape:

> **Inference → strict audit → PASS/FAIL verdict**

Even when inference is “offline” (deterministic) for public CI, the contract and gate are real.

---

## The core failure mode: “Invented specifics”

Given vague input, a naive LLM often adds details like:

- Roles: “Security Team must review…”
- Data classes: “PII / PHI is collected…”
- Legal claims: “GDPR requires…”
- Cadence: “Quarterly reviews / 30-day retention…”

If those details are not in the input, they are not “helpful guesses”.
They are **unverified claims** — and that’s how organizations ship incorrect policies,
make wrong compliance assertions, or mislead stakeholders.

---

## A concrete example (before / after)

### Input (messy policy text)

> “We keep logs for troubleshooting and share data with vendors when needed.”

This input does **not** specify:

- retention duration
- which data types
- which vendors
- which approvals
- which legal basis

### Naive output (sounds good, but unsafe)

> “Logs are retained for 90 days and contain PII. The Security Team reviews vendor access quarterly. GDPR requires…”

This output invented:

- a number (90 days)
- a data class (PII)
- a role + cadence (Security Team / quarterly)
- a legal claim (GDPR requires…)

### What we want instead (bounded + auditable)

A safer output:

- structures what is known
- flags unknowns as **TBD**
- avoids invented specifics

Example:

- “Retention duration: TBD”
- “Data classification: not specified”
- “Vendor approval workflow: TBD”
- “Legal basis: not specified”

This is less flashy — and far more usable in real environments.

---

## What this repo’s demo proves

This public repo implements a strict audit that checks whether the output introduces:

- roles not present in input
- data classes not present in input
- legal claims not present in input
- numbers/cadence not present in input

If the output invents these, the evaluator flags it and the demo produces a clear FAIL verdict.

So the system shifts from:

> “Trust the model”

to

> “Trust the contract + verification”

---

## Why this is a “security mindset” applied to AI

In adversarial environments, you learn quickly:

- attackers adapt
- signals are incomplete
- ambiguity is constant
- “perfect detection” doesn’t exist

So you build systems that:

- define boundaries
- measure failure
- degrade safely
- improve iteratively

This repo applies the same philosophy to LLM behavior.

---

## How to explain it to a non-technical stakeholder

A simple framing:

> “We don’t let the model make up facts. If the input didn’t say it, the output can’t claim it. Unknowns are explicit. The system is auditable.”
