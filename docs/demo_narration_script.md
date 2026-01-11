# Demo Narration Script (short, recruiter-friendly)

Use this as a 2–5 minute spoken walkthrough while screensharing.

---

## 0) One-liner opener (10 seconds)

“This repo shows how I build **reliable LLM behavior** for GRC/Privacy — where the main risk is the model inventing facts. The key idea is I enforce a strict contract with CPU-only evaluation in CI.”

---

## 1) What problem I’m solving (20 seconds)

“In governance domains, ‘sounds plausible’ is not good enough. If an LLM invents a cadence, a team name, or a legal claim, that’s a compliance risk. So I treat hallucination control as an engineering problem: define constraints, enforce them, and measure regressions.”

---

## 2) Run the demo (45–60 seconds)

“I can show the repo is runnable without GPUs or adapters.”

- Run: `make ci`
- Mention: “lint + unit tests + offline strict audit”
- Then open a report in `reports/` and show pass/fail plus categories.

---

## 3) Show the strict contract (60 seconds)

Open `unit_testing/strict_rules.py`.

“Here are the guardrails. I explicitly flag categories that are common sources of hallucination in GRC:
- roles/teams/vendors
- numbers/timelines/cadence
- legal claims
- data classes like credentials or personal data

Strict mode requires the output to stick to evidence and mark unknowns.”

---

## 4) Show the evaluator design (45 seconds)

Open `unit_testing/eval_offline_public.py`.

“Instead of running the model in CI, I evaluate reference outputs stored in the dataset. That keeps CI deterministic and fast, and it lets me enforce behavior regressions early. In production, inference is expensive; evaluation should be cheap and always-on.”

---

## 5) Close with why this isn’t easily replaceable (30 seconds)

“LLMs can write code quickly, but the value here is in **system design and correctness**:
- defining the right contracts
- building measurable gates
- understanding domain risk and failure modes
- wiring it into CI so it stays correct over time

That work requires human judgment and engineering discipline.”

---

## 6) If they ask: how would you extend it? (30 seconds)

“I’d add adversarial test generation, expand rule categories, and run a private inference-based evaluator in GPU CI, while keeping the public CPU gate as the always-on baseline.”
