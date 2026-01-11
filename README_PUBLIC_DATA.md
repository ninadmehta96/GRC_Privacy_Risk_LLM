# Public data pack (recommended)

This folder contains small, fully synthetic JSONL subsets intended for a public GitHub repo.

## Included files

- `data/smoke/cluster1_smoke_inputs.jsonl` (30 rows)
  - Minimal smoke inputs used by your smoke gate.

- `data/public_samples/policy_refactor_v2_devtest.jsonl` (dev+test subset)
- `data/public_samples/risk_narrative_v2_devtest.jsonl` (dev+test subset)
- `data/public_samples/control_narrative_v1_devtest.jsonl` (dev+test subset)

These "devtest" files are best for reproducible CI/evaluation in a public repo.

- `data/public_samples/policy_refactor_v2_tinytrain.jsonl` (20 train rows)
- `data/public_samples/risk_narrative_v2_tinytrain.jsonl` (20 train rows)
- `data/public_samples/control_narrative_v1_tinytrain.jsonl` (8 train rows)

These "tinytrain" files are optional; they enable a lightweight demo fine-tune without shipping the full training corpus.

## Why not publish full training_data?
Even when synthetic, keeping the full corpus private makes the public repo lighter and reduces accidental future leakage. You can always expand later.
