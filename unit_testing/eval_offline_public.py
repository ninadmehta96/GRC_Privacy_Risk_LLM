#!/usr/bin/env python3
"""
Offline evaluator for public samples.

- NO model inference.
- NO torch/transformers/peft dependencies.
- Uses the reference assistant output already present in the JSONL.
- Extracts evidence/raw from the user prompt between --- delimiters.
- Runs strict_rules.compute_flags() and fails CI if any sample flags.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Important: this must stay stdlib-only. strict_rules should also be stdlib-only.
# Import strict_rules in a way that works both as:
#   - `python -m unit_testing.eval_offline_public ...` (recommended)
#   - `python unit_testing/eval_offline_public.py ...` (direct execution)
try:
    from unit_testing.strict_rules import compute_flags, any_flagged  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # When executed as a script, Python sets sys.path[0] to the script directory (`unit_testing/`),
    # which can break absolute package imports. Add repo root to sys.path and retry.
    import sys
    from pathlib import Path as _Path

    _repo_root = str(_Path(__file__).resolve().parents[1])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from unit_testing.strict_rules import compute_flags, any_flagged  # type: ignore



def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_by_split(rows: List[Dict[str, Any]], split: Optional[str]) -> List[Dict[str, Any]]:
    if not split:
        return rows
    s = split.lower()
    return [r for r in rows if str(r.get("split", "")).lower() == s]


def first_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def last_assistant_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


def extract_raw_from_user_prompt(user_text: str) -> str:
    """
    Your public samples use:
      instructions
      ---
      RAW / evidence
      ---

    If we can't find delimiters, fall back to the full user prompt.
    """
    lines = (user_text or "").splitlines()
    idx = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(idx) >= 2:
        raw_lines = lines[idx[0] + 1 : idx[1]]
        raw = "\n".join(raw_lines).strip()
        return raw
    return (user_text or "").strip()


def short(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


@dataclass
class Counts:
    numbers: int = 0
    roles: int = 0
    data_classes: int = 0
    cadence: int = 0
    legal: int = 0


def bump_counts(counts: Counts, flags: Dict[str, List[str]]) -> None:
    if flags.get("numbers_not_in_input"):
        counts.numbers += 1
    if flags.get("roles_not_in_input"):
        counts.roles += 1
    if flags.get("data_classes_not_in_input"):
        counts.data_classes += 1
    if flags.get("cadence_not_in_input"):
        counts.cadence += 1
    if flags.get("legal_claims_not_in_input"):
        counts.legal += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data/public_samples/*.jsonl")
    ap.add_argument("--split", default=None, help="Optional split filter (e.g., test)")
    ap.add_argument("--report", default="reports/public_offline_eval.jsonl")
    ap.add_argument(
        "--task",
        default=None,
        help="Optional task_type override. If not provided, uses row['task_type']",
    )
    args = ap.parse_args()

    data_path = Path(args.data)
    rows = filter_by_split(read_jsonl(data_path), args.split)
    if args.split and not rows:
        print(f"No rows found for split='{args.split}' in {data_path}")
        return 2

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fails = 0
    counts = Counts()

    with report_path.open("w", encoding="utf-8") as rf:
        for r in rows:
            rid = r.get("id", "unknown")
            task_type = args.task or r.get("task_type") or r.get("task") or "unknown"
            messages = r.get("messages", []) or []

            user_prompt = first_user_message(messages)
            raw = extract_raw_from_user_prompt(user_prompt)
            reference_out = last_assistant_message(messages)

            if not reference_out.strip():
                # If a row has no assistant completion, it isn't evaluable in offline mode.
                flags = {
                    "numbers_not_in_input": ["<missing_assistant_output>"],
                    "roles_not_in_input": [],
                    "data_classes_not_in_input": [],
                    "cadence_not_in_input": [],
                    "legal_claims_not_in_input": [],
                }
            else:
                flags = compute_flags(reference_out, raw)

            failed = any_flagged(flags)
            if failed:
                fails += 1
                bump_counts(counts, flags)

            rf.write(
                json.dumps(
                    {
                        "id": rid,
                        "task_type": task_type,
                        "split": r.get("split"),
                        "pass": not failed,
                        "flags": flags,
                        "raw_excerpt": short(raw, 600),
                        "output_excerpt": short(reference_out, 1200),
                        "data_file": str(data_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(rows)
    pct = (fails / total * 100.0) if total else 0.0
    print("\n=== OFFLINE STRICT AUDIT SUMMARY ===")
    print(f"Dataset: {data_path}")
    print(f"Split: {args.split or 'ALL'}")
    print(f"Samples: {total}")
    print(f"Failing samples: {fails} ({pct:.1f}%)")
    print("By category (count of failing samples triggering):")
    print(f"  - numbers:      {counts.numbers}")
    print(f"  - roles:        {counts.roles}")
    print(f"  - data_classes: {counts.data_classes}")
    print(f"  - cadence:      {counts.cadence}")
    print(f"  - legal:        {counts.legal}")
    print(f"Report written to: {report_path.resolve()}")

    # Make CI fail if anything is flagged.
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

