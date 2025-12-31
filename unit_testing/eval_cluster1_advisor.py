#!/usr/bin/env python3
"""
eval_cluster1_advisor.py

Lightweight eval harness for Cluster 1 "advisor" style.

What it checks (heuristics, not formal proof):
- Output contains the expected headings for the task (advisor contract).
- Output does NOT claim key entities were "not mentioned" when the raw notes contain them
  (e.g., raw mentions DPO but output says "DPO not mentioned").

This is intentionally simple and fast; it is meant to catch regressions in prompt/style,
not to fully validate truthfulness.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from peft import PeftModel


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "cluster1_cli.py").exists():
            return p
    return start.parent


REPO_ROOT = find_repo_root(Path(__file__))
sys.path.insert(0, str(REPO_ROOT))

from cluster1_cli import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_LORA_POLICY,
    DEFAULT_LORA_RISK,
    load_model_and_tokenizer,
    build_messages,
    generate,
    strip_preamble,
)

WS_RE = re.compile(r"\s+")


def canon(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip()).lower()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_fenced_text(s: str) -> Optional[str]:
    m = re.search(r"```text\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def get_raw_from_record(rec: Dict[str, Any]) -> str:
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ""
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if not user_msgs:
        return ""
    last_user = user_msgs[-1].get("content", "")
    if not isinstance(last_user, str):
        return ""
    fenced = extract_fenced_text(last_user)
    return fenced if fenced is not None else last_user.strip()


def has_required_headings(task: str, out: str) -> List[str]:
    """Return missing headings (regex patterns)."""
    t = canon(out)

    if task == "risk_narrative":
        required = [
            r"facts\s*\(from notes\)\s*:",
            r"unknowns\s*\(not in notes\)\s*:",
            r"recommendations\s*\(non-binding\)\s*:",
        ]
    else:
        return []

    missing = []
    for pat in required:
        if not re.search(pat, t, flags=re.IGNORECASE):
            missing.append(pat)
    return missing


def contradiction_not_mentioned(raw: str, out: str) -> List[str]:
    """If raw contains an entity, output should not say it wasn't mentioned."""
    raw_c = canon(raw)
    out_c = canon(out)

    issues: List[str] = []

    entities = [
        ("dpo", r"\bdpo\b"),
        ("regulator", r"\bregulator\b"),
        ("ciso", r"\bciso\b"),
    ]

    for label, ent_pat in entities:
        if re.search(ent_pat, raw_c):
            if re.search(rf"\b{label}\b.*\bnot\b.*\bmentioned\b", out_c) or re.search(
                rf"\bnot\b.*\bmentioned\b.*\b{label}\b", out_c
            ):
                issues.append(f"{label}_claimed_not_mentioned")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["policy_refactor", "risk_narrative"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--lora-dir", default=None)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=650)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", default="eval_report_advisor.jsonl")
    args = ap.parse_args()

    lora_dir = args.lora_dir
    if lora_dir is None:
        lora_dir = DEFAULT_LORA_POLICY if args.task == "policy_refactor" else DEFAULT_LORA_RISK

    rows = read_jsonl(args.data)
    rows = [r for r in rows if r.get("split", "train") == args.split]
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit(f"No rows found for split='{args.split}' in {args.data}")

    print(f"Loading base model: {args.base_model}")
    base_model, tokenizer = load_model_and_tokenizer(args.base_model, args.dtype)
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    total = 0
    any_fail = 0
    fail_headings = 0
    fail_contradiction = 0

    out_path = Path(args.report)
    with out_path.open("w", encoding="utf-8") as out_f:
        for i, rec in enumerate(rows):
            total += 1
            rec_id = rec.get("id", f"{args.split}:{i}")
            raw = get_raw_from_record(rec)

            msgs = build_messages(args.task, "advisor", raw)
            out = generate(
                model=model,
                tokenizer=tokenizer,
                messages=msgs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                debug_prompt=False,
                label="ADVISOR EVAL",
            )
            out = strip_preamble(out)

            missing = has_required_headings(args.task, out)
            contrad = contradiction_not_mentioned(raw, out)

            failed = bool(missing or contrad)
            if failed:
                any_fail += 1
                if missing:
                    fail_headings += 1
                if contrad:
                    fail_contradiction += 1

            out_rec = {
                "id": rec_id,
                "task": args.task,
                "split": args.split,
                "style": "advisor",
                "flags": {
                    "missing_headings": missing,
                    "contradiction_not_mentioned": contrad,
                },
                "raw_excerpt": raw[:600],
                "output_excerpt": out[:1200],
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            if total % 10 == 0:
                print(f"Processed {total}/{len(rows)}... fails so far: {any_fail}")

    print("\n=== ADVISOR EVAL SUMMARY ===")
    print(f"Dataset: {args.data}")
    print(f"Task: {args.task}")
    print(f"Split: {args.split}")
    print(f"Samples: {total}")
    print(f"Any-flag fails: {any_fail} ({(any_fail/total)*100:.1f}%)")
    print(f"  - Missing required headings: {fail_headings}")
    print(f"  - 'not mentioned' contradictions: {fail_contradiction}")
    print(f"Report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
