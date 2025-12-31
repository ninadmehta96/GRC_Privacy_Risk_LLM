#!/usr/bin/env python3
"""
eval_cluster1_strict.py

Strict-mode eval harness for Cluster 1:
- policy_refactor strict (+ optional audit pass)
- risk_narrative strict (+ optional audit pass)

It loads a JSONL with records like:
  {"id": "...", "split": "dev|test|train", "messages": [ ... ]}

It extracts the "raw input evidence" from the last user message, preferring text inside
```text ...``` fences (if present), then runs the model and flags likely hallucinations:
- numbers not present in raw evidence (ignores list numbering like "1." / "2)")
- cadence words not present in raw evidence
- roles/teams not present in raw evidence (EXACT phrase match; no gap-words)
- data classes not present in raw evidence (supports small gaps between words)
- legal/regulatory claims not present in raw evidence (EXACT phrase match; no gap-words)

Notes:
- We canonicalize common acronyms/expansions (DPO, CISO, SOC 2, ISO 27001).
- We also canonicalize "notify regulator" <-> "regulatory notification" so paraphrases
  don't get flagged as hallucinations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from peft import PeftModel


# ---------------------------
# Repo wiring
# ---------------------------

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
    build_audit_messages,
    generate,
    strip_preamble,
)


# ---------------------------
# Term lists (focused to avoid false positives)
# ---------------------------

ROLE_TERMS = [
    # roles/teams/titles (avoid generic "legal"/"compliance" adjectives)
    "dpo",
    "privacy office",
    "privacy team",
    "legal team",
    "legal counsel",
    "risk committee",
    "ciso",
    "security team",
    "compliance team",
]

DATA_CLASS_TERMS = [
    "logs",
    "log data",
    "backup",
    "backups",
    "physical records",
    "paper records",
    "system backups",
    "audit logs",
    "application logs",
]

CADENCE_TERMS = [
    "annual",
    "annually",
    "quarterly",
    "monthly",
    "weekly",
    "daily",
    "every year",
    "every quarter",
    "every month",
    "every week",
]

LEGAL_CLAIM_TERMS = [
    # keep focused on strong claims; avoid generic "regulatory"/"regulator"
    "required by law",
    "statutory",
    "gdpr",
    "ccpa",
    "hipaa",
    "pci",
    "soc 2",
    "iso 27001",
    "regulatory requirement",
    "regulatory obligation",
    "regulatory notification",
]


# ---------------------------
# JSONL helpers
# ---------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSONL parse error at line {line_no}: {e}") from e
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


# ---------------------------
# Normalization + canonicalization
# ---------------------------

_WS_RE = re.compile(r"\s+")
_SOC2_RE = re.compile(r"\bsoc\s*2\b", flags=re.IGNORECASE)
_ISO_RE = re.compile(r"\biso\s*27001\b", flags=re.IGNORECASE)
_DPO_EXPANSION_RE = re.compile(r"\bdata\s+protection\s+officer\b", flags=re.IGNORECASE)
_CISO_EXPANSION_RE = re.compile(r"\bchief\s+information\s+security\s+officer\b", flags=re.IGNORECASE)


def normalize_spaces(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip()).lower()


def canonicalize_text(s: str) -> str:
    """
    Canonicalize common acronyms/expansions so raw vs output mismatches
    don't get flagged as hallucination.
    Also canonicalize regulator notification phrasing so paraphrases align.
    """
    t = normalize_spaces(s)

    # DPO expansion -> dpo
    t = _DPO_EXPANSION_RE.sub("dpo", t)
    # CISO expansion -> ciso
    t = _CISO_EXPANSION_RE.sub("ciso", t)

    # SOC2 / ISO27001 spacing
    t = _SOC2_RE.sub("soc 2", t)
    t = _ISO_RE.sub("iso 27001", t)

    # Normalize "regulatory body" -> "regulator"
    t = re.sub(r"\brelevant regulatory body\b", "regulator", t, flags=re.IGNORECASE)
    t = re.sub(r"\bregulatory body\b", "regulator", t, flags=re.IGNORECASE)

    # Treat "notify regulator" as equivalent to "regulatory notification"
    t = re.sub(r"\bnotify(?:ing|ies)?\s+(?:the\s+)?regulator\b", "regulatory notification", t, flags=re.IGNORECASE)
    t = re.sub(r"\bnotification\s+to\s+(?:the\s+)?regulator\b", "regulatory notification", t, flags=re.IGNORECASE)

    return t


# ---------------------------
# Term matching
#   - gap-match: good for "application ... logs"
#   - exact-match: required for roles/legal to avoid false positives like
#       "privacy ... our team" matching "privacy team"
# ---------------------------

MAX_GAP_WORDS = 6
_TERM_CACHE_GAP: Dict[Tuple[str, int], re.Pattern] = {}
_TERM_CACHE_EXACT: Dict[str, re.Pattern] = {}


def _term_regex_gap(term: str, gap_words: int = MAX_GAP_WORDS) -> re.Pattern:
    """
    Regex that matches term with word boundaries, allowing up to `gap_words`
    intervening words between parts. Use for data classes.
    """
    term_n = canonicalize_text(term)
    parts = term_n.split()

    if len(parts) == 1:
        pat = r"\b" + re.escape(parts[0]) + r"\b"
        return re.compile(pat, flags=re.IGNORECASE)

    joiner = rf"(?:\W+\w+){{0,{gap_words}}}\W+"
    pat = r"\b" + joiner.join(re.escape(p) for p in parts) + r"\b"
    return re.compile(pat, flags=re.IGNORECASE)


def _term_regex_exact(term: str) -> re.Pattern:
    """
    Exact phrase match (no intervening words), but allows punctuation/whitespace
    between the words. Use for roles/legal terms.
    """
    term_n = canonicalize_text(term)
    parts = term_n.split()

    if len(parts) == 1:
        pat = r"\b" + re.escape(parts[0]) + r"\b"
    else:
        # allow punctuation/space between words, but NO extra words
        pat = r"\b" + r"\W+".join(re.escape(p) for p in parts) + r"\b"
    return re.compile(pat, flags=re.IGNORECASE)


def contains_any_term_gap(text: str, terms: List[str]) -> List[str]:
    t = canonicalize_text(text)
    hits: List[str] = []
    for term in terms:
        key = (term, MAX_GAP_WORDS)
        if key not in _TERM_CACHE_GAP:
            _TERM_CACHE_GAP[key] = _term_regex_gap(term, MAX_GAP_WORDS)
        if _TERM_CACHE_GAP[key].search(t):
            hits.append(term)
    return hits


def contains_any_term_exact(text: str, terms: List[str]) -> List[str]:
    t = canonicalize_text(text)
    hits: List[str] = []
    for term in terms:
        if term not in _TERM_CACHE_EXACT:
            _TERM_CACHE_EXACT[term] = _term_regex_exact(term)
        if _TERM_CACHE_EXACT[term].search(t):
            hits.append(term)
    return hits


def diff_terms_gap(output: str, raw: str, terms: List[str]) -> List[str]:
    out_hits = set(contains_any_term_gap(output, terms))
    raw_hits = set(contains_any_term_gap(raw, terms))
    return sorted(out_hits - raw_hits)


def diff_terms_exact(output: str, raw: str, terms: List[str]) -> List[str]:
    out_hits = set(contains_any_term_exact(output, terms))
    raw_hits = set(contains_any_term_exact(raw, terms))
    return sorted(out_hits - raw_hits)


# ---------------------------
# Numbers (ignore list numbering)
# ---------------------------

_NUM_RE = re.compile(r"(?<!\w)[~]?\d[\d,]*\.?\d*(?!\w)")
_LIST_NUM_PREFIX_RE = re.compile(
    r"(?m)^\s*\(?\d{1,3}\)?\s*(?:[.)\]:]|[-–])\s+"
)


def strip_list_numbering(s: str) -> str:
    """
    Remove common list prefixes at line start:
      1. foo
      2) foo
      (3) foo
      4: foo
      5 - foo
      6 – foo
    """
    return _LIST_NUM_PREFIX_RE.sub("", s or "")


def _canon_num(tok: str) -> str:
    t = tok.strip()
    if t.startswith("~"):
        t = t[1:]
    t = t.replace(",", "")
    if t.endswith(".") and t.count(".") == 1:
        t = t[:-1]
    return t


def extract_numbers(s: str) -> List[str]:
    return [_canon_num(m.group(0)) for m in _NUM_RE.finditer(s or "")]


def diff_numbers(output: str, raw: str) -> List[str]:
    output2 = strip_list_numbering(output)
    raw2 = strip_list_numbering(raw)
    out_nums = set(extract_numbers(output2))
    raw_nums = set(extract_numbers(raw2))
    return sorted(n for n in out_nums - raw_nums if n)


# ---------------------------
# Run model
# ---------------------------

def run_one(
    task: str,
    raw: str,
    model,
    tokenizer,
    audit: bool,
    max_new_tokens: int,
    seed: int,
) -> str:
    messages = build_messages(task, "strict", raw)
    out1 = generate(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
        debug_prompt=False,
        label="EVAL PASS 1",
    )
    out1 = strip_preamble(out1)

    if audit:
        audit_msgs = build_audit_messages(task, raw, out1)
        out2 = generate(
            model=model,
            tokenizer=tokenizer,
            messages=audit_msgs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            debug_prompt=False,
            label="EVAL AUDIT",
        )
        return strip_preamble(out2)

    return out1


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["policy_refactor", "risk_narrative"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--lora-dir", default=None)
    ap.add_argument("--audit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=550)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", default="eval_report_strict.jsonl")
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

    out_path = Path(args.report)
    total = 0
    any_fail = 0
    fail_numbers = 0
    fail_roles = 0
    fail_data = 0
    fail_cadence = 0
    fail_legal = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(rows):
            total += 1
            rec_id = rec.get("id", f"{args.split}:{idx}")
            raw = get_raw_from_record(rec)

            output = run_one(
                task=args.task,
                raw=raw,
                model=model,
                tokenizer=tokenizer,
                audit=args.audit,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )

            num_diff = diff_numbers(output, raw)

            # Roles/legal should be exact phrase matches (no gap words)
            role_diff = diff_terms_exact(output, raw, ROLE_TERMS)
            legal_diff = diff_terms_exact(output, raw, LEGAL_CLAIM_TERMS)

            # Data classes benefit from gap-matching ("application, db, infra logs")
            data_diff = diff_terms_gap(output, raw, DATA_CLASS_TERMS)

            # Cadence can stay gap-based (phrases like "every ... month" appear with punctuation)
            cadence_diff = diff_terms_gap(output, raw, CADENCE_TERMS)

            failed = bool(num_diff or role_diff or data_diff or cadence_diff or legal_diff)
            if failed:
                any_fail += 1
                if num_diff:
                    fail_numbers += 1
                if role_diff:
                    fail_roles += 1
                if data_diff:
                    fail_data += 1
                if cadence_diff:
                    fail_cadence += 1
                if legal_diff:
                    fail_legal += 1

            out_rec = {
                "id": rec_id,
                "task": args.task,
                "split": args.split,
                "audit": args.audit,
                "flags": {
                    "numbers_not_in_input": num_diff,
                    "roles_not_in_input": role_diff,
                    "data_classes_not_in_input": data_diff,
                    "cadence_not_in_input": cadence_diff,
                    "legal_claims_not_in_input": legal_diff,
                },
                "raw_excerpt": raw[:600],
                "output_excerpt": output[:1200],
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            if total % 10 == 0:
                print(f"Processed {total}/{len(rows)}... fails so far: {any_fail}")

    print("\n=== STRICT EVAL SUMMARY ===")
    print(f"Dataset: {args.data}")
    print(f"Task: {args.task}")
    print(f"Split: {args.split}")
    print(f"Audit pass: {args.audit}")
    print(f"Samples: {total}")
    print(f"Any-flag fails: {any_fail} ({(any_fail/total)*100:.1f}%)")
    print(f"  - Numbers introduced: {fail_numbers}")
    print(f"  - Roles introduced: {fail_roles}")
    print(f"  - Data classes introduced: {fail_data}")
    print(f"  - Cadence introduced: {fail_cadence}")
    print(f"  - Legal/reg claims introduced: {fail_legal}")
    print(f"Report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
