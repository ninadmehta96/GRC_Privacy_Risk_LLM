#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_DEFAULT = "mistralai/Mistral-7B-Instruct-v0.3"

ROLE_TERMS = [
    "dpo", "ciso", "cio", "cto",
    "legal counsel", "privacy team", "security team",
    "data owner", "data steward",
    "contractor", "third party",
]

ROLE_ALIASES: Dict[str, List[str]] = {
    "dpo": [r"\bdpo\b", r"data protection officer"],
    "ciso": [r"\bciso\b", r"chief information security officer"],
    "cio": [r"\bcio\b", r"chief information officer"],
    "cto": [r"\bcto\b", r"chief technology officer"],
    "legal counsel": [r"\blegal counsel\b", r"\battorney\b", r"\blawyer\b"],
    "privacy team": [r"\bprivacy team\b"],
    "security team": [r"\bsecurity team\b", r"\binformation security\b"],
    "data owner": [r"\bdata owner\b"],
    "data steward": [r"\bdata steward\b"],
    "contractor": [r"\bcontractor(s)?\b"],
    # Include plural to avoid false positives when raw says "third parties".
    "third party": [r"\bthird[- ]part(y|ies)\b"],
}

DATA_CLASS_TERMS = [
    "pii",
    "personal data",
    "personally identifiable information",
    "phi",
    "health data",
    "financial data",
    "payment card data",
    "pci",
    "biometric data",
    "credentials",
    "passwords",
    "api keys",
]

DATA_CLASS_ALIASES: Dict[str, List[str]] = {
    "pii": [r"\bpii\b", r"personally identifiable information", r"personal information"],
    "personal data": [r"personal data", r"personal information"],
    "personally identifiable information": [r"personally identifiable information", r"\bpii\b"],
    "phi": [r"\bphi\b", r"protected health information"],
    "health data": [r"health data", r"medical data", r"patient data"],
    "financial data": [r"financial data", r"bank account", r"routing number"],
    "payment card data": [r"payment card", r"card number", r"cvv"],
    "pci": [r"\bpci\b", r"payment card"],
    "biometric data": [r"biometric", r"fingerprint", r"faceprint"],
    "credentials": [r"credentials", r"login", r"username"],
    "passwords": [r"password(s)?"],
    "api keys": [r"api key(s)?", r"access token(s)?", r"secret key(s)?"],
}

CADENCE_TERMS = [
    "daily", "weekly", "monthly", "quarterly", "annually",
    "within 24 hours", "within 72 hours", "within 30 days",
]
# Terms that often imply a legal/regulatory conclusion or obligation.
# In strict mode, these should be treated as "introduced" unless they appear in the input.
LEGAL_CLAIM_TERMS = [
    "reportable",
    "breach notification",
    "notify regulators",
    "notify the regulator",
    "notify authorities",
    "regulatory notification",
    "regulatory reporting",
    "report to regulators",
    "report to the regulator",
    "report to authorities",
    "law enforcement",
    "attorney general",
    "required by law",
    "legal requirement",
    "regulatory requirement",
    "gdpr",
    "ccpa",
    "hipaa",
]


LEGAL_TERMS = [
    "gdpr", "ccpa", "hipaa", "sox", "glba",
    "regulator", "supervisory authority", "regulatory notification",
    "breach notification", "notify affected individuals",
]

PII_EVIDENCE_PATTERNS = [
    r"\bname(s)?\b",
    r"\bphone number(s)?\b|\btelephone\b|\bmobile\b",
    r"\bemail\b|\be-mail\b",
    r"\baddress\b",
    r"\bip address\b|\bip\b",
    r"\bssn\b|\bsocial security\b",
    r"\bpassport\b|\bdriver'?s license\b",
]

def raw_supports_pii(raw: str) -> bool:
    raw_l = raw.lower()
    return any(re.search(p, raw_l) for p in PII_EVIDENCE_PATTERNS)

BLOCK_RE = re.compile(r"---\s*(.*?)\s*---", re.S)

def extract_raw_for_audit(task: str, user_prompt: str) -> str:
    if re.search(r"(?i)original notes\s*:", user_prompt):
        m = re.search(r"(?is)original notes\s*:\s*---\s*(.*?)\s*---", user_prompt)
        if m:
            return m.group(1).strip()
    m = BLOCK_RE.search(user_prompt)
    if m:
        return m.group(1).strip()
    return user_prompt.strip()

def _present(text: str, canonical: str, aliases: Optional[Dict[str, List[str]]] = None) -> bool:
    t = text.lower()
    if aliases and canonical in aliases:
        return any(re.search(pat, t) for pat in aliases[canonical])
    return canonical in t

def diff_terms_gap(output: str, raw: str, terms: List[str], aliases: Optional[Dict[str, List[str]]] = None) -> List[str]:
    out_l = output.lower()
    raw_l = raw.lower()
    introduced: List[str] = []
    for term in terms:
        if _present(out_l, term, aliases) and not _present(raw_l, term, aliases):
            introduced.append(term)
    return introduced

NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

def _numbers_not_in_raw(output: str, raw: str) -> List[str]:
    raw_nums = set(NUM_RE.findall(raw))
    out_nums = NUM_RE.findall(output)

    ignored = set()
    for m in re.finditer(r"(?m)^\s*(\d{1,2})\s*[\.)]\s+", output):
        ignored.add(m.group(1))
    for m in re.finditer(r"(?i)\bstep\s+(\d{1,2})\b", output):
        ignored.add(m.group(1))

    bad = []
    for n in out_nums:
        if n in raw_nums or n in ignored:
            continue
        bad.append(n)

    seen=set()
    out=[]
    for n in bad:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def compute_flags(output: str, raw: str) -> Dict[str, List[str]]:
    roles = diff_terms_gap(output, raw, ROLE_TERMS, ROLE_ALIASES)
    data_classes = diff_terms_gap(output, raw, DATA_CLASS_TERMS, DATA_CLASS_ALIASES)

    if raw_supports_pii(raw):
        data_classes = [t for t in data_classes if t not in ("pii", "personal data", "personally identifiable information")]

    cadence = diff_terms_gap(output, raw, CADENCE_TERMS)
    # Treat both "legal claim" phrases and general legal/regulatory terms as introduced.
    legal_terms = list(dict.fromkeys(LEGAL_CLAIM_TERMS + LEGAL_TERMS))
    legal = diff_terms_gap(output, raw, legal_terms)
    numbers = _numbers_not_in_raw(output, raw)

    return {
        "numbers_not_in_input": numbers,
        "roles_not_in_input": roles,
        "data_classes_not_in_input": data_classes,
        "cadence_not_in_input": cadence,
        "legal_claims_not_in_input": legal,
    }

def any_flagged(flags: Dict[str, List[str]]) -> bool:
    return any(bool(v) for v in flags.values())

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(base_model: str, lora_dir: str, dtype: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()
    return model, tok

def build_prompt(tokenizer: AutoTokenizer, messages: List[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts=[]
        for m in messages:
            role=m.get("role","user")
            content=m.get("content","")
            parts.append(f"{role.upper()}:\n{content}\n")
        parts.append("ASSISTANT:\n")
        return "\n".join(parts)

def generate_strict(model, tokenizer, prompt: str, max_new_tokens: int, seed: int) -> str:
    set_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if prompt in decoded:
        return decoded.split(prompt, 1)[1].strip()
    return decoded.strip()

def load_rows(path: Path) -> List[dict]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def filter_by_split(rows: List[dict], split: Optional[str]) -> List[dict]:
    if not split:
        return rows
    return [r for r in rows if str(r.get("split","")).lower() == split.lower()]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["policy_refactor", "risk_narrative"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--report", default="eval_report_strict.jsonl")
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    rows = filter_by_split(load_rows(Path(args.data)), args.split)
    if args.split and not rows:
        print(f"No rows found for split='{args.split}' in {args.data}")
        return 1

    model, tok = load_model(args.base_model, args.lora_dir, args.dtype)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fails = 0
    counts = {"numbers":0, "roles":0, "data_classes":0, "cadence":0, "legal":0}

    with report_path.open("w", encoding="utf-8") as rf:
        for r in rows:
            rid = r.get("id")
            msgs = r.get("messages", [])
            prompt_msgs = msgs[:-1] if msgs and msgs[-1].get("role") == "assistant" else msgs

            user_prompt = ""
            for m in prompt_msgs:
                if m.get("role") == "user":
                    user_prompt = m.get("content","")
                    break
            raw = extract_raw_for_audit(args.task, user_prompt)

            prompt = build_prompt(tok, prompt_msgs)
            out = generate_strict(model, tok, prompt, args.max_new_tokens, args.seed)

            flags = compute_flags(out, raw)

            rf.write(json.dumps({
                "id": rid,
                "task": args.task,
                "split": args.split,
                "lora_dir": args.lora_dir,
                "base_model": args.base_model,
                "audit_pass": not args.no_audit,
                "flags": flags,
                "raw_excerpt": raw[:1200],
                "output_excerpt": out[:1200],
            }, ensure_ascii=False) + "\n")

            if any_flagged(flags):
                fails += 1
                if flags["numbers_not_in_input"]:
                    counts["numbers"] += 1
                if flags["roles_not_in_input"]:
                    counts["roles"] += 1
                if flags["data_classes_not_in_input"]:
                    counts["data_classes"] += 1
                if flags["cadence_not_in_input"]:
                    counts["cadence"] += 1
                if flags["legal_claims_not_in_input"]:
                    counts["legal"] += 1

    total = len(rows)
    pct = (fails / total * 100.0) if total else 0.0
    print("\n=== STRICT EVAL SUMMARY ===")
    print(f"Dataset: {args.data}")
    print(f"Task: {args.task}")
    print(f"Split: {args.split or 'ALL'}")
    print(f"Audit pass: {not args.no_audit}")
    print(f"Samples: {total}")
    print(f"Any-flag fails: {fails} ({pct:.1f}%)")
    print(f"  - Numbers introduced: {counts['numbers']}")
    print(f"  - Roles introduced: {counts['roles']}")
    print(f"  - Data classes introduced: {counts['data_classes']}")
    print(f"  - Cadence introduced: {counts['cadence']}")
    print(f"  - Legal/reg claims introduced: {counts['legal']}")
    print(f"Report written to: {report_path.resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Compatibility shims for smoke_gate.py
# smoke_gate will look for diff_terms/diff_numbers or the *_gap variants.
# We expose both, delegating to the internal helpers above.
# ---------------------------------------------------------------------------

def diff_numbers_gap(output: str, raw: str):
    """Return numeric tokens present in output but not in raw, with list-index noise filtered."""
    return _numbers_not_in_raw(output, raw)

def diff_terms(output: str, raw: str, terms):
    """Back-compat alias for diff_terms_gap.

    Older callers (e.g., smoke_gate.py) pass only (output, raw, terms). We still want
    alias-aware matching for roles and data-classes, so we auto-select aliases when
    the provided terms match known term lists.
    """
    aliases = None
    try:
        if list(terms) == ROLE_TERMS:
            aliases = ROLE_ALIASES
        elif list(terms) == DATA_CLASS_TERMS:
            aliases = DATA_CLASS_ALIASES
    except Exception:
        aliases = None
    return diff_terms_gap(output, raw, list(terms), aliases)

def diff_numbers(output: str, raw: str):
    """Back-compat alias for diff_numbers_gap."""
    return diff_numbers_gap(output, raw)
