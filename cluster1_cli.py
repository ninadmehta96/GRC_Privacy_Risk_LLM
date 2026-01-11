#!/usr/bin/env python3
"""Cluster1 CLI (policy_refactor + risk_narrative).

This file doubles as:
1) The interactive CLI you run locally, and
2) A library imported by unit_testing/smoke_gate.py.

So we keep the core functions (build_messages / generate / audit / strict_generate_with_repairs)
stable and deterministic in strict mode.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

DEFAULT_LORA_POLICY = "./mistral7b-cluster1-policy-refactor-lora-v1"
DEFAULT_LORA_RISK   = "./mistral7b-cluster1-risk-narrative-lora-v1"


SYSTEM_POLICY = (
    "You are a senior GRC & privacy engineer. "
    "You refactor security/privacy policies to improve clarity and structure for engineers, "
    "product managers, and auditors — without changing obligations."
)

SYSTEM_RISK = (
    "You are a senior risk & privacy leader. "
    "You turn raw incident/risk/DPIA information into concise executive narratives, "
    "clearly separating facts from assessment."
)


def scrub_introduced_roles(text: str, introduced_roles: List[str], ev) -> str:
    """
    Deterministic post-processor used ONLY when verifier flags invented roles.

    Behavior:
      1) If the evaluator module provides ev.scrub_introduced_roles, prefer it.
      2) Otherwise fall back to a conservative local scrub that uses ev.ROLE_ALIASES when available.

    This keeps smoke/strict stable while allowing you to later move strict rules into
    a lightweight module (e.g., unit_testing/strict_rules.py) without touching this file again.
    """
    if not introduced_roles:
        return text

    # Prefer evaluator-provided scrubber if available (future-proof)
    if ev is not None:
        fn = getattr(ev, "scrub_introduced_roles", None)
        if callable(fn):
            try:
                return fn(text, introduced_roles)  # strict_rules style
            except TypeError:
                try:
                    return fn(text, introduced_roles, ev)  # legacy style (unlikely)
                except TypeError:
                    pass

    t = text

    # Neutral placeholder (shouldn't match ROLE_TERMS in your evaluator)
    repl = "designated responsible owner (TBD)"

    role_aliases = getattr(ev, "ROLE_ALIASES", {}) if ev is not None else {}

    patterns: List[str] = []
    for role in introduced_roles:
        # Special-case: historically most common autopilot role injection.
        if role == "security team":
            patterns += [
                r"\bsecurity team\b",
                r"\binformation security\b",
                r"\binfosec\b",
                r"\binfo\s*sec\b",
                r"\bISO\b",
                r"\bsecurity officer\b",
                r"\binformation security officer\b",
            ]
        else:
            patterns += role_aliases.get(role, [])
            patterns += [re.escape(role)]

    for pat in patterns:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)

    # Cleanup minor formatting artifacts
    t = re.sub(r"\s{2,}", " ", t)
    return t


def read_text(text: Optional[str], text_file: Optional[str]) -> str:
    if (text is None) == (text_file is None):
        raise ValueError("Provide exactly one of --text or --text-file.")
    if text_file:
        return Path(text_file).read_text(encoding="utf-8").strip()
    return text.strip()


def strip_preamble(s: str) -> str:
    """
    Remove model-added headers like 'Refactored ...' and leading separators.
    Keep from the first real heading if present.
    """
    s = s.strip()
    # Drop common separators
    s = re.sub(r"^\s*-{3,}\s*", "", s).strip()

    # Keep from first heading (Scope: is our anchor)
    m = re.search(r"(?mi)^(scope\s*:)", s)
    if m:
        s = s[m.start():].strip()

    # Remove repeated leading separators again
    s = re.sub(r"^\s*-{3,}\s*", "", s).strip()
    return s


def build_messages(mode: str, style: str, raw: str) -> List[Dict[str, str]]:
    """
    style:
      - "strict": no invention. Unknown => TBD / Not yet determined.
      - "assumptive": may propose suggestions, clearly labeled as suggestions.
    """

    if mode == "policy_refactor":
        if style == "strict":
            user = (
                "Policy draft (messy):\n\n"
                f"{raw}\n\n"
                "Task: Refactor this into a clear policy section with headings:\n"
                "Scope\n"
                "Policy\n"
                "Timelines\n"
                "Exceptions & Approvals\n"
                "Engineering Notes\n"
                "Required Inputs\n\n"
                "STRICT guardrails (must follow):\n"
                "- You may ONLY state facts/obligations explicitly present in the draft.\n"
                "- Do NOT introduce any new retention periods, deadlines, cadences (annual/quarterly), or numbers.\n"
                "- Do NOT introduce new data classes (e.g., logs, backups, physical records) unless the draft mentions them.\n"
                "- Do NOT introduce new roles/owners (e.g., Legal, DPO, Privacy Office) unless the draft mentions them.\n"
                "- If timelines/owners/criteria are missing, write 'TBD' and list what is needed under Required Inputs.\n"
                "- Improve wording/structure ONLY; do not expand scope or increase commitments.\n\n"
                "Return ONLY the refactored policy section."
            )
        elif style == "assumptive":
            user = (
                "Policy draft (messy):\n\n"
                f"{raw}\n\n"
                "Task: Refactor this into a clear policy section with headings:\n"
                "Scope\n"
                "Policy\n"
                "Timelines\n"
                "Exceptions & Approvals\n"
                "Engineering Notes\n\n"
                "ASSUMPTIVE mode rules:\n"
                "- You MAY propose reasonable example defaults (e.g., retention windows, review cadence) IF missing.\n"
                "- You MUST NOT present suggestions as existing policy.\n"
                "- Place any guesses under a final heading: Suggested Defaults (Confirm with Legal/Privacy).\n"
                "- Keep the main Policy conservative and faithful to the draft.\n\n"
                "Return ONLY the refactored policy section."
            )
        else:
            raise ValueError("style must be 'strict' or 'assumptive'")

        return [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": user},
        ]

    if mode == "risk_narrative":
        if style == "strict":
            user = (
                "Raw incident / risk notes:\n\n"
                f"{raw}\n\n"
                "Task: Write a 3–5 paragraph board-level executive summary focusing on:\n"
                "- what happened\n"
                "- who is impacted\n"
                "- current risk\n"
                "- planned remediation (ONLY what is explicitly stated)\n\n"
                "STRICT guardrails (must follow):\n"
                "- Do NOT claim actions occurred unless explicitly stated (e.g., notifications sent, forensics completed).\n"
                "- Do NOT invent facts, numbers, dates, root cause details, scope, or legal conclusions.\n"
                "- If something is unknown, say 'Not yet determined' and what will be verified.\n"
                "- Keep assessment phrased as risk/uncertainty; do not over-certify.\n\n"
                "Return ONLY the executive summary."
            )
        elif style == "assumptive":
            user = (
                "Raw incident / risk notes:\n\n"
                f"{raw}\n\n"
                "Task: Write a 3–5 paragraph board-level executive summary focusing on:\n"
                "- what happened\n"
                "- who is impacted\n"
                "- current risk\n"
                "- planned remediation\n\n"
                "ASSUMPTIVE mode rules:\n"
                "- You MAY propose sensible next steps as recommendations.\n"
                "- You MUST NOT present recommendations as actions already taken.\n"
                "- Do NOT invent facts/numbers/dates; unknown => 'Not yet determined'.\n\n"
                "Output:\n"
                "- 3–5 paragraphs summary\n"
                "- Then a short bullet list titled 'Recommended Next Steps' (clearly recommendations)\n\n"
                "Return ONLY the narrative + recommendations."
            )
        else:
            raise ValueError("style must be 'strict' or 'assumptive'")

        return [
            {"role": "system", "content": SYSTEM_RISK},
            {"role": "user", "content": user},
        ]

    raise ValueError(f"Unknown mode: {mode}")


def build_audit_messages(mode: str, raw: str, draft_output: str) -> List[Dict[str, str]]:
    """
    Second pass: remove anything not supported by raw input.
    This is what stops strict-mode hallucinated timelines/roles/data types.
    """
    if mode == "policy_refactor":
        user = (
            "You are auditing a policy refactor for strict non-invention.\n\n"
            "Original draft:\n"
            f"{raw}\n\n"
            "Refactored output to audit:\n"
            f"{draft_output}\n\n"
            "Rules:\n"
            "- Remove or rewrite any statement not explicitly supported by the original draft.\n"
            "- Remove any invented numbers/timelines/cadences.\n"
            "- Remove invented data classes (e.g., logs/backups) unless present in the draft.\n"
            "- Remove invented roles/owners (e.g., Legal/DPO/Privacy Office) unless present.\n"
            "- Keep headings and produce the corrected final policy section only.\n"
            "- If something is missing, use 'TBD' and put it under Required Inputs.\n\n"
            "Return ONLY the corrected policy section."
        )
        return [{"role": "system", "content": SYSTEM_POLICY}, {"role": "user", "content": user}]

    if mode == "risk_narrative":
        user = (
            "You are auditing an executive incident narrative for strict non-invention.\n\n"
            "Original notes:\n"
            f"{raw}\n\n"
            "Narrative to audit:\n"
            f"{draft_output}\n\n"
            "Rules:\n"
            "- Remove or rewrite any statement not explicitly supported by the original notes.\n"
            "- Do not add facts. Unknown => 'Not yet determined'.\n"
            "- Return ONLY the corrected executive summary (no extra headers).\n"
        )
        return [{"role": "system", "content": SYSTEM_RISK}, {"role": "user", "content": user}]

    raise ValueError(f"Unknown mode for audit: {mode}")


def normalize_hyphen_bullets(text: str) -> str:
    # Normalize common bullet glyphs (e.g., Unicode bullets/dashes, '*') to a plain '- ' bullet.
    # Formatting-only: helps keep strict output consistent.
    text = re.sub(r"(?m)^\s*[\u2022\u25CF\u2023\u2043\u2219\u00B7•]\s+", "- ", text)
    text = re.sub(r"(?m)^\s*[\u2013\u2014\u2212–—-]\s+", "- ", text)  # en/em/minus dashes
    text = re.sub(r"(?m)^\s*\*\s+", "- ", text)
    text = re.sub(r"(?m)^-\s{2,}", "- ", text)
    return text


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(base_model_id: str, dtype_str: str):
    if dtype_str == "bf16":
        dtype = torch.bfloat16
    elif dtype_str == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("--dtype must be bf16 or fp16")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.backends.cuda.matmul.allow_tf32 = True

    # Transformers has flipped between `torch_dtype` and `dtype` depending on version.
    # Try `dtype` first to avoid warnings; on older versions fall back to `torch_dtype`.
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map="auto",
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
    base_model.eval()
    return base_model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens, temperature, top_p, seed):
    _set_seed(seed)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs: Dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Only pass sampling params when sampling.
    if temperature is not None and float(temperature) > 0.0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": float(top_p),
        })
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _try_load_strict_evaluator():
    """Best-effort import of strict evaluation rules.

    Preference order:
      1) unit_testing.strict_rules (lightweight, CPU-friendly)
      2) unit_testing.eval_cluster1_strict (legacy)
      3) None (no strict checks)
    """
    for modname in ("unit_testing.strict_rules", "unit_testing.eval_cluster1_strict"):
        try:
            mod = __import__(modname, fromlist=["*"])
            return mod
        except Exception:
            continue
    return None


def strict_generate_with_repairs(
    *,
    task: str,
    raw: str,
    model,
    tokenizer,
    max_new_tokens: int = 550,
    seed: int = 7,
    audit: bool = True,
    debug_prompt: bool = False,
    max_repairs: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """Strict pipeline used by unit_testing/smoke_gate.py.

    Pipeline intuition (compiler-style):
      raw input -> (draft) -> (audit) -> (verify flags) -> (repair if needed) -> (verify)

    Repairs are targeted: we tell the model exactly which phrases are forbidden because
    they were not supported by the input. This is much more reliable than hoping the
    first strict prompt never emits defaults like "security team".
    """
    if task not in ("policy_refactor", "risk_narrative"):
        raise ValueError(f"Unknown task: {task}")

    ev = _try_load_strict_evaluator()

    def _flags_for(text: str) -> Dict[str, List[str]]:
        if ev is None:
            return {
                "numbers_not_in_input": [],
                "roles_not_in_input": [],
                "data_classes_not_in_input": [],
                "cadence_not_in_input": [],
                "legal_claims_not_in_input": [],
            }
        compute_flags_fn = getattr(ev, "compute_flags", None)
        if callable(compute_flags_fn):
            return compute_flags_fn(text, raw)
        # If evaluator doesn't implement compute_flags, be conservative.
        return {
            "numbers_not_in_input": [],
            "roles_not_in_input": [],
            "data_classes_not_in_input": [],
            "cadence_not_in_input": [],
            "legal_claims_not_in_input": [],
        }

    def _any_flagged(flags: Dict[str, List[str]]) -> bool:
        if ev is None:
            return False
        any_flagged_fn = getattr(ev, "any_flagged", None)
        if callable(any_flagged_fn):
            return bool(any_flagged_fn(flags))
        return any(bool(v) for v in flags.values())

    dbg: Dict[str, Any] = {"attempts": []}

    # 1) Draft
    msgs = build_messages(task, "strict", raw)
    out = strip_preamble(generate(
        model=model,
        tokenizer=tokenizer,
        messages=msgs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
    ))
    out = normalize_hyphen_bullets(out)

    # 2) Audit
    if audit:
        audit_msgs = build_audit_messages(task, raw, out)
        out = strip_preamble(generate(
            model=model,
            tokenizer=tokenizer,
            messages=audit_msgs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
        ))
        out = normalize_hyphen_bullets(out)

    flags = _flags_for(out)
    dbg["attempts"].append({"stage": "draft+audit", "flags": flags})

    # Deterministic scrub: if verifier says we introduced roles, remove them.
    if ev is not None and flags.get("roles_not_in_input"):
        out2 = scrub_introduced_roles(out, flags["roles_not_in_input"], ev)
        if out2 != out:
            out = out2
            flags = _flags_for(out)
            dbg["attempts"].append({"stage": "draft+audit_scrub_roles", "flags": flags})

    if not _any_flagged(flags):
        return out, dbg

    # 3) Repair loop (targeted rewrite)
    system = SYSTEM_POLICY if task == "policy_refactor" else SYSTEM_RISK

    for k in range(1, max_repairs + 1):
        forbidden_lines = []
        for cat, items in flags.items():
            if items:
                forbidden_lines.append(f"- {cat}: {', '.join(items)}")
        forbidden_blob = "\n".join(forbidden_lines) if forbidden_lines else "- (none)"

        if task == "policy_refactor":
            structural_rule = (
                "Keep the SAME headings in the same order: Scope, Policy, Timelines, Exceptions & Approvals, "
                "Engineering Notes, Required Inputs."
            )
            missing_rule = "If an owner/timeline is missing, write 'TBD' and list required info under Required Inputs."
        else:
            structural_rule = "Keep it as a 3–5 paragraph executive summary (no bullet lists unless the input had them)."
            missing_rule = "If something is unknown, say 'Not yet determined' without adding new details."

        user = (
            "You are repairing a STRICT output to remove unsupported introduced terms.\n\n"
            "Original input (source of truth):\n"
            f"---\n{raw}\n---\n\n"
            "Current output (needs repair):\n"
            f"---\n{out}\n---\n\n"
            "The following items were detected as INTRODUCED (not present in the input) and MUST be removed or rewritten "
            "so that none of these phrases/terms appear in the final answer:\n"
            f"{forbidden_blob}\n\n"
            "Rules:\n"
            f"- {structural_rule}\n"
            f"- {missing_rule}\n"
            "- Do NOT introduce any new facts, numbers, timelines, cadences, roles, data classes, or legal conclusions.\n"
            "- Return ONLY the repaired final output (no preamble)."
        )

        repair_msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        out = strip_preamble(generate(
            model=model,
            tokenizer=tokenizer,
            messages=repair_msgs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
        ))
        out = normalize_hyphen_bullets(out)

        if audit:
            audit_msgs = build_audit_messages(task, raw, out)
            out = strip_preamble(generate(
                model=model,
                tokenizer=tokenizer,
                messages=audit_msgs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=seed,
            ))
            out = normalize_hyphen_bullets(out)

        flags = _flags_for(out)
        dbg["attempts"].append({"stage": f"repair_{k}", "flags": flags})

        # scrub roles again after each repair iteration
        if ev is not None and flags.get("roles_not_in_input"):
            out2 = scrub_introduced_roles(out, flags["roles_not_in_input"], ev)
            if out2 != out:
                out = out2
                flags = _flags_for(out)
                dbg["attempts"].append({"stage": f"repair_{k}_scrub_roles", "flags": flags})

        if not _any_flagged(flags):
            break

    if debug_prompt:
        dbg["last_output"] = out
    return out, dbg


def main():
    parser = argparse.ArgumentParser(
        description="Cluster-1 router CLI: policy_refactor + risk_narrative LoRAs"
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Make these optional so we can set safer defaults per style
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("policy_refactor")
    p1.add_argument("--lora-dir", default=DEFAULT_LORA_POLICY)
    p1.add_argument("--style", default="strict", choices=["strict", "assumptive"])
    p1.add_argument("--audit", action=argparse.BooleanOptionalAction, default=None,
                    help="Run strict audit/repair pass (default: on for strict, off for assumptive)")
    p1.add_argument("--text", default=None)
    p1.add_argument("--text-file", default=None)

    p2 = sub.add_parser("risk_narrative")
    p2.add_argument("--lora-dir", default=DEFAULT_LORA_RISK)
    p2.add_argument("--style", default="strict", choices=["strict", "assumptive"])
    p2.add_argument("--audit", action=argparse.BooleanOptionalAction, default=None,
                    help="Run strict audit/repair pass (default: on for strict, off for assumptive)")
    p2.add_argument("--text", default=None)
    p2.add_argument("--text-file", default=None)

    args = parser.parse_args()

    raw = read_text(args.text, args.text_file)

    # Safe defaults by style
    style = args.style
    audit = args.audit if args.audit is not None else (style == "strict")

    # Deterministic for strict by default
    temperature = args.temperature if args.temperature is not None else (0.0 if style == "strict" else 0.25)
    top_p = args.top_p if args.top_p is not None else (1.0 if style == "strict" else 0.9)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (650 if style == "assumptive" else 550)

    messages = build_messages(args.mode, style, raw)

    base_model, tokenizer = load_model_and_tokenizer(args.base_model, args.dtype)
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model.eval()

    # First pass
    out1 = generate(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=args.seed,
    )
    out1 = strip_preamble(out1)

    # Strict audit/repair pass (recommended)
    if audit and style == "strict":
        audit_msgs = build_audit_messages(args.mode, raw, out1)
        out2 = generate(
            model=model,
            tokenizer=tokenizer,
            messages=audit_msgs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=args.seed,
        )
        output = strip_preamble(out2)
    else:
        output = out1

    if args.json:
        print(json.dumps(
            {
                "mode": args.mode,
                "style": style,
                "audit": bool(audit and style == "strict"),
                "temperature": temperature,
                "top_p": top_p,
                "output": output,
            },
            ensure_ascii=False,
            indent=2
        ))
    else:
        print(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[cluster1_cli] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
