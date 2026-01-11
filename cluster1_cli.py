#!/usr/bin/env python3
"""
Cluster1 CLI router + strict repair pipeline.

Tasks:
  - policy_refactor
  - risk_narrative
  - control_narrative (structure + headings; can reuse risk adapter until trained)

Strict mode:
  generate (deterministic) -> verify -> scrub roles (deterministic) -> audit rewrite (deterministic) -> verify
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

DEFAULT_LORA_POLICY = "./mistral7b-cluster1-policy-refactor-lora-v1"
DEFAULT_LORA_RISK = "./mistral7b-cluster1-risk-narrative-lora-v1"
DEFAULT_LORA_CONTROL = DEFAULT_LORA_RISK  # until you train a dedicated control adapter


SYSTEM_POLICY = (
    "You are a senior GRC & privacy engineer. "
    "You refactor security/privacy policies to improve clarity and structure for engineers, "
    "product managers, and auditors — without changing obligations. "
    "You do not invent timelines, owners, laws, systems, vendors, or data classes."
)

SYSTEM_RISK = (
    "You are a senior risk & privacy leader. "
    "You turn raw incident/risk/DPIA information into concise executive narratives, "
    "clearly separating facts from unknowns. "
    "You do not invent numbers, dates, systems, laws, vendors, or roles."
)

SYSTEM_CONTROL = (
    "You are a senior GRC control owner and auditor. "
    "You produce audit-ready control narratives grounded strictly in the provided evidence/notes. "
    "You do not invent owners, cadence, scope, tools, systems, data classes, laws, or numbers."
)

CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)


# ---- Optional strict verifier support ----
try:
    from unit_testing.strict_rules import compute_flags, any_flagged, scrub_introduced_roles
except Exception:
    compute_flags = None
    any_flagged = None
    scrub_introduced_roles = None


def read_text(text: Optional[str], text_file: Optional[str]) -> str:
    if bool(text) == bool(text_file):
        raise ValueError("Provide exactly one of --text or --text-file.")
    if text_file:
        return Path(text_file).read_text(encoding="utf-8").strip()
    return (text or "").strip()


def strip_preamble(s: str) -> str:
    s = (s or "").strip()
    # drop leading separators
    s = re.sub(r"^\s*-{3,}\s*", "", s).strip()

    # remove common model preambles
    s = re.sub(r"(?is)^refactored\s+policy\s+section\s*:?\s*", "", s).strip()
    s = re.sub(r"(?is)^risk\s+narrative\s*:?\s*", "", s).strip()
    s = re.sub(r"(?is)^control\s+narrative\s*:?\s*", "", s).strip()

    return s.strip()


def normalize_hyphen_bullets(s: str) -> str:
    s = (s or "")
    return s.replace("•", "-").replace("–", "-").replace("—", "-")


def build_messages(task: str, style: str, raw: str) -> List[Dict[str, str]]:
    raw = (raw or "").strip()

    if task == "policy_refactor":
        user = (
            "Policy text (source of truth):\n\n"
            f"{raw}\n\n"
            "Task: Refactor into the following headings (exact):\n"
            "Scope\n"
            "Policy\n"
            "Timelines\n"
            "Exceptions & Approvals\n"
            "Engineering Notes\n"
            "Required Inputs\n\n"
        )
        if style == "strict":
            user += (
                "STRICT guardrails:\n"
                "- Do NOT invent timelines, owners/roles, systems, vendors, laws, or data classes.\n"
                "- Do NOT introduce any numbers not present in the source text.\n"
                "- If missing, write 'TBD' and list required info under Required Inputs.\n"
                "Return ONLY the refactored policy."
            )
        else:
            user += (
                "ASSUMPTIVE mode:\n"
                "- The main output must remain faithful (no invention).\n"
                "- You MAY add a final section: Suggested Defaults (Confirm) with clearly-marked recommendations.\n"
                "Return ONLY the refactored policy (including suggested defaults if used)."
            )
        return [{"role": "system", "content": SYSTEM_POLICY}, {"role": "user", "content": user}]

    if task == "risk_narrative":
        user = (
            "Notes (source of truth):\n\n"
            f"{raw}\n\n"
            "Task: Produce a 3–5 paragraph executive risk narrative.\n"
        )
        if style == "strict":
            user += (
                "STRICT guardrails:\n"
                "- Only state facts explicitly present in the notes.\n"
                "- Do NOT invent numbers, dates, systems, vendors, roles, laws, or data classes.\n"
                "- If unknown, say 'Not yet determined'.\n"
                "Return only the narrative."
            )
        else:
            user += (
                "ASSUMPTIVE mode:\n"
                "- Keep facts grounded in notes.\n"
                "- You MAY add a final paragraph labeled 'Recommendations (Non-binding)'.\n"
                "Return only the narrative."
            )
        return [{"role": "system", "content": SYSTEM_RISK}, {"role": "user", "content": user}]

    if task == "control_narrative":
        user = (
            "Evidence / notes (source of truth):\n\n"
            f"{raw}\n\n"
            "Task: Produce an audit-ready CONTROL NARRATIVE with these headings (exact):\n"
            "Control Objective\n"
            "Control Statement\n"
            "Scope\n"
            "How the Control Operates\n"
            "Frequency\n"
            "Roles & Responsibilities\n"
            "Evidence\n"
            "Exceptions\n"
            "Gaps / Evidence Needed\n\n"
        )
        if style == "strict":
            user += (
                "STRICT guardrails:\n"
                "- Only state facts explicitly present in the evidence/notes.\n"
                "- Do NOT invent owners/roles, cadence, scope, tools/systems/vendors, numbers, laws, or data classes.\n"
                "- If missing, write 'TBD' in that field and list required info under Gaps / Evidence Needed.\n"
                "Return ONLY the control narrative."
            )
        else:
            user += (
                "ASSUMPTIVE mode:\n"
                "- The main narrative must remain faithful (no invention).\n"
                "- You MAY add a final section: Suggested Defaults (Confirm) with clearly-marked recommendations.\n"
                "Return ONLY the control narrative."
            )
        return [{"role": "system", "content": SYSTEM_CONTROL}, {"role": "user", "content": user}]

    raise ValueError(f"Unknown task: {task}")


def build_audit_messages(task: str, raw: str, draft_output: str) -> List[Dict[str, str]]:
    raw = (raw or "").strip()
    draft_output = (draft_output or "").strip()

    if task == "policy_refactor":
        sysmsg = SYSTEM_POLICY
        user = (
            "You are auditing a refactored policy for strict non-invention.\n\n"
            "SOURCE (only truth):\n"
            f"{raw}\n\n"
            "DRAFT OUTPUT:\n"
            f"{draft_output}\n\n"
            "Fix rules:\n"
            "- Remove or rewrite any statement not supported by the source.\n"
            "- Remove any invented numbers, dates, timelines, roles, systems, vendors, laws, or data classes.\n"
            "- Keep the SAME headings and order.\n"
            "- Unknowns => 'TBD' and list needed info under Required Inputs.\n"
            "Return ONLY the corrected refactored policy."
        )
    elif task == "risk_narrative":
        sysmsg = SYSTEM_RISK
        user = (
            "You are auditing a risk narrative for strict non-invention.\n\n"
            "SOURCE (only truth):\n"
            f"{raw}\n\n"
            "DRAFT OUTPUT:\n"
            f"{draft_output}\n\n"
            "Fix rules:\n"
            "- Remove or rewrite any statement not supported by the notes.\n"
            "- Remove any invented numbers, dates, roles, systems, vendors, laws, or data classes.\n"
            "- Keep it 3–5 paragraphs.\n"
            "Return ONLY the corrected narrative."
        )
    elif task == "control_narrative":
        sysmsg = SYSTEM_CONTROL
        user = (
            "You are auditing a control narrative for strict non-invention.\n\n"
            "SOURCE (only truth):\n"
            f"{raw}\n\n"
            "DRAFT OUTPUT:\n"
            f"{draft_output}\n\n"
            "Fix rules:\n"
            "- Remove or rewrite any statement not supported by the evidence/notes.\n"
            "- Remove any invented numbers, owners/roles, cadence, scope, tools/systems/vendors, laws, or data classes.\n"
            "- Keep the SAME headings and order.\n"
            "- Missing fields => 'TBD' and list needed info under Gaps / Evidence Needed.\n"
            "Return ONLY the corrected control narrative."
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]


def load_model_and_tokenizer(base_model: str, dtype: str):
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # transformers changed torch_dtype -> dtype; support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map="auto")

    model.eval()
    return model, tok


def generate(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip the prompt if echoed
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    return decoded.strip()


def strict_generate_with_repairs(
    task: str,
    raw: str,
    model,
    tokenizer,
    max_new_tokens: int = 650,
    seed: int = 7,
    max_repairs: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """
    Strict pipeline:
      1) deterministic generation (no sampling)
      2) verify strict_rules (if available)
      3) deterministic scrub roles (if available)
      4) audit rewrite (LLM, deterministic)
      5) verify again
    """
    meta: Dict[str, Any] = {"repairs": []}

    msgs = build_messages(task, "strict", raw)
    out = generate(
        model=model,
        tokenizer=tokenizer,
        messages=msgs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
    )
    out = normalize_hyphen_bullets(strip_preamble(out))

    if compute_flags is None or any_flagged is None:
        meta["final_flags"] = {}
        return out, meta

    flags = compute_flags(out, raw)

    # deterministic role scrub (only roles)
    if any_flagged(flags) and scrub_introduced_roles is not None:
        introduced_roles = flags.get("roles_not_in_input", [])
        if introduced_roles:
            out2 = scrub_introduced_roles(out, introduced_roles)
            out2 = normalize_hyphen_bullets(strip_preamble(out2))
            flags2 = compute_flags(out2, raw)
            meta["repairs"].append(
                {"type": "scrub_roles", "roles": introduced_roles, "flags_before": flags, "flags_after": flags2}
            )
            out, flags = out2, flags2

    # audit rewrite loop
    for _ in range(max_repairs):
        if not any_flagged(flags):
            break
        audit_msgs = build_audit_messages(task, raw, out)
        out2 = generate(
            model=model,
            tokenizer=tokenizer,
            messages=audit_msgs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
        )
        out2 = normalize_hyphen_bullets(strip_preamble(out2))
        flags2 = compute_flags(out2, raw)
        meta["repairs"].append({"type": "audit_rewrite", "flags_before": flags, "flags_after": flags2})
        out, flags = out2, flags2

    meta["final_flags"] = flags
    return out, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster1 router CLI")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=650)

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pol = sub.add_parser("policy_refactor")
    p_pol.add_argument("--lora-dir", default=DEFAULT_LORA_POLICY)
    p_pol.add_argument("--style", default="strict", choices=["strict", "assumptive"])
    p_pol.add_argument("--text", default=None)
    p_pol.add_argument("--text-file", default=None)

    p_risk = sub.add_parser("risk_narrative")
    p_risk.add_argument("--lora-dir", default=DEFAULT_LORA_RISK)
    p_risk.add_argument("--style", default="strict", choices=["strict", "assumptive"])
    p_risk.add_argument("--text", default=None)
    p_risk.add_argument("--text-file", default=None)

    p_ctrl = sub.add_parser("control_narrative")
    p_ctrl.add_argument("--lora-dir", default=DEFAULT_LORA_CONTROL)
    p_ctrl.add_argument("--style", default="strict", choices=["strict", "assumptive"])
    p_ctrl.add_argument("--text", default=None)
    p_ctrl.add_argument("--text-file", default=None)

    args = ap.parse_args()
    raw = read_text(getattr(args, "text", None), getattr(args, "text_file", None))

    base_model, tokenizer = load_model_and_tokenizer(args.base_model, args.dtype)
    model = PeftModel.from_pretrained(base_model, getattr(args, "lora_dir"))
    model.eval()

    if args.style == "strict":
        out, meta = strict_generate_with_repairs(
            task=args.cmd,
            raw=raw,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    else:
        msgs = build_messages(args.cmd, "assumptive", raw)
        out = generate(
            model=model,
            tokenizer=tokenizer,
            messages=msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.25,
            top_p=0.9,
            seed=args.seed,
        )
        out = normalize_hyphen_bullets(strip_preamble(out))
    print("\n---\n" + out.strip() + "\n---\n")
    # Uncomment for debugging:
    # print(meta)  # debug: contains strict repair metadata when style=strict


if __name__ == "__main__":
    main()
