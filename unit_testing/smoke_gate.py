#!/usr/bin/env python3
"""
unit_testing/smoke_gate.py (v5)

SMOKE GATE for Cluster1 STRICT mode.

What this gate enforces:
- For each smoke case, run STRICT generation (with audit)
- Apply strict hallucination checks:
    numbers/roles/data classes/cadence/legal terms introduced that are not in input
- Fail CI if ANY case fails.

Key engineering detail:
- Loads BOTH adapters into ONE PEFT model and switches adapters per task (no adapter stacking).
- If cluster1_cli.py exposes strict_generate_with_repairs(...), we use it so SMOKE matches
  the production STRICT pipeline (generate -> audit -> diff-based repair -> verify).

Evaluator loading:
- Prefer unit_testing/strict_rules.py if present (lightweight, CPU-friendly rules module)
- Otherwise fall back to unit_testing/eval_cluster1_strict.py (legacy)

Usage:
  python unit_testing/smoke_gate.py \
    --smoke-data data/smoke/cluster1_smoke_inputs.jsonl \
    --policy-lora adapters/...policy... \
    --risk-lora adapters/...risk...

Exit code:
  0 on pass, 1 on fail.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def _repo_root() -> Path:
    # unit_testing/smoke_gate.py -> repo root
    return Path(__file__).resolve().parents[1]


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load module: {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_eval_mod(repo: Path):
    """
    Prefer strict_rules.py if present; else fall back to eval_cluster1_strict.py.
    """
    strict_path = repo / "unit_testing" / "strict_rules.py"
    if strict_path.exists():
        return _load_module_from_path("strict_rules", strict_path)

    eval_path = repo / "unit_testing" / "eval_cluster1_strict.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluator: {eval_path}")
    return _load_module_from_path("eval_cluster1_strict", eval_path)


def _pick_diff_fns(eval_mod):
    diff_terms_fn = getattr(eval_mod, "diff_terms", None) or getattr(eval_mod, "diff_terms_gap", None)
    diff_numbers_fn = getattr(eval_mod, "diff_numbers", None) or getattr(eval_mod, "diff_numbers_gap", None)
    if diff_terms_fn is None or diff_numbers_fn is None:
        raise AttributeError("Evaluator missing diff_terms/diff_numbers (or *_gap variants).")
    return diff_terms_fn, diff_numbers_fn


def _compute_flags(eval_mod, output: str, raw: str) -> Dict[str, List[str]]:
    # Prefer the evaluator's canonical flag computation if it exists.
    # This matters because compute_flags() is where alias-handling and
    # PII-evidence heuristics live (e.g., personal information <-> personal data).
    compute_flags_fn = getattr(eval_mod, "compute_flags", None)
    if callable(compute_flags_fn):
        return compute_flags_fn(output, raw)

    # Fallback for older evaluators: use diff_* helpers + explicit term lists.
    diff_terms_fn, diff_numbers_fn = _pick_diff_fns(eval_mod)

    role_terms = getattr(eval_mod, "ROLE_TERMS", None)
    data_terms = getattr(eval_mod, "DATA_CLASS_TERMS", None)
    cadence_terms = getattr(eval_mod, "CADENCE_TERMS", None)
    # If LEGAL_CLAIM_TERMS is missing, fall back to LEGAL_TERMS.
    legal_terms = getattr(eval_mod, "LEGAL_CLAIM_TERMS", None) or getattr(eval_mod, "LEGAL_TERMS", None)

    if any(x is None for x in [role_terms, data_terms, cadence_terms, legal_terms]):
        raise AttributeError("Evaluator missing one or more TERM lists.")

    return {
        "numbers_not_in_input": diff_numbers_fn(output, raw),
        "roles_not_in_input": diff_terms_fn(output, raw, role_terms),
        "data_classes_not_in_input": diff_terms_fn(output, raw, data_terms),
        "cadence_not_in_input": diff_terms_fn(output, raw, cadence_terms),
        "legal_claims_not_in_input": diff_terms_fn(output, raw, legal_terms),
    }


def _any_fail(flags: Dict[str, List[str]]) -> bool:
    return any(bool(v) for v in flags.values())


def _short(s: str, n: int = 600) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _load_base_model_and_tokenizer(base_model: str, dtype: str):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    # Transformers has flipped between `torch_dtype` and `dtype` depending on version.
    # Try `dtype` first to avoid warnings; fall back to `torch_dtype` on older versions.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch_dtype,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
    model.eval()
    return model, tok


def _load_adapters_single_model(base_model, policy_lora: str, risk_lora: str):
    """
    Load both adapters into a single PEFT model and name them by task.
    """
    # First adapter
    model = PeftModel.from_pretrained(base_model, policy_lora, adapter_name="policy_refactor")
    # Second adapter
    model.load_adapter(risk_lora, adapter_name="risk_narrative")
    return model


def _strict_generate_for_case(
    *,
    cli_mod,
    model,
    tokenizer,
    task: str,
    raw: str,
    audit: bool,
    max_new_tokens: int,
    seed: int,
) -> str:
    """
    Use the production STRICT pipeline if available.
    Falls back to (strict generation + audit) using cluster1_cli helpers otherwise.
    """
    # Preferred: production pipeline
    fn = getattr(cli_mod, "strict_generate_with_repairs", None)
    if callable(fn):
        try:
            out, _dbg = fn(
                task=task,
                raw=raw,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                seed=seed,
                audit=audit,
                debug_prompt=False,
            )
            return out
        except TypeError:
            # signature mismatch; fall back
            pass

    # Fallback: strict + audit only
    build_messages = getattr(cli_mod, "build_messages")
    build_audit_messages = getattr(cli_mod, "build_audit_messages", None)
    generate = getattr(cli_mod, "generate")
    strip_preamble = getattr(cli_mod, "strip_preamble", lambda x: (x or "").strip())

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

    if audit and callable(build_audit_messages):
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

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-data", required=True)
    ap.add_argument("--policy-lora", required=True)
    ap.add_argument("--risk-lora", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--audit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-new-tokens", type=int, default=650)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    repo = _repo_root()
    smoke_path = (repo / args.smoke_data).resolve()
    reports_dir = (repo / args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"smoke_report_{_now_stamp()}.jsonl"

    # Ensure we can import repo-local files
    sys.path.insert(0, str(repo))

    # Load modules
    eval_mod = _load_eval_mod(repo)
    cli_path = repo / "cluster1_cli.py"
    if not cli_path.exists():
        raise FileNotFoundError(f"Missing cluster1_cli.py at repo root: {cli_path}")
    cli_mod = _load_module_from_path("cluster1_cli", cli_path)

    print("=== Cluster1 SMOKE GATE (strict) ===")
    print(f"Repo:        {repo}")
    print(f"Smoke data:  {smoke_path}")
    print(f"Report:      {report_path}")
    print(f"Base model:  {args.base_model}")
    print(f"Dtype:       {args.dtype}")
    print(f"Audit:       {args.audit}")
    print(f"Policy LoRA: {args.policy_lora}")
    print(f"Risk LoRA:   {args.risk_lora}\n")

    rows = _read_jsonl(smoke_path)

    base_model, tokenizer = _load_base_model_and_tokenizer(args.base_model, args.dtype)

    print("--- Loading adapters into a single PEFT model ---")
    model = _load_adapters_single_model(base_model, args.policy_lora, args.risk_lora)
    model.eval()

    # Run
    failing: List[str] = []
    counts = {"numbers": 0, "roles": 0, "data_classes": 0, "cadence": 0, "legal": 0}
    total = 0
    failed_cases = 0

    with report_path.open("w", encoding="utf-8") as outf:
        for r in rows:
            total += 1
            cid = r.get("id", f"case_{total}")
            task = r.get("task")
            raw = r.get("raw", "")
            if task not in ("policy_refactor", "risk_narrative"):
                raise ValueError(f"Bad task in smoke row {cid}: {task}")

            # switch adapter
            try:
                model.set_adapter(task)
            except Exception:
                # older PEFT uses active_adapter attr
                model.active_adapter = task  # type: ignore

            out = _strict_generate_for_case(
                cli_mod=cli_mod,
                model=model,
                tokenizer=tokenizer,
                task=task,
                raw=raw,
                audit=bool(args.audit),
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )

            flags = _compute_flags(eval_mod, out, raw)
            failed = _any_fail(flags)

            if failed:
                failed_cases += 1
                failing.append(cid)
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

            outf.write(json.dumps({
                "id": cid,
                "task": task,
                "failed": failed,
                "flags": flags,
                "raw_excerpt": _short(raw, 300),
                "output_excerpt": _short(out, 900),
            }, ensure_ascii=False) + "\n")

    print("=== SMOKE SUMMARY ===")
    print(f"Cases: {total}")
    print(f"Failing cases: {failed_cases}")
    print("By category (count of cases triggering):")
    print(f"  - numbers:      {counts['numbers']}")
    print(f"  - roles:        {counts['roles']}")
    print(f"  - data_classes: {counts['data_classes']}")
    print(f"  - cadence:      {counts['cadence']}")
    print(f"  - legal:        {counts['legal']}")
    print(f"Report written to: {report_path}")

    if failing:
        print("Failing IDs:")
        for cid in failing:
            print(f"  - {cid}")
        print("=== RESULT: FAIL ===")
        return 1

    print("=== RESULT: PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
