#!/usr/bin/env python3
"""
Cluster1 STRICT smoke gate.

- Loads BOTH adapters into ONE PEFT model and switches adapters per task.
- Runs strict_generate_with_repairs() so smoke matches production STRICT behavior.
- Writes JSONL report with excerpts + flags + repair meta.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cluster1_cli import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    load_model_and_tokenizer,
    strict_generate_with_repairs,
)
from unit_testing.strict_rules import compute_flags, any_flagged  # noqa: E402


def _short(s: str, n: int = 600) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_multi_adapter_model(base_model, policy_lora: str, risk_lora: str, control_lora: str | None = None):
    model = PeftModel.from_pretrained(base_model, policy_lora, adapter_name="policy_refactor")
    model.load_adapter(risk_lora, adapter_name="risk_narrative")
    if control_lora:
        model.load_adapter(control_lora, adapter_name="control_narrative")
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-data", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--audit", action=argparse.BooleanOptionalAction, default=True)  # kept for compat (strict uses repairs)
    ap.add_argument("--policy-lora", required=True)
    ap.add_argument("--risk-lora", required=True)
    ap.add_argument("--control-lora", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=650)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    smoke_path = Path(args.smoke_data)
    if not smoke_path.is_absolute():
        smoke_path = (REPO_ROOT / smoke_path).resolve()

    reports_dir = (REPO_ROOT / "reports").resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"smoke_report_{ts}.jsonl"

    print("=== Cluster1 SMOKE GATE (strict) ===")
    print(f"Repo:        {REPO_ROOT}")
    print(f"Smoke data:  {smoke_path}")
    print(f"Report:      {report_path}")
    print(f"Base model:  {args.base_model}")
    print(f"Dtype:       {args.dtype}")
    print(f"Audit:       {args.audit}")
    print(f"Policy LoRA: {args.policy_lora}")
    print(f"Risk LoRA:   {args.risk_lora}")
    if args.control_lora:
        print(f"Control LoRA:{args.control_lora}")
    print("")

    rows = read_jsonl(smoke_path)
    if not rows:
        print("No smoke rows found.")
        return 2

    base_model, tokenizer = load_model_and_tokenizer(args.base_model, args.dtype)

    print("--- Loading adapters into a single PEFT model ---")
    model = load_multi_adapter_model(base_model, args.policy_lora, args.risk_lora, args.control_lora)
    model.eval()

    failing_ids: List[str] = []
    counts = {"numbers": 0, "roles": 0, "data_classes": 0, "cadence": 0, "legal": 0}

    with report_path.open("w", encoding="utf-8") as outf:
        for rec in rows:
            cid = rec.get("id", "unknown")
            task = rec.get("task")
            raw = rec.get("raw", "")

            if task not in ("policy_refactor", "risk_narrative", "control_narrative"):
                raise SystemExit(f"Unknown task in {cid}: {task}")

            adapter = task
            if task == "control_narrative" and not args.control_lora:
                # until a dedicated control adapter exists, reuse risk adapter
                adapter = "risk_narrative"

            try:
                model.set_adapter(adapter)
            except Exception:
                model.active_adapter = adapter  # type: ignore

            out, meta = strict_generate_with_repairs(
                task=task,
                raw=raw,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )

            flags = compute_flags(out, raw)
            failed = any_flagged(flags)

            if failed:
                failing_ids.append(cid)
                if flags.get("numbers_not_in_input"):
                    counts["numbers"] += 1
                if flags.get("roles_not_in_input"):
                    counts["roles"] += 1
                if flags.get("data_classes_not_in_input"):
                    counts["data_classes"] += 1
                if flags.get("cadence_not_in_input"):
                    counts["cadence"] += 1
                if flags.get("legal_claims_not_in_input"):
                    counts["legal"] += 1

            outf.write(
                json.dumps(
                    {
                        "id": cid,
                        "task": task,
                        "pass": not failed,
                        "flags": flags,
                        "raw_excerpt": _short(raw, 350),
                        "output_excerpt": _short(out, 1200),
                        "meta": meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=== SMOKE SUMMARY ===")
    print(f"Cases: {len(rows)}")
    print(f"Failing cases: {len(failing_ids)}")
    print("By category (count of cases triggering):")
    print(f"  - numbers:      {counts['numbers']}")
    print(f"  - roles:        {counts['roles']}")
    print(f"  - data_classes: {counts['data_classes']}")
    print(f"  - cadence:      {counts['cadence']}")
    print(f"  - legal:        {counts['legal']}")
    print(f"Report written to: {report_path}")

    if failing_ids:
        print("Failing IDs:")
        for x in failing_ids:
            print(f"  - {x}")
        print("=== RESULT: FAIL ===")
        return 1

    print("=== RESULT: PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
