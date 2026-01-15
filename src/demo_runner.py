from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from src.inference import run_inference
from unit_testing.strict_rules import compute_flags, any_flagged


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _fmt_check(name: str, passed: bool, evidence: List[str]) -> str:
    if passed:
        return f"✔ {name}"
    ev = ""
    if evidence:
        ev = f" ({', '.join(evidence[:4])}{'…' if len(evidence) > 4 else ''})"
    return f"✖ {name}{ev}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["policy_refactor", "risk_narrative"], required=True)
    ap.add_argument("--case", choices=["good", "trap"], default="good")
    ap.add_argument("--mode", choices=["offline", "hf"], default="offline")
    ap.add_argument("--lora-dir", default=None)
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    repo = _repo_root()
    inputs_dir = repo / "demo" / "inputs"

    input_path = inputs_dir / (
        "policy_good.txt" if args.task == "policy_refactor" and args.case == "good"
        else "policy_trap.txt" if args.task == "policy_refactor"
        else "risk_good.txt" if args.case == "good"
        else "risk_trap.txt"
    )

    raw = _read_text(input_path)

    inf = run_inference(
        task=args.task,
        case=args.case,
        raw=raw,
        mode=args.mode,
        base_model=args.base_model,
        lora_dir=args.lora_dir,
    )
    out_text = (inf.text or "").strip()

    flags: Dict[str, List[str]] = compute_flags(out_text, raw)
    failed = any_flagged(flags)

    print(f"CASE: {args.task}:{args.case}  (mode={inf.mode}, model={inf.model})")
    print(_fmt_check("no_numbers_added", not bool(flags["numbers_not_in_input"]), flags["numbers_not_in_input"]))
    print(_fmt_check("no_roles_added", not bool(flags["roles_not_in_input"]), flags["roles_not_in_input"]))
    print(_fmt_check("no_new_data_classes", not bool(flags["data_classes_not_in_input"]), flags["data_classes_not_in_input"]))
    print(_fmt_check("no_new_cadence", not bool(flags["cadence_not_in_input"]), flags["cadence_not_in_input"]))
    print(_fmt_check("no_new_legal_claims", not bool(flags["legal_claims_not_in_input"]), flags["legal_claims_not_in_input"]))

    verdict = "FAIL" if failed else "PASS"
    print(f"Result: {verdict}")

    reports_dir = (repo / args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"demo_{args.task}_{args.case}_{_now_stamp()}.json"

    report = {
        "task": args.task,
        "case": args.case,
        "mode": inf.mode,
        "model": inf.model,
        "adapter": inf.adapter,
        "verdict": verdict,
        "flags": flags,
        "raw": raw,
        "output": out_text,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

