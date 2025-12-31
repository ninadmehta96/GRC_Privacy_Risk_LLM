#!/usr/bin/env python3
"""
Testing/test_cluster1_policy_refactor.py

Runs ONE policy_refactor generation using the same prompt + generation path as cluster1_cli.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    load_model_and_tokenizer,
    build_messages,
    build_audit_messages,
    generate,
    strip_preamble,
    read_text as _read_text_exactly_one,
)


SAMPLE_POLICY_DRAFT = """The Company shall endeavor at all times to delete, anonymize or otherwise dispose of information in a timely manner. When data is longer used, ..."""


def read_optional_text(text: str | None, text_file: str | None) -> str:
    if text is None and text_file is None:
        return SAMPLE_POLICY_DRAFT
    return _read_text_exactly_one(text, text_file)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--lora-dir", default=DEFAULT_LORA_POLICY)
    ap.add_argument("--style", default="strict", choices=["strict", "assumptive"])
    ap.add_argument("--audit", action=argparse.BooleanOptionalAction, default=None)

    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)

    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--debug-prompt", action="store_true")
    args = ap.parse_args()

    raw = read_optional_text(args.text, args.text_file)

    style = args.style
    audit = args.audit if args.audit is not None else (style == "strict")

    temperature = args.temperature if args.temperature is not None else (0.0 if style == "strict" else 0.25)
    top_p = args.top_p if args.top_p is not None else (1.0 if style == "strict" else 0.9)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (650 if style == "assumptive" else 550)

    print("Loading base model + tokenizer...")
    base_model, tokenizer = load_model_and_tokenizer(args.base_model, args.dtype)
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model.eval()

    messages = build_messages("policy_refactor", style, raw)
    out1 = generate(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=args.seed,
        debug_prompt=args.debug_prompt,
        label="PASS 1",
    )
    out1 = strip_preamble(out1)

    final_out = out1
    if audit and style == "strict":
        audit_msgs = build_audit_messages("policy_refactor", raw, out1)
        out2 = generate(
            model=model,
            tokenizer=tokenizer,
            messages=audit_msgs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=args.seed,
            debug_prompt=args.debug_prompt,
            label="AUDIT PASS",
        )
        final_out = strip_preamble(out2)

    print("\n=== MODEL OUTPUT ===\n")
    print(final_out)


if __name__ == "__main__":
    main()
