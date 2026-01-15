from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class InferenceResult:
    """Normalized inference output.

    Public repo supports two modes:

    - offline: deterministic canned outputs (CI-safe; no GPU, no downloads)
    - hf:      real Transformers inference with optional LoRA (local use only)

    The point is to make inference an explicit boundary that can be swapped
    without changing the strict audit contract.
    """

    task: str
    mode: str
    model: str
    adapter: Optional[str]
    text: str


def _repo_root() -> Path:
    # src/inference.py -> repo root
    return Path(__file__).resolve().parents[1]


def _load_offline_outputs(repo: Path) -> dict:
    path = repo / "demo" / "offline_outputs.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing offline outputs file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_inference(
    *,
    task: str,
    case: str,
    raw: str,
    mode: str = "offline",
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    lora_dir: Optional[str] = None,
    max_new_tokens: int = 650,
    seed: int = 7,
) -> InferenceResult:
    """Run inference for a given task/case.

    Args:
        task: policy_refactor | risk_narrative
        case: good | trap
        raw:  input text (used in hf mode)
        mode: offline | hf
        base_model: HF model id for hf mode
        lora_dir: optional path to a LoRA adapter
    """
    task = (task or "").strip()
    case = (case or "").strip()
    mode = (mode or "offline").strip().lower()

    if mode == "offline":
        data = _load_offline_outputs(_repo_root())
        try:
            out_text = data[task][case]
        except KeyError as e:
            raise KeyError(f"offline_outputs.json missing key: {e}") from e
        return InferenceResult(task=task, mode=mode, model="offline_stub", adapter=None, text=out_text)

    if mode != "hf":
        raise ValueError(f"Unsupported mode: {mode}. Use offline or hf.")

    # Optional local Transformers inference.
    # NOTE: This is intentionally NOT used in public CI.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(
            "HF mode requires torch + transformers + peft installed. "
            "Install requirements-ml.lock.txt (or your local equivalent)."
        ) from e

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    adapter_name = None
    if lora_dir:
        adapter_name = Path(lora_dir).name
        model = PeftModel.from_pretrained(model, lora_dir)

    # Minimal prompt: public keeps this simple; private repo has richer templates/repair loops.
    prompt = (
        f"Task: {task}\n"
        "STRICT guardrails:\n"
        "- Do NOT invent timelines, owners/roles, systems, vendors, laws, or data classes.\n"
        "- Do NOT introduce any numbers not present in the source text.\n"
        "- If missing, write 'TBD' and list required inputs.\n\n"
        f"Source text:\n{raw}\n\n"
        "Return ONLY the result.\n"
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
    decoded = tok.decode(out_ids[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    return InferenceResult(task=task, mode=mode, model=base_model, adapter=adapter_name, text=decoded.strip())

