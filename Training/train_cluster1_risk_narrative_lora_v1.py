#!/usr/bin/env python3
"""
train_cluster1_risk_narrative_lora_v1.py

Same as policy trainer, but for risk_narrative.
assistant_only_loss MUST be False because dataset is mapped to plain text.
"""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "cluster1_cli.py").exists():
            return p
    return start.parent


REPO_ROOT = find_repo_root(Path(__file__))


def _pick_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


DEFAULT_DATA_PATH = str(
    _pick_existing(
        [
            REPO_ROOT / "data" / "training_data" / "v2" / "grc_risk_narrative_v2_strict.jsonl",
            REPO_ROOT / "data" / "training_data" / "grc_risk_narrative_v2_strict.jsonl",
            REPO_ROOT / "data" / "training_data" / "v1" / "grc_risk_narrative_v1.jsonl",
            REPO_ROOT / "data" / "training_data" / "grc_risk_narrative_v1.jsonl",
        ]
    )
)

DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "adapters" / "mistral7b-cluster1-risk-narrative-lora-v2-strict")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="dev")

    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--num-train-epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--max-length", type=int, default=2048)

    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume-from-checkpoint", default=None)
    return ap.parse_args()


def _build_sft_config_compat(**kwargs) -> SFTConfig:
    sig = inspect.signature(SFTConfig.__init__)
    params = sig.parameters

    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    if "max_length" in kwargs and "max_length" not in params and "max_seq_length" in params:
        kwargs["max_seq_length"] = kwargs.pop("max_length")

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return SFTConfig(**filtered)


def _build_trainer_compat(**kwargs) -> SFTTrainer:
    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters

    if "processing_class" not in params and "tokenizer" in params and "processing_class" in kwargs:
        kwargs["tokenizer"] = kwargs.pop("processing_class")

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return SFTTrainer(**filtered)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_ds = load_dataset("json", data_files=args.data_path, split="train")

    def pick_split(ds, split_name: str):
        if "split" in ds.column_names:
            return ds.filter(lambda ex: ex.get("split", "train") == split_name)
        return ds if split_name == "train" else None

    train_ds = pick_split(raw_ds, args.train_split)
    eval_ds = pick_split(raw_ds, args.eval_split)

    if train_ds is None or len(train_ds) == 0:
        raise ValueError(f"No training rows found for split='{args.train_split}' in {args.data_path}")

    def format_example(example):
        msgs = example["messages"]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    if eval_ds is not None and len(eval_ds) > 0:
        eval_ds = eval_ds.map(format_example, remove_columns=eval_ds.column_names)
    else:
        eval_ds = None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else -1
    eval_strategy_value = "steps" if eval_ds is not None else "no"

    sft_args = _build_sft_config_compat(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs if max_steps == -1 else 1.0,
        max_steps=max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_length=args.max_length,
        packing=False,
        dataset_text_field="text",

        # IMPORTANT: must be False because dataset is plain text (not conversational)
        assistant_only_loss=False,

        report_to="none",
        evaluation_strategy=eval_strategy_value,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_total_limit=3,
        seed=args.seed,
    )

    trainer = _build_trainer_compat(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved risk_narrative LoRA to:", args.output_dir)


if __name__ == "__main__":
    main()
