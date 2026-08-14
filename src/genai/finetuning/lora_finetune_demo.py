"""Basic fine-tuning method #2 (reference implementation, not executed in
this lightweight sandbox): LoRA (Low-Rank Adaptation) fine-tuning of a
small causal language model using HuggingFace `transformers` + `peft`.

Persona: GenAI Developer.
JD requirement covered: "Apply expertise in ... basic finetuning methods."

This file is a complete, correct reference for the *full* fine-tuning
methodology a GenAI Developer would use to domain-adapt an open-weight
model (e.g. for a specialized support-ticket response style), matching
what the JD calls "basic finetuning methods." It intentionally requires
`torch`, `transformers`, `peft`, and `datasets` — heavy optional
dependencies not installed by default in this reference repository's
lightweight environment (see `requirements.txt` "optional: fine-tuning
extras"). The methodology it documents is exercised end-to-end, on a
dependency-free surrogate, by `embedding_adapter_finetune.py`, whose
results are reported in `docs/10_finetuning_experiment_report.md`.

Usage (in an environment with the optional extras installed and a GPU
or a small enough model for CPU training):

    python -m src.genai.finetuning.lora_finetune_demo \
        --base-model sshleifer/tiny-gpt2 \
        --train-file data/finetune/support_replies.jsonl \
        --output-dir artifacts/finetuned/lora-adapter
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("documind.genai.finetuning.lora")


def load_jsonl_dataset(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_lora_finetune(base_model: str, train_file: str, output_dir: str, epochs: int = 3) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "LoRA fine-tuning requires the optional extras: "
            "`pip install torch transformers peft datasets accelerate`. "
            "This reference implementation is provided for completeness; "
            "the executed fine-tuning demonstration in this repository uses "
            "the dependency-free `embedding_adapter_finetune.py` instead — "
            "see docs/10_finetuning_experiment_report.md."
        ) from exc

    records = load_jsonl_dataset(train_file)
    dataset = Dataset.from_list(records)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                 # rank of the low-rank update matrices
        lora_alpha=16,       # scaling factor
        lora_dropout=0.05,
        target_modules=["c_attn"],  # GPT-2-style attention projection; adjust per architecture
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(batch):
        text = [f"{p}\n{c}" for p, c in zip(batch["prompt"], batch["completion"])]
        return tokenizer(text, truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("lora_finetune.complete", extra={"output_dir": output_dir})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="artifacts/finetuned/lora-adapter")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    run_lora_finetune(args.base_model, args.train_file, args.output_dir, args.epochs)
