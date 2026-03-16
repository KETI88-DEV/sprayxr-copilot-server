#!/usr/bin/env python3
"""QLoRA fine-tuning for Qwen3-4B using peft + trl SFTTrainer."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def _patch_torch_set_submodule() -> None:
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def _set_submodule(self: torch.nn.Module, target: str, module: torch.nn.Module) -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)

    torch.nn.Module.set_submodule = _set_submodule  # type: ignore[attr-defined]
    logger.info("Patched torch.nn.Module.set_submodule for torch %s", torch.__version__)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_sft_dataset(path: Path) -> Dataset:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    logger.info("Loaded %d SFT records from %s", len(records), path)
    return Dataset.from_list(records)


def build_model_and_tokenizer(config: Dict[str, Any]) -> tuple:
    _patch_torch_set_submodule()

    model_cfg = config["model"]
    model_path = model_cfg["model_path"]

    compute_dtype = getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    logger.info("Loading model in 4-bit NF4 from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=model_cfg.get("device_map", "auto"),
        use_safetensors=model_cfg.get("use_safetensors", True),
        dtype=compute_dtype,
        attn_implementation="eager",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    vram_gb = torch.cuda.memory_allocated() / 1e9
    logger.info("Model loaded. VRAM used: %.2f GB", vram_gb)

    return model, tokenizer


TRAINING_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'system' %}"
    "{{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'user' %}"
    "{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message.content }}"
    "{%- endgeneration %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{{- '<|im_start|>user\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def _build_training_chat_template() -> str:
    return TRAINING_CHAT_TEMPLATE


def _verify_assistant_mask(tokenizer: Any) -> None:
    test_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = tokenizer.apply_chat_template(
        test_msgs, tokenize=True, return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = result.get("assistant_masks", [])
    if 1 not in mask:
        raise RuntimeError(
            "Training chat template failed: no assistant tokens detected. "
            "Check TRAINING_CHAT_TEMPLATE for missing {%% generation %%} markers."
        )
    assistant_ratio = sum(mask) / len(mask) if mask else 0
    logger.info(
        "Assistant-only loss enabled: %d/%d tokens (%.1f%%) will receive loss",
        sum(mask), len(mask), assistant_ratio * 100,
    )


def build_lora_config(config: Dict[str, Any]) -> LoraConfig:
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
    )


def build_training_args(config: Dict[str, Any]) -> SFTConfig:
    train_cfg = config["training"]
    output_cfg = config["output"]

    return SFTConfig(
        output_dir=output_cfg["checkpoint_dir"],
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=train_cfg.get("optim", "adamw_8bit"),
        logging_steps=train_cfg.get("logging_steps", 5),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        assistant_only_loss=True,
        seed=train_cfg.get("seed", 42),
        max_length=train_cfg.get("max_seq_length", 2048),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=0,
        packing=False,
    )


def run_training(
    model: Any,
    tokenizer: Any,
    lora_config: LoraConfig,
    training_args: SFTConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    output_cfg = config["output"]
    adapter_dir = Path(output_cfg["adapter_dir"])
    log_dir = Path(output_cfg.get("log_dir", "/logs"))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = {"messages", "chat_template_kwargs"}
    cols_to_drop = [c for c in train_dataset.column_names if c not in keep_cols]
    if cols_to_drop:
        train_dataset = train_dataset.remove_columns(cols_to_drop)
        val_dataset = val_dataset.remove_columns(cols_to_drop)
        logger.info("Dropped non-messages columns: %s", cols_to_drop)

    if "chat_template_kwargs" not in train_dataset.column_names:
        ct_kwargs = {"enable_thinking": False}
        train_dataset = train_dataset.map(lambda _: {"chat_template_kwargs": ct_kwargs})
        val_dataset = val_dataset.map(lambda _: {"chat_template_kwargs": ct_kwargs})
        logger.info("Injected chat_template_kwargs: %s", ct_kwargs)

    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))
    logger.info("First example roles: %s", [m["role"] for m in train_dataset[0]["messages"]])

    original_chat_template = tokenizer.chat_template
    tokenizer.chat_template = _build_training_chat_template()
    _verify_assistant_mask(tokenizer)

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting QLoRA training...")
    t0 = time.time()

    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM detected! Applying fallback settings...")
        torch.cuda.empty_cache()
        oom_cfg = config.get("oom_fallback", {})
        training_args.per_device_train_batch_size = oom_cfg.get("per_device_train_batch_size", 1)
        training_args.gradient_accumulation_steps = oom_cfg.get("gradient_accumulation_steps", 16)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )
        train_result = trainer.train()

    elapsed = time.time() - t0
    logger.info("Training completed in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    logger.info("Saving LoRA adapter to %s", adapter_dir)
    tokenizer.chat_template = original_chat_template
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log_history = trainer.state.log_history
    train_log_path = log_dir / "train_qlora.log.json"
    with train_log_path.open("w", encoding="utf-8") as fh:
        json.dump(log_history, fh, ensure_ascii=False, indent=2)
    logger.info("Training log saved to %s", train_log_path)

    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    result = {
        "elapsed_s": round(elapsed, 1),
        "elapsed_min": round(elapsed / 60, 1),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2),
        "adapter_dir": str(adapter_dir),
        "initial_train_loss": train_losses[0] if train_losses else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "eval_losses": eval_losses,
        "total_steps": trainer.state.global_step,
        "epochs_completed": train_result.metrics.get("epoch", 0),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }

    summary_path = adapter_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    logger.info("Training summary saved to %s", summary_path)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning with peft + trl SFTTrainer."
    )
    parser.add_argument(
        "--config",
        default="/workspace/trainer/configs/qlora_config.yaml",
        type=Path,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.config.is_file():
        logger.error("Config not found: %s", args.config)
        return 1

    config = load_config(args.config)

    data_cfg = config["data"]
    train_path = Path(data_cfg["train_path"])
    val_path = Path(data_cfg["val_path"])

    if not train_path.is_file():
        logger.error("Training data not found: %s", train_path)
        return 1
    if not val_path.is_file():
        logger.error("Validation data not found: %s", val_path)
        return 1

    train_dataset = load_sft_dataset(train_path)
    val_dataset = load_sft_dataset(val_path)

    model, tokenizer = build_model_and_tokenizer(config)
    lora_config = build_lora_config(config)
    training_args = build_training_args(config)

    result = run_training(
        model=model,
        tokenizer=tokenizer,
        lora_config=lora_config,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )

    print("\n" + "=" * 60)
    print("QLoRA Training Summary")
    print("=" * 60)
    print(f"  Training time:      {result['elapsed_min']:.1f} min")
    print(f"  Train samples:      {result['train_samples']}")
    print(f"  Trainable params:   {result['trainable_params']:,} / {result['total_params']:,} ({result['trainable_pct']}%)")
    print(f"  Initial train loss: {result['initial_train_loss']}")
    print(f"  Final train loss:   {result['final_train_loss']}")
    print(f"  Eval losses:        {result['eval_losses']}")
    print(f"  Peak VRAM:          {result['peak_vram_gb']} GB")
    print(f"  Adapter saved:      {result['adapter_dir']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
