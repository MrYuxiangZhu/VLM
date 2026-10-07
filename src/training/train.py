from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments, set_seed

from ..data.datasets import build_dataset
from ..common.logging_utils import configure_logging


@dataclass
class VisionCollator:
    processor: Any
    max_length: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        texts, image_inputs, video_inputs = [], [], []
        for feature in features:
            messages = feature["messages"]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            images, videos = process_vision_info(messages)
            image_inputs.extend(images or [])
            video_inputs.extend(videos or [])
        kwargs: dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt", "truncation": True, "max_length": self.max_length}
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs
        batch = self.processor(**kwargs)
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    logger = configure_logging(cfg["log_dir"])
    logger.info("loaded config=%s", args.config)
    set_seed(cfg["seed"])

    processor = AutoProcessor.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)
    dtype = torch.bfloat16 if cfg["bf16"] else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"], task_type=TaskType.CAUSAL_LM,
    ))
    model.print_trainable_parameters()
    train_set = build_dataset(cfg["train_dataset"])
    eval_set = build_dataset(cfg["validation_dataset"])
    logger.info("dataset sizes train=%d validation=%d", len(train_set), len(eval_set))
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"], per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"], warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"], save_steps=cfg["save_steps"], eval_strategy="steps", eval_steps=cfg["eval_steps"],
        save_total_limit=cfg["save_total_limit"], bf16=cfg["bf16"], gradient_checkpointing=True, remove_unused_columns=False,
        report_to=cfg.get("report_to", "none"), logging_dir=cfg["log_dir"], dataloader_num_workers=cfg.get("num_workers", 2),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=eval_set, data_collator=VisionCollator(processor, cfg["max_seq_length"]))
    logger.info("training started")
    trainer.train(resume_from_checkpoint=os.getenv("RESUME_FROM_CHECKPOINT"))
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])
    logger.info("training finished output=%s", cfg["output_dir"])


if __name__ == "__main__":
    main()
