from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

LOGGER = logging.getLogger("vlm.model")
SUPPORTED_FINETUNE_MODES = ("lora", "qlora", "dora", "full")


def build_model(cfg: dict[str, Any], mode: str):
    mode = mode.lower()
    if mode not in SUPPORTED_FINETUNE_MODES:
        raise ValueError(f"finetune mode must be one of {SUPPORTED_FINETUNE_MODES}, got {mode!r}")

    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if mode == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA GPU and bitsandbytes")
        load_kwargs.update({
            "device_map": "auto",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            ),
        })
    elif not cfg.get("distributed", False):
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], **load_kwargs)
    if mode == "full":
        LOGGER.info("finetune mode=full, all model parameters are trainable")
        return model

    if mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias=cfg.get("lora_bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        use_dora=mode == "dora",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    LOGGER.info("finetune mode=%s", mode)
    return model
