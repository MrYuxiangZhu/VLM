from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

LOGGER = logging.getLogger("vlm.model")
SUPPORTED_FINETUNE_MODES = ("lora", "qlora", "dora", "full")


def _dtype(cfg: dict[str, Any]) -> torch.dtype:
    if cfg.get("bf16", True):
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            LOGGER.warning("BF16 is not supported by this GPU; falling back to FP16")
            return torch.float16
        return torch.bfloat16
    return torch.float16


def _load_kwargs(cfg: dict[str, Any], mode: str, dtype: torch.dtype) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": True}
    if mode == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA GPU")
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("QLoRA requires bitsandbytes; install requirements.txt") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        )
        kwargs["device_map"] = cfg.get("device_map", "auto")
    elif cfg.get("device_map"):
        kwargs["device_map"] = cfg["device_map"]
    elif not cfg.get("distributed", False) and mode != "full":
        kwargs["device_map"] = "auto"
    return kwargs


def build_model(cfg: dict[str, Any], mode: str):
    mode = mode.lower()
    if mode not in SUPPORTED_FINETUNE_MODES:
        raise ValueError(f"finetune mode must be one of {SUPPORTED_FINETUNE_MODES}, got {mode!r}")

    dtype = _dtype(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        **_load_kwargs(cfg, mode, dtype),
    )
    if getattr(model.config, "use_cache", False):
        model.config.use_cache = False

    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        LOGGER.info("finetune mode=full, all model parameters are trainable")
        return model

    if mode == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        )
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
    if cfg.get("gradient_checkpointing", True) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    LOGGER.info("finetune mode=%s", mode)
    return model
