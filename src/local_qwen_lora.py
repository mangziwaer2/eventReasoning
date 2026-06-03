from __future__ import annotations

from pathlib import Path
from typing import Any


class LoraUnavailable(RuntimeError):
    pass


def import_qwen_lora_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise LoraUnavailable(
            "Qwen LoRA training requires `peft`, `transformers`, and `torch` in the active environment."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model


def load_qwen_with_lora(
    model_path: Path,
    target_modules: list[str] | None = None,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
):
    torch, auto_model_cls, auto_tokenizer_cls, lora_config_cls, get_peft_model = import_qwen_lora_stack()

    tokenizer = auto_tokenizer_cls.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = auto_model_cls.from_pretrained(model_path, trust_remote_code=False)
    lora_config = lora_config_cls(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer, torch
