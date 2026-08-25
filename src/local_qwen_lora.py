from __future__ import annotations

from pathlib import Path
from typing import Any


class LoraUnavailable(RuntimeError):
    pass


def _disable_incompatible_torchao_for_peft() -> None:
    try:
        import peft.import_utils as peft_import_utils
    except ImportError:
        return

    try:
        peft_import_utils.is_torchao_available()
    except ImportError:
        peft_import_utils.is_torchao_available = lambda: False


def import_qwen_lora_stack() -> tuple[Any, Any, Any, Any, Any, Any]:

    import torch
    _disable_incompatible_torchao_for_peft()
    from peft import LoraConfig
    from peft import PeftModel
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model, PeftModel


def _inference_model_kwargs(torch: Any) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {"trust_remote_code": False, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16
    return model_kwargs


def _move_to_device(model, torch):
    if torch.cuda.is_available():
        return model.to("cuda")
    return model


def _require_local_adapter(adapter_path: Path) -> str:
    adapter = Path(adapter_path)
    config_path = adapter / "adapter_config.json"
    if not adapter.is_dir() or not config_path.is_file():
        raise LoraUnavailable(
            f"LoRA adapter directory is incomplete: {adapter}. Expected {config_path}. "
            "Use a best_adapter/latest_adapter directory created by train_coarse_graph_qwen.py, "
            "or omit the adapter flag to run frozen Qwen3-4B."
        )
    return str(adapter)


def load_qwen_with_lora(
    model_path: Path,
    adapter_path: Path | None = None,
    target_modules: list[str] | None = None,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
):
    torch, auto_model_cls, auto_tokenizer_cls, lora_config_cls, get_peft_model, peft_model_cls = import_qwen_lora_stack()

    adapter_id = _require_local_adapter(adapter_path) if adapter_path is not None else None
    tokenizer_source = adapter_path if adapter_path is not None and (adapter_path / "tokenizer_config.json").exists() else model_path
    tokenizer = auto_tokenizer_cls.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = auto_model_cls.from_pretrained(model_path, **_inference_model_kwargs(torch))
    if adapter_path is not None:
        try:
            model = peft_model_cls.from_pretrained(model, adapter_id, is_trainable=True)
        except TypeError:
            model = peft_model_cls.from_pretrained(model, adapter_id)
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = True
    else:
        lora_config = lora_config_cls(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model = _move_to_device(model, torch)
    return model, tokenizer, torch


def load_trained_qwen_lora(
    base_model_path: Path,
    adapter_path: Path,
):
    torch, auto_model_cls, auto_tokenizer_cls, _, _, peft_model_cls = import_qwen_lora_stack()

    adapter_id = _require_local_adapter(adapter_path)
    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model_path
    tokenizer = auto_tokenizer_cls.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = auto_model_cls.from_pretrained(base_model_path, **_inference_model_kwargs(torch))
    model = peft_model_cls.from_pretrained(model, adapter_id)
    model = _move_to_device(model, torch)
    return model, tokenizer, torch


def load_qwen_for_inference(
    base_model_path: Path,
    adapter_path: Path | None = None,
):
    """Load a causal LM for inference, optionally attaching a frozen LoRA adapter."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise LoraUnavailable(
            "Inference requires `torch` and `transformers`."
        ) from exc

    adapter_id = _require_local_adapter(adapter_path) if adapter_path is not None else None
    tokenizer_source = adapter_path if adapter_path is not None and (adapter_path / "tokenizer_config.json").exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_path, **_inference_model_kwargs(torch))
    if adapter_path is not None:
        try:
            _disable_incompatible_torchao_for_peft()
            from peft import PeftModel
        except ImportError as exc:
            raise LoraUnavailable(
                "Loading a coarse LoRA adapter requires `peft`."
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_id)
    model = _move_to_device(model, torch)
    return model, tokenizer, torch
