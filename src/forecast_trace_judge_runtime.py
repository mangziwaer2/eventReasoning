from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forecast_trace_judge import JUDGE_SYSTEM_PROMPT
from forecast_trace_judge import build_judge_prompt
from forecast_trace_judge import parse_judge_response
from local_qwen_lora import load_qwen_for_inference
from path_utils import resolve_repo_path


class FrozenQwenJudgeRuntime:
    """Frozen Qwen runtime with BatchEncoding-safe chat-template handling."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        max_new_tokens: int = 256,
        thinking: bool = False,
        cache_path: Path | str | None = None,
        max_context_chars: int = 12000,
    ) -> None:
        self.model_path = resolve_repo_path(str(model_path))
        self.max_new_tokens = max(32, int(max_new_tokens))
        self.thinking = bool(thinking)
        self.max_context_chars = max(1000, int(max_context_chars))
        self.cache_path = resolve_repo_path(str(cache_path)) if cache_path else None
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self.cache = {str(key): value for key, value in payload.items() if isinstance(value, dict)}
            except (OSError, json.JSONDecodeError):
                self.cache = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self.model, self.tokenizer, self.torch = load_qwen_for_inference(self.model_path)
        self.model.eval()
        self._loaded = True

    def _encode_messages(self, prompt: str):
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(encoded, "input_ids"):
                input_ids = encoded.input_ids
            elif isinstance(encoded, dict):
                input_ids = encoded["input_ids"]
            else:
                input_ids = encoded
        else:
            text = (
                f"<|im_start|>system\n{JUDGE_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if not hasattr(input_ids, "to"):
            input_ids = self.torch.tensor(input_ids, dtype=self.torch.long)
        return input_ids.to(next(self.model.parameters()).device)

    def _generate(self, prompt: str) -> str:
        self._load()
        input_ids = self._encode_messages(prompt)
        attention_mask = self.torch.ones_like(input_ids)
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with self.torch.no_grad():
            output = self.model.generate(**kwargs)
        return self.tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()

    def score(self, prediction: dict[str, Any], *, query: Any = None, graph: Any = None, context_prompt: str = "") -> dict[str, Any]:
        prompt = build_judge_prompt(
            prediction,
            query=query,
            graph=graph,
            context_prompt=context_prompt,
            max_context_chars=self.max_context_chars,
        )
        # The prompt is deterministic, so cache scores by the exact judge input.
        import hashlib

        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key in self.cache:
            return {**self.cache[key], "cached": True}
        try:
            raw = self._generate(prompt if self.thinking else prompt + "\n/no_think")
            result = parse_judge_response(raw)
        except Exception as exc:
            result = {
                "parsed_json": False,
                "support": 0.0,
                "causal": 0.0,
                "temporal": 0.0,
                "answer_link": 0.0,
                "hallucination": 1.0,
                "overall": 0.0,
                "reason": f"judge_error: {exc}",
                "raw_response": "",
            }
        self.cache[key] = result
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

