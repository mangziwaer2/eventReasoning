from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from path_utils import resolve_repo_path


DEFAULT_LOCAL_MODEL = "models/Qwen3-4B-Thinking-2507"
DEFAULT_HF_MODEL = "Qwen/Qwen3-4B-Thinking-2507"


class Qwen3ThinkingUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class ThinkingResponse:
    thinking: str
    answer: str
    raw_text: str
    generated_tokens: int


def _version_tuple(version: str) -> tuple[int, int, int]:
    numbers = [int(item) for item in re.findall(r"\d+", version)[:3]]
    return tuple((numbers + [0, 0, 0])[:3])


def resolve_model_source(model_path: str, allow_download: bool) -> str:
    raw_path = Path(model_path).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(resolve_repo_path(raw_path))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    if allow_download:
        if model_path == DEFAULT_LOCAL_MODEL:
            return DEFAULT_HF_MODEL
        return model_path
    raise FileNotFoundError(
        f"Qwen3 model was not found at {model_path!r}. Download {DEFAULT_HF_MODEL!r} "
        f"to {DEFAULT_LOCAL_MODEL!r}, or pass --model-path {DEFAULT_HF_MODEL} --allow-download."
    )


def _import_runtime() -> tuple[Any, Any, Any, str]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise Qwen3ThinkingUnavailable(
            "Qwen3 inference requires torch, transformers>=4.51.0, and accelerate."
        ) from exc
    if _version_tuple(transformers.__version__) < (4, 51, 0):
        raise Qwen3ThinkingUnavailable(
            f"Qwen3 requires transformers>=4.51.0; found {transformers.__version__}."
        )
    return torch, AutoModelForCausalLM, AutoTokenizer, transformers.__version__


class Qwen3ThinkingChat:
    def __init__(
        self,
        model_path: str = DEFAULT_LOCAL_MODEL,
        allow_download: bool = False,
        device: str = "auto",
    ) -> None:
        torch, auto_model_cls, auto_tokenizer_cls, transformers_version = _import_runtime()
        self._torch = torch
        self.transformers_version = transformers_version
        self.model_source = resolve_model_source(model_path, allow_download=allow_download)
        local_files_only = not allow_download

        self.tokenizer = auto_tokenizer_cls.from_pretrained(
            self.model_source,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": False,
            "local_files_only": local_files_only,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if device == "auto" and torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        self.model = auto_model_cls.from_pretrained(self.model_source, **model_kwargs)

        if device != "auto":
            target_device = torch.device(device)
            self.model = self.model.to(target_device)
        elif not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()
        self.input_device = self._resolve_input_device()

        self.think_end_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        if (
            not isinstance(self.think_end_token_id, int)
            or self.think_end_token_id < 0
            or self.think_end_token_id == self.tokenizer.unk_token_id
        ):
            self.think_end_token_id = None

    def _resolve_input_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            return next(self.model.parameters()).device

    @property
    def device_summary(self) -> str:
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            devices = sorted({str(value) for value in device_map.values()})
            return ",".join(devices)
        return str(self.input_device)

    def _split_response(self, output_ids: list[int]) -> ThinkingResponse:
        split_index = 0
        if self.think_end_token_id is not None:
            for index, token_id in enumerate(output_ids):
                if token_id == self.think_end_token_id:
                    split_index = index + 1

        raw_text = self.tokenizer.decode(output_ids, skip_special_tokens=False).strip()
        if split_index > 0:
            thinking = self.tokenizer.decode(output_ids[:split_index], skip_special_tokens=True).strip()
            answer = self.tokenizer.decode(output_ids[split_index:], skip_special_tokens=True).strip()
        elif "</think>" in raw_text:
            thinking_text, answer_text = raw_text.rsplit("</think>", maxsplit=1)
            thinking = thinking_text.replace("<think>", "", 1).strip()
            answer = answer_text.strip()
        else:
            thinking = ""
            answer = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return ThinkingResponse(
            thinking=thinking,
            answer=answer,
            raw_text=raw_text,
            generated_tokens=len(output_ids),
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
    ) -> ThinkingResponse:
        if temperature <= 0:
            raise ValueError(
                "Qwen3-4B-Thinking-2507 should use sampling; temperature must be greater than zero."
            )
        if seed is not None:
            self._torch.manual_seed(seed)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(seed)

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {key: value.to(self.input_device) for key, value in model_inputs.items()}
        input_length = int(model_inputs["input_ids"].shape[-1])

        with self._torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        output_ids = generated[0, input_length:].tolist()
        return self._split_response(output_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive local chat with Qwen3-4B-Thinking-2507.")
    parser.add_argument("--model-path", default=DEFAULT_LOCAL_MODEL, help="Local model directory or Hugging Face model id.")
    parser.add_argument("--allow-download", action="store_true", help="Allow Transformers to download a Hugging Face model id.")
    parser.add_argument("--device", default="auto", help="Device such as auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--prompt", default=None, help="Optional one-shot prompt. Omit for interactive chat.")
    parser.add_argument("--system-prompt", default="You are a careful and helpful question-answering assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-thinking", action="store_true", help="Print the model's thinking section before its final answer.")
    return parser.parse_args()


def _print_response(response: ThinkingResponse, show_thinking: bool) -> None:
    if show_thinking:
        print("\n[thinking]")
        print(response.thinking or "(thinking block was not parsed)")
    print("\n[answer]")
    print(response.answer or "(empty final answer)")
    print(f"\n[generated_tokens] {response.generated_tokens}")


def main() -> None:
    args = parse_args()
    print(f"loading Qwen3 Thinking model from {args.model_path} ...", flush=True)
    chat = Qwen3ThinkingChat(
        model_path=args.model_path,
        allow_download=args.allow_download,
        device=args.device,
    )
    print(
        f"loaded | source={chat.model_source} | device={chat.device_summary} "
        f"| transformers={chat.transformers_version}",
        flush=True,
    )

    history: list[dict[str, str]] = []

    def ask(user_text: str) -> ThinkingResponse:
        messages = [{"role": "system", "content": args.system_prompt}, *history]
        messages.append({"role": "user", "content": user_text})
        response = chat.generate(
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed + len(history),
        )
        history.append({"role": "user", "content": user_text})
        # Qwen recommends keeping only the final answer, not thinking content, in multi-turn history.
        history.append({"role": "assistant", "content": response.answer})
        return response

    if args.prompt:
        _print_response(ask(args.prompt), show_thinking=args.show_thinking)
        return

    print("Interactive commands: /clear resets history, /exit quits.")
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("bye")
            break
        if user_text.lower() == "/clear":
            history.clear()
            print("history cleared")
            continue
        _print_response(ask(user_text), show_thinking=args.show_thinking)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, Qwen3ThinkingUnavailable, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
