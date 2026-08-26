from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_qwen_lora import LoraUnavailable, load_qwen_with_lora
from mirai_dataset import load_mirai_event_code_choices
from path_utils import REPO_ROOT, resolve_repo_path

CODEBOOK_SYSTEM = "Map a CAMEO event base code to its canonical event description. Return JSON only."
FORECAST_SYSTEM = "Predict future CAMEO event base codes from historical events and a causal graph. Return JSON only."


@dataclass(slots=True)
class Example:
    sample_id: str
    prompt: str
    completion: str


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.streams[0], name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage MIRAI code and answer-list SFT.")
    parser.add_argument("--stage", choices=["codebook", "forecast"], required=True)
    parser.add_argument("--input", nargs="+", default=[str(REPO_ROOT / "outputs" / "grpo_context_mirai_rule_train_no_refine" / "grpo_context.jsonl")])
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MIRAI_data.zip"))
    parser.add_argument("--model-path", default=str(REPO_ROOT / "models" / "Qwen3-4B"))
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=int, default=8)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--max-sequence-length", type=int, default=2304, help="Maximum prompt plus target tokens used for SFT.")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSONL.") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_codebook(dataset_path: Path) -> dict[str, str]:
    choices = load_mirai_event_code_choices(dataset_path)
    codebook = {
        str(item.get("event_code", "")).strip(): str(item.get("description", "")).strip()
        for item in choices
        if str(item.get("event_code", "")).strip() and str(item.get("description", "")).strip()
    }
    if not codebook:
        raise RuntimeError(f"No codebook entries found in {dataset_path}.")
    return dict(sorted(codebook.items()))


def codebook_examples(codebook: dict[str, str]) -> list[Example]:
    return [
        Example(
            f"codebook:{code}",
            f"Event base code: {code}",
            json.dumps({"event_code": code, "event_description": description}, ensure_ascii=False, separators=(",", ":")),
        )
        for code, description in codebook.items()
    ]


def forecast_examples(rows: list[dict[str, Any]], codebook: dict[str, str]) -> tuple[list[Example], int]:
    examples, skipped = [], 0
    for index, row in enumerate(rows, start=1):
        query = row.get("mirai_query", {})
        codes = query.get("answer_list", []) if isinstance(query, dict) else []
        codes = sorted({str(code).strip() for code in codes if str(code).strip()}) if isinstance(codes, list) else []
        raw_prompt = str(row.get("forecast_prompt", "")).strip()
        context_start = raw_prompt.find("QueryId:")
        context = raw_prompt[context_start:].strip() if context_start >= 0 else raw_prompt
        if not codes or not context:
            continue
        if any(code not in codebook for code in codes):
            skipped += 1
            continue
        completion = json.dumps(
            {"answers": [{"event_code": code, "event_description": codebook[code]} for code in codes]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = (
            "Predict every likely future CAMEO event-base code supported by the historical events and causal graph. "
            "Return JSON only with this schema:\n"
            '{"answers":[{"event_code":"three digit code","event_description":"canonical description"}]}\n\n'
            + context
        )
        examples.append(Example(str(row.get("query_id", f"row_{index}")), prompt, completion))
    return examples, skipped


def split_examples(examples: list[Example], ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    if not 0.0 <= ratio < 0.5:
        raise ValueError("--validation-ratio must be in [0.0, 0.5).")
    if len(examples) < 2 or ratio == 0:
        return examples, []
    ids = list(range(len(examples)))
    random.Random(seed).shuffle(ids)
    validation_ids = set(ids[: max(1, round(len(examples) * ratio))])
    return (
        [item for index, item in enumerate(examples) if index not in validation_ids],
        [item for index, item in enumerate(examples) if index in validation_ids],
    )


def _as_token_ids(encoded: Any, *, source: str) -> list[int]:
    # Transformers versions may return a list, tensor, or BatchEncoding here.
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = list(encoded[0])
    if not isinstance(encoded, list):
        raise ValueError(f"{source} did not return a token-id sequence.")
    try:
        return [int(token_id) for token_id in encoded]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} returned non-integer token IDs.") from exc


def message_ids(tokenizer, messages: list[dict[str, str]], generation: bool) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation)
        return _as_token_ids(encoded, source="tokenizer.apply_chat_template")
    text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
    if generation:
        text += "\nassistant:"
    return _as_token_ids(tokenizer.encode(text, add_special_tokens=True), source="tokenizer.encode")


def encode(tokenizer, item: Example, system: str, max_prompt: int, max_completion: int, max_sequence: int) -> dict[str, Any]:
    user_prompt = item.prompt.rstrip()
    if not user_prompt.endswith("/no_think"):
        user_prompt += "\n/no_think"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    prefix = message_ids(tokenizer, messages, True)
    full = message_ids(tokenizer, messages + [{"role": "assistant", "content": item.completion}], False)
    if full[: len(prefix)] == prefix:
        answer = full[len(prefix) :]
    else:
        answer = _as_token_ids(tokenizer.encode(item.completion, add_special_tokens=False), source="tokenizer.encode")
        if tokenizer.eos_token_id is not None:
            answer.append(tokenizer.eos_token_id)
    original_length = len(answer)
    answer = answer[:max_completion]
    prompt_budget = min(max_prompt, max(1, max_sequence - len(answer)))
    prompt = prefix[-prompt_budget:]
    return {
        "input_ids": prompt + answer,
        "labels": [-100] * len(prompt) + answer,
        "completion_tokens": original_length,
        "truncated": original_length > len(answer),
    }


def collate(items: list[dict[str, Any]], tokenizer, torch) -> dict[str, Any]:
    width = max(len(item["input_ids"]) for item in items)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids, masks, labels = [], [], []
    for item in items:
        count = width - len(item["input_ids"])
        input_ids.append([pad] * count + item["input_ids"])
        masks.append([0] * count + [1] * len(item["input_ids"]))
        labels.append([-100] * count + item["labels"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def loss_on(model, items, tokenizer, torch, device, batch_size) -> float | None:
    if not items:
        return None
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = collate(items[start : start + batch_size], tokenizer, torch)
            output = model(**{key: value.to(device) for key, value in batch.items()})
            values.append(float(output.loss.detach().float().cpu()))
    return sum(values) / len(values)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.stage == "forecast" and not args.adapter_path:
        raise ValueError("--stage forecast requires --adapter-path from codebook SFT.")
    if args.max_prompt_length < 1 or args.max_completion_length < 1 or args.max_sequence_length < 2 or args.num_train_epochs < 1:
        raise ValueError("Token limits and epochs must be positive.")
    if args.max_completion_length >= args.max_sequence_length:
        raise ValueError("--max-sequence-length must exceed --max-completion-length.")
    dataset_path = resolve_repo_path(args.dataset)
    codebook = load_codebook(dataset_path)
    input_paths = [resolve_repo_path(value) for value in args.input]
    if args.stage == "codebook":
        examples, system, skipped = codebook_examples(codebook), CODEBOOK_SYSTEM, 0
    else:
        rows = [row for path in input_paths for row in load_jsonl(path)]
        examples, skipped = forecast_examples(rows, codebook)
        system = FORECAST_SYSTEM
    if args.max_samples > 0:
        examples = examples[: args.max_samples]
    validation_ratio = 0.0 if args.stage == "codebook" else args.validation_ratio
    train, validation = split_examples(examples, validation_ratio, args.seed)
    if not train:
        raise RuntimeError("No usable SFT examples.")
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    if log_path.exists():
        log_path.unlink()
    with log_path.open("a", encoding="utf-8") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(old_stdout, log_file), Tee(old_stderr, log_file)
        try:
            print(f"forecast SFT | stage={args.stage} | examples={len(examples)} | train={len(train)} | validation={len(validation)} | codebook={len(codebook)}", flush=True)
            model, tokenizer, torch = load_qwen_with_lora(
                resolve_repo_path(args.model_path),
                resolve_repo_path(args.adapter_path) if args.adapter_path else None,
                args.target_modules,
                args.lora_r,
                args.lora_alpha,
                args.lora_dropout,
            )
            model.config.use_cache = False
            device = next(model.parameters()).device
            train_items = [encode(tokenizer, item, system, args.max_prompt_length, args.max_completion_length, args.max_sequence_length) for item in train]
            validation_items = [encode(tokenizer, item, system, args.max_prompt_length, args.max_completion_length, args.max_sequence_length) for item in validation]
            all_items = train_items + validation_items
            truncated = sum(int(item["truncated"]) for item in all_items)
            if truncated:
                raise RuntimeError(f"{truncated} targets exceed --max-completion-length={args.max_completion_length}.")
            parameters = [value for value in model.parameters() if value.requires_grad]
            if not parameters:
                raise RuntimeError("No trainable LoRA parameters.")
            optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
            best_loss, global_step, started, history = float("inf"), 0, time.time(), []
            best_adapter = output_dir / "best_adapter"
            for epoch in range(1, args.num_train_epochs + 1):
                model.train()
                order = list(range(len(train_items)))
                random.Random(args.seed + epoch).shuffle(order)
                losses, accumulation = [], 0
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, len(order), args.per_device_train_batch_size):
                    selected = [train_items[index] for index in order[start : start + args.per_device_train_batch_size]]
                    output = model(**{key: value.to(device) for key, value in collate(selected, tokenizer, torch).items()})
                    if not torch.isfinite(output.loss):
                        raise RuntimeError(f"Non-finite loss at epoch={epoch}, offset={start}.")
                    losses.append(float(output.loss.detach().float().cpu()))
                    (output.loss / args.gradient_accumulation_steps).backward()
                    accumulation += 1
                    if accumulation == args.gradient_accumulation_steps or start + args.per_device_train_batch_size >= len(order):
                        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        accumulation = 0
                        global_step += 1
                        if args.logging_steps and global_step % args.logging_steps == 0:
                            print(f"step={global_step} | epoch={epoch} | loss={losses[-1]:.6f}", flush=True)
                train_loss = sum(losses) / len(losses)
                validation_loss = loss_on(model, validation_items, tokenizer, torch, device, args.per_device_train_batch_size)
                selected_loss = validation_loss if validation_loss is not None else train_loss
                record = {"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "validation_loss": validation_loss, "elapsed_seconds": round(time.time() - started, 3)}
                history.append(record)
                print(json.dumps(record), flush=True)
                if selected_loss <= best_loss:
                    best_loss = selected_loss
                    model.save_pretrained(best_adapter)
                    tokenizer.save_pretrained(best_adapter)
            final_adapter = output_dir / "final_adapter"
            model.save_pretrained(final_adapter)
            tokenizer.save_pretrained(final_adapter)
            config = {
                **vars(args),
                "dataset_path": str(dataset_path),
                "input_paths": [str(path) for path in input_paths],
                "examples": len(examples),
                "train_examples": len(train),
                "validation_examples": len(validation),
                "effective_validation_ratio": validation_ratio,
                "codebook_size": len(codebook),
                "skipped_unknown_code_rows": skipped,
                "max_target_completion_tokens": max(item["completion_tokens"] for item in all_items),
                "best_adapter": str(best_adapter),
                "final_adapter": str(final_adapter),
            }
            save_json(output_dir / "train_config.json", config)
            save_json(output_dir / "train_history.json", history)
            save_json(output_dir / "metrics.json", {"stage": args.stage, "examples": len(examples), "best_selection_loss": best_loss, "truncated_targets": truncated, "best_adapter": str(best_adapter), "final_adapter": str(final_adapter)})
            print(f"SFT complete | best_adapter={best_adapter} | final_adapter={final_adapter}", flush=True)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


if __name__ == "__main__":
    try:
        main()
    except (LoraUnavailable, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
