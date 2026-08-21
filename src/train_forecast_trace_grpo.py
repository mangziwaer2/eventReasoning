from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

from forecast_trace_grpo_rewards import ForecastTraceGRPOReward
from forecast_trace_grpo_rewards import rollout_rows_to_grpo_samples
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_qwen_with_lora
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train forecast-trace LoRA with TRL GRPO and deterministic structured rewards."
    )
    parser.add_argument("--input", nargs="+", required=True, help="predictions.jsonl files from evaluate_local_qwen_pipeline.py")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "models" / "Qwen2.5-0.5B"))
    parser.add_argument("--adapter-path", default=None, help="Existing trainable LoRA adapter to continue from.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "forecast_trace_grpo"))
    parser.add_argument("--policy", default="forecast_trace_reward")
    parser.add_argument("--reward-key", default="total", help="Reward breakdown key passed to GRPO.")
    parser.add_argument("--error-reward", type=float, default=-0.25, help="Finite fallback reward for malformed reward inputs.")
    parser.add_argument("--max-samples", type=int, default=0, help="Use 0 for all rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generations", type=int, default=4, help="Completions sampled per prompt for GRPO groups.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-chat-prompt", dest="chat_prompt", action="store_false", default=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_grpo_samples(
    input_paths: list[str],
    max_samples: int = 0,
    *,
    chat_prompt: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_arg in input_paths:
        rows.extend(load_jsonl(resolve_repo_path(input_arg)))
    samples = rollout_rows_to_grpo_samples(rows, chat_prompt=chat_prompt)
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def build_training_config(config_cls: Any, args: argparse.Namespace) -> Any:
    """Build GRPOConfig across TRL releases with different optional fields."""

    values = {
        "output_dir": str(resolve_repo_path(args.output_dir)),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "seed": args.seed,
    }
    accepted = inspect.signature(config_cls).parameters
    return config_cls(**{key: value for key, value in values.items() if key in accepted})


def build_trainer(trainer_cls: Any, model: Any, tokenizer: Any, config: Any, dataset: Any, reward_fn: Any) -> Any:
    values = {
        "model": model,
        "reward_funcs": reward_fn,
        "args": config,
        "train_dataset": dataset,
    }
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        values["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        values["tokenizer"] = tokenizer
    return trainer_cls(**{key: value for key, value in values.items() if key in parameters})


def main() -> None:
    args = parse_args()
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2 for GRPO group-relative advantages.")
    if args.per_device_train_batch_size < args.num_generations:
        raise ValueError(
            "--per-device-train-batch-size must be at least --num-generations "
            "for a single-process GRPO run."
        )
    if args.per_device_train_batch_size % args.num_generations != 0:
        raise ValueError(
            "--per-device-train-batch-size must be divisible by --num-generations "
            "for a single-process GRPO run."
        )
    random.seed(args.seed)
    samples = load_grpo_samples(args.input, max_samples=args.max_samples, chat_prompt=args.chat_prompt)
    if not samples:
        raise RuntimeError("No usable rows found. Ensure predictions JSONL contains forecast_prompt.")

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise LoraUnavailable(
            "GRPO training requires `trl` and `datasets`; install requirements.txt before running this command."
        ) from exc

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(samples)
    model, tokenizer, _ = load_qwen_with_lora(
        model_path=resolve_repo_path(args.model_path),
        adapter_path=resolve_repo_path(args.adapter_path) if args.adapter_path else None,
        target_modules=args.target_modules,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    reward_fn = ForecastTraceGRPOReward(
        policy_name=args.policy,
        reward_key=args.reward_key,
        error_reward=args.error_reward,
    )
    config = build_training_config(GRPOConfig, args)
    trainer = build_trainer(GRPOTrainer, model, tokenizer, config, dataset, reward_fn)
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(output_dir / "final_adapter")
    (output_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "samples": len(samples)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"GRPO training complete | samples={len(samples)} | output={output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (LoraUnavailable, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
