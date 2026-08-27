from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from forecast_trace_grpo_rewards import ForecastTraceGRPOReward
from forecast_trace_grpo_rewards import rollout_rows_to_grpo_samples
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_qwen_with_lora
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


class TeeStream:
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


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def log_line(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train forecast-trace LoRA with TRL GRPO and deterministic structured rewards."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(REPO_ROOT / "outputs" / "grpo_context_mirai_forecast_train_no_refine" / "grpo_context.jsonl")],
        help="Prompt-only GRPO context JSONL files from the prepare-grpo-context stage.",
    )
    parser.add_argument("--allow-legacy-rollout-input", action="store_true", help="Allow old predictions.jsonl rows; disabled by default for online-only training.")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "models" / "Qwen3-4B"))
    parser.add_argument("--adapter-path", default=None, help="Required SFT LoRA adapter from the single-stage codebook+forecast SFT.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "forecast_trace_grpo"))
    parser.add_argument("--policy", default="forecast_trace_reward")
    parser.add_argument("--reward-key", default="total", help="Reward breakdown key passed to GRPO.")
    parser.add_argument("--reward-log-every", type=int, default=1, help="Write one aggregate reward audit row every N reward calls.")
    parser.add_argument("--sample-log-every", type=int, default=1, help="Write sampled prompt/completion rows every N reward calls.")
    parser.add_argument("--sample-log-count", type=int, default=2, help="Maximum generated samples written per sampled reward call.")
    parser.add_argument("--error-reward", type=float, default=-0.25, help="Finite fallback reward for malformed reward inputs.")
    parser.add_argument("--max-samples", type=int, default=0, help="Use 0 for all rows.")
    parser.add_argument("--min-coarse-edges", type=int, default=0, help="Optional minimum refined/coarse graph edge count. 0 keeps every prompt context.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generations", type=int, default=4, help="Completions sampled per prompt for GRPO groups.")
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--lora-r", type=int, default=16)
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


def normalize_input_paths(input_paths: Any) -> list[Path]:
    if isinstance(input_paths, (str, Path)):
        return [resolve_repo_path(str(input_paths))]
    return [resolve_repo_path(str(item)) for item in input_paths]


def load_grpo_samples(
    input_paths: Any,
    max_samples: int = 0,
    *,
    chat_prompt: bool = True,
    allow_legacy_rollout_input: bool = False,
    min_coarse_edges: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_path in normalize_input_paths(input_paths):
        rows.extend(load_jsonl(input_path))
    legacy_rows = [row for row in rows if any(key in row for key in ("forecast_prediction", "raw_forecast", "reward", "reward_breakdown"))]
    if legacy_rows and not allow_legacy_rollout_input:
        raise ValueError(
            "Online GRPO requires prompt-only context rows. Found precomputed forecast/reward fields; "
            "run --stage prepare-grpo-context or pass --allow-legacy-rollout-input explicitly."
        )
    rows, _ = filter_rollout_rows_by_edge_count(rows, min_coarse_edges)
    samples = rollout_rows_to_grpo_samples(rows, chat_prompt=chat_prompt)
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def rollout_row_edge_count(row: dict[str, Any]) -> int:
    trajectory = row.get("trajectory", {})
    if not isinstance(trajectory, dict):
        return 0
    metadata = trajectory.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    graph = metadata.get("refined_graph")
    if isinstance(graph, dict):
        edges = graph.get("edges")
        if isinstance(edges, list):
            return len(edges)
    for section in ("refinement", "coarse"):
        summary = row.get(section, {})
        if isinstance(summary, dict) and "edge_count" in summary:
            try:
                return max(0, int(summary["edge_count"]))
            except (TypeError, ValueError):
                continue
    return 0


def filter_rollout_rows_by_edge_count(rows: list[dict[str, Any]], min_coarse_edges: int) -> tuple[list[dict[str, Any]], int]:
    if min_coarse_edges <= 0:
        return rows, 0
    kept = [row for row in rows if rollout_row_edge_count(row) >= min_coarse_edges]
    return kept, len(rows) - len(kept)
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
    if args.min_coarse_edges < 0:
        raise ValueError("--min-coarse-edges must be non-negative.")
    if not args.adapter_path:
        raise ValueError("--adapter-path is required: start GRPO from the completed codebook+forecast SFT adapter.")
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
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    if log_path.exists():
        log_path.unlink()
    for audit_name in ("reward_history.jsonl", "rollout_samples.jsonl", "sample_generations.txt"):
        audit_path = output_dir / audit_name
        if audit_path.exists():
            audit_path.unlink()

    input_paths = normalize_input_paths(args.input)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            log_line(
                " | ".join(
                    [
                        "forecast trace GRPO loading model",
                        f"inputs={len(input_paths)}",
                        f"chat_prompt={args.chat_prompt}",
                        f"num_generations={args.num_generations}",
                        f"batch_size={args.per_device_train_batch_size}",
                        f"min_coarse_edges={args.min_coarse_edges}",
                        f"sample_log_every={args.sample_log_every}",
                        f"sample_log_count={args.sample_log_count}",
                    ]
                )
            )
            source_rows = [row for input_path in input_paths for row in load_jsonl(input_path)]
            source_row_count = len(source_rows)
            _, low_edge_filtered_rows = filter_rollout_rows_by_edge_count(
                source_rows,
                args.min_coarse_edges,
            )
            samples = load_grpo_samples(
                input_paths,
                max_samples=args.max_samples,
                chat_prompt=args.chat_prompt,
                allow_legacy_rollout_input=args.allow_legacy_rollout_input,
                min_coarse_edges=args.min_coarse_edges,
            )
            if not samples:
                raise RuntimeError("No usable rows found. Ensure GRPO context JSONL contains forecast_prompt and trajectory.")

            save_json(
                output_dir / "sample_summary.json",
                {
                    "input_paths": [str(path) for path in input_paths],
                    "source_rows": source_row_count,
                    "low_edge_filtered_rows": low_edge_filtered_rows,
                    "samples": len(samples),
                    "chat_prompt": bool(args.chat_prompt),
                    "max_samples": int(args.max_samples),
                    "min_coarse_edges": int(args.min_coarse_edges),
                    "policy": args.policy,
                    "reward_key": args.reward_key,
                },
            )
            log_line(
                f"loaded prompt rows | source_rows={source_row_count} | samples={len(samples)} | "
                f"low_edge_filtered={low_edge_filtered_rows} | inputs={len(input_paths)} | "
                f"min_coarse_edges={args.min_coarse_edges}"
            )

            try:
                from datasets import Dataset
                from trl import GRPOConfig, GRPOTrainer
            except ImportError as exc:
                raise LoraUnavailable(
                    "GRPO training requires `trl` and `datasets`; install requirements.txt before running this command."
                ) from exc

            dataset = Dataset.from_list(samples)
            log_line(f"dataset ready | rows={len(samples)} | output={output_dir}")
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
                audit_path=output_dir / "reward_history.jsonl",
                audit_every=args.reward_log_every,
                sample_audit_path=output_dir / "rollout_samples.jsonl",
                sample_audit_every=args.sample_log_every,
                sample_audit_limit=args.sample_log_count,
                sample_human_path=output_dir / "sample_generations.txt",
            )
            config = build_training_config(GRPOConfig, args)
            trainer = build_trainer(GRPOTrainer, model, tokenizer, config, dataset, reward_fn)
            train_result = trainer.train()
            if hasattr(trainer, "save_state"):
                try:
                    trainer.save_state()
                except Exception:
                    pass

            final_adapter = output_dir / "final_adapter"
            trainer.save_model(str(final_adapter))
            tokenizer.save_pretrained(final_adapter)

            history = [item for item in getattr(trainer.state, "log_history", []) if isinstance(item, dict)]
            train_metrics = dict(getattr(train_result, "metrics", {}) or {})
            run_config = {
                **vars(args),
                "samples": len(samples),
                "input_paths": [str(path) for path in input_paths],
                "model_path": str(resolve_repo_path(args.model_path)),
                "adapter_path": str(resolve_repo_path(args.adapter_path)) if args.adapter_path else None,
                "output_dir": str(output_dir),
                "final_adapter": str(final_adapter),
            }
            save_json(output_dir / "run_config.json", run_config)
            save_json(output_dir / "train_history.json", history)
            save_json(
                output_dir / "metrics.json",
                {
                    "samples": len(samples),
                    "input_paths": [str(path) for path in input_paths],
                    "output_dir": str(output_dir),
                    "final_adapter": str(final_adapter),
                    "reward_history": str(output_dir / "reward_history.jsonl"),
                    "rollout_samples": str(output_dir / "rollout_samples.jsonl"),
                    "history_entries": len(history),
                    "train_result_metrics": train_metrics,
                    "elapsed_seconds": round(time.time() - started, 3),
                },
            )
            log_line(
                " | ".join(
                    [
                        "GRPO training complete",
                        f"samples={len(samples)}",
                        f"output={output_dir}",
                        f"log={log_path}",
                    ]
                )
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    try:
        main()
    except (LoraUnavailable, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
