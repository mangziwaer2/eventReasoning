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

STAGE_MANIFEST_NAME = "grpo_stage_manifest.json"


try:
    from transformers.trainer_callback import TrainerCallback
except ImportError:  # Keep argument parsing/imports usable without Transformers installed.
    class TrainerCallback:  # type: ignore[no-redef]
        pass


class CollapseGuardCallback(TrainerCallback):
    """Stop GRPO before a collapsed policy is written as the final adapter.

    TRL exposes both metrics through ``on_log``. A single zero-variance group
    is normal, so the guard requires a configurable number of consecutive log
    entries before stopping.
    """

    def __init__(
        self,
        *,
        patience: int = 8,
        min_entropy: float = 0.03,
        max_zero_std_ratio: float = 0.95,
    ) -> None:
        self.patience = max(0, int(patience))
        self.min_entropy = float(min_entropy)
        self.max_zero_std_ratio = float(max_zero_std_ratio)
        self._entropy_bad_steps = 0
        self._zero_std_bad_steps = 0
        self.triggered = False

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if self.patience <= 0 or not logs:
            return control
        entropy = logs.get("entropy")
        if isinstance(entropy, (int, float)):
            self._entropy_bad_steps = self._entropy_bad_steps + 1 if entropy < self.min_entropy else 0
        zero_std = logs.get("frac_reward_zero_std")
        if isinstance(zero_std, (int, float)):
            self._zero_std_bad_steps = self._zero_std_bad_steps + 1 if zero_std >= self.max_zero_std_ratio else 0
        if self._entropy_bad_steps >= self.patience or self._zero_std_bad_steps >= self.patience:
            self.triggered = True
            control.should_training_stop = True
            print(
                "GRPO collapse guard triggered | "
                f"entropy_bad_steps={self._entropy_bad_steps} | "
                f"zero_std_bad_steps={self._zero_std_bad_steps}",
                flush=True,
            )
        return control


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
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Hard step cap (default: 0, disabled). A positive value takes precedence over epochs.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="Reference-policy KL coefficient. Keep positive to prevent LoRA drift.",
    )
    parser.add_argument(
        "--grpo-stage",
        choices=("single", "bootstrap", "kl"),
        default="single",
        help=(
            "Training stage. bootstrap requires beta=0 and writes a health manifest; "
            "kl requires beta>0 and a passed bootstrap adapter."
        ),
    )
    parser.add_argument(
        "--allow-unverified-bootstrap-adapter",
        action="store_true",
        help="Allow the KL stage to start without a passed bootstrap manifest (unsafe escape hatch).",
    )
    parser.add_argument(
        "--stage-min-reward-groups",
        type=int,
        default=20,
        help="Minimum audited reward groups required for a staged run to pass.",
    )
    parser.add_argument(
        "--stage-min-valid-answer-rate",
        type=float,
        default=0.5,
        help="Minimum mean rate of non-empty three-digit answers required for staged-run acceptance.",
    )
    parser.add_argument(
        "--stage-min-format-score",
        type=float,
        default=0.75,
        help="Minimum mean structured-output format score required for staged-run acceptance.",
    )
    parser.add_argument(
        "--stage-min-answer-hit-group-rate",
        type=float,
        default=0.05,
        help="Minimum fraction of reward groups containing at least one gold answer hit.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--wrong-answer-trace-scale",
        type=float,
        default=0.05,
        help="Trace reward multiplier when no gold event code is hit; keep small to preserve exploration without format hacking.",
    )
    parser.add_argument(
        "--collapse-patience",
        type=int,
        default=8,
        help="Consecutive log entries required before the collapse guard stops training; 0 disables it.",
    )
    parser.add_argument("--collapse-min-entropy", type=float, default=0.03)
    parser.add_argument("--collapse-max-zero-std-ratio", type=float, default=0.95)
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
        "num_train_epochs": getattr(args, "num_train_epochs", 1.0),
        "max_steps": getattr(args, "max_steps", 0) if getattr(args, "max_steps", 0) > 0 else -1,
        "warmup_ratio": getattr(args, "warmup_ratio", 0.1),
        "max_grad_norm": getattr(args, "max_grad_norm", 0.5),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "seed": args.seed,
        "beta": getattr(args, "beta", 0.04),
        "loss_type": "grpo",
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "save_total_limit": 3,
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


def bootstrap_manifest_path(adapter_path: str | Path) -> Path:
    return resolve_repo_path(str(adapter_path)).parent / STAGE_MANIFEST_NAME


def validate_two_stage_args(args: argparse.Namespace) -> None:
    stage = str(getattr(args, "grpo_stage", "single"))
    beta = float(getattr(args, "beta", 0.0))
    policy = str(getattr(args, "policy", "forecast_trace_reward"))
    wrong_answer_trace_scale = float(getattr(args, "wrong_answer_trace_scale", 0.05))
    min_groups = int(getattr(args, "stage_min_reward_groups", 20))
    if min_groups <= 0:
        raise ValueError("--stage-min-reward-groups must be positive.")
    for name in (
        "stage_min_valid_answer_rate",
        "stage_min_format_score",
        "stage_min_answer_hit_group_rate",
    ):
        value = float(getattr(args, name, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")

    if stage == "single":
        return
    if policy != "forecast_trace_reward":
        raise ValueError("Two-stage GRPO must preserve main's forecast_trace_reward policy.")
    if wrong_answer_trace_scale != 0.05:
        raise ValueError(
            "Two-stage GRPO must preserve main's --wrong-answer-trace-scale 0.05; use --grpo-stage single for ablations."
        )
    if stage == "bootstrap":
        if beta != 0.0:
            raise ValueError("--grpo-stage bootstrap requires --beta 0.")
        return
    if stage != "kl":
        raise ValueError(f"Unsupported GRPO stage: {stage}")
    if beta <= 0.0:
        raise ValueError("--grpo-stage kl requires a positive --beta.")
    if bool(getattr(args, "allow_unverified_bootstrap_adapter", False)):
        return

    manifest_path = bootstrap_manifest_path(args.adapter_path)
    if not manifest_path.is_file():
        raise ValueError(
            f"KL stage requires a passed bootstrap manifest at {manifest_path}. "
            "Run the bootstrap stage first or pass --allow-unverified-bootstrap-adapter deliberately."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read bootstrap manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("stage") != "bootstrap" or manifest.get("status") != "passed":
        raise ValueError(f"Bootstrap manifest has not passed its health gate: {manifest_path}")
    if (
        manifest.get("reward_policy") != "forecast_trace_reward"
        or float(manifest.get("wrong_answer_trace_scale", -1.0)) != 0.05
    ):
        raise ValueError(
            f"Bootstrap manifest does not preserve the verified main reward configuration: {manifest_path}"
        )


def reference_policy_mode(model: Any, beta: float) -> str:
    if beta <= 0.0:
        return "disabled"
    peft_config = getattr(model, "peft_config", None)
    if peft_config is None:
        return "separate_model"
    try:
        adapter_names = set(peft_config.keys())
    except (AttributeError, TypeError):
        adapter_names = set()
    return "ref_adapter" if "ref" in adapter_names else "base_disabled"


def summarize_stage_health(
    reward_history_path: Path,
    args: argparse.Namespace,
    *,
    collapse_triggered: bool = False,
) -> dict[str, Any]:
    rows = load_jsonl(reward_history_path) if reward_history_path.is_file() else []
    sample_count = sum(max(1, int(row.get("batch_size", 1))) for row in rows)

    def weighted_mean(key: str) -> float:
        if sample_count <= 0:
            return 0.0
        total = 0.0
        for row in rows:
            weight = max(1, int(row.get("batch_size", 1)))
            breakdown = row.get("breakdown_mean", {})
            value = breakdown.get(key, 0.0) if isinstance(breakdown, dict) else 0.0
            total += weight * float(value)
        return total / sample_count

    answer_hit_groups = 0
    for row in rows:
        breakdown = row.get("breakdown_mean", {})
        if isinstance(breakdown, dict) and float(breakdown.get("answer", 0.0)) > 0.0:
            answer_hit_groups += 1
    group_count = len(rows)
    metrics = {
        "reward_groups": group_count,
        "completion_samples": sample_count,
        "valid_answer_rate": round(weighted_mean("valid_answer_format"), 6),
        "mean_format_score": round(weighted_mean("format"), 6),
        "answer_hit_group_rate": round(answer_hit_groups / group_count, 6) if group_count else 0.0,
        "collapse_triggered": bool(collapse_triggered),
    }
    thresholds = {
        "min_reward_groups": int(args.stage_min_reward_groups),
        "min_valid_answer_rate": float(args.stage_min_valid_answer_rate),
        "min_format_score": float(args.stage_min_format_score),
        "min_answer_hit_group_rate": float(args.stage_min_answer_hit_group_rate),
    }
    failures: list[str] = []
    if metrics["reward_groups"] < thresholds["min_reward_groups"]:
        failures.append("insufficient_reward_groups")
    if metrics["valid_answer_rate"] < thresholds["min_valid_answer_rate"]:
        failures.append("valid_answer_rate")
    if metrics["mean_format_score"] < thresholds["min_format_score"]:
        failures.append("format_score")
    if metrics["answer_hit_group_rate"] < thresholds["min_answer_hit_group_rate"]:
        failures.append("answer_hit_group_rate")
    if collapse_triggered:
        failures.append("collapse_guard")
    return {
        "passed": not failures,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    if args.min_coarse_edges < 0:
        raise ValueError("--min-coarse-edges must be non-negative.")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative.")
    if args.num_train_epochs <= 0:
        raise ValueError("--num-train-epochs must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.beta < 0:
        raise ValueError("--beta must be non-negative.")
    if not 0.0 <= args.wrong_answer_trace_scale <= 1.0:
        raise ValueError("--wrong-answer-trace-scale must be between 0 and 1.")
    if not args.adapter_path:
        raise ValueError("--adapter-path is required: start GRPO from the completed codebook+forecast SFT adapter.")
    validate_two_stage_args(args)
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
    for audit_name in (
        "reward_history.jsonl",
        "rollout_samples.jsonl",
        "sample_generations.txt",
        STAGE_MANIFEST_NAME,
    ):
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
                        f"grpo_stage={args.grpo_stage}",
                        f"beta={args.beta}",
                        f"learning_rate={args.learning_rate}",
                        f"wrong_answer_trace_scale={args.wrong_answer_trace_scale}",
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
                wrong_answer_trace_scale=args.wrong_answer_trace_scale,
                audit_path=output_dir / "reward_history.jsonl",
                audit_every=args.reward_log_every,
                sample_audit_path=output_dir / "rollout_samples.jsonl",
                sample_audit_every=args.sample_log_every,
                sample_audit_limit=args.sample_log_count,
                sample_human_path=output_dir / "sample_generations.txt",
            )
            config = build_training_config(GRPOConfig, args)
            trainer = build_trainer(GRPOTrainer, model, tokenizer, config, dataset, reward_fn)
            reference_mode = reference_policy_mode(model, args.beta)
            log_line(f"GRPO reference policy | stage={args.grpo_stage} | beta={args.beta} | mode={reference_mode}")
            if args.grpo_stage == "kl" and reference_mode != "ref_adapter":
                raise RuntimeError(
                    "KL stage requires a frozen copy of the bootstrap adapter as the reference policy; "
                    f"detected mode={reference_mode}. Use TRL 1.10 with PEFT >= 0.20."
                )
            collapse_guard = None
            if args.collapse_patience > 0:
                collapse_guard = CollapseGuardCallback(
                    patience=args.collapse_patience,
                    min_entropy=args.collapse_min_entropy,
                    max_zero_std_ratio=args.collapse_max_zero_std_ratio,
                )
                trainer.add_callback(collapse_guard)
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
            if args.grpo_stage != "single":
                health = summarize_stage_health(
                    output_dir / "reward_history.jsonl",
                    args,
                    collapse_triggered=bool(collapse_guard and collapse_guard.triggered),
                )
                status = "passed" if health["passed"] else "failed"
                manifest_path = output_dir / STAGE_MANIFEST_NAME
                save_json(
                    manifest_path,
                    {
                        "schema_version": "two-stage-grpo-v1",
                        "stage": args.grpo_stage,
                        "status": status,
                        "source_adapter": str(resolve_repo_path(args.adapter_path)),
                        "final_adapter": str(final_adapter),
                        "beta": float(args.beta),
                        "reference_policy_mode": reference_mode,
                        "reward_policy": args.policy,
                        "wrong_answer_trace_scale": float(args.wrong_answer_trace_scale),
                        "health": health,
                    },
                )
                log_line(f"GRPO stage manifest | stage={args.grpo_stage} | status={status} | path={manifest_path}")
                if status != "passed":
                    failures = ", ".join(health.get("failures", [])) if isinstance(health, dict) else "unknown"
                    raise RuntimeError(
                        "GRPO stage failed its health gate; do not use this adapter for subsequent training or evaluation. "
                        f"stage={args.grpo_stage} | failures={failures} | manifest={manifest_path}"
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
