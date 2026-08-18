from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_qwen_with_lora
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path
from rl_pipeline_hooks import PipelineStep
from rl_pipeline_hooks import PipelineTrajectory
from rl_pipeline_hooks import build_pipeline_policy


DEFAULT_FORECAST_SYSTEM_PROMPT = "You output structured forecast_trace and closed-set final_answer JSON only."


@dataclass(slots=True)
class ForecastTraceRLSample:
    sample_id: str
    query_id: str
    prompt: str
    system_prompt: str
    completion: str
    reward: float
    reward_breakdown: dict[str, float]
    weight: float = 1.0
    metadata: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline reward-weighted LoRA training for LoRA B forecast_trace generation. "
            "Input is predictions.jsonl produced by evaluate_local_qwen_pipeline.py."
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="One or more predictions.jsonl / rescored JSONL files.")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "models" / "Qwen2.5-0.5B"), help="Base Qwen model path for LoRA B.")
    parser.add_argument("--adapter-path", default=None, help="Optional existing LoRA B adapter to continue from.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "forecast_trace_rl_lora"), help="Output directory.")
    parser.add_argument("--policy", default="forecast_trace_reward", help="Reward policy used when recomputing rewards.")
    parser.add_argument("--reward-source", choices=["recompute", "stored"], default="recompute", help="Use deterministic reward recomputation or stored row reward.")
    parser.add_argument("--completion-source", choices=["raw", "parsed"], default="raw", help="Train on raw model completion or normalized parsed JSON.")
    parser.add_argument("--min-reward", type=float, default=0.0, help="Drop rollouts with reward below this value.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap after filtering. Use 0 for all.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Held-out ratio for RL rollout validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--epochs", type=int, default=1, help="Number of reward-weighted training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Validation batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for LoRA B RL continuation.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm. Use 0 to disable.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum prompt + completion token length.")
    parser.add_argument("--max-completion-tokens", type=int, default=512, help="Maximum completion tokens kept for training.")
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"], help="LoRA target modules for a fresh adapter.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank for a fresh adapter.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha for a fresh adapter.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout for a fresh adapter.")

    parser.add_argument("--reward-baseline", choices=["mean", "median", "zero"], default="mean", help="Baseline b used in reward weighting.")
    parser.add_argument("--weighting", choices=["exp", "linear", "binary", "uniform"], default="exp", help="Reward-to-loss weight transform.")
    parser.add_argument("--reward-temperature", type=float, default=1.0, help="Temperature tau for exp reward weighting.")
    parser.add_argument("--min-weight", type=float, default=0.05, help="Minimum sample weight after transform.")
    parser.add_argument("--max-weight", type=float, default=3.0, help="Maximum sample weight after transform.")

    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N optimizer steps. Use 0 to disable.")
    parser.add_argument("--debug-samples", type=int, default=3, help="Number of rollout samples summarized in debug JSONL.")
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def log_line(message: str, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL.") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def build_trajectory(row: dict[str, Any]) -> PipelineTrajectory:
    raw_trajectory = row.get("trajectory", {})
    if not isinstance(raw_trajectory, dict):
        raw_trajectory = {}
    trajectory = PipelineTrajectory(
        sample_id=str(raw_trajectory.get("sample_id", row.get("query_id", ""))),
        final_reward=raw_trajectory.get("final_reward"),
        metadata=dict(raw_trajectory.get("metadata", {})) if isinstance(raw_trajectory.get("metadata"), dict) else {},
    )
    for step in raw_trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        trajectory.steps.append(
            PipelineStep(
                name=str(step.get("name", "")),
                observation=dict(step.get("observation", {})) if isinstance(step.get("observation"), dict) else {},
                action=dict(step.get("action", {})) if isinstance(step.get("action"), dict) else {},
                reward=step.get("reward"),
                metadata=dict(step.get("metadata", {})) if isinstance(step.get("metadata"), dict) else {},
            )
        )
    if "choices" in row:
        trajectory.metadata.setdefault("choices", row["choices"])
    return trajectory


def latest_step(trajectory: PipelineTrajectory, name: str) -> PipelineStep | None:
    for step in reversed(trajectory.steps):
        if step.name == name:
            return step
    return None


def clean_trace_event(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_event_id": item.get("trace_event_id", ""),
        "event": {
            "trigger": item.get("trigger", ""),
            "mention": item.get("event", ""),
            "actors": item.get("actors", []),
            "relative_time": item.get("relative_time", ""),
        },
        "supporting_event_ids": item.get("supporting_event_ids", []),
        "supporting_edge_ids": item.get("supporting_edge_ids", []),
        "expected_effect": item.get("expected_effect", ""),
        "confidence": item.get("confidence", 0.0),
    }


def clean_trace_edge(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_edge_id": item.get("trace_edge_id", ""),
        "source_id": item.get("source_id", ""),
        "target_id": item.get("target_id", ""),
        "relation_type": item.get("relation_type", ""),
        "confidence": item.get("confidence", 0.0),
    }


def parsed_completion(prediction: dict[str, Any]) -> str:
    trace = prediction.get("forecast_trace", {})
    if not isinstance(trace, dict):
        trace = {}
    final_answer = prediction.get("final_answer", {})
    if not isinstance(final_answer, dict):
        final_answer = {}
    payload = {
        "forecast_trace": {
            "intermediate_events": [
                clean_trace_event(item)
                for item in trace.get("intermediate_events", [])
                if isinstance(item, dict)
            ],
            "trace_edges": [
                clean_trace_edge(item)
                for item in trace.get("trace_edges", [])
                if isinstance(item, dict)
            ],
        },
        "final_answer": {
            "choice_id": final_answer.get("choice_id", ""),
            "event_code": final_answer.get("event_code", prediction.get("predicted_event_base_code", "")),
            "event": final_answer.get("event", prediction.get("forecast_event", "")),
            "confidence": final_answer.get("confidence", prediction.get("confidence", 0.0)),
            "supporting_event_ids": final_answer.get("supporting_event_ids", []),
            "supporting_edge_ids": final_answer.get("supporting_edge_ids", []),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def extract_prompt_and_completion(row: dict[str, Any], completion_source: str) -> tuple[str, str, str]:
    prompt = str(row.get("forecast_prompt", "")).strip()
    system_prompt = str(row.get("forecast_system_prompt", DEFAULT_FORECAST_SYSTEM_PROMPT)).strip() or DEFAULT_FORECAST_SYSTEM_PROMPT
    trajectory = build_trajectory(row)
    forecast_step = latest_step(trajectory, "forecast")
    raw_completion = ""
    if forecast_step is not None:
        raw_completion = str(forecast_step.metadata.get("raw_response", "")).strip()
    raw_completion = raw_completion or str(row.get("raw_response", "")).strip()

    prediction = row.get("forecast_prediction", {})
    if not isinstance(prediction, dict):
        prediction = {}
    completion = raw_completion if completion_source == "raw" else parsed_completion(prediction)
    if not completion and prediction:
        completion = parsed_completion(prediction)
    return prompt, system_prompt, completion.strip()


def compute_reward_breakdown(row: dict[str, Any], policy_name: str, reward_source: str) -> dict[str, float]:
    if reward_source == "stored":
        stored = row.get("reward_breakdown", {})
        if isinstance(stored, dict) and "total" in stored:
            return {str(key): float(value) for key, value in stored.items() if isinstance(value, (int, float))}
        return {"total": float(row.get("reward", 0.0))}

    policy = build_pipeline_policy(policy_name)
    prediction = row.get("forecast_prediction", {})
    if not isinstance(prediction, dict):
        prediction = {}
    gold = row.get("mirai_query", {})
    if not isinstance(gold, dict):
        gold = {}
    trajectory = build_trajectory(row)
    return policy.compute_reward_breakdown(prediction, gold, trajectory)


def load_rollout_samples(args: argparse.Namespace) -> list[ForecastTraceRLSample]:
    samples: list[ForecastTraceRLSample] = []
    missing_prompt = 0
    missing_completion = 0
    for input_arg in args.input:
        input_path = resolve_repo_path(input_arg)
        for row in load_jsonl(input_path):
            prompt, system_prompt, completion = extract_prompt_and_completion(row, args.completion_source)
            if not prompt:
                missing_prompt += 1
                continue
            if not completion:
                missing_completion += 1
                continue
            breakdown = compute_reward_breakdown(row, args.policy, args.reward_source)
            reward = float(breakdown.get("total", row.get("reward", 0.0)))
            if reward < args.min_reward:
                continue
            prediction = row.get("forecast_prediction", {})
            if not isinstance(prediction, dict):
                prediction = {}
            samples.append(
                ForecastTraceRLSample(
                    sample_id=str(row.get("query_id", row.get("sample_id", ""))),
                    query_id=str(row.get("query_id", "")),
                    prompt=prompt,
                    system_prompt=system_prompt,
                    completion=completion,
                    reward=reward,
                    reward_breakdown=breakdown,
                    metadata={
                        "predicted_event_base_code": prediction.get("predicted_event_base_code", ""),
                        "parsed_json": prediction.get("parsed_json", False),
                        "input_path": str(input_path),
                    },
                )
            )
    if missing_prompt:
        print(
            f"warning: skipped {missing_prompt} rows without forecast_prompt. "
            "Regenerate rollouts without --no-save-forecast-prompts.",
            flush=True,
        )
    if missing_completion:
        print(f"warning: skipped {missing_completion} rows without a forecast completion.", flush=True)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    return samples


def reward_baseline(rewards: list[float], mode: str) -> float:
    if not rewards or mode == "zero":
        return 0.0
    if mode == "median":
        ordered = sorted(rewards)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0
    return sum(rewards) / len(rewards)


def compute_sample_weights(samples: list[ForecastTraceRLSample], args: argparse.Namespace) -> None:
    rewards = [sample.reward for sample in samples]
    baseline = reward_baseline(rewards, args.reward_baseline)
    temperature = max(float(args.reward_temperature), 1e-6)
    raw_weights: list[float] = []
    for sample in samples:
        advantage = sample.reward - baseline
        if args.weighting == "uniform":
            weight = 1.0
        elif args.weighting == "binary":
            weight = 1.0 if advantage >= 0 else 0.0
        elif args.weighting == "linear":
            weight = max(0.0, advantage)
        else:
            weight = math.exp(advantage / temperature)
        if weight > 0:
            weight = min(max(weight, args.min_weight), args.max_weight)
        raw_weights.append(weight)

    positive_mean = sum(raw_weights) / max(sum(1 for weight in raw_weights if weight > 0), 1)
    if positive_mean <= 0:
        positive_mean = 1.0
    for sample, weight in zip(samples, raw_weights):
        sample.weight = float(weight / positive_mean) if weight > 0 else 0.0


def split_samples(
    samples: list[ForecastTraceRLSample],
    validation_ratio: float,
    seed: int,
) -> tuple[list[ForecastTraceRLSample], list[ForecastTraceRLSample]]:
    if validation_ratio <= 0 or len(samples) < 2:
        return samples, []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio))
    if validation_size >= len(shuffled):
        validation_size = len(shuffled) - 1
    return shuffled[validation_size:], shuffled[:validation_size]


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def build_prompt_ids(tokenizer, prompt: str, system_prompt: str) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=None)
        return [int(item) for item in ids]
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return [int(item) for item in tokenizer(text, add_special_tokens=False).input_ids]


def build_completion_ids(tokenizer, completion: str, max_completion_tokens: int) -> list[int]:
    ids = [int(item) for item in tokenizer(completion, add_special_tokens=False).input_ids]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        ids.append(int(eos_id))
    if max_completion_tokens > 0 and len(ids) > max_completion_tokens:
        ids = ids[:max_completion_tokens]
        if eos_id is not None:
            ids[-1] = int(eos_id)
    return ids


class ForecastTraceTokenDataset:
    def __init__(self, samples: list[ForecastTraceRLSample], tokenizer, args: argparse.Namespace) -> None:
        self.items: list[dict[str, Any]] = []
        skipped_too_long_completion = 0
        for sample in samples:
            prompt_ids = build_prompt_ids(tokenizer, sample.prompt, sample.system_prompt)
            completion_ids = build_completion_ids(tokenizer, sample.completion, args.max_completion_tokens)
            if not completion_ids or len(completion_ids) >= args.max_length:
                skipped_too_long_completion += 1
                continue
            available_prompt_tokens = args.max_length - len(completion_ids)
            if len(prompt_ids) > available_prompt_tokens:
                prompt_ids = prompt_ids[-available_prompt_tokens:]
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            self.items.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "weight": sample.weight,
                    "sample_id": sample.sample_id,
                    "reward": sample.reward,
                }
            )
        if skipped_too_long_completion:
            print(f"warning: skipped {skipped_too_long_completion} samples whose completion exceeded max_length.", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def make_collate_fn(tokenizer, torch):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        attention_mask = []
        weights = []
        rewards = []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
            weights.append(float(item["weight"]))
            rewards.append(float(item["reward"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
        }

    return collate


def weighted_token_loss(model, batch: dict[str, Any], torch, device) -> tuple[Any, dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    weights = batch["weights"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    token_counts = mask.sum(dim=1).clamp_min(1.0)
    per_sample_loss = (token_loss * mask).sum(dim=1) / token_counts
    loss = (per_sample_loss * weights).mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "unweighted_loss": float(per_sample_loss.mean().detach().cpu()),
        "mean_weight": float(weights.mean().detach().cpu()),
        "mean_tokens": float(token_counts.mean().detach().cpu()),
    }
    return loss, metrics


def evaluate_loss(model, dataloader, torch, device) -> dict[str, float]:
    if dataloader is None:
        return {}
    model.eval()
    total_loss = 0.0
    total_unweighted = 0.0
    total_weight = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            loss, metrics = weighted_token_loss(model, batch, torch, device)
            batch_size = int(batch["input_ids"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_unweighted += metrics["unweighted_loss"] * batch_size
            total_weight += metrics["mean_weight"] * batch_size
            count += batch_size
    model.train()
    return {
        "val_loss": total_loss / max(count, 1),
        "val_unweighted_loss": total_unweighted / max(count, 1),
        "val_mean_weight": total_weight / max(count, 1),
    }


def summarize_samples(samples: list[ForecastTraceRLSample]) -> dict[str, Any]:
    rewards = [sample.reward for sample in samples]
    weights = [sample.weight for sample in samples]
    return {
        "samples": len(samples),
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "weight_min": min(weights) if weights else 0.0,
        "weight_mean": sum(weights) / len(weights) if weights else 0.0,
        "weight_max": max(weights) if weights else 0.0,
    }


def write_debug_samples(path: Path, samples: list[ForecastTraceRLSample], limit: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in sorted(samples, key=lambda item: item.reward, reverse=True)[: max(limit, 0)]:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "query_id": sample.query_id,
                        "reward": sample.reward,
                        "weight": sample.weight,
                        "reward_breakdown": sample.reward_breakdown,
                        "metadata": sample.metadata or {},
                        "completion_preview": sample.completion[:600],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    if log_path.exists():
        log_path.unlink()
    random.seed(args.seed)

    rollout_samples = load_rollout_samples(args)
    if not rollout_samples:
        raise RuntimeError(
            "No usable RL rollout samples were loaded. "
            "Run evaluate_local_qwen_pipeline.py first and keep forecast_prompt in predictions.jsonl."
        )
    compute_sample_weights(rollout_samples, args)
    rollout_samples = [sample for sample in rollout_samples if sample.weight > 0]
    if not rollout_samples:
        raise RuntimeError("All rollout samples received zero reward weight. Lower --min-reward or change --weighting.")

    train_samples, validation_samples = split_samples(rollout_samples, args.validation_ratio, args.seed)
    save_json(output_dir / "train_config.json", vars(args))
    save_json(
        output_dir / "rollout_summary.json",
        {
            "all": summarize_samples(rollout_samples),
            "train": summarize_samples(train_samples),
            "validation": summarize_samples(validation_samples),
        },
    )
    write_debug_samples(output_dir / "debug_rollout_samples.jsonl", rollout_samples, args.debug_samples)

    log_line(
        " | ".join(
            [
                "forecast trace RL loading model",
                f"samples={len(rollout_samples)}",
                f"train={len(train_samples)}",
                f"validation={len(validation_samples)}",
                f"completion_source={args.completion_source}",
                f"weighting={args.weighting}",
            ]
        ),
        log_path,
    )

    try:
        adapter_path = resolve_repo_path(args.adapter_path) if args.adapter_path else None
        model, tokenizer, torch = load_qwen_with_lora(
            model_path=resolve_repo_path(args.model_path),
            adapter_path=adapter_path,
            target_modules=args.target_modules,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    except LoraUnavailable as exc:
        raise RuntimeError(str(exc)) from exc

    set_seed(args.seed, torch)
    device = next(model.parameters()).device
    train_dataset = ForecastTraceTokenDataset(train_samples, tokenizer, args)
    validation_dataset = ForecastTraceTokenDataset(validation_samples, tokenizer, args) if validation_samples else None
    if len(train_dataset) == 0:
        raise RuntimeError("No tokenized training samples remain after max-length filtering.")

    collate = make_collate_fn(tokenizer, torch)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        collate_fn=collate,
    )
    validation_loader = (
        torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=max(1, args.eval_batch_size),
            shuffle=False,
            collate_fn=collate,
        )
        if validation_dataset is not None and len(validation_dataset) > 0
        else None
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    global_step = 0
    started = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_unweighted = 0.0
        epoch_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            loss, metrics = weighted_token_loss(model, batch, torch, device)
            scaled_loss = loss / max(1, args.gradient_accumulation_steps)
            scaled_loss.backward()
            epoch_loss += metrics["loss"]
            epoch_unweighted += metrics["unweighted_loss"]
            epoch_batches += 1

            if batch_index % max(1, args.gradient_accumulation_steps) == 0 or batch_index == len(train_loader):
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        args.grad_clip,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.log_every > 0 and global_step % args.log_every == 0:
                    log_line(
                        " | ".join(
                            [
                                f"epoch={epoch}",
                                f"step={global_step}",
                                f"loss={metrics['loss']:.4f}",
                                f"unweighted={metrics['unweighted_loss']:.4f}",
                                f"mean_weight={metrics['mean_weight']:.3f}",
                                f"time={format_seconds(time.time() - started)}",
                            ]
                        ),
                        log_path,
                    )

        train_metrics = {
            "epoch": float(epoch),
            "global_step": float(global_step),
            "loss": epoch_loss / max(epoch_batches, 1),
            "unweighted_loss": epoch_unweighted / max(epoch_batches, 1),
            "lr": float(args.lr),
        }
        train_metrics.update(evaluate_loss(model, validation_loader, torch, device))
        history.append(train_metrics)

        latest_adapter = output_dir / "latest_adapter"
        model.save_pretrained(latest_adapter)
        tokenizer.save_pretrained(latest_adapter)
        current_val = train_metrics.get("val_loss", train_metrics["loss"])
        if current_val <= best_val_loss:
            best_val_loss = current_val
            best_adapter = output_dir / "best_adapter"
            model.save_pretrained(best_adapter)
            tokenizer.save_pretrained(best_adapter)

        save_json(output_dir / "train_history.json", history)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": vars(args),
            },
            output_dir / "latest_training_state.pt",
        )
        log_line(
            " | ".join(
                [
                    f"epoch={epoch}/{args.epochs}",
                    f"loss={train_metrics['loss']:.4f}",
                    f"val_loss={train_metrics.get('val_loss', 0.0):.4f}" if "val_loss" in train_metrics else "val_loss=n/a",
                    f"best={best_val_loss:.4f}",
                    f"time={format_seconds(time.time() - started)}",
                ]
            ),
            log_path,
        )

    metrics = {
        "samples": len(rollout_samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "tokenized_train_samples": len(train_dataset),
        "tokenized_validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
        "best_val_loss": best_val_loss,
        "outputs": {
            "best_adapter": str(output_dir / "best_adapter"),
            "latest_adapter": str(output_dir / "latest_adapter"),
            "history": str(output_dir / "train_history.json"),
            "summary": str(output_dir / "rollout_summary.json"),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json(output_dir / "metrics.json", metrics)
    log_line(
        " | ".join(
            [
                "forecast trace RL complete",
                f"samples={metrics['samples']}",
                f"best_val_loss={best_val_loss:.4f}",
                f"output={output_dir}",
            ]
        ),
        log_path,
    )


if __name__ == "__main__":
    main()
