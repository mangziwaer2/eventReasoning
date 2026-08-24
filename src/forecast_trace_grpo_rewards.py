from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forecast_trace_schema import parse_structured_forecast
from forecast_trace_prompt import FORECAST_TRACE_SYSTEM_PROMPT
from rl_pipeline_hooks import PipelineStep
from rl_pipeline_hooks import PipelineTrajectory
from rl_pipeline_hooks import build_pipeline_policy


DEFAULT_FORECAST_SYSTEM_PROMPT = FORECAST_TRACE_SYSTEM_PROMPT
DEFAULT_ERROR_REWARD = -0.25


def _json_loads_if_needed(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _as_dict(value: Any) -> dict[str, Any]:
    value = _json_loads_if_needed(value)
    return value if isinstance(value, dict) else {}


def _as_choices(value: Any) -> list[dict[str, Any]]:
    value = _json_loads_if_needed(value)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_batch(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _looks_like_chat_messages(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict) and ("content" in item or "text" in item) for item in value
    )


def _value_at(value: Any, index: int, batch_size: int, *, scalar_list: bool = False) -> Any:
    if scalar_list:
        return value
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == batch_size:
        return value[index]
    return value


def _kw_value(
    kwargs: dict[str, Any],
    names: tuple[str, ...],
    index: int,
    batch_size: int,
    default: Any = None,
    *,
    scalar_list: bool = False,
) -> Any:
    for name in names:
        if name in kwargs:
            return _value_at(kwargs[name], index, batch_size, scalar_list=scalar_list)
    return default


def completion_to_text(completion: Any) -> str:
    """Normalize TRL text or chat-style completion payloads to assistant text."""

    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion.strip()
    completion = _json_loads_if_needed(completion)
    if isinstance(completion, dict):
        if not any(key in completion for key in ("content", "text", "generated_text")):
            return json.dumps(completion, ensure_ascii=False)
        content = completion.get("content", completion.get("text", completion.get("generated_text", "")))
        if isinstance(content, list):
            return " ".join(completion_to_text(item) for item in content).strip()
        return str(content or "").strip()
    if isinstance(completion, list):
        if _looks_like_chat_messages(completion):
            assistant_messages = [
                item for item in completion if str(item.get("role", "")).strip().lower() == "assistant"
            ]
            message = assistant_messages[-1] if assistant_messages else completion[-1]
            return completion_to_text(message)
        return " ".join(completion_to_text(item) for item in completion).strip()
    return str(completion).strip()


def pipeline_trajectory_from_payload(payload: Any, sample_id: str = "") -> PipelineTrajectory:
    """Build a PipelineTrajectory from JSON/JSONL payloads used by rollout logs."""

    if isinstance(payload, PipelineTrajectory):
        return payload
    raw = _as_dict(payload)
    trajectory = PipelineTrajectory(
        sample_id=str(raw.get("sample_id", sample_id)),
        final_reward=raw.get("final_reward"),
        metadata=dict(raw.get("metadata", {})) if isinstance(raw.get("metadata"), dict) else {},
    )
    steps = raw.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
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
    return trajectory


def _context_row_from_kwargs(kwargs: dict[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    row = _kw_value(kwargs, ("row", "rows", "reward_context", "reward_contexts"), index, batch_size, {})
    return _as_dict(row)


def _context_value(
    row: dict[str, Any],
    kwargs: dict[str, Any],
    names: tuple[str, ...],
    index: int,
    batch_size: int,
    default: Any = None,
    *,
    scalar_list: bool = False,
) -> Any:
    for name in names:
        if name in kwargs:
            return _kw_value(kwargs, names, index, batch_size, default, scalar_list=scalar_list)
    for name in names:
        if name in row:
            return row[name]
    return default


@dataclass(slots=True)
class ForecastTraceGRPOContext:
    gold: dict[str, Any]
    trajectory: PipelineTrajectory
    choices: list[dict[str, Any]]


def build_grpo_context(kwargs: dict[str, Any], index: int, batch_size: int) -> ForecastTraceGRPOContext:
    row = _context_row_from_kwargs(kwargs, index, batch_size)
    sample_id = str(
        _context_value(row, kwargs, ("sample_id", "query_id"), index, batch_size, row.get("query_id", ""))
    )
    gold = _as_dict(
        _context_value(
            row,
            kwargs,
            ("mirai_query", "gold", "gold_summary", "reference", "label", "labels"),
            index,
            batch_size,
            {},
        )
    )
    trajectory = pipeline_trajectory_from_payload(
        _context_value(row, kwargs, ("trajectory",), index, batch_size, {}),
        sample_id=sample_id,
    )
    choices = _as_choices(_context_value(row, kwargs, ("choices", "choice_pool"), index, batch_size, []))
    if choices:
        trajectory.metadata.setdefault("choices", choices)
    for key in ("refined_graph", "event_ref_to_id", "edge_ref_to_id"):
        value = _context_value(row, kwargs, (key,), index, batch_size, None)
        if value is not None:
            trajectory.metadata.setdefault(key, _json_loads_if_needed(value))
    return ForecastTraceGRPOContext(gold=gold, trajectory=trajectory, choices=choices)


class ForecastTraceGRPOReward:
    """TRL GRPO-compatible reward callable for structured forecast traces.

    GRPOTrainer calls reward functions with a batch of generated completions and
    the non-prompt dataset columns as keyword arguments. This adapter parses each
    completion, reuses the deterministic pipeline reward policy, and returns one
    finite float per completion.
    """

    def __init__(
        self,
        policy_name: str = "forecast_trace_reward",
        reward_key: str = "total",
        error_reward: float = DEFAULT_ERROR_REWARD,
        audit_path: Path | str | None = None,
        audit_every: int = 1,
    ) -> None:
        self.policy_name = policy_name
        self.reward_key = reward_key
        self.error_reward = float(error_reward)
        self.policy = build_pipeline_policy(policy_name)
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit_every = max(1, int(audit_every))
        self.call_count = 0
        self.last_breakdowns: list[dict[str, float]] = []

    def score_completion(
        self,
        completion: Any,
        *,
        gold: dict[str, Any] | None = None,
        trajectory: PipelineTrajectory | dict[str, Any] | str | None = None,
        choices: list[dict[str, Any]] | str | None = None,
    ) -> dict[str, float]:
        trajectory_obj = pipeline_trajectory_from_payload(trajectory or {})
        normalized_choices = _as_choices(choices or [])
        if normalized_choices:
            trajectory_obj.metadata.setdefault("choices", normalized_choices)
        prediction = parse_structured_forecast(completion_to_text(completion), choices=normalized_choices)
        breakdown = self.policy.compute_reward_breakdown(prediction, gold or {}, trajectory_obj)
        return {str(key): float(value) for key, value in breakdown.items() if isinstance(value, (int, float))}

    def __call__(self, prompts: Any = None, completions: Any = None, **kwargs: Any) -> list[float]:
        if completions is None:
            completions = _kw_value(kwargs, ("completion", "response", "raw_response"), 0, 1, [])
        batch = _as_batch(completions)
        batch_size = len(batch)
        rewards: list[float] = []
        breakdowns: list[dict[str, float]] = []
        for index, completion in enumerate(batch):
            try:
                context = build_grpo_context(kwargs, index, batch_size)
                breakdown = self.score_completion(
                    completion,
                    gold=context.gold,
                    trajectory=context.trajectory,
                    choices=context.choices,
                )
                reward = float(breakdown.get(self.reward_key, breakdown.get("total", self.error_reward)))
                if not math.isfinite(reward):
                    reward = self.error_reward
                    breakdown = {**breakdown, self.reward_key: reward, "error_reward": 1.0}
            except Exception:
                reward = self.error_reward
                breakdown = {self.reward_key: reward, "error_reward": 1.0}
            rewards.append(reward)
            breakdowns.append(breakdown)
        self.last_breakdowns = breakdowns
        self.call_count += 1
        if self.audit_path is not None and self.call_count % self.audit_every == 0:
            numeric_keys = sorted({key for breakdown in breakdowns for key in breakdown})
            averages = {
                key: round(
                    sum(float(breakdown.get(key, 0.0)) for breakdown in breakdowns) / max(1, len(breakdowns)),
                    6,
                )
                for key in numeric_keys
            }
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "reward_call": self.call_count,
                            "batch_size": len(batch),
                            "reward_mean": round(sum(rewards) / max(1, len(rewards)), 6),
                            "reward_min": round(min(rewards), 6) if rewards else 0.0,
                            "reward_max": round(max(rewards), 6) if rewards else 0.0,
                            "breakdown_mean": averages,
                            "error_count": sum(1 for breakdown in breakdowns if breakdown.get("error_reward")),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return rewards


def make_forecast_trace_grpo_reward_func(
    policy_name: str = "forecast_trace_reward",
    reward_key: str = "total",
    error_reward: float = DEFAULT_ERROR_REWARD,
) -> ForecastTraceGRPOReward:
    return ForecastTraceGRPOReward(policy_name=policy_name, reward_key=reward_key, error_reward=error_reward)


def rollout_row_to_grpo_sample(row: dict[str, Any], *, chat_prompt: bool = True) -> dict[str, Any]:
    """Convert one pipeline rollout row into a GRPOTrainer dataset row."""

    prompt_text = str(row.get("forecast_prompt", "")).strip()
    system_prompt = str(row.get("forecast_system_prompt", DEFAULT_FORECAST_SYSTEM_PROMPT)).strip()
    if chat_prompt:
        prompt: Any = [
            {"role": "system", "content": system_prompt or DEFAULT_FORECAST_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
    else:
        prompt = prompt_text
    reward_context = {
        "schema_version": str(row.get("schema_version", "forecast-trace-grpo-context-v1")),
        "stage": str(row.get("stage", "prompt-context")),
        "query_id": str(row.get("query_id", row.get("sample_id", ""))),
        "mirai_query": row.get("mirai_query", {}),
        "trajectory": row.get("trajectory", {}),
        "choices": row.get("choices", []),
    }
    return {
        "prompt": prompt,
        "query_id": reward_context["query_id"],
        "reward_context": json.dumps(reward_context, ensure_ascii=False),
    }


def rollout_rows_to_grpo_samples(rows: list[dict[str, Any]], *, chat_prompt: bool = True) -> list[dict[str, Any]]:
    return [rollout_row_to_grpo_sample(row, chat_prompt=chat_prompt) for row in rows if row.get("forecast_prompt")]
