from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from forecast_trace_grpo_rewards import _as_batch
from forecast_trace_grpo_rewards import _value_at
from forecast_trace_grpo_rewards import build_grpo_context
from forecast_trace_grpo_rewards import completion_to_text
from forecast_trace_grpo_rewards import value_to_log_text
from forecast_trace_grpo_rewards import ForecastTraceGRPOReward
from forecast_trace_judge_runtime_v3 import RolloutAwareFrozenQwenJudge
from forecast_trace_schema import parse_structured_forecast


class AuditedJudgeAugmentedGRPOReward:
    """Frozen-Qwen trace judge plus deterministic reward, with rollout audits."""

    def __init__(
        self,
        model_path: str,
        *,
        policy_name: str = "forecast_trace_reward",
        reward_key: str = "total",
        error_reward: float = -0.25,
        judge_weight: float = 0.2,
        max_new_tokens: int = 256,
        thinking: bool = False,
        cache_path: str | None = None,
        audit_path: Path | str | None = None,
        audit_every: int = 1,
        sample_audit_path: Path | str | None = None,
        sample_audit_every: int = 1,
        sample_audit_limit: int = 2,
    ) -> None:
        self.base = ForecastTraceGRPOReward(
            policy_name=policy_name,
            reward_key=reward_key,
            error_reward=error_reward,
        )
        self.judge = RolloutAwareFrozenQwenJudge(
            model_path,
            max_new_tokens=max_new_tokens,
            thinking=thinking,
            cache_path=cache_path,
        )
        self.judge_weight = max(0.0, float(judge_weight))
        self.error_reward = float(error_reward)
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit_every = max(1, int(audit_every))
        self.sample_audit_path = Path(sample_audit_path) if sample_audit_path else None
        self.sample_audit_every = max(1, int(sample_audit_every))
        self.sample_audit_limit = max(0, int(sample_audit_limit))
        self.call_count = 0
        self.last_breakdowns: list[dict[str, float]] = []

    def __call__(self, prompts: Any = None, completions: Any = None, **kwargs: Any) -> list[float]:
        if completions is None:
            completions = kwargs.get("completion", kwargs.get("response", []))
        batch = _as_batch(completions)
        rewards: list[float] = []
        breakdowns: list[dict[str, float]] = []
        audit_rows: list[dict[str, Any]] = []

        for index, completion in enumerate(batch):
            batch_size = len(batch)
            prompt_value = _value_at(prompts, index, batch_size) if prompts is not None else ""
            completion_text = completion_to_text(completion)
            context = None
            judge: dict[str, Any] = {}
            try:
                context = build_grpo_context(kwargs, index, batch_size)
                prediction = parse_structured_forecast(completion_text, choices=context.choices)
                base = self.base.policy.compute_reward_breakdown(prediction, context.gold, context.trajectory)
                judge = self.judge.score(
                    prediction,
                    query=context.trajectory.metadata.get("query", context.gold),
                    graph=context.trajectory.metadata.get("refined_graph"),
                    context_prompt=completion_to_text(prompt_value),
                )
                answer = float(base.get("answer", 0.0))
                gate = 1.0 if answer > 0.0 else 0.2
                judge_score = float(judge.get("overall", 0.0))
                total = answer + gate * float(base.get("trace", 0.0)) + self.judge_weight * gate * judge_score
                if not math.isfinite(total):
                    raise ValueError("non-finite reward")
                breakdown = {
                    **base,
                    "judge_support": float(judge.get("support", 0.0)),
                    "judge_causal": float(judge.get("causal", 0.0)),
                    "judge_temporal": float(judge.get("temporal", 0.0)),
                    "judge_answer_link": float(judge.get("answer_link", 0.0)),
                    "judge_hallucination": float(judge.get("hallucination", 0.0)),
                    "judge_overall": judge_score,
                    "judge_parsed": float(bool(judge.get("parsed_json", False))),
                    "judge_gate": gate,
                    "total": total,
                }
            except Exception as exc:
                prediction = parse_structured_forecast(completion_text)
                judge = {"parsed_json": False, "reason": f"reward_error: {exc}"}
                total = self.error_reward
                breakdown = {"total": total, "error_reward": 1.0, "judge_parsed": 0.0}

            rewards.append(total)
            breakdowns.append(breakdown)
            audit_rows.append(
                {
                    "query_id": context.trajectory.sample_id if context is not None else "",
                    "prompt": value_to_log_text(prompt_value),
                    "completion": completion_text,
                    "parsed_prediction": prediction,
                    "gold": context.gold if context is not None else {},
                    "reward": total,
                    "reward_breakdown": breakdown,
                    "judge": judge,
                }
            )

        self.last_breakdowns = breakdowns
        self.call_count += 1
        self._write_audit(rewards, breakdowns, audit_rows)
        return rewards

    def _write_audit(
        self,
        rewards: list[float],
        breakdowns: list[dict[str, float]],
        audit_rows: list[dict[str, Any]],
    ) -> None:
        if self.audit_path is not None and self.call_count % self.audit_every == 0:
            keys = sorted({key for breakdown in breakdowns for key in breakdown})
            averages = {
                key: round(sum(float(item.get(key, 0.0)) for item in breakdowns) / max(1, len(breakdowns)), 6)
                for key in keys
            }
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "reward_call": self.call_count,
                            "batch_size": len(rewards),
                            "reward_mean": round(sum(rewards) / max(1, len(rewards)), 6),
                            "reward_min": round(min(rewards), 6) if rewards else 0.0,
                            "reward_max": round(max(rewards), 6) if rewards else 0.0,
                            "breakdown_mean": averages,
                            "judge_parse_rate": averages.get("judge_parsed", 0.0),
                            "error_count": sum(1 for item in breakdowns if item.get("error_reward")),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if self.sample_audit_path is not None and self.sample_audit_limit > 0 and self.call_count % self.sample_audit_every == 0:
            self.sample_audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.sample_audit_path.open("a", encoding="utf-8") as handle:
                for index, row in enumerate(audit_rows[: self.sample_audit_limit]):
                    handle.write(
                        json.dumps({"reward_call": self.call_count, "batch_index": index, **row}, ensure_ascii=False, default=str)
                        + "\n"
                    )
