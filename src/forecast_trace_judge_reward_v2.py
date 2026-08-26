from __future__ import annotations

from typing import Any

from forecast_trace_grpo_rewards import ForecastTraceGRPOReward
from forecast_trace_grpo_rewards import _value_at
from forecast_trace_grpo_rewards import build_grpo_context
from forecast_trace_grpo_rewards import completion_to_text
from forecast_trace_judge_runtime_v3 import RolloutAwareFrozenQwenJudge
from forecast_trace_schema import parse_structured_forecast


class JudgeAugmentedGRPOReward:
    """Optional GRPO callable combining deterministic checks and frozen Qwen."""

    def __init__(
        self,
        model_path: str,
        *,
        judge_weight: float = 0.2,
        max_new_tokens: int = 256,
        thinking: bool = False,
        cache_path: str | None = None,
    ) -> None:
        self.base = ForecastTraceGRPOReward()
        self.judge = RolloutAwareFrozenQwenJudge(
            model_path,
            max_new_tokens=max_new_tokens,
            thinking=thinking,
            cache_path=cache_path,
        )
        self.judge_weight = max(0.0, float(judge_weight))
        self.last_breakdowns: list[dict[str, float]] = []

    def __call__(self, prompts: Any = None, completions: Any = None, **kwargs: Any) -> list[float]:
        if completions is None:
            completions = kwargs.get("completion", kwargs.get("response", []))
        if isinstance(completions, (str, dict)):
            completions = [completions]
        batch = list(completions or [])
        rewards: list[float] = []
        breakdowns: list[dict[str, float]] = []
        for index, completion in enumerate(batch):
            batch_size = len(batch)
            context = build_grpo_context(kwargs, index, batch_size)
            prompt_value = _value_at(prompts, index, batch_size) if prompts is not None else ""
            prediction = parse_structured_forecast(completion_to_text(completion), choices=context.choices)
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
            breakdowns.append(
                {
                    **base,
                    "judge_support": float(judge.get("support", 0.0)),
                    "judge_causal": float(judge.get("causal", 0.0)),
                    "judge_temporal": float(judge.get("temporal", 0.0)),
                    "judge_answer_link": float(judge.get("answer_link", 0.0)),
                    "judge_hallucination": float(judge.get("hallucination", 0.0)),
                    "judge_overall": judge_score,
                    "judge_gate": gate,
                    "total": total,
                }
            )
            rewards.append(total)
        self.last_breakdowns = breakdowns
        return rewards

