from __future__ import annotations

from typing import Any

from forecast_trace_judge_reward_v3 import AuditedJudgeAugmentedGRPOReward
from forecast_trace_judge_runtime_v6 import PromptReferenceRobustFrozenQwenJudge


class PromptReferenceRobustJudgeGRPOReward(AuditedJudgeAugmentedGRPOReward):
    """Audited judge reward with prompt-aligned references and truncation recovery."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        super().__init__(model_path, **kwargs)
        self.judge = PromptReferenceRobustFrozenQwenJudge(
            model_path,
            max_new_tokens=kwargs.get("max_new_tokens", 384),
            thinking=kwargs.get("thinking", False),
            cache_path=kwargs.get("cache_path"),
        )
