from __future__ import annotations

from typing import Any

from forecast_trace_judge_reward_v3 import AuditedJudgeAugmentedGRPOReward
from forecast_trace_judge_runtime_v5 import PromptReferenceFrozenQwenJudge


class PromptReferenceAuditedJudgeGRPOReward(AuditedJudgeAugmentedGRPOReward):
    """Audited judge reward with prompt-aligned graph references."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        super().__init__(model_path, **kwargs)
        self.judge = PromptReferenceFrozenQwenJudge(
            model_path,
            max_new_tokens=kwargs.get("max_new_tokens", 384),
            thinking=kwargs.get("thinking", False),
            cache_path=kwargs.get("cache_path"),
        )
