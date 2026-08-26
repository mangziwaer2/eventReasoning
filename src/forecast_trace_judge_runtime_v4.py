from __future__ import annotations

from typing import Any

from forecast_trace_judge_parser import parse_judge_response_robust
from forecast_trace_judge_runtime_v3 import RolloutAwareFrozenQwenJudge


class RobustRolloutAwareFrozenQwenJudge(RolloutAwareFrozenQwenJudge):
    """Rollout-aware judge that preserves scalar scores from truncated JSON."""

    def score(self, prediction: dict[str, Any], *, query: Any = None, graph: Any = None, context_prompt: str = "") -> dict[str, Any]:
        result = super().score(prediction, query=query, graph=graph, context_prompt=context_prompt)
        if result.get("parsed_json") or not result.get("raw_response"):
            return result
        return parse_judge_response_robust(str(result.get("raw_response", ""))) | {
            "cached": bool(result.get("cached", False)),
        }
