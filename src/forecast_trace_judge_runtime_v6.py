from __future__ import annotations

from typing import Any

from forecast_trace_judge_runtime_v2 import graph_from_forecast_context
from forecast_trace_judge_runtime_v3 import normalize_context_prompt
from forecast_trace_judge_runtime_v4 import RobustRolloutAwareFrozenQwenJudge


class PromptReferenceRobustFrozenQwenJudge(RobustRolloutAwareFrozenQwenJudge):
    """Judge using prompt-visible Hxx/Rxx references and robust JSON recovery."""

    def score(self, prediction: dict[str, Any], *, query: Any = None, graph: Any = None, context_prompt: str = "") -> dict[str, Any]:
        normalized_context = normalize_context_prompt(context_prompt)
        prompt_graph = graph_from_forecast_context(normalized_context) if normalized_context else {}
        # The policy cites Hxx/Rxx, so prefer the graph in exactly that namespace.
        if prompt_graph.get("events") or prompt_graph.get("edges"):
            graph = prompt_graph
        return super().score(
            prediction,
            query=query,
            graph=graph,
            context_prompt=normalized_context,
        )
