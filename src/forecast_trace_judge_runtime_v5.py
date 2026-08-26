from __future__ import annotations

from typing import Any

from forecast_trace_judge_runtime_v2 import graph_from_forecast_context
from forecast_trace_judge_runtime_v3 import RolloutAwareFrozenQwenJudge
from forecast_trace_judge_runtime_v3 import normalize_context_prompt


class PromptReferenceFrozenQwenJudge(RolloutAwareFrozenQwenJudge):
    """Judge in the same Hxx/Rxx reference space exposed to the forecasting policy."""

    def score(self, prediction: dict[str, Any], *, query: Any = None, graph: Any = None, context_prompt: str = "") -> dict[str, Any]:
        normalized_context = normalize_context_prompt(context_prompt)
        prompt_graph = graph_from_forecast_context(normalized_context) if normalized_context else {}
        # Forecast traces are trained to cite Hxx/Rxx. Prefer that prompt-visible
        # graph rather than internal eN/pred_edge_N identifiers from trajectory metadata.
        if prompt_graph.get("events") or prompt_graph.get("edges"):
            graph = prompt_graph
        return super().score(
            prediction,
            query=query,
            graph=graph,
            context_prompt=normalized_context,
        )
