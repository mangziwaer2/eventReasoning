from __future__ import annotations

import json
from typing import Any

from forecast_trace_judge_runtime_v2 import ContextAwareFrozenQwenJudge
from forecast_trace_judge_runtime_v2 import graph_from_forecast_context


def normalize_context_prompt(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                payload = json.loads(text)
                if isinstance(payload, list):
                    parts = []
                    for item in payload:
                        if isinstance(item, dict) and item.get("content"):
                            parts.append(str(item["content"]))
                    if parts:
                        return "\n\n".join(parts)
            except json.JSONDecodeError:
                pass
        return text
    return str(value or "").strip()


class RolloutAwareFrozenQwenJudge(ContextAwareFrozenQwenJudge):
    def score(self, prediction: dict[str, Any], *, query: Any = None, graph: Any = None, context_prompt: str = "") -> dict[str, Any]:
        normalized_context = normalize_context_prompt(context_prompt)
        if graph is None and normalized_context:
            graph = graph_from_forecast_context(normalized_context)
        return super().score(
            prediction,
            query=query,
            graph=graph,
            context_prompt=normalized_context,
        )

