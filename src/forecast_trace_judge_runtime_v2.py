from __future__ import annotations

import re
from typing import Any

from forecast_trace_judge_runtime import FrozenQwenJudgeRuntime


def graph_from_forecast_context(context: str) -> dict[str, list[dict[str, Any]]]:
    """Recover prompt-visible Hxx/Rxx references for rollout audit rows."""
    events: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for line in str(context).splitlines():
        event = re.match(r"\s*-\s*(H\d+)\s*\|.*?mention=(.*?)\s*\|\s*participants=", line)
        if event:
            events.append({"event_id": event.group(1), "text": event.group(2).strip()})
            continue
        edge = re.match(
            r"\s*-\s*(R\d+)\s*\|.*?\s(H\d+)\s*->\s*([^| ]+)\s*\|\s*relation=([^|]+)\|\s*confidence=([0-9.]+)",
            line,
        )
        if edge:
            edges.append(
                {
                    "edge_id": edge.group(1),
                    "source_event_id": edge.group(2),
                    "target_event_id": edge.group(3),
                    "relation_type": edge.group(4).strip(),
                    "score": float(edge.group(5)),
                }
            )
    return {"events": events, "edges": edges}


class ContextAwareFrozenQwenJudge(FrozenQwenJudgeRuntime):
    def score(self, prediction: dict[str, Any], *, query: Any = None, graph: Any = None, context_prompt: str = "") -> dict[str, Any]:
        if graph is None and context_prompt:
            graph = graph_from_forecast_context(context_prompt)
        return super().score(prediction, query=query, graph=graph, context_prompt=context_prompt)

