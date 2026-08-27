from __future__ import annotations

import json
from typing import Any


JUDGE_SYSTEM_PROMPT = (
    "You are a strict critic of structured future-event traces. "
    "Judge only whether the proposed trace is supported by the supplied historical events and graph, "
    "whether its causal direction and time are coherent, and whether it connects to the proposed answer. "
    "Historical events are evidence, not forecast steps: exact or near-verbatim restatements of them "
    "as intermediate trace events are invalid. "
    "Do not use outside knowledge and do not decide correctness from the hidden gold label. Return JSON only."
)


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _query_text(query: Any) -> str:
    if isinstance(query, dict):
        return str(query.get("text", query.get("query_text", ""))).strip()
    return str(query or "").strip()


def _graph_text(graph: Any) -> str:
    if not isinstance(graph, dict):
        return "- graph unavailable"
    lines: list[str] = []
    for event in graph.get("events", []):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id", "")).strip()
        text = str(
            event.get("text")
            or event.get("normalized_text")
            or event.get("mention")
            or event.get("event", "")
        ).strip()
        if event_id:
            lines.append(f"- {event_id}: {text or '(no event text)'}")
    if not lines:
        lines.append("- no graph events")
    lines.append("Edges:")
    edge_lines = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source_event_id", "")).strip()
        target = str(edge.get("target_event_id", "")).strip()
        relation = str(edge.get("relation_type", "")).strip()
        score = _clamp(edge.get("score", edge.get("confidence", 0.0)))
        if source and target:
            edge_lines.append(f"- {source} -> {target} | {relation} | confidence={score:.3f}")
    lines.extend(edge_lines or ["- no graph edges"])
    return "\n".join(lines)


def build_judge_prompt(
    prediction: dict[str, Any],
    *,
    query: Any = None,
    graph: Any = None,
    context_prompt: str = "",
    max_context_chars: int = 12000,
) -> str:
    """Build the only prompt format used by the frozen Qwen trace judge."""
    context = str(context_prompt or "").strip()
    if len(context) > max_context_chars:
        context = context[-max_context_chars:]
    trace = prediction.get("forecast_trace", {}) if isinstance(prediction, dict) else {}
    answer = prediction.get("final_answer", {}) if isinstance(prediction, dict) else {}
    return (
        "Score the candidate trace on a 0 to 1 scale. A high score requires concrete events, "
        "valid support in the graph, correct edge direction, a pre-target relative time, and a direct "
        "connection to the proposed answer. A trace event must be a distinct future hypothesis even when it cites a historical event. "
        "Penalize exact or near-verbatim historical copies, copied placeholders, generic claims, unsupported "
        "facts, and answer links that are not explained by the trace.\n\n"
        "Output exactly:\n"
        '{"support":0.0,"causal":0.0,"temporal":0.0,"answer_link":0.0,"hallucination":0.0,"overall":0.0,"reason":"short reason"}\n\n'
        f"Query: {_query_text(query)}\n\n"
        "Graph:\n"
        f"{_graph_text(graph)}\n\n"
        "Candidate trace:\n"
        f"{_json(trace)}\n\n"
        "Candidate answer:\n"
        f"{_json(answer)}\n\n"
        + ("Original forecasting context:\n" + context + "\n" if context else "")
    )


def build_description_judge_prompt(
    code: str,
    canonical_description: str,
    generated_description: str,
) -> str:
    """Build a semantic, non-exact-match code-description consistency prompt."""
    return (
        "Evaluate semantic equivalence, not exact wording. The generated description may be a "
        "concise paraphrase, but it must describe the same event type as the canonical description. "
        "Do not judge whether the code is correct for a query; only compare the two descriptions.\n\n"
        "Output exactly:\n"
        '{"match":0.0,"reason":"short reason"}\n\n'
        f"Event code: {code}\n"
        f"Canonical description: {canonical_description}\n"
        f"Generated description: {generated_description}\n"
    )
