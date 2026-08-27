from __future__ import annotations

import heapq
from typing import Any


def _as_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return item if isinstance(item, dict) else {}


def graph_events(graph: Any) -> list[dict[str, Any]]:
    graph_dict = _as_dict(graph)
    return [item for item in graph_dict.get("events", []) if isinstance(item, dict)]


def graph_edges(graph: Any) -> list[dict[str, Any]]:
    graph_dict = _as_dict(graph)
    return [item for item in graph_dict.get("edges", []) if isinstance(item, dict)]


def graph_event_ids(graph: Any) -> set[str]:
    return {str(item.get("event_id", "")).strip() for item in graph_events(graph) if str(item.get("event_id", "")).strip()}


def graph_edge_ids(graph: Any) -> set[str]:
    return {str(item.get("edge_id", "")).strip() for item in graph_edges(graph) if str(item.get("edge_id", "")).strip()}


def resolve_event_ref(ref: str, event_ref_to_id: dict[str, str] | None = None) -> str:
    text = str(ref).strip()
    if event_ref_to_id and text in event_ref_to_id:
        return event_ref_to_id[text]
    return text


def resolve_edge_ref(ref: str, edge_ref_to_id: dict[str, str] | None = None) -> str:
    text = str(ref).strip()
    if edge_ref_to_id and text in edge_ref_to_id:
        return edge_ref_to_id[text]
    return text


def answer_aliases(final_answer: dict[str, Any], answers: list[dict[str, Any]] | None = None) -> set[str]:
    aliases: set[str] = {"answers"} if answers else set()
    values = [final_answer.get("event_code", "")]
    for answer in answers or []:
        if isinstance(answer, dict):
            values.append(answer.get("event_code", ""))
    for raw_value in values:
        value = str(raw_value).strip()
        if not value:
            continue
        aliases.update({value, f"answer_{value}", f"choice_{value}", f"code_{value}"})
    return aliases


def trace_event_ids(forecast_trace: dict[str, Any]) -> set[str]:
    return {
        str(item.get("trace_event_id", "")).strip()
        for item in forecast_trace.get("intermediate_events", [])
        if isinstance(item, dict) and str(item.get("trace_event_id", "")).strip()
    }


def build_augmented_adjacency(
    graph: Any,
    forecast_trace: dict[str, Any],
    event_ref_to_id: dict[str, str] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for edge in graph_edges(graph):
        source = str(edge.get("source_event_id", "")).strip()
        target = str(edge.get("target_event_id", "")).strip()
        if not source or not target:
            continue
        confidence = edge.get("score", edge.get("confidence", 0.0))
        try:
            score = max(0.0, min(float(confidence), 1.0))
        except (TypeError, ValueError):
            score = 0.0
        adjacency.setdefault(source, []).append((target, score))

    for trace_edge in forecast_trace.get("trace_edges", []):
        if not isinstance(trace_edge, dict):
            continue
        source = resolve_event_ref(trace_edge.get("source_id", ""), event_ref_to_id)
        target = resolve_event_ref(trace_edge.get("target_id", ""), event_ref_to_id)
        if not source or not target:
            continue
        try:
            score = max(0.0, min(float(trace_edge.get("confidence", 0.0)), 1.0))
        except (TypeError, ValueError):
            score = 0.0
        adjacency.setdefault(source, []).append((target, score))
    return adjacency


def best_path_score(
    adjacency: dict[str, list[tuple[str, float]]],
    start_nodes: set[str],
    target_nodes: set[str],
    max_depth: int = 6,
) -> float:
    if not start_nodes or not target_nodes:
        return 0.0
    heap: list[tuple[float, str, int]] = []
    best_seen: dict[tuple[str, int], float] = {}
    for node in start_nodes:
        heapq.heappush(heap, (-1.0, node, 0))
    best_target_score = 0.0
    while heap:
        negative_score, node, depth = heapq.heappop(heap)
        score = -negative_score
        if node in target_nodes and depth > 0:
            best_target_score = max(best_target_score, score)
            continue
        if depth >= max_depth:
            continue
        for next_node, edge_score in adjacency.get(node, []):
            next_score = score * max(0.0, min(edge_score, 1.0))
            state = (next_node, depth + 1)
            if next_score <= best_seen.get(state, -1.0):
                continue
            best_seen[state] = next_score
            heapq.heappush(heap, (-next_score, next_node, depth + 1))
    return best_target_score


def graph_bridge_score(
    graph: Any,
    prediction: dict[str, Any],
    event_ref_to_id: dict[str, str] | None = None,
) -> float:
    forecast_trace = prediction.get("forecast_trace", {})
    if not isinstance(forecast_trace, dict):
        return 0.0
    final_answer = prediction.get("final_answer", {})
    if not isinstance(final_answer, dict):
        return 0.0

    start_nodes: set[str] = set()
    for event in forecast_trace.get("intermediate_events", []):
        if not isinstance(event, dict):
            continue
        for ref in event.get("supporting_event_ids", []):
            resolved = resolve_event_ref(ref, event_ref_to_id)
            if resolved:
                start_nodes.add(resolved)
    for ref in final_answer.get("supporting_event_ids", []):
        resolved = resolve_event_ref(ref, event_ref_to_id)
        if resolved:
            start_nodes.add(resolved)
    if not start_nodes:
        start_nodes = graph_event_ids(graph)

    answers = prediction.get("answers", [])
    targets = answer_aliases(final_answer, answers if isinstance(answers, list) else [])
    if not targets:
        return 0.0
    adjacency = build_augmented_adjacency(graph, forecast_trace, event_ref_to_id=event_ref_to_id)
    return best_path_score(adjacency, start_nodes=start_nodes, target_nodes=targets)
