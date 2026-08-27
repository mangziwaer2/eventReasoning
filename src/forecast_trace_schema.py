from __future__ import annotations

import json
import re
from typing import Any


GENERIC_EVENT_PATTERNS = (
    "tensions rise",
    "tension rises",
    "situation worsens",
    "situation deteriorates",
    "further developments occur",
    "events unfold",
    "things get worse",
    "conflict continues",
    "relations worsen",
)


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text).strip()
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(stripped[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def extract_tagged_json_object(text: str, tag: str) -> dict[str, Any] | None:
    pattern = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(str(text))
    if not match:
        return None
    return extract_first_json_object(match.group(1))


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _as_string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    return max(0.0, min(_safe_float(value), 1.0))


def _event_text_from_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("mention", "normalized_event", "event", "text", "trigger"):
            text = str(value.get(key, "")).strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _event_actor_list(item: dict[str, Any]) -> list[str]:
    if isinstance(item.get("event"), dict):
        event_payload = item["event"]
        actors = event_payload.get("actors", event_payload.get("participants", []))
        if actors:
            return _as_string_list(actors)
    return _as_string_list(item.get("actors", item.get("participants", [])))


def _support_event_refs(item: dict[str, Any]) -> list[str]:
    refs = _as_string_list(item.get("supporting_event_ids", item.get("support_event_ids", [])))
    for support in _as_list(item.get("supporting_events", item.get("support_events", []))):
        if isinstance(support, dict):
            ref = support.get("event_ref", support.get("event_id", support.get("ref")))
            if str(ref or "").strip():
                refs.append(str(ref).strip())
        elif str(support).strip():
            refs.append(str(support).strip())
    return _dedupe_preserve_order(refs)


def _support_edge_refs(item: dict[str, Any]) -> list[str]:
    refs = _as_string_list(item.get("supporting_edge_ids", item.get("supporting_edge_refs", item.get("support_edge_ids", []))))
    for support in _as_list(item.get("supporting_edges", item.get("support_edges", []))):
        if isinstance(support, dict):
            ref = support.get("edge_ref", support.get("edge_id", support.get("ref")))
            if str(ref or "").strip():
                refs.append(str(ref).strip())
        elif str(support).strip():
            refs.append(str(support).strip())
    return _dedupe_preserve_order(refs)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def normalize_trace_event(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        item = {"event": str(item)}
    event_payload = item.get("event")
    event_text = _event_text_from_value(event_payload if event_payload is not None else item.get("mention", item.get("text", "")))
    relative_time = str(item.get("relative_time", "")).strip()
    if not relative_time and isinstance(event_payload, dict):
        relative_time = str(event_payload.get("relative_time", "")).strip()
    return {
        "trace_event_id": str(item.get("trace_event_id", item.get("id", f"ft_{index + 1}"))).strip() or f"ft_{index + 1}",
        "event": event_text,
        "trigger": str(item.get("trigger", event_payload.get("trigger", "") if isinstance(event_payload, dict) else "")).strip(),
        "actors": _event_actor_list(item),
        "relative_time": relative_time,
        "supporting_event_ids": _support_event_refs(item),
        "supporting_edge_ids": _support_edge_refs(item),
        "expected_effect": str(item.get("expected_effect", item.get("effect", ""))).strip(),
        "confidence": _clamp01(item.get("confidence", 0.0)),
        "raw": item,
    }


def normalize_trace_edge(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        item = {}
    source = item.get("source_ref", item.get("source_id", item.get("source", item.get("source_event_id", ""))))
    target = item.get("target_ref", item.get("target_id", item.get("target", item.get("target_event_id", ""))))
    return {
        "trace_edge_id": str(item.get("trace_edge_id", item.get("edge_id", f"fte_{index + 1}"))).strip() or f"fte_{index + 1}",
        "source_id": str(source or "").strip(),
        "target_id": str(target or "").strip(),
        "relation_type": str(item.get("relation_type", item.get("relation", "causes"))).strip() or "causes",
        "confidence": _clamp01(item.get("confidence", item.get("score", 0.0))),
        "raw": item,
    }


def normalize_forecast_trace(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    intermediate_events = [
        normalize_trace_event(item, index)
        for index, item in enumerate(_as_list(payload.get("intermediate_events", payload.get("events", []))))
    ]
    trace_edges = [
        normalize_trace_edge(item, index)
        for index, item in enumerate(_as_list(payload.get("trace_edges", payload.get("edges", []))))
    ]
    return {
        "intermediate_events": intermediate_events,
        "trace_edges": trace_edges,
    }


def normalize_final_answer(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    event_code = str(payload.get("event_code", payload.get("predicted_event_base_code", ""))).strip()
    support_event_ids = _support_event_refs(payload)
    support_edge_ids = _support_edge_refs(payload)
    return {
        "event_code": event_code,
        "event_description": str(payload.get("event_description", payload.get("description", payload.get("event", payload.get("forecast_event", ""))))).strip(),
        "event": str(payload.get("event", payload.get("forecast_event", payload.get("event_description", payload.get("description", ""))))).strip(),
        "confidence": _clamp01(payload.get("confidence", 0.0)),
        "supporting_event_ids": support_event_ids,
        "supporting_edge_ids": support_edge_ids,
        "raw": payload,
    }


def parse_structured_forecast(raw_response: str) -> dict[str, Any]:
    payload = extract_first_json_object(raw_response)
    tagged_trace = extract_tagged_json_object(raw_response, "forecast_trace")
    tagged_answer = extract_tagged_json_object(raw_response, "final_answer")
    parsed_json = payload is not None or tagged_trace is not None or tagged_answer is not None

    if payload is None:
        payload = {}
    forecast_trace_payload: Any = tagged_trace if tagged_trace is not None else payload.get("forecast_trace")
    if forecast_trace_payload is None and ("intermediate_events" in payload or "trace_edges" in payload):
        forecast_trace_payload = payload

    final_answer_payload: Any = tagged_answer if tagged_answer is not None else payload.get("final_answer")
    if final_answer_payload is None:
        final_answer_payload = {
            "event_code": payload.get("event_code", payload.get("predicted_event_base_code", "")),
            "forecast_event": payload.get("forecast_event", ""),
            "event_description": payload.get("event_description", payload.get("description", "")),
            "confidence": payload.get("confidence", 0.0),
            "supporting_event_ids": payload.get("supporting_event_ids", payload.get("support_event_ids", [])),
            "supporting_edge_ids": payload.get("supporting_edge_ids", []),
        }

    if not parsed_json:
        code_match = re.search(r"\b\d{3}\b", str(raw_response))
        final_answer_payload = {"event_code": code_match.group(0) if code_match else "", "confidence": 0.0}

    forecast_trace = normalize_forecast_trace(forecast_trace_payload)
    final_answer = normalize_final_answer(final_answer_payload)
    answer_items = payload.get("answers", [])
    normalized_answers = [
        normalize_final_answer(item)
        for item in answer_items
        if isinstance(item, dict)
    ] if isinstance(answer_items, list) else []
    normalized_answers = [item for item in normalized_answers if item["event_code"]]
    if normalized_answers:
        final_answer = normalized_answers[0]
    elif final_answer["event_code"]:
        normalized_answers = [final_answer]
    alternatives = _as_string_list(payload.get("alternative_event_base_codes", []))
    alternatives.extend(item["event_code"] for item in normalized_answers[1:])
    predicted_codes = _dedupe_preserve_order(
        [final_answer["event_code"]] + alternatives
    )
    alternatives = [item for item in predicted_codes if item != final_answer["event_code"]]
    support_event_ids = _dedupe_preserve_order(
        final_answer["supporting_event_ids"]
        + [
            ref
            for event in forecast_trace["intermediate_events"]
            for ref in event.get("supporting_event_ids", [])
        ]
    )
    return {
        "parsed_json": parsed_json,
        "abstain": bool(payload.get("abstain", False)),
        "forecast_trace": forecast_trace,
        "final_answer": final_answer,
        "predicted_event_base_code": final_answer["event_code"],
        "predicted_event_base_codes": predicted_codes,
        "alternative_event_base_codes": alternatives,
        "answers": normalized_answers,
        "predicted_relation_name": str(payload.get("predicted_relation_name", "")).strip(),
        "forecast_event": final_answer["event"]
        or "; ".join(event["event"] for event in forecast_trace["intermediate_events"] if event.get("event")),
        "confidence": final_answer["confidence"],
        "rationale": str(payload.get("rationale", payload.get("explanation", ""))).strip(),
        "support_event_ids": support_event_ids,
        "raw_payload": payload if parsed_json else None,
    }


def is_generic_event(text: str) -> bool:
    normalized = " ".join(str(text).lower().split())
    if not normalized:
        return True
    if any(pattern in normalized for pattern in GENERIC_EVENT_PATTERNS):
        return True
    tokens = normalized.split()
    return len(tokens) < 3 or len(tokens) > 32


def parse_relative_time_score(value: str) -> float:
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return 0.0
    if text in {"beforet", "pre-t", "pret"}:
        return 0.8
    match = re.fullmatch(r"t-(\d+)", text)
    if match:
        return 1.0 if int(match.group(1)) > 0 else 0.0
    match = re.fullmatch(r"t\+(\d+)", text)
    if match:
        return 0.0
    if "before" in text or "prior" in text:
        return 0.6
    return 0.0
