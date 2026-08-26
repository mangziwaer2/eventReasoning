from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from forecast_trace_schema import extract_first_json_object
from local_llm import LocalQwenGenerator
from path_utils import resolve_repo_path


JUDGE_SYSTEM_PROMPT = (
    "You are a strict critic of structured future-event traces. "
    "Judge only whether the proposed trace is supported by the supplied historical events and graph, "
    "whether its causal direction and time are coherent, and whether it connects to the proposed answer. "
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
    context = str(context_prompt or "").strip()
    if len(context) > max_context_chars:
        context = context[-max_context_chars:]
    trace = prediction.get("forecast_trace", {}) if isinstance(prediction, dict) else {}
    answer = prediction.get("final_answer", {}) if isinstance(prediction, dict) else {}
    return (
        "Score the candidate trace on a 0 to 1 scale. A high score requires concrete events, "
        "valid support in the graph, correct edge direction, a pre-target relative time, and a direct "
        "connection to the proposed answer. Penalize copied placeholders, generic claims, unsupported "
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


def parse_judge_response(raw_response: str) -> dict[str, Any]:
    payload = extract_first_json_object(str(raw_response)) or {}
    scores = {
        key: _clamp(payload.get(key, 0.0))
        for key in ("support", "causal", "temporal", "answer_link", "hallucination", "overall")
    }
    if "overall" not in payload:
        scores["overall"] = (
            0.3 * scores["support"]
            + 0.25 * scores["causal"]
            + 0.2 * scores["temporal"]
            + 0.25 * scores["answer_link"]
        ) * (1.0 - scores["hallucination"])
    return {
        "parsed_json": bool(payload),
        **scores,
        "reason": str(payload.get("reason", "")).strip(),
        "raw_response": str(raw_response).strip(),
    }


class FrozenQwenTraceJudge:
    """Lazy, frozen Qwen critic used as an optional process reward."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        max_new_tokens: int = 256,
        thinking: bool = False,
        cache_path: Path | str | None = None,
        max_context_chars: int = 12000,
    ) -> None:
        self.model_path = resolve_repo_path(str(model_path))
        self.max_new_tokens = max(32, int(max_new_tokens))
        self.thinking = bool(thinking)
        self.max_context_chars = max(1000, int(max_context_chars))
        self.cache_path = resolve_repo_path(str(cache_path)) if cache_path else None
        self._generator: LocalQwenGenerator | None = None
        self._cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self._cache = {str(key): value for key, value in payload.items() if isinstance(value, dict)}
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    def _load(self) -> LocalQwenGenerator:
        if self._generator is None:
            self._generator = LocalQwenGenerator(self.model_path, max_new_tokens=self.max_new_tokens)
        return self._generator

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def score(
        self,
        prediction: dict[str, Any],
        *,
        query: Any = None,
        graph: Any = None,
        context_prompt: str = "",
    ) -> dict[str, Any]:
        judge_prompt = build_judge_prompt(
            prediction,
            query=query,
            graph=graph,
            context_prompt=context_prompt,
            max_context_chars=self.max_context_chars,
        )
        cache_key = hashlib.sha256(judge_prompt.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return {**self._cache[cache_key], "cached": True}
        try:
            prompt = judge_prompt if self.thinking else judge_prompt + "\n/no_think"
            raw = self._load().generate(
                prompt,
                temperature=0.0,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                max_new_tokens=self.max_new_tokens,
            )
            result = parse_judge_response(raw)
        except Exception as exc:
            result = {
                "parsed_json": False,
                "support": 0.0,
                "causal": 0.0,
                "temporal": 0.0,
                "answer_link": 0.0,
                "hallucination": 1.0,
                "overall": 0.0,
                "reason": f"judge_error: {exc}",
                "raw_response": "",
            }
        self._cache[cache_key] = result
        self._save_cache()
        return result

