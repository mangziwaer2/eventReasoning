from __future__ import annotations

import re
from datetime import date
from datetime import datetime
from datetime import timedelta
from difflib import SequenceMatcher

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from forecast_trace_graph import graph_bridge_score
from forecast_trace_graph import graph_edge_ids
from forecast_trace_graph import graph_event_ids
from forecast_trace_graph import resolve_edge_ref
from forecast_trace_graph import resolve_event_ref
from forecast_trace_schema import is_generic_event
from forecast_trace_schema import parse_relative_time_score


@dataclass(slots=True)
class PipelineStep:
    name: str
    observation: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineTrajectory:
    sample_id: str
    steps: list[PipelineStep] = field(default_factory=list)
    final_reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        name: str,
        observation: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
        reward: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.steps.append(
            PipelineStep(
                name=name,
                observation=observation or {},
                action=action or {},
                reward=reward,
                metadata=metadata or {},
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "steps": [step.to_dict() for step in self.steps],
            "final_reward": self.final_reward,
            "metadata": self.metadata,
        }


class PipelinePolicy:
    """Interface for future RL over graph construction with fixed event inputs and forecaster."""

    name = "base"

    def select_coarse_threshold(self, default_threshold: float, observation: dict[str, Any]) -> float:
        return default_threshold

    def select_refinement_threshold(self, default_threshold: float, observation: dict[str, Any]) -> float:
        return default_threshold

    def compute_reward(self, prediction: dict[str, Any], gold: dict[str, Any], trajectory: PipelineTrajectory) -> float:
        return 0.0

    def compute_reward_breakdown(
        self,
        prediction: dict[str, Any],
        gold: dict[str, Any],
        trajectory: PipelineTrajectory,
    ) -> dict[str, float]:
        reward = self.compute_reward(prediction, gold, trajectory)
        return {"total": reward}


class NoOpPipelinePolicy(PipelinePolicy):
    name = "noop"


class MiraiCodeReward:
    """Simple terminal reward for MIRAI event-base-code prediction."""

    def __call__(self, prediction: dict[str, Any], gold: dict[str, Any]) -> float:
        gold_codes = {str(item).strip() for item in gold.get("answer_list", []) if str(item).strip()}
        predicted_codes = prediction.get("predicted_event_base_codes", [])
        if not isinstance(predicted_codes, list):
            predicted_codes = []
        if not predicted_codes:
            primary_code = str(prediction.get("predicted_event_base_code", "")).strip()
            alternatives = prediction.get("alternative_event_base_codes", [])
            predicted_codes = [primary_code] + (
                [str(item).strip() for item in alternatives if str(item).strip()]
                if isinstance(alternatives, list) else []
            )
        predicted_set = {str(code).strip() for code in predicted_codes if str(code).strip()}
        if not gold_codes or not predicted_set:
            return 0.0
        hits = len(predicted_set & gold_codes)
        precision = hits / len(predicted_set)
        recall = hits / len(gold_codes)
        return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


class MiraiCodeRewardPolicy(NoOpPipelinePolicy):
    name = "mirai_code_reward"

    def __init__(self) -> None:
        self.reward_fn = MiraiCodeReward()

    def compute_reward(self, prediction: dict[str, Any], gold: dict[str, Any], trajectory: PipelineTrajectory) -> float:
        return self.reward_fn(prediction, gold)

    def compute_reward_breakdown(
        self,
        prediction: dict[str, Any],
        gold: dict[str, Any],
        trajectory: PipelineTrajectory,
    ) -> dict[str, float]:
        answer = self.reward_fn(prediction, gold)
        return {"answer": answer, "total": answer}


def _latest_step(trajectory: PipelineTrajectory, name: str) -> PipelineStep | None:
    for step in reversed(trajectory.steps):
        if step.name == name:
            return step
    return None


def _valid_event_code_score(prediction: dict[str, Any]) -> float:
    final_answer = prediction.get("final_answer", {})
    if not isinstance(final_answer, dict):
        final_answer = {}
    event_code = str(final_answer.get("event_code", prediction.get("predicted_event_base_code", ""))).strip()
    return 1.0 if re.fullmatch(r"\d{3}", event_code) else 0.0


def _valid_answer_format_score(prediction: dict[str, Any]) -> float:
    answers = prediction.get("answers", [])
    if not isinstance(answers, list) or not answers:
        return 0.0
    codes = [
        str(item.get("event_code", "")).strip()
        for item in answers
        if isinstance(item, dict)
    ]
    valid = len(codes) == len(answers) and all(
        re.fullmatch(r"\d{3}", code) for code in codes)
    return 1.0 if valid else 0.0


def _support_refs(prediction: dict[str, Any]) -> tuple[list[str], list[str]]:
    event_refs: list[str] = []
    edge_refs: list[str] = []
    trace = prediction.get("forecast_trace", {})
    if isinstance(trace, dict):
        for item in trace.get("intermediate_events", []):
            if not isinstance(item, dict):
                continue
            event_refs.extend(str(ref).strip() for ref in item.get("supporting_event_ids", []) if str(ref).strip())
            edge_refs.extend(str(ref).strip() for ref in item.get("supporting_edge_ids", []) if str(ref).strip())
    final_answer = prediction.get("final_answer", {})
    if isinstance(final_answer, dict):
        event_refs.extend(str(ref).strip() for ref in final_answer.get("supporting_event_ids", []) if str(ref).strip())
        edge_refs.extend(str(ref).strip() for ref in final_answer.get("supporting_edge_ids", []) if str(ref).strip())
    event_refs.extend(str(ref).strip() for ref in prediction.get("support_event_ids", []) if str(ref).strip())
    return event_refs, edge_refs


def _ratio_valid_refs(refs: list[str], valid_ids: set[str], mapping: dict[str, str] | None, kind: str) -> float:
    if not refs:
        return 0.0
    valid = 0
    for ref in refs:
        resolved = resolve_event_ref(ref, mapping) if kind == "event" else resolve_edge_ref(ref, mapping)
        if resolved in valid_ids:
            valid += 1
    return valid / len(refs)


def _trace_temporal_score(prediction: dict[str, Any]) -> float:
    trace = prediction.get("forecast_trace", {})
    if not isinstance(trace, dict):
        return 0.0
    events = [item for item in trace.get("intermediate_events", []) if isinstance(item, dict)]
    if not events:
        return 0.0
    scores = [parse_relative_time_score(str(item.get("relative_time", ""))) for item in events]
    return sum(scores) / len(scores)


def _parse_calendar_time(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _graph_event_time_lookup(graph: Any, mapping: dict[str, str]) -> dict[str, date]:
    if not graph:
        return {}
    events = graph.get("events", []) if isinstance(graph, dict) else getattr(graph, "events", [])
    result: dict[str, date] = {}
    for event in events if isinstance(events, list) else []:
        if isinstance(event, dict):
            event_id = str(event.get("event_id", "")).strip()
            metadata = event.get("metadata", {})
            value = event.get("event_time") or event.get("date") or event.get("publish_time")
            if not value and isinstance(metadata, dict):
                value = metadata.get("event_time") or metadata.get("date") or metadata.get("publish_time")
        else:
            event_id = str(getattr(event, "event_id", "")).strip()
            metadata = getattr(event, "metadata", {})
            value = (metadata.get("event_time") or metadata.get("date") or metadata.get("publish_time")) if isinstance(metadata, dict) else None
        parsed = _parse_calendar_time(value)
        if event_id and parsed is not None:
            result[event_id] = parsed
    for ref, event_id in (mapping or {}).items():
        if event_id in result:
            result[str(ref)] = result[event_id]
    return result


def _trace_temporal_components(
    prediction: dict[str, Any],
    gold: dict[str, Any],
    trajectory: PipelineTrajectory,
    graph: Any,
    event_ref_to_id: dict[str, str],
) -> tuple[float, float, float]:
    """Return (overall, interval validity, ordering) for forecast trace times.

    Absolute event_time is authoritative. Relative t-k values are converted from
    the target answer date when that date is available, otherwise from the cutoff.
    Without calendar metadata this deliberately falls back to the legacy relative-time score.
    """
    trace = prediction.get("forecast_trace", {})
    events = [item for item in trace.get("intermediate_events", []) if isinstance(item, dict)] if isinstance(trace, dict) else []
    if not events:
        return 0.0, 0.0, 0.0
    query = trajectory.metadata.get("query", {}) if isinstance(trajectory.metadata, dict) else {}
    if not isinstance(query, dict):
        query = {}
    graph_query = graph.get("query", {}) if isinstance(graph, dict) else {}
    if not isinstance(graph_query, dict):
        graph_query = {}
    cutoff = _parse_calendar_time(
        query.get("observation_time") or query.get("cutoff_time") or query.get("date_str")
        or graph_query.get("observation_time") or graph_query.get("cutoff_time") or graph_query.get("date_str")
    )
    target = _parse_calendar_time(query.get("target_time") or query.get("target_date"))
    target_events = gold.get("target_events", []) if isinstance(gold, dict) else []
    predicted_codes = prediction.get("predicted_event_base_codes", []) if isinstance(prediction, dict) else []
    if not isinstance(predicted_codes, list):
        predicted_codes = []
    predicted_codes = {str(code).strip() for code in predicted_codes if str(code).strip()}
    candidate_target_dates: list[date] = []
    if isinstance(target_events, list):
        for item in target_events:
            if not isinstance(item, dict):
                continue
            code = str(item.get("event_code", "")).strip()
            parsed_date = _parse_calendar_time(item.get("date"))
            if parsed_date is not None and (not predicted_codes or code in predicted_codes):
                candidate_target_dates.append(parsed_date)
    if candidate_target_dates:
        target = min(candidate_target_dates)
    if target is None:
        target = _parse_calendar_time(gold.get("target_time") or gold.get("target_date")) if isinstance(gold, dict) else None
    support_times = _graph_event_time_lookup(graph, event_ref_to_id)
    observed_times = [value for key, value in support_times.items() if key not in event_ref_to_id]
    latest_observed = max(observed_times) if observed_times else None
    predicted_times: list[date | None] = []
    legacy_scores: list[float] = []
    future_scores: list[float] = []
    for item in events:
        relative = str(item.get("relative_time", "")).strip().lower().replace(" ", "")
        legacy = parse_relative_time_score(relative)
        legacy_scores.append(legacy)
        explicit_value = item.get("event_time") or item.get("date")
        has_explicit = bool(str(explicit_value or "").strip())
        explicit = _parse_calendar_time(explicit_value)
        predicted = explicit
        relative_anchor = target or cutoff
        if not has_explicit and predicted is None and relative_anchor is not None:
            match = re.fullmatch(r"t([+-])(\d+|0)", relative)
            if match:
                delta = int(match.group(2)) * (1 if match.group(1) == "+" else -1)
                predicted = relative_anchor + timedelta(days=delta)
        predicted_times.append(predicted)
        # A forecast step must be strictly after observation and before the answer date.
        if cutoff is None and predicted is None and not has_explicit:
            future_scores.append(legacy)
            continue
        valid = (explicit is not None) if has_explicit else (bool(re.fullmatch(r"t[+-](\d+|0)", relative)) or relative == "t")
        if predicted is not None and cutoff is not None:
            valid = valid and predicted > cutoff
        if predicted is not None and target is not None:
            valid = valid and predicted < target
        support_refs = _support_refs_for_trace_item(item)
        timed_supports = [support_times[ref] for ref in support_refs if ref in support_times]
        reference_times = timed_supports + ([latest_observed] if latest_observed is not None else [])
        if predicted is not None and reference_times:
            valid = valid and predicted > max(reference_times)
        future_scores.append(1.0 if valid else 0.0)
    if cutoff is None and all(item is None for item in predicted_times):
        return sum(legacy_scores) / len(legacy_scores), sum(legacy_scores) / len(legacy_scores), 1.0
    known_times = [item for item in predicted_times if item is not None]
    ordering = 1.0
    if len(known_times) >= 2:
        ordering = sum(1.0 for left, right in zip(known_times, known_times[1:]) if left < right) / (len(known_times) - 1)
    validity = sum(future_scores) / len(future_scores)
    # Ordering is only meaningful for events that are themselves inside the
    # observation-to-target interval; invalid timestamps must not receive a
    # positive temporal reward from a correctly ordered list.
    return validity * (0.7 + 0.3 * ordering), validity, ordering


def _support_refs_for_trace_item(item: dict[str, Any]) -> list[str]:
    refs = item.get("supporting_event_ids", item.get("support_event_ids", []))
    if not isinstance(refs, list):
        refs = [refs] if refs else []
    return [str(ref).strip() for ref in refs if str(ref).strip()]


VAGUE_EFFECT_PATTERNS = (
    "increased likelihood of this event",
    "increases likelihood of this event",
    "raises likelihood",
    "increases likelihood",
    "makes this more likely",
    "more likely to happen",
)
MECHANISM_TERMS = ("because", "caus", "lead", "trigger", "response", "force", "allow", "enable", "prevent", "constraint", "retaliat", "pressure", "commit", "mobil")


def _expected_effect_penalty(prediction: dict[str, Any]) -> float:
    trace = prediction.get("forecast_trace", {})
    events = [item for item in trace.get("intermediate_events", []) if isinstance(item, dict)] if isinstance(trace, dict) else []
    if not events:
        return 0.0
    penalties: list[float] = []
    for item in events:
        effect = " ".join(str(item.get("expected_effect", "")).lower().split())
        if not effect:
            penalties.append(1.0)
        elif any(pattern in effect for pattern in VAGUE_EFFECT_PATTERNS) and not any(term in effect for term in MECHANISM_TERMS):
            penalties.append(1.0)
        else:
            penalties.append(0.0)
    return sum(penalties) / len(penalties)


def _generic_penalty(prediction: dict[str, Any]) -> float:
    trace = prediction.get("forecast_trace", {})
    if not isinstance(trace, dict):
        return 1.0
    events = [item for item in trace.get("intermediate_events", []) if isinstance(item, dict)]
    if not events:
        return 1.0
    generic = sum(1 for item in events if is_generic_event(str(item.get("event", ""))))
    return generic / len(events)


def _copy_tokens(value: Any) -> list[str]:
    """Normalize an event mention enough to detect copied historical evidence."""
    if isinstance(value, dict):
        value = (
            value.get("mention")
            or value.get("event_context")
            or value.get("text")
            or value.get("normalized_text")
            or value.get("event")
            or ""
        )
    text = str(value or "").lower()
    text = re.sub(r"\b(?:trigger|mention|actors|participants)\s*=\s*", " ", text)
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def _historical_event_candidates(graph: Any) -> list[list[str]]:
    if isinstance(graph, dict):
        events = graph.get("events", [])
    else:
        events = getattr(graph, "events", [])
    candidates: list[list[str]] = []
    for event in events if isinstance(events, list) else []:
        values: list[Any] = []
        if isinstance(event, dict):
            values.extend(
                event.get(key, "")
                for key in ("text", "normalized_text", "mention", "event_context", "event_mention")
            )
            metadata = event.get("metadata", {})
            if isinstance(metadata, dict):
                values.extend(metadata.get(key, "") for key in ("event_context", "event_mention"))
        else:
            values.extend(getattr(event, key, "") for key in ("text", "normalized_text"))
            metadata = getattr(event, "metadata", {})
            if isinstance(metadata, dict):
                values.extend(metadata.get(key, "") for key in ("event_context", "event_mention"))
        for value in values:
            tokens = _copy_tokens(value)
            if tokens and tokens not in candidates:
                candidates.append(tokens)
    return candidates


def _historical_copy_penalty(prediction: dict[str, Any], graph: Any) -> float:
    """Return the mean strongest exact/near-verbatim copy match in the trace."""
    if graph is None:
        return 0.0
    trace = prediction.get("forecast_trace", {})
    if not isinstance(trace, dict):
        return 0.0
    historical = _historical_event_candidates(graph)
    if not historical:
        return 0.0
    scores: list[float] = []
    for item in trace.get("intermediate_events", []):
        if not isinstance(item, dict):
            continue
        tokens = _copy_tokens(item.get("event", ""))
        if not tokens:
            continue
        best = 0.0
        token_set = set(tokens)
        for candidate in historical:
            candidate_set = set(candidate)
            if tokens == candidate:
                best = 1.0
                break
            if len(token_set) < 4:
                continue
            if len(candidate_set) < 4:
                continue
            overlap = len(token_set & candidate_set) / min(len(token_set), len(candidate_set))
            length_ratio = min(len(tokens), len(candidate)) / max(len(tokens), len(candidate))
            sequence = SequenceMatcher(None, tokens, candidate).ratio()
            if overlap >= 0.8 and length_ratio >= 0.8:
                best = max(best, min(1.0, max(overlap, sequence)))
            elif sequence >= 0.92:
                best = max(best, sequence)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def _density_penalty(prediction: dict[str, Any], max_events: int = 3, max_edges: int = 5) -> float:
    trace = prediction.get("forecast_trace", {})
    if not isinstance(trace, dict):
        return 0.0
    event_count = len([item for item in trace.get("intermediate_events", []) if isinstance(item, dict)])
    edge_count = len([item for item in trace.get("trace_edges", []) if isinstance(item, dict)])
    overflow = max(0, event_count - max_events) + max(0, edge_count - max_edges)
    return min(float(overflow), 4.0) / 4.0


def _format_score(prediction: dict[str, Any]) -> float:
    parsed = 1.0 if prediction.get("parsed_json") else 0.0
    trace = prediction.get("forecast_trace", {})
    final_answer = prediction.get("final_answer", {})
    has_trace = 1.0 if isinstance(trace, dict) and isinstance(trace.get("intermediate_events"), list) and isinstance(trace.get("trace_edges"), list) else 0.0
    has_answer = 1.0 if isinstance(final_answer, dict) and str(final_answer.get("event_code", prediction.get("predicted_event_base_code", ""))).strip() else 0.0
    valid_event_code = _valid_event_code_score(prediction)
    return 0.25 * parsed + 0.25 * has_trace + 0.25 * has_answer + 0.25 * valid_event_code


class ForecastTraceReward:
    """Reward for structured trace plus MIRAI closed-set answer.

    This is intentionally deterministic and offline-friendly. It can be used as
    a reward model for RL rollouts, or as a validation metric before PPO/GRPO.
    """

    def __init__(
        self,
        answer_weight: float = 1.0,
        format_weight: float = 0.2,
        grounding_weight: float = 0.2,
        temporal_weight: float = 0.2,
        graph_bridge_weight: float = 0.3,
        generic_penalty_weight: float = 0.15,
        density_penalty_weight: float = 0.15,
        historical_copy_penalty_weight: float = 0.5,
        expected_effect_penalty_weight: float = 0.15,
        wrong_answer_trace_scale: float = 0.2,
    ) -> None:
        self.answer_reward = MiraiCodeReward()
        self.answer_weight = answer_weight
        self.format_weight = format_weight
        self.grounding_weight = grounding_weight
        self.temporal_weight = temporal_weight
        self.graph_bridge_weight = graph_bridge_weight
        self.generic_penalty_weight = generic_penalty_weight
        self.density_penalty_weight = density_penalty_weight
        self.historical_copy_penalty_weight = max(0.0, float(historical_copy_penalty_weight))
        self.expected_effect_penalty_weight = max(0.0, float(expected_effect_penalty_weight))
        self.wrong_answer_trace_scale = max(0.0, min(float(wrong_answer_trace_scale), 1.0))

    def __call__(self, prediction: dict[str, Any], gold: dict[str, Any], trajectory: PipelineTrajectory) -> dict[str, float]:
        forecast_step = _latest_step(trajectory, "forecast")
        metadata = forecast_step.metadata if forecast_step is not None else {}
        refined_graph = metadata.get("refined_graph") or trajectory.metadata.get("refined_graph")
        event_ref_to_id = metadata.get("event_ref_to_id") or trajectory.metadata.get("event_ref_to_id") or {}
        edge_ref_to_id = metadata.get("edge_ref_to_id") or trajectory.metadata.get("edge_ref_to_id") or {}

        answer = self.answer_reward(prediction, gold)
        fmt = _format_score(prediction)
        valid_answer_format = _valid_answer_format_score(prediction)

        if refined_graph is None:
            valid_event_ratio = 0.0
            valid_edge_ratio = 0.0
            bridge = 0.0
        else:
            event_refs, edge_refs = _support_refs(prediction)
            valid_event_ratio = _ratio_valid_refs(
                event_refs,
                graph_event_ids(refined_graph),
                event_ref_to_id if isinstance(event_ref_to_id, dict) else {},
                kind="event",
            )
            valid_edge_ratio = _ratio_valid_refs(
                edge_refs,
                graph_edge_ids(refined_graph),
                edge_ref_to_id if isinstance(edge_ref_to_id, dict) else {},
                kind="edge",
            )
            bridge = graph_bridge_score(
                refined_graph,
                prediction,
                event_ref_to_id=event_ref_to_id if isinstance(event_ref_to_id, dict) else {},
            )

        grounding = 0.65 * valid_event_ratio + 0.35 * valid_edge_ratio
        temporal, temporal_validity, temporal_order = _trace_temporal_components(
            prediction,
            gold,
            trajectory,
            refined_graph,
            event_ref_to_id if isinstance(event_ref_to_id, dict) else {},
        )
        generic_penalty = _generic_penalty(prediction)
        density_penalty = _density_penalty(prediction)
        historical_copy_penalty = _historical_copy_penalty(prediction, refined_graph)
        expected_effect_penalty = _expected_effect_penalty(prediction)
        trace_unscaled = (
            self.format_weight * fmt
            + self.grounding_weight * grounding
            + self.temporal_weight * temporal
            + self.graph_bridge_weight * bridge
            - self.generic_penalty_weight * generic_penalty
            - self.density_penalty_weight * density_penalty
            - self.historical_copy_penalty_weight * historical_copy_penalty
            - self.expected_effect_penalty_weight * expected_effect_penalty
        )
        trace_component = trace_unscaled
        if answer <= 0.0:
            # Keep answer reward dominant without flattening all incorrect rollouts.
            trace_component *= self.wrong_answer_trace_scale
        total = self.answer_weight * answer + trace_component
        return {
            "answer": round(answer, 6),
            "format": round(fmt, 6),
            "valid_answer_format": round(valid_answer_format, 6),
            "grounding": round(grounding, 6),
            "valid_event_ref_ratio": round(valid_event_ratio, 6),
            "valid_edge_ref_ratio": round(valid_edge_ratio, 6),
            "temporal": round(temporal, 6),
            "temporal_validity": round(temporal_validity, 6),
            "temporal_order": round(temporal_order, 6),
            "graph_bridge": round(bridge, 6),
            "generic_penalty": round(generic_penalty, 6),
            "density_penalty": round(density_penalty, 6),
            "historical_copy_penalty": round(historical_copy_penalty, 6),
            "expected_effect_penalty": round(expected_effect_penalty, 6),
            "trace_unscaled": round(trace_unscaled, 6),
            "trace": round(trace_component, 6),
            "total": round(total, 6),
        }


class ForecastTraceRewardPolicy(NoOpPipelinePolicy):
    name = "forecast_trace_reward"

    def __init__(self) -> None:
        self.reward_fn = ForecastTraceReward()

    def compute_reward(self, prediction: dict[str, Any], gold: dict[str, Any], trajectory: PipelineTrajectory) -> float:
        return self.reward_fn(prediction, gold, trajectory)["total"]

    def compute_reward_breakdown(
        self,
        prediction: dict[str, Any],
        gold: dict[str, Any],
        trajectory: PipelineTrajectory,
    ) -> dict[str, float]:
        return self.reward_fn(prediction, gold, trajectory)


def build_pipeline_policy(name: str = "noop") -> PipelinePolicy:
    normalized = name.strip().lower()
    if normalized in {"noop", "none", ""}:
        return NoOpPipelinePolicy()
    if normalized in {"mirai_code_reward", "mirai-code-reward"}:
        return MiraiCodeRewardPolicy()
    if normalized in {"forecast_trace_reward", "forecast-trace-reward", "trace_reward", "trace-reward"}:
        return ForecastTraceRewardPolicy()
    raise ValueError(f"Unsupported pipeline policy: {name}")
