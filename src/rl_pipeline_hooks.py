from __future__ import annotations

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
        predicted_codes = []
        primary_code = str(prediction.get("predicted_event_base_code", "")).strip()
        if primary_code:
            predicted_codes.append(primary_code)
        alternatives = prediction.get("alternative_event_base_codes", [])
        if isinstance(alternatives, list):
            predicted_codes.extend(str(item).strip() for item in alternatives if str(item).strip())
        if not gold_codes:
            return 0.0
        if primary_code in gold_codes:
            return 1.0
        if any(code in gold_codes for code in predicted_codes):
            return 0.5
        return 0.0


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


def _valid_choice_score(prediction: dict[str, Any], choices: list[dict[str, Any]]) -> float:
    if not choices:
        return 1.0
    final_answer = prediction.get("final_answer", {})
    if not isinstance(final_answer, dict):
        final_answer = {}
    allowed_codes = {str(choice.get("event_code", "")).strip() for choice in choices if str(choice.get("event_code", "")).strip()}
    allowed_choice_ids = {str(choice.get("choice_id", "")).strip() for choice in choices if str(choice.get("choice_id", "")).strip()}
    event_code = str(final_answer.get("event_code", prediction.get("predicted_event_base_code", ""))).strip()
    choice_id = str(final_answer.get("choice_id", "")).strip()
    code_ok = bool(event_code and event_code in allowed_codes)
    choice_ok = bool(choice_id and choice_id in allowed_choice_ids)
    return 1.0 if code_ok or choice_ok else 0.0


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


def _generic_penalty(prediction: dict[str, Any]) -> float:
    trace = prediction.get("forecast_trace", {})
    if not isinstance(trace, dict):
        return 1.0
    events = [item for item in trace.get("intermediate_events", []) if isinstance(item, dict)]
    if not events:
        return 1.0
    generic = sum(1 for item in events if is_generic_event(str(item.get("event", ""))))
    return generic / len(events)


def _density_penalty(prediction: dict[str, Any], max_events: int = 3, max_edges: int = 5) -> float:
    trace = prediction.get("forecast_trace", {})
    if not isinstance(trace, dict):
        return 0.0
    event_count = len([item for item in trace.get("intermediate_events", []) if isinstance(item, dict)])
    edge_count = len([item for item in trace.get("trace_edges", []) if isinstance(item, dict)])
    overflow = max(0, event_count - max_events) + max(0, edge_count - max_edges)
    return min(float(overflow), 4.0) / 4.0


def _format_score(prediction: dict[str, Any], choices: list[dict[str, Any]]) -> float:
    parsed = 1.0 if prediction.get("parsed_json") else 0.0
    trace = prediction.get("forecast_trace", {})
    final_answer = prediction.get("final_answer", {})
    has_trace = 1.0 if isinstance(trace, dict) and isinstance(trace.get("intermediate_events"), list) and isinstance(trace.get("trace_edges"), list) else 0.0
    has_answer = 1.0 if isinstance(final_answer, dict) and str(final_answer.get("event_code", prediction.get("predicted_event_base_code", ""))).strip() else 0.0
    valid_choice = _valid_choice_score(prediction, choices)
    return 0.25 * parsed + 0.25 * has_trace + 0.25 * has_answer + 0.25 * valid_choice


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
        self.wrong_answer_trace_scale = max(0.0, min(float(wrong_answer_trace_scale), 1.0))

    def __call__(self, prediction: dict[str, Any], gold: dict[str, Any], trajectory: PipelineTrajectory) -> dict[str, float]:
        forecast_step = _latest_step(trajectory, "forecast")
        metadata = forecast_step.metadata if forecast_step is not None else {}
        refined_graph = metadata.get("refined_graph") or trajectory.metadata.get("refined_graph")
        choices = metadata.get("choices") or trajectory.metadata.get("choices") or []
        event_ref_to_id = metadata.get("event_ref_to_id") or trajectory.metadata.get("event_ref_to_id") or {}
        edge_ref_to_id = metadata.get("edge_ref_to_id") or trajectory.metadata.get("edge_ref_to_id") or {}

        answer = self.answer_reward(prediction, gold)
        fmt = _format_score(prediction, choices if isinstance(choices, list) else [])

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
        temporal = _trace_temporal_score(prediction)
        generic_penalty = _generic_penalty(prediction)
        density_penalty = _density_penalty(prediction)
        trace_unscaled = (
            self.format_weight * fmt
            + self.grounding_weight * grounding
            + self.temporal_weight * temporal
            + self.graph_bridge_weight * bridge
            - self.generic_penalty_weight * generic_penalty
            - self.density_penalty_weight * density_penalty
        )
        trace_component = trace_unscaled
        if answer <= 0.0:
            # Keep answer reward dominant without flattening all incorrect rollouts.
            trace_component *= self.wrong_answer_trace_scale
        total = self.answer_weight * answer + trace_component
        return {
            "answer": round(answer, 6),
            "format": round(fmt, 6),
            "grounding": round(grounding, 6),
            "valid_event_ref_ratio": round(valid_event_ratio, 6),
            "valid_edge_ref_ratio": round(valid_edge_ratio, 6),
            "temporal": round(temporal, 6),
            "graph_bridge": round(bridge, 6),
            "generic_penalty": round(generic_penalty, 6),
            "density_penalty": round(density_penalty, 6),
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
