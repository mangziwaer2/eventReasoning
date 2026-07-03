from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any


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
    """Interface for future reinforcement learning over the full local pipeline."""

    name = "base"

    def build_event_extraction_prompt(self, default_prompt: str, observation: dict[str, Any]) -> str:
        return default_prompt

    def select_event_limit(self, default_limit: int, observation: dict[str, Any]) -> int:
        return default_limit

    def select_coarse_threshold(self, default_threshold: float, observation: dict[str, Any]) -> float:
        return default_threshold

    def select_refinement_threshold(self, default_threshold: float, observation: dict[str, Any]) -> float:
        return default_threshold

    def build_forecast_prompt(self, default_prompt: str, observation: dict[str, Any]) -> str:
        return default_prompt

    def compute_reward(self, prediction: dict[str, Any], gold: dict[str, Any], trajectory: PipelineTrajectory) -> float:
        return 0.0


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


def build_pipeline_policy(name: str = "noop") -> PipelinePolicy:
    normalized = name.strip().lower()
    if normalized in {"noop", "none", ""}:
        return NoOpPipelinePolicy()
    if normalized in {"mirai_code_reward", "mirai-code-reward"}:
        return MiraiCodeRewardPolicy()
    raise ValueError(f"Unsupported pipeline policy: {name}")
