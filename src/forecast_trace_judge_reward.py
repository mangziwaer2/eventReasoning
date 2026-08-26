from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from forecast_trace_grpo_rewards import (
    ForecastTraceGRPOReward,
    _as_batch,
    _value_at,
    build_grpo_context,
    completion_to_text,
    value_to_log_text,
)
from forecast_trace_judge_runtime import FrozenQwenTraceJudge
from forecast_trace_schema import parse_structured_forecast
from mirai_dataset import load_mirai_event_code_choices
from path_utils import REPO_ROOT, resolve_repo_path


def _answer_items(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    items = prediction.get("answers", [])
    if isinstance(items, list):
        normalized = [item for item in items if isinstance(item, dict)]
        if normalized:
            return normalized[:3]
    final_answer = prediction.get("final_answer", {})
    return [final_answer] if isinstance(final_answer, dict) else []


class JudgeGRPOReward:
    """Deterministic forecast reward plus frozen-Qwen trace/semantic judges."""

    def __init__(
        self,
        judge_model_path: str,
        *,
        policy_name: str = "forecast_trace_reward",
        reward_key: str = "total",
        error_reward: float = -0.25,
        judge_weight: float = 0.2,
        description_weight: float = 0.05,
        judge_max_new_tokens: int = 384,
        description_max_new_tokens: int = 96,
        judge_thinking: bool = False,
        judge_cache_path: str | Path | None = None,
        codebook_dataset_path: str | Path | None = None,
        audit_path: str | Path | None = None,
        audit_every: int = 1,
        sample_audit_path: str | Path | None = None,
        sample_audit_every: int = 1,
        sample_audit_limit: int = 2,
    ) -> None:
        self.base = ForecastTraceGRPOReward(policy_name=policy_name, reward_key=reward_key, error_reward=error_reward)
        self.judge = FrozenQwenTraceJudge(
            judge_model_path,
            max_new_tokens=judge_max_new_tokens,
            description_max_new_tokens=description_max_new_tokens,
            thinking=judge_thinking,
            cache_path=judge_cache_path,
        )
        self.error_reward = float(error_reward)
        self.judge_weight = max(0.0, float(judge_weight))
        self.description_weight = max(0.0, float(description_weight))
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit_every = max(1, int(audit_every))
        self.sample_audit_path = Path(sample_audit_path) if sample_audit_path else None
        self.sample_audit_every = max(1, int(sample_audit_every))
        self.sample_audit_limit = max(0, int(sample_audit_limit))
        self.codebook_dataset_path = resolve_repo_path(
            str(codebook_dataset_path or (REPO_ROOT / "datasets" / "MIRAI_data.zip"))
        )
        try:
            choices = load_mirai_event_code_choices(self.codebook_dataset_path)
        except (OSError, KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot load MIRAI codebook for semantic reward: {self.codebook_dataset_path}") from exc
        self.codebook = {
            str(item.get("event_code", "")).strip(): str(item.get("description", "")).strip()
            for item in choices
            if str(item.get("event_code", "")).strip() and str(item.get("description", "")).strip()
        }
        if not self.codebook:
            raise RuntimeError(f"MIRAI codebook is empty: {self.codebook_dataset_path}")
        self.call_count = 0
        self.last_breakdowns: list[dict[str, float]] = []

    def _score_descriptions(self, prediction: dict[str, Any]) -> tuple[float, float, list[dict[str, Any]]]:
        if self.description_weight <= 0.0:
            return 0.0, 0.0, []
        scores: list[float] = []
        parsed: list[float] = []
        audit: list[dict[str, Any]] = []
        for item in _answer_items(prediction):
            code = str(item.get("event_code", item.get("predicted_event_base_code", ""))).strip()
            generated = str(item.get("event_description", item.get("description", item.get("event", "")))).strip()
            canonical = self.codebook.get(code, "")
            if not code or not generated or not canonical:
                continue
            result = self.judge.score_description(code, canonical, generated)
            scores.append(float(result.get("match", 0.0)))
            parsed.append(float(bool(result.get("parsed_json", False))))
            audit.append({
                "code": code,
                "canonical_description": canonical,
                "generated_description": generated,
                **result,
            })
        if not scores:
            return 0.0, 0.0, audit
        return sum(scores) / len(scores), sum(parsed) / len(parsed), audit

    def __call__(self, prompts: Any = None, completions: Any = None, **kwargs: Any) -> list[float]:
        if completions is None:
            completions = kwargs.get("completion", kwargs.get("response", []))
        batch = _as_batch(completions)
        rewards: list[float] = []
        breakdowns: list[dict[str, float]] = []
        audit_rows: list[dict[str, Any]] = []

        for index, completion in enumerate(batch):
            batch_size = len(batch)
            prompt_value = _value_at(prompts, index, batch_size) if prompts is not None else ""
            completion_text = completion_to_text(completion)
            context = None
            prediction = parse_structured_forecast(completion_text)
            judge: dict[str, Any] = {}
            description_judge: list[dict[str, Any]] = []
            try:
                context = build_grpo_context(kwargs, index, batch_size)
                prediction = parse_structured_forecast(completion_text, choices=context.choices)
                base = self.base.policy.compute_reward_breakdown(prediction, context.gold, context.trajectory)
                judge = self.judge.score(
                    prediction,
                    query=context.trajectory.metadata.get("query", context.gold),
                    graph=context.trajectory.metadata.get("refined_graph"),
                    context_prompt=completion_to_text(prompt_value),
                )
                description_score, description_parsed, description_judge = self._score_descriptions(prediction)
                answer = float(base.get("answer", 0.0))
                gate = 1.0 if answer > 0.0 else 0.2
                judge_score = float(judge.get("overall", 0.0))
                description_reward = self.description_weight * gate * description_score
                total = answer + gate * float(base.get("trace", 0.0)) + self.judge_weight * gate * judge_score + description_reward
                if not math.isfinite(total):
                    raise ValueError("non-finite reward")
                breakdown = {
                    **base,
                    "judge_support": float(judge.get("support", 0.0)),
                    "judge_causal": float(judge.get("causal", 0.0)),
                    "judge_temporal": float(judge.get("temporal", 0.0)),
                    "judge_answer_link": float(judge.get("answer_link", 0.0)),
                    "judge_hallucination": float(judge.get("hallucination", 0.0)),
                    "judge_overall": judge_score,
                    "judge_parsed": float(bool(judge.get("parsed_json", False))),
                    "judge_partial": float(bool(judge.get("partial_json", False))),
                    "judge_gate": gate,
                    "description_match": description_score,
                    "description_judge_parsed": description_parsed,
                    "description_gate": gate,
                    "description_reward": description_reward,
                    "description_weight": self.description_weight,
                    "total": total,
                }
            except Exception as exc:
                judge = {"parsed_json": False, "reason": f"reward_error: {exc}"}
                total = self.error_reward
                breakdown = {"total": total, "error_reward": 1.0, "judge_parsed": 0.0, "description_judge_parsed": 0.0}
            rewards.append(float(total))
            breakdowns.append(breakdown)
            audit_rows.append(
                {
                    "query_id": context.trajectory.sample_id if context is not None else "",
                    "prompt": value_to_log_text(prompt_value),
                    "completion": completion_text,
                    "parsed_prediction": prediction,
                    "gold": context.gold if context is not None else {},
                    "reward": total,
                    "reward_breakdown": breakdown,
                    "judge": judge,
                    "description_judge": description_judge,
                }
            )

        self.last_breakdowns = breakdowns
        self.call_count += 1
        self._write_audits(rewards, breakdowns, audit_rows)
        return rewards

    def _write_audits(self, rewards: list[float], breakdowns: list[dict[str, float]], rows: list[dict[str, Any]]) -> None:
        if self.audit_path is not None and self.call_count % self.audit_every == 0:
            keys = sorted({key for item in breakdowns for key in item})
            means = {
                key: round(sum(float(item.get(key, 0.0)) for item in breakdowns) / max(1, len(breakdowns)), 6)
                for key in keys
            }
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "reward_call": self.call_count,
                    "batch_size": len(rewards),
                    "reward_mean": round(sum(rewards) / max(1, len(rewards)), 6),
                    "reward_min": round(min(rewards), 6) if rewards else 0.0,
                    "reward_max": round(max(rewards), 6) if rewards else 0.0,
                    "breakdown_mean": means,
                    "judge_parse_rate": means.get("judge_parsed", 0.0),
                    "judge_partial_rate": means.get("judge_partial", 0.0),
                    "description_match": means.get("description_match", 0.0),
                    "description_judge_parse_rate": means.get("description_judge_parsed", 0.0),
                    "error_count": sum(1 for item in breakdowns if item.get("error_reward")),
                }, ensure_ascii=False) + "\n")
        if self.sample_audit_path is not None and self.sample_audit_limit > 0 and self.call_count % self.sample_audit_every == 0:
            self.sample_audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.sample_audit_path.open("a", encoding="utf-8") as handle:
                for index, row in enumerate(rows[: self.sample_audit_limit]):
                    handle.write(json.dumps({"reward_call": self.call_count, "batch_index": index, **row}, ensure_ascii=False, default=str) + "\n")
