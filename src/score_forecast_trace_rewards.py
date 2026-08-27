from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from path_utils import resolve_repo_path
from rl_pipeline_hooks import PipelineStep
from rl_pipeline_hooks import PipelineTrajectory
from rl_pipeline_hooks import build_pipeline_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute structured forecast-trace rewards from pipeline predictions.jsonl.")
    parser.add_argument("--input", required=True, help="Path to predictions.jsonl from evaluate_local_qwen_pipeline.py.")
    parser.add_argument("--output", default=None, help="Optional rescored JSONL path.")
    parser.add_argument("--metrics-output", default=None, help="Optional reward metrics JSON path.")
    parser.add_argument("--policy", default="forecast_trace_reward", help="Reward policy name.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_trajectory(row: dict[str, Any]) -> PipelineTrajectory:
    raw_trajectory = row.get("trajectory", {})
    trajectory = PipelineTrajectory(
        sample_id=str(raw_trajectory.get("sample_id", row.get("query_id", ""))),
        final_reward=raw_trajectory.get("final_reward"),
        metadata=dict(raw_trajectory.get("metadata", {})) if isinstance(raw_trajectory.get("metadata", {}), dict) else {},
    )
    steps = raw_trajectory.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            trajectory.steps.append(
                PipelineStep(
                    name=str(step.get("name", "")),
                    observation=dict(step.get("observation", {})) if isinstance(step.get("observation", {}), dict) else {},
                    action=dict(step.get("action", {})) if isinstance(step.get("action", {}), dict) else {},
                    reward=step.get("reward"),
                    metadata=dict(step.get("metadata", {})) if isinstance(step.get("metadata", {}), dict) else {},
                )
            )
    return trajectory


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    rows = load_jsonl(input_path)
    policy = build_pipeline_policy(args.policy)
    output_rows: list[dict[str, Any]] = []
    reward_keys: set[str] = set()

    for row in rows:
        prediction = row.get("forecast_prediction", {})
        gold = row.get("mirai_query", {})
        trajectory = build_trajectory(row)
        breakdown = policy.compute_reward_breakdown(prediction, gold, trajectory)
        reward_keys.update(breakdown)
        output_rows.append(
            {
                **row,
                "reward": float(breakdown.get("total", row.get("reward", 0.0))),
                "reward_breakdown": breakdown,
            }
        )

    metrics = {
        "input": str(input_path),
        "samples": len(output_rows),
        "policy": policy.name,
        "average_reward": average([float(row["reward"]) for row in output_rows]),
        "average_reward_breakdown": {
            key: average([float(row.get("reward_breakdown", {}).get(key, 0.0)) for row in output_rows])
            for key in sorted(reward_keys)
        },
    }

    output_path = resolve_repo_path(args.output) if args.output else input_path.with_name(input_path.stem + ".rescored.jsonl")
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_path = (
        resolve_repo_path(args.metrics_output)
        if args.metrics_output
        else output_path.with_name(output_path.stem + ".metrics.json")
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        " | ".join(
            [
                f"rescored={len(output_rows)}",
                f"average_reward={metrics['average_reward']:.4f}",
                f"output={output_path}",
                f"metrics={metrics_path}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
