from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from forecast_trace_judge import FrozenQwenTraceJudge
from forecast_trace_judge import parse_judge_response
from forecast_trace_schema import parse_structured_forecast
from path_utils import resolve_repo_path


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _graph_from_row(row: dict[str, Any]) -> Any:
    trajectory = row.get("trajectory", {})
    if isinstance(trajectory, dict):
        metadata = trajectory.get("metadata", {})
        if isinstance(metadata, dict) and isinstance(metadata.get("refined_graph"), dict):
            return metadata["refined_graph"]
        for step in trajectory.get("steps", []):
            if isinstance(step, dict):
                value = step.get("metadata", {})
                if isinstance(value, dict) and isinstance(value.get("refined_graph"), dict):
                    return value["refined_graph"]
    return None


def _query_from_row(row: dict[str, Any]) -> Any:
    trajectory = row.get("trajectory", {})
    if isinstance(trajectory, dict):
        metadata = trajectory.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("query") is not None:
            return metadata["query"]
    return row.get("mirai_query", row.get("gold", {}))


def _prediction_from_row(row: dict[str, Any]) -> dict[str, Any]:
    prediction = row.get("forecast_prediction")
    if isinstance(prediction, dict):
        return prediction
    prediction = row.get("parsed_prediction")
    if isinstance(prediction, dict):
        return prediction
    raw = row.get("raw_response", row.get("completion", ""))
    return parse_structured_forecast(str(raw))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score forecast traces with a frozen local Qwen judge.")
    parser.add_argument("--input", required=True, help="Pipeline predictions.jsonl or GRPO rollout_samples.jsonl.")
    parser.add_argument("--output", required=True, help="Output JSONL with judge scores.")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--thinking", action="store_true", help="Allow hidden Qwen reasoning for the judge.")
    parser.add_argument("--cache-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output)
    rows = _load_rows(input_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    judge = FrozenQwenTraceJudge(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        thinking=args.thinking,
        cache_path=args.cache_path,
        max_context_chars=args.max_context_chars,
    )
    output_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    errors = 0
    for row in rows:
        prediction = _prediction_from_row(row)
        result = judge.score(
            prediction,
            query=_query_from_row(row),
            graph=_graph_from_row(row),
            context_prompt=str(row.get("forecast_prompt", row.get("prompt", ""))),
        )
        errors += int(not result.get("parsed_json", False))
        scores.append(float(result.get("overall", 0.0)))
        output_rows.append({**row, "judge": result})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    metrics = {
        "input": str(input_path),
        "output": str(output_path),
        "samples": len(output_rows),
        "judge_parse_rate": (len(output_rows) - errors) / len(output_rows) if output_rows else 0.0,
        "judge_error_count": errors,
        "judge_overall_mean": statistics.mean(scores) if scores else 0.0,
        "judge_overall_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
    }
    metrics_path = resolve_repo_path(args.metrics_output) if args.metrics_output else output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
