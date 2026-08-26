from __future__ import annotations

import argparse
import json
import statistics

from forecast_trace_judge_runtime_v3 import RolloutAwareFrozenQwenJudge
from path_utils import resolve_repo_path
from run_trace_judge import _graph_from_row
from run_trace_judge import _load_rows
from run_trace_judge import _prediction_from_row
from run_trace_judge import _query_from_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a rollout-aware frozen Qwen judge on forecast traces.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--cache-path", default=None)
    args = parser.parse_args()
    input_path = resolve_repo_path(args.input)
    rows = _load_rows(input_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    judge = RolloutAwareFrozenQwenJudge(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        thinking=args.thinking,
        cache_path=args.cache_path,
        max_context_chars=args.max_context_chars,
    )
    output_rows = []
    scores = []
    errors = 0
    for row in rows:
        result = judge.score(
            _prediction_from_row(row),
            query=_query_from_row(row),
            graph=_graph_from_row(row),
            context_prompt=str(row.get("forecast_prompt", row.get("prompt", ""))),
        )
        errors += int(not result.get("parsed_json", False))
        scores.append(float(result.get("overall", 0.0)))
        output_rows.append({**row, "judge": result})
    output_path = resolve_repo_path(args.output)
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
