from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_refinement import load_model
from evaluate_refinement import metric_counts
from evaluate_refinement import predict_probabilities
from evaluate_refinement import predict_relation_labels
from evaluate_refinement import relation_report
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path
from refinement_dataset import load_cached_refinement_samples
from train_refinement import split_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate refinement on the exact deterministic training subset split.")
    parser.add_argument("--cache", default=str(REPO_ROOT / "outputs" / "maven_qwen_refinement_cache_v3" / "samples.jsonl"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Must match train_refinement --limit; 0 means all cache samples.")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    samples = load_cached_refinement_samples(
        resolve_repo_path(args.cache),
        limit=args.limit if args.limit > 0 else None,
        require_complete_maven_cache=True,
    )
    _, validation_samples = split_samples(samples, args.validation_ratio, args.seed)
    if not validation_samples:
        raise RuntimeError("No validation samples; increase cache size or validation ratio.")
    model = load_model(resolve_repo_path(args.model_path), torch)
    labels, probabilities, graph_candidate_counts = predict_probabilities(model, validation_samples, torch)
    relation_gold, relation_predicted = predict_relation_labels(model, validation_samples, torch)
    baseline = metric_counts(labels, [True] * len(labels))
    rows: list[dict[str, Any]] = []
    for threshold in sorted(set(args.thresholds)):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("All --thresholds values must be in [0, 1].")
        rows.append({"threshold": threshold, **metric_counts(labels, [score >= threshold for score in probabilities])})
    best = max(rows, key=lambda row: (float(row["f1"]), float(row["precision"]), -float(row["threshold"])))
    result = {
        "evaluation_contract": "candidate-edge filtering only; held-out samples use train_refinement's deterministic split",
        "cache_limit": args.limit,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "validation_samples": len(validation_samples),
        "validation_candidate_edges": len(labels),
        "mean_candidates_per_graph": round(sum(graph_candidate_counts) / max(1, len(graph_candidate_counts)), 4),
        "baseline_keep_all": baseline,
        "relation_report": relation_report(relation_gold, relation_predicted),
        "threshold_results": rows,
        "best_f1": best,
        "improvement_over_keep_all": {
            "precision": round(float(best["precision"]) - float(baseline["precision"]), 6),
            "recall": round(float(best["recall"]) - float(baseline["recall"]), 6),
            "f1": round(float(best["f1"]) - float(baseline["f1"]), 6),
            "edges_removed": int(baseline["kept_edges"]) - int(best["kept_edges"]),
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve_repo_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
