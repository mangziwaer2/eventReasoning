from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from path_utils import REPO_ROOT
from path_utils import resolve_repo_path
from refinement_dataset import EDGE_FEATURE_DIM
from refinement_dataset import ID_TO_RELATION_LABEL
from refinement_dataset import RefinementSample
from refinement_dataset import load_cached_refinement_samples
from train_refinement import split_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a refinement checkpoint against held-out frozen-Qwen coarse edges."
    )
    parser.add_argument(
        "--cache",
        default=str(REPO_ROOT / "outputs" / "maven_qwen_refinement_cache_v3" / "samples.jsonl"),
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def metric_counts(labels: list[int], predictions: list[bool]) -> dict[str, float | int]:
    tp = sum(1 for label, predicted in zip(labels, predictions) if label and predicted)
    fp = sum(1 for label, predicted in zip(labels, predictions) if not label and predicted)
    fn = sum(1 for label, predicted in zip(labels, predictions) if label and not predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "kept_edges": tp + fp,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def load_model(model_path: Path, torch):
    from refinement_model import TemporalRelationalEdgeRefiner

    config_path = model_path.parent / "train_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    model = TemporalRelationalEdgeRefiner(
        edge_dim=int(config.get("edge_feature_dim", EDGE_FEATURE_DIM)),
        hidden_dim=int(config.get("hidden_dim", 192)),
        num_message_passing_steps=int(config.get("message_steps", 4)),
        dropout=0.0,
    )
    payload = torch.load(model_path, map_location="cpu")
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()
    return model


def predict_probabilities(model, samples: list[RefinementSample], torch) -> tuple[list[int], list[float], list[int]]:
    labels: list[int] = []
    probabilities: list[float] = []
    graph_candidate_counts: list[int] = []
    with torch.no_grad():
        for sample in samples:
            outputs = model(
                node_features=torch.tensor(sample.node_features, dtype=torch.float32),
                edge_index=torch.tensor(sample.edge_index, dtype=torch.long),
                edge_features=torch.tensor(sample.edge_features, dtype=torch.float32),
            )
            probs = torch.sigmoid(outputs["edge_keep_logits"]).tolist()
            labels.extend(int(value) for value in sample.edge_labels)
            probabilities.extend(float(value) for value in probs)
            graph_candidate_counts.append(len(sample.edge_labels))
    return labels, probabilities, graph_candidate_counts


def predict_relation_labels(model, samples: list[RefinementSample], torch) -> tuple[list[int], list[int]]:
    """Return gold/predicted relation ids for every cached candidate edge."""
    gold: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for sample in samples:
            outputs = model(
                node_features=torch.tensor(sample.node_features, dtype=torch.float32),
                edge_index=torch.tensor(sample.edge_index, dtype=torch.long),
                edge_features=torch.tensor(sample.edge_features, dtype=torch.float32),
            )
            relation_ids = outputs["edge_relation_logits"].argmax(dim=-1).tolist()
            labels = sample.edge_relation_labels
            if len(labels) != len(relation_ids):
                labels = [0] * len(relation_ids)
            for keep, actual, guess in zip(sample.edge_labels, labels, relation_ids):
                if keep > 0.5:
                    gold.append(int(actual))
                    predicted.append(int(guess))
    return gold, predicted


def relation_report(gold: list[int], predicted: list[int]) -> dict[str, Any]:
    classes = sorted(set(gold) | set(predicted))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id in classes:
        tp = sum(1 for actual, guess in zip(gold, predicted) if actual == class_id and guess == class_id)
        fp = sum(1 for actual, guess in zip(gold, predicted) if actual != class_id and guess == class_id)
        fn = sum(1 for actual, guess in zip(gold, predicted) if actual == class_id and guess != class_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[ID_TO_RELATION_LABEL.get(class_id, str(class_id))] = {
            "id": class_id,
            "support": tp + fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {
        "support": len(gold),
        "accuracy": round(sum(actual == guess for actual, guess in zip(gold, predicted)) / max(1, len(gold)), 6),
        "macro_f1": round(sum(f1_values) / max(1, len(f1_values)), 6),
        "per_class": per_class,
    }


def main() -> None:
    import torch

    args = parse_args()
    samples = load_cached_refinement_samples(
        resolve_repo_path(args.cache),
        require_complete_maven_cache=True,
    )
    _, validation_samples = split_samples(samples, args.validation_ratio, args.seed)
    if not validation_samples:
        raise RuntimeError("No validation samples; increase cache size or validation ratio.")
    model = load_model(resolve_repo_path(args.model_path), torch)
    labels, probabilities, graph_candidate_counts = predict_probabilities(model, validation_samples, torch)
    relation_gold, relation_predicted = predict_relation_labels(model, validation_samples, torch)
    baseline = metric_counts(labels, [True] * len(labels))
    rows = []
    for threshold in sorted(set(args.thresholds)):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("All --thresholds values must be in [0, 1].")
        row = {"threshold": threshold, **metric_counts(labels, [value >= threshold for value in probabilities])}
        rows.append(row)
    best = max(rows, key=lambda row: (float(row["f1"]), float(row["precision"]), -float(row["threshold"])))
    result: dict[str, Any] = {
        "evaluation_contract": "candidate-edge filtering only; recall is relative to Qwen coarse candidates, not all gold graph edges",
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
