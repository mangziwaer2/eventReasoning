from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from causal_graph import CoarseCausalGraph
from coarse_graph_dataset import PAIR_RELATION_TYPES
from coarse_graph_dataset import RELATION_PRIORITY
from coarse_graph_dataset import build_event_pair_inference_samples
from coarse_graph_dataset import build_graph_from_pair_predictions
from coarse_graph_dataset import load_maven_document_graph_samples
from coarse_graph_dataset import parse_pair_payload
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_trained_qwen_lora
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path
from refinement_dataset import EDGE_FEATURE_DIM
from refinement_dataset import ID_TO_RELATION
from refinement_dataset import load_refinement_sample_from_coarse_graph
from run_refinement import build_refined_graph
from run_refinement import load_model_config
from run_refinement import load_model_state


LABELS = tuple(PAIR_RELATION_TYPES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MAVEN coarse Qwen generation and refinement metrics end to end.",
    )
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="test", help="MAVEN split name: test, valid, train, etc.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum MAVEN rows. Use 0 for full split.")
    parser.add_argument("--max-events", type=int, default=16, help="Maximum events kept per document graph.")

    parser.add_argument("--base-model-path", default=str(REPO_ROOT / "models" / "Qwen2.5-0.5B"), help="Base Qwen model directory.")
    parser.add_argument("--coarse-adapter-path", default=str(REPO_ROOT / "outputs" / "coarse_graph_qwen_lora_4090_full" / "best_adapter"), help="Trained coarse Qwen LoRA adapter directory.")
    parser.add_argument("--max-sentence-gap", type=int, default=3, help="Maximum same-document sentence gap for coarse candidate pairs.")
    parser.add_argument("--max-pairs", type=int, default=64, help="Maximum coarse candidate pairs per document. Use 0 for all generated candidates.")
    parser.add_argument("--coarse-keep-threshold", type=float, default=0.5, help="Minimum coarse relation score to keep an edge.")
    parser.add_argument("--coarse-batch-size", type=int, default=8, help="Batch size for Qwen generation during evaluation.")
    parser.add_argument("--coarse-max-length", type=int, default=1024, help="Maximum coarse prompt length.")
    parser.add_argument("--coarse-max-new-tokens", type=int, default=48, help="Maximum generated tokens for coarse JSON.")
    parser.add_argument("--include-query", action="store_true", help="Include query title in coarse Qwen prompts.")
    parser.add_argument("--document-mode", choices=["none", "title", "snippet", "summary", "full"], default="title", help="Document context used in coarse Qwen prompts.")
    parser.add_argument("--max-document-chars", type=int, default=240, help="Maximum document snippet chars when document-mode=snippet.")

    parser.add_argument("--refinement-model-path", default=str(REPO_ROOT / "outputs" / "refinement_graph_4090_full" / "refinement_model.pt"), help="Trained refinement model path.")
    parser.add_argument("--refinement-keep-threshold", type=float, default=0.5, help="Minimum refinement keep probability for an edge.")
    parser.add_argument("--include-completion-candidates", dest="include_completion_candidates", action="store_true", default=True, help="Add heuristic completion candidates for refinement.")
    parser.add_argument("--no-completion-candidates", dest="include_completion_candidates", action="store_false", help="Only refine Qwen coarse edges.")
    parser.add_argument("--max-completion-edges", type=int, default=0, help="Maximum refinement completion candidates. Use 0 for no cap.")

    parser.add_argument("--output", default=str(REPO_ROOT / "outputs" / "maven_pipeline_eval" / "metrics.json"), help="Path for aggregate metrics JSON.")
    parser.add_argument("--per-sample-output", default=None, help="Optional JSONL path for per-sample metrics.")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N samples. Use 0 to disable.")
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def set_report(predicted: set[tuple[Any, ...]], gold: set[tuple[Any, ...]]) -> dict[str, float | int]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    report = prf(tp, fp, fn)
    report["predicted"] = len(predicted)
    report["gold"] = len(gold)
    return report


def normalize_label(label: str | None) -> str:
    normalized = str(label or "none").strip().lower()
    return normalized if normalized in LABELS else "none"


def classification_report(gold_labels: list[str], pred_labels: list[str]) -> dict[str, Any]:
    total = len(gold_labels)
    correct = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred)
    per_label: dict[str, Any] = {}
    observed_labels: list[str] = []
    for label in LABELS:
        tp = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred != label)
        support = sum(1 for gold in gold_labels if gold == label)
        predicted = sum(1 for pred in pred_labels if pred == label)
        row = prf(tp, fp, fn)
        row["support"] = support
        row["predicted"] = predicted
        per_label[label] = row
        if support > 0 or predicted > 0:
            observed_labels.append(label)

    macro_f1 = safe_div(sum(float(per_label[label]["f1"]) for label in observed_labels), len(observed_labels))
    macro_precision = safe_div(sum(float(per_label[label]["precision"]) for label in observed_labels), len(observed_labels))
    macro_recall = safe_div(sum(float(per_label[label]["recall"]) for label in observed_labels), len(observed_labels))
    return {
        "total": total,
        "accuracy": safe_div(correct, total),
        "correct": correct,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label,
    }


def binary_report(gold_positive: list[bool], pred_positive: list[bool]) -> dict[str, float | int]:
    tp = sum(1 for gold, pred in zip(gold_positive, pred_positive) if gold and pred)
    fp = sum(1 for gold, pred in zip(gold_positive, pred_positive) if not gold and pred)
    fn = sum(1 for gold, pred in zip(gold_positive, pred_positive) if gold and not pred)
    tn = sum(1 for gold, pred in zip(gold_positive, pred_positive) if not gold and not pred)
    report = prf(tp, fp, fn)
    report["tn"] = tn
    report["accuracy"] = safe_div(tp + tn, tp + fp + fn + tn)
    return report


def relation_label_map(graph: CoarseCausalGraph) -> dict[tuple[str, str], str]:
    pair_to_label: dict[tuple[str, str], str] = {}
    for edge in graph.edges:
        pair = (edge.source_event_id, edge.target_event_id)
        current = pair_to_label.get(pair)
        candidate = normalize_label(edge.relation_type)
        if current is None or RELATION_PRIORITY.get(candidate, 0) > RELATION_PRIORITY.get(current, 0):
            pair_to_label[pair] = candidate
    return pair_to_label


def graph_edge_sets(graph: CoarseCausalGraph, sample_id: str) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str, str]]]:
    unlabeled = {
        (sample_id, edge.source_event_id, edge.target_event_id)
        for edge in graph.edges
    }
    labeled = {
        (sample_id, edge.source_event_id, edge.target_event_id, normalize_label(edge.relation_type))
        for edge in graph.edges
    }
    return unlabeled, labeled


def format_prompt(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou classify directed relations between event pairs.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def resolve_generation_eos_ids(tokenizer) -> list[int] | int | None:
    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
    if not eos_ids:
        return None
    return eos_ids if len(eos_ids) > 1 else eos_ids[0]


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def generate_coarse_pair_predictions(
    model,
    tokenizer,
    torch,
    device,
    pair_samples,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any] | None], list[str]]:
    predictions: list[dict[str, Any] | None] = []
    raw_generations: list[str] = []
    if not pair_samples:
        return predictions, raw_generations

    tokenizer.padding_side = "left"
    batch_size = max(1, int(args.coarse_batch_size))
    eos_token_id = resolve_generation_eos_ids(tokenizer)
    with torch.no_grad():
        for start in range(0, len(pair_samples), batch_size):
            batch = pair_samples[start : start + batch_size]
            prompts = []
            for pair_sample in batch:
                item = pair_sample.to_instruction_example(
                    include_query=args.include_query,
                    document_mode=args.document_mode,
                    max_document_chars=args.max_document_chars,
                )
                prompts.append(format_prompt(item["prompt"]))
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=args.coarse_max_length,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.coarse_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_token_id,
            )
            prompt_width = input_ids.shape[-1]
            for row_index in range(len(batch)):
                generated = tokenizer.decode(outputs[row_index][prompt_width:], skip_special_tokens=True).strip()
                raw_generations.append(generated)
                predictions.append(parse_pair_payload(generated))
    return predictions, raw_generations


def build_refinement_sample_via_inference_path(
    coarse_graph: CoarseCausalGraph,
    sample_id: str,
    temp_dir: Path,
    args: argparse.Namespace,
):
    graph_path = temp_dir / f"{sample_id}.json"
    graph_path.write_text(json.dumps({"coarse_graph": coarse_graph.to_dict()}, ensure_ascii=False), encoding="utf-8")
    return load_refinement_sample_from_coarse_graph(
        graph_path,
        sample_id=sample_id,
        include_completion_candidates=args.include_completion_candidates,
        max_completion_edges=args.max_completion_edges if args.max_completion_edges > 0 else None,
    )


def run_refinement_model(model, torch, device, refinement_sample, keep_threshold: float, coarse_graph: CoarseCausalGraph) -> tuple[CoarseCausalGraph, list[float], list[int], list[float]]:
    node_features = torch.tensor(refinement_sample.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(refinement_sample.edge_index, dtype=torch.long, device=device)
    edge_features = torch.tensor(refinement_sample.edge_features, dtype=torch.float32, device=device)
    query_features = torch.tensor(refinement_sample.query_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            query_features=query_features,
        )
    keep_probs = torch.sigmoid(outputs["edge_keep_logits"]).detach().cpu().tolist()
    type_predictions = outputs["edge_type_logits"].argmax(dim=-1).detach().cpu().tolist()
    strength_predictions = outputs["edge_strengths"].detach().cpu().tolist()
    refined_graph = build_refined_graph(
        coarse_graph=coarse_graph,
        edge_descriptions=list(refinement_sample.metadata.get("edge_descriptions", [])),
        keep_probs=keep_probs,
        type_predictions=type_predictions,
        strength_predictions=strength_predictions,
        keep_threshold=keep_threshold,
    )
    return refined_graph, keep_probs, type_predictions, strength_predictions


def load_refinement_model(args: argparse.Namespace, torch, device):
    from refinement_model import TemporalRelationalEdgeRefiner

    model_path = resolve_repo_path(args.refinement_model_path)
    hidden_dim, message_steps, edge_dim, dropout = load_model_config(
        model_path,
        hidden_dim=None,
        message_steps=None,
        edge_dim=None,
        dropout=None,
    )
    model = TemporalRelationalEdgeRefiner(
        edge_dim=edge_dim or EDGE_FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_message_passing_steps=message_steps,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(load_model_state(model_path, torch))
    model.eval()
    return model


def sample_graph_metrics(pred_graph: CoarseCausalGraph, gold_graph: CoarseCausalGraph) -> dict[str, Any]:
    pred_unlabeled, pred_labeled = graph_edge_sets(pred_graph, "sample")
    gold_unlabeled, gold_labeled = graph_edge_sets(gold_graph, "sample")
    return {
        "unlabeled": set_report(pred_unlabeled, gold_unlabeled),
        "labeled": set_report(pred_labeled, gold_labeled),
    }


def main() -> None:
    args = parse_args()
    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_sample_path = resolve_repo_path(args.per_sample_output) if args.per_sample_output else output_path.with_suffix(".samples.jsonl")
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    if per_sample_path.exists():
        per_sample_path.unlink()

    samples = load_maven_document_graph_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit if args.limit > 0 else None,
        max_events=args.max_events,
    )
    if not samples:
        raise RuntimeError(f"No MAVEN samples loaded from split={args.split!r}.")

    try:
        coarse_model, tokenizer, torch = load_trained_qwen_lora(
            base_model_path=resolve_repo_path(args.base_model_path),
            adapter_path=resolve_repo_path(args.coarse_adapter_path),
        )
    except LoraUnavailable as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return
    coarse_model.eval()
    device = next(coarse_model.parameters()).device
    refinement_model = load_refinement_model(args, torch, device)

    coarse_gold_raw_labels: list[str] = []
    coarse_pred_raw_labels: list[str] = []
    coarse_pred_thresholded_labels: list[str] = []
    refinement_gold_labels: list[str] = []
    refinement_pred_labels: list[str] = []
    refinement_gold_keep: list[bool] = []
    refinement_pred_keep: list[bool] = []

    coarse_parse_total = 0
    coarse_parse_success = 0
    coarse_candidate_pairs_global: set[tuple[str, str, str]] = set()
    refinement_candidate_pairs_global: set[tuple[str, str, str]] = set()
    gold_unlabeled_global: set[tuple[str, str, str]] = set()
    gold_labeled_global: set[tuple[str, str, str, str]] = set()
    coarse_unlabeled_global: set[tuple[str, str, str]] = set()
    coarse_labeled_global: set[tuple[str, str, str, str]] = set()
    refined_unlabeled_global: set[tuple[str, str, str]] = set()
    refined_labeled_global: set[tuple[str, str, str, str]] = set()
    per_sample_rows: list[dict[str, Any]] = []

    started = time.time()
    print(
        " | ".join(
            [
                "maven pipeline evaluation",
                f"split={args.split}",
                f"samples={len(samples)}",
                f"device={device}",
                f"max_events={args.max_events}",
                f"max_pairs={args.max_pairs}",
            ]
        ),
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="maven_pipeline_eval_") as temp_name:
        temp_dir = Path(temp_name)
        for sample_index, document_sample in enumerate(samples, start=1):
            sample_id = document_sample.sample_id
            gold_graph = document_sample.gold_graph
            if gold_graph is None:
                continue
            gold_label_map = relation_label_map(gold_graph)
            gold_unlabeled, gold_labeled = graph_edge_sets(gold_graph, sample_id)
            gold_unlabeled_global.update(gold_unlabeled)
            gold_labeled_global.update(gold_labeled)

            pair_samples = build_event_pair_inference_samples(
                sample=document_sample,
                max_sentence_gap=args.max_sentence_gap,
                max_pairs=args.max_pairs,
            )
            pair_predictions, _ = generate_coarse_pair_predictions(
                model=coarse_model,
                tokenizer=tokenizer,
                torch=torch,
                device=device,
                pair_samples=pair_samples,
                args=args,
            )
            coarse_parse_total += len(pair_predictions)
            coarse_parse_success += sum(1 for prediction in pair_predictions if prediction is not None)

            for pair_sample, prediction in zip(pair_samples, pair_predictions):
                pair = (pair_sample.source_event_id, pair_sample.target_event_id)
                coarse_candidate_pairs_global.add((sample_id, pair[0], pair[1]))
                gold_label = gold_label_map.get(pair, "none")
                pred_raw = normalize_label(prediction.get("relation_type") if prediction else "none")
                pred_score = float(prediction.get("score", 0.0)) if prediction else 0.0
                pred_thresholded = pred_raw if pred_raw != "none" and pred_score >= args.coarse_keep_threshold else "none"
                coarse_gold_raw_labels.append(gold_label)
                coarse_pred_raw_labels.append(pred_raw)
                coarse_pred_thresholded_labels.append(pred_thresholded)

            coarse_graph = build_graph_from_pair_predictions(
                document_sample=document_sample,
                pair_samples=pair_samples,
                pair_predictions=pair_predictions,
                keep_threshold=args.coarse_keep_threshold,
            )
            coarse_unlabeled, coarse_labeled = graph_edge_sets(coarse_graph, sample_id)
            coarse_unlabeled_global.update(coarse_unlabeled)
            coarse_labeled_global.update(coarse_labeled)

            refinement_sample = build_refinement_sample_via_inference_path(
                coarse_graph=coarse_graph,
                sample_id=sample_id,
                temp_dir=temp_dir,
                args=args,
            )
            edge_descriptions = list(refinement_sample.metadata.get("edge_descriptions", []))
            for edge_desc in edge_descriptions:
                refinement_candidate_pairs_global.add(
                    (
                        sample_id,
                        str(edge_desc.get("source_event_id", "")),
                        str(edge_desc.get("target_event_id", "")),
                    )
                )

            refined_graph, keep_probs, type_predictions, _ = run_refinement_model(
                model=refinement_model,
                torch=torch,
                device=device,
                refinement_sample=refinement_sample,
                keep_threshold=args.refinement_keep_threshold,
                coarse_graph=coarse_graph,
            )
            for edge_desc, keep_prob, type_prediction in zip(edge_descriptions, keep_probs, type_predictions):
                pair = (
                    str(edge_desc.get("source_event_id", "")),
                    str(edge_desc.get("target_event_id", "")),
                )
                gold_label = gold_label_map.get(pair, "none")
                pred_keep = float(keep_prob) >= args.refinement_keep_threshold
                pred_relation = normalize_label(ID_TO_RELATION.get(int(type_prediction), "precedes")) if pred_keep else "none"
                refinement_gold_labels.append(gold_label)
                refinement_pred_labels.append(pred_relation)
                refinement_gold_keep.append(gold_label != "none")
                refinement_pred_keep.append(pred_keep)

            refined_unlabeled, refined_labeled = graph_edge_sets(refined_graph, sample_id)
            refined_unlabeled_global.update(refined_unlabeled)
            refined_labeled_global.update(refined_labeled)

            sample_row = {
                "sample_id": sample_id,
                "gold_edges": len(gold_graph.edges),
                "coarse_candidate_pairs": len(pair_samples),
                "coarse_pred_edges": len(coarse_graph.edges),
                "refinement_candidate_edges": len(edge_descriptions),
                "refined_pred_edges": len(refined_graph.edges),
                "coarse_parse_rate": safe_div(
                    sum(1 for prediction in pair_predictions if prediction is not None),
                    len(pair_predictions),
                ),
                "coarse_graph": sample_graph_metrics(coarse_graph, gold_graph),
                "refined_graph": sample_graph_metrics(refined_graph, gold_graph),
            }
            per_sample_rows.append(sample_row)
            with per_sample_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample_row, ensure_ascii=False) + "\n")

            if args.log_every > 0 and (sample_index % args.log_every == 0 or sample_index == len(samples)):
                print(
                    f"evaluated {sample_index}/{len(samples)} samples | elapsed={format_seconds(time.time() - started)}",
                    flush=True,
                )

    coarse_candidate_covered_gold = len(coarse_candidate_pairs_global & gold_unlabeled_global)
    refinement_candidate_covered_gold = len(refinement_candidate_pairs_global & gold_unlabeled_global)
    metrics = {
        "config": {
            **vars(args),
            "dataset": str(resolve_repo_path(args.dataset)),
            "base_model_path": str(resolve_repo_path(args.base_model_path)),
            "coarse_adapter_path": str(resolve_repo_path(args.coarse_adapter_path)),
            "refinement_model_path": str(resolve_repo_path(args.refinement_model_path)),
            "device": str(device),
        },
        "dataset": {
            "samples": len(samples),
            "gold_edges": len(gold_unlabeled_global),
            "gold_labeled_edges": len(gold_labeled_global),
        },
        "coarse": {
            "parse_rate": safe_div(coarse_parse_success, coarse_parse_total),
            "candidate_pairs": len(coarse_candidate_pairs_global),
            "candidate_gold_coverage": safe_div(coarse_candidate_covered_gold, len(gold_unlabeled_global)),
            "candidate_gold_covered": coarse_candidate_covered_gold,
            "pair_classification_raw": classification_report(coarse_gold_raw_labels, coarse_pred_raw_labels),
            "pair_classification_thresholded": classification_report(coarse_gold_raw_labels, coarse_pred_thresholded_labels),
            "edge_binary_thresholded": binary_report(
                [label != "none" for label in coarse_gold_raw_labels],
                [label != "none" for label in coarse_pred_thresholded_labels],
            ),
            "graph_unlabeled": set_report(coarse_unlabeled_global, gold_unlabeled_global),
            "graph_labeled": set_report(coarse_labeled_global, gold_labeled_global),
        },
        "refinement": {
            "candidate_edges": len(refinement_candidate_pairs_global),
            "candidate_gold_coverage": safe_div(refinement_candidate_covered_gold, len(gold_unlabeled_global)),
            "candidate_gold_covered": refinement_candidate_covered_gold,
            "keep_binary": binary_report(refinement_gold_keep, refinement_pred_keep),
            "candidate_label_classification": classification_report(refinement_gold_labels, refinement_pred_labels),
            "graph_unlabeled": set_report(refined_unlabeled_global, gold_unlabeled_global),
            "graph_labeled": set_report(refined_labeled_global, gold_labeled_global),
        },
        "outputs": {
            "metrics": str(output_path),
            "per_sample": str(per_sample_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        " | ".join(
            [
                f"saved metrics to {output_path}",
                f"coarse_labeled_f1={metrics['coarse']['graph_labeled']['f1']:.4f}",
                f"refined_labeled_f1={metrics['refinement']['graph_labeled']['f1']:.4f}",
                f"elapsed={format_seconds(float(metrics['elapsed_seconds']))}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
