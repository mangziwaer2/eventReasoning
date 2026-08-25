from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from coarse_graph_dataset import build_event_pair_inference_samples
from coarse_graph_dataset import build_graph_from_pair_predictions
from coarse_graph_dataset import load_maven_document_graph_samples
from coarse_graph_dataset import parse_pair_payload
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_qwen_for_inference
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path
from refinement_dataset import gold_and_coarse_graph_to_refinement_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen coarse graphs and cache MAVEN-ERE refinement supervision."
    )
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="Rows to process; 0 means the full split.")
    parser.add_argument("--max-events", type=int, default=16)
    parser.add_argument("--max-sentence-gap", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=64, help="Qwen candidate pairs per document; 0 means all.")
    parser.add_argument("--coarse-keep-threshold", type=float, default=0.5)
    parser.add_argument("--coarse-topology-mode", choices=["none", "temporal-dag"], default="temporal-dag")
    parser.add_argument("--base-model-path", default=str(REPO_ROOT / "models" / "Qwen3-4B"))
    parser.add_argument("--coarse-adapter-path", default=None)
    parser.add_argument("--coarse-batch-size", type=int, default=1)
    parser.add_argument("--coarse-max-length", type=int, default=1024)
    parser.add_argument("--coarse-max-new-tokens", type=int, default=48)
    parser.add_argument("--include-query", action="store_true")
    parser.add_argument("--document-mode", choices=["none", "title", "snippet", "summary", "full"], default="title")
    parser.add_argument("--max-document-chars", type=int, default=240)
    parser.add_argument(
        "--negative-completion-ratio",
        type=float,
        default=0.0,
        help="Optional heuristic negatives per ERE gold edge. Missing gold pairs are never injected.",
    )
    parser.add_argument("--max-completion-edges", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing samples.jsonl in output-dir.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "maven_qwen_refinement_cache"),
        help="Directory containing samples.jsonl and cache_manifest.json.",
    )
    return parser.parse_args()


def format_prompt(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou classify directed relations between event pairs."
        "<|im_end|>\n<|im_start|>user\n"
        f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
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


def generate_pair_predictions(model, tokenizer, torch, device, pair_samples, args):
    predictions: list[dict[str, Any] | None] = []
    raw_generations: list[str] = []
    tokenizer.padding_side = "left"
    eos_token_id = resolve_generation_eos_ids(tokenizer)
    batch_size = max(1, int(args.coarse_batch_size))
    with torch.no_grad():
        for start in range(0, len(pair_samples), batch_size):
            batch = pair_samples[start : start + batch_size]
            prompts = [
                format_prompt(
                    pair.to_instruction_example(
                        include_query=args.include_query,
                        document_mode=args.document_mode,
                        max_document_chars=args.max_document_chars,
                    )["prompt"]
                )
                for pair in batch
            ]
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
                generated = tokenizer.decode(
                    outputs[row_index][prompt_width:], skip_special_tokens=True
                ).strip()
                raw_generations.append(generated)
                predictions.append(parse_pair_payload(generated))
    return predictions, raw_generations


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    manifest_path = output_dir / "cache_manifest.json"
    if samples_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Cache already exists: {samples_path}. Pass --overwrite to replace it."
            )
        samples_path.unlink()

    document_samples = load_maven_document_graph_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit if args.limit > 0 else None,
        max_events=args.max_events if args.max_events > 0 else None,
    )
    if not document_samples:
        raise RuntimeError(f"No MAVEN-ERE samples loaded for split={args.split!r}.")

    try:
        adapter_path = resolve_repo_path(args.coarse_adapter_path) if args.coarse_adapter_path else None
        model, tokenizer, torch = load_qwen_for_inference(
            base_model_path=resolve_repo_path(args.base_model_path),
            adapter_path=adapter_path,
        )
    except LoraUnavailable as exc:
        raise RuntimeError(str(exc)) from exc

    model.eval()
    device = next(model.parameters()).device
    stats = {
        "source_rows": len(document_samples),
        "cached_samples": 0,
        "trainable_samples": 0,
        "skipped_no_pairs": 0,
        "zero_candidate_samples": 0,
        "coarse_pairs": 0,
        "coarse_parse_success": 0,
        "coarse_edges": 0,
        "refinement_candidates": 0,
        "refinement_positive_edges": 0,
    }
    started = time.time()

    with samples_path.open("w", encoding="utf-8") as handle:
        for row_index, document_sample in enumerate(document_samples, start=1):
            gold_graph = document_sample.gold_graph
            if gold_graph is None or len(document_sample.events) < 2:
                continue
            pair_samples = build_event_pair_inference_samples(
                document_sample,
                max_sentence_gap=args.max_sentence_gap,
                max_pairs=args.max_pairs,
            )
            if not pair_samples:
                stats["skipped_no_pairs"] += 1
                continue
            predictions, raw_generations = generate_pair_predictions(
                model, tokenizer, torch, device, pair_samples, args
            )
            stats["coarse_pairs"] += len(pair_samples)
            stats["coarse_parse_success"] += sum(item is not None for item in predictions)
            coarse_graph = build_graph_from_pair_predictions(
                document_sample,
                pair_samples,
                predictions,
                keep_threshold=args.coarse_keep_threshold,
                topology_mode=args.coarse_topology_mode,
            )
            stats["coarse_edges"] += len(coarse_graph.edges)
            refinement_sample = gold_and_coarse_graph_to_refinement_sample(
                sample_id=document_sample.sample_id,
                gold_graph=gold_graph,
                coarse_graph=coarse_graph,
                negative_completion_ratio=args.negative_completion_ratio,
                max_completion_edges=args.max_completion_edges or None,
                seed=args.seed + row_index,
                include_missing_gold_pairs=False,
            )

            pair_rows = []
            for pair, prediction, raw in zip(pair_samples, predictions, raw_generations):
                pair_rows.append(
                    {
                        "sample_id": pair.sample_id,
                        "source_event_id": pair.source_event_id,
                        "target_event_id": pair.target_event_id,
                        "candidate_score": pair.metadata.get("candidate_score"),
                        "prediction": prediction,
                        "raw_generation": raw,
                    }
                )
            entry = {
                "schema_version": "maven-ere-qwen-refinement-v1",
                "sample_id": document_sample.sample_id,
                "gold_graph": gold_graph.to_dict(),
                "coarse_graph": coarse_graph.to_dict(),
                "refinement_candidate_graph": coarse_graph.to_dict(),
                "refinement_sample": refinement_sample.to_dict(),
                "coarse_pair_results": pair_rows,
                "metadata": {
                    "dataset": "MAVEN-ERE",
                    "split": args.split,
                    "base_model_path": str(resolve_repo_path(args.base_model_path)),
                    "coarse_adapter_path": str(adapter_path) if adapter_path else None,
                    "coarse_keep_threshold": args.coarse_keep_threshold,
                    "coarse_topology_mode": args.coarse_topology_mode,
                    "gold_completion_candidates_enabled": False,
                },
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            stats["cached_samples"] += 1
            stats["refinement_candidates"] += len(refinement_sample.edge_index)
            stats["refinement_positive_edges"] += sum(refinement_sample.edge_labels)
            if refinement_sample.edge_index:
                stats["trainable_samples"] += 1
            else:
                stats["zero_candidate_samples"] += 1
            if row_index % 10 == 0 or row_index == len(document_samples):
                print(
                    f"cached {row_index}/{len(document_samples)} | "
                    f"rows={stats['cached_samples']} | trainable={stats['trainable_samples']} | "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    manifest = {
        "schema_version": "maven-ere-qwen-refinement-v1",
        "config": vars(args),
        "resolved_base_model_path": str(resolve_repo_path(args.base_model_path)),
        "resolved_dataset": str(resolve_repo_path(args.dataset)),
        "device": str(device),
        "stats": {
            **stats,
            "coarse_parse_rate": (
                stats["coarse_parse_success"] / stats["coarse_pairs"]
                if stats["coarse_pairs"] else 0.0
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
