from __future__ import annotations

import argparse
import json
from pathlib import Path

from coarse_graph_dataset import build_event_pair_inference_samples
from coarse_graph_dataset import build_graph_from_pair_predictions
from coarse_graph_dataset import load_jsonl_document_graph_sample
from coarse_graph_dataset import load_mirai_document_graph_sample
from coarse_graph_dataset import parse_pair_payload
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_trained_qwen_lora
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen LoRA event-pair relation classification and assemble a coarse graph.")
    parser.add_argument("--input-mode", choices=["mirai", "jsonl"], default="mirai", help="Input source.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MIRAI_data.zip"), help="Path to MIRAI zip file when input-mode=mirai.")
    parser.add_argument("--query-id", default="1", help="MIRAI QueryId when input-mode=mirai.")
    parser.add_argument("--split", default="test", help="MIRAI split name.")
    parser.add_argument("--input", default=None, help="Path to news JSONL file when input-mode=jsonl.")
    parser.add_argument("--query", default=None, help="Query text when input-mode=jsonl.")
    parser.add_argument("--cutoff", default=None, help="Cutoff time when input-mode=jsonl.")
    parser.add_argument("--event-extractor", default="rule", help="Event extractor backend name.")
    parser.add_argument("--max-docs", type=int, default=6, help="Maximum retrieved documents.")
    parser.add_argument("--max-events-per-doc", type=int, default=6, help="Maximum events kept per document.")
    parser.add_argument("--max-events", type=int, default=12, help="Maximum total events passed to pair classification.")
    parser.add_argument("--max-sentence-gap", type=int, default=3, help="Maximum sentence gap for same-document candidate pairs.")
    parser.add_argument("--max-pairs", type=int, default=64, help="Maximum number of candidate pairs scored.")
    parser.add_argument("--keep-threshold", type=float, default=0.5, help="Minimum pair score required to keep an edge.")
    parser.add_argument("--base-model-path", default=str(REPO_ROOT / "models" / "Qwen2.5-0.5B"), help="Base Qwen model directory.")
    parser.add_argument("--adapter-path", default=str(REPO_ROOT / "outputs" / "coarse_graph_qwen_lora"), help="Trained LoRA adapter directory.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum prompt length.")
    parser.add_argument("--include-query", action="store_true", help="Include query text in the prompt.")
    parser.add_argument("--document-mode", choices=["none", "title", "snippet", "summary", "full"], default="title", help="How much document text to include in the prompt.")
    parser.add_argument("--max-document-chars", type=int, default=240, help="Maximum characters kept per document snippet when document-mode=snippet.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def _format_prompt(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou classify directed relations between event pairs.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    args = parse_args()
    if args.input_mode == "mirai":
        document_sample = load_mirai_document_graph_sample(
            dataset_path=resolve_repo_path(args.dataset),
            query_id=args.query_id,
            split=args.split,
            event_extractor_name=args.event_extractor,
            max_docs=args.max_docs,
            max_events_per_doc=args.max_events_per_doc,
            max_events=args.max_events,
        )
    else:
        if not args.input or not args.query:
            raise ValueError("--input and --query are required when --input-mode=jsonl")
        document_sample = load_jsonl_document_graph_sample(
            input_path=resolve_repo_path(args.input),
            query_text=args.query,
            cutoff_time=args.cutoff,
            event_extractor_name=args.event_extractor,
            max_docs=args.max_docs,
            max_events_per_doc=args.max_events_per_doc,
            max_events=args.max_events,
        )

    pair_samples = build_event_pair_inference_samples(
        sample=document_sample,
        max_sentence_gap=args.max_sentence_gap,
        max_pairs=args.max_pairs,
    )

    try:
        model, tokenizer, torch = load_trained_qwen_lora(
            base_model_path=resolve_repo_path(args.base_model_path),
            adapter_path=resolve_repo_path(args.adapter_path),
        )
    except LoraUnavailable as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return

    model.eval()
    device = next(model.parameters()).device
    predictions: list[dict[str, object] | None] = []
    pair_previews: list[dict[str, object]] = []

    with torch.no_grad():
        for pair_sample in pair_samples:
            item = pair_sample.to_instruction_example(
                include_query=args.include_query,
                document_mode=args.document_mode,
                max_document_chars=args.max_document_chars,
            )
            prompt = _format_prompt(item["prompt"])
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
            parsed = parse_pair_payload(generated)
            predictions.append(parsed)
            pair_previews.append(
                {
                    "sample_id": pair_sample.sample_id,
                    "source_event_id": pair_sample.source_event_id,
                    "target_event_id": pair_sample.target_event_id,
                    "candidate_score": pair_sample.metadata.get("candidate_score", pair_sample.score),
                    "raw_generation": generated,
                    "parsed_prediction": parsed,
                }
            )

    coarse_graph = build_graph_from_pair_predictions(
        document_sample=document_sample,
        pair_samples=pair_samples,
        pair_predictions=predictions,
        keep_threshold=args.keep_threshold,
    )
    result = {
        "sample_id": document_sample.sample_id,
        "metadata": document_sample.metadata,
        "candidate_pair_count": len(pair_samples),
        "pair_predictions": pair_previews,
        "coarse_graph": coarse_graph.to_dict(),
    }
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
