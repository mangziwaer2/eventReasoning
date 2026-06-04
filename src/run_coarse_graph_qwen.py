from __future__ import annotations

import argparse
import json
from pathlib import Path

from coarse_graph_dataset import load_jsonl_document_graph_sample
from coarse_graph_dataset import load_mirai_document_graph_sample
from coarse_graph_dataset import sample_graph_from_model_output
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_trained_qwen_lora
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen LoRA document-to-coarse-graph inference.")
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
    parser.add_argument("--max-events", type=int, default=12, help="Maximum total events passed to Qwen.")
    parser.add_argument("--base-model-path", default=str(REPO_ROOT / "models" / "Qwen2.5-0.5B"), help="Base Qwen model directory.")
    parser.add_argument("--adapter-path", default=str(REPO_ROOT / "outputs" / "coarse_graph_qwen_lora"), help="Trained LoRA adapter directory.")
    parser.add_argument("--max-length", type=int, default=1536, help="Maximum prompt length.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def _format_prompt(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou build coarse causal graphs from retrieved news evidence.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    args = parse_args()
    if args.input_mode == "mirai":
        sample = load_mirai_document_graph_sample(
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
        sample = load_jsonl_document_graph_sample(
            input_path=resolve_repo_path(args.input),
            query_text=args.query,
            cutoff_time=args.cutoff,
            event_extractor_name=args.event_extractor,
            max_docs=args.max_docs,
            max_events_per_doc=args.max_events_per_doc,
            max_events=args.max_events,
        )

    item = sample.to_instruction_example()
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
    prompt = _format_prompt(item["prompt"])
    with torch.no_grad():
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()

    coarse_graph = sample_graph_from_model_output(sample, generated)
    result = {
        "sample_id": sample.sample_id,
        "metadata": sample.metadata,
        "prompt": item["prompt"],
        "raw_generation": generated,
        "coarse_graph": coarse_graph.to_dict(),
    }
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
