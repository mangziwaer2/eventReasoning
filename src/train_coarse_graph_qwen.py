from __future__ import annotations

import argparse
import json
import random

from coarse_graph_dataset import DocumentGraphSample
from coarse_graph_dataset import load_maven_document_graph_samples
from coarse_graph_dataset import sample_graph_from_model_output
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_qwen_with_lora
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen LoRA coarse-graph generator on document-level graph samples.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "models" / "Qwen2.5-0.5B"), help="Local Qwen model directory.")
    parser.add_argument("--split", default="train", help="MAVEN split name.")
    parser.add_argument("--limit", type=int, default=128, help="Maximum number of MAVEN rows.")
    parser.add_argument("--max-events", type=int, default=12, help="Maximum events kept per training sample.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--max-length", type=int, default=1536, help="Maximum token length.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--debug-samples", type=int, default=2, help="Number of validation samples printed each epoch.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for splits and debug sampling.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "coarse_graph_qwen_lora"), help="Training output directory.")
    return parser.parse_args()


def _format_text(prompt: str, target: str) -> str:
    return (
        "<|im_start|>system\nYou build coarse causal graphs from retrieved news evidence.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )


def _format_prompt_only(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou build coarse causal graphs from retrieved news evidence.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def split_samples(
    samples: list[DocumentGraphSample],
    validation_ratio: float,
    seed: int,
) -> tuple[list[DocumentGraphSample], list[DocumentGraphSample]]:
    if validation_ratio <= 0 or len(samples) < 2:
        return samples, []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio))
    if validation_size >= len(shuffled):
        validation_size = len(shuffled) - 1
    return shuffled[validation_size:], shuffled[:validation_size]


def compute_text_loss(model, tokenizer, device, text: str, max_length: int):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    return outputs.loss


def evaluate(model, tokenizer, device, validation_samples: list[DocumentGraphSample], max_length: int):
    if not validation_samples:
        return None
    model.eval()
    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for sample in validation_samples:
            item = sample.to_instruction_example()
            text = _format_text(item["prompt"], item["target"])
            loss = compute_text_loss(model, tokenizer, device, text, max_length)
            total_loss += float(loss.item())
            sample_count += 1
    return {"val_loss": total_loss / max(sample_count, 1)}


def print_debug_samples(model, tokenizer, device, validation_samples: list[DocumentGraphSample], max_length: int, debug_samples: int, seed: int) -> None:
    if not validation_samples or debug_samples <= 0:
        return
    rng = random.Random(seed)
    chosen = rng.sample(validation_samples, min(debug_samples, len(validation_samples)))
    model.eval()
    with torch.no_grad():
        for sample in chosen:
            item = sample.to_instruction_example()
            prompt_only = _format_prompt_only(item["prompt"])
            encoded = tokenizer(
                prompt_only,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
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
            prediction = tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
            pred_graph = sample_graph_from_model_output(sample, prediction)
            print(
                json.dumps(
                    {
                        "debug_stage": "coarse_qwen_validation_sample",
                        "sample_id": sample.sample_id,
                        "query": sample.query.text,
                        "event_count": len(sample.events),
                        "gold_edge_count": len(sample.gold_graph.edges) if sample.gold_graph else 0,
                        "pred_edge_count": len(pred_graph.edges),
                        "target": item["target"],
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
            )


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_maven_document_graph_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit,
        max_events=args.max_events,
    )
    train_samples, validation_samples = split_samples(samples, args.validation_ratio, args.seed)

    try:
        model, tokenizer, torch = load_qwen_with_lora(resolve_repo_path(args.model_path))
    except LoraUnavailable as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    device = next(model.parameters()).device
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for sample in train_samples:
            item = sample.to_instruction_example()
            text = _format_text(item["prompt"], item["target"])
            loss = compute_text_loss(model, tokenizer, device, text, args.max_length)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            sample_count += 1

        record = {
            "epoch": float(epoch + 1),
            "loss": total_loss / max(sample_count, 1),
        }
        validation_record = evaluate(model, tokenizer, device, validation_samples, args.max_length)
        if validation_record is not None:
            record.update(validation_record)
        history.append(record)
        print(json.dumps(record))
        print_debug_samples(
            model,
            tokenizer,
            device,
            validation_samples,
            args.max_length,
            args.debug_samples,
            args.seed + epoch,
        )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "limit": args.limit,
                "max_events": args.max_events,
                "model_path": args.model_path,
                "validation_ratio": args.validation_ratio,
                "debug_samples": args.debug_samples,
                "task": "document_to_coarse_graph",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved LoRA outputs to {output_dir}")


if __name__ == "__main__":
    main()
