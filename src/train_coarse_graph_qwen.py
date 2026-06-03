from __future__ import annotations

import argparse
import json
from pathlib import Path

from coarse_graph_dataset import load_maven_pair_samples
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_qwen_with_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen LoRA coarse-graph proposer on MAVEN-ERE event-pair samples.")
    parser.add_argument("--dataset", default="datasets/MAVEN_ERE.zip", help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--model-path", default="models/Qwen2.5-0.5B", help="Local Qwen model directory.")
    parser.add_argument("--split", default="train", help="MAVEN split name.")
    parser.add_argument("--limit", type=int, default=128, help="Maximum number of MAVEN rows.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative to positive pair ratio.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--output-dir", default="outputs/coarse_graph_qwen_lora", help="Training output directory.")
    return parser.parse_args()


def _format_text(prompt: str, target: str) -> str:
    return (
        "<|im_start|>system\nYou infer causal relations between structured events.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_maven_pair_samples(
        dataset_path=Path(args.dataset),
        split=args.split,
        limit=args.limit,
        negative_ratio=args.negative_ratio,
    )
    instruction_samples = [sample.to_instruction_example() for sample in samples]

    try:
        model, tokenizer, torch = load_qwen_with_lora(Path(args.model_path))
    except LoraUnavailable as exc:
        error_payload = {"error": str(exc)}
        print(json.dumps(error_payload, ensure_ascii=False))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    device = next(model.parameters()).device
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        total_loss = 0.0
        sample_count = 0
        for item in instruction_samples:
            text = _format_text(item["prompt"], item["target"])
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            sample_count += 1

        record = {
            "epoch": float(epoch + 1),
            "loss": total_loss / max(sample_count, 1),
        }
        history.append(record)
        print(json.dumps(record))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "limit": args.limit,
                "negative_ratio": args.negative_ratio,
                "model_path": args.model_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved LoRA outputs to {output_dir}")


if __name__ == "__main__":
    main()
