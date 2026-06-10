from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from coarse_graph_dataset import EventPairSample
from coarse_graph_dataset import load_maven_event_pair_samples
from coarse_graph_dataset import parse_pair_payload
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_qwen_with_lora
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen LoRA event-pair relation classifier.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "models" / "Qwen2.5-0.5B"), help="Local Qwen model directory.")
    parser.add_argument("--train-split", default="train", help="Training split name from the dataset.")
    parser.add_argument("--validation-split", default="valid", help="Validation split name from the dataset.")
    parser.add_argument("--train-limit", type=int, default=128, help="Maximum number of training rows.")
    parser.add_argument("--validation-limit", type=int, default=32, help="Maximum number of validation rows.")
    parser.add_argument("--max-events", type=int, default=12, help="Maximum events kept per source document.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative pair sampling ratio.")
    parser.add_argument("--max-sentence-gap", type=int, default=3, help="Maximum sentence gap for candidate pair construction.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum token length.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Validation batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--debug-samples", type=int, default=2, help="Number of validation samples printed each epoch.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--include-query", action="store_true", help="Include query text in the training prompt.")
    parser.add_argument("--document-mode", choices=["none", "title", "snippet", "summary", "full"], default="title", help="How much document text to include in the prompt.")
    parser.add_argument("--max-document-chars", type=int, default=240, help="Maximum characters kept per document snippet when document-mode=snippet.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "coarse_graph_qwen_lora"), help="Training output directory.")
    return parser.parse_args()


def _format_prompt_only(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou classify directed relations between event pairs.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _format_target(target: str) -> str:
    return f"{target}<|im_end|>"


@dataclass(slots=True)
class EncodedSample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


def _build_text_example(sample: EventPairSample, args: argparse.Namespace) -> dict[str, str]:
    item = sample.to_instruction_example(
        include_query=args.include_query,
        document_mode=args.document_mode,
        max_document_chars=args.max_document_chars,
    )
    return {
        "prompt_only": _format_prompt_only(item["prompt"]),
        "target_text": _format_target(item["target"]),
        "prompt": item["prompt"],
        "target": item["target"],
    }


def encode_sample(sample: EventPairSample, tokenizer, args: argparse.Namespace) -> EncodedSample:
    item = _build_text_example(sample, args)
    prompt_ids = tokenizer.encode(item["prompt_only"], add_special_tokens=False)
    target_ids = tokenizer.encode(item["target_text"], add_special_tokens=False)

    if not target_ids:
        raise ValueError(f"Empty target encoding for sample {sample.sample_id}")

    if len(target_ids) >= args.max_length:
        target_ids = target_ids[: max(1, args.max_length - 1)]

    available_prompt_tokens = max(1, args.max_length - len(target_ids))
    prompt_ids = prompt_ids[:available_prompt_tokens]

    input_ids = prompt_ids + target_ids
    labels = ([-100] * len(prompt_ids)) + target_ids
    attention_mask = [1] * len(input_ids)
    return EncodedSample(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def collate_encoded_samples(batch, tokenizer, torch):
    if not batch:
        raise ValueError("Empty batch.")
    max_length = max(len(item.input_ids) for item in batch)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer pad token is required for batching.")

    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        pad_size = max_length - len(item.input_ids)
        input_ids.append(item.input_ids + [pad_token_id] * pad_size)
        labels.append(item.labels + ([-100] * pad_size))
        attention_mask.append(item.attention_mask + ([0] * pad_size))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def build_dataloader(samples, tokenizer, args: argparse.Namespace, torch, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    encoded_samples = [encode_sample(sample, tokenizer, args) for sample in samples]
    return DataLoader(
        encoded_samples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_encoded_samples(batch, tokenizer, torch),
    )


def compute_batch_loss(model, batch, device):
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    return outputs.loss


def evaluate(model, dataloader, device, torch):
    if dataloader is None:
        return None
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = compute_batch_loss(model, batch, device)
            total_loss += float(loss.item())
            batch_count += 1
    return {"val_loss": total_loss / max(batch_count, 1)}


def print_debug_samples(
    model,
    tokenizer,
    device,
    validation_samples: list[EventPairSample],
    args: argparse.Namespace,
    debug_samples: int,
    seed: int,
) -> None:
    if not validation_samples or debug_samples <= 0:
        return
    rng = random.Random(seed)
    chosen = rng.sample(validation_samples, min(debug_samples, len(validation_samples)))
    model.eval()
    with torch.no_grad():
        for sample in chosen:
            item = _build_text_example(sample, args)
            encoded = tokenizer(
                item["prompt_only"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
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
            prediction = tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
            parsed = parse_pair_payload(prediction)
            print(
                json.dumps(
                    {
                        "debug_stage": "pair_qwen_validation_sample",
                        "sample_id": sample.sample_id,
                        "source_event_id": sample.source_event_id,
                        "target_event_id": sample.target_event_id,
                        "gold_target": item["target"],
                        "prediction": prediction,
                        "parsed_prediction": parsed,
                    },
                    ensure_ascii=False,
                )
            )


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_maven_event_pair_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.train_split,
        limit=args.train_limit,
        max_events=args.max_events,
        negative_ratio=args.negative_ratio,
        max_sentence_gap=args.max_sentence_gap,
        seed=args.seed,
    )
    validation_samples = load_maven_event_pair_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.validation_split,
        limit=args.validation_limit,
        max_events=args.max_events,
        negative_ratio=args.negative_ratio,
        max_sentence_gap=args.max_sentence_gap,
        seed=args.seed + 1,
    )

    try:
        model, tokenizer, torch = load_qwen_with_lora(resolve_repo_path(args.model_path))
    except LoraUnavailable as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return

    train_dataloader = build_dataloader(
        train_samples,
        tokenizer,
        args,
        torch,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_dataloader = None
    if validation_samples:
        validation_dataloader = build_dataloader(
            validation_samples,
            tokenizer,
            args,
            torch,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    device = next(model.parameters()).device
    history: list[dict[str, float]] = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batch_count = 0

        for batch_index, batch in enumerate(train_dataloader, start=1):
            loss = compute_batch_loss(model, batch, device)
            detached_loss = float(loss.item())
            scaled_loss = loss / max(args.gradient_accumulation_steps, 1)
            scaled_loss.backward()

            if batch_index % max(args.gradient_accumulation_steps, 1) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss += detached_loss
            batch_count += 1

        if batch_count % max(args.gradient_accumulation_steps, 1) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        record = {
            "epoch": float(epoch + 1),
            "loss": total_loss / max(batch_count, 1),
            "train_batches": float(batch_count),
            "optimizer_steps": float(global_step),
            "train_pair_samples": float(len(train_samples)),
        }
        validation_record = evaluate(model, validation_dataloader, device, torch)
        if validation_record is not None:
            record.update(validation_record)
        history.append(record)
        print(json.dumps(record))
        print_debug_samples(
            model,
            tokenizer,
            device,
            validation_samples,
            args,
            args.debug_samples,
            args.seed + epoch,
        )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "train_split": args.train_split,
                "validation_split": args.validation_split,
                "train_limit": args.train_limit,
                "validation_limit": args.validation_limit,
                "max_events": args.max_events,
                "negative_ratio": args.negative_ratio,
                "max_sentence_gap": args.max_sentence_gap,
                "model_path": args.model_path,
                "include_query": args.include_query,
                "document_mode": args.document_mode,
                "max_document_chars": args.max_document_chars,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "task": "event_pair_relation_classification",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved LoRA outputs to {output_dir}")


if __name__ == "__main__":
    main()
