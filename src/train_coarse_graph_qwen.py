from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coarse_graph_dataset import EventPairSample
from coarse_graph_dataset import PAIR_RELATION_TYPES
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
    parser.add_argument("--train-limit", type=int, default=4, help="Maximum number of training rows.")
    parser.add_argument("--validation-limit", type=int, default=1, help="Maximum number of validation rows.")
    parser.add_argument("--max-train-pairs", type=int, default=0, help="Optional cap after train rows are expanded into event pairs.")
    parser.add_argument("--max-validation-pairs", type=int, default=0, help="Optional cap after validation rows are expanded into event pairs.")
    parser.add_argument("--max-events", type=int, default=16, help="Maximum events kept per source document.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative pair sampling ratio.")
    parser.add_argument("--max-sentence-gap", type=int, default=3, help="Maximum sentence gap for candidate pair construction.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum token length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Validation batch size.")
    parser.add_argument("--tokenize-batch-size", type=int, default=512, help="Batch size used during prompt tokenization.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--log-every", type=int, default=25, help="Print one progress line every N batches. Use 0 to disable.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--debug-samples", type=int, default=2, help="Number of validation samples printed and saved each epoch.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--include-query", action="store_true", help="Include query text in the training prompt.")
    parser.add_argument("--document-mode", choices=["none", "title", "snippet", "summary", "full"], default="title", help="How much document text to include in the prompt.")
    parser.add_argument("--max-document-chars", type=int, default=240, help="Maximum characters kept per document snippet when document-mode=snippet.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "coarse_graph_qwen_lora"), help="Training output directory.")
    parser.add_argument("--resume-from", default=None, help="Resume from an output dir, adapter dir, or latest_training_state.pt.")
    return parser.parse_args()


def save_json(path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def log_line(message: str, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")


def resolve_resume_sources(resume_from: str | None, output_dir: Path) -> tuple[Path | None, Path | None]:
    if not resume_from:
        return None, None

    resume_path = resolve_repo_path(resume_from)
    state_path: Path | None = None
    adapter_path: Path | None = None

    if resume_path.is_dir():
        latest_state = resume_path / "latest_training_state.pt"
        if latest_state.exists():
            state_path = latest_state
        latest_adapter = resume_path / "latest_adapter"
        best_adapter = resume_path / "best_adapter"
        if latest_adapter.exists():
            adapter_path = latest_adapter
        elif (resume_path / "adapter_config.json").exists():
            adapter_path = resume_path
        elif best_adapter.exists():
            adapter_path = best_adapter
    elif resume_path.is_file():
        if resume_path.suffix.lower() in {".pt", ".pth"}:
            state_path = resume_path
            sibling_adapter = resume_path.parent / ("best_adapter" if "best" in resume_path.stem else "latest_adapter")
            if sibling_adapter.exists():
                adapter_path = sibling_adapter
        elif resume_path.name == "adapter_config.json":
            adapter_path = resume_path.parent

    if adapter_path is None:
        fallback_adapter = output_dir / "latest_adapter"
        if fallback_adapter.exists():
            adapter_path = fallback_adapter

    if adapter_path is None:
        raise FileNotFoundError(
            f"Could not find a LoRA adapter to resume from. "
            f"Checked resume_from={resume_path} and output_dir={output_dir}."
        )
    return state_path, adapter_path


def load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_torch_checkpoint(path: Path, torch, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_training_state(
    path: Path,
    optimizer,
    args: argparse.Namespace,
    history: list[dict[str, float]],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    adapter_path: Path,
    torch,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": vars(args),
            "adapter_path": str(adapter_path),
        },
        path,
    )


def infer_resume_progress(
    output_dir: Path,
    state_path: Path | None,
    optimizer,
    torch,
    device,
    log_path: Path | None,
) -> tuple[int, int, float, list[dict[str, float]]]:
    history: list[dict[str, float]] = []
    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 0

    if state_path is not None and state_path.exists():
        checkpoint = load_torch_checkpoint(state_path, torch, device)
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", len(history)))
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        log_line(
            f"resumed training state from {state_path} | start_epoch={start_epoch} | global_step={global_step}",
            log_path,
        )
    else:
        loaded_history = load_json_if_exists(output_dir / "train_history.json", [])
        if isinstance(loaded_history, list):
            history = loaded_history
        start_epoch = len(history)
        if history:
            global_step = int(history[-1].get("optimizer_steps", 0))
            best_val_loss = min(float(item.get("val_loss", item.get("loss", float("inf")))) for item in history)
            log_line(
                "resumed from adapter without optimizer state"
                f" | inferred_start_epoch={start_epoch}"
                f" | inferred_global_step={global_step}"
                f" | best_val_loss={best_val_loss:.4f}",
                log_path,
            )

    if best_val_loss == float("inf") and history:
        best_val_loss = min(float(item.get("val_loss", item.get("loss", float("inf")))) for item in history)
    return start_epoch, global_step, best_val_loss, history


def summarize_pair_samples(samples: list[EventPairSample]) -> dict[str, Any]:
    relation_counts = {relation: 0 for relation in PAIR_RELATION_TYPES}
    for sample in samples:
        relation_counts[sample.relation_type] = relation_counts.get(sample.relation_type, 0) + 1
    return {
        "samples": len(samples),
        "relation_counts": relation_counts,
        "positive_samples": len(samples) - relation_counts.get("none", 0),
        "negative_samples": relation_counts.get("none", 0),
        "positive_ratio": (len(samples) - relation_counts.get("none", 0)) / len(samples) if samples else 0.0,
    }


def limit_pair_samples(samples: list[EventPairSample], max_pairs: int, seed: int) -> list[EventPairSample]:
    if max_pairs <= 0 or len(samples) <= max_pairs:
        return samples
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    return shuffled[:max_pairs]


def make_progress(iterable, total: int | None, desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print(f"{desc} started", flush=True)
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
    )


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def shorten_text(text: str, max_chars: int = 220) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def _format_prompt_only(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou classify directed relations between event pairs.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _format_target(target: str) -> str:
    return f"{target}<|im_end|>"


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


def encode_samples_batched(
    samples: list[EventPairSample],
    tokenizer,
    args: argparse.Namespace,
    name: str,
    progress_enabled: bool,
    log_path: Path | None,
) -> list[EncodedSample]:
    encoded_samples: list[EncodedSample] = []
    total_samples = len(samples)
    chunk_size = max(1, int(args.tokenize_batch_size))
    ranges = range(0, total_samples, chunk_size)
    for start in make_progress(
        ranges,
        total=(total_samples + chunk_size - 1) // chunk_size if total_samples else 0,
        desc=f"{name} tokenizing",
        enabled=progress_enabled,
    ):
        chunk = samples[start : start + chunk_size]
        text_items = [_build_text_example(sample, args) for sample in chunk]
        prompt_texts = [item["prompt_only"] for item in text_items]
        target_texts = [item["target_text"] for item in text_items]
        prompt_batch = tokenizer(prompt_texts, add_special_tokens=False, padding=False)
        target_batch = tokenizer(target_texts, add_special_tokens=False, padding=False)

        for sample, prompt_ids, target_ids in zip(chunk, prompt_batch["input_ids"], target_batch["input_ids"]):
            if not target_ids:
                raise ValueError(f"Empty target encoding for sample {sample.sample_id}")

            if len(target_ids) >= args.max_length:
                target_ids = target_ids[: max(1, args.max_length - 1)]

            available_prompt_tokens = max(1, args.max_length - len(target_ids))
            prompt_ids = prompt_ids[:available_prompt_tokens]

            input_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids
            attention_mask = [1] * len(input_ids)
            encoded_samples.append(
                EncodedSample(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
            )
    log_line(f"{name} tokenized {len(encoded_samples)}/{total_samples} pair samples", log_path)
    return encoded_samples


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


def build_dataloader(
    samples,
    tokenizer,
    args: argparse.Namespace,
    torch,
    batch_size: int,
    shuffle: bool,
    name: str,
    progress_enabled: bool,
    log_path: Path | None,
):
    from torch.utils.data import DataLoader

    encoded_samples = encode_samples_batched(
        samples=list(samples),
        tokenizer=tokenizer,
        args=args,
        name=name,
        progress_enabled=progress_enabled,
        log_path=log_path,
    )
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
    torch,
    device,
    validation_samples: list[EventPairSample],
    args: argparse.Namespace,
    debug_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not validation_samples or debug_samples <= 0:
        return [], []
    rng = random.Random(seed)
    chosen = rng.sample(validation_samples, min(debug_samples, len(validation_samples)))
    rows: list[dict[str, Any]] = []
    readable_blocks: list[str] = []
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
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=resolve_generation_eos_ids(tokenizer),
            )
            prediction = tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
            parsed = parse_pair_payload(prediction)
            event_lookup = {event.event_id: event for event in sample.events}
            source_event = event_lookup.get(sample.source_event_id)
            target_event = event_lookup.get(sample.target_event_id)
            row = {
                "debug_stage": "pair_qwen_validation_sample",
                "sample_id": sample.sample_id,
                "source_event_id": sample.source_event_id,
                "target_event_id": sample.target_event_id,
                "gold_relation_type": sample.relation_type,
                "gold_score": sample.score,
                "prediction": prediction,
                "parsed_prediction": parsed,
                "source_event": source_event.text if source_event is not None else "",
                "target_event": target_event.text if target_event is not None else "",
            }
            rows.append(row)
            readable_blocks.append(
                "\n".join(
                    [
                        f"coarse qwen debug sample={sample.sample_id}",
                        f"gold={sample.relation_type}:{sample.score:.3f} parsed={parsed}",
                        f"raw={shorten_text(prediction, 240)}",
                        f"source_event: {shorten_text(row['source_event'])}",
                        f"target_event: {shorten_text(row['target_event'])}",
                    ]
                )
            )
    return rows, readable_blocks


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    progress_enabled = not args.no_progress
    resume_state_path, resume_adapter_path = resolve_resume_sources(args.resume_from, output_dir)
    is_resuming = resume_adapter_path is not None
    training_log_path = output_dir / "training.log"
    if training_log_path.exists() and not is_resuming:
        training_log_path.unlink()

    log_line(
        "loading MAVEN pair samples"
        f" | train_limit={args.train_limit}"
        f" | validation_limit={args.validation_limit}"
        f" | max_events={args.max_events}"
        f" | negative_ratio={args.negative_ratio}",
        training_log_path,
    )
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
    original_train_pair_count = len(train_samples)
    original_validation_pair_count = len(validation_samples)
    train_samples = limit_pair_samples(train_samples, args.max_train_pairs, args.seed + 17)
    validation_samples = limit_pair_samples(validation_samples, args.max_validation_pairs, args.seed + 23)
    train_stats = summarize_pair_samples(train_samples)
    validation_stats = summarize_pair_samples(validation_samples)
    log_line(
        "loaded pair samples"
        f" | train_samples={train_stats['samples']}"
        f" | val_samples={validation_stats['samples']}"
        f" | original_train_pairs={original_train_pair_count}"
        f" | original_val_pairs={original_validation_pair_count}"
        f" | train_pos_ratio={train_stats['positive_ratio']:.3f}",
        training_log_path,
    )

    try:
        if resume_adapter_path is not None:
            log_line(
                f"loading Qwen LoRA model from {resolve_repo_path(args.model_path)} with adapter {resume_adapter_path}",
                training_log_path,
            )
        else:
            log_line(f"loading Qwen LoRA model from {resolve_repo_path(args.model_path)}", training_log_path)
        model, tokenizer, torch = load_qwen_with_lora(
            resolve_repo_path(args.model_path),
            adapter_path=resume_adapter_path,
        )
    except LoraUnavailable as exc:
        log_line(json.dumps({"error": str(exc)}, ensure_ascii=False), training_log_path)
        return
    device = next(model.parameters()).device
    log_line(f"loaded Qwen LoRA model | device={device}", training_log_path)

    train_dataloader = build_dataloader(
        train_samples,
        tokenizer,
        args,
        torch,
        batch_size=args.batch_size,
        shuffle=True,
        name="train",
        progress_enabled=progress_enabled,
        log_path=training_log_path,
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
            name="validation",
            progress_enabled=progress_enabled,
            log_path=training_log_path,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    history: list[dict[str, float]] = []
    global_step = 0
    best_val_loss = float("inf")
    if is_resuming:
        start_epoch, global_step, best_val_loss, history = infer_resume_progress(
            output_dir=output_dir,
            state_path=resume_state_path,
            optimizer=optimizer,
            torch=torch,
            device=device,
            log_path=training_log_path,
        )
    train_started = time.time()
    debug_jsonl_path = output_dir / "debug_predictions.jsonl"
    debug_readable_path = output_dir / "debug_readable.log"
    if debug_jsonl_path.exists() and start_epoch == 0:
        debug_jsonl_path.unlink()
    if debug_readable_path.exists() and start_epoch == 0:
        debug_readable_path.unlink()

    train_config = {
        **vars(args),
        "train_stats": train_stats,
        "validation_stats": validation_stats,
        "original_train_pair_count": original_train_pair_count,
        "original_validation_pair_count": original_validation_pair_count,
        "resume_from": str(resolve_repo_path(args.resume_from)) if args.resume_from else None,
        "resume_state_path": str(resume_state_path) if resume_state_path else None,
        "resume_adapter_path": str(resume_adapter_path) if resume_adapter_path else None,
        "start_epoch": start_epoch,
        "task": "qwen_event_pair_to_coarse_graph",
    }
    save_json(output_dir / "train_config.json", train_config)
    log_line(
        " | ".join(
            [
                "coarse qwen training",
                f"device={device}",
                f"train_samples={train_stats['samples']}",
                f"val_samples={validation_stats['samples']}",
                f"pos_ratio={train_stats['positive_ratio']:.3f}",
                f"batch_size={args.batch_size}",
                f"grad_accum={args.gradient_accumulation_steps}",
                f"lr={args.lr:.2e}",
            ]
        ),
        training_log_path,
    )
    log_line(f"train_relation_counts={json.dumps(train_stats['relation_counts'], ensure_ascii=False)}", training_log_path)
    log_line(f"val_relation_counts={json.dumps(validation_stats['relation_counts'], ensure_ascii=False)}", training_log_path)
    if start_epoch >= args.epochs:
        log_line(
            f"resume start_epoch={start_epoch} is already >= epochs={args.epochs}; no remaining epochs to train.",
            training_log_path,
        )

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        window_loss = 0.0
        batch_count = 0
        window_count = 0

        train_iterator = make_progress(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"epoch {epoch + 1:03d}/{args.epochs:03d}",
            enabled=progress_enabled,
        )
        for batch_index, batch in enumerate(train_iterator, start=1):
            loss = compute_batch_loss(model, batch, device)
            detached_loss = float(loss.item())
            scaled_loss = loss / max(args.gradient_accumulation_steps, 1)
            scaled_loss.backward()

            if batch_index % max(args.gradient_accumulation_steps, 1) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss += detached_loss
            window_loss += detached_loss
            batch_count += 1
            window_count += 1

            if progress_enabled and hasattr(train_iterator, "set_postfix"):
                train_iterator.set_postfix(
                    loss=f"{window_loss / max(window_count, 1):.4f}",
                    step=global_step,
                )

            if (
                not progress_enabled
                and args.log_every > 0
                and (batch_index % args.log_every == 0 or batch_index == len(train_dataloader))
            ):
                log_line(
                    " | ".join(
                        [
                            f"epoch {epoch + 1:03d}/{args.epochs:03d} batch {batch_index:04d}/{len(train_dataloader):04d}",
                            f"loss={window_loss / max(window_count, 1):.4f}",
                            f"global_step={global_step}",
                            f"time={format_seconds(time.time() - epoch_started)}",
                        ]
                    ),
                    training_log_path,
                )
                window_loss = 0.0
                window_count = 0

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
        val_loss = float(record.get("val_loss", record["loss"]))
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_adapter_dir = output_dir / "best_adapter"
            model.save_pretrained(best_adapter_dir)
            tokenizer.save_pretrained(best_adapter_dir)
            save_training_state(
                output_dir / "best_training_state.pt",
                optimizer=optimizer,
                args=args,
                history=history,
                epoch=epoch + 1,
                global_step=global_step,
                best_val_loss=best_val_loss,
                adapter_path=best_adapter_dir,
                torch=torch,
            )
        latest_adapter_dir = output_dir / "latest_adapter"
        model.save_pretrained(latest_adapter_dir)
        tokenizer.save_pretrained(latest_adapter_dir)
        save_training_state(
            output_dir / "latest_training_state.pt",
            optimizer=optimizer,
            args=args,
            history=history,
            epoch=epoch + 1,
            global_step=global_step,
            best_val_loss=best_val_loss,
            adapter_path=latest_adapter_dir,
            torch=torch,
        )
        save_json(output_dir / "train_history.json", history)

        log_line(
            " | ".join(
                [
                    f"epoch {epoch + 1:03d}/{args.epochs:03d} done" + (" best" if is_best else ""),
                    f"loss={record['loss']:.4f}",
                    f"val_loss={record.get('val_loss', float('nan')):.4f}" if "val_loss" in record else "val_loss=n/a",
                    f"optimizer_steps={global_step}",
                    f"time={format_seconds(time.time() - epoch_started)}",
                ]
            ),
            training_log_path,
        )
        debug_rows, debug_blocks = print_debug_samples(
            model,
            tokenizer,
            torch,
            device,
            validation_samples,
            args,
            args.debug_samples,
            args.seed + epoch,
        )
        if debug_rows:
            with debug_jsonl_path.open("a", encoding="utf-8") as debug_file:
                for row in debug_rows:
                    debug_file.write(json.dumps({"epoch": epoch + 1, **row}, ensure_ascii=False) + "\n")
        if debug_blocks:
            readable_text = "\n\n".join(f"[epoch {epoch + 1:03d}] {block}" for block in debug_blocks)
            log_line(readable_text, training_log_path)
            with debug_readable_path.open("a", encoding="utf-8") as readable_file:
                readable_file.write(readable_text + "\n\n")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json(output_dir / "train_history.json", history)
    log_line(f"Saved LoRA outputs to {output_dir} | total_time={format_seconds(time.time() - train_started)}", training_log_path)


if __name__ == "__main__":
    main()
