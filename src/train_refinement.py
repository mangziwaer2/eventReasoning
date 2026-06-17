from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from refinement_dataset import ID_TO_RELATION
from refinement_dataset import RELATION_TO_ID
from refinement_dataset import EDGE_FEATURE_DIM
from refinement_dataset import RefinementSample
from refinement_dataset import RefinementTensorDataset
from refinement_dataset import generate_synthetic_refinement_samples
from refinement_dataset import load_maven_refinement_samples
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def collate_single(batch):
    return batch[0]


def split_samples(
    samples: list[RefinementSample],
    validation_ratio: float,
    seed: int,
) -> tuple[list[RefinementSample], list[RefinementSample]]:
    if validation_ratio <= 0 or len(samples) < 2:
        return samples, []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio))
    if validation_size >= len(shuffled):
        validation_size = len(shuffled) - 1
    return shuffled[validation_size:], shuffled[:validation_size]


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str, torch):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def move_batch_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if hasattr(value, "to") else value
    return moved


def summarize_samples(samples: list[RefinementSample]) -> dict[str, Any]:
    total_edges = sum(len(sample.edge_labels) for sample in samples)
    positive_edges = sum(sum(1 for label in sample.edge_labels if label > 0.5) for sample in samples)
    relation_counts = {name: 0 for name in RELATION_TO_ID}
    for sample in samples:
        for keep_label, type_label in zip(sample.edge_labels, sample.edge_type_labels):
            if keep_label > 0.5:
                relation_counts[ID_TO_RELATION.get(int(type_label), str(type_label))] = (
                    relation_counts.get(ID_TO_RELATION.get(int(type_label), str(type_label)), 0) + 1
                )
    return {
        "samples": len(samples),
        "edges": total_edges,
        "positive_edges": positive_edges,
        "positive_ratio": positive_edges / total_edges if total_edges else 0.0,
        "relation_counts": relation_counts,
    }


def compute_keep_pos_weight(samples: list[RefinementSample], max_weight: float, torch, device):
    positives = sum(sum(1 for label in sample.edge_labels if label > 0.5) for sample in samples)
    total = sum(len(sample.edge_labels) for sample in samples)
    negatives = max(total - positives, 0)
    if positives <= 0:
        return None
    weight = negatives / positives if negatives > 0 else 1.0
    weight = max(1.0, min(float(weight), max_weight))
    return torch.tensor(weight, dtype=torch.float32, device=device)


def compute_type_class_weights(samples: list[RefinementSample], max_weight: float, torch, device):
    counts = [0 for _ in range(len(RELATION_TO_ID))]
    for sample in samples:
        for keep_label, type_label in zip(sample.edge_labels, sample.edge_type_labels):
            if keep_label > 0.5 and 0 <= int(type_label) < len(counts):
                counts[int(type_label)] += 1
    observed = [count for count in counts if count > 0]
    if len(observed) <= 1:
        return None
    mean_count = sum(observed) / len(observed)
    weights = []
    for count in counts:
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(max(1.0 / max_weight, min(math.sqrt(mean_count / count), max_weight)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_type_loss(type_class_weights, label_smoothing: float, torch, nn):
    try:
        return nn.CrossEntropyLoss(weight=type_class_weights, label_smoothing=label_smoothing)
    except TypeError:
        if label_smoothing > 0:
            print("warning: installed torch does not support label_smoothing; using plain CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=type_class_weights)


def make_grad_scaler(torch, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_autocast_context(torch, enabled: bool):
    if not enabled:
        return nullcontext
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return lambda: torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def print_metrics(prefix: str, metrics: dict[str, float | int | None], elapsed_seconds: float) -> None:
    ordered_keys = (
        "loss",
        "keep_loss",
        "type_loss",
        "strength_loss",
        "density_loss",
        "val_loss",
        "val_keep_loss",
        "val_type_loss",
        "val_strength_loss",
        "val_density_loss",
        "lr",
    )
    parts = [prefix]
    for key in ordered_keys:
        if key in metrics:
            value = metrics[key]
            if key == "lr" and isinstance(value, float):
                parts.append(f"{key}={value:.2e}")
            elif isinstance(value, (float, int)):
                parts.append(f"{key}={format_metric(float(value))}")
            else:
                parts.append(f"{key}={value}")
    parts.append(f"time={format_seconds(elapsed_seconds)}")
    print(" | ".join(parts), flush=True)


def compute_losses(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    keep_loss_fn,
    type_loss_fn,
    strength_loss_fn,
    loss_weights: dict[str, float],
):
    keep_loss = keep_loss_fn(outputs["edge_keep_logits"], batch["edge_labels"])
    keep_probs = outputs["edge_keep_logits"].sigmoid()
    if keep_probs.numel() > 0:
        density_loss = (keep_probs.mean() - batch["edge_labels"].mean()).pow(2)
    else:
        density_loss = keep_loss.new_zeros(())
    positive_mask = batch["edge_labels"] > 0.5
    if positive_mask.any():
        type_loss = type_loss_fn(
            outputs["edge_type_logits"][positive_mask],
            batch["edge_type_labels"][positive_mask],
        )
        strength_loss = strength_loss_fn(
            outputs["edge_strengths"][positive_mask],
            batch["edge_strengths"][positive_mask],
        )
    else:
        zero = keep_loss.new_zeros(())
        type_loss = zero
        strength_loss = zero
    loss = (
        loss_weights["keep"] * keep_loss
        + loss_weights["type"] * type_loss
        + loss_weights["strength"] * strength_loss
        + loss_weights["density"] * density_loss
    )
    return loss, keep_loss, type_loss, strength_loss, density_loss


def evaluate(
    model,
    dataloader,
    keep_loss_fn,
    type_loss_fn,
    strength_loss_fn,
    loss_weights: dict[str, float],
    torch,
    device,
):
    model.eval()
    totals = {
        "val_loss": 0.0,
        "val_keep_loss": 0.0,
        "val_type_loss": 0.0,
        "val_strength_loss": 0.0,
        "val_density_loss": 0.0,
    }
    batch_count = 0
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(
                node_features=batch["node_features"],
                edge_index=batch["edge_index"],
                edge_features=batch["edge_features"],
                query_features=batch["query_features"],
            )
            loss, keep_loss, type_loss, strength_loss, density_loss = compute_losses(
                outputs,
                batch,
                keep_loss_fn,
                type_loss_fn,
                strength_loss_fn,
                loss_weights,
            )
            totals["val_loss"] += float(loss.item())
            totals["val_keep_loss"] += float(keep_loss.item())
            totals["val_type_loss"] += float(type_loss.item())
            totals["val_strength_loss"] += float(strength_loss.item())
            totals["val_density_loss"] += float(density_loss.item())
            batch_count += 1
    if batch_count == 0:
        return None
    return {key: value / batch_count for key, value in totals.items()}


def collect_debug_samples(model, samples: list[RefinementSample], debug_samples: int, seed: int, torch, device) -> list[dict[str, Any]]:
    if not samples or debug_samples <= 0:
        return []
    rng = random.Random(seed)
    chosen = rng.sample(samples, min(debug_samples, len(samples)))
    debug_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for sample in chosen:
            node_features = torch.tensor(sample.node_features, dtype=torch.float32, device=device)
            edge_index = torch.tensor(sample.edge_index, dtype=torch.long, device=device)
            edge_features = torch.tensor(sample.edge_features, dtype=torch.float32, device=device)
            query_features = torch.tensor(sample.query_features, dtype=torch.float32, device=device)
            outputs = model(
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                query_features=query_features,
            )
            keep_probs = torch.sigmoid(outputs["edge_keep_logits"]).detach().cpu().tolist()
            pred_types = outputs["edge_type_logits"].argmax(dim=-1).detach().cpu().tolist()
            pred_strengths = outputs["edge_strengths"].detach().cpu().tolist()
            edge_descriptions = sample.metadata.get("edge_descriptions", [])
            for idx, edge_desc in enumerate(edge_descriptions[: min(3, len(edge_descriptions))]):
                debug_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "query": sample.metadata.get("query_text", ""),
                        "source_event": edge_desc.get("source_text", ""),
                        "target_event": edge_desc.get("target_text", ""),
                        "coarse_relation": edge_desc.get("coarse_relation_type", ""),
                        "gold_relation": edge_desc.get("gold_relation_type", ""),
                        "gold_keep": sample.edge_labels[idx] if idx < len(sample.edge_labels) else None,
                        "pred_keep_prob": keep_probs[idx] if idx < len(keep_probs) else None,
                        "gold_type": sample.edge_type_labels[idx] if idx < len(sample.edge_type_labels) else None,
                        "pred_type": pred_types[idx] if idx < len(pred_types) else None,
                        "gold_strength": sample.edge_strengths[idx] if idx < len(sample.edge_strengths) else None,
                        "pred_strength": pred_strengths[idx] if idx < len(pred_strengths) else None,
                    }
                )
    return debug_rows


def shorten_text(text: Any, max_chars: int = 120) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def relation_name(relation_id: int | None) -> str:
    if relation_id is None:
        return "none"
    return ID_TO_RELATION.get(int(relation_id), str(relation_id))


def edge_debug_row(
    idx: int,
    edge_desc: dict[str, Any],
    keep_prob: float,
    pred_type: int,
    pred_strength: float,
    gold_keep: int | float | None,
    gold_type: int | None,
    gold_strength: float | None,
    keep_threshold: float,
) -> dict[str, Any]:
    candidate_source = str(edge_desc.get("candidate_source", "coarse"))
    pred_relation = relation_name(pred_type)
    gold_relation = edge_desc.get("gold_relation_type") or relation_name(gold_type)
    action = "KEEP" if keep_prob >= keep_threshold else "DROP"
    if candidate_source == "completion" and keep_prob >= keep_threshold:
        action = "ADD"
    elif candidate_source == "completion":
        action = "REJECT"
    gold_action = "KEEP" if gold_keep and float(gold_keep) > 0.5 else "DROP"
    return {
        "idx": idx,
        "action": action,
        "gold_action": gold_action,
        "candidate_source": candidate_source,
        "keep_prob": float(keep_prob),
        "pred_relation": pred_relation,
        "pred_strength": float(pred_strength),
        "gold_relation": str(gold_relation),
        "gold_strength": gold_strength,
        "coarse_relation": edge_desc.get("coarse_relation_type", ""),
        "coarse_score": edge_desc.get("coarse_score", 0.0),
        "source_text": edge_desc.get("source_text", ""),
        "target_text": edge_desc.get("target_text", ""),
    }


def select_debug_edges(rows: list[dict[str, Any]], max_edges: int) -> list[dict[str, Any]]:
    if max_edges <= 0 or len(rows) <= max_edges:
        return rows
    priority: list[dict[str, Any]] = []
    priority.extend([row for row in rows if row["action"] == "ADD"])
    priority.extend([row for row in rows if row["action"] == "DROP" and row["candidate_source"] == "coarse"])
    priority.extend([row for row in rows if row["gold_action"] == "KEEP" and row["action"] not in {"KEEP", "ADD"}])
    priority.extend(sorted(rows, key=lambda row: row["keep_prob"], reverse=True))

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in priority:
        row_id = int(row["idx"])
        if row_id in seen:
            continue
        selected.append(row)
        seen.add(row_id)
        if len(selected) >= max_edges:
            break
    selected.sort(key=lambda row: int(row["idx"]))
    return selected


def format_readable_debug_sample(
    sample: RefinementSample,
    keep_probs: list[float],
    pred_types: list[int],
    pred_strengths: list[float],
    epoch: int,
    keep_threshold: float,
    max_edges: int,
) -> str:
    edge_descriptions = list(sample.metadata.get("edge_descriptions", []))
    rows = [
        edge_debug_row(
            idx=idx,
            edge_desc=edge_desc,
            keep_prob=keep_probs[idx] if idx < len(keep_probs) else 0.0,
            pred_type=pred_types[idx] if idx < len(pred_types) else 0,
            pred_strength=pred_strengths[idx] if idx < len(pred_strengths) else 0.0,
            gold_keep=sample.edge_labels[idx] if idx < len(sample.edge_labels) else None,
            gold_type=sample.edge_type_labels[idx] if idx < len(sample.edge_type_labels) else None,
            gold_strength=sample.edge_strengths[idx] if idx < len(sample.edge_strengths) else None,
            keep_threshold=keep_threshold,
        )
        for idx, edge_desc in enumerate(edge_descriptions)
    ]
    coarse_count = sum(1 for row in rows if row["candidate_source"] == "coarse")
    completion_count = sum(1 for row in rows if row["candidate_source"] == "completion")
    kept_count = sum(1 for row in rows if row["action"] == "KEEP")
    added_count = sum(1 for row in rows if row["action"] == "ADD")
    dropped_count = sum(1 for row in rows if row["action"] == "DROP")
    rejected_count = sum(1 for row in rows if row["action"] == "REJECT")
    gold_kept_count = sum(1 for row in rows if row["gold_action"] == "KEEP")

    lines = [
        "",
        f"[epoch {epoch:03d}] refinement debug sample={sample.sample_id}",
        f"query: {shorten_text(sample.metadata.get('query_text', ''), 180)}",
        (
            "graph: "
            f"events={len(sample.node_features)} "
            f"coarse_edges={coarse_count} completion_candidates={completion_count} "
            f"gold_edges={gold_kept_count} -> refined_keep={kept_count} refined_add={added_count} "
            f"drop={dropped_count} reject={rejected_count}"
        ),
        "edges:",
    ]

    for row in select_debug_edges(rows, max_edges):
        lines.extend(
            [
                (
                    f"  #{row['idx']:03d} {row['action']} "
                    f"source={row['candidate_source']} "
                    f"keep={row['keep_prob']:.3f} "
                    f"pred={row['pred_relation']}:{row['pred_strength']:.3f} "
                    f"candidate_prior={row['coarse_relation']}:{float(row['coarse_score']):.3f} "
                    f"gold={row['gold_action']}:{row['gold_relation']}"
                ),
                f"       source_event: {shorten_text(row['source_text'])}",
                f"       target_event: {shorten_text(row['target_text'])}",
            ]
        )
    return "\n".join(lines)


def collect_readable_debug_samples(
    model,
    samples: list[RefinementSample],
    debug_samples: int,
    seed: int,
    torch,
    device,
    epoch: int,
    keep_threshold: float,
    max_edges: int,
) -> list[str]:
    if not samples or debug_samples <= 0:
        return []
    rng = random.Random(seed)
    chosen = rng.sample(samples, min(debug_samples, len(samples)))
    blocks: list[str] = []
    model.eval()
    with torch.no_grad():
        for sample in chosen:
            node_features = torch.tensor(sample.node_features, dtype=torch.float32, device=device)
            edge_index = torch.tensor(sample.edge_index, dtype=torch.long, device=device)
            edge_features = torch.tensor(sample.edge_features, dtype=torch.float32, device=device)
            query_features = torch.tensor(sample.query_features, dtype=torch.float32, device=device)
            outputs = model(
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                query_features=query_features,
            )
            keep_probs = torch.sigmoid(outputs["edge_keep_logits"]).detach().cpu().tolist()
            pred_types = outputs["edge_type_logits"].argmax(dim=-1).detach().cpu().tolist()
            pred_strengths = outputs["edge_strengths"].detach().cpu().tolist()
            blocks.append(
                format_readable_debug_sample(
                    sample=sample,
                    keep_probs=keep_probs,
                    pred_types=pred_types,
                    pred_strengths=pred_strengths,
                    epoch=epoch,
                    keep_threshold=keep_threshold,
                    max_edges=max_edges,
                )
            )
    return blocks


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    args: argparse.Namespace,
    history: list[dict[str, float]],
    epoch: int,
    best_metric: float,
    torch,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "history": history,
        "config": vars(args),
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the refinement model on coarse-graph samples.")
    parser.add_argument("--dataset-mode", choices=["synthetic", "maven"], default="maven", help="Training dataset mode.")
    parser.add_argument("--maven-dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="train", help="MAVEN split when dataset-mode=maven.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of synthetic training samples.")
    parser.add_argument("--limit", type=int, default=2048, help="Maximum number of MAVEN samples when dataset-mode=maven.")
    parser.add_argument("--max-events", type=int, default=16, help="Maximum events kept in each MAVEN graph sample.")
    parser.add_argument("--negative-completion-ratio", type=float, default=0.75, help="Extra non-gold completion candidates per gold edge.")
    parser.add_argument("--max-completion-edges", type=int, default=0, help="Optional cap on added completion candidates per graph.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for splits and debug sampling.")

    parser.add_argument("--hidden-dim", type=int, default=192, help="Refinement model hidden dimension.")
    parser.add_argument("--message-steps", type=int, default=4, help="Number of graph message passing steps.")
    parser.add_argument("--dropout", type=float, default=0.12, help="Dropout used by the graph refiner.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping. Use 0 to disable.")
    parser.add_argument("--keep-loss-weight", type=float, default=1.0, help="Weight for edge keep BCE loss.")
    parser.add_argument("--type-loss-weight", type=float, default=0.7, help="Weight for relation type CE loss.")
    parser.add_argument("--strength-loss-weight", type=float, default=0.3, help="Weight for strength regression loss.")
    parser.add_argument("--density-loss-weight", type=float, default=0.08, help="Weight for predicted graph density regularization.")
    parser.add_argument("--keep-pos-weight", default="auto", help="BCE positive weight: auto, none, or a float.")
    parser.add_argument("--max-pos-weight", type=float, default=12.0, help="Upper bound for auto keep positive weight.")
    parser.add_argument("--type-class-weighting", choices=["auto", "none"], default="auto", help="Use inverse-frequency weights for relation type loss.")
    parser.add_argument("--max-type-class-weight", type=float, default=4.0, help="Upper bound for auto relation type class weights.")
    parser.add_argument("--type-label-smoothing", type=float, default=0.02, help="Label smoothing for relation type loss.")

    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--log-every", type=int, default=25, help="Print one progress line every N train steps. Use 0 to disable.")
    parser.add_argument("--debug-samples", type=int, default=1, help="Validation samples printed and saved each epoch. Use 0 to disable.")
    parser.add_argument("--debug-max-edges", type=int, default=12, help="Maximum candidate edges shown per readable debug sample.")
    parser.add_argument("--debug-keep-threshold", type=float, default=0.5, help="Keep threshold used only for readable debug summaries.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "refinement"), help="Training output directory.")
    parser.add_argument("--resume-from", default=None, help="Optional training checkpoint created by this script.")
    return parser.parse_args()


def main() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from refinement_model import TemporalRelationalEdgeRefiner

    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, torch)
    device = resolve_device(args.device, torch)
    amp_enabled = (args.amp == "on" or (args.amp == "auto" and device.type == "cuda"))
    if args.amp == "on" and device.type != "cuda":
        raise ValueError("--amp on requires a CUDA device.")

    if args.dataset_mode == "synthetic":
        all_samples = generate_synthetic_refinement_samples(num_samples=args.num_samples, seed=args.seed)
    else:
        all_samples = load_maven_refinement_samples(
            dataset_path=resolve_repo_path(args.maven_dataset),
            split=args.split,
            limit=args.limit,
            max_events=args.max_events,
            negative_completion_ratio=args.negative_completion_ratio,
            max_completion_edges=args.max_completion_edges or None,
            seed=args.seed,
        )
    train_samples, validation_samples = split_samples(all_samples, args.validation_ratio, args.seed)
    train_stats = summarize_samples(train_samples)
    validation_stats = summarize_samples(validation_samples)

    dataset = RefinementTensorDataset(train_samples)
    validation_dataset = RefinementTensorDataset(validation_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TemporalRelationalEdgeRefiner(
        edge_dim=EDGE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_message_passing_steps=args.message_steps,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = make_grad_scaler(torch, amp_enabled)

    if args.keep_pos_weight == "auto":
        keep_pos_weight = compute_keep_pos_weight(train_samples, args.max_pos_weight, torch, device)
    elif args.keep_pos_weight == "none":
        keep_pos_weight = None
    else:
        keep_pos_weight = torch.tensor(float(args.keep_pos_weight), dtype=torch.float32, device=device)
    type_class_weights = (
        compute_type_class_weights(train_samples, args.max_type_class_weight, torch, device)
        if args.type_class_weighting == "auto"
        else None
    )

    keep_loss_fn = nn.BCEWithLogitsLoss(pos_weight=keep_pos_weight)
    type_loss_fn = make_type_loss(type_class_weights, args.type_label_smoothing, torch, nn)
    strength_loss_fn = nn.SmoothL1Loss(beta=0.08)
    loss_weights = {
        "keep": args.keep_loss_weight,
        "type": args.type_loss_weight,
        "strength": args.strength_loss_weight,
        "density": args.density_loss_weight,
    }

    history: list[dict[str, float]] = []
    start_epoch = 0
    best_metric = float("inf")
    if args.resume_from:
        checkpoint = torch.load(resolve_repo_path(args.resume_from), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))

    config = {
        **vars(args),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "train_stats": train_stats,
        "validation_stats": validation_stats,
        "keep_pos_weight": float(keep_pos_weight.item()) if keep_pos_weight is not None else None,
        "type_class_weights": type_class_weights.detach().cpu().tolist() if type_class_weights is not None else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "task": "coarse_graph_to_refined_graph",
    }
    save_json(output_dir / "train_config.json", config)

    print(
        " | ".join(
            [
                "refinement training",
                f"device={device}",
                f"amp={amp_enabled}",
                f"train_samples={train_stats['samples']}",
                f"train_edges={train_stats['edges']}",
                f"pos_ratio={train_stats['positive_ratio']:.3f}",
                f"hidden_dim={args.hidden_dim}",
                f"steps={args.message_steps}",
                f"params={config['parameter_count']}",
            ]
        ),
        flush=True,
    )
    print(f"relation_counts={json.dumps(train_stats['relation_counts'], ensure_ascii=False)}", flush=True)

    debug_path = output_dir / "debug_predictions.jsonl"
    readable_debug_path = output_dir / "debug_readable.log"
    if start_epoch == 0 and debug_path.exists():
        debug_path.unlink()
    if start_epoch == 0 and readable_debug_path.exists():
        readable_debug_path.unlink()

    autocast_context = make_autocast_context(torch, amp_enabled)
    train_started = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.time()
        model.train()
        totals = {
            "loss": 0.0,
            "keep_loss": 0.0,
            "type_loss": 0.0,
            "strength_loss": 0.0,
            "density_loss": 0.0,
        }
        window_totals = dict(totals)
        batch_count = 0
        window_count = 0

        for step, raw_batch in enumerate(dataloader, start=1):
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                outputs = model(
                    node_features=batch["node_features"],
                    edge_index=batch["edge_index"],
                    edge_features=batch["edge_features"],
                    query_features=batch["query_features"],
                )
                loss, keep_loss, type_loss, strength_loss, density_loss = compute_losses(
                    outputs,
                    batch,
                    keep_loss_fn,
                    type_loss_fn,
                    strength_loss_fn,
                    loss_weights,
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step_metrics = {
                "loss": float(loss.item()),
                "keep_loss": float(keep_loss.item()),
                "type_loss": float(type_loss.item()),
                "strength_loss": float(strength_loss.item()),
                "density_loss": float(density_loss.item()),
            }
            for key, value in step_metrics.items():
                totals[key] += value
                window_totals[key] += value
            batch_count += 1
            window_count += 1

            if args.log_every > 0 and (step % args.log_every == 0 or step == len(dataloader)):
                averaged = {key: value / max(window_count, 1) for key, value in window_totals.items()}
                averaged["lr"] = optimizer.param_groups[0]["lr"]
                print_metrics(
                    f"epoch {epoch + 1:03d}/{args.epochs:03d} step {step:04d}/{len(dataloader):04d}",
                    averaged,
                    time.time() - epoch_started,
                )
                window_totals = {key: 0.0 for key in window_totals}
                window_count = 0

        record: dict[str, float] = {
            "epoch": float(epoch + 1),
            "loss": totals["loss"] / max(batch_count, 1),
            "keep_loss": totals["keep_loss"] / max(batch_count, 1),
            "type_loss": totals["type_loss"] / max(batch_count, 1),
            "strength_loss": totals["strength_loss"] / max(batch_count, 1),
            "density_loss": totals["density_loss"] / max(batch_count, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        validation_record = evaluate(
            model,
            validation_dataloader,
            keep_loss_fn,
            type_loss_fn,
            strength_loss_fn,
            loss_weights,
            torch,
            device,
        )
        if validation_record is not None:
            record.update(validation_record)

        scheduler_metric = float(record.get("val_loss", record["loss"]))
        scheduler.step(scheduler_metric)
        history.append(record)

        is_best = scheduler_metric < best_metric
        if is_best:
            best_metric = scheduler_metric
            torch.save(model.state_dict(), output_dir / "refinement_model.pt")
            save_checkpoint(
                output_dir / "best_training_state.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                history,
                epoch + 1,
                best_metric,
                torch,
            )

        torch.save(model.state_dict(), output_dir / "refinement_model_latest.pt")
        save_checkpoint(
            output_dir / "latest_training_state.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            history,
            epoch + 1,
            best_metric,
            torch,
        )
        save_json(output_dir / "train_history.json", history)

        record["lr"] = float(optimizer.param_groups[0]["lr"])
        suffix = " best" if is_best else ""
        print_metrics(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} done{suffix}",
            record,
            time.time() - epoch_started,
        )
        debug_rows = collect_debug_samples(model, validation_samples, args.debug_samples, args.seed + epoch, torch, device)
        if debug_rows:
            with debug_path.open("a", encoding="utf-8") as debug_file:
                for row in debug_rows:
                    debug_file.write(json.dumps({"epoch": epoch + 1, **row}, ensure_ascii=False) + "\n")
        readable_debug_blocks = collect_readable_debug_samples(
            model=model,
            samples=validation_samples,
            debug_samples=args.debug_samples,
            seed=args.seed + epoch,
            torch=torch,
            device=device,
            epoch=epoch + 1,
            keep_threshold=args.debug_keep_threshold,
            max_edges=args.debug_max_edges,
        )
        if readable_debug_blocks:
            readable_text = "\n".join(readable_debug_blocks)
            print(readable_text, flush=True)
            with readable_debug_path.open("a", encoding="utf-8") as readable_debug_file:
                readable_debug_file.write(readable_text + "\n")

    print(f"saved outputs to {output_dir} | total_time={format_seconds(time.time() - train_started)}", flush=True)


if __name__ == "__main__":
    main()
