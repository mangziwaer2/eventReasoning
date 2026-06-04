from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from refinement_dataset import RefinementTensorDataset
from refinement_dataset import RefinementSample
from refinement_dataset import generate_synthetic_refinement_samples
from refinement_dataset import load_maven_refinement_samples
from refinement_model import TemporalRelationalEdgeRefiner
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def collate_single(batch):
    return batch[0]


def split_samples(samples: list[RefinementSample], validation_ratio: float, seed: int) -> tuple[list[RefinementSample], list[RefinementSample]]:
    if validation_ratio <= 0 or len(samples) < 2:
        return samples, []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio))
    if validation_size >= len(shuffled):
        validation_size = len(shuffled) - 1
    return shuffled[validation_size:], shuffled[:validation_size]


def evaluate(model, dataloader, keep_loss_fn, type_loss_fn, strength_loss_fn):
    model.eval()
    total_loss = 0.0
    total_keep_loss = 0.0
    total_type_loss = 0.0
    total_strength_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                node_features=batch["node_features"],
                edge_index=batch["edge_index"],
                edge_features=batch["edge_features"],
                query_features=batch["query_features"],
            )
            keep_loss = keep_loss_fn(outputs["edge_keep_logits"], batch["edge_labels"])
            type_loss = type_loss_fn(outputs["edge_type_logits"], batch["edge_type_labels"])
            strength_loss = strength_loss_fn(outputs["edge_strengths"], batch["edge_strengths"])
            loss = keep_loss + type_loss + strength_loss
            total_loss += float(loss.item())
            total_keep_loss += float(keep_loss.item())
            total_type_loss += float(type_loss.item())
            total_strength_loss += float(strength_loss.item())
            batch_count += 1
    if batch_count == 0:
        return None
    return {
        "val_loss": total_loss / batch_count,
        "val_keep_loss": total_keep_loss / batch_count,
        "val_type_loss": total_type_loss / batch_count,
        "val_strength_loss": total_strength_loss / batch_count,
    }


def print_debug_samples(model, samples: list[RefinementSample], debug_samples: int, seed: int) -> None:
    if not samples or debug_samples <= 0:
        return
    rng = random.Random(seed)
    chosen = rng.sample(samples, min(debug_samples, len(samples)))
    model.eval()
    with torch.no_grad():
        for sample in chosen:
            node_features = torch.tensor(sample.node_features, dtype=torch.float32)
            edge_index = torch.tensor(sample.edge_index, dtype=torch.long)
            edge_features = torch.tensor(sample.edge_features, dtype=torch.float32)
            query_features = torch.tensor(sample.query_features, dtype=torch.float32)
            outputs = model(
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                query_features=query_features,
            )
            keep_probs = torch.sigmoid(outputs["edge_keep_logits"]).tolist()
            pred_types = outputs["edge_type_logits"].argmax(dim=-1).tolist()
            pred_strengths = outputs["edge_strengths"].tolist()
            edge_descriptions = sample.metadata.get("edge_descriptions", [])
            for idx, edge_desc in enumerate(edge_descriptions[: min(3, len(edge_descriptions))]):
                print(
                    json.dumps(
                        {
                            "debug_stage": "refinement_validation_sample",
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
                        },
                        ensure_ascii=False,
                    )
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the refinement baseline on synthetic coarse-graph samples.")
    parser.add_argument("--dataset-mode", choices=["synthetic", "maven"], default="maven", help="Training dataset mode.")
    parser.add_argument("--maven-dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="train", help="MAVEN split when dataset-mode=maven.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of synthetic training samples.")
    parser.add_argument("--limit", type=int, default=128, help="Maximum number of MAVEN samples when dataset-mode=maven.")
    parser.add_argument("--max-events", type=int, default=12, help="Maximum events kept in each MAVEN graph sample.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--debug-samples", type=int, default=2, help="Number of validation samples printed each epoch.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for splits and debug sampling.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "refinement"), help="Training output directory.")
    return parser.parse_args()


def main() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_mode == "synthetic":
        all_samples = generate_synthetic_refinement_samples(num_samples=args.num_samples, seed=args.seed)
    else:
        all_samples = load_maven_refinement_samples(
            dataset_path=resolve_repo_path(args.maven_dataset),
            split=args.split,
            limit=args.limit,
            max_events=args.max_events,
            seed=args.seed,
        )
    train_samples, validation_samples = split_samples(all_samples, args.validation_ratio, args.seed)
    dataset = RefinementTensorDataset(train_samples)
    validation_dataset = RefinementTensorDataset(validation_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=collate_single)

    model = TemporalRelationalEdgeRefiner()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    keep_loss_fn = nn.BCEWithLogitsLoss()
    type_loss_fn = nn.CrossEntropyLoss()
    strength_loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_keep_loss = 0.0
        total_type_loss = 0.0
        total_strength_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                node_features=batch["node_features"],
                edge_index=batch["edge_index"],
                edge_features=batch["edge_features"],
                query_features=batch["query_features"],
            )
            keep_logits = outputs["edge_keep_logits"]
            type_logits = outputs["edge_type_logits"]
            strength_preds = outputs["edge_strengths"]

            keep_loss = keep_loss_fn(keep_logits, batch["edge_labels"])
            type_loss = type_loss_fn(type_logits, batch["edge_type_labels"])
            strength_loss = strength_loss_fn(strength_preds, batch["edge_strengths"])
            loss = keep_loss + type_loss + strength_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_keep_loss += float(keep_loss.item())
            total_type_loss += float(type_loss.item())
            total_strength_loss += float(strength_loss.item())
            batch_count += 1

        record = {
            "epoch": float(epoch + 1),
            "loss": total_loss / max(batch_count, 1),
            "keep_loss": total_keep_loss / max(batch_count, 1),
            "type_loss": total_type_loss / max(batch_count, 1),
            "strength_loss": total_strength_loss / max(batch_count, 1),
        }
        validation_record = evaluate(model, validation_dataloader, keep_loss_fn, type_loss_fn, strength_loss_fn)
        if validation_record is not None:
            record.update(validation_record)
        history.append(record)
        print(json.dumps(record))
        print_debug_samples(model, validation_samples, args.debug_samples, args.seed + epoch)

    torch.save(model.state_dict(), output_dir / "refinement_model.pt")
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "dataset_mode": args.dataset_mode,
                "num_samples": args.num_samples,
                "limit": args.limit,
                "max_events": args.max_events,
                "split": args.split,
                "validation_ratio": args.validation_ratio,
                "debug_samples": args.debug_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
