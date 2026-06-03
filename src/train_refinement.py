from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from refinement_dataset import RefinementTensorDataset
from refinement_dataset import RefinementSample
from refinement_dataset import generate_synthetic_refinement_samples
from refinement_dataset import load_maven_refinement_samples
from refinement_model import TemporalRelationalEdgeRefiner


def collate_single(batch):
    return batch[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the refinement baseline on synthetic coarse-graph samples.")
    parser.add_argument("--dataset-mode", choices=["synthetic", "maven"], default="synthetic", help="Training dataset mode.")
    parser.add_argument("--maven-dataset", default="datasets/MAVEN_ERE.zip", help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="train", help="MAVEN split when dataset-mode=maven.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of synthetic training samples.")
    parser.add_argument("--limit", type=int, default=128, help="Maximum number of MAVEN samples when dataset-mode=maven.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output-dir", default="outputs/refinement_synthetic", help="Training output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_mode == "synthetic":
        train_samples = generate_synthetic_refinement_samples(num_samples=args.num_samples)
    else:
        train_samples = load_maven_refinement_samples(
            dataset_path=Path(args.maven_dataset),
            split=args.split,
            limit=args.limit,
        )
    dataset = RefinementTensorDataset(train_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

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
        history.append(record)
        print(json.dumps(record))

    torch.save(model.state_dict(), output_dir / "refinement_model.pt")
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "dataset_mode": args.dataset_mode,
                "num_samples": args.num_samples,
                "limit": args.limit,
                "split": args.split,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
