from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from coarse_graph_dataset import CoarseGraphPairDataset
from coarse_graph_dataset import load_maven_pair_samples
from coarse_graph_model import CoarseEdgeProposer


def collate_batch(batch):
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "relation_label": torch.stack([item["relation_label"] for item in batch], dim=0),
        "edge_score": torch.stack([item["edge_score"] for item in batch], dim=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the coarse graph proposer on MAVEN-ERE event-pair samples.")
    parser.add_argument("--dataset", default="datasets/MAVEN_ERE.zip", help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="train", help="MAVEN split name.")
    parser.add_argument("--limit", type=int, default=128, help="Maximum number of MAVEN rows.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative to positive pair ratio.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output-dir", default="outputs/coarse_graph_maven", help="Training output directory.")
    return parser.parse_args()


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
    dataset = CoarseGraphPairDataset(samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    model = CoarseEdgeProposer()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    relation_loss_fn = nn.CrossEntropyLoss()
    score_loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_relation_loss = 0.0
        total_score_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch["features"])
            relation_loss = relation_loss_fn(outputs["relation_logits"], batch["relation_label"])
            score_loss = score_loss_fn(outputs["edge_scores"], batch["edge_score"])
            loss = relation_loss + score_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_relation_loss += float(relation_loss.item())
            total_score_loss += float(score_loss.item())
            batch_count += 1

        record = {
            "epoch": float(epoch + 1),
            "loss": total_loss / max(batch_count, 1),
            "relation_loss": total_relation_loss / max(batch_count, 1),
            "score_loss": total_score_loss / max(batch_count, 1),
        }
        history.append(record)
        print(json.dumps(record))

    torch.save(model.state_dict(), output_dir / "coarse_graph_model.pt")
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "limit": args.limit,
                "negative_ratio": args.negative_ratio,
                "batch_size": args.batch_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
