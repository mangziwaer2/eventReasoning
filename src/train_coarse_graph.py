from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from coarse_graph_dataset import CoarseGraphPairDataset
from coarse_graph_dataset import ID_TO_RELATION
from coarse_graph_dataset import CoarseGraphPairSample
from coarse_graph_dataset import load_maven_pair_samples
from coarse_graph_model import CoarseEdgeProposer
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def collate_batch(batch):
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "relation_label": torch.stack([item["relation_label"] for item in batch], dim=0),
        "edge_score": torch.stack([item["edge_score"] for item in batch], dim=0),
    }


def split_samples(samples: list[CoarseGraphPairSample], validation_ratio: float, seed: int) -> tuple[list[CoarseGraphPairSample], list[CoarseGraphPairSample]]:
    if validation_ratio <= 0 or len(samples) < 2:
        return samples, []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio))
    if validation_size >= len(shuffled):
        validation_size = len(shuffled) - 1
    return shuffled[validation_size:], shuffled[:validation_size]


def evaluate(model, dataloader, relation_loss_fn, score_loss_fn):
    model.eval()
    total_loss = 0.0
    total_relation_loss = 0.0
    total_score_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch["features"])
            relation_loss = relation_loss_fn(outputs["relation_logits"], batch["relation_label"])
            score_loss = score_loss_fn(outputs["edge_scores"], batch["edge_score"])
            loss = relation_loss + score_loss
            total_loss += float(loss.item())
            total_relation_loss += float(relation_loss.item())
            total_score_loss += float(score_loss.item())
            batch_count += 1
    if batch_count == 0:
        return None
    return {
        "val_loss": total_loss / batch_count,
        "val_relation_loss": total_relation_loss / batch_count,
        "val_score_loss": total_score_loss / batch_count,
    }


def print_debug_samples(model, samples: list[CoarseGraphPairSample], debug_samples: int, seed: int) -> None:
    if not samples or debug_samples <= 0:
        return
    rng = random.Random(seed)
    chosen = rng.sample(samples, min(debug_samples, len(samples)))
    model.eval()
    with torch.no_grad():
        for sample in chosen:
            features = torch.tensor(sample.features, dtype=torch.float32).unsqueeze(0)
            outputs = model(features)
            pred_relation_id = int(outputs["relation_logits"].argmax(dim=-1).item())
            pred_score = float(outputs["edge_scores"].item())
            print(
                json.dumps(
                    {
                        "debug_stage": "coarse_validation_sample",
                        "sample_id": sample.sample_id,
                        "query": sample.query_text,
                        "event_a": sample.event_a_text,
                        "event_b": sample.event_b_text,
                        "gold_relation": ID_TO_RELATION.get(sample.relation_label, "none"),
                        "pred_relation": ID_TO_RELATION.get(pred_relation_id, "none"),
                        "gold_score": sample.edge_score,
                        "pred_score": pred_score,
                    },
                    ensure_ascii=False,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the coarse graph proposer on MAVEN-ERE event-pair samples.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="train", help="MAVEN split name.")
    parser.add_argument("--limit", type=int, default=128, help="Maximum number of MAVEN rows.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative to positive pair ratio.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--debug-samples", type=int, default=2, help="Number of validation samples printed each epoch.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for splits and debug sampling.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "coarse_graph_maven"), help="Training output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_maven_pair_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    train_samples, validation_samples = split_samples(samples, args.validation_ratio, args.seed)
    train_dataset = CoarseGraphPairDataset(train_samples)
    validation_dataset = CoarseGraphPairDataset(validation_samples)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

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
        validation_record = evaluate(model, validation_dataloader, relation_loss_fn, score_loss_fn)
        if validation_record is not None:
            record.update(validation_record)
        history.append(record)
        print(json.dumps(record))
        print_debug_samples(model, validation_samples, args.debug_samples, args.seed + epoch)

    torch.save(model.state_dict(), output_dir / "coarse_graph_model.pt")
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "limit": args.limit,
                "negative_ratio": args.negative_ratio,
                "batch_size": args.batch_size,
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
