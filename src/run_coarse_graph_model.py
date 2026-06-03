from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coarse_graph_dataset import ID_TO_RELATION
from coarse_graph_dataset import load_maven_pair_samples
from coarse_graph_model import CoarseEdgeProposer
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run coarse graph proposer inference on MAVEN-ERE event-pair samples.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="valid", help="MAVEN split name.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of MAVEN rows.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative to positive pair ratio.")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "outputs" / "coarse_graph_maven" / "coarse_graph_model.pt"), help="Trained model path.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_maven_pair_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit,
        negative_ratio=args.negative_ratio,
    )
    if not samples:
        raise RuntimeError("No coarse graph samples were loaded for inference.")

    model = CoarseEdgeProposer()
    model.load_state_dict(torch.load(resolve_repo_path(args.model_path), map_location="cpu"))
    model.eval()

    preview = []
    with torch.no_grad():
        for sample in samples[:16]:
            features = torch.tensor(sample.features, dtype=torch.float32).unsqueeze(0)
            outputs = model(features)
            relation_id = int(outputs["relation_logits"].argmax(dim=-1).item())
            edge_score = float(outputs["edge_scores"].item())
            preview.append(
                {
                    "sample_id": sample.sample_id,
                    "event_a_text": sample.event_a_text,
                    "event_b_text": sample.event_b_text,
                    "gold_relation": ID_TO_RELATION.get(sample.relation_label, "none"),
                    "predicted_relation": ID_TO_RELATION.get(relation_id, "none"),
                    "gold_edge_score": sample.edge_score,
                    "predicted_edge_score": edge_score,
                }
            )

    payload = {"predictions": preview}
    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
