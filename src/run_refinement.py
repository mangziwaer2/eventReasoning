from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from refinement_dataset import coarse_graph_to_refinement_sample
from refinement_dataset import export_mirai_refinement_sample
from refinement_model import TemporalRelationalEdgeRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refinement inference on a MIRAI-derived coarse graph sample.")
    parser.add_argument("--dataset", default="datasets/MIRAI_data.zip", help="Path to MIRAI zip file.")
    parser.add_argument("--query-id", default="1", help="MIRAI QueryId.")
    parser.add_argument("--split", default="test", help="MIRAI split name.")
    parser.add_argument("--event-extractor", default="rule", help="Event extractor backend name.")
    parser.add_argument("--model-path", default="outputs/refinement_synthetic/refinement_model.pt", help="Trained refinement model path.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_mirai_refinement_sample(
        dataset_path=Path(args.dataset),
        query_id=args.query_id,
        split=args.split,
        event_extractor_name=args.event_extractor,
    )
    sample_dict = payload["refinement_sample"]
    node_features = torch.tensor(sample_dict["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(sample_dict["edge_index"], dtype=torch.long)
    edge_features = torch.tensor(sample_dict["edge_features"], dtype=torch.float32)
    query_features = torch.tensor(sample_dict["query_features"], dtype=torch.float32)

    model = TemporalRelationalEdgeRefiner()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        outputs = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            query_features=query_features,
        )

    result = {
        "mirai_query": payload["mirai_query"],
        "refinement_sample": sample_dict,
        "refinement_output": {
            "edge_keep_probabilities": torch.sigmoid(outputs["edge_keep_logits"]).tolist(),
            "edge_type_predictions": outputs["edge_type_logits"].argmax(dim=-1).tolist(),
            "edge_strengths": outputs["edge_strengths"].tolist(),
            "frontier_scores": outputs["frontier_scores"].tolist(),
        },
    }

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
