from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_graph import CoarseCausalEdge
from causal_graph import CoarseCausalGraph
from coarse_graph_dataset import load_coarse_graph
from refinement_dataset import ID_TO_RELATION
from refinement_dataset import EDGE_FEATURE_DIM
from refinement_dataset import load_refinement_sample_from_coarse_graph
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refinement inference on a coarse graph JSON file.")
    parser.add_argument("--coarse-graph", required=True, help="Path to a coarse graph JSON file.")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "outputs" / "refinement" / "refinement_model.pt"), help="Trained refinement model path.")
    parser.add_argument("--hidden-dim", type=int, default=192, help="Model hidden dimension. Defaults to train_config.json next to model path, then 128.")
    parser.add_argument("--message-steps", type=int, default=4, help="Model message passing steps. Defaults to train_config.json next to model path, then 3.")
    parser.add_argument("--edge-dim", type=int, default=192, help="Model edge feature dimension. Defaults to train_config.json next to model path, then current dataset schema.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Model dropout. Defaults to train_config.json next to model path, then 0.0 for inference.")
    parser.add_argument("--include-completion-candidates", action="store_true", help="Also score plausible non-coarse event pairs so refinement can add missing edges.")
    parser.add_argument("--max-completion-edges", type=int, default=64, help="Maximum non-coarse candidate edges added when completion is enabled.")
    parser.add_argument("--keep-threshold", type=float, default=0.5, help="Minimum keep probability for refined edges.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def load_model_config(
    model_path: Path,
    hidden_dim: int | None,
    message_steps: int | None,
    edge_dim: int | None,
    dropout: float | None,
) -> tuple[int, int, int, float]:
    config_path = model_path.parent / "train_config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved_hidden_dim = hidden_dim if hidden_dim is not None else int(config.get("hidden_dim", 192))
    resolved_message_steps = message_steps if message_steps is not None else int(config.get("message_steps", 4))
    resolved_edge_dim = edge_dim if edge_dim is not None else int(config.get("edge_feature_dim", EDGE_FEATURE_DIM))
    resolved_dropout = dropout if dropout is not None else float(config.get("dropout", 0.0))
    return resolved_hidden_dim, resolved_message_steps, resolved_edge_dim, resolved_dropout


def load_model_state(model_path: Path, torch):
    payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def build_refined_graph(
    coarse_graph: CoarseCausalGraph,
    edge_descriptions: list[dict[str, object]],
    keep_probs: list[float],
    type_predictions: list[int],
    strength_predictions: list[float],
    keep_threshold: float,
) -> CoarseCausalGraph:
    refined_edges: list[CoarseCausalEdge] = []
    coarse_lookup = {
        (edge.source_event_id, edge.target_event_id): edge
        for edge in coarse_graph.edges
    }
    event_lookup = {event.event_id: event for event in coarse_graph.events}
    for edge_index, (edge_desc, keep_prob, type_prediction, strength_prediction) in enumerate(zip(
        edge_descriptions,
        keep_probs,
        type_predictions,
        strength_predictions,
    )):
        if keep_prob < keep_threshold:
            continue
        source_event_id = str(edge_desc.get("source_event_id", ""))
        target_event_id = str(edge_desc.get("target_event_id", ""))
        edge = coarse_lookup.get((source_event_id, target_event_id))
        source_event = event_lookup.get(source_event_id)
        target_event = event_lookup.get(target_event_id)
        if source_event is None or target_event is None:
            continue
        relation_type = ID_TO_RELATION.get(int(type_prediction), str(edge_desc.get("coarse_relation_type", "precedes")))
        refined_edges.append(
            CoarseCausalEdge(
                edge_id=edge.edge_id if edge is not None else f"refined_added_edge_{edge_index}",
                source_event_id=source_event_id,
                target_event_id=target_event_id,
                relation_type=relation_type,
                score=round(float(strength_prediction), 4),
                evidence=edge.evidence if edge is not None else source_event.evidence + target_event.evidence,
                feature_scores=edge.feature_scores if edge is not None else {},
                metadata={
                    **(edge.metadata if edge is not None else {}),
                    "refined_keep_probability": round(float(keep_prob), 4),
                    "refined_from_model": True,
                    "candidate_source": edge_desc.get("candidate_source", "coarse"),
                },
            )
        )
    return CoarseCausalGraph(
        query=coarse_graph.query,
        documents=coarse_graph.documents,
        events=coarse_graph.events,
        edges=refined_edges,
        trace=coarse_graph.trace,
    )


def summarize_edges(
    edge_descriptions: list[dict[str, object]],
    keep_probs: list[float],
    type_predictions: list[int],
    strength_predictions: list[float],
) -> list[dict[str, object]]:
    preview: list[dict[str, object]] = []
    for edge_desc, keep_prob, type_prediction, strength_prediction in zip(
        edge_descriptions,
        keep_probs,
        type_predictions,
        strength_predictions,
    ):
        preview.append(
            {
                "source_event_id": edge_desc.get("source_event_id", ""),
                "target_event_id": edge_desc.get("target_event_id", ""),
                "source_text": edge_desc.get("source_text", ""),
                "target_text": edge_desc.get("target_text", ""),
                "candidate_source": edge_desc.get("candidate_source", "coarse"),
                "coarse_relation_type": edge_desc.get("coarse_relation_type", ""),
                "pred_relation_type": ID_TO_RELATION.get(int(type_prediction), str(edge_desc.get("coarse_relation_type", "precedes"))),
                "coarse_score": edge_desc.get("coarse_score", 0.0),
                "pred_strength": round(float(strength_prediction), 4),
                "keep_probability": round(float(keep_prob), 4),
            }
        )
    return preview


def summarize_frontier_nodes(coarse_graph: CoarseCausalGraph, frontier_scores: list[float]) -> list[dict[str, object]]:
    ranked = []
    for event, score in zip(coarse_graph.events, frontier_scores):
        ranked.append(
            {
                "event_id": event.event_id,
                "text": event.text,
                "frontier_score": round(float(score), 4),
            }
        )
    ranked.sort(key=lambda item: item["frontier_score"], reverse=True)
    return ranked


def main() -> None:
    import torch
    from refinement_model import TemporalRelationalEdgeRefiner

    args = parse_args()
    coarse_graph_path = resolve_repo_path(args.coarse_graph)
    payload = json.loads(coarse_graph_path.read_text(encoding="utf-8"))
    coarse_graph_dict = payload.get("coarse_graph", payload)
    sample = load_refinement_sample_from_coarse_graph(
        coarse_graph_path,
        include_completion_candidates=args.include_completion_candidates,
        max_completion_edges=args.max_completion_edges if args.include_completion_candidates else None,
    )

    node_features = torch.tensor(sample.node_features, dtype=torch.float32)
    edge_index = torch.tensor(sample.edge_index, dtype=torch.long)
    edge_features = torch.tensor(sample.edge_features, dtype=torch.float32)
    query_features = torch.tensor(sample.query_features, dtype=torch.float32)

    model_path = resolve_repo_path(args.model_path)
    hidden_dim, message_steps, edge_dim, dropout = load_model_config(
        model_path,
        args.hidden_dim,
        args.message_steps,
        args.edge_dim,
        args.dropout,
    )
    model = TemporalRelationalEdgeRefiner(
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_message_passing_steps=message_steps,
        dropout=dropout,
    )
    model.load_state_dict(load_model_state(model_path, torch))
    model.eval()

    with torch.no_grad():
        outputs = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            query_features=query_features,
        )

    keep_probs = torch.sigmoid(outputs["edge_keep_logits"]).tolist()
    type_predictions = outputs["edge_type_logits"].argmax(dim=-1).tolist()
    strength_predictions = outputs["edge_strengths"].tolist()
    frontier_scores = outputs["frontier_scores"].tolist()

    coarse_graph = load_coarse_graph(resolve_repo_path(args.coarse_graph))
    edge_descriptions = list(sample.metadata.get("edge_descriptions", []))
    refined_graph = build_refined_graph(
        coarse_graph=coarse_graph,
        edge_descriptions=edge_descriptions,
        keep_probs=keep_probs,
        type_predictions=type_predictions,
        strength_predictions=strength_predictions,
        keep_threshold=args.keep_threshold,
    )
    refined_edges_preview = summarize_edges(
        edge_descriptions=edge_descriptions,
        keep_probs=keep_probs,
        type_predictions=type_predictions,
        strength_predictions=strength_predictions,
    )
    frontier_nodes = summarize_frontier_nodes(coarse_graph, frontier_scores)

    result = {
        "sample_id": sample.sample_id,
        "coarse_graph": coarse_graph_dict,
        "refinement_output": {
            "edge_keep_probabilities": keep_probs,
            "edge_type_predictions": type_predictions,
            "edge_strengths": strength_predictions,
            "frontier_scores": frontier_scores,
        },
        "refined_edges_preview": refined_edges_preview,
        "frontier_nodes": frontier_nodes,
        "refined_graph": refined_graph.to_dict(),
    }

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
