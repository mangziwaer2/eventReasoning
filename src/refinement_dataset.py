from __future__ import annotations

import argparse
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import io

import torch
from torch.utils.data import Dataset

from causal_graph import CoarseCausalGraph
from causal_graph import CoarseCausalEdge
from causal_graph import EvidenceSpan
from causal_graph import EventNode
from causal_graph import GraphBuildTrace
from causal_graph import NewsDocument
from causal_graph import QuerySpec
from coarse_graph_builder import CoarseCausalGraphBuilder
from mirai_dataset import export_mirai_query_snapshot
from mirai_dataset import get_mirai_query_by_id
from mirai_dataset import load_mirai_news_for_docids
from query_causal_graph import QueryCausalGraphBuilder


RELATION_TO_ID = {
    "precedes": 0,
    "causes": 1,
    "escalates": 2,
    "mitigates": 3,
}


@dataclass(slots=True)
class RefinementSample:
    sample_id: str
    node_features: list[list[float]]
    edge_index: list[list[int]]
    edge_features: list[list[float]]
    edge_labels: list[int]
    edge_type_labels: list[int]
    edge_strengths: list[float]
    query_features: list[float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
            "edge_labels": self.edge_labels,
            "edge_type_labels": self.edge_type_labels,
            "edge_strengths": self.edge_strengths,
            "query_features": self.query_features,
            "metadata": self.metadata,
        }


def _safe_feature(edge: CoarseCausalEdge, key: str) -> float:
    return float(edge.feature_scores.get(key, 0.0))


def _event_node_feature(event: EventNode, query_text: str) -> list[float]:
    participants_count = float(len(event.participants))
    token_count = float(len(event.normalized_text.split()))
    sentence_index = float(event.sentence_index)
    confidence = float(event.confidence)
    query_overlap = 1.0 if any(token in event.normalized_text for token in query_text.lower().split()) else 0.0
    trigger_length = float(len(str(event.metadata.get("trigger", ""))))
    return [
        participants_count,
        token_count,
        sentence_index,
        confidence,
        query_overlap,
        trigger_length,
    ]


def _edge_feature(edge: CoarseCausalEdge) -> list[float]:
    relation_id = float(RELATION_TO_ID.get(edge.relation_type, 0))
    return [
        float(edge.score),
        relation_id,
        _safe_feature(edge, "temporal_score"),
        _safe_feature(edge, "entity_overlap_score"),
        _safe_feature(edge, "lexical_support_score"),
        _safe_feature(edge, "marker_score"),
        _safe_feature(edge, "query_alignment_score"),
    ]


def coarse_graph_to_refinement_sample(sample_id: str, graph: CoarseCausalGraph) -> RefinementSample:
    node_id_to_index = {event.event_id: index for index, event in enumerate(graph.events)}
    node_features = [_event_node_feature(event, graph.query.text) for event in graph.events]

    edge_index: list[list[int]] = []
    edge_features: list[list[float]] = []
    edge_labels: list[int] = []
    edge_type_labels: list[int] = []
    edge_strengths: list[float] = []

    for edge in graph.edges:
        if edge.source_event_id not in node_id_to_index or edge.target_event_id not in node_id_to_index:
            continue
        edge_index.append([node_id_to_index[edge.source_event_id], node_id_to_index[edge.target_event_id]])
        edge_features.append(_edge_feature(edge))
        edge_labels.append(int(edge.score >= 0.65))
        edge_type_labels.append(RELATION_TO_ID.get(edge.relation_type, 0))
        edge_strengths.append(float(edge.score))

    query_features = [
        float(len(graph.query.focus_entities)),
        float(len(graph.documents)),
        float(len(graph.events)),
        float(len(graph.edges)),
    ]

    return RefinementSample(
        sample_id=sample_id,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_labels=edge_labels,
        edge_type_labels=edge_type_labels,
        edge_strengths=edge_strengths,
        query_features=query_features,
        metadata={
            "query_id": graph.query.query_id,
            "cutoff_time": graph.query.cutoff_time,
        },
    )


class RefinementTensorDataset(Dataset):
    def __init__(self, samples: list[RefinementSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "node_features": torch.tensor(sample.node_features, dtype=torch.float32),
            "edge_index": torch.tensor(sample.edge_index, dtype=torch.long),
            "edge_features": torch.tensor(sample.edge_features, dtype=torch.float32),
            "edge_labels": torch.tensor(sample.edge_labels, dtype=torch.float32),
            "edge_type_labels": torch.tensor(sample.edge_type_labels, dtype=torch.long),
            "edge_strengths": torch.tensor(sample.edge_strengths, dtype=torch.float32),
            "query_features": torch.tensor(sample.query_features, dtype=torch.float32),
            "metadata": sample.metadata,
        }


def generate_synthetic_refinement_samples(num_samples: int = 32, seed: int = 7) -> list[RefinementSample]:
    random.seed(seed)
    samples: list[RefinementSample] = []

    for sample_index in range(num_samples):
        node_count = random.randint(4, 8)
        edge_count = random.randint(node_count - 1, min(node_count * 2, 12))

        node_features: list[list[float]] = []
        for node_index in range(node_count):
            node_features.append(
                [
                    random.uniform(0, 6),
                    random.uniform(3, 18),
                    float(node_index),
                    random.uniform(0.3, 0.95),
                    random.choice([0.0, 1.0]),
                    random.uniform(3, 10),
                ]
            )

        edge_index: list[list[int]] = []
        edge_features: list[list[float]] = []
        edge_labels: list[int] = []
        edge_type_labels: list[int] = []
        edge_strengths: list[float] = []

        used_pairs: set[tuple[int, int]] = set()
        while len(edge_index) < edge_count:
            source = random.randint(0, node_count - 2)
            target = random.randint(source + 1, node_count - 1)
            if (source, target) in used_pairs:
                continue
            used_pairs.add((source, target))
            temporal_score = random.uniform(0.5, 1.0)
            entity_score = random.uniform(0.0, 1.0)
            lexical_score = random.uniform(0.0, 1.0)
            marker_score = random.choice([0.0, 1.0])
            query_score = random.uniform(0.0, 1.0)
            coarse_score = 0.34 * temporal_score + 0.22 * entity_score + 0.18 * lexical_score + 0.16 * marker_score + 0.10 * query_score
            relation_id = random.choice([0.0, 1.0, 2.0, 3.0])

            edge_index.append([source, target])
            edge_features.append(
                [
                    coarse_score,
                    relation_id,
                    temporal_score,
                    entity_score,
                    lexical_score,
                    marker_score,
                    query_score,
                ]
            )
            edge_labels.append(int(coarse_score >= 0.62))
            edge_type_labels.append(int(relation_id))
            edge_strengths.append(coarse_score)

        samples.append(
            RefinementSample(
                sample_id=f"synthetic_{sample_index}",
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                edge_labels=edge_labels,
                edge_type_labels=edge_type_labels,
                edge_strengths=edge_strengths,
                query_features=[
                    random.uniform(1, 3),
                    random.uniform(2, 6),
                    float(node_count),
                    float(edge_count),
                ],
                metadata={"synthetic": True},
            )
        )

    return samples


def export_mirai_refinement_sample(
    dataset_path: Path,
    query_id: str,
    split: str = "test",
    event_extractor_name: str = "rule",
    max_docs: int = 6,
    max_events_per_doc: int = 6,
) -> dict[str, Any]:
    example = get_mirai_query_by_id(dataset_path, query_id=query_id, split=split)
    documents = load_mirai_news_for_docids(dataset_path, example.docids)
    graph_builder = QueryCausalGraphBuilder(
        max_docs=max_docs,
        max_events_per_doc=max_events_per_doc,
        event_extractor_name=event_extractor_name,
    )
    query = example.build_query_spec()
    local_graph = graph_builder.build(query, documents)
    coarse_builder = CoarseCausalGraphBuilder()
    coarse_graph = coarse_builder.build(query, local_graph.documents, local_graph.events, local_graph.trace)
    sample = coarse_graph_to_refinement_sample(sample_id=f"mirai_{query_id}", graph=coarse_graph)
    return {
        "mirai_query": json.loads(export_mirai_query_snapshot(example)),
        "refinement_sample": sample.to_dict(),
    }


def _normalize_maven_sentence(sentence) -> str:
    if isinstance(sentence, list):
        return " ".join(sentence)
    return str(sentence)


def _read_maven_rows(zip_path: Path, split: str) -> list[dict[str, Any]]:
    member_name = f"MAVEN_ERE/{split}.jsonl"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            text_stream = io.TextIOWrapper(handle, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _maven_relation_to_type(relation_name: str) -> str:
    if relation_name == "CAUSE":
        return "causes"
    if relation_name == "PRECONDITION":
        return "precedes"
    if relation_name == "BEFORE":
        return "precedes"
    if relation_name in {"OVERLAP", "SIMULTANEOUS", "CONTAINS", "ENDS-ON", "BEGINS-ON"}:
        return "precedes"
    return "precedes"


def maven_row_to_gold_graph(row: dict[str, Any]) -> CoarseCausalGraph:
    query = QuerySpec(
        query_id=str(row["id"]),
        text=row.get("title", "MAVEN-ERE sample"),
        cutoff_time=None,
        focus_entities=[],
        metadata={"dataset": "MAVEN-ERE"},
    )
    documents = [
        NewsDocument(
            document_id=str(row["id"]),
            title=row.get("title", ""),
            text="\n".join(_normalize_maven_sentence(sentence) for sentence in row.get("sentences", [])),
            source="MAVEN-ERE",
        )
    ]

    events: list[EventNode] = []
    mention_id_to_event_id: dict[str, str] = {}
    for event_index, event in enumerate(row.get("events", [])):
        mentions = event.get("mention", [])
        if not mentions:
            continue
        first_mention = mentions[0]
        sent_id = int(first_mention.get("sent_id", 0))
        sentence_text = _normalize_maven_sentence(row.get("sentences", [])[sent_id])
        trigger_word = str(first_mention.get("trigger_word", "")).strip()
        event_id = f"maven_e{event_index}"
        mention_id_to_event_id[event["id"]] = event_id
        events.append(
            EventNode(
                event_id=event_id,
                text=sentence_text,
                normalized_text=trigger_word.lower() if trigger_word else sentence_text.lower(),
                document_id=str(row["id"]),
                sentence_index=sent_id,
                participants=[],
                node_type="observed",
                confidence=1.0,
                evidence=[
                    EvidenceSpan(
                        document_id=str(row["id"]),
                        sentence_index=sent_id,
                        text=sentence_text,
                    )
                ],
                metadata={
                    "trigger": trigger_word,
                    "event_type": event.get("type", ""),
                    "event_type_id": event.get("type_id", -1),
                    "source_event_id": event.get("id", ""),
                },
            )
        )

    edges: list[CoarseCausalEdge] = []
    edge_index = 0

    for relation_name, pairs in row.get("causal_relations", {}).items():
        for source_id, target_id in pairs:
            if source_id not in mention_id_to_event_id or target_id not in mention_id_to_event_id:
                continue
            edges.append(
                CoarseCausalEdge(
                    edge_id=f"gold_edge_{edge_index}",
                    source_event_id=mention_id_to_event_id[source_id],
                    target_event_id=mention_id_to_event_id[target_id],
                    relation_type=_maven_relation_to_type(relation_name),
                    score=1.0,
                    evidence=[],
                    feature_scores={"gold_relation": 1.0},
                    metadata={"source_relation": relation_name, "gold": True},
                )
            )
            edge_index += 1

    for relation_name, pairs in row.get("temporal_relations", {}).items():
        for source_id, target_id in pairs:
            if source_id not in mention_id_to_event_id or target_id not in mention_id_to_event_id:
                continue
            edges.append(
                CoarseCausalEdge(
                    edge_id=f"gold_edge_{edge_index}",
                    source_event_id=mention_id_to_event_id[source_id],
                    target_event_id=mention_id_to_event_id[target_id],
                    relation_type=_maven_relation_to_type(relation_name),
                    score=0.85,
                    evidence=[],
                    feature_scores={"gold_relation": 1.0},
                    metadata={"source_relation": relation_name, "gold": True},
                )
            )
            edge_index += 1

    return CoarseCausalGraph(
        query=query,
        documents=documents,
        events=events,
        edges=_dedupe_coarse_edges(edges),
        trace=GraphBuildTrace(),
    )


def _dedupe_coarse_edges(edges: list[CoarseCausalEdge]) -> list[CoarseCausalEdge]:
    deduped: list[CoarseCausalEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = (edge.source_event_id, edge.target_event_id, edge.relation_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def perturb_gold_graph_to_coarse(
    graph: CoarseCausalGraph,
    drop_rate: float = 0.25,
    flip_rate: float = 0.15,
    add_negative_rate: float = 0.25,
    seed: int = 7,
) -> CoarseCausalGraph:
    random.seed(seed)
    kept_edges: list[CoarseCausalEdge] = []

    relation_choices = ["precedes", "causes", "escalates", "mitigates"]
    for edge in graph.edges:
        if random.random() < drop_rate:
            continue
        relation_type = edge.relation_type
        if random.random() < flip_rate:
            relation_type = random.choice([item for item in relation_choices if item != relation_type])
        score = random.uniform(0.35, 0.9)
        kept_edges.append(
            CoarseCausalEdge(
                edge_id=edge.edge_id,
                source_event_id=edge.source_event_id,
                target_event_id=edge.target_event_id,
                relation_type=relation_type,
                score=round(score, 4),
                evidence=edge.evidence,
                feature_scores={
                    "temporal_score": random.uniform(0.3, 1.0),
                    "entity_overlap_score": random.uniform(0.0, 1.0),
                    "lexical_support_score": random.uniform(0.0, 1.0),
                    "marker_score": random.choice([0.0, 1.0]),
                    "query_alignment_score": random.uniform(0.0, 1.0),
                },
                metadata={"gold": False, "derived_from_gold": True},
            )
        )

    event_ids = [event.event_id for event in graph.events]
    gold_pairs = {(edge.source_event_id, edge.target_event_id) for edge in graph.edges}
    target_negatives = max(1, int(len(graph.edges) * add_negative_rate))
    negative_index = 0
    while negative_index < target_negatives and len(event_ids) >= 2:
        source_id, target_id = random.sample(event_ids, 2)
        if (source_id, target_id) in gold_pairs:
            continue
        kept_edges.append(
            CoarseCausalEdge(
                edge_id=f"noisy_edge_{negative_index}",
                source_event_id=source_id,
                target_event_id=target_id,
                relation_type=random.choice(relation_choices),
                score=round(random.uniform(0.2, 0.55), 4),
                evidence=[],
                feature_scores={
                    "temporal_score": random.uniform(0.0, 0.8),
                    "entity_overlap_score": random.uniform(0.0, 0.6),
                    "lexical_support_score": random.uniform(0.0, 0.5),
                    "marker_score": random.choice([0.0, 1.0]),
                    "query_alignment_score": random.uniform(0.0, 0.4),
                },
                metadata={"gold": False, "synthetic_negative": True},
            )
        )
        negative_index += 1

    return CoarseCausalGraph(
        query=graph.query,
        documents=graph.documents,
        events=graph.events,
        edges=_dedupe_coarse_edges(kept_edges),
        trace=graph.trace,
    )


def gold_and_coarse_graph_to_refinement_sample(
    sample_id: str,
    gold_graph: CoarseCausalGraph,
    coarse_graph: CoarseCausalGraph,
) -> RefinementSample:
    node_id_to_index = {event.event_id: index for index, event in enumerate(gold_graph.events)}
    node_features = [_event_node_feature(event, gold_graph.query.text) for event in gold_graph.events]

    gold_edge_map = {
        (edge.source_event_id, edge.target_event_id): edge
        for edge in gold_graph.edges
    }

    edge_index: list[list[int]] = []
    edge_features: list[list[float]] = []
    edge_labels: list[int] = []
    edge_type_labels: list[int] = []
    edge_strengths: list[float] = []

    for edge in coarse_graph.edges:
        pair = (edge.source_event_id, edge.target_event_id)
        if edge.source_event_id not in node_id_to_index or edge.target_event_id not in node_id_to_index:
            continue
        edge_index.append([node_id_to_index[edge.source_event_id], node_id_to_index[edge.target_event_id]])
        edge_features.append(_edge_feature(edge))
        if pair in gold_edge_map:
            edge_labels.append(1)
            edge_type_labels.append(RELATION_TO_ID.get(gold_edge_map[pair].relation_type, 0))
            edge_strengths.append(float(gold_edge_map[pair].score))
        else:
            edge_labels.append(0)
            edge_type_labels.append(0)
            edge_strengths.append(0.0)

    query_features = [
        float(len(gold_graph.query.focus_entities)),
        float(len(gold_graph.documents)),
        float(len(gold_graph.events)),
        float(len(coarse_graph.edges)),
    ]

    return RefinementSample(
        sample_id=sample_id,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_labels=edge_labels,
        edge_type_labels=edge_type_labels,
        edge_strengths=edge_strengths,
        query_features=query_features,
        metadata={
            "query_id": gold_graph.query.query_id,
            "dataset": gold_graph.query.metadata.get("dataset", "unknown"),
        },
    )


def load_maven_refinement_samples(
    dataset_path: Path,
    split: str = "train",
    limit: int | None = None,
    seed: int = 7,
) -> list[RefinementSample]:
    rows = _read_maven_rows(dataset_path, split=split)
    if limit is not None:
        rows = rows[:limit]

    samples: list[RefinementSample] = []
    for index, row in enumerate(rows):
        gold_graph = maven_row_to_gold_graph(row)
        if len(gold_graph.events) < 2 or not gold_graph.edges:
            continue
        coarse_graph = perturb_gold_graph_to_coarse(
            gold_graph,
            seed=seed + index,
        )
        sample = gold_and_coarse_graph_to_refinement_sample(
            sample_id=f"maven_{row['id']}",
            gold_graph=gold_graph,
            coarse_graph=coarse_graph,
        )
        if sample.edge_index:
            samples.append(sample)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export synthetic, MIRAI-based, or MAVEN-ERE-based refinement samples.")
    parser.add_argument("--mode", choices=["synthetic", "mirai", "maven"], default="synthetic")
    parser.add_argument("--dataset", default="datasets/MIRAI_data.zip", help="Path to MIRAI zip file.")
    parser.add_argument("--maven-dataset", default="datasets/MAVEN_ERE.zip", help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--query-id", default="1", help="MIRAI QueryId when mode=mirai.")
    parser.add_argument("--split", default="test", help="MIRAI split name.")
    parser.add_argument("--event-extractor", default="rule", help="Event extractor backend name.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of synthetic samples.")
    parser.add_argument("--limit", type=int, default=4, help="Maximum number of MAVEN rows exported when mode=maven.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic":
        samples = generate_synthetic_refinement_samples(num_samples=args.num_samples)
        payload = {"mode": "synthetic", "samples": [sample.to_dict() for sample in samples]}
    elif args.mode == "mirai":
        payload = export_mirai_refinement_sample(
            dataset_path=Path(args.dataset),
            query_id=args.query_id,
            split=args.split,
            event_extractor_name=args.event_extractor,
        )
    else:
        samples = load_maven_refinement_samples(
            dataset_path=Path(args.maven_dataset),
            split=args.split,
            limit=args.limit,
        )
        payload = {"mode": "maven", "samples": [sample.to_dict() for sample in samples]}

    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
