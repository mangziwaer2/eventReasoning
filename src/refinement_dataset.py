from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from causal_graph import CoarseCausalGraph
from causal_graph import CoarseCausalEdge
from causal_graph import EventNode
from coarse_graph_dataset import load_coarse_graph
from coarse_graph_dataset import load_maven_document_graph_samples
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - fallback for data export and CLI help
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


RELATION_TO_ID = {
    "precedes": 0,
    "causes": 1,
    "escalates": 2,
    "mitigates": 3,
}
ID_TO_RELATION = {value: key for key, value in RELATION_TO_ID.items()}
TIME_NORMALIZATION_DAYS = 30.0
MAX_RELATIVE_TIME_VALUE = 365.0
EDGE_FEATURE_DIM = 20
DEFAULT_CANDIDATE_RELATION = "precedes"


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


class RefinementTensorDataset(Dataset):
    def __init__(self, samples: list[RefinementSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("RefinementTensorDataset requires torch in the active environment.")
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


def _safe_feature(edge: CoarseCausalEdge, key: str) -> float:
    return float(edge.feature_scores.get(key, 0.0))


def _clip_feature(value: float, minimum: float = -MAX_RELATIVE_TIME_VALUE, maximum: float = MAX_RELATIVE_TIME_VALUE) -> float:
    return max(min(float(value), maximum), minimum)


def _parse_time_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    normalized = text.replace("/", "-")
    candidates = (
        normalized,
        normalized.replace(" ", "T"),
    )
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp() / 86400.0

    for pattern in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(normalized, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        return parsed.timestamp() / 86400.0
    return None


def _resolve_raw_event_time(
    event: EventNode,
    document_time_lookup: dict[str, float | None],
) -> tuple[float, float]:
    publish_time = _parse_time_value(event.metadata.get("publish_time"))
    if publish_time is not None:
        return publish_time, 1.0

    document_time = document_time_lookup.get(event.document_id)
    if document_time is not None:
        return document_time, 1.0

    return float(event.sentence_index), 0.0


def _build_temporal_context(
    query,
    documents: list[Any],
    events: list[EventNode],
) -> tuple[dict[str, float], dict[str, float], float, float]:
    document_time_lookup = {document.document_id: _parse_time_value(document.publish_time) for document in documents}

    raw_event_times: dict[str, float] = {}
    event_time_validity: dict[str, float] = {}
    for event in events:
        raw_time, observed = _resolve_raw_event_time(event, document_time_lookup)
        raw_event_times[event.event_id] = raw_time
        event_time_validity[event.event_id] = observed

    observed_times = [value for event_id, value in raw_event_times.items() if event_time_validity[event_id] > 0.5]
    cutoff_raw = _parse_time_value(query.cutoff_time)

    if observed_times:
        anchor_time = min(observed_times)
    elif cutoff_raw is not None:
        anchor_time = cutoff_raw
    else:
        anchor_time = 0.0

    if cutoff_raw is not None:
        anchor_time = min(anchor_time, cutoff_raw)

    event_time_values = {
        event_id: _clip_feature((raw_time - anchor_time) / TIME_NORMALIZATION_DAYS)
        for event_id, raw_time in raw_event_times.items()
    }
    cutoff_value = 0.0
    if cutoff_raw is not None:
        cutoff_value = _clip_feature((cutoff_raw - anchor_time) / TIME_NORMALIZATION_DAYS)
    cutoff_present = 1.0 if cutoff_raw is not None else 0.0
    return event_time_values, event_time_validity, cutoff_value, cutoff_present


def _event_node_feature(
    event: EventNode,
    query_text: str,
    event_time_value: float,
    time_validity: float,
) -> list[float]:
    participants_count = float(len(event.participants))
    token_count = float(len(event.normalized_text.split()))
    sentence_index = float(event.sentence_index)
    confidence = float(event.confidence)
    query_overlap = 1.0 if any(token in event.normalized_text for token in query_text.lower().split()) else 0.0
    trigger_length = float(len(str(event.metadata.get("trigger", ""))))
    is_bridge_hypothesis = 1.0 if event.node_type != "observed" else 0.0
    is_title_event = 1.0 if event.metadata.get("is_title") else 0.0
    return [
        participants_count,
        token_count,
        sentence_index,
        confidence,
        query_overlap,
        trigger_length,
        event_time_value,
        time_validity,
        is_bridge_hypothesis,
        is_title_event,
    ]


def _edge_feature(
    edge: CoarseCausalEdge,
    source_event: EventNode,
    target_event: EventNode,
    event_time_values: dict[str, float],
    event_degrees: dict[str, tuple[int, int]] | None = None,
    candidate_source: str = "coarse",
) -> list[float]:
    relation_id = float(RELATION_TO_ID.get(edge.relation_type, 0))
    source_time_value = float(event_time_values.get(edge.source_event_id, 0.0))
    target_time_value = float(event_time_values.get(edge.target_event_id, 0.0))
    delta_time_value = _clip_feature(target_time_value - source_time_value)
    abs_delta_time_value = min(abs(delta_time_value), MAX_RELATIVE_TIME_VALUE)
    sentence_gap = _clip_feature((float(target_event.sentence_index) - float(source_event.sentence_index)) / 8.0, minimum=-32.0, maximum=32.0)
    is_cross_document = 1.0 if source_event.document_id != target_event.document_id else 0.0
    source_out_degree, source_in_degree = event_degrees.get(source_event.event_id, (0, 0)) if event_degrees else (0, 0)
    target_out_degree, target_in_degree = event_degrees.get(target_event.event_id, (0, 0)) if event_degrees else (0, 0)
    same_sentence = 1.0 if source_event.document_id == target_event.document_id and source_event.sentence_index == target_event.sentence_index else 0.0
    candidate_source = candidate_source.strip().lower()
    is_coarse_edge = 1.0 if candidate_source == "coarse" else 0.0
    is_completion_candidate = 1.0 if candidate_source == "completion" else 0.0
    return [
        float(edge.score),
        relation_id,
        _safe_feature(edge, "temporal_score"),
        _safe_feature(edge, "entity_overlap_score"),
        _safe_feature(edge, "lexical_support_score"),
        _safe_feature(edge, "marker_score"),
        _safe_feature(edge, "query_alignment_score"),
        source_time_value,
        target_time_value,
        delta_time_value,
        abs_delta_time_value,
        sentence_gap,
        is_cross_document,
        float(source_out_degree),
        float(source_in_degree),
        float(target_out_degree),
        float(target_in_degree),
        same_sentence,
        is_coarse_edge,
        is_completion_candidate,
    ]


def _query_features(
    graph: CoarseCausalGraph,
    edge_count: int,
    cutoff_value: float,
    cutoff_present: float,
) -> list[float]:
    return [
        float(len(graph.query.focus_entities)),
        float(len(graph.documents)),
        float(len(graph.events)),
        float(edge_count),
        cutoff_value,
        cutoff_present,
    ]


def _dedupe_edges(edges: list[CoarseCausalEdge]) -> list[CoarseCausalEdge]:
    deduped: list[CoarseCausalEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = (edge.source_event_id, edge.target_event_id, edge.relation_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _event_degrees(edges: list[CoarseCausalEdge], event_ids: set[str]) -> dict[str, tuple[int, int]]:
    out_degrees = {event_id: 0 for event_id in event_ids}
    in_degrees = {event_id: 0 for event_id in event_ids}
    for edge in edges:
        if edge.source_event_id in out_degrees:
            out_degrees[edge.source_event_id] += 1
        if edge.target_event_id in in_degrees:
            in_degrees[edge.target_event_id] += 1
    return {
        event_id: (
            out_degrees.get(event_id, 0),
            in_degrees.get(event_id, 0),
        )
        for event_id in event_ids
    }


def _pair_candidate_score(source_event: EventNode, target_event: EventNode, query_text: str) -> float:
    if source_event.event_id == target_event.event_id:
        return 0.0
    same_document = 1.0 if source_event.document_id == target_event.document_id else 0.0
    sentence_gap = abs(target_event.sentence_index - source_event.sentence_index)
    distance_score = 1.0 / (1.0 + float(sentence_gap))
    shared_participants = len(set(source_event.participants) & set(target_event.participants))
    source_tokens = set(source_event.normalized_text.lower().split())
    target_tokens = set(target_event.normalized_text.lower().split())
    if source_tokens or target_tokens:
        lexical_score = len(source_tokens & target_tokens) / max(len(source_tokens | target_tokens), 1)
    else:
        lexical_score = 0.0
    query_tokens = set(query_text.lower().split())
    source_query_score = len(query_tokens & source_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
    target_query_score = len(query_tokens & target_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
    forward_time = 1.0 if target_event.sentence_index >= source_event.sentence_index else 0.5
    return (
        0.25 * same_document
        + 0.20 * distance_score
        + 0.20 * lexical_score
        + 0.15 * min(shared_participants, 3) / 3.0
        + 0.10 * ((source_query_score + target_query_score) / 2.0)
        + 0.10 * forward_time
    )


def _completion_edge(
    edge_id: str,
    source_event: EventNode,
    target_event: EventNode,
    query_text: str,
    event_time_values: dict[str, float],
) -> CoarseCausalEdge:
    delta_time = event_time_values.get(target_event.event_id, 0.0) - event_time_values.get(source_event.event_id, 0.0)
    temporal_score = 1.0 / (1.0 + abs(delta_time))
    candidate_score = _pair_candidate_score(source_event, target_event, query_text)
    return CoarseCausalEdge(
        edge_id=edge_id,
        source_event_id=source_event.event_id,
        target_event_id=target_event.event_id,
        relation_type=DEFAULT_CANDIDATE_RELATION,
        score=round(min(max(candidate_score, 0.0), 0.49), 4),
        evidence=source_event.evidence + target_event.evidence,
        feature_scores={
            "temporal_score": temporal_score,
            "entity_overlap_score": min(len(set(source_event.participants) & set(target_event.participants)) / 3.0, 1.0),
            "lexical_support_score": candidate_score,
            "marker_score": 0.0,
            "query_alignment_score": candidate_score,
        },
        metadata={
            "candidate_source": "completion",
            "generated_for_refinement": True,
        },
    )


def _rank_completion_pairs(
    events: list[EventNode],
    query_text: str,
    forbidden_pairs: set[tuple[str, str]],
) -> list[tuple[float, EventNode, EventNode]]:
    candidates: list[tuple[float, EventNode, EventNode]] = []
    for source_event in events:
        for target_event in events:
            pair = (source_event.event_id, target_event.event_id)
            if pair in forbidden_pairs or source_event.event_id == target_event.event_id:
                continue
            if source_event.document_id != target_event.document_id:
                shared_participants = set(source_event.participants) & set(target_event.participants)
                if not shared_participants:
                    continue
            score = _pair_candidate_score(source_event, target_event, query_text)
            candidates.append((score, source_event, target_event))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def add_completion_candidates(
    gold_graph: CoarseCausalGraph,
    coarse_graph: CoarseCausalGraph,
    event_time_values: dict[str, float],
    negative_ratio: float = 0.75,
    max_completion_edges: int | None = None,
    seed: int = 7,
) -> list[CoarseCausalEdge]:
    rng = random.Random(seed)
    event_lookup = {event.event_id: event for event in gold_graph.events}
    coarse_edges = list(coarse_graph.edges)
    coarse_pairs = {(edge.source_event_id, edge.target_event_id) for edge in coarse_edges}
    gold_pairs = {(edge.source_event_id, edge.target_event_id) for edge in gold_graph.edges}

    completion_edges: list[CoarseCausalEdge] = []
    missing_gold_pairs = [pair for pair in gold_pairs if pair not in coarse_pairs]
    rng.shuffle(missing_gold_pairs)
    for source_id, target_id in missing_gold_pairs:
        source_event = event_lookup.get(source_id)
        target_event = event_lookup.get(target_id)
        if source_event is None or target_event is None:
            continue
        completion_edges.append(
            _completion_edge(
                edge_id=f"completion_gold_{len(completion_edges)}",
                source_event=source_event,
                target_event=target_event,
                query_text=gold_graph.query.text,
                event_time_values=event_time_values,
            )
        )

    target_negative_count = max(1, int(max(len(gold_pairs), 1) * max(negative_ratio, 0.0)))
    forbidden_pairs = set(coarse_pairs) | set(gold_pairs)
    ranked_negatives = _rank_completion_pairs(gold_graph.events, gold_graph.query.text, forbidden_pairs)
    for _, source_event, target_event in ranked_negatives[:target_negative_count]:
        completion_edges.append(
            _completion_edge(
                edge_id=f"completion_neg_{len(completion_edges)}",
                source_event=source_event,
                target_event=target_event,
                query_text=gold_graph.query.text,
                event_time_values=event_time_values,
            )
        )

    if max_completion_edges is not None and max_completion_edges > 0:
        completion_edges = completion_edges[:max_completion_edges]
    return _dedupe_edges(coarse_edges + completion_edges)


def perturb_gold_graph_to_coarse(
    graph: CoarseCausalGraph,
    drop_rate: float = 0.25,
    flip_rate: float = 0.15,
    add_negative_rate: float = 0.25,
    score_min: float = 0.35,
    score_max: float = 0.9,
    seed: int = 7,
) -> CoarseCausalGraph:
    rng = random.Random(seed)
    kept_edges: list[CoarseCausalEdge] = []
    relation_choices = list(RELATION_TO_ID.keys())

    for edge in graph.edges:
        if rng.random() < drop_rate:
            continue
        relation_type = edge.relation_type
        if rng.random() < flip_rate:
            relation_type = rng.choice([item for item in relation_choices if item != relation_type])
        score = rng.uniform(score_min, score_max)
        kept_edges.append(
            CoarseCausalEdge(
                edge_id=edge.edge_id,
                source_event_id=edge.source_event_id,
                target_event_id=edge.target_event_id,
                relation_type=relation_type,
                score=round(score, 4),
                evidence=edge.evidence,
                feature_scores={
                    "temporal_score": rng.uniform(0.3, 1.0),
                    "entity_overlap_score": rng.uniform(0.0, 1.0),
                    "lexical_support_score": rng.uniform(0.0, 1.0),
                    "marker_score": rng.choice([0.0, 1.0]),
                    "query_alignment_score": rng.uniform(0.0, 1.0),
                },
                metadata={"gold": False, "derived_from_gold": True},
            )
        )

    event_ids = [event.event_id for event in graph.events]
    gold_pairs = {(edge.source_event_id, edge.target_event_id) for edge in graph.edges}
    target_negatives = max(1, int(len(graph.edges) * add_negative_rate))
    negative_index = 0
    seen_negative_pairs: set[tuple[str, str]] = set()
    while negative_index < target_negatives and len(event_ids) >= 2:
        source_id, target_id = rng.sample(event_ids, 2)
        pair = (source_id, target_id)
        if pair in gold_pairs or pair in seen_negative_pairs:
            continue
        seen_negative_pairs.add(pair)
        kept_edges.append(
            CoarseCausalEdge(
                edge_id=f"noisy_edge_{negative_index}",
                source_event_id=source_id,
                target_event_id=target_id,
                relation_type=rng.choice(relation_choices),
                score=round(rng.uniform(0.2, 0.55), 4),
                evidence=[],
                feature_scores={
                    "temporal_score": rng.uniform(0.0, 0.8),
                    "entity_overlap_score": rng.uniform(0.0, 0.6),
                    "lexical_support_score": rng.uniform(0.0, 0.5),
                    "marker_score": rng.choice([0.0, 1.0]),
                    "query_alignment_score": rng.uniform(0.0, 0.4),
                },
                metadata={"gold": False, "synthetic_negative": True},
            )
        )
        negative_index += 1

    return CoarseCausalGraph(
        query=graph.query,
        documents=graph.documents,
        events=graph.events,
        edges=_dedupe_edges(kept_edges),
        trace=graph.trace,
    )


def gold_and_coarse_graph_to_refinement_sample(
    sample_id: str,
    gold_graph: CoarseCausalGraph,
    coarse_graph: CoarseCausalGraph,
    negative_completion_ratio: float = 0.75,
    max_completion_edges: int | None = None,
    seed: int = 7,
) -> RefinementSample:
    node_id_to_index = {event.event_id: index for index, event in enumerate(gold_graph.events)}
    event_lookup = {event.event_id: event for event in gold_graph.events}
    event_time_values, event_time_validity, cutoff_value, cutoff_present = _build_temporal_context(
        query=gold_graph.query,
        documents=gold_graph.documents,
        events=gold_graph.events,
    )
    node_features = [
        _event_node_feature(
            event,
            gold_graph.query.text,
            event_time_values.get(event.event_id, 0.0),
            event_time_validity.get(event.event_id, 0.0),
        )
        for event in gold_graph.events
    ]
    candidate_edges = add_completion_candidates(
        gold_graph=gold_graph,
        coarse_graph=coarse_graph,
        event_time_values=event_time_values,
        negative_ratio=negative_completion_ratio,
        max_completion_edges=max_completion_edges,
        seed=seed,
    )
    event_degrees = _event_degrees(
        coarse_graph.edges,
        event_ids=set(node_id_to_index.keys()),
    )

    gold_edge_map = {(edge.source_event_id, edge.target_event_id): edge for edge in gold_graph.edges}
    edge_index: list[list[int]] = []
    edge_features: list[list[float]] = []
    edge_labels: list[int] = []
    edge_type_labels: list[int] = []
    edge_strengths: list[float] = []
    edge_descriptions: list[dict[str, Any]] = []

    for edge in candidate_edges:
        pair = (edge.source_event_id, edge.target_event_id)
        if edge.source_event_id not in node_id_to_index or edge.target_event_id not in node_id_to_index:
            continue
        source_event = event_lookup[edge.source_event_id]
        target_event = event_lookup[edge.target_event_id]
        candidate_source = str(edge.metadata.get("candidate_source", "coarse"))
        edge_index.append([node_id_to_index[edge.source_event_id], node_id_to_index[edge.target_event_id]])
        edge_features.append(
            _edge_feature(
                edge,
                source_event,
                target_event,
                event_time_values,
                event_degrees=event_degrees,
                candidate_source=candidate_source,
            )
        )
        gold_edge = gold_edge_map.get(pair)
        if gold_edge is not None:
            edge_labels.append(1)
            edge_type_labels.append(RELATION_TO_ID.get(gold_edge.relation_type, 0))
            edge_strengths.append(float(gold_edge.score))
            gold_relation = gold_edge.relation_type
        else:
            edge_labels.append(0)
            edge_type_labels.append(0)
            edge_strengths.append(0.0)
            gold_relation = "none"
        edge_descriptions.append(
            {
                "source_event_id": edge.source_event_id,
                "target_event_id": edge.target_event_id,
                "source_text": source_event.text,
                "target_text": target_event.text,
                "coarse_relation_type": edge.relation_type,
                "gold_relation_type": gold_relation,
                "coarse_score": edge.score,
                "candidate_source": candidate_source,
                "delta_time": round(event_time_values.get(edge.target_event_id, 0.0) - event_time_values.get(edge.source_event_id, 0.0), 4),
            }
        )

    completion_count = sum(1 for edge in candidate_edges if edge.metadata.get("candidate_source") == "completion")
    return RefinementSample(
        sample_id=sample_id,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_labels=edge_labels,
        edge_type_labels=edge_type_labels,
        edge_strengths=edge_strengths,
        query_features=_query_features(
            graph=gold_graph,
            edge_count=len(edge_index),
            cutoff_value=cutoff_value,
            cutoff_present=cutoff_present,
        ),
        metadata={
            "query_id": gold_graph.query.query_id,
            "dataset": gold_graph.query.metadata.get("dataset", "unknown"),
            "query_text": gold_graph.query.text,
            "event_texts": [event.text for event in gold_graph.events],
            "edge_descriptions": edge_descriptions,
            "coarse_edge_count": len(coarse_graph.edges),
            "completion_candidate_count": completion_count,
            "candidate_edge_count": len(edge_index),
        },
    )


def load_maven_refinement_samples(
    dataset_path: Path,
    split: str = "train",
    limit: int | None = None,
    max_events: int | None = None,
    negative_completion_ratio: float = 0.75,
    max_completion_edges: int | None = None,
    seed: int = 7,
) -> list[RefinementSample]:
    graph_samples = load_maven_document_graph_samples(
        dataset_path=dataset_path,
        split=split,
        limit=limit,
        max_events=max_events,
    )
    samples: list[RefinementSample] = []
    for index, graph_sample in enumerate(graph_samples):
        gold_graph = graph_sample.gold_graph
        if gold_graph is None or len(gold_graph.events) < 2 or not gold_graph.edges:
            continue
        coarse_graph = perturb_gold_graph_to_coarse(gold_graph, seed=seed + index)
        sample = gold_and_coarse_graph_to_refinement_sample(
            sample_id=graph_sample.sample_id,
            gold_graph=gold_graph,
            coarse_graph=coarse_graph,
            negative_completion_ratio=negative_completion_ratio,
            max_completion_edges=max_completion_edges,
            seed=seed + index,
        )
        if sample.edge_index:
            samples.append(sample)
    return samples


def load_refinement_sample_from_coarse_graph(
    coarse_graph_path: Path,
    sample_id: str | None = None,
    include_completion_candidates: bool = False,
    max_completion_edges: int | None = None,
) -> RefinementSample:
    graph = load_coarse_graph(coarse_graph_path)
    inferred_sample_id = sample_id or coarse_graph_path.stem
    node_id_to_index = {event.event_id: index for index, event in enumerate(graph.events)}
    event_lookup = {event.event_id: event for event in graph.events}
    event_time_values, event_time_validity, cutoff_value, cutoff_present = _build_temporal_context(
        query=graph.query,
        documents=graph.documents,
        events=graph.events,
    )
    event_degrees = _event_degrees(graph.edges, event_ids=set(node_id_to_index.keys()))
    candidate_edges = list(graph.edges)
    if include_completion_candidates:
        forbidden_pairs = {(edge.source_event_id, edge.target_event_id) for edge in graph.edges}
        ranked_candidates = _rank_completion_pairs(graph.events, graph.query.text, forbidden_pairs)
        selected_candidates = ranked_candidates
        if max_completion_edges is not None and max_completion_edges > 0:
            selected_candidates = selected_candidates[:max_completion_edges]
        for _, source_event, target_event in selected_candidates:
            candidate_edges.append(
                _completion_edge(
                    edge_id=f"inference_completion_{len(candidate_edges)}",
                    source_event=source_event,
                    target_event=target_event,
                    query_text=graph.query.text,
                    event_time_values=event_time_values,
                )
            )

    edge_index: list[list[int]] = []
    edge_features: list[list[float]] = []
    edge_labels: list[int] = []
    edge_type_labels: list[int] = []
    edge_strengths: list[float] = []
    edge_descriptions: list[dict[str, Any]] = []

    for edge in candidate_edges:
        if edge.source_event_id not in node_id_to_index or edge.target_event_id not in node_id_to_index:
            continue
        source_event = event_lookup[edge.source_event_id]
        target_event = event_lookup[edge.target_event_id]
        candidate_source = str(edge.metadata.get("candidate_source", "coarse"))
        edge_index.append([node_id_to_index[edge.source_event_id], node_id_to_index[edge.target_event_id]])
        edge_features.append(
            _edge_feature(
                edge,
                source_event,
                target_event,
                event_time_values,
                event_degrees=event_degrees,
                candidate_source=candidate_source,
            )
        )
        edge_labels.append(0)
        edge_type_labels.append(RELATION_TO_ID.get(edge.relation_type, 0))
        edge_strengths.append(float(edge.score))
        edge_descriptions.append(
            {
                "source_event_id": edge.source_event_id,
                "target_event_id": edge.target_event_id,
                "source_text": source_event.text,
                "target_text": target_event.text,
                "coarse_relation_type": edge.relation_type,
                "coarse_score": edge.score,
                "candidate_source": candidate_source,
                "delta_time": round(event_time_values.get(edge.target_event_id, 0.0) - event_time_values.get(edge.source_event_id, 0.0), 4),
            }
        )

    return RefinementSample(
        sample_id=inferred_sample_id,
        node_features=[
            _event_node_feature(
                event,
                graph.query.text,
                event_time_values.get(event.event_id, 0.0),
                event_time_validity.get(event.event_id, 0.0),
            )
            for event in graph.events
        ],
        edge_index=edge_index,
        edge_features=edge_features,
        edge_labels=edge_labels,
        edge_type_labels=edge_type_labels,
        edge_strengths=edge_strengths,
        query_features=_query_features(
            graph=graph,
            edge_count=len(edge_index),
            cutoff_value=cutoff_value,
            cutoff_present=cutoff_present,
        ),
        metadata={
            "query_id": graph.query.query_id,
            "dataset": graph.query.metadata.get("dataset", "unknown"),
            "query_text": graph.query.text,
            "event_texts": [event.text for event in graph.events],
            "edge_descriptions": edge_descriptions,
            "coarse_edge_count": len(graph.edges),
            "completion_candidate_count": max(len(edge_descriptions) - len(graph.edges), 0),
            "candidate_edge_count": len(edge_descriptions),
        },
    )


def generate_synthetic_refinement_samples(num_samples: int = 32, seed: int = 7) -> list[RefinementSample]:
    rng = random.Random(seed)
    samples: list[RefinementSample] = []
    relation_names = list(RELATION_TO_ID.keys())

    for sample_index in range(num_samples):
        node_count = rng.randint(4, 8)
        edge_count = rng.randint(node_count - 1, min(node_count * 2, 12))

        node_features: list[list[float]] = []
        for node_index in range(node_count):
            node_features.append(
                [
                    rng.uniform(0, 6),
                    rng.uniform(3, 18),
                    float(node_index),
                    rng.uniform(0.3, 0.95),
                    rng.choice([0.0, 1.0]),
                    rng.uniform(3, 10),
                    rng.uniform(0.0, 8.0),
                    rng.choice([0.0, 1.0]),
                    rng.choice([0.0, 1.0]),
                    rng.choice([0.0, 1.0]),
                ]
            )

        edge_index: list[list[int]] = []
        edge_features: list[list[float]] = []
        edge_labels: list[int] = []
        edge_type_labels: list[int] = []
        edge_strengths: list[float] = []
        used_pairs: set[tuple[int, int]] = set()

        while len(edge_index) < edge_count:
            source = rng.randint(0, node_count - 2)
            target = rng.randint(source + 1, node_count - 1)
            if (source, target) in used_pairs:
                continue
            used_pairs.add((source, target))
            temporal_score = rng.uniform(0.5, 1.0)
            entity_score = rng.uniform(0.0, 1.0)
            lexical_score = rng.uniform(0.0, 1.0)
            marker_score = rng.choice([0.0, 1.0])
            query_score = rng.uniform(0.0, 1.0)
            coarse_score = 0.34 * temporal_score + 0.22 * entity_score + 0.18 * lexical_score + 0.16 * marker_score + 0.10 * query_score
            relation_id = rng.randint(0, len(relation_names) - 1)

            edge_index.append([source, target])
            edge_features.append(
                [
                    coarse_score,
                    float(relation_id),
                    temporal_score,
                    entity_score,
                    lexical_score,
                    marker_score,
                    query_score,
                    float(source),
                    float(target),
                    float(target - source),
                    float(abs(target - source)),
                    float(target - source) / 4.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                ]
            )
            edge_labels.append(int(coarse_score >= 0.62))
            edge_type_labels.append(relation_id)
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
                    rng.uniform(1, 3),
                    rng.uniform(2, 6),
                    float(node_count),
                    float(edge_count),
                    rng.uniform(0.0, 8.0),
                    rng.choice([0.0, 1.0]),
                ],
                metadata={
                    "synthetic": True,
                    "query_text": "synthetic query",
                    "event_texts": [f"synthetic_event_{i}" for i in range(node_count)],
                    "edge_descriptions": [
                        {
                            "source_event_id": f"n{src}",
                            "target_event_id": f"n{tgt}",
                            "source_text": f"synthetic_event_{src}",
                            "target_text": f"synthetic_event_{tgt}",
                            "coarse_relation_type": relation_names[int(edge_features[idx][1])],
                            "coarse_score": edge_strengths[idx],
                        }
                        for idx, (src, tgt) in enumerate(edge_index)
                    ],
                },
            )
        )

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export refinement samples for training or inference.")
    parser.add_argument("--mode", choices=["synthetic", "maven", "coarse-graph"], default="maven")
    parser.add_argument("--maven-dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="train", help="MAVEN split name.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of synthetic samples.")
    parser.add_argument("--limit", type=int, default=4, help="Maximum number of MAVEN rows exported.")
    parser.add_argument("--max-events", type=int, default=12, help="Maximum events kept in each MAVEN graph sample.")
    parser.add_argument("--negative-completion-ratio", type=float, default=0.75, help="Extra non-gold completion candidates per gold edge.")
    parser.add_argument("--max-completion-edges", type=int, default=0, help="Optional cap on added completion candidates per graph.")
    parser.add_argument("--include-completion-candidates", action="store_true", help="Add non-coarse candidate edges when mode=coarse-graph.")
    parser.add_argument("--coarse-graph", default=None, help="Path to a coarse graph JSON file when mode=coarse-graph.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic":
        payload = {
            "mode": "synthetic",
            "samples": [sample.to_dict() for sample in generate_synthetic_refinement_samples(num_samples=args.num_samples)],
        }
    elif args.mode == "coarse-graph":
        if not args.coarse_graph:
            raise ValueError("--coarse-graph is required when --mode=coarse-graph")
        sample = load_refinement_sample_from_coarse_graph(
            resolve_repo_path(args.coarse_graph),
            include_completion_candidates=args.include_completion_candidates,
            max_completion_edges=args.max_completion_edges or None,
        )
        payload = {"mode": "coarse-graph", "sample": sample.to_dict()}
    else:
        samples = load_maven_refinement_samples(
            dataset_path=resolve_repo_path(args.maven_dataset),
            split=args.split,
            limit=args.limit,
            max_events=args.max_events,
            negative_completion_ratio=args.negative_completion_ratio,
            max_completion_edges=args.max_completion_edges or None,
        )
        payload = {"mode": "maven", "samples": [sample.to_dict() for sample in samples]}

    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        resolve_repo_path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
