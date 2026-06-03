from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_graph import CoarseCausalEdge
from causal_graph import CoarseCausalGraph
from causal_graph import EventNode
from causal_graph import GraphBuildTrace
from event_extraction import lexical_overlap
from event_extraction import tokenize
from mirai_dataset import export_mirai_query_snapshot
from mirai_dataset import get_mirai_query_by_id
from mirai_dataset import load_mirai_news_for_docids
from query_causal_graph import QueryCausalGraphBuilder


CAUSAL_MARKERS = {
    "after",
    "because",
    "cause",
    "caused",
    "causing",
    "following",
    "led",
    "prompted",
    "resulted",
    "sparked",
    "triggered",
}

ESCALATION_TOKENS = {
    "attack",
    "bomb",
    "condemn",
    "criticize",
    "denounce",
    "deploy",
    "escalat",
    "kill",
    "protest",
    "sanction",
    "strike",
    "warn",
    "withdraw",
}

MITIGATION_TOKENS = {
    "aid",
    "ceasefire",
    "deescalat",
    "evacuate",
    "mediate",
    "negotiat",
    "open",
    "pause",
    "rescue",
    "support",
    "talk",
    "truce",
}


def _stem_token(token: str) -> str:
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _temporal_score(source: EventNode, target: EventNode) -> float:
    if source.document_id == target.document_id:
        if source.sentence_index < target.sentence_index:
            return 0.9
        return 0.0
    source_time = source.metadata.get("publish_time")
    target_time = target.metadata.get("publish_time")
    if source_time and target_time and str(source_time) <= str(target_time):
        return 0.55
    return 0.3


def _entity_overlap_score(source: EventNode, target: EventNode) -> float:
    left = {item.lower() for item in source.participants}
    right = {item.lower() for item in target.participants}
    if not left or not right:
        return 0.0
    matched = left & right
    union = left | right
    return len(matched) / len(union)


def _lexical_support_score(source: EventNode, target: EventNode) -> float:
    score, _ = lexical_overlap(source.text, target.text)
    return score


def _marker_score(source: EventNode, target: EventNode) -> float:
    combined = f"{source.text} {target.text}".lower()
    return 1.0 if any(marker in combined for marker in CAUSAL_MARKERS) else 0.0


def _query_alignment_score(query_text: str, event: EventNode) -> float:
    score, _ = lexical_overlap(query_text, event.text)
    return score


def _relation_type(source: EventNode, target: EventNode) -> str:
    source_tokens = {_stem_token(token) for token in tokenize(source.text)}
    target_tokens = {_stem_token(token) for token in tokenize(target.text)}
    if source_tokens & MITIGATION_TOKENS or target_tokens & MITIGATION_TOKENS:
        return "mitigates"
    if source_tokens & ESCALATION_TOKENS or target_tokens & ESCALATION_TOKENS:
        return "escalates"
    if _marker_score(source, target) > 0:
        return "causes"
    return "precedes"


class CoarseCausalGraphBuilder:
    def __init__(self, min_edge_score: float = 0.55, max_edges: int = 40) -> None:
        self.min_edge_score = min_edge_score
        self.max_edges = max_edges

    def build(self, query, documents, events, trace: GraphBuildTrace | None = None) -> CoarseCausalGraph:
        build_trace = trace or GraphBuildTrace()
        event_lookup = {event.event_id: event for event in events}
        edges: list[CoarseCausalEdge] = []
        edge_index = 0

        for source in events:
            for target in events:
                if source.event_id == target.event_id:
                    continue

                temporal_score = _temporal_score(source, target)
                if temporal_score <= 0:
                    continue

                entity_score = _entity_overlap_score(source, target)
                lexical_score = _lexical_support_score(source, target)
                marker_score = _marker_score(source, target)
                query_score = (_query_alignment_score(query.text, source) + _query_alignment_score(query.text, target)) / 2

                score = (
                    0.34 * temporal_score
                    + 0.22 * entity_score
                    + 0.18 * lexical_score
                    + 0.16 * marker_score
                    + 0.10 * query_score
                )
                score = round(min(score, 0.99), 4)
                if score < self.min_edge_score:
                    continue

                relation_type = _relation_type(source, target)
                edge = CoarseCausalEdge(
                    edge_id=f"cg_edge_{edge_index}",
                    source_event_id=source.event_id,
                    target_event_id=target.event_id,
                    relation_type=relation_type,
                    score=score,
                    evidence=source.evidence + target.evidence,
                    feature_scores={
                        "temporal_score": round(temporal_score, 4),
                        "entity_overlap_score": round(entity_score, 4),
                        "lexical_support_score": round(lexical_score, 4),
                        "marker_score": round(marker_score, 4),
                        "query_alignment_score": round(query_score, 4),
                    },
                    metadata={
                        "source_document_id": source.document_id,
                        "target_document_id": target.document_id,
                    },
                )
                edges.append(edge)
                edge_index += 1

        edges.sort(key=lambda item: item.score, reverse=True)
        edges = self._dedupe_edges(edges[: self.max_edges])
        for edge in edges:
            build_trace.event_notes.append(
                f"{edge.source_event_id}->{edge.target_event_id} type={edge.relation_type} score={edge.score:.2f}"
            )

        return CoarseCausalGraph(
            query=query,
            documents=documents,
            events=events,
            edges=edges,
            trace=build_trace,
        )

    def _dedupe_edges(self, edges: list[CoarseCausalEdge]) -> list[CoarseCausalEdge]:
        deduped: list[CoarseCausalEdge] = []
        seen_pairs: set[tuple[str, str]] = set()
        for edge in edges:
            pair = (edge.source_event_id, edge.target_event_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            deduped.append(edge)
        return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a coarse causal graph from a MIRAI query, retrieved evidence, and extracted events."
    )
    parser.add_argument("--dataset", default="datasets/MIRAI_data.zip", help="Path to MIRAI zip file.")
    parser.add_argument("--query-id", required=True, help="MIRAI QueryId.")
    parser.add_argument("--split", default="test", help="MIRAI split name: test or test_subset.")
    parser.add_argument("--event-extractor", default="rule", help="Event extractor backend name.")
    parser.add_argument("--max-docs", type=int, default=6, help="Maximum evidence documents.")
    parser.add_argument("--max-events-per-doc", type=int, default=6, help="Maximum events kept per document.")
    parser.add_argument("--min-edge-score", type=float, default=0.55, help="Minimum score for keeping a coarse edge.")
    parser.add_argument("--max-edges", type=int, default=40, help="Maximum number of coarse edges kept.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    example = get_mirai_query_by_id(dataset_path, query_id=args.query_id, split=args.split)
    documents = load_mirai_news_for_docids(dataset_path, example.docids)

    graph_builder = QueryCausalGraphBuilder(
        max_docs=args.max_docs,
        max_events_per_doc=args.max_events_per_doc,
        event_extractor_name=args.event_extractor,
    )
    query = example.build_query_spec()
    local_graph = graph_builder.build(query, documents)

    coarse_builder = CoarseCausalGraphBuilder(
        min_edge_score=args.min_edge_score,
        max_edges=args.max_edges,
    )
    coarse_graph = coarse_builder.build(
        query=query,
        documents=local_graph.documents,
        events=local_graph.events,
        trace=local_graph.trace,
    )

    payload = {
        "mirai_query": json.loads(export_mirai_query_snapshot(example)),
        "event_extractor": args.event_extractor,
        "coarse_graph": coarse_graph.to_dict(),
    }
    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
