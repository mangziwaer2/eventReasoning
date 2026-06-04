from __future__ import annotations

import argparse
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal_graph import CoarseCausalEdge
from causal_graph import CoarseCausalGraph
from causal_graph import EvidenceSpan
from causal_graph import EventNode
from causal_graph import GraphBuildTrace
from causal_graph import NewsDocument
from causal_graph import QuerySpec
from event_extraction import extract_titlecase_entities
from mirai_dataset import export_mirai_query_snapshot
from mirai_dataset import get_mirai_query_by_id
from mirai_dataset import load_mirai_news_for_docids
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path
from query_causal_graph import QueryCausalGraphBuilder
from query_causal_graph import build_query
from query_causal_graph import load_news_jsonl


RELATION_TYPES = ("precedes", "causes", "escalates", "mitigates")
RELATION_TO_ID = {relation: index for index, relation in enumerate(RELATION_TYPES)}
ID_TO_RELATION = {index: relation for relation, index in RELATION_TO_ID.items()}


@dataclass(slots=True)
class DocumentGraphSample:
    sample_id: str
    query: QuerySpec
    documents: list[NewsDocument]
    events: list[EventNode]
    gold_graph: CoarseCausalGraph | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query.to_dict(),
            "documents": [document.to_dict() for document in self.documents],
            "events": [event.to_dict() for event in self.events],
            "gold_graph": self.gold_graph.to_dict() if self.gold_graph is not None else None,
            "metadata": self.metadata,
        }

    def to_instruction_example(self) -> dict[str, Any]:
        prompt_lines = [
            "You build a coarse causal graph from retrieved evidence documents and extracted events.",
            "Return strict JSON with the schema {\"edges\": [...]} only.",
            "Each edge must contain source_event_id, target_event_id, relation_type, and score.",
            "Allowed relation_type values: precedes, causes, escalates, mitigates.",
            "Only use the listed event ids. Do not create new nodes.",
            "",
            f"Query: {self.query.text}",
            "Documents:",
            self.render_documents(),
            "Events:",
            self.render_events(),
        ]
        return {
            "sample_id": self.sample_id,
            "prompt": "\n".join(prompt_lines),
            "target": self.render_gold_target(),
            "metadata": self.metadata,
        }

    def render_documents(self) -> str:
        parts: list[str] = []
        for document in self.documents:
            parts.append(
                "\n".join(
                    [
                        f"[Document {document.document_id}]",
                        f"Title: {document.title}",
                        f"Text: {document.text}",
                    ]
                )
            )
        return "\n\n".join(parts)

    def render_events(self) -> str:
        lines: list[str] = []
        for event in self.events:
            trigger = str(event.metadata.get("trigger", "")).strip()
            trigger_text = f" | trigger={trigger}" if trigger else ""
            lines.append(
                f"- {event.event_id} | doc={event.document_id} | sent={event.sentence_index}{trigger_text} | text={event.text}"
            )
        return "\n".join(lines)

    def render_gold_target(self) -> str:
        if self.gold_graph is None:
            return json.dumps({"edges": []}, ensure_ascii=False)
        return json.dumps({"edges": graph_edges_to_payload(self.gold_graph)}, ensure_ascii=False)


def _read_maven_rows(zip_path: Path, split: str) -> list[dict[str, Any]]:
    member_name = f"MAVEN_ERE/{split}.jsonl"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            text_stream = io.TextIOWrapper(handle, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _normalize_sentence(sentence: Any) -> str:
    if isinstance(sentence, list):
        return " ".join(str(token) for token in sentence)
    return str(sentence)


def _maven_relation_to_type(relation_name: str) -> str:
    if relation_name == "CAUSE":
        return "causes"
    if relation_name in {"PRECONDITION", "BEFORE", "OVERLAP", "SIMULTANEOUS", "CONTAINS", "ENDS-ON", "BEGINS-ON"}:
        return "precedes"
    return "precedes"


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


def _event_degrees(graph: CoarseCausalGraph) -> dict[str, int]:
    degrees: dict[str, int] = {event.event_id: 0 for event in graph.events}
    for edge in graph.edges:
        if edge.source_event_id in degrees:
            degrees[edge.source_event_id] += 1
        if edge.target_event_id in degrees:
            degrees[edge.target_event_id] += 1
    return degrees


def truncate_graph_events(graph: CoarseCausalGraph, max_events: int | None = None) -> CoarseCausalGraph:
    if max_events is None or max_events <= 0 or len(graph.events) <= max_events:
        return graph

    selected_ids: list[str] = []
    sorted_edges = sorted(graph.edges, key=lambda edge: edge.score, reverse=True)
    for edge in sorted_edges:
        for event_id in (edge.source_event_id, edge.target_event_id):
            if event_id in selected_ids:
                continue
            selected_ids.append(event_id)
            if len(selected_ids) >= max_events:
                break
        if len(selected_ids) >= max_events:
            break

    degrees = _event_degrees(graph)
    remaining_events = sorted(
        graph.events,
        key=lambda event: (
            -degrees.get(event.event_id, 0),
            -float(event.confidence),
            int(event.sentence_index),
            event.event_id,
        ),
    )
    for event in remaining_events:
        if len(selected_ids) >= max_events:
            break
        if event.event_id not in selected_ids:
            selected_ids.append(event.event_id)

    selected_set = set(selected_ids)
    filtered_events = [event for event in graph.events if event.event_id in selected_set]
    filtered_edges = [
        edge
        for edge in graph.edges
        if edge.source_event_id in selected_set and edge.target_event_id in selected_set
    ]
    return CoarseCausalGraph(
        query=graph.query,
        documents=graph.documents,
        events=filtered_events,
        edges=_dedupe_edges(filtered_edges),
        trace=graph.trace,
    )


def truncate_events(events: list[EventNode], max_events: int | None = None) -> list[EventNode]:
    if max_events is None or max_events <= 0 or len(events) <= max_events:
        return events
    ranked = sorted(
        events,
        key=lambda event: (-float(event.confidence), int(event.sentence_index), event.event_id),
    )
    selected_ids = {event.event_id for event in ranked[:max_events]}
    return [event for event in events if event.event_id in selected_ids]


def maven_row_to_gold_graph(row: dict[str, Any]) -> CoarseCausalGraph:
    sentences = [_normalize_sentence(sentence) for sentence in row.get("sentences", [])]
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
            text="\n".join(sentences),
            source="MAVEN-ERE",
        )
    ]

    events: list[EventNode] = []
    source_event_id_to_node_id: dict[str, str] = {}
    for event_index, event in enumerate(row.get("events", [])):
        mentions = event.get("mention", [])
        if not mentions:
            continue
        first_mention = mentions[0]
        sent_id = int(first_mention.get("sent_id", 0))
        if sent_id >= len(sentences):
            continue
        sentence_text = sentences[sent_id]
        trigger_word = str(first_mention.get("trigger_word", "")).strip()
        node_id = f"maven_e{event_index}"
        source_event_id_to_node_id[str(event.get("id", node_id))] = node_id
        events.append(
            EventNode(
                event_id=node_id,
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
            source_node = source_event_id_to_node_id.get(source_id)
            target_node = source_event_id_to_node_id.get(target_id)
            if source_node is None or target_node is None:
                continue
            edges.append(
                CoarseCausalEdge(
                    edge_id=f"gold_edge_{edge_index}",
                    source_event_id=source_node,
                    target_event_id=target_node,
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
            source_node = source_event_id_to_node_id.get(source_id)
            target_node = source_event_id_to_node_id.get(target_id)
            if source_node is None or target_node is None:
                continue
            edges.append(
                CoarseCausalEdge(
                    edge_id=f"gold_edge_{edge_index}",
                    source_event_id=source_node,
                    target_event_id=target_node,
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
        edges=_dedupe_edges(edges),
        trace=GraphBuildTrace(),
    )


def load_maven_document_graph_samples(
    dataset_path: Path,
    split: str = "train",
    limit: int | None = None,
    max_events: int | None = None,
) -> list[DocumentGraphSample]:
    rows = _read_maven_rows(dataset_path, split=split)
    if limit and limit > 0:
        rows = rows[:limit]

    samples: list[DocumentGraphSample] = []
    for row in rows:
        gold_graph = maven_row_to_gold_graph(row)
        gold_graph = truncate_graph_events(gold_graph, max_events=max_events)
        if len(gold_graph.events) < 2 or not gold_graph.edges:
            continue
        samples.append(
            DocumentGraphSample(
                sample_id=f"maven_{row['id']}",
                query=gold_graph.query,
                documents=gold_graph.documents,
                events=gold_graph.events,
                gold_graph=gold_graph,
                metadata={"dataset": "MAVEN-ERE", "row_id": row["id"]},
            )
        )
    return samples


def build_document_graph_inference_sample(
    query: QuerySpec,
    documents: list[NewsDocument],
    sample_id: str,
    event_extractor_name: str = "rule",
    max_docs: int = 6,
    max_events_per_doc: int = 6,
    max_events: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentGraphSample:
    builder = QueryCausalGraphBuilder(
        max_docs=max_docs,
        max_events_per_doc=max_events_per_doc,
        event_extractor_name=event_extractor_name,
    )
    local_graph = builder.build(query, documents)
    events = truncate_events(local_graph.events, max_events=max_events)
    return DocumentGraphSample(
        sample_id=sample_id,
        query=query,
        documents=local_graph.documents,
        events=events,
        gold_graph=None,
        metadata=metadata or {},
    )


def load_mirai_document_graph_sample(
    dataset_path: Path,
    query_id: str,
    split: str = "test",
    event_extractor_name: str = "rule",
    max_docs: int = 6,
    max_events_per_doc: int = 6,
    max_events: int | None = None,
) -> DocumentGraphSample:
    example = get_mirai_query_by_id(dataset_path, query_id=query_id, split=split)
    query = example.build_query_spec()
    documents = load_mirai_news_for_docids(dataset_path, example.docids)
    return build_document_graph_inference_sample(
        query=query,
        documents=documents,
        sample_id=f"mirai_{query_id}",
        event_extractor_name=event_extractor_name,
        max_docs=max_docs,
        max_events_per_doc=max_events_per_doc,
        max_events=max_events,
        metadata={
            "dataset": "MIRAI",
            "mirai_query": json.loads(export_mirai_query_snapshot(example)),
        },
    )


def load_jsonl_document_graph_sample(
    input_path: Path,
    query_text: str,
    cutoff_time: str | None = None,
    sample_id: str = "jsonl_0",
    event_extractor_name: str = "rule",
    max_docs: int = 6,
    max_events_per_doc: int = 6,
    max_events: int | None = None,
) -> DocumentGraphSample:
    query = build_query(query_text, cutoff_time)
    if not query.focus_entities:
        query.focus_entities = extract_titlecase_entities(query.text)
    documents = load_news_jsonl(input_path)
    return build_document_graph_inference_sample(
        query=query,
        documents=documents,
        sample_id=sample_id,
        event_extractor_name=event_extractor_name,
        max_docs=max_docs,
        max_events_per_doc=max_events_per_doc,
        max_events=max_events,
        metadata={"dataset": "jsonl"},
    )


def graph_edges_to_payload(graph: CoarseCausalGraph) -> list[dict[str, Any]]:
    return [
        {
            "source_event_id": edge.source_event_id,
            "target_event_id": edge.target_event_id,
            "relation_type": edge.relation_type,
            "score": round(float(edge.score), 4),
        }
        for edge in graph.edges
    ]


def _extract_json_block(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return None


def parse_edge_payload(text: str) -> list[dict[str, Any]]:
    json_block = _extract_json_block(text)
    if json_block is None:
        return []
    try:
        payload = json.loads(json_block)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        raw_edges = payload
    elif isinstance(payload, dict):
        raw_edges = payload.get("edges", [])
    else:
        return []
    if not isinstance(raw_edges, list):
        return []
    return [edge for edge in raw_edges if isinstance(edge, dict)]


def build_graph_from_edge_payload(
    query: QuerySpec,
    documents: list[NewsDocument],
    events: list[EventNode],
    edge_payload: list[dict[str, Any]],
) -> CoarseCausalGraph:
    valid_event_ids = {event.event_id for event in events}
    event_lookup = {event.event_id: event for event in events}
    edges: list[CoarseCausalEdge] = []

    for edge_index, row in enumerate(edge_payload):
        source_event_id = str(row.get("source_event_id", "")).strip()
        target_event_id = str(row.get("target_event_id", "")).strip()
        relation_type = str(row.get("relation_type", "")).strip().lower()
        if (
            source_event_id not in valid_event_ids
            or target_event_id not in valid_event_ids
            or source_event_id == target_event_id
            or relation_type not in RELATION_TO_ID
        ):
            continue
        raw_score = row.get("score", 0.5)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.5
        score = round(min(max(score, 0.0), 1.0), 4)
        source_event = event_lookup[source_event_id]
        target_event = event_lookup[target_event_id]
        edges.append(
            CoarseCausalEdge(
                edge_id=f"pred_edge_{edge_index}",
                source_event_id=source_event_id,
                target_event_id=target_event_id,
                relation_type=relation_type,
                score=score,
                evidence=source_event.evidence + target_event.evidence,
                feature_scores={},
                metadata={"generated_by": "qwen"},
            )
        )

    return CoarseCausalGraph(
        query=query,
        documents=documents,
        events=events,
        edges=_dedupe_edges(edges),
        trace=GraphBuildTrace(),
    )


def sample_graph_from_model_output(sample: DocumentGraphSample, text: str) -> CoarseCausalGraph:
    return build_graph_from_edge_payload(
        query=sample.query,
        documents=sample.documents,
        events=sample.events,
        edge_payload=parse_edge_payload(text),
    )


def _parse_query(data: dict[str, Any]) -> QuerySpec:
    return QuerySpec(
        query_id=str(data.get("query_id", "")),
        text=str(data.get("text", "")),
        cutoff_time=data.get("cutoff_time"),
        focus_entities=[str(item) for item in data.get("focus_entities", [])],
        metadata=dict(data.get("metadata", {})),
    )


def _parse_document(data: dict[str, Any]) -> NewsDocument:
    return NewsDocument(
        document_id=str(data.get("document_id", "")),
        title=str(data.get("title", "")),
        text=str(data.get("text", "")),
        publish_time=data.get("publish_time"),
        source=str(data.get("source", "unknown")),
        metadata=dict(data.get("metadata", {})),
    )


def _parse_evidence(data: dict[str, Any]) -> EvidenceSpan:
    return EvidenceSpan(
        document_id=str(data.get("document_id", "")),
        sentence_index=int(data.get("sentence_index", 0)),
        text=str(data.get("text", "")),
    )


def _parse_event(data: dict[str, Any]) -> EventNode:
    return EventNode(
        event_id=str(data.get("event_id", "")),
        text=str(data.get("text", "")),
        normalized_text=str(data.get("normalized_text", "")),
        document_id=str(data.get("document_id", "")),
        sentence_index=int(data.get("sentence_index", 0)),
        participants=[str(item) for item in data.get("participants", [])],
        node_type=str(data.get("node_type", "observed")),
        confidence=float(data.get("confidence", 0.0)),
        evidence=[_parse_evidence(item) for item in data.get("evidence", [])],
        metadata=dict(data.get("metadata", {})),
    )


def _parse_edge(data: dict[str, Any]) -> CoarseCausalEdge:
    return CoarseCausalEdge(
        edge_id=str(data.get("edge_id", "")),
        source_event_id=str(data.get("source_event_id", "")),
        target_event_id=str(data.get("target_event_id", "")),
        relation_type=str(data.get("relation_type", "")),
        score=float(data.get("score", 0.0)),
        evidence=[_parse_evidence(item) for item in data.get("evidence", [])],
        feature_scores={str(key): float(value) for key, value in dict(data.get("feature_scores", {})).items()},
        metadata=dict(data.get("metadata", {})),
    )


def load_coarse_graph(path: Path) -> CoarseCausalGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph_data = payload.get("coarse_graph", payload)
    return CoarseCausalGraph(
        query=_parse_query(graph_data.get("query", {})),
        documents=[_parse_document(item) for item in graph_data.get("documents", [])],
        events=[_parse_event(item) for item in graph_data.get("events", [])],
        edges=[_parse_edge(item) for item in graph_data.get("edges", [])],
        trace=GraphBuildTrace(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export document-level graph samples for coarse graph training.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="train", help="MAVEN split name.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of MAVEN rows.")
    parser.add_argument("--max-events", type=int, default=0, help="Optional cap on total events per sample.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_maven_document_graph_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit,
        max_events=args.max_events or None,
    )
    payload = {
        "samples": [sample.to_dict() for sample in samples],
        "instruction_samples": [sample.to_instruction_example() for sample in samples[: min(8, len(samples))]],
    }
    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        resolve_repo_path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
