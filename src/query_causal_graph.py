from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_graph import CausalEdge
from causal_graph import EvidenceSpan
from causal_graph import EventNode
from causal_graph import GraphBuildTrace
from causal_graph import LocalCausalGraph
from causal_graph import NewsDocument
from causal_graph import QuerySpec
from causal_graph import RetrievalHit
from event_extractor import BaseEventExtractor
from event_extractor import build_event_extractor
from event_extraction import extract_titlecase_entities
from event_extraction import format_event_mention
from event_extraction import lexical_overlap
from event_extraction import normalize_text
from event_extraction import tokenize

CAUSAL_MARKERS = {
    "after",
    "because",
    "caused",
    "causing",
    "driving",
    "following",
    "led",
    "prompted",
    "sparked",
    "triggered",
}

BRIDGE_PATTERNS = [
    (
        {"raise", "cut", "ban", "sanction", "approve", "vote"},
        {"delay", "drop", "expand", "protest", "respond"},
        "conditions around {participants} shifted",
    ),
    (
        {"attack", "strike", "deploy"},
        {"warn", "evacuate", "respond"},
        "security pressure around {participants} intensified",
    ),
]


class QueryCausalGraphBuilder:
    def __init__(
        self,
        max_docs: int = 6,
        max_events_per_doc: int = 4,
        event_extractor: BaseEventExtractor | None = None,
        event_extractor_name: str = "rule",
    ) -> None:
        self.max_docs = max_docs
        self.max_events_per_doc = max_events_per_doc
        self.event_extractor = event_extractor or build_event_extractor(event_extractor_name)

    def build(self, query: QuerySpec, documents: list[NewsDocument]) -> LocalCausalGraph:
        trace = GraphBuildTrace()
        retrieved_docs = self.retrieve_documents(query, documents, trace)

        events: list[EventNode] = []
        edges: list[CausalEdge] = []
        event_counter = 0
        edge_counter = 0

        for document in retrieved_docs:
            doc_events = self.extract_document_events(query, document, trace, event_counter)
            event_counter += len(doc_events)
            events.extend(doc_events)

            temporal_edges = self.build_temporal_edges(doc_events, edge_counter)
            edge_counter += len(temporal_edges)
            edges.extend(temporal_edges)

            causal_edges = self.build_local_causal_edges(doc_events, edge_counter)
            edge_counter += len(causal_edges)
            edges.extend(causal_edges)

        bridge_events, bridge_edges = self.induce_bridge_events(query, events, edge_counter, event_counter, trace)
        events.extend(bridge_events)
        edges.extend(bridge_edges)

        return LocalCausalGraph(
            query=query,
            documents=retrieved_docs,
            events=events,
            edges=edges,
            trace=trace,
        )

    def retrieve_documents(
        self,
        query: QuerySpec,
        documents: list[NewsDocument],
        trace: GraphBuildTrace,
    ) -> list[NewsDocument]:
        scored: list[tuple[float, list[str], NewsDocument]] = []
        for document in documents:
            score_title, title_terms = lexical_overlap(query.text, document.title)
            score_body, body_terms = lexical_overlap(query.text, document.text)
            score = score_title * 1.4 + score_body
            matched_terms = sorted(set(title_terms + body_terms))
            if score <= 0:
                continue
            scored.append((score, matched_terms, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_docs = [document for _, _, document in scored[: self.max_docs]]
        trace.retrieval_hits = [
            RetrievalHit(document_id=document.document_id, score=round(score, 4), matched_terms=matched_terms)
            for score, matched_terms, document in scored[: self.max_docs]
        ]

        if not top_docs:
            trace.warnings.append("No evidence document matched the query.")
        return top_docs

    def extract_document_events(
        self,
        query: QuerySpec,
        document: NewsDocument,
        trace: GraphBuildTrace,
        start_index: int,
    ) -> list[EventNode]:
        candidates: list[tuple[float, int, str, str, list[str], str]] = []
        document_extraction = self.event_extractor.extract_document(query, document)

        for sentence_extraction in document_extraction.sentence_extractions:
            sentence_index = sentence_extraction.sentence_index
            for atomic_event in sentence_extraction.events:
                overlap_score, matched = lexical_overlap(query.text, atomic_event.text)
                candidates.append(
                    (
                        atomic_event.score,
                        sentence_index,
                        atomic_event.text,
                        atomic_event.normalized_text,
                        atomic_event.participants,
                        atomic_event.trigger,
                    )
                )
                trace.event_notes.append(
                    f"{document.document_id}[{sentence_index}] trigger={atomic_event.trigger} "
                    f"score={atomic_event.score:.2f} matched={','.join(matched) or '-'}"
                )

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[: self.max_events_per_doc]
        events: list[EventNode] = []
        for offset, (score, sentence_index, sentence, normalized_text, participants, trigger) in enumerate(selected):
            event_id = f"e{start_index + offset}"
            event_text = format_event_mention(trigger=trigger, context=sentence)
            events.append(
                EventNode(
                    event_id=event_id,
                    text=event_text,
                    normalized_text=normalized_text,
                    document_id=document.document_id,
                    sentence_index=sentence_index,
                    participants=participants,
                    confidence=round(min(score, 0.95), 4),
                    evidence=[
                        EvidenceSpan(
                            document_id=document.document_id,
                            sentence_index=sentence_index,
                            text=sentence,
                        )
                    ],
                    metadata={
                        "trigger": trigger,
                        "event_mention": event_text,
                        "event_context": sentence,
                        "publish_time": document.publish_time,
                        "is_title": sentence_index == 0,
                        "document_source": document.source,
                    },
                )
            )
        return sorted(events, key=lambda item: item.sentence_index)

    def build_temporal_edges(self, events: list[EventNode], start_index: int) -> list[CausalEdge]:
        edges: list[CausalEdge] = []
        for index in range(len(events) - 1):
            source = events[index]
            target = events[index + 1]
            edges.append(
                CausalEdge(
                    edge_id=f"edge{start_index + index}",
                    source_event_id=source.event_id,
                    target_event_id=target.event_id,
                    relation_type="precedes",
                    confidence=0.55,
                    evidence=source.evidence + target.evidence,
                )
            )
        return edges

    def build_local_causal_edges(self, events: list[EventNode], start_index: int) -> list[CausalEdge]:
        edges: list[CausalEdge] = []
        offset = 0
        for source, target in zip(events, events[1:]):
            combined = f"{source.text} {target.text}".lower()
            if not any(marker in combined for marker in CAUSAL_MARKERS):
                continue
            edges.append(
                CausalEdge(
                    edge_id=f"edge{start_index + offset}",
                    source_event_id=source.event_id,
                    target_event_id=target.event_id,
                    relation_type="causes",
                    confidence=0.72,
                    evidence=source.evidence + target.evidence,
                )
            )
            offset += 1
        return edges

    def induce_bridge_events(
        self,
        query: QuerySpec,
        events: list[EventNode],
        start_edge_index: int,
        start_event_index: int,
        trace: GraphBuildTrace,
    ) -> tuple[list[EventNode], list[CausalEdge]]:
        bridge_events: list[EventNode] = []
        bridge_edges: list[CausalEdge] = []
        bridge_event_index = 0
        bridge_edge_index = 0

        query_entities = query.focus_entities or extract_titlecase_entities(query.text)
        if not query_entities:
            return bridge_events, bridge_edges

        for source in events:
            for target in events:
                if source.event_id == target.event_id:
                    continue
                if source.document_id == target.document_id and source.sentence_index >= target.sentence_index:
                    continue
                participants = sorted(set(source.participants) & set(target.participants) & set(query_entities))
                if not participants:
                    continue
                source_tokens = set(tokenize(source.text))
                target_tokens = set(tokenize(target.text))
                if len(source_tokens & target_tokens) >= 3:
                    continue

                template = self._pick_bridge_template(source, target)
                if template is None:
                    continue

                bridge_text = template.format(participants=", ".join(participants))
                bridge_id = f"e{start_event_index + bridge_event_index}"
                bridge_event_index += 1
                bridge_node = EventNode(
                    event_id=bridge_id,
                    text=bridge_text,
                    normalized_text=normalize_text(bridge_text),
                    document_id="bridge",
                    sentence_index=0,
                    participants=participants,
                    node_type="bridge_hypothesis",
                    confidence=0.38,
                    evidence=source.evidence + target.evidence,
                    metadata={"query_conditioned": True},
                )
                bridge_events.append(bridge_node)
                trace.bridge_notes.append(
                    f"{source.event_id}->{bridge_id}->{target.event_id} participants={','.join(participants)}"
                )

                bridge_edges.extend(
                    [
                        CausalEdge(
                            edge_id=f"edge{start_edge_index + bridge_edge_index}",
                            source_event_id=source.event_id,
                            target_event_id=bridge_id,
                            relation_type="enables",
                            confidence=0.4,
                            evidence=source.evidence,
                            is_hypothesis=True,
                        ),
                        CausalEdge(
                            edge_id=f"edge{start_edge_index + bridge_edge_index + 1}",
                            source_event_id=bridge_id,
                            target_event_id=target.event_id,
                            relation_type="causes",
                            confidence=0.4,
                            evidence=target.evidence,
                            is_hypothesis=True,
                        ),
                    ]
                )
                bridge_edge_index += 2

                if bridge_event_index >= 2:
                    return bridge_events, bridge_edges
        return bridge_events, bridge_edges

    def _pick_bridge_template(self, source: EventNode, target: EventNode) -> str | None:
        source_tokens = set(tokenize(source.text))
        target_tokens = set(tokenize(target.text))
        for source_set, target_set, template in BRIDGE_PATTERNS:
            if source_tokens & source_set and target_tokens & target_set:
                return template
        return None


def load_news_jsonl(path: Path) -> list[NewsDocument]:
    documents: list[NewsDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            documents.append(
                NewsDocument(
                    document_id=str(row["document_id"]),
                    title=row.get("title", ""),
                    text=row.get("text", ""),
                    publish_time=row.get("publish_time"),
                    source=row.get("source", "unknown"),
                    metadata={key: value for key, value in row.items() if key not in {"document_id", "title", "text", "publish_time", "source"}},
                )
            )
    return documents


def dump_graph(graph: LocalCausalGraph, output_path: Path | None) -> None:
    payload = json.dumps(graph.to_dict(), ensure_ascii=False, indent=2)
    if output_path is None:
        print(payload)
        return
    output_path.write_text(payload, encoding="utf-8")


def build_query(query_text: str, cutoff_time: str | None) -> QuerySpec:
    return QuerySpec(
        query_id="query_001",
        text=query_text,
        cutoff_time=cutoff_time,
        focus_entities=extract_titlecase_entities(query_text),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local causal graph from a query and a news JSONL file.")
    parser.add_argument("--input", required=True, help="Path to a news JSONL file.")
    parser.add_argument("--query", required=True, help="Forecast query text.")
    parser.add_argument("--cutoff", default=None, help="Cutoff time string.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument("--max-docs", type=int, default=6, help="Maximum number of retrieved documents.")
    parser.add_argument("--max-events-per-doc", type=int, default=4, help="Maximum number of events kept from each document.")
    parser.add_argument("--event-extractor", default="rule", help="Event extractor backend name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = build_query(args.query, args.cutoff)
    documents = load_news_jsonl(Path(args.input))
    builder = QueryCausalGraphBuilder(
        max_docs=args.max_docs,
        max_events_per_doc=args.max_events_per_doc,
        event_extractor_name=args.event_extractor,
    )
    graph = builder.build(query, documents)
    dump_graph(graph, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
