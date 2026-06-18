from __future__ import annotations

import argparse
import io
import json
import random
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
from event_extraction import format_event_mention
from event_extraction import lexical_overlap
from mirai_dataset import export_mirai_query_snapshot
from mirai_dataset import get_mirai_query_by_id
from mirai_dataset import load_mirai_news_for_docids
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path
from query_causal_graph import QueryCausalGraphBuilder
from query_causal_graph import build_query
from query_causal_graph import load_news_jsonl


RELATION_TYPES = ("precedes", "causes", "escalates", "mitigates")
PAIR_RELATION_TYPES = ("none",) + RELATION_TYPES
RELATION_TO_ID = {relation: index for index, relation in enumerate(RELATION_TYPES)}
PAIR_RELATION_TO_ID = {relation: index for index, relation in enumerate(PAIR_RELATION_TYPES)}
ID_TO_RELATION = {index: relation for relation, index in RELATION_TO_ID.items()}
PAIR_ID_TO_RELATION = {index: relation for relation, index in PAIR_RELATION_TO_ID.items()}
RELATION_PRIORITY = {
    "causes": 4,
    "escalates": 3,
    "mitigates": 2,
    "precedes": 1,
}


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

    def render_events(self) -> str:
        lines: list[str] = []
        for event in self.events:
            trigger = str(event.metadata.get("trigger", "")).strip()
            trigger_text = f" | trigger={trigger}" if trigger else ""
            lines.append(
                f"- {event.event_id} | doc={event.document_id} | sent={event.sentence_index}{trigger_text} | event={event.text}"
            )
        return "\n".join(lines)

    def render_documents(self, mode: str = "snippet", max_chars_per_doc: int = 320) -> str:
        normalized_mode = _normalize_document_mode(mode)
        if normalized_mode == "none":
            return ""
        parts: list[str] = []
        for document in self.documents:
            lines = [f"[Document {document.document_id}]"]
            if document.title:
                lines.append(f"Title: {document.title}")
            if normalized_mode == "full":
                lines.append(f"Text: {document.text}")
            elif normalized_mode == "snippet":
                snippet = _compact_text(document.text, max_chars=max_chars_per_doc)
                if snippet:
                    lines.append(f"Snippet: {snippet}")
            elif normalized_mode != "title":
                raise ValueError(f"Unsupported document render mode: {mode}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def render_gold_target(self) -> str:
        if self.gold_graph is None:
            return json.dumps({"edges": []}, ensure_ascii=False)
        return json.dumps({"edges": graph_edges_to_payload(self.gold_graph)}, ensure_ascii=False)


@dataclass(slots=True)
class EventPairSample:
    sample_id: str
    query: QuerySpec
    documents: list[NewsDocument]
    events: list[EventNode]
    source_event_id: str
    target_event_id: str
    relation_type: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query.to_dict(),
            "documents": [document.to_dict() for document in self.documents],
            "events": [event.to_dict() for event in self.events],
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "relation_type": self.relation_type,
            "score": self.score,
            "metadata": self.metadata,
        }

    def to_instruction_example(
        self,
        include_query: bool = False,
        document_mode: str = "title",
        max_document_chars: int = 240,
    ) -> dict[str, str]:
        event_lookup = {event.event_id: event for event in self.events}
        document_lookup = {document.document_id: document for document in self.documents}
        source_event = event_lookup[self.source_event_id]
        target_event = event_lookup[self.target_event_id]
        source_document = document_lookup.get(source_event.document_id)
        target_document = document_lookup.get(target_event.document_id)

        prompt_lines = [
            "You classify the relation between two candidate events.",
            "Return strict JSON with the schema {\"relation_type\": ..., \"score\": ...} only.",
            "Allowed relation_type values: none, precedes, causes, escalates, mitigates.",
            "Use none when there is no supported directed relation from source_event to target_event.",
        ]
        if include_query and self.query.text:
            prompt_lines.extend(["", f"Query: {self.query.text}"])
        prompt_lines.extend(
            [
                "",
                "Source Event:",
                self._render_event_block(source_event, source_document),
                "",
                "Target Event:",
                self._render_event_block(target_event, target_document),
            ]
        )

        document_block = self._render_document_context(
            source_event=source_event,
            target_event=target_event,
            source_document=source_document,
            target_document=target_document,
            mode=document_mode,
            max_document_chars=max_document_chars,
        )
        if document_block:
            prompt_lines.extend(["", "Document Context:", document_block])

        prompt_lines.extend(["", "Metadata:", self._render_pair_metadata(source_event, target_event)])
        return {
            "sample_id": self.sample_id,
            "prompt": "\n".join(prompt_lines),
            "target": json.dumps(
                {"relation_type": self.relation_type, "score": round(float(self.score), 4)},
                ensure_ascii=False,
            ),
            "metadata": self.metadata,
        }

    def _render_event_block(self, event: EventNode, document: NewsDocument | None) -> str:
        trigger = str(event.metadata.get("trigger", "")).strip()
        lines = [
            f"id={event.event_id}",
            f"doc={event.document_id}",
            f"sent={event.sentence_index}",
        ]
        if trigger:
            lines.append(f"trigger={trigger}")
        if document is not None and document.title:
            lines.append(f"title={document.title}")
        lines.append(f"event={event.text}")
        context = str(event.metadata.get("event_context") or event.metadata.get("sentence_text") or "").strip()
        if context and context != event.text:
            lines.append(f"context={context}")
        return "\n".join(lines)

    def _render_document_context(
        self,
        source_event: EventNode,
        target_event: EventNode,
        source_document: NewsDocument | None,
        target_document: NewsDocument | None,
        mode: str,
        max_document_chars: int,
    ) -> str:
        normalized_mode = _normalize_document_mode(mode)
        if normalized_mode == "none":
            return ""

        rows: list[str] = []
        seen_document_ids: set[str] = set()
        for event, document, role in (
            (source_event, source_document, "Source"),
            (target_event, target_document, "Target"),
        ):
            if document is None or document.document_id in seen_document_ids:
                continue
            seen_document_ids.add(document.document_id)
            lines = [f"[{role} Document {document.document_id}]"]
            if document.title:
                lines.append(f"Title: {document.title}")
            if normalized_mode == "full":
                lines.append(f"Text: {document.text}")
            elif normalized_mode == "snippet":
                lines.append(f"Snippet: {_compact_text(document.text, max_chars=max_document_chars)}")
            rows.append("\n".join(lines))
        return "\n\n".join(rows)

    def _render_pair_metadata(self, source_event: EventNode, target_event: EventNode) -> str:
        same_document = int(source_event.document_id == target_event.document_id)
        sentence_gap = target_event.sentence_index - source_event.sentence_index
        shared_participants = sorted(set(source_event.participants) & set(target_event.participants))
        shared_text = ", ".join(shared_participants) if shared_participants else "none"
        return "\n".join(
            [
                f"same_document={same_document}",
                f"sentence_gap={sentence_gap}",
                f"shared_participants={shared_text}",
            ]
        )


def _normalize_document_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "summary":
        return "snippet"
    return normalized


def _compact_text(text: str, max_chars: int = 320) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max(0, max_chars - 3)].rstrip() + "..."


def _read_maven_rows(zip_path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        requested_split = split.strip().lower()
        if requested_split in {"validation", "valid", "val", "dev"}:
            split_candidates = ["validation", "valid", "val", "dev"]
        else:
            split_candidates = [requested_split]
        available_members = set(archive.namelist())
        member_name = None
        for candidate in split_candidates:
            candidate_name = f"MAVEN_ERE/{candidate}.jsonl"
            if candidate_name in available_members:
                member_name = candidate_name
                break
        if member_name is None:
            available_jsonl = sorted(
                name for name in available_members if name.startswith("MAVEN_ERE/") and name.endswith(".jsonl")
            )
            raise FileNotFoundError(
                f"Split '{split}' not found in {zip_path}. Available MAVEN jsonl members: {available_jsonl}"
            )
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


def _mention_context(sentence_tokens: list[str] | None, sentence_text: str, offset: Any, trigger_word: str) -> str:
    if not sentence_tokens or not isinstance(offset, list) or len(offset) != 2:
        return sentence_text
    try:
        start = max(0, int(offset[0]))
        end = max(start + 1, int(offset[1]))
    except (TypeError, ValueError):
        return sentence_text
    left = max(0, start - 8)
    right = min(len(sentence_tokens), end + 8)
    window = sentence_tokens[left:right]
    if not window:
        return sentence_text
    context = " ".join(str(token) for token in window)
    if left > 0:
        context = "... " + context
    if right < len(sentence_tokens):
        context = context + " ..."
    return context or trigger_word or sentence_text


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
    token_sentences = row.get("tokens", [])
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
        offset = first_mention.get("offset", [])
        sentence_tokens = token_sentences[sent_id] if sent_id < len(token_sentences) else None
        mention_context = _mention_context(sentence_tokens, sentence_text, offset, trigger_word)
        event_mention = format_event_mention(trigger=trigger_word, context=mention_context)
        node_id = f"maven_e{event_index}"
        source_event_id_to_node_id[str(event.get("id", node_id))] = node_id
        events.append(
            EventNode(
                event_id=node_id,
                text=event_mention,
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
                    "event_mention": event_mention,
                    "event_context": mention_context,
                    "sentence_text": sentence_text,
                    "mention_id": first_mention.get("id", ""),
                    "offset": offset,
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


def _relation_label_map(graph: CoarseCausalGraph) -> dict[tuple[str, str], tuple[str, float]]:
    pair_to_label: dict[tuple[str, str], tuple[str, float]] = {}
    for edge in graph.edges:
        pair = (edge.source_event_id, edge.target_event_id)
        current = pair_to_label.get(pair)
        candidate = (edge.relation_type, float(edge.score))
        if current is None or RELATION_PRIORITY.get(candidate[0], 0) > RELATION_PRIORITY.get(current[0], 0):
            pair_to_label[pair] = candidate
    return pair_to_label


def _pair_candidate_score(source_event: EventNode, target_event: EventNode, query_text: str) -> float:
    shared_participants = len(set(source_event.participants) & set(target_event.participants))
    lexical_score, _ = lexical_overlap(source_event.text, target_event.text)
    query_source_score, _ = lexical_overlap(query_text, source_event.text)
    query_target_score, _ = lexical_overlap(query_text, target_event.text)
    same_document = 1.0 if source_event.document_id == target_event.document_id else 0.0
    sentence_gap = abs(target_event.sentence_index - source_event.sentence_index)
    distance_score = 1.0 / (1.0 + float(sentence_gap))
    return (
        0.35 * same_document
        + 0.25 * distance_score
        + 0.20 * lexical_score
        + 0.10 * min(shared_participants, 3) / 3.0
        + 0.10 * ((query_source_score + query_target_score) / 2.0)
    )


def _candidate_pairs_for_training(
    sample: DocumentGraphSample,
    max_sentence_gap: int,
) -> list[tuple[str, str, float]]:
    candidates: list[tuple[str, str, float]] = []
    for source_event in sample.events:
        for target_event in sample.events:
            if source_event.event_id == target_event.event_id:
                continue
            if source_event.document_id != target_event.document_id:
                continue
            if abs(target_event.sentence_index - source_event.sentence_index) > max_sentence_gap:
                continue
            score = _pair_candidate_score(source_event, target_event, sample.query.text)
            candidates.append((source_event.event_id, target_event.event_id, score))
    return candidates


def load_maven_event_pair_samples(
    dataset_path: Path,
    split: str = "train",
    limit: int | None = None,
    max_events: int | None = None,
    negative_ratio: float = 1.0,
    max_sentence_gap: int = 3,
    seed: int = 7,
) -> list[EventPairSample]:
    rng = random.Random(seed)
    document_samples = load_maven_document_graph_samples(
        dataset_path=dataset_path,
        split=split,
        limit=limit,
        max_events=max_events,
    )

    pair_samples: list[EventPairSample] = []
    for document_sample in document_samples:
        if document_sample.gold_graph is None:
            continue
        positive_map = _relation_label_map(document_sample.gold_graph)
        if not positive_map:
            continue

        for index, ((source_event_id, target_event_id), (relation_type, relation_score)) in enumerate(positive_map.items()):
            pair_samples.append(
                EventPairSample(
                    sample_id=f"{document_sample.sample_id}_pos_{index}",
                    query=document_sample.query,
                    documents=document_sample.documents,
                    events=document_sample.events,
                    source_event_id=source_event_id,
                    target_event_id=target_event_id,
                    relation_type=relation_type,
                    score=relation_score,
                    metadata={
                        **document_sample.metadata,
                        "pair_label": relation_type,
                        "is_negative": False,
                    },
                )
            )

        negative_candidates = [
            (source_event_id, target_event_id, candidate_score)
            for source_event_id, target_event_id, candidate_score in _candidate_pairs_for_training(
                document_sample,
                max_sentence_gap=max_sentence_gap,
            )
            if (source_event_id, target_event_id) not in positive_map
        ]
        rng.shuffle(negative_candidates)
        negative_limit = max(1, int(len(positive_map) * max(negative_ratio, 0.0)))
        for index, (source_event_id, target_event_id, candidate_score) in enumerate(negative_candidates[:negative_limit]):
            pair_samples.append(
                EventPairSample(
                    sample_id=f"{document_sample.sample_id}_neg_{index}",
                    query=document_sample.query,
                    documents=document_sample.documents,
                    events=document_sample.events,
                    source_event_id=source_event_id,
                    target_event_id=target_event_id,
                    relation_type="none",
                    score=min(round(candidate_score, 4), 0.49),
                    metadata={
                        **document_sample.metadata,
                        "pair_label": "none",
                        "is_negative": True,
                    },
                )
            )
    return pair_samples


def build_event_pair_inference_samples(
    sample: DocumentGraphSample,
    max_sentence_gap: int = 3,
    max_pairs: int = 64,
) -> list[EventPairSample]:
    pair_candidates: list[tuple[float, str, str]] = []
    for source_event in sample.events:
        for target_event in sample.events:
            if source_event.event_id == target_event.event_id:
                continue
            if source_event.document_id == target_event.document_id:
                if abs(target_event.sentence_index - source_event.sentence_index) > max_sentence_gap:
                    continue
            else:
                shared_participants = set(source_event.participants) & set(target_event.participants)
                lexical_score, _ = lexical_overlap(source_event.text, target_event.text)
                if not shared_participants and lexical_score < 0.12:
                    continue
            score = _pair_candidate_score(source_event, target_event, sample.query.text)
            pair_candidates.append((score, source_event.event_id, target_event.event_id))

    pair_candidates.sort(key=lambda item: item[0], reverse=True)
    if max_pairs is None or max_pairs <= 0:
        selected = pair_candidates
    else:
        selected = pair_candidates[:max_pairs]
    pair_samples: list[EventPairSample] = []
    for index, (candidate_score, source_event_id, target_event_id) in enumerate(selected):
        pair_samples.append(
            EventPairSample(
                sample_id=f"{sample.sample_id}_pair_{index}",
                query=sample.query,
                documents=sample.documents,
                events=sample.events,
                source_event_id=source_event_id,
                target_event_id=target_event_id,
                relation_type="none",
                score=min(round(candidate_score, 4), 0.49),
                metadata={
                    **sample.metadata,
                    "candidate_score": round(candidate_score, 4),
                    "inference_pair": True,
                },
            )
        )
    return pair_samples


def parse_pair_payload(text: str) -> dict[str, Any] | None:
    json_block = _extract_json_block(text)
    if json_block is None:
        return None
    try:
        payload = json.loads(json_block)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    relation_type = str(payload.get("relation_type", "")).strip().lower()
    if relation_type not in PAIR_RELATION_TO_ID:
        return None
    raw_score = payload.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    score = round(min(max(score, 0.0), 1.0), 4)
    return {
        "relation_type": relation_type,
        "score": score,
    }


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
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return None


def build_graph_from_pair_predictions(
    document_sample: DocumentGraphSample,
    pair_samples: list[EventPairSample],
    pair_predictions: list[dict[str, Any] | None],
    keep_threshold: float = 0.5,
) -> CoarseCausalGraph:
    event_lookup = {event.event_id: event for event in document_sample.events}
    edges: list[CoarseCausalEdge] = []
    for edge_index, (pair_sample, prediction) in enumerate(zip(pair_samples, pair_predictions)):
        if prediction is None:
            continue
        relation_type = str(prediction.get("relation_type", "")).strip().lower()
        score = float(prediction.get("score", 0.0))
        if relation_type == "none" or relation_type not in RELATION_TO_ID or score < keep_threshold:
            continue
        source_event = event_lookup[pair_sample.source_event_id]
        target_event = event_lookup[pair_sample.target_event_id]
        edges.append(
            CoarseCausalEdge(
                edge_id=f"pred_edge_{edge_index}",
                source_event_id=pair_sample.source_event_id,
                target_event_id=pair_sample.target_event_id,
                relation_type=relation_type,
                score=round(score, 4),
                evidence=source_event.evidence + target_event.evidence,
                feature_scores={},
                metadata={
                    "generated_by": "qwen_pair_classifier",
                    "candidate_score": pair_sample.metadata.get("candidate_score"),
                },
            )
        )
    return CoarseCausalGraph(
        query=document_sample.query,
        documents=document_sample.documents,
        events=document_sample.events,
        edges=_dedupe_edges(edges),
        trace=GraphBuildTrace(),
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
    parser = argparse.ArgumentParser(description="Export document and pair-level samples for coarse graph training.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="train", help="MAVEN split name.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of MAVEN rows.")
    parser.add_argument("--max-events", type=int, default=0, help="Optional cap on total events per sample.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative pair sampling ratio.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document_samples = load_maven_document_graph_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit,
        max_events=args.max_events or None,
    )
    pair_samples = load_maven_event_pair_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit,
        max_events=args.max_events or None,
        negative_ratio=args.negative_ratio,
    )
    payload = {
        "document_samples": [sample.to_dict() for sample in document_samples],
        "pair_samples": [sample.to_dict() for sample in pair_samples[: min(32, len(pair_samples))]],
        "instruction_samples": [
            sample.to_instruction_example(include_query=False, document_mode="title")
            for sample in pair_samples[: min(8, len(pair_samples))]
        ],
    }
    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        resolve_repo_path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
