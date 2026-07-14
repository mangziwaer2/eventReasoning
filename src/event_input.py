from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from causal_graph import EvidenceSpan
from causal_graph import EventNode
from causal_graph import NewsDocument
from causal_graph import QuerySpec
from event_extraction import format_event_mention
from event_extraction import normalize_text
from path_utils import resolve_repo_path


EVENT_INPUT_SCHEMA_VERSION = "event-input-v1"


class EventInputValidationError(ValueError):
    pass


@dataclass(slots=True)
class EventInputRecord:
    sample_id: str
    query_id: str
    events: list[EventNode]
    query: QuerySpec | None = None
    documents: list[NewsDocument] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = EVENT_INPUT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "query_id": self.query_id,
            "query": self.query.to_dict() if self.query is not None else None,
            "documents": [document.to_dict() for document in self.documents],
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
        }


def _as_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EventInputValidationError(f"{field_name} must be a JSON object.")
    return value


def _as_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise EventInputValidationError(f"{field_name} must be a JSON array.")
    return value


def _required_text(value: Any, field_name: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        raise EventInputValidationError(f"{field_name} is required and cannot be empty.")
    return text


def _parse_query(data: dict[str, Any]) -> QuerySpec:
    query_id = _required_text(data.get("query_id"), "query.query_id")
    text = _required_text(data.get("text"), "query.text")
    focus_entities = _as_list(data.get("focus_entities", []), "query.focus_entities")
    metadata = _as_dict(data.get("metadata", {}), "query.metadata")
    return QuerySpec(
        query_id=query_id,
        text=text,
        cutoff_time=data.get("cutoff_time"),
        focus_entities=[str(item) for item in focus_entities],
        metadata=dict(metadata),
    )


def _parse_document(data: dict[str, Any], index: int) -> NewsDocument:
    prefix = f"documents[{index}]"
    document_id = _required_text(data.get("document_id", data.get("doc_id")), f"{prefix}.document_id")
    metadata = _as_dict(data.get("metadata", {}), f"{prefix}.metadata")
    return NewsDocument(
        document_id=document_id,
        title=str(data.get("title", "")),
        text=str(data.get("text", "")),
        publish_time=data.get("publish_time", data.get("published_at")),
        source=str(data.get("source", "preextracted")),
        metadata=dict(metadata),
    )


def _parse_formatted_event_text(text: str) -> tuple[str, str]:
    match = re.fullmatch(r"\s*trigger=(.*?);\s*mention=(.*)\s*", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def _parse_evidence(
    value: Any,
    event_index: int,
    document_id: str,
    sentence_index: int,
    mention: str,
) -> list[EvidenceSpan]:
    if value is None:
        raw_items: list[Any] = []
    elif isinstance(value, (dict, str)):
        raw_items = [value]
    else:
        raw_items = _as_list(value, f"events[{event_index}].evidence")

    evidence: list[EvidenceSpan] = []
    for evidence_index, item in enumerate(raw_items):
        if isinstance(item, str):
            text = item.strip()
            if text:
                evidence.append(EvidenceSpan(document_id=document_id, sentence_index=sentence_index, text=text))
            continue
        item_data = _as_dict(item, f"events[{event_index}].evidence[{evidence_index}]")
        evidence_document_id = str(item_data.get("document_id", item_data.get("doc_id", document_id))).strip()
        evidence_text = str(item_data.get("text", item_data.get("sentence", ""))).strip()
        try:
            evidence_sentence_index = int(item_data.get("sentence_index", sentence_index))
        except (TypeError, ValueError) as exc:
            raise EventInputValidationError(
                f"events[{event_index}].evidence[{evidence_index}].sentence_index must be an integer."
            ) from exc
        if not evidence_document_id or not evidence_text:
            raise EventInputValidationError(
                f"events[{event_index}].evidence[{evidence_index}] requires document_id and text."
            )
        evidence.append(
            EvidenceSpan(
                document_id=evidence_document_id,
                sentence_index=evidence_sentence_index,
                text=evidence_text,
            )
        )
    if not evidence:
        evidence.append(EvidenceSpan(document_id=document_id, sentence_index=sentence_index, text=mention))
    return evidence


def _parse_event(data: dict[str, Any], index: int) -> EventNode:
    prefix = f"events[{index}]"
    metadata = dict(_as_dict(data.get("metadata", {}), f"{prefix}.metadata"))
    event_id = _required_text(data.get("event_id"), f"{prefix}.event_id")
    document_id = _required_text(data.get("document_id", data.get("doc_id")), f"{prefix}.document_id")

    trigger = str(data.get("trigger", metadata.get("trigger", ""))).strip()
    mention = str(
        data.get(
            "mention",
            data.get("event", metadata.get("event_context", "")),
        )
    ).strip()
    if not trigger or not mention:
        parsed_trigger, parsed_mention = _parse_formatted_event_text(str(data.get("text", "")))
        trigger = trigger or parsed_trigger
        mention = mention or parsed_mention
    trigger = _required_text(trigger, f"{prefix}.trigger")
    mention = _required_text(mention, f"{prefix}.mention")

    try:
        sentence_index = int(data.get("sentence_index", 0))
    except (TypeError, ValueError) as exc:
        raise EventInputValidationError(f"{prefix}.sentence_index must be an integer.") from exc
    if sentence_index < 0:
        raise EventInputValidationError(f"{prefix}.sentence_index must be non-negative.")

    try:
        confidence = float(data.get("confidence", 1.0))
    except (TypeError, ValueError) as exc:
        raise EventInputValidationError(f"{prefix}.confidence must be numeric.") from exc
    if not 0.0 <= confidence <= 1.0:
        raise EventInputValidationError(f"{prefix}.confidence must be within [0, 1].")

    participants = _as_list(data.get("participants", []), f"{prefix}.participants")
    event_text = format_event_mention(trigger=trigger, context=mention)
    metadata.update(
        {
            "trigger": trigger,
            "event_mention": event_text,
            "event_context": mention,
            "event_input_schema": EVENT_INPUT_SCHEMA_VERSION,
        }
    )
    evidence = _parse_evidence(
        data.get("evidence"),
        event_index=index,
        document_id=document_id,
        sentence_index=sentence_index,
        mention=mention,
    )
    return EventNode(
        event_id=event_id,
        text=event_text,
        normalized_text=str(data.get("normalized_text", "")).strip() or normalize_text(mention),
        document_id=document_id,
        sentence_index=sentence_index,
        participants=[str(item) for item in participants],
        node_type=str(data.get("node_type", "observed")),
        confidence=confidence,
        evidence=evidence,
        metadata=metadata,
    )


def parse_event_input_record(data: dict[str, Any], source: str = "<memory>") -> EventInputRecord:
    schema_version = str(data.get("schema_version", EVENT_INPUT_SCHEMA_VERSION)).strip()
    if schema_version != EVENT_INPUT_SCHEMA_VERSION:
        raise EventInputValidationError(
            f"{source}: unsupported schema_version={schema_version!r}; expected {EVENT_INPUT_SCHEMA_VERSION!r}."
        )
    if data.get("edges"):
        raise EventInputValidationError(
            f"{source}: event input must not contain relation edges; graph edges are model targets or predictions."
        )

    query_data = data.get("query")
    query = _parse_query(_as_dict(query_data, "query")) if query_data is not None else None
    query_id = str(data.get("query_id", query.query_id if query is not None else "")).strip()
    query_id = _required_text(query_id, "query_id")
    if query is not None and query.query_id != query_id:
        raise EventInputValidationError(
            f"{source}: query_id={query_id!r} does not match query.query_id={query.query_id!r}."
        )

    sample_id = str(data.get("sample_id", f"events_{query_id}")).strip() or f"events_{query_id}"
    raw_documents = _as_list(data.get("documents", []), "documents")
    documents = [_parse_document(_as_dict(item, f"documents[{index}]"), index) for index, item in enumerate(raw_documents)]
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise EventInputValidationError(f"{source}: document_id values must be unique.")

    raw_events = _as_list(data.get("events"), "events")
    if not raw_events:
        raise EventInputValidationError(f"{source}: events cannot be empty.")
    events = [_parse_event(_as_dict(item, f"events[{index}]"), index) for index, item in enumerate(raw_events)]
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise EventInputValidationError(f"{source}: event_id values must be unique within a sample.")

    if documents:
        validate_events_against_documents(events, documents, source=source)
    metadata = dict(_as_dict(data.get("metadata", {}), "metadata"))
    metadata.setdefault("event_source", "precomputed")
    metadata.setdefault("source_file", source)
    return EventInputRecord(
        sample_id=sample_id,
        query_id=query_id,
        query=query,
        documents=documents,
        events=events,
        metadata=metadata,
        schema_version=schema_version,
    )


def _read_payloads(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        payloads: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EventInputValidationError(f"{path}:{line_number}: invalid JSON: {exc.msg}.") from exc
            payloads.append(_as_dict(payload, f"{path}:{line_number}"))
        return payloads

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EventInputValidationError(f"{path}: invalid JSON: {exc.msg}.") from exc
    if isinstance(payload, list):
        return [_as_dict(item, f"{path}[{index}]") for index, item in enumerate(payload)]
    payload_data = _as_dict(payload, str(path))
    if "samples" in payload_data:
        return [
            _as_dict(item, f"{path}.samples[{index}]")
            for index, item in enumerate(_as_list(payload_data["samples"], "samples"))
        ]
    return [payload_data]


def load_event_input_records(path: Path) -> list[EventInputRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Pre-extracted event input was not found: {path}")
    records = [parse_event_input_record(payload, source=str(path)) for payload in _read_payloads(path)]
    if not records:
        raise EventInputValidationError(f"{path}: no event input records were found.")
    query_ids = [record.query_id for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise EventInputValidationError(f"{path}: query_id values must be unique across records.")
    return records


def load_event_input_index(path: Path) -> dict[str, EventInputRecord]:
    return {record.query_id: record for record in load_event_input_records(path)}


def select_event_input_record(path: Path, query_id: str | None = None) -> EventInputRecord:
    records = load_event_input_records(path)
    if query_id is None:
        if len(records) != 1:
            raise EventInputValidationError(
                f"{path}: contains {len(records)} records; provide query_id to select one."
            )
        return records[0]
    for record in records:
        if record.query_id == str(query_id):
            return record
    raise EventInputValidationError(f"{path}: query_id={query_id!r} was not found.")


def validate_events_against_documents(
    events: list[EventNode],
    documents: list[NewsDocument],
    source: str = "event input",
) -> None:
    document_ids = {document.document_id for document in documents}
    missing = sorted({event.document_id for event in events if event.document_id not in document_ids})
    if missing:
        raise EventInputValidationError(
            f"{source}: events reference document_id values not present in the active documents: {missing}."
        )
    for event in events:
        for evidence in event.evidence:
            if evidence.document_id not in document_ids:
                raise EventInputValidationError(
                    f"{source}: event {event.event_id!r} has evidence for unknown document_id={evidence.document_id!r}."
                )


def materialize_event_input(
    record: EventInputRecord,
    query: QuerySpec | None = None,
    documents: list[NewsDocument] | None = None,
    max_events: int | None = None,
) -> tuple[QuerySpec, list[NewsDocument], list[EventNode]]:
    active_query = query or record.query
    if active_query is None:
        raise EventInputValidationError(
            f"query_id={record.query_id!r}: query is missing from both the event input and dataset context."
        )
    if active_query.query_id != record.query_id:
        raise EventInputValidationError(
            f"Event input query_id={record.query_id!r} does not match active query_id={active_query.query_id!r}."
        )

    active_documents = list(documents) if documents is not None else list(record.documents)
    if not active_documents:
        raise EventInputValidationError(
            f"query_id={record.query_id!r}: documents are missing from both the event input and dataset context."
        )
    validate_events_against_documents(record.events, active_documents, source=f"query_id={record.query_id}")
    events = list(record.events)
    if max_events is not None and max_events > 0:
        events = events[:max_events]
    if len(events) < 2:
        raise EventInputValidationError(
            f"query_id={record.query_id!r}: at least two valid events are required for graph construction."
        )
    return active_query, active_documents, events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pre-extracted event input files.")
    parser.add_argument("--input", required=True, help="event-input-v1 JSON or JSONL path.")
    parser.add_argument("--query-id", default=None, help="Optional query id to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    records = load_event_input_records(input_path)
    if args.query_id is not None:
        records = [select_event_input_record(input_path, args.query_id)]
    summary = {
        "schema_version": EVENT_INPUT_SCHEMA_VERSION,
        "path": str(input_path),
        "records": len(records),
        "events": sum(len(record.events) for record in records),
        "query_ids": [record.query_id for record in records[:20]],
        "records_with_embedded_query": sum(record.query is not None for record in records),
        "records_with_embedded_documents": sum(bool(record.documents) for record in records),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
