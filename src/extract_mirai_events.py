from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from causal_graph import EvidenceSpan
from causal_graph import EventNode
from causal_graph import NewsDocument
from causal_graph import QuerySpec
from event_extraction import format_event_mention
from event_extraction import normalize_text
from event_extraction import split_sentences
from event_extractor import build_event_extractor
from mirai_dataset import load_mirai_news_for_docids
from mirai_dataset import load_mirai_queries
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


SCHEMA_VERSION = "event-input-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MIRAI documents into event-input-v1 JSONL. "
            "Default rule mode does not load any model; qwen mode is optional offline extraction."
        )
    )
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MIRAI_data.zip"), help="MIRAI_data.zip path.")
    parser.add_argument("--split", default="test", help="MIRAI split name.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum MIRAI query rows. Use 0 for full split.")
    parser.add_argument("--query-id", default=None, help="Optional single MIRAI QueryId.")
    parser.add_argument("--output", default=str(REPO_ROOT / "datasets" / "mirai_events_debug.jsonl"), help="Output event-input-v1 JSONL. Use '-' for stdout.")
    parser.add_argument("--extractor", choices=["rule", "qwen"], default="rule", help="Event extractor backend.")
    parser.add_argument("--max-docs", type=int, default=4, help="Maximum MIRAI documents per query.")
    parser.add_argument("--max-events", type=int, default=16, help="Maximum events per query.")
    parser.add_argument("--max-events-per-doc", type=int, default=6, help="Maximum events retained per document.")
    parser.add_argument("--min-events", type=int, default=2, help="Skip records with fewer events; downstream graph construction needs at least 2.")

    parser.add_argument("--model-path", default=str(REPO_ROOT / "models" / "Qwen3-4B"), help="Qwen model path, only used with --extractor qwen.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Qwen extraction temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Qwen extraction max new tokens.")
    parser.add_argument("--max-document-chars", type=int, default=900, help="Maximum document text chars in qwen extraction prompt.")
    parser.add_argument("--include-raw-qwen-response", action="store_true", help="Store raw qwen response in metadata for debugging.")
    return parser.parse_args()


def compact_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text).strip()
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
                try:
                    payload = json.loads(stripped[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def build_qwen_event_extraction_prompt(
    query: QuerySpec,
    documents: list[NewsDocument],
    max_events: int,
    max_document_chars: int,
) -> str:
    blocks: list[str] = []
    for document in documents:
        blocks.append(
            "\n".join(
                [
                    f"[Document {document.document_id}]",
                    f"Title: {compact_text(document.title, 180)}",
                    f"Date: {document.publish_time or '-'}",
                    f"Text: {compact_text(document.text, max_document_chars)}",
                ]
            )
        )
    return (
        "Extract concrete observed event mentions useful for forecasting the query.\n"
        "Use only the provided cutoff-before documents. Do not infer future events.\n"
        "Do not output relations or answers. Return strict JSON only.\n"
        "Schema:\n"
        "{\n"
        '  "events": [\n'
        "    {\n"
        '      "document_id": "same id as source document",\n'
        '      "trigger": "short event trigger word or phrase",\n'
        '      "mention": "one concrete event mention from the document",\n'
        '      "evidence": "source sentence or clause",\n'
        '      "participants": ["entity"],\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Return at most {max_events} events.\n\n"
        f"Query: {query.text}\n"
        f"Cutoff date: {query.cutoff_time or '-'}\n"
        f"Focus entities: {', '.join(query.focus_entities) or '-'}\n\n"
        "Documents:\n"
        + "\n\n".join(blocks)
    )


def locate_sentence(document: NewsDocument, evidence: str) -> tuple[int, str]:
    candidates = [document.title] + split_sentences(document.text)
    evidence_norm = normalize_text(evidence)
    for index, sentence in enumerate(candidates):
        sentence_norm = normalize_text(sentence)
        if evidence_norm and (evidence_norm in sentence_norm or sentence_norm in evidence_norm):
            return index, sentence
    return 0, evidence or document.title


def clamp01(value: Any, default: float = 0.7) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


def event_to_payload(event: EventNode) -> dict[str, Any]:
    trigger = str(event.metadata.get("trigger", "")).strip()
    mention = str(event.metadata.get("event_context", "")).strip()
    if not mention:
        match = re.fullmatch(r"\s*trigger=(.*?);\s*mention=(.*)\s*", event.text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            trigger = trigger or match.group(1).strip()
            mention = match.group(2).strip()
    if not mention:
        mention = event.text
    return {
        "event_id": event.event_id,
        "trigger": trigger or mention.split(" ")[0],
        "mention": mention,
        "participants": event.participants,
        "document_id": event.document_id,
        "sentence_index": event.sentence_index,
        "confidence": event.confidence,
        "evidence": [span.to_dict() for span in event.evidence],
        "metadata": event.metadata,
    }


def dedupe_events(events: list[EventNode]) -> list[EventNode]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[EventNode] = []
    for event in events:
        trigger = str(event.metadata.get("trigger", "")).lower()
        key = (event.document_id, trigger, normalize_text(event.text))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def extract_rule_events(
    query: QuerySpec,
    documents: list[NewsDocument],
    max_events: int,
    max_events_per_doc: int,
) -> tuple[list[EventNode], dict[str, Any]]:
    extractor = build_event_extractor("rule")
    events: list[EventNode] = []
    per_doc_counts: dict[str, int] = {}
    for document in documents:
        extracted = extractor.extract_document(query, document)
        doc_count = 0
        for sentence_index, atomic_event in extracted.iter_events():
            if doc_count >= max_events_per_doc:
                break
            event_id = f"e{len(events) + 1}"
            mention = atomic_event.text
            trigger = atomic_event.trigger or mention.split(" ")[0]
            sentence_text = mention
            split_candidates = [document.title] + split_sentences(document.text)
            if 0 <= sentence_index < len(split_candidates):
                sentence_text = split_candidates[sentence_index]
            event_text = format_event_mention(trigger=trigger, context=mention)
            events.append(
                EventNode(
                    event_id=event_id,
                    text=event_text,
                    normalized_text=atomic_event.normalized_text,
                    document_id=document.document_id,
                    sentence_index=sentence_index,
                    participants=atomic_event.participants,
                    node_type="observed",
                    confidence=atomic_event.score,
                    evidence=[EvidenceSpan(document_id=document.document_id, sentence_index=sentence_index, text=sentence_text)],
                    metadata={
                        "trigger": trigger,
                        "event_mention": event_text,
                        "event_context": mention,
                        "sentence_text": sentence_text,
                        "publish_time": document.publish_time,
                        "extracted_by": "rule",
                    },
                )
            )
            doc_count += 1
            if len(events) >= max_events:
                break
        per_doc_counts[document.document_id] = doc_count
        if len(events) >= max_events:
            break
    events = dedupe_events(events)[:max_events]
    for index, event in enumerate(events, start=1):
        event.event_id = f"e{index}"
    return events, {"extractor_name": "rule", "per_doc_event_counts": per_doc_counts}


def extract_qwen_events(
    query: QuerySpec,
    documents: list[NewsDocument],
    args: argparse.Namespace,
) -> tuple[list[EventNode], dict[str, Any]]:
    from local_llm import LocalQwenGenerator

    generator = LocalQwenGenerator(resolve_repo_path(args.model_path), max_new_tokens=args.max_new_tokens)
    prompt = build_qwen_event_extraction_prompt(
        query=query,
        documents=documents,
        max_events=args.max_events,
        max_document_chars=args.max_document_chars,
    )
    raw_response = generator.generate(
        prompt,
        temperature=args.temperature,
        system_prompt="You extract concrete observed event mentions and return strict JSON only.",
        max_new_tokens=args.max_new_tokens,
    )
    payload = extract_first_json_object(raw_response)
    metadata: dict[str, Any] = {
        "extractor_name": "qwen",
        "parsed_json": payload is not None,
        "prompt_chars": len(prompt),
    }
    if args.include_raw_qwen_response:
        metadata["raw_qwen_response"] = raw_response
    if payload is None or not isinstance(payload.get("events", []), list):
        return [], {**metadata, "raw_event_count": 0}

    doc_lookup = {document.document_id: document for document in documents}
    events: list[EventNode] = []
    raw_events = payload.get("events", [])
    metadata["raw_event_count"] = len(raw_events)
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        document_id = str(item.get("document_id", "")).strip()
        document = doc_lookup.get(document_id)
        if document is None:
            continue
        mention = str(item.get("mention", item.get("event", ""))).strip()
        evidence = str(item.get("evidence", mention)).strip()
        if not mention and not evidence:
            continue
        trigger = str(item.get("trigger", "")).strip() or (mention or evidence).split(" ")[0]
        sentence_index, sentence_text = locate_sentence(document, evidence or mention)
        participants = item.get("participants", [])
        if not isinstance(participants, list):
            participants = []
        event_text = format_event_mention(trigger=trigger, context=mention or sentence_text)
        events.append(
            EventNode(
                event_id=f"e{len(events) + 1}",
                text=event_text,
                normalized_text=normalize_text(mention or sentence_text),
                document_id=document_id,
                sentence_index=sentence_index,
                participants=[str(participant) for participant in participants],
                node_type="observed",
                confidence=clamp01(item.get("confidence", 0.7)),
                evidence=[EvidenceSpan(document_id=document_id, sentence_index=sentence_index, text=sentence_text)],
                metadata={
                    "trigger": trigger,
                    "event_mention": event_text,
                    "event_context": mention or sentence_text,
                    "sentence_text": sentence_text,
                    "publish_time": document.publish_time,
                    "extracted_by": "qwen",
                },
            )
        )
        if len(events) >= args.max_events:
            break
    events = dedupe_events(events)[: args.max_events]
    for index, event in enumerate(events, start=1):
        event.event_id = f"e{index}"
    return events, metadata


def build_event_input_record(
    query: QuerySpec,
    documents: list[NewsDocument],
    events: list[EventNode],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"mirai_{query.query_id}",
        "query_id": query.query_id,
        "query": query.to_dict(),
        "documents": [document.to_dict() for document in documents],
        "events": [event_to_payload(event) for event in events],
        "metadata": metadata,
    }


def write_jsonl(path_arg: str, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path_arg == "-":
        sys.stdout.write(text)
        return
    path = resolve_repo_path(path_arg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    if args.query_id:
        examples = [example for example in load_mirai_queries(dataset_path, split=args.split) if example.query_id == str(args.query_id)]
    else:
        examples = load_mirai_queries(dataset_path, split=args.split, limit=args.limit if args.limit > 0 else 0)
    if not examples:
        raise RuntimeError(f"No MIRAI examples found for split={args.split!r}, query_id={args.query_id!r}.")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_events = 0
    for index, example in enumerate(examples, start=1):
        query = example.build_query_spec()
        documents = load_mirai_news_for_docids(dataset_path, example.docids)[: args.max_docs]
        if args.extractor == "rule":
            events, extraction_metadata = extract_rule_events(
                query=query,
                documents=documents,
                max_events=args.max_events,
                max_events_per_doc=args.max_events_per_doc,
            )
        else:
            events, extraction_metadata = extract_qwen_events(query=query, documents=documents, args=args)

        if len(events) < args.min_events:
            skipped.append({"query_id": example.query_id, "event_count": len(events), "reason": "too_few_events"})
            continue
        total_events += len(events)
        rows.append(
            build_event_input_record(
                query=query,
                documents=documents,
                events=events,
                metadata={
                    "dataset": "MIRAI",
                    "split": args.split,
                    "event_source": f"{args.extractor}_offline_extractor",
                    "extractor_name": args.extractor,
                    "source_docids": example.docids[: args.max_docs],
                    "answer_list": example.answer_list,
                    "relation_name": example.relation_name,
                    "event_base_code": example.event_base_code,
                    "extraction": extraction_metadata,
                },
            )
        )
        if args.output != "-" and (index % 25 == 0 or index == len(examples)):
            print(f"processed {index}/{len(examples)} | kept={len(rows)} | skipped={len(skipped)}", flush=True)

    write_jsonl(args.output, rows)
    summary = {
        "extractor": args.extractor,
        "split": args.split,
        "requested_examples": len(examples),
        "written_records": len(rows),
        "skipped_records": len(skipped),
        "total_events": total_events,
        "average_events_per_record": total_events / len(rows) if rows else 0.0,
        "output": args.output,
        "skipped_preview": skipped[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr if args.output == "-" else sys.stdout)


if __name__ == "__main__":
    main()
