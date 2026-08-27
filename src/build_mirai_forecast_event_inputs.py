from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from causal_graph import NewsDocument, QuerySpec
from event_input import parse_event_input_record
from extract_mirai_events import build_event_input_record, extract_rule_events
from path_utils import REPO_ROOT, resolve_repo_path


SOURCE_SCHEMA_VERSION = "mirai-forecast-v1"
OUTPUT_SCHEMA_VERSION = "event-input-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert mirai_forecast JSONL samples to event-input-v1 using the "
            "offline rule event extractor."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            str(REPO_ROOT / "datasets" / "mirai_forecast" / "train.jsonl"),
            str(REPO_ROOT / "datasets" / "mirai_forecast" / "dev.jsonl"),
            str(REPO_ROOT / "datasets" / "mirai_forecast" / "holdout.jsonl"),
        ],
        help="One or more mirai_forecast JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "datasets" / "mirai_event_inputs_rule"),
        help="Directory for converted JSONL files and manifest.",
    )
    parser.add_argument("--max-docs", type=int, default=0, help="Maximum history documents per sample; 0 keeps all.")
    parser.add_argument("--max-events", type=int, default=16, help="Maximum extracted events per sample.")
    parser.add_argument("--max-events-per-doc", type=int, default=6, help="Maximum extracted events per document.")
    parser.add_argument("--min-events", type=int, default=2, help="Skip samples with fewer extracted events.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows per input file; 0 processes all rows.")
    parser.add_argument("--log-every", type=int, default=250, help="Progress interval per input file.")
    return parser.parse_args()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(payload)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _query_from_row(row: dict[str, Any]) -> QuerySpec:
    raw_query = _as_dict(row.get("query"))
    query_id = _text(raw_query.get("query_id") or row.get("sample_id"))
    text = _text(raw_query.get("text"))
    if not query_id or not text:
        raise ValueError("mirai_forecast row requires query.query_id and query.text")
    focus_entities = [_text(item) for item in _as_list(raw_query.get("focus_entities")) if _text(item)]
    return QuerySpec(
        query_id=query_id,
        text=text,
        cutoff_time=_text(raw_query.get("cutoff_time")) or None,
        focus_entities=focus_entities,
        metadata=_as_dict(raw_query.get("metadata")),
    )


def _documents_from_row(row: dict[str, Any], max_docs: int) -> list[NewsDocument]:
    documents: list[NewsDocument] = []
    seen_ids: set[str] = set()
    raw_documents = _as_list(row.get("documents"))
    for raw_document in raw_documents:
        data = _as_dict(raw_document)
        document_id = _text(data.get("document_id") or data.get("doc_id"))
        if not document_id or document_id in seen_ids:
            continue
        seen_ids.add(document_id)
        documents.append(
            NewsDocument(
                document_id=document_id,
                title=str(data.get("title", "")),
                text=str(data.get("text", "")),
                publish_time=_text(data.get("publish_time")) or None,
                source=str(data.get("source", "MIRAI")),
                metadata=_as_dict(data.get("metadata")),
            )
        )
        if max_docs > 0 and len(documents) >= max_docs:
            break
    return documents


def _target_metadata(row: dict[str, Any], query: QuerySpec) -> tuple[list[str], list[dict[str, Any]]]:
    targets = _as_dict(row.get("targets"))
    target_events: list[dict[str, Any]] = []
    for item in _as_list(targets.get("events")):
        if not isinstance(item, dict):
            continue
        event_code = _text(item.get("event_code"))
        if not event_code:
            continue
        target_events.append(
            {
                "date": _text(item.get("date")),
                "event_code": event_code,
                "relation_name": _text(item.get("relation_name")),
                "docids": [_text(docid) for docid in _as_list(item.get("docids")) if _text(docid)],
            }
        )
    codes = {_text(code) for code in _as_list(targets.get("event_codes")) if _text(code)}
    if not codes:
        codes = {
            _text(code)
            for code in _as_list(query.metadata.get("gold_answer_list"))
            if _text(code)
        }
    codes.update(item["event_code"] for item in target_events)
    return sorted(codes), target_events


def convert_row(
    row: dict[str, Any],
    *,
    source_split: str,
    max_docs: int = 0,
    max_events: int = 16,
    max_events_per_doc: int = 6,
    min_events: int = 2,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    query = _query_from_row(row)
    documents = _documents_from_row(row, max_docs)
    if not documents:
        return None, {"query_id": query.query_id, "event_count": 0, "reason": "no_documents"}
    events, extraction_metadata = extract_rule_events(
        query=query,
        documents=documents,
        max_events=max_events,
        max_events_per_doc=max_events_per_doc,
    )
    if len(events) < min_events:
        return None, {
            "query_id": query.query_id,
            "event_count": len(events),
            "reason": "too_few_events",
        }
    answer_list, target_events = _target_metadata(row, query)
    targets = _as_dict(row.get("targets"))
    source_metadata = _as_dict(row.get("metadata"))
    payload = build_event_input_record(
        query=query,
        documents=documents,
        events=events,
        metadata={
            "dataset": "MIRAI",
            "source_dataset": "mirai_forecast",
            "source_schema_version": str(row.get("schema_version", SOURCE_SCHEMA_VERSION)),
            "source_split": source_split,
            "forecast_sample_id": _text(row.get("sample_id")),
            "event_source": "rule_offline_extractor",
            "extractor_name": "rule",
            "source_docids": [document.document_id for document in documents],
            "answer_list": answer_list,
            "target_events": target_events,
            "target_horizon_days": targets.get("horizon_days"),
            "target_label_start": _text(targets.get("label_start")) or None,
            "target_label_end": _text(targets.get("label_end")) or None,
            "history_metadata": source_metadata,
            "extraction": extraction_metadata,
        },
    )
    payload["schema_version"] = OUTPUT_SCHEMA_VERSION
    return payload, None


def _output_name(input_path: Path) -> str:
    return f"mirai_forecast_event_input_{input_path.stem}.jsonl"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(input_path, limit=args.limit)
    converted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        try:
            payload, skip = convert_row(
                row,
                source_split=input_path.stem,
                max_docs=args.max_docs,
                max_events=args.max_events,
                max_events_per_doc=args.max_events_per_doc,
                min_events=args.min_events,
            )
        except (TypeError, ValueError, KeyError) as exc:
            skip = {
                "query_id": _text(_as_dict(row.get("query")).get("query_id") or row.get("sample_id")),
                "event_count": 0,
                "reason": "invalid_row",
                "error": str(exc),
            }
            payload = None
        if payload is not None:
            # Validate before writing so malformed event references fail close to their source row.
            parse_event_input_record(payload, source=str(input_path))
            converted.append(payload)
        elif skip is not None:
            skipped.append(skip)
        if args.log_every > 0 and (index % args.log_every == 0 or index == len(rows)):
            print(
                f"{input_path.name}: processed={index}/{len(rows)} kept={len(converted)} skipped={len(skipped)}",
                flush=True,
            )
    write_jsonl(output_path, converted)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "requested_rows": len(rows),
        "written_records": len(converted),
        "skipped_records": len(skipped),
        "total_events": sum(len(row.get("events", [])) for row in converted),
        "average_events_per_record": (
            sum(len(row.get("events", [])) for row in converted) / len(converted)
            if converted
            else 0.0
        ),
        "skipped_preview": skipped[:10],
    }


def main() -> None:
    args = parse_args()
    if args.max_docs < 0 or args.max_events < 1 or args.max_events_per_doc < 1 or args.min_events < 0:
        raise ValueError("max-docs must be >= 0; max-events and max-events-per-doc must be positive; min-events must be >= 0")
    input_paths = [resolve_repo_path(path) for path in args.input]
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSONL was not found: {input_path}")
        summaries.append(convert_file(input_path, output_dir / _output_name(input_path), args))
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "extractor": "rule",
        "config": {**vars(args), "input": [str(path) for path in input_paths], "output_dir": str(output_dir)},
        "files": summaries,
        "total_written_records": sum(item["written_records"] for item in summaries),
        "total_skipped_records": sum(item["skipped_records"] for item in summaries),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
