from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import io
import json
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from event_extraction import clean_document_text
from path_utils import REPO_ROOT, resolve_repo_path


SCHEMA_VERSION = "mirai-forecast-v1"
DEFAULT_START = "2023-02-01"
DEFAULT_TRAIN_END = "2023-08-31"
DEFAULT_DEV_END = "2023-09-30"
DEFAULT_HOLDOUT_END = "2023-10-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build chronological MIRAI forecasting data from data_kg.csv and data_news.csv. "
            "History is on or before cutoff; labels are events in the future horizon."
        )
    )
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MIRAI_data.zip"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "datasets" / "mirai_forecast"))
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--dev-end", default=DEFAULT_DEV_END)
    parser.add_argument("--holdout-end", default=DEFAULT_HOLDOUT_END)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--max-history-events", type=int, default=24)
    parser.add_argument("--max-docs", type=int, default=8)
    parser.add_argument("--min-history-events", type=int, default=2)
    parser.add_argument("--min-history-docs", type=int, default=1)
    parser.add_argument("--max-samples-train", type=int, default=5000)
    parser.add_argument("--max-samples-dev", type=int, default=1000)
    parser.add_argument("--max-samples-holdout", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-log-every", type=int, default=250)
    parser.add_argument("--sample-log-count", type=int, default=1)
    return parser.parse_args()


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date: {value!r}") from exc


def parse_docids(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value or "[]")
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        text = str(item).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def compact(text: str, limit: int = 600) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: max(0, limit - 3)].rstrip() + "..."


def read_kg(path: Path) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[tuple[str, str, str], list[dict[str, Any]]]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_pair_date: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with zipfile.ZipFile(path) as archive:
        with archive.open("MIRAI/data_kg.csv") as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), delimiter="\t")
            for raw in reader:
                date = str(raw.get("DateStr", "")).strip()
                a1 = str(raw.get("Actor1CountryCode", "")).strip()
                a2 = str(raw.get("Actor2CountryCode", "")).strip()
                code = str(raw.get("EventBaseCode", "")).strip()
                if not date or not a1 or not a2 or not code:
                    continue
                row = {
                    "date": date,
                    "actor1_code": a1,
                    "actor2_code": a2,
                    "actor1_name": str(raw.get("Actor1CountryName", a1)).strip(),
                    "actor2_name": str(raw.get("Actor2CountryName", a2)).strip(),
                    "event_code": code,
                    "relation_name": str(raw.get("RelName", "")).strip(),
                    "quad_event_code": str(raw.get("QuadEventCode", "")).strip(),
                    "docid": str(raw.get("Docid", "")).strip(),
                    "docids": parse_docids(raw.get("Docids", "")),
                }
                pair = (a1, a2)
                by_pair[pair].append(row)
                by_pair_date[(date, a1, a2)].append(row)
    for rows in by_pair.values():
        rows.sort(key=lambda item: (item["date"], item["docid"], item["event_code"]))
    return dict(by_pair), dict(by_pair_date)


def read_news(path: Path, needed_docids: set[str]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    if not needed_docids:
        return selected
    with zipfile.ZipFile(path) as archive:
        with archive.open("MIRAI/data_news.csv") as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), delimiter="\t")
            for raw in reader:
                docid = str(raw.get("Docid", "")).strip()
                if not docid or docid not in needed_docids:
                    continue
                title = str(raw.get("Title", "")).strip()
                text = str(raw.get("Text", "")).strip()
                abstract = str(raw.get("Abstract", "")).strip()
                body = "\n".join(item for item in (abstract, text) if item)
                selected[docid] = {
                    "document_id": docid,
                    "title": title,
                    "text": clean_document_text(title=title, text=body),
                    "publish_time": str(raw.get("Date", "")).strip() or None,
                    "source": "MIRAI",
                    "metadata": {"url": str(raw.get("URL", "")).strip(), "md5": str(raw.get("MD5", "")).strip()},
                }
    return selected


def event_payload(row: dict[str, Any], index: int, news: dict[str, Any]) -> dict[str, Any]:
    docid = row["docid"]
    title = str(news.get("title", "")).strip()
    relation = row["relation_name"] or f"CAMEO event {row['event_code']}"
    mention = f"{row['actor1_name']} {relation} {row['actor2_name']}"
    evidence_text = title or mention
    return {
        "event_id": f"e{index}",
        "trigger": row["relation_name"] or row["event_code"],
        "mention": mention,
        "normalized_text": mention.lower(),
        "document_id": docid,
        "sentence_index": 0,
        "participants": [row["actor1_name"], row["actor2_name"]],
        "confidence": 1.0,
        "evidence": [{"document_id": docid, "sentence_index": 0, "text": evidence_text}],
        "metadata": {
            "source": "MIRAI/data_kg.csv",
            "date": row["date"],
            "event_base_code": row["event_code"],
            "relation_name": row["relation_name"],
            "quad_event_code": row["quad_event_code"],
            "structured_event": True,
            "evidence_type": "source_document_title",
            "evidence_note": "CAMEO event label from MIRAI/data_kg.csv; headline is provenance and is not a manually aligned span.",
        },
    }


def query_id(cutoff: dt.date, pair: tuple[str, str]) -> str:
    return f"{cutoff.isoformat()}_{pair[0]}_{pair[1]}"


def build_sample(
    split: str,
    cutoff: dt.date,
    pair: tuple[str, str],
    history: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    news: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], set[str]] | None:
    unique_history: list[dict[str, Any]] = []
    seen_events: set[tuple[str, str, str]] = set()
    history_start = cutoff - dt.timedelta(days=args.history_days) if args.history_days > 0 else None
    for row in reversed(history):
        row_date = parse_date(row["date"])
        if row_date > cutoff or (history_start is not None and row_date < history_start):
            continue
        key = (row["date"], row["event_code"], row["quad_event_code"] or row["event_code"])
        if key in seen_events:
            continue
        seen_events.add(key)
        unique_history.append(row)
        if len(unique_history) >= args.max_history_events:
            break
    unique_history.reverse()
    history_docids = list(dict.fromkeys(row["docid"] for row in unique_history if row["docid"]))
    history_docids = history_docids[-args.max_docs :] if args.max_docs > 0 else history_docids
    if len(unique_history) < args.min_history_events or len(history_docids) < args.min_history_docs:
        return None
    active_docs = [news[docid] for docid in history_docids if docid in news]
    if len(active_docs) < args.min_history_docs:
        return None
    active_docids = {doc["document_id"] for doc in active_docs}
    unique_history = [row for row in unique_history if row["docid"] in active_docids]
    if len(unique_history) < args.min_history_events:
        return None
    codes = sorted({row["event_code"] for row in targets})
    if not codes:
        return None
    first_target = targets[0]
    query = {
        "query_id": query_id(cutoff, pair),
        "text": f"As of {cutoff.isoformat()}, what important event may happen next between {first_target['actor1_name']} and {first_target['actor2_name']}?",
        "cutoff_time": cutoff.isoformat(),
        "focus_entities": [first_target["actor1_name"], first_target["actor2_name"]],
        "metadata": {"dataset": "MIRAI", "actor1_country_code": pair[0], "actor2_country_code": pair[1], "gold_answer_list": codes},
    }
    event_rows = [event_payload(row, index, news[row["docid"]]) for index, row in enumerate(unique_history, start=1)]
    future_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in targets:
        key = (row["date"], row["event_code"], row["quad_event_code"] or row["event_code"])
        item = future_by_key.setdefault(
            key,
            {"date": row["date"], "event_code": row["event_code"], "relation_name": row["relation_name"], "docids": []},
        )
        for docid in row["docids"] or ([row["docid"]] if row["docid"] else []):
            if docid and docid not in item["docids"]:
                item["docids"].append(docid)
    future_events = sorted(future_by_key.values(), key=lambda item: (item["date"], item["event_code"]))
    row = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": query["query_id"],
        "query": query,
        "documents": active_docs,
        "events": event_rows,
        "targets": {
            "horizon_days": args.horizon_days,
            "label_start": (cutoff + dt.timedelta(days=1)).isoformat(),
            "label_end": (cutoff + dt.timedelta(days=args.horizon_days)).isoformat(),
            "event_codes": codes,
            "events": future_events,
        },
        "metadata": {
            "dataset": "MIRAI",
            "source_files": ["MIRAI/data_kg.csv", "MIRAI/data_news.csv"],
            "split": split,
            "history_inclusive_cutoff": True,
            "history_start": history_start.isoformat() if history_start else None,
            "history_end_exclusive": (cutoff + dt.timedelta(days=1)).isoformat(),
            "history_docids": history_docids,
            "target_docids": sorted({docid for item in future_events for docid in item["docids"]}),
            "event_source": "gdelt_cameo_structured",
        },
    }
    target_docids = {docid for item in future_events for docid in item["docids"]}
    return row, target_docids


def render_sample(row: dict[str, Any], number: int) -> str:
    query = row["query"]
    lines = [
        f"=== MIRAI forecast sample {number} | split={row['metadata']['split']} ===",
        f"QueryId: {query['query_id']}",
        f"Query: {query['text']}",
        f"Cutoff: {query['cutoff_time']} | Horizon: {row['targets']['horizon_days']} days",
        "Historical documents (on or before cutoff):",
    ]
    for doc in row["documents"][:8]:
        lines.append(f"- {doc['document_id']} | {doc.get('publish_time') or '-'} | {compact(doc.get('title', ''), 180)}")
    lines.append("Historical events:")
    for event in row["events"][:12]:
        lines.append(f"- {event['event_id']} | {event['metadata'].get('date')} | {event['metadata'].get('event_base_code')} | {event['mention']} | doc={event['document_id']}")
    lines.append("Future supervision (not shown to the model):")
    for event in row["targets"]["events"][:12]:
        lines.append(f"- {event['date']} | {event['event_code']} | {event['relation_name']} | docs={','.join(event['docids'][:4])}")
    return "\n".join(lines) + "\n\n"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.horizon_days < 1 or args.history_days < 0:
        raise ValueError("--horizon-days must be positive and --history-days must be non-negative")
    if args.max_docs < 1 or args.max_history_events < 1:
        raise ValueError("--max-docs and --max-history-events must be positive")
    start = parse_date(args.start_date)
    train_end = parse_date(args.train_end)
    dev_end = parse_date(args.dev_end)
    holdout_end = parse_date(args.holdout_end)
    if not start <= train_end < dev_end < holdout_end:
        raise ValueError("Expected start-date <= train-end < dev-end < holdout-end")
    dataset = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"reading MIRAI KG | dataset={dataset}", flush=True)
    by_pair, by_pair_date = read_kg(dataset)
    print(f"loaded structured events | pairs={len(by_pair)} | rows={sum(len(v) for v in by_pair.values())}", flush=True)

    # Split on future-label dates, not only on cutoff. This prevents a training
    # sample near a boundary from carrying labels that belong to dev or holdout.
    split_specs = {
        "train": (start, train_end, args.max_samples_train),
        "dev": (train_end + dt.timedelta(days=1), dev_end, args.max_samples_dev),
        "holdout": (dev_end + dt.timedelta(days=1), holdout_end, args.max_samples_holdout),
    }
    candidates: dict[str, list[tuple[dt.date, tuple[str, str], list[dict[str, Any]], list[dict[str, Any]]]]] = defaultdict(list)
    for pair, rows in by_pair.items():
        dates = sorted({parse_date(row["date"]) for row in rows})
        for target_date in dates:
            cutoff = target_date - dt.timedelta(days=1)
            target_rows: list[dict[str, Any]] = []
            for offset in range(args.horizon_days):
                target_rows.extend(by_pair_date.get(((target_date + dt.timedelta(days=offset)).isoformat(), pair[0], pair[1]), []))
            if not target_rows:
                continue
            history_rows = [row for row in rows if parse_date(row["date"]) <= cutoff]
            for split, (label_start, label_end, limit) in split_specs.items():
                if label_start <= target_date and target_date + dt.timedelta(days=args.horizon_days - 1) <= label_end:
                    candidates[split].append((cutoff, pair, history_rows, target_rows))
                    break

    rng = random.Random(args.seed)
    all_needed_docs: set[str] = set()
    selected_candidates: dict[str, list[tuple[dt.date, tuple[str, str], list[dict[str, Any]], list[dict[str, Any]]]]] = {}
    for split, values in candidates.items():
        rng.shuffle(values)
        limit = split_specs[split][2]
        selected = values[:limit] if limit > 0 else values
        selected_candidates[split] = selected
        for cutoff, _, history_rows, _ in selected:
            history_start = cutoff - dt.timedelta(days=args.history_days) if args.history_days > 0 else None
            for row in reversed(history_rows):
                if row["date"] > cutoff.isoformat() or (history_start and row["date"] < history_start.isoformat()):
                    continue
                all_needed_docs.add(row["docid"])
                if len(all_needed_docs) >= 1_000_000:
                    break
    print(f"loading source documents | needed_docids={len(all_needed_docs)}", flush=True)
    news = read_news(dataset, all_needed_docs)
    print(f"loaded source documents | found={len(news)} | missing={len(all_needed_docs - set(news))}", flush=True)

    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "config": {**vars(args), "dataset": str(dataset), "output_dir": str(output_dir)}, "splits": {}}
    for split in ("train", "dev", "holdout"):
        rows: list[dict[str, Any]] = []
        skipped: dict[str, int] = defaultdict(int)
        sample_path = output_dir / f"{split}_samples.txt"
        with sample_path.open("w", encoding="utf-8") as sample_handle:
            for index, (cutoff, pair, history_rows, target_rows) in enumerate(selected_candidates.get(split, []), start=1):
                built = build_sample(split, cutoff, pair, history_rows, target_rows, news, args)
                if built is None:
                    skipped["insufficient_history_or_documents"] += 1
                    continue
                row, _ = built
                rows.append(row)
                if args.sample_log_every > 0 and args.sample_log_count > 0 and (len(rows) % args.sample_log_every == 0 or len(rows) <= args.sample_log_count):
                    sample_handle.write(render_sample(row, len(rows)))
                if index % 500 == 0 or index == len(selected_candidates.get(split, [])):
                    print(f"{split}: processed={index}/{len(selected_candidates.get(split, []))} kept={len(rows)} skipped={sum(skipped.values())}", flush=True)
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, rows)
        summary["splits"][split] = {
            "candidate_samples": len(selected_candidates.get(split, [])),
            "written_samples": len(rows),
            "skipped": dict(skipped),
            "output": str(output_path),
            "human_readable_samples": str(sample_path),
            "label_events": sum(len(row["targets"]["events"]) for row in rows),
            "average_history_events": sum(len(row["events"]) for row in rows) / len(rows) if rows else 0.0,
        }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
