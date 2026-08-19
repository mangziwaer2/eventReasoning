from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Iterable

from causal_graph import NewsDocument
from event_extraction import clean_document_text
from extract_mirai_events import build_event_input_record
from extract_mirai_events import extract_rule_events
from mirai_dataset import MiraiQueryExample
from mirai_dataset import load_mirai_queries
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed pseudo train/dev/test MIRAI event-input-v1 JSONL files "
            "from the public MIRAI/test split using the no-model rule extractor."
        )
    )
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MIRAI_data.zip"), help="MIRAI_data.zip path.")
    parser.add_argument("--source-split", default="test", help="Source MIRAI split. Public package usually only has test/test_subset.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "datasets" / "mirai_event_inputs_rule"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for pseudo split.")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Pseudo train ratio.")
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Pseudo dev ratio.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap before splitting. Use 0 for all.")
    parser.add_argument("--max-docs", type=int, default=4, help="Maximum documents per query.")
    parser.add_argument("--max-events", type=int, default=16, help="Maximum events per query.")
    parser.add_argument("--max-events-per-doc", type=int, default=6, help="Maximum events retained per document.")
    parser.add_argument("--min-events", type=int, default=2, help="Skip records with fewer extracted events.")
    parser.add_argument("--log-every", type=int, default=50, help="Progress interval.")
    return parser.parse_args()


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def load_news_index(zip_path: Path, needed_docids: set[str]) -> dict[str, NewsDocument]:
    selected: dict[str, NewsDocument] = {}
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("MIRAI/data_news.csv") as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), delimiter="\t")
            for row in reader:
                docid = str(row.get("Docid", "")).strip()
                if docid not in needed_docids or docid in selected:
                    continue
                title = row.get("Title", "").strip()
                text = row.get("Text", "").strip()
                abstract = row.get("Abstract", "").strip()
                parts: list[str] = []
                if abstract:
                    parts.append(abstract)
                if text:
                    parts.append(text)
                selected[docid] = NewsDocument(
                    document_id=docid,
                    title=title,
                    text=clean_document_text(title=title, text="\n".join(parts)),
                    publish_time=row.get("Date"),
                    source="MIRAI",
                    metadata={"url": row.get("URL", ""), "md5": row.get("MD5", "")},
                )
                if len(selected) >= len(needed_docids):
                    break
    return selected


def split_examples(
    examples: list[MiraiQueryExample],
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, list[MiraiQueryExample]]:
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("--train-ratio must be > 0 and train+dev must be < 1.")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    train_size = int(len(shuffled) * train_ratio)
    dev_size = int(len(shuffled) * dev_ratio)
    return {
        "train": shuffled[:train_size],
        "dev": shuffled[train_size : train_size + dev_size],
        "test": shuffled[train_size + dev_size :],
    }


def build_records(
    split_name: str,
    examples: list[MiraiQueryExample],
    news_index: dict[str, NewsDocument],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for index, example in enumerate(examples, start=1):
        query = example.build_query_spec()
        docids = unique_preserve_order(example.docids)[: args.max_docs]
        documents = [news_index[docid] for docid in docids if docid in news_index]
        events, extraction_metadata = extract_rule_events(
            query=query,
            documents=documents,
            max_events=args.max_events,
            max_events_per_doc=args.max_events_per_doc,
        )
        if len(events) < args.min_events:
            skipped.append({"query_id": example.query_id, "event_count": len(events), "reason": "too_few_events"})
            continue
        rows.append(
            build_event_input_record(
                query=query,
                documents=documents,
                events=events,
                metadata={
                    "dataset": "MIRAI",
                    "source_split": args.source_split,
                    "pseudo_split": split_name,
                    "event_source": "rule_offline_extractor",
                    "extractor_name": "rule",
                    "source_docids": docids,
                    "answer_list": example.answer_list,
                    "answer_dict": example.answer_dict,
                    "relation_name": example.relation_name,
                    "event_base_code": example.event_base_code,
                    "extraction": extraction_metadata,
                },
            )
        )
        if args.log_every > 0 and (index % args.log_every == 0 or index == len(examples)):
            print(
                f"{split_name}: processed {index}/{len(examples)} | kept={len(rows)} | skipped={len(skipped)}",
                flush=True,
            )
    return rows, skipped


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir)
    examples = load_mirai_queries(dataset_path, split=args.source_split, limit=args.limit if args.limit > 0 else 0)
    if not examples:
        raise RuntimeError(f"No MIRAI examples loaded from split={args.source_split!r}.")

    needed_docids = {docid for example in examples for docid in unique_preserve_order(example.docids)[: args.max_docs]}
    print(f"loading news index | examples={len(examples)} | needed_docids={len(needed_docids)}", flush=True)
    news_index = load_news_index(dataset_path, needed_docids)
    missing_docids = sorted(needed_docids - set(news_index))
    if missing_docids:
        print(f"warning: missing {len(missing_docids)} documents; preview={missing_docids[:10]}", flush=True)

    splits = split_examples(
        examples=examples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )
    manifest = {
        "dataset": str(dataset_path),
        "source_split": args.source_split,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "dev_ratio": args.dev_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.dev_ratio, 6),
        "max_docs": args.max_docs,
        "max_events": args.max_events,
        "max_events_per_doc": args.max_events_per_doc,
        "min_events": args.min_events,
        "splits": {name: [example.query_id for example in split_examples] for name, split_examples in splits.items()},
    }
    write_json(output_dir / "mirai_pseudo_splits_seed42.json", manifest)

    all_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"config": {**vars(args), "dataset": str(dataset_path)}, "splits": {}}
    for split_name, split_examples_list in splits.items():
        rows, skipped = build_records(split_name, split_examples_list, news_index, args)
        output_path = output_dir / f"mirai_event_input_{split_name}.jsonl"
        write_jsonl(output_path, rows)
        all_rows.extend(rows)
        summary["splits"][split_name] = {
            "requested_queries": len(split_examples_list),
            "written_records": len(rows),
            "skipped_records": len(skipped),
            "total_events": sum(len(row.get("events", [])) for row in rows),
            "average_events_per_record": (
                sum(len(row.get("events", [])) for row in rows) / len(rows) if rows else 0.0
            ),
            "output": str(output_path),
            "skipped_preview": skipped[:10],
        }

    combined_path = output_dir / "mirai_event_input_all.jsonl"
    write_jsonl(combined_path, all_rows)
    summary["combined_output"] = str(combined_path)
    summary["manifest"] = str(output_dir / "mirai_pseudo_splits_seed42.json")
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
