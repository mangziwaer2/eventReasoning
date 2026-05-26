from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


def load_news_map(zip_path: str | Path) -> dict[str, dict[str, Any]]:
    from mind_adapter import parse_news_line

    news_map: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("news.tsv")), None)
        if member is None:
            raise FileNotFoundError(f"No news.tsv found inside {zip_path}")

        with archive.open(member) as handle:
            for raw_line in handle:
                parsed = parse_news_line(raw_line.decode("utf-8", errors="replace"))
                if parsed is not None:
                    news_map[parsed["document_id"]] = parsed
    return news_map


def iter_behaviors(zip_path: str | Path):
    with zipfile.ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("behaviors.tsv")), None)
        if member is None:
            raise FileNotFoundError(f"No behaviors.tsv found inside {zip_path}")

        with archive.open(member) as handle:
            for raw_line in handle:
                parts = raw_line.decode("utf-8", errors="replace").rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                yield {
                    "impression_id": parts[0],
                    "user_id": parts[1],
                    "timestamp": parts[2],
                    "history": parts[3].split() if parts[3].strip() else [],
                    "impressions": parts[4].split() if parts[4].strip() else [],
                }


def build_queries(
    *,
    news_zip: str | Path,
    behaviors_zip: str | Path,
    output_path: str | Path,
    category: str | None = None,
    limit: int = 100,
    min_history: int = 1,
) -> int:
    news_map = load_news_map(news_zip)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as writer:
        for behavior in iter_behaviors(behaviors_zip):
            if len(behavior["history"]) < min_history:
                continue

            positive_targets = [item.split("-")[0] for item in behavior["impressions"] if item.endswith("-1")]
            if not positive_targets:
                continue

            history_ids = [news_id for news_id in behavior["history"] if news_id in news_map]
            target_id = next((news_id for news_id in positive_targets if news_id in news_map), None)
            if not history_ids or target_id is None:
                continue

            target_news = news_map[target_id]
            if category and target_news["category"].lower() != category.lower():
                continue

            last_history_id = history_ids[-1]
            last_history_news = news_map[last_history_id]
            query_text = last_history_news["title"] or last_history_news["text"]
            if not query_text:
                continue

            row = {
                "query_id": f"mind_query_{behavior['impression_id']}",
                "query_type": "forecast_next_event",
                "query_text": query_text,
                "focus_event_text": query_text,
                "history_news_ids": history_ids,
                "target_news_id": target_id,
                "target_text": target_news["title"] or target_news["text"],
                "target_category": target_news["category"],
                "target_subcategory": target_news["subcategory"],
            }
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simple forecast queries from MIND behaviors.")
    parser.add_argument("--news-zip", required=True, help="Path to MIND zip containing news.tsv.")
    parser.add_argument("--behaviors-zip", required=True, help="Path to MIND zip containing behaviors.tsv.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--category", help="Optional category filter on target news.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of queries to write.")
    parser.add_argument("--min-history", type=int, default=1, help="Minimum behavior history length.")
    args = parser.parse_args()

    count = build_queries(
        news_zip=args.news_zip,
        behaviors_zip=args.behaviors_zip,
        output_path=args.output,
        category=args.category,
        limit=args.limit,
        min_history=args.min_history,
    )
    print(f"Wrote {count} queries to {args.output}")


if __name__ == "__main__":
    main()
