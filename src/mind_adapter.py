from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


def parse_news_line(line: str) -> dict[str, Any] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 7:
        return None

    news_id = parts[0].strip()
    category = parts[1].strip()
    subcategory = parts[2].strip()
    title = parts[3].strip()
    abstract = parts[4].strip()
    url = parts[5].strip()
    title_entities_raw = parts[6].strip() if len(parts) > 6 else "[]"
    abstract_entities_raw = parts[7].strip() if len(parts) > 7 else "[]"

    try:
        title_entities = json.loads(title_entities_raw) if title_entities_raw else []
    except json.JSONDecodeError:
        title_entities = []
    try:
        abstract_entities = json.loads(abstract_entities_raw) if abstract_entities_raw else []
    except json.JSONDecodeError:
        abstract_entities = []

    return {
        "document_id": news_id,
        "category": category,
        "subcategory": subcategory,
        "title": title,
        "text": abstract,
        "url": url,
        "title_entities": title_entities,
        "abstract_entities": abstract_entities,
    }


def iter_zip_news(zip_path: str | Path):
    with zipfile.ZipFile(zip_path) as archive:
        news_member = None
        for member in archive.namelist():
            if member.endswith("news.tsv"):
                news_member = member
                break
        if news_member is None:
            raise FileNotFoundError(f"No news.tsv found inside {zip_path}")

        with archive.open(news_member) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8", errors="replace")
                parsed = parse_news_line(line)
                if parsed is not None:
                    yield parsed


def convert_mind_zip_to_jsonl(
    zip_path: str | Path,
    output_path: str | Path,
    *,
    category: str | None = None,
    subcategory: str | None = None,
    limit: int | None = None,
    require_abstract: bool = True,
) -> int:
    count = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as writer:
        for item in iter_zip_news(zip_path):
            if category and item["category"].lower() != category.lower():
                continue
            if subcategory and item["subcategory"].lower() != subcategory.lower():
                continue
            if require_abstract and not item["text"]:
                continue

            row = {
                "document_id": item["document_id"],
                "title": item["title"],
                "text": item["text"],
                "publish_time": None,
                "source": "MIND",
                "category": item["category"],
                "subcategory": item["subcategory"],
                "url": item["url"],
                "entities": sorted(
                    {
                        surface
                        for entity in item["title_entities"] + item["abstract_entities"]
                        for surface in entity.get("SurfaceForms", [])
                        if surface
                    }
                ),
            }
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a MIND zip split into Lite Causal EventRAG JSONL.")
    parser.add_argument("--input-zip", required=True, help="Path to MINDlarge_train.zip or MINDlarge_dev.zip.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--category", help="Optional category filter, e.g. news or finance-like proxy category.")
    parser.add_argument("--subcategory", help="Optional subcategory filter.")
    parser.add_argument("--limit", type=int, help="Optional row limit.")
    parser.add_argument(
        "--allow-empty-abstract",
        action="store_true",
        help="Include rows whose abstract text is empty.",
    )
    args = parser.parse_args()

    count = convert_mind_zip_to_jsonl(
        args.input_zip,
        args.output,
        category=args.category,
        subcategory=args.subcategory,
        limit=args.limit,
        require_abstract=not args.allow_empty_abstract,
    )
    print(f"Wrote {count} MIND news rows to {args.output}")


if __name__ == "__main__":
    main()
