from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                key = str(row.get("query_id", row.get("sample_id", ""))).strip()
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge resumable GRPO context JSONL shards with query-id deduplication.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = load_rows([Path(item) for item in args.input])
    if args.limit > 0:
        rows = rows[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows), "sources": len(args.input)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
