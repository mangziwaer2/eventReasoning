from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split event-input JSONL into resumable shards.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=25)
    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), args.shard_size):
        shard = rows[start : start + args.shard_size]
        path = output_dir / f"events_{start // args.shard_size:04d}.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in shard) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "shards": (len(rows) + args.shard_size - 1) // args.shard_size, "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
