from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


def evaluate_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    abstain_count = 0
    hit_at_1 = 0
    hit_at_3 = 0

    for row in rows:
        if row.get("decision") == "abstain":
            abstain_count += 1

        target = normalize(row.get("target_text"))
        predictions = row.get("predictions", [])
        texts = [normalize(item.get("text")) for item in predictions]

        if target and texts:
            if texts[0] and (texts[0] in target or target in texts[0]):
                hit_at_1 += 1
            top_three = texts[:3]
            if any(candidate and (candidate in target or target in candidate) for candidate in top_three):
                hit_at_3 += 1

    denominator = max(1, total)
    return {
        "total_queries": total,
        "abstain_count": abstain_count,
        "abstain_rate": round(abstain_count / denominator, 4),
        "hit_at_1": hit_at_1,
        "hit_at_1_rate": round(hit_at_1 / denominator, 4),
        "hit_at_3": hit_at_3,
        "hit_at_3_rate": round(hit_at_3 / denominator, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Lite Causal EventRAG forecast outputs.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    args = parser.parse_args()

    rows = load_jsonl(args.predictions)
    report = evaluate_prediction_rows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
