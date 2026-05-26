from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lite_cerf import LiteCausalEventRAG


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_batch_forecast(
    memory_path: str | Path,
    query_path: str | Path,
    output_path: str | Path,
    *,
    top_k: int = 3,
    use_llm: bool = False,
) -> int:
    cerf = LiteCausalEventRAG(use_llm_reranker=use_llm)
    cerf.load_memory(memory_path)

    queries = load_jsonl(query_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as writer:
        for query in queries:
            result = cerf.forecast(query_text=query["query_text"], top_k=top_k)
            row = {
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "target_text": query.get("target_text"),
                "decision": result["decision"],
                "predictions": result["predictions"],
                "abstention_reason": result.get("abstention_reason"),
            }
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch forecasting for Lite Causal EventRAG.")
    parser.add_argument("--memory", required=True, help="Memory JSON path.")
    parser.add_argument("--queries", required=True, help="Query JSONL path.")
    parser.add_argument("--output", required=True, help="Prediction JSONL path.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions per query.")
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI-compatible reranker.")
    args = parser.parse_args()

    count = run_batch_forecast(
        memory_path=args.memory,
        query_path=args.queries,
        output_path=args.output,
        top_k=args.top_k,
        use_llm=args.use_llm,
    )
    print(f"Wrote {count} forecast rows to {args.output}")


if __name__ == "__main__":
    main()
