from __future__ import annotations

import argparse
from pathlib import Path

from batch_forecast import run_batch_forecast
from evaluate_forecast import evaluate_prediction_rows, load_jsonl
from mind_adapter import convert_mind_zip_to_jsonl
from mind_query_builder import build_queries
from lite_cerf import LiteCausalEventRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an end-to-end small MIND pipeline for Lite Causal EventRAG.")
    parser.add_argument("--mind-zip", required=True, help="Path to a MIND zip split, e.g. MINDlarge_dev.zip.")
    parser.add_argument("--category", default="news", help="Category filter.")
    parser.add_argument("--news-limit", type=int, default=100, help="Number of news rows to keep.")
    parser.add_argument("--query-limit", type=int, default=30, help="Number of forecast queries to build.")
    parser.add_argument("--prefix", default="mind_dev_small", help="Output prefix under data/.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    news_jsonl = data_dir / f"{args.prefix}_news.jsonl"
    memory_json = data_dir / f"{args.prefix}_memory.json"
    queries_jsonl = data_dir / f"{args.prefix}_queries.jsonl"
    predictions_jsonl = data_dir / f"{args.prefix}_predictions.jsonl"

    count_news = convert_mind_zip_to_jsonl(
        args.mind_zip,
        news_jsonl,
        category=args.category,
        limit=args.news_limit,
    )
    print(f"[1/4] Wrote {count_news} news rows to {news_jsonl}")

    cerf = LiteCausalEventRAG()
    memory = cerf.build_memory_from_jsonl(news_jsonl)
    cerf.save_memory(memory_json)
    print(f"[2/4] Built memory: {len(memory.documents)} documents, {len(memory.events)} events, {len(memory.edges)} edges")

    count_queries = build_queries(
        news_zip=args.mind_zip,
        behaviors_zip=args.mind_zip,
        output_path=queries_jsonl,
        category=args.category,
        limit=args.query_limit,
    )
    print(f"[3/4] Wrote {count_queries} queries to {queries_jsonl}")

    count_predictions = run_batch_forecast(
        memory_path=memory_json,
        query_path=queries_jsonl,
        output_path=predictions_jsonl,
        top_k=3,
    )
    print(f"[4/4] Wrote {count_predictions} predictions to {predictions_jsonl}")

    report = evaluate_prediction_rows(load_jsonl(predictions_jsonl))
    print("Evaluation:")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
