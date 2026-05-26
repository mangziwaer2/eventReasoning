from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_llm import LocalGenerationUnavailable
from local_llm import LocalQwenGenerator
from local_llm import build_forecast_prompt
from local_llm import parse_forecast_response
from mirai_dataset import export_mirai_query_snapshot
from mirai_dataset import get_mirai_query_by_id
from mirai_dataset import load_mirai_news_for_docids
from query_causal_graph import QueryCausalGraphBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MIRAI query-conditioned local causal graph and optionally run a local model forecast."
    )
    parser.add_argument("--dataset", default="datasets/MIRAI_data.zip", help="Path to MIRAI zip file.")
    parser.add_argument("--query-id", required=True, help="MIRAI QueryId.")
    parser.add_argument("--split", default="test", help="MIRAI split name: test or test_subset.")
    parser.add_argument("--model-path", default="models/Qwen2.5-0.5B", help="Local Hugging Face model directory.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--max-docs", type=int, default=6, help="Maximum retrieved documents kept in the graph.")
    parser.add_argument("--max-events-per-doc", type=int, default=4, help="Maximum events kept per document.")
    parser.add_argument("--event-extractor", default="rule", help="Event extractor backend name.")
    parser.add_argument("--skip-model", action="store_true", help="Only build the graph and prompt without generation.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature for the local model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    example = get_mirai_query_by_id(dataset_path, query_id=args.query_id, split=args.split)
    documents = load_mirai_news_for_docids(dataset_path, example.docids)

    builder = QueryCausalGraphBuilder(
        max_docs=args.max_docs,
        max_events_per_doc=args.max_events_per_doc,
        event_extractor_name=args.event_extractor,
    )
    query = example.build_query_spec()
    graph = builder.build(query, documents)
    prompt = build_forecast_prompt(graph)

    payload = {
        "mirai_query": json.loads(export_mirai_query_snapshot(example)),
        "event_extractor": args.event_extractor,
        "graph": graph.to_dict(),
        "prompt": prompt,
    }

    if not args.skip_model:
        try:
            generator = LocalQwenGenerator(Path(args.model_path))
            raw_response = generator.generate(prompt, temperature=args.temperature)
            forecast_result = parse_forecast_response(
                query_id=example.query_id,
                prompt=prompt,
                raw_response=raw_response,
                gold=example.gold_summary(),
            )
            payload["forecast_result"] = forecast_result.to_dict()
        except LocalGenerationUnavailable as exc:
            payload["forecast_result"] = {
                "error": str(exc),
                "gold": example.gold_summary(),
            }

    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
