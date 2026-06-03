from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_graph import GraphBuildTrace
from event_extractor import build_event_extractor
from mirai_dataset import export_mirai_query_snapshot
from mirai_dataset import get_mirai_query_by_id
from mirai_dataset import load_mirai_news_for_docids
from query_causal_graph import QueryCausalGraphBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a MIRAI query and its linked news documents into retrieved evidence and extracted atomic events."
    )
    parser.add_argument("--dataset", default="datasets/MIRAI_data.zip", help="Path to MIRAI zip file.")
    parser.add_argument("--query-id", required=True, help="MIRAI QueryId.")
    parser.add_argument("--split", default="test", help="MIRAI split name: test or test_subset.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument("--max-docs", type=int, default=6, help="Maximum retrieved evidence documents.")
    parser.add_argument("--max-events-per-doc", type=int, default=6, help="Maximum structured events per document.")
    parser.add_argument("--event-extractor", default="rule", help="Event extractor backend name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    example = get_mirai_query_by_id(dataset_path, query_id=args.query_id, split=args.split)
    query = example.build_query_spec()
    candidate_documents = load_mirai_news_for_docids(dataset_path, example.docids)
    event_extractor = build_event_extractor(args.event_extractor)

    builder = QueryCausalGraphBuilder(
        max_docs=args.max_docs,
        max_events_per_doc=args.max_events_per_doc,
        event_extractor=event_extractor,
    )
    trace = GraphBuildTrace()
    retrieved_documents = builder.retrieve_documents(query, candidate_documents, trace)

    document_outputs: list[dict[str, object]] = []
    total_events = 0

    for document in retrieved_documents:
        sentence_events: list[dict[str, object]] = []
        document_extraction = event_extractor.extract_document(query, document)
        for sentence_extraction in document_extraction.sentence_extractions:
            events = sentence_extraction.events[: args.max_events_per_doc]
            sentence_events.append(
                {
                    "sentence_index": sentence_extraction.sentence_index,
                    "sentence_text": sentence_extraction.sentence_text,
                    "events": [
                        {
                            "text": event.text,
                            "normalized_text": event.normalized_text,
                            "trigger": event.trigger,
                            "participants": event.participants,
                            "score": event.score,
                        }
                        for event in events
                    ],
                }
            )
            total_events += len(events)

        document_outputs.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "publish_time": document.publish_time,
                "source": document.source,
                "extractor_name": document_extraction.extractor_name,
                "event_sentences": sentence_events,
            }
        )

    payload = {
        "mirai_query": json.loads(export_mirai_query_snapshot(example)),
        "query_spec": query.to_dict(),
        "event_extractor": event_extractor.name,
        "retrieval_hits": [item.to_dict() for item in trace.retrieval_hits],
        "retrieved_document_count": len(retrieved_documents),
        "extracted_event_count": total_events,
        "documents": document_outputs,
    }

    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
