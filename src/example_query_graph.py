from __future__ import annotations

import json

from causal_graph import NewsDocument
from query_causal_graph import QueryCausalGraphBuilder
from query_causal_graph import build_query


def main() -> None:
    query = build_query(
        "What may happen after the Central Bank raises rates for manufacturers?",
        "2025-05-25",
    )

    documents = [
        NewsDocument(
            document_id="news_001",
            title="Central Bank raises rates",
            text=(
                "The Central Bank raised interest rates on Tuesday. "
                "Borrowing costs increased for manufacturers after the decision. "
                "Several firms delayed expansion plans because financing became more expensive."
            ),
            publish_time="2025-05-25",
            source="demo",
        ),
        NewsDocument(
            document_id="news_002",
            title="Manufacturers revise investment plans",
            text=(
                "Major Manufacturers reviewed factory expansion plans this week. "
                "Executives warned that higher financing costs could slow new hiring."
            ),
            publish_time="2025-05-24",
            source="demo",
        ),
        NewsDocument(
            document_id="news_003",
            title="Coffee exports improve",
            text=(
                "Coffee exporters reported better harvest conditions. "
                "Analysts said shipment volumes may rise next month."
            ),
            publish_time="2025-05-23",
            source="demo",
        ),
    ]

    builder = QueryCausalGraphBuilder(max_docs=3, max_events_per_doc=3)
    graph = builder.build(query, documents)
    print(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
