from __future__ import annotations

from pathlib import Path

from lite_cerf import LiteCausalEventRAG


def main() -> None:
    data_path = Path(__file__).resolve().parent.parent / "data" / "lite_news_demo.jsonl"
    memory_path = Path(__file__).resolve().parent.parent / "data" / "lite_memory_demo.json"

    cerf = LiteCausalEventRAG()
    memory = cerf.build_memory_from_jsonl(data_path)
    cerf.save_memory(memory_path)

    print(
        f"Memory built: {len(memory.documents)} documents, {len(memory.events)} events, {len(memory.edges)} edges.\n"
    )

    for query in [
        "central bank raised interest rates",
        "battery recall announced",
        "coffee harvest improved",
    ]:
        result = cerf.forecast(query, top_k=3)
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Decision: {result['decision']}")
        if result["abstention_reason"]:
            print(f"Reason: {result['abstention_reason']}")
        for prediction in result["predictions"]:
            path = " -> ".join(prediction["support_path"])
            print(f"- {prediction['text']} | score={prediction['heuristic_score']:.3f} | support={path}")
        print()


if __name__ == "__main__":
    main()
