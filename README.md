# Query-Conditioned Local Causal Graph

This repository is now a research scaffold for one direction:

**build a local causal graph from a query and time-bounded news evidence, then use that graph to support future event prediction.**

The previous `MIND -> memory -> forecast` prototype has been removed on purpose. It no longer matched the current research target and was blocking clean iteration.

## Current Focus

The repository now centers on four questions:

1. What should the forecasting input look like?
2. How should a query-conditioned local causal graph be constructed?
3. How should bridge events and conflicting explanations be represented?
4. How should graph quality and downstream forecasting quality be evaluated together?

## Repository Layout

```text
lite_causal_eventrag/
  README.md
  requirements.txt
  docs/
    query_graph_method.md
    当前创新点.md
    当前项目功能书.md
    创新点变动书.md
  src/
    causal_graph.py
    query_causal_graph.py
    example_query_graph.py
    mirai_dataset.py
    local_llm.py
    query_forecast.py
  TODO.md
```

## Current Code Scope

The current code is intentionally small.

- `src/causal_graph.py`
  Defines the research data structures for query, evidence documents, event nodes, causal edges, and local graphs.
- `src/query_causal_graph.py`
  Provides a lightweight baseline builder for `query -> retrieved evidence -> local causal graph`.
- `src/example_query_graph.py`
  Runs an in-memory demo and prints the resulting graph.

This is a starting point, not the final method.

## Quick Start

Run the demo:

```bash
python src/example_query_graph.py
```

Run the CLI on a JSONL file:

```bash
python src/query_causal_graph.py --input news.jsonl --query "What may happen after the central bank raises rates?" --cutoff 2025-05-25
```

Run a MIRAI-based local validation:

```bash
python src/query_forecast.py --query-id 1 --skip-model
```

Run the same path with the local Qwen model:

```bash
python src/query_forecast.py --query-id 1 --model-path models/Qwen2.5-0.5B
```

Each JSONL row should contain:

```json
{
  "document_id": "news_001",
  "title": "Central bank raises rates",
  "text": "The central bank raised rates. Borrowing costs climbed for manufacturers.",
  "publish_time": "2025-05-25",
  "source": "demo"
}
```

## Planned Benchmark Setup

- Main downstream benchmark: `MIRAI`
- Auxiliary graph-quality evaluation: `MAVEN-ERE`, `Event StoryLine Corpus`, `Causal News Corpus`

Those choices and later changes are tracked in [docs/创新点变动书.md](/E:/project/EventDetection/lite_causal_eventrag/docs/创新点变动书.md).

## What Was Intentionally Removed

The following old direction has been removed:

- `MIND` conversion scripts
- offline memory construction
- heuristic batch forecasting
- legacy evaluation scripts tied to the old prototype

If an old file is gone, it was removed because it no longer served the query-conditioned causal-graph direction.
