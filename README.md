# Lite Causal EventRAG

Lite Causal EventRAG is a small, solo-research-friendly prototype for event forecasting.

This project is intentionally narrow:

- small event memory
- query-centric causal retrieval
- lightweight reranking
- forecast or abstain

It does not try to be a full benchmark system.

## What It Does

Given a small news collection, the prototype:

1. extracts lightweight event instances
2. builds a small causal event memory
3. retrieves local support paths for a query
4. predicts likely next events or consequences
5. abstains when support is weak

## Project Layout

```text
lite_causal_eventrag/
  README.md
  requirements.txt
  datasets/
    MINDlarge_*.zip
  data/
    lite_news_demo.jsonl
    mind_dev_news_100.jsonl
  src/
    lite_cerf.py
    example_lite_cerf.py
    mind_adapter.py
    mind_query_builder.py
    batch_forecast.py
    evaluate_forecast.py
    run_mind_pipeline.py
  docs/
    method_overview.md
    dataset_notes.md
    使用说明.md
    流程示例.md
```

## Installation

```bash
pip install -r requirements.txt
```

Optional:

- set `OPENAI_API_KEY`
- set `OPENAI_API_BASE`
- set `LITECERF_MODEL_NAME`

These are only needed if you want to use the optional LLM reranker.

## Quick Start

Run the demo:

```bash
python lite_causal_eventrag/src/example_lite_cerf.py
```

Build a memory file:

```bash
python lite_causal_eventrag/src/lite_cerf.py build-memory --input lite_causal_eventrag/data/lite_news_demo.jsonl --output lite_causal_eventrag/data/lite_memory.json
```

Run forecasting:

```bash
python lite_causal_eventrag/src/lite_cerf.py forecast --memory lite_causal_eventrag/data/lite_memory.json --query "central bank raised interest rates"
```

Optional LLM reranking:

```bash
python lite_causal_eventrag/src/lite_cerf.py forecast --memory lite_causal_eventrag/data/lite_memory.json --query "battery recall announced" --use-llm
```

In the current workspace setup, the most reliable way to run commands is from the repository root using the explicit `lite_causal_eventrag/...` paths above.

## End-to-End MIND Pipeline

You can also run a small end-to-end MIND pipeline:

```bash
python lite_causal_eventrag/src/run_mind_pipeline.py --mind-zip lite_causal_eventrag/datasets/MINDlarge_dev.zip --category news --news-limit 100 --query-limit 30 --prefix mind_dev_small
```

This pipeline will:

1. convert MIND `news.tsv` into JSONL
2. build a small memory
3. build simple forecast queries from `behaviors.tsv`
4. run batch forecasting
5. print a small evaluation summary

## Input Format

The input document file is `jsonl`.

Each line should look like:

```json
{
  "document_id": "econ_001",
  "title": "Central bank raises interest rates",
  "text": "The central bank raised interest rates on Tuesday. Borrowing costs increased shortly afterward.",
  "publish_time": "2025-05-25",
  "source": "demo"
}
```

You may also provide an optional `events` list if you already have extracted events.

## Current Scope

This version is designed for:

- single-domain news
- small datasets
- prototype experiments

It does not include:

- large-scale data engineering
- multi-agent debate
- complex graph databases
- full branch forecasting

## Core Documents

If you are new to this project, read these in order:

1. `docs/使用说明.md`
2. `docs/流程示例.md`
3. `docs/dataset_notes.md`
4. `docs/method_overview.md`
5. `TODO.md`

## Recommended Next Step

Download a small, time-ordered news dataset and convert it into the JSONL format above.
