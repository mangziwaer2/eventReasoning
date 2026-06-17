# Query-Conditioned Causal Graph for Event Forecasting

This repository studies one main problem:

**use query-conditioned local causal graphs to improve future event forecasting from news.**

The project is organized around the following pipeline:

`query + cutoff time + candidate news -> retrieved evidence -> extracted events -> coarse causal graph -> refined causal graph -> future event hypotheses`

At the current stage, the implemented code covers the graph construction and refinement path:

`query + cutoff time + candidate news -> retrieved evidence -> extracted events -> coarse causal graph -> refined causal graph`

## Current Scope

The repository does not treat event extraction as the main research contribution.
Event extraction is implemented as an upstream, replaceable module.

The current research focus is:

1. How to build a useful coarse causal graph from extracted events
2. How to refine that coarse graph with graph learning
3. How to use the refined graph for event forecasting

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
    项目方法书.md
    refinement设计书.md
  src/
    causal_graph.py
    event_extraction.py
    event_extractor.py
    mirai_dataset.py
    query_to_events.py
    query_causal_graph.py
    coarse_graph_builder.py
    coarse_graph_dataset.py
    coarse_graph_model.py
    train_coarse_graph.py
    run_coarse_graph_model.py
    local_qwen_lora.py
    train_coarse_graph_qwen.py
    run_coarse_graph_qwen.py
    refinement_dataset.py
    refinement_model.py
    train_refinement.py
    run_refinement.py
    query_forecast.py
    local_llm.py
  temp/
    export_dataset_readable_samples.py
    log_event_extraction_samples.py
  TODO.md
```

## Current Code

- `src/event_extractor.py`
  Pluggable event extraction interface. The default backend is `rule`.
- `src/query_to_events.py`
  Converts a `MIRAI` query into retrieved evidence and extracted atomic events.
- `src/query_causal_graph.py`
  Builds the current local graph baseline from retrieved evidence and extracted events.
- `src/coarse_graph_builder.py`
  Builds the current coarse causal graph from extracted events.
- `src/refinement_dataset.py`
  Converts a complete coarse graph into graph-level refinement tensors. This is not the Qwen event-pair task; it scores all candidate edges in a graph and can add completion candidates for missing edges.
- `src/refinement_model.py`
  Query-conditioned temporal relational graph refiner for coarse-graph-to-refined-graph learning.
- `src/query_forecast.py`
  Keeps the graph-conditioned forecasting entrypoint scaffold.

## Quick Start

Inspect extracted events:

```bash
python src/query_to_events.py --query-id 1 --event-extractor rule
```

Build a coarse causal graph:

```bash
python src/coarse_graph_builder.py --query-id 1 --event-extractor rule
```

Export MAVEN-ERE event-pair samples for coarse graph training:

```bash
python src/coarse_graph_dataset.py --split train --limit 2
```

Train the coarse graph proposer:

```bash
python src/train_coarse_graph.py --split train --limit 128
```

Train a Qwen LoRA coarse graph proposer:

```bash
python src/train_coarse_graph_qwen.py --split train --limit 128 --model-path models/Qwen2.5-0.5B
```

Run coarse graph proposer inference:

```bash
python src/run_coarse_graph_model.py --split valid --limit 1
```

Run Qwen LoRA coarse graph proposer inference:

```bash
python src/run_coarse_graph_qwen.py --split valid --limit 1 --model-path outputs/coarse_graph_qwen_lora
```

Generate synthetic refinement samples:

```bash
python src/refinement_dataset.py --mode synthetic --num-samples 8
```

Train graph-level refinement on MAVEN-ERE:

```bash
python src/train_refinement.py --dataset-mode maven --limit 2048 --epochs 30
```

Run refinement inference on a coarse graph JSON:

```bash
python src/run_refinement.py --coarse-graph outputs/coarse_graph.json --include-completion-candidates
```

Generate a readable event extraction log:

```bash
python temp/log_event_extraction_samples.py --query-ids 1 2 3 --event-extractor rule
```

## Current Benchmarks

- Main downstream benchmark: `MIRAI`
- Auxiliary graph supervision / evaluation:
  - `MAVEN-ERE`
  - `Event StoryLine Corpus`
  - `Causal News Corpus`
  - optional `MATRES`

## Current Position

The project has already implemented:

1. `MIRAI` query ingestion
2. query-conditioned evidence retrieval
3. replaceable event extraction
4. coarse causal graph construction

The next research step is:

**train and refine the coarse graph, rather than keep extending extraction heuristics.**
