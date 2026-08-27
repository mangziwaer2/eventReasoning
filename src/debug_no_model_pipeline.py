from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from coarse_graph_dataset import DocumentGraphSample
from coarse_graph_dataset import build_event_pair_inference_samples
from coarse_graph_dataset import build_graph_from_pair_predictions
from event_input import EventInputRecord
from event_input import load_event_input_index
from event_input import materialize_event_input
from forecast_trace_prompt import FORECAST_TRACE_SYSTEM_PROMPT
from forecast_trace_prompt import build_structured_forecast_prompt
from forecast_trace_schema import parse_structured_forecast
from mirai_dataset import export_mirai_query_snapshot
from mirai_dataset import get_mirai_query_by_id
from mirai_dataset import load_mirai_news_for_docids
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path
from rl_pipeline_hooks import PipelineTrajectory
from rl_pipeline_hooks import build_pipeline_policy


DEFAULT_EVENT_CODE = "010"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "No-model dry-run for the event-to-graph-to-forecast-to-reward flow. "
            "This script never loads Qwen, torch, transformers, or LoRA adapters."
        )
    )
    parser.add_argument("--input", default=str(REPO_ROOT / "examples" / "event_input.example.json"), help="event-input-v1 JSON/JSONL.")
    parser.add_argument("--query-id", default=None, help="Record selector for JSONL input. Defaults to first/single record.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MIRAI_data.zip"), help="MIRAI zip, only used with --mirai-query-id.")
    parser.add_argument("--split", default="test", help="MIRAI split, only used with --mirai-query-id.")
    parser.add_argument("--mirai-query-id", default=None, help="Optional MIRAI QueryId used to supply query, documents, and gold labels.")
    parser.add_argument("--max-docs", type=int, default=4, help="Maximum documents for MIRAI mode.")
    parser.add_argument("--max-events", type=int, default=16, help="Maximum events kept.")
    parser.add_argument("--max-pairs", type=int, default=64, help="Maximum candidate event pairs.")
    parser.add_argument("--max-sentence-gap", type=int, default=3, help="Candidate event-pair sentence gap.")
    parser.add_argument("--coarse-keep-threshold", type=float, default=0.5, help="Mock coarse confidence threshold.")
    parser.add_argument("--policy", default="forecast_trace_reward", help="Reward policy name.")
    parser.add_argument("--output", default=str(REPO_ROOT / "outputs" / "debug_no_model_pipeline.json"), help="Debug JSON output path. Use '-' to print JSON to stdout.")
    parser.add_argument("--predictions-output", default=None, help="Optional one-row predictions.jsonl compatible with RL training. Use '-' to print it to stdout; omit with --output - to avoid file writes.")
    return parser.parse_args()


def load_record(input_path: Path, query_id: str | None) -> EventInputRecord:
    index = load_event_input_index(input_path)
    if query_id:
        if query_id not in index:
            raise KeyError(f"query_id={query_id!r} not found in {input_path}. Available preview={list(index)[:8]}")
        return index[query_id]
    if not index:
        raise RuntimeError(f"No event input records found in {input_path}.")
    return next(iter(index.values()))


def build_document_sample(args: argparse.Namespace) -> tuple[DocumentGraphSample, dict[str, Any]]:
    input_path = resolve_repo_path(args.input)
    record = load_record(input_path, args.query_id)

    if args.mirai_query_id:
        dataset_path = resolve_repo_path(args.dataset)
        example = get_mirai_query_by_id(dataset_path, query_id=args.mirai_query_id, split=args.split)
        query = example.build_query_spec()
        documents = load_mirai_news_for_docids(dataset_path, example.docids)[: args.max_docs]
        gold = json.loads(export_mirai_query_snapshot(example))
    else:
        if record.query is None:
            raise RuntimeError("The event input record has no query. Provide --mirai-query-id or include query in input.")
        query = record.query
        documents = record.documents
        gold = {
            "query_id": query.query_id,
            "date_str": query.cutoff_time,
            "answer_list": [DEFAULT_EVENT_CODE],
            "debug_gold": True,
        }

    _, _, events = materialize_event_input(record, query=query, documents=documents, max_events=args.max_events)
    sample = DocumentGraphSample(
        sample_id=record.sample_id,
        query=query,
        documents=documents,
        events=events,
        gold_graph=None,
        metadata={
            **record.metadata,
            "debug_no_model": True,
            "source_input": str(input_path),
            "mirai_query_id": args.mirai_query_id,
        },
    )
    return sample, gold


def lexical_overlap_score(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in left.replace(";", " ").replace(",", " ").split() if len(token) > 2}
    right_tokens = {token.lower() for token in right.replace(";", " ").replace(",", " ").split() if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def mock_pair_prediction(pair_sample) -> dict[str, Any]:
    event_lookup = {event.event_id: event for event in pair_sample.events}
    source = event_lookup[pair_sample.source_event_id]
    target = event_lookup[pair_sample.target_event_id]
    same_doc = source.document_id == target.document_id
    sentence_gap = int(target.sentence_index) - int(source.sentence_index)
    shared_participants = set(source.participants) & set(target.participants)
    overlap = lexical_overlap_score(source.text, target.text)
    candidate_prior = float(pair_sample.metadata.get("candidate_score", pair_sample.score) or 0.0)

    if same_doc and sentence_gap > 0:
        relation_type = "precedes"
        confidence = 0.55 + min(0.25, 0.08 * sentence_gap)
    elif shared_participants:
        relation_type = "causes"
        confidence = 0.62
    elif overlap >= 0.15:
        relation_type = "causes"
        confidence = 0.58
    else:
        relation_type = "none"
        confidence = max(0.55, min(0.95, 1.0 - candidate_prior))

    return {
        "relation_type": relation_type,
        "confidence": round(max(0.0, min(confidence, 1.0)), 4),
        "debug_features": {
            "same_doc": same_doc,
            "sentence_gap": sentence_gap,
            "shared_participants": sorted(shared_participants),
            "lexical_overlap": round(overlap, 4),
            "candidate_prior": round(candidate_prior, 4),
        },
    }


def mock_forecast_json(prompt_bundle, gold: dict[str, Any]) -> str:
    gold_codes = [str(item) for item in gold.get("answer_list", []) if str(item)]
    selected_code = gold_codes[0] if gold_codes else DEFAULT_EVENT_CODE

    event_refs = list(prompt_bundle.event_ref_to_id)
    edge_refs = list(prompt_bundle.edge_ref_to_id)
    support_event = event_refs[0] if event_refs else ""
    support_edge = edge_refs[0] if edge_refs else ""
    payload = {
        "forecast_trace": {
            "intermediate_events": [
                {
                    "trace_event_id": "ft_1",
                    "event": {
                        "trigger": "respond",
                        "mention": "one actor responds to the prior observed event",
                        "actors": [],
                        "relative_time": "t-1",
                    },
                    "supporting_event_ids": [support_event] if support_event else [],
                    "supporting_edge_ids": [support_edge] if support_edge else [],
                    "expected_effect": "debug hypothesis linking visible history to the final event code",
                    "confidence": 0.66,
                }
            ],
            "trace_edges": [
                {"source_id": support_event, "target_id": "ft_1", "relation_type": "causes", "confidence": 0.61},
                {
                    "source_id": "ft_1",
                    "target_id": f"answer_{selected_code}",
                    "relation_type": "raises_likelihood",
                    "confidence": 0.6,
                },
            ]
            if support_event
            else [],
        },
        "final_answer": {
            "event_code": selected_code,
            "event": str(gold.get("relation_name", "debug target event")),
            "confidence": 0.64,
            "supporting_event_ids": [support_event] if support_event else [],
            "supporting_edge_ids": [support_edge] if support_edge else [],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    document_sample, gold = build_document_sample(args)
    pair_samples = build_event_pair_inference_samples(
        sample=document_sample,
        max_sentence_gap=args.max_sentence_gap,
        max_pairs=args.max_pairs,
    )
    pair_predictions = [mock_pair_prediction(pair_sample) for pair_sample in pair_samples]
    coarse_graph = build_graph_from_pair_predictions(
        document_sample=document_sample,
        pair_samples=pair_samples,
        pair_predictions=pair_predictions,
        keep_threshold=args.coarse_keep_threshold,
    )
    refined_graph = coarse_graph

    prompt_bundle = build_structured_forecast_prompt(
        query=document_sample.query,
        documents=document_sample.documents,
        refined_graph=refined_graph,
    )
    raw_forecast = mock_forecast_json(prompt_bundle, gold=gold)
    forecast_prediction = parse_structured_forecast(raw_forecast)

    trajectory = PipelineTrajectory(
        sample_id=document_sample.sample_id,
        metadata={
            "policy": args.policy,
            "query": document_sample.query.to_dict(),
            "event_ref_to_id": prompt_bundle.event_ref_to_id,
            "edge_ref_to_id": prompt_bundle.edge_ref_to_id,
            "refined_graph": refined_graph.to_dict(),
        },
    )
    trajectory.add_step(
        "event_input",
        observation={"document_count": len(document_sample.documents), "query": document_sample.query.text},
        action={"source": "event-input-v1", "max_events": args.max_events},
        metadata={"event_count": len(document_sample.events), "valid": True, "parsed_json": True},
    )
    trajectory.add_step(
        "coarse_graph",
        observation={"event_count": len(document_sample.events), "candidate_pairs": len(pair_samples)},
        action={"mock": True, "keep_threshold": args.coarse_keep_threshold},
        metadata={"coarse_edge_count": len(coarse_graph.edges), "raw_preview": pair_predictions[:5]},
    )
    trajectory.add_step(
        "refinement",
        observation={"coarse_edge_count": len(coarse_graph.edges)},
        action={"skip_refinement": True},
        metadata={"skipped": True, "refined_edge_count": len(refined_graph.edges), "graph_source": "coarse_graph"},
    )

    policy = build_pipeline_policy(args.policy)
    reward_breakdown = policy.compute_reward_breakdown(forecast_prediction, gold, trajectory)
    reward = float(reward_breakdown.get("total", 0.0))
    trajectory.final_reward = reward
    trajectory.add_step(
        "forecast",
        observation={"event_count": len(refined_graph.events), "edge_count": len(refined_graph.edges)},
        action={"mock": True, "prompt_chars": len(prompt_bundle.prompt)},
        reward=reward,
        metadata={
            "prediction_mode": "forecast-trace",
            "event_ref_to_id": prompt_bundle.event_ref_to_id,
            "edge_ref_to_id": prompt_bundle.edge_ref_to_id,
            "refined_graph": refined_graph.to_dict(),
            "raw_response": raw_forecast,
            "prediction": forecast_prediction,
            "reward_breakdown": reward_breakdown,
        },
    )

    output = {
        "debug_mode": "no_model",
        "model_loading": False,
        "sample_id": document_sample.sample_id,
        "inputs": {
            "query": document_sample.query.to_dict(),
            "documents": [document.to_dict() for document in document_sample.documents],
            "events": [event.to_dict() for event in document_sample.events],
            "gold": gold,
        },
        "coarse": {
            "candidate_pair_count": len(pair_samples),
            "pair_predictions_preview": [
                {
                    "source_event_id": pair.source_event_id,
                    "target_event_id": pair.target_event_id,
                    "prediction": prediction,
                }
                for pair, prediction in list(zip(pair_samples, pair_predictions))[:12]
            ],
            "graph": coarse_graph.to_dict(),
        },
        "refinement": {"skipped": True, "graph": refined_graph.to_dict()},
        "forecast": {
            "prompt": prompt_bundle.prompt,
            "event_ref_to_id": prompt_bundle.event_ref_to_id,
            "edge_ref_to_id": prompt_bundle.edge_ref_to_id,
            "raw_response": raw_forecast,
            "prediction": forecast_prediction,
        },
        "reward": {"value": reward, "breakdown": reward_breakdown},
        "rl_training_preview": {
            "input_file_expected_by_train_forecast_trace_rl": "predictions.jsonl with forecast_prompt + raw_response + trajectory",
            "loss": "weighted token NLL over forecast completion only",
            "weight": "reward-transformed sample weight",
        },
        "trajectory": trajectory.to_dict(),
    }

    output_text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(output_text)
        output_path_text = "<stdout>"
    else:
        output_path = resolve_repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        output_path_text = str(output_path)

    predictions_output = (
        resolve_repo_path(args.predictions_output)
        if args.predictions_output and args.predictions_output != "-"
        else None
    )
    row = {
        "query_id": document_sample.query.query_id,
        "mirai_query": gold,
        "document_count": len(document_sample.documents),
        "event_input": {"source": "debug_no_model", "valid": True, "parsed_json": True, "event_count": len(document_sample.events)},
        "coarse": {"generation_mode": "mock_no_model", "candidate_pairs": len(pair_samples), "edge_count": len(coarse_graph.edges)},
        "refinement": {"skipped": True, "edge_count": len(refined_graph.edges), "graph_source": "coarse_graph"},
        "forecast_prompt": prompt_bundle.prompt,
        "forecast_system_prompt": FORECAST_TRACE_SYSTEM_PROMPT,
        "forecast_prediction": forecast_prediction,
        "reward": reward,
        "reward_breakdown": reward_breakdown,
        "trajectory": trajectory.to_dict(),
    }
    row_text = json.dumps(row, ensure_ascii=False) + "\n"
    if args.predictions_output == "-":
        print(row_text, end="")
        predictions_output_text = "<stdout>"
    elif predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions_output.write_text(row_text, encoding="utf-8")
        predictions_output_text = str(predictions_output)
    elif args.output != "-":
        fallback_predictions_output = resolve_repo_path(args.output).with_suffix(".predictions.jsonl")
        fallback_predictions_output.write_text(row_text, encoding="utf-8")
        predictions_output_text = str(fallback_predictions_output)
    else:
        predictions_output_text = "<not written>"

    print(
        " | ".join(
            [
                "debug no-model pipeline complete",
                f"events={len(document_sample.events)}",
                f"pairs={len(pair_samples)}",
                f"coarse_edges={len(coarse_graph.edges)}",
                f"reward={reward:.4f}",
                f"output={output_path_text}",
                f"rl_jsonl={predictions_output_text}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
