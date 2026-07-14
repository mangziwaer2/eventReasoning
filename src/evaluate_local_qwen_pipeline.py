from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from causal_graph import CoarseCausalGraph
from causal_graph import EvidenceSpan
from causal_graph import EventNode
from coarse_graph_dataset import DocumentGraphSample
from coarse_graph_dataset import build_event_pair_inference_samples
from coarse_graph_dataset import build_graph_from_pair_predictions
from coarse_graph_dataset import parse_pair_payload
from event_extraction import format_event_mention
from event_extraction import normalize_text
from event_extraction import split_sentences
from event_input import EventInputValidationError
from event_input import load_event_input_index
from event_input import materialize_event_input
from local_llm import LocalGenerationUnavailable
from local_llm import LocalQwenGenerator
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_trained_qwen_lora
from mirai_dataset import export_mirai_query_snapshot
from mirai_dataset import load_mirai_news_for_docids
from mirai_dataset import load_mirai_queries
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path
from refinement_dataset import EDGE_FEATURE_DIM
from refinement_dataset import load_refinement_sample_from_coarse_graph
from rl_pipeline_hooks import PipelineTrajectory
from rl_pipeline_hooks import build_pipeline_policy
from run_refinement import build_refined_graph
from run_refinement import load_model_config
from run_refinement import load_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local no-API evaluation: pre-extracted events or an optional frozen Qwen extractor "
            "provide event nodes, Qwen LoRA builds the coarse graph, refinement produces a causal graph, "
            "and native Qwen forecasts future events."
        )
    )
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MIRAI_data.zip"), help="Path to MIRAI_data.zip.")
    parser.add_argument("--split", default="test", help="MIRAI split name.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum query examples. Use 0 for the full split.")
    parser.add_argument("--query-id", default=None, help="Optional single MIRAI QueryId.")

    parser.add_argument("--model-path", default=str(REPO_ROOT / "models" / "Qwen2.5-0.5B"), help="Native local Qwen model path for forecasting and optional event extraction.")
    parser.add_argument("--coarse-base-model-path", default=None, help="Base Qwen model for the coarse LoRA adapter. Defaults to --model-path.")
    parser.add_argument("--coarse-adapter-path", default=str(REPO_ROOT / "outputs" / "coarse_graph_qwen_lora_4090_full" / "best_adapter"), help="Coarse graph Qwen LoRA adapter path.")
    parser.add_argument("--refinement-model-path", default=str(REPO_ROOT / "outputs" / "refinement_graph_4090_full" / "refinement_model.pt"), help="Refinement model path.")

    parser.add_argument("--event-source", choices=["precomputed", "qwen"], default="precomputed", help="Event input source. precomputed is the research setting; qwen is an optional frozen-extractor baseline.")
    parser.add_argument("--precomputed-events", default=None, help="event-input-v1 JSON/JSONL path required when event-source=precomputed.")
    parser.add_argument("--max-docs", type=int, default=4, help="Maximum MIRAI documents used as authoritative context.")
    parser.add_argument("--max-document-chars", type=int, default=900, help="Maximum characters per document in the optional Qwen extraction prompt.")
    parser.add_argument("--max-events", type=int, default=16, help="Maximum pre-extracted or Qwen-extracted events kept per query.")
    parser.add_argument("--event-extraction-temperature", type=float, default=0.0, help="Native Qwen temperature for event extraction.")
    parser.add_argument("--event-extraction-max-new-tokens", type=int, default=768, help="Maximum native Qwen tokens for event extraction JSON.")

    parser.add_argument("--max-sentence-gap", type=int, default=3, help="Maximum same-document sentence gap for coarse candidate pairs.")
    parser.add_argument("--max-pairs", type=int, default=64, help="Maximum coarse event pairs. Use 0 for all candidates.")
    parser.add_argument("--coarse-batch-size", type=int, default=8, help="Batch size for Qwen LoRA pair generation.")
    parser.add_argument("--coarse-max-length", type=int, default=1024, help="Maximum coarse pair prompt length.")
    parser.add_argument("--coarse-max-new-tokens", type=int, default=48, help="Maximum generated tokens for coarse relation JSON.")
    parser.add_argument("--coarse-keep-threshold", type=float, default=0.5, help="Minimum coarse relation score kept as an edge.")

    parser.add_argument("--include-completion-candidates", dest="include_completion_candidates", action="store_true", default=True, help="Add heuristic completion candidates before refinement.")
    parser.add_argument("--no-completion-candidates", dest="include_completion_candidates", action="store_false", help="Disable refinement completion candidates.")
    parser.add_argument("--max-completion-edges", type=int, default=64, help="Maximum completion candidates for refinement. Use 0 for no cap.")
    parser.add_argument("--refinement-keep-threshold", type=float, default=0.5, help="Minimum refinement keep probability.")

    parser.add_argument("--forecast-temperature", type=float, default=0.0, help="Native Qwen temperature for final forecasting.")
    parser.add_argument("--forecast-max-new-tokens", type=int, default=320, help="Maximum native Qwen tokens for forecast JSON.")
    parser.add_argument("--max-graph-events-in-prompt", type=int, default=24, help="Maximum graph events shown to forecast Qwen.")
    parser.add_argument("--max-graph-edges-in-prompt", type=int, default=48, help="Maximum graph edges shown to forecast Qwen.")

    parser.add_argument("--policy", default="mirai_code_reward", help="RL hook policy name: noop or mirai_code_reward.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "local_qwen_pipeline_eval"), help="Evaluation output directory.")
    parser.add_argument("--log-every", type=int, default=1, help="Print progress every N samples. Use 0 to disable.")
    return parser.parse_args()


def compact_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(stripped[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def resolve_generation_eos_ids(tokenizer) -> list[int] | int | None:
    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
    if not eos_ids:
        return None
    return eos_ids if len(eos_ids) > 1 else eos_ids[0]


def build_event_extraction_prompt(query_text: str, documents, max_document_chars: int, max_events: int) -> str:
    document_blocks: list[str] = []
    for document in documents:
        document_blocks.append(
            "\n".join(
                [
                    f"[Document {document.document_id}]",
                    f"Title: {compact_text(document.title, 180)}",
                    f"Date: {document.publish_time or '-'}",
                    f"Text: {compact_text(document.text, max_document_chars)}",
                ]
            )
        )
    return (
        "Extract concrete event mentions that are useful for forecasting the query.\n"
        "Use only the provided documents. Do not invent events.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "events": [\n'
        "    {\n"
        '      "document_id": "same id as the source document",\n'
        '      "trigger": "short event trigger word or phrase",\n'
        '      "event": "one concrete event mention",\n'
        '      "evidence": "short source sentence or clause",\n'
        '      "participants": ["entity"],\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Return at most {max_events} events.\n\n"
        f"Query: {query_text}\n\n"
        "Documents:\n"
        + "\n\n".join(document_blocks)
    )


def locate_sentence(document, evidence: str) -> tuple[int, str]:
    candidates = [document.title] + split_sentences(document.text)
    evidence_norm = normalize_text(evidence)
    for index, sentence in enumerate(candidates):
        sentence_norm = normalize_text(sentence)
        if evidence_norm and (evidence_norm in sentence_norm or sentence_norm in evidence_norm):
            return index, sentence
    return 0, evidence or document.title


def parse_qwen_events(raw_response: str, query, documents, max_events: int) -> tuple[list[EventNode], dict[str, Any]]:
    payload = extract_first_json_object(raw_response)
    metadata = {"parsed_json": payload is not None, "raw_event_count": 0}
    if payload is None:
        return [], metadata
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list):
        return [], metadata
    metadata["raw_event_count"] = len(raw_events)

    doc_lookup = {document.document_id: document for document in documents}
    events: list[EventNode] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        document_id = str(item.get("document_id", "")).strip()
        document = doc_lookup.get(document_id)
        if document is None and documents:
            document = documents[0]
            document_id = document.document_id
        if document is None:
            continue

        trigger = str(item.get("trigger", "")).strip()
        event_text_raw = str(item.get("event", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not event_text_raw and not evidence:
            continue
        sentence_index, sentence_text = locate_sentence(document, evidence or event_text_raw)
        event_text = format_event_mention(trigger=trigger or event_text_raw.split(" ")[0], context=event_text_raw or sentence_text)
        key = (document_id, trigger.lower(), normalize_text(event_text))
        if key in seen:
            continue
        seen.add(key)

        participants = item.get("participants", [])
        if not isinstance(participants, list):
            participants = []
        try:
            confidence = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        event_id = f"qwen_e{len(events)}"
        events.append(
            EventNode(
                event_id=event_id,
                text=event_text,
                normalized_text=normalize_text(event_text_raw or sentence_text),
                document_id=document_id,
                sentence_index=sentence_index,
                participants=[str(participant) for participant in participants],
                node_type="observed",
                confidence=max(0.0, min(confidence, 1.0)),
                evidence=[
                    EvidenceSpan(
                        document_id=document_id,
                        sentence_index=sentence_index,
                        text=sentence_text,
                    )
                ],
                metadata={
                    "trigger": trigger,
                    "event_mention": event_text,
                    "event_context": event_text_raw or sentence_text,
                    "sentence_text": sentence_text,
                    "publish_time": document.publish_time,
                    "extracted_by": "native_qwen",
                },
            )
        )
        if len(events) >= max_events:
            break
    return events, metadata


def build_document_sample_from_events(
    example,
    query,
    documents,
    events: list[EventNode],
    event_source: str,
) -> DocumentGraphSample:
    return DocumentGraphSample(
        sample_id=f"mirai_{example.query_id}",
        query=query,
        documents=documents,
        events=events,
        gold_graph=None,
        metadata={
            "dataset": "MIRAI",
            "query_id": example.query_id,
            "event_source": event_source,
        },
    )


def format_pair_prompt(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou classify directed relations between event pairs.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_coarse_predictions(model, tokenizer, torch, device, pair_samples, args: argparse.Namespace):
    predictions: list[dict[str, Any] | None] = []
    raw_generations: list[str] = []
    if not pair_samples:
        return predictions, raw_generations
    tokenizer.padding_side = "left"
    batch_size = max(1, args.coarse_batch_size)
    eos_token_id = resolve_generation_eos_ids(tokenizer)
    with torch.no_grad():
        for start in range(0, len(pair_samples), batch_size):
            batch = pair_samples[start : start + batch_size]
            prompts = []
            for pair_sample in batch:
                item = pair_sample.to_instruction_example(include_query=False, document_mode="title")
                prompts.append(format_pair_prompt(item["prompt"]))
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=args.coarse_max_length,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.coarse_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_token_id,
            )
            prompt_width = input_ids.shape[-1]
            for row_index in range(len(batch)):
                raw = tokenizer.decode(outputs[row_index][prompt_width:], skip_special_tokens=True).strip()
                raw_generations.append(raw)
                predictions.append(parse_pair_payload(raw))
    return predictions, raw_generations


def load_refinement_model(args: argparse.Namespace, torch, device):
    from refinement_model import TemporalRelationalEdgeRefiner

    model_path = resolve_repo_path(args.refinement_model_path)
    hidden_dim, message_steps, edge_dim, dropout = load_model_config(
        model_path,
        hidden_dim=None,
        message_steps=None,
        edge_dim=None,
        dropout=None,
    )
    model = TemporalRelationalEdgeRefiner(
        edge_dim=edge_dim or EDGE_FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_message_passing_steps=message_steps,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(load_model_state(model_path, torch))
    model.eval()
    return model


def refine_graph(coarse_graph: CoarseCausalGraph, sample_id: str, temp_dir: Path, refinement_model, torch, device, args: argparse.Namespace) -> CoarseCausalGraph:
    graph_path = temp_dir / f"{sample_id}_coarse.json"
    graph_path.write_text(json.dumps({"coarse_graph": coarse_graph.to_dict()}, ensure_ascii=False), encoding="utf-8")
    sample = load_refinement_sample_from_coarse_graph(
        graph_path,
        sample_id=sample_id,
        include_completion_candidates=args.include_completion_candidates,
        max_completion_edges=args.max_completion_edges if args.max_completion_edges > 0 else None,
    )
    node_features = torch.tensor(sample.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(sample.edge_index, dtype=torch.long, device=device)
    edge_features = torch.tensor(sample.edge_features, dtype=torch.float32, device=device)
    query_features = torch.tensor(sample.query_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = refinement_model(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            query_features=query_features,
        )
    keep_probs = torch.sigmoid(outputs["edge_keep_logits"]).detach().cpu().tolist()
    type_predictions = outputs["edge_type_logits"].argmax(dim=-1).detach().cpu().tolist()
    strength_predictions = outputs["edge_strengths"].detach().cpu().tolist()
    return build_refined_graph(
        coarse_graph=coarse_graph,
        edge_descriptions=list(sample.metadata.get("edge_descriptions", [])),
        keep_probs=keep_probs,
        type_predictions=type_predictions,
        strength_predictions=strength_predictions,
        keep_threshold=args.refinement_keep_threshold,
    )


def render_refined_graph_prompt(example, query, graph: CoarseCausalGraph, args: argparse.Namespace) -> str:
    event_lines = []
    for event in graph.events[: args.max_graph_events_in_prompt]:
        event_lines.append(
            f"- {event.event_id} | doc={event.document_id} | sent={event.sentence_index} | {event.text}"
        )
    edge_lines = []
    for edge in sorted(graph.edges, key=lambda item: float(item.score), reverse=True)[: args.max_graph_edges_in_prompt]:
        edge_lines.append(
            f"- {edge.source_event_id} --{edge.relation_type}:{float(edge.score):.3f}--> {edge.target_event_id}"
        )
    doc_lines = [
        f"- {document.document_id} | {document.publish_time or '-'} | {compact_text(document.title, 160)}"
        for document in graph.documents
    ]
    return (
        "Predict the next important event for the query using the refined causal graph.\n"
        "Use only the graph and the document summaries. Do not use outside knowledge.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "abstain": false,\n'
        '  "predicted_event_base_code": "three digit CAMEO event base code, or empty string if unknown",\n'
        '  "alternative_event_base_codes": ["optional three digit code"],\n'
        '  "predicted_relation_name": "short relation name",\n'
        '  "forecast_event": "one concrete future event hypothesis",\n'
        '  "confidence": 0.0,\n'
        '  "rationale": "brief graph-grounded explanation",\n'
        '  "support_event_ids": ["event id"]\n'
        "}\n\n"
        f"QueryId: {example.query_id}\n"
        f"Query: {query.text}\n"
        f"Cutoff date: {query.cutoff_time or '-'}\n"
        f"Actors: {', '.join(query.focus_entities) or '-'}\n\n"
        "Dataset documents supplied to this local run:\n"
        + "\n".join(doc_lines)
        + "\n\nRefined graph events:\n"
        + ("\n".join(event_lines) if event_lines else "- none")
        + "\n\nRefined causal edges:\n"
        + ("\n".join(edge_lines) if edge_lines else "- none")
    )


def parse_forecast_json(raw_response: str) -> dict[str, Any]:
    payload = extract_first_json_object(raw_response)
    if payload is None:
        code_match = re.search(r"\b\d{3}\b", raw_response)
        return {
            "parsed_json": False,
            "abstain": False,
            "predicted_event_base_code": code_match.group(0) if code_match else "",
            "alternative_event_base_codes": [],
            "forecast_event": raw_response.strip(),
            "confidence": 0.0,
            "rationale": "",
            "support_event_ids": [],
        }
    alternatives = payload.get("alternative_event_base_codes", [])
    if not isinstance(alternatives, list):
        alternatives = []
    support_event_ids = payload.get("support_event_ids", [])
    if not isinstance(support_event_ids, list):
        support_event_ids = []
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "parsed_json": True,
        "abstain": bool(payload.get("abstain", False)),
        "predicted_event_base_code": str(payload.get("predicted_event_base_code", "")).strip(),
        "alternative_event_base_codes": [str(item).strip() for item in alternatives if str(item).strip()],
        "predicted_relation_name": str(payload.get("predicted_relation_name", "")).strip(),
        "forecast_event": str(payload.get("forecast_event", "")).strip(),
        "confidence": max(0.0, min(confidence, 1.0)),
        "rationale": str(payload.get("rationale", "")).strip(),
        "support_event_ids": [str(item) for item in support_event_ids],
    }


def score_prediction(prediction: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    gold_codes = {str(item).strip() for item in gold.get("answer_list", []) if str(item).strip()}
    primary_code = str(prediction.get("predicted_event_base_code", "")).strip()
    alternatives = prediction.get("alternative_event_base_codes", [])
    if not isinstance(alternatives, list):
        alternatives = []
    all_codes = [primary_code] + [str(item).strip() for item in alternatives if str(item).strip()]
    return {
        "gold_codes": sorted(gold_codes),
        "predicted_code": primary_code,
        "code_hit": bool(primary_code and primary_code in gold_codes),
        "code_hit_with_alternatives": any(code in gold_codes for code in all_codes),
    }


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    if predictions_path.exists():
        predictions_path.unlink()

    dataset_path = resolve_repo_path(args.dataset)
    if args.query_id:
        examples = [example for example in load_mirai_queries(dataset_path, split=args.split) if example.query_id == str(args.query_id)]
    else:
        examples = load_mirai_queries(
            dataset_path,
            split=args.split,
            limit=args.limit if args.limit > 0 else None,
        )
    if not examples:
        raise RuntimeError(f"No MIRAI examples found for split={args.split!r}, query_id={args.query_id!r}.")

    precomputed_event_index = {}
    precomputed_events_path: Path | None = None
    if args.event_source == "precomputed":
        if not args.precomputed_events:
            raise EventInputValidationError("--precomputed-events is required when --event-source=precomputed")
        precomputed_events_path = resolve_repo_path(args.precomputed_events)
        precomputed_event_index = load_event_input_index(precomputed_events_path)
        missing_query_ids = [example.query_id for example in examples if example.query_id not in precomputed_event_index]
        if missing_query_ids:
            preview = missing_query_ids[:10]
            raise EventInputValidationError(
                f"Precomputed event input is missing {len(missing_query_ids)} requested MIRAI query ids; preview={preview}."
            )

    policy = build_pipeline_policy(args.policy)
    native_qwen = LocalQwenGenerator(resolve_repo_path(args.model_path), max_new_tokens=max(args.event_extraction_max_new_tokens, args.forecast_max_new_tokens))
    try:
        coarse_base_model_path = resolve_repo_path(args.coarse_base_model_path) if args.coarse_base_model_path else resolve_repo_path(args.model_path)
        coarse_model, coarse_tokenizer, torch = load_trained_qwen_lora(
            base_model_path=coarse_base_model_path,
            adapter_path=resolve_repo_path(args.coarse_adapter_path),
        )
    except LoraUnavailable as exc:
        raise RuntimeError(str(exc)) from exc
    coarse_model.eval()
    device = next(coarse_model.parameters()).device
    refinement_model = load_refinement_model(args, torch, device)

    started = time.time()
    rows: list[dict[str, Any]] = []
    print(
        " | ".join(
            [
                "local qwen pipeline eval",
                f"split={args.split}",
                f"samples={len(examples)}",
                f"event_source={args.event_source}",
                f"native_device={native_qwen.device}",
                f"coarse_device={device}",
            ]
        ),
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="local_qwen_pipeline_") as temp_name:
        temp_dir = Path(temp_name)
        for index, example in enumerate(examples, start=1):
            query = example.build_query_spec()
            documents = load_mirai_news_for_docids(dataset_path, example.docids)[: args.max_docs]
            trajectory = PipelineTrajectory(
                sample_id=example.query_id,
                metadata={
                    "policy": policy.name,
                    "query": query.to_dict(),
                },
            )

            if args.event_source == "precomputed":
                record = precomputed_event_index[example.query_id]
                _, _, events = materialize_event_input(
                    record,
                    query=query,
                    documents=documents,
                    max_events=args.max_events,
                )
                event_input_metadata = {
                    "source": "precomputed",
                    "valid": True,
                    "parsed_json": True,
                    "schema_version": record.schema_version,
                    "raw_event_count": len(record.events),
                    "event_count": len(events),
                    "extractor_name": record.metadata.get("extractor_name"),
                }
                event_input_action = {
                    "source": "precomputed",
                    "event_input_path": str(precomputed_events_path),
                    "max_events": args.max_events,
                }
            else:
                extraction_prompt = build_event_extraction_prompt(
                    query_text=query.text,
                    documents=documents,
                    max_document_chars=args.max_document_chars,
                    max_events=args.max_events,
                )
                raw_event_response = native_qwen.generate(
                    extraction_prompt,
                    temperature=args.event_extraction_temperature,
                    system_prompt="You extract structured event mentions and return strict JSON only.",
                    max_new_tokens=args.event_extraction_max_new_tokens,
                )
                events, event_parse_metadata = parse_qwen_events(raw_event_response, query, documents, args.max_events)
                event_input_metadata = {
                    **event_parse_metadata,
                    "source": "qwen",
                    "valid": bool(event_parse_metadata.get("parsed_json")) and len(events) >= 2,
                    "event_count": len(events),
                    "raw_response": raw_event_response,
                }
                event_input_action = {
                    "source": "qwen",
                    "prompt_chars": len(extraction_prompt),
                    "max_events": args.max_events,
                }
            trajectory.add_step(
                "event_input",
                observation={"document_count": len(documents), "query": query.text},
                action=event_input_action,
                metadata=event_input_metadata,
            )

            document_sample = build_document_sample_from_events(
                example,
                query,
                documents,
                events,
                event_source=args.event_source,
            )
            pair_samples = build_event_pair_inference_samples(
                sample=document_sample,
                max_sentence_gap=args.max_sentence_gap,
                max_pairs=args.max_pairs,
            )
            pair_predictions, pair_raw_generations = generate_coarse_predictions(
                model=coarse_model,
                tokenizer=coarse_tokenizer,
                torch=torch,
                device=device,
                pair_samples=pair_samples,
                args=args,
            )
            coarse_graph = build_graph_from_pair_predictions(
                document_sample=document_sample,
                pair_samples=pair_samples,
                pair_predictions=pair_predictions,
                keep_threshold=args.coarse_keep_threshold,
            )
            trajectory.add_step(
                "coarse_graph",
                observation={"event_count": len(events), "candidate_pairs": len(pair_samples)},
                action={"keep_threshold": args.coarse_keep_threshold, "max_pairs": args.max_pairs},
                metadata={
                    "parse_rate": safe_div(sum(1 for item in pair_predictions if item is not None), len(pair_predictions)),
                    "coarse_edge_count": len(coarse_graph.edges),
                    "raw_preview": pair_raw_generations[:3],
                },
            )

            refined_graph = refine_graph(
                coarse_graph=coarse_graph,
                sample_id=f"mirai_{example.query_id}",
                temp_dir=temp_dir,
                refinement_model=refinement_model,
                torch=torch,
                device=device,
                args=args,
            )
            trajectory.add_step(
                "refinement",
                observation={"coarse_edge_count": len(coarse_graph.edges)},
                action={
                    "keep_threshold": args.refinement_keep_threshold,
                    "include_completion_candidates": args.include_completion_candidates,
                    "max_completion_edges": args.max_completion_edges,
                },
                metadata={"refined_edge_count": len(refined_graph.edges)},
            )

            forecast_prompt = render_refined_graph_prompt(example, query, refined_graph, args)
            raw_forecast = native_qwen.generate(
                forecast_prompt,
                temperature=args.forecast_temperature,
                system_prompt="You forecast future events from a refined causal graph and return strict JSON only.",
                max_new_tokens=args.forecast_max_new_tokens,
            )
            forecast_prediction = parse_forecast_json(raw_forecast)
            score = score_prediction(forecast_prediction, example.gold_summary())
            reward = policy.compute_reward(forecast_prediction, example.gold_summary(), trajectory)
            trajectory.final_reward = reward
            trajectory.add_step(
                "forecast",
                observation={"event_count": len(refined_graph.events), "edge_count": len(refined_graph.edges)},
                action={"prompt_chars": len(forecast_prompt)},
                reward=reward,
                metadata={"raw_response": raw_forecast, "prediction": forecast_prediction, "score": score},
            )

            row = {
                "query_id": example.query_id,
                "mirai_query": json.loads(export_mirai_query_snapshot(example)),
                "document_count": len(documents),
                "event_input": event_input_metadata,
                "coarse": {
                    "candidate_pairs": len(pair_samples),
                    "parse_rate": safe_div(sum(1 for item in pair_predictions if item is not None), len(pair_predictions)),
                    "edge_count": len(coarse_graph.edges),
                },
                "refinement": {
                    "edge_count": len(refined_graph.edges),
                },
                "forecast_prediction": forecast_prediction,
                "score": score,
                "reward": reward,
                "trajectory": trajectory.to_dict(),
            }
            rows.append(row)
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            if args.log_every > 0 and (index % args.log_every == 0 or index == len(examples)):
                print(
                    " | ".join(
                        [
                            f"evaluated {index}/{len(examples)}",
                            f"events={len(events)}",
                            f"coarse_edges={len(coarse_graph.edges)}",
                            f"refined_edges={len(refined_graph.edges)}",
                            f"code_hit={int(score['code_hit'])}",
                            f"reward={reward:.3f}",
                            f"time={format_seconds(time.time() - started)}",
                        ]
                    ),
                    flush=True,
                )

    metrics = {
        "config": {
            **vars(args),
            "dataset": str(dataset_path),
            "model_path": str(resolve_repo_path(args.model_path)),
            "coarse_base_model_path": str(resolve_repo_path(args.coarse_base_model_path)) if args.coarse_base_model_path else str(resolve_repo_path(args.model_path)),
            "coarse_adapter_path": str(resolve_repo_path(args.coarse_adapter_path)),
            "refinement_model_path": str(resolve_repo_path(args.refinement_model_path)),
            "precomputed_events": str(precomputed_events_path) if precomputed_events_path is not None else None,
            "policy": policy.name,
        },
        "samples": len(rows),
        "event_input_success_rate": safe_div(sum(1 for row in rows if row["event_input"]["valid"]), len(rows)),
        "event_extraction_parse_rate": (
            safe_div(sum(1 for row in rows if row["event_input"]["parsed_json"]), len(rows))
            if args.event_source == "qwen"
            else None
        ),
        "forecast_parse_rate": safe_div(sum(1 for row in rows if row["forecast_prediction"]["parsed_json"]), len(rows)),
        "code_hit_rate": safe_div(sum(1 for row in rows if row["score"]["code_hit"]), len(rows)),
        "code_hit_with_alternatives_rate": safe_div(
            sum(1 for row in rows if row["score"]["code_hit_with_alternatives"]),
            len(rows),
        ),
        "average_reward": safe_div(sum(float(row["reward"]) for row in rows), len(rows)),
        "average_event_count": safe_div(sum(int(row["event_input"]["event_count"]) for row in rows), len(rows)),
        "average_coarse_edge_count": safe_div(sum(int(row["coarse"]["edge_count"]) for row in rows), len(rows)),
        "average_refined_edge_count": safe_div(sum(int(row["refinement"]["edge_count"]) for row in rows), len(rows)),
        "outputs": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        " | ".join(
            [
                f"saved metrics to {metrics_path}",
                f"code_hit_rate={metrics['code_hit_rate']:.4f}",
                f"avg_reward={metrics['average_reward']:.4f}",
                f"elapsed={format_seconds(float(metrics['elapsed_seconds']))}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (LocalGenerationUnavailable, EventInputValidationError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
