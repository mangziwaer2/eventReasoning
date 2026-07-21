from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from chat_qwen3_thinking import DEFAULT_LOCAL_MODEL
from chat_qwen3_thinking import Qwen3ThinkingChat
from chat_qwen3_thinking import Qwen3ThinkingUnavailable
from coarse_graph_dataset import DocumentGraphSample
from coarse_graph_dataset import load_maven_document_graph_samples
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


ALLOWED_RELATIONS = ("precedes", "causes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot Qwen3 document-to-coarse-graph evaluation against MAVEN-ERE gold edges."
    )
    parser.add_argument("--model-path", default=DEFAULT_LOCAL_MODEL, help="Local Qwen3 model directory or Hugging Face model id.")
    parser.add_argument("--allow-download", action="store_true", help="Allow Transformers to download a Hugging Face model id.")
    parser.add_argument("--device", default="auto", help="Device such as auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"))
    parser.add_argument("--split", default="valid", help="MAVEN split with gold relations. Use valid for initial testing.")
    parser.add_argument("--limit", type=int, default=3, help="Number of readable samples to evaluate.")
    parser.add_argument("--sample-offset", type=int, default=0, help="Skip this many valid graph samples.")
    parser.add_argument("--shuffle", action="store_true", help="Randomly sample from the loaded split instead of taking the first rows.")
    parser.add_argument("--max-events", type=int, default=16, help="Maximum gold event mentions shown per document.")
    parser.add_argument(
        "--gold-scope",
        choices=["causal", "all"],
        default="causal",
        help="causal compares MAVEN CAUSE/PRECONDITION only; all also includes temporal relations.",
    )
    parser.add_argument("--max-document-chars", type=int, default=12000, help="Maximum document characters shown to Qwen3.")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-thinking", action="store_true", help="Include a bounded thinking preview in readable output.")
    parser.add_argument("--save-thinking", action="store_true", help="Save full thinking text in predictions.jsonl.")
    parser.add_argument("--thinking-preview-chars", type=int, default=2000)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "qwen3_zero_shot_coarse"))
    parser.add_argument("--dry-run", action="store_true", help="Print one prompt and gold graph without loading Qwen3.")
    return parser.parse_args()


def compact_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    if max_chars <= 0 or len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def event_trigger(event) -> str:
    return str(event.metadata.get("trigger", "")).strip()


def event_mention(event) -> str:
    return str(event.metadata.get("event_context", event.text)).strip()


def build_graph_prompt(sample: DocumentGraphSample, max_document_chars: int) -> str:
    document = sample.documents[0]
    event_lines: list[str] = []
    for event in sample.events:
        evidence = event.evidence[0].text if event.evidence else ""
        event_lines.append(
            " | ".join(
                [
                    f"event_id={event.event_id}",
                    f"sentence_index={event.sentence_index}",
                    f"trigger={event_trigger(event)}",
                    f"mention={event_mention(event)}",
                    f"evidence={compact_text(evidence, 300)}",
                ]
            )
        )

    return (
        "Construct a directed coarse event relation graph from the document and the provided event mentions.\n"
        "The event mentions are authoritative: do not add, delete, merge, or rename event IDs.\n"
        "Predict only relations supported by the document. Omit unrelated event pairs.\n"
        "Allowed relation types:\n"
        "- precedes: the source temporally precedes, enables, or is a precondition for the target.\n"
        "- causes: the source explicitly or strongly causes the target.\n"
        "Direction always runs from source_event_id to target_event_id.\n"
        "Confidence must be between 0 and 1.\n"
        "After thinking, return only one valid JSON object in the final answer, without Markdown fences or prose:\n"
        "{\n"
        '  "edges": [\n'
        "    {\n"
        '      "source_event_id": "maven_e0",\n'
        '      "target_event_id": "maven_e1",\n'
        '      "relation_type": "precedes",\n'
        '      "confidence": 0.75,\n'
        '      "evidence": "short supporting text"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Document title: {document.title}\n"
        f"Document text:\n{compact_text(document.text, max_document_chars)}\n\n"
        "Event mentions:\n"
        + "\n".join(f"- {line}" for line in event_lines)
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
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


def parse_predicted_edges(
    answer: str,
    valid_event_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    payload = extract_json_object(answer)
    if payload is None:
        return [], ["final answer did not contain a valid JSON object"], False
    raw_edges = payload.get("edges", [])
    if not isinstance(raw_edges, list):
        return [], ["JSON field 'edges' was not an array"], False

    parsed: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for index, item in enumerate(raw_edges):
        if not isinstance(item, dict):
            warnings.append(f"edge[{index}] was not an object")
            continue
        source = str(item.get("source_event_id", "")).strip()
        target = str(item.get("target_event_id", "")).strip()
        relation = str(item.get("relation_type", "")).strip().lower()
        if source not in valid_event_ids or target not in valid_event_ids:
            warnings.append(f"edge[{index}] referenced an unknown event id: {source}->{target}")
            continue
        if source == target:
            warnings.append(f"edge[{index}] was a self-loop: {source}")
            continue
        if relation not in ALLOWED_RELATIONS:
            warnings.append(f"edge[{index}] used unsupported relation_type={relation!r}")
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
            warnings.append(f"edge[{index}] had non-numeric confidence")
        key = (source, target, relation)
        if key in seen:
            warnings.append(f"edge[{index}] duplicated {source} --{relation}--> {target}")
            continue
        seen.add(key)
        parsed.append(
            {
                "source_event_id": source,
                "target_event_id": target,
                "relation_type": relation,
                "confidence": round(max(0.0, min(confidence, 1.0)), 4),
                "evidence": str(item.get("evidence", "")).strip(),
            }
        )
    return parsed, warnings, True


def gold_edge_in_scope(edge, gold_scope: str) -> bool:
    if edge.relation_type not in ALLOWED_RELATIONS:
        return False
    if gold_scope == "all":
        return True
    source_relation = str(edge.metadata.get("source_relation", "")).upper()
    return source_relation in {"CAUSE", "PRECONDITION"}


def prf_counts(predicted: set[tuple], gold: set[tuple]) -> dict[str, int | float]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_edges(
    sample: DocumentGraphSample,
    predicted_edges: list[dict[str, Any]],
    gold_scope: str = "causal",
) -> dict[str, Any]:
    gold_graph = sample.gold_graph
    if gold_graph is None:
        raise ValueError(f"Sample {sample.sample_id} does not contain a gold graph.")
    gold_typed = {
        (edge.source_event_id, edge.target_event_id, edge.relation_type)
        for edge in gold_graph.edges
        if gold_edge_in_scope(edge, gold_scope)
    }
    pred_typed = {
        (edge["source_event_id"], edge["target_event_id"], edge["relation_type"])
        for edge in predicted_edges
    }
    gold_pairs = {(source, target) for source, target, _ in gold_typed}
    pred_pairs = {(source, target) for source, target, _ in pred_typed}
    typed = prf_counts(pred_typed, gold_typed)
    pair = prf_counts(pred_pairs, gold_pairs)
    universe = len(sample.events) * max(len(sample.events) - 1, 0)
    pair_tn = max(universe - int(pair["tp"]) - int(pair["fp"]) - int(pair["fn"]), 0)
    pair_accuracy = (int(pair["tp"]) + pair_tn) / universe if universe else 0.0

    per_relation = {}
    for relation in ALLOWED_RELATIONS:
        per_relation[relation] = prf_counts(
            {item for item in pred_typed if item[2] == relation},
            {item for item in gold_typed if item[2] == relation},
        )
    exact_pairs = {(source, target) for source, target, _ in pred_typed & gold_typed}
    wrong_type_pairs = (pred_pairs & gold_pairs) - exact_pairs
    return {
        "typed": typed,
        "pair": {**pair, "tn": pair_tn, "accuracy": pair_accuracy},
        "per_relation": per_relation,
        "wrong_type_pair_count": len(wrong_type_pairs),
        "gold_scope": gold_scope,
        "gold_typed_edges": [list(item) for item in sorted(gold_typed)],
        "predicted_typed_edges": [list(item) for item in sorted(pred_typed)],
    }


def edge_text(edge: tuple[str, str, str]) -> str:
    source, target, relation = edge
    return f"{source} --{relation}--> {target}"


def render_readable_sample(
    sample: DocumentGraphSample,
    predicted_edges: list[dict[str, Any]],
    metrics: dict[str, Any],
    warnings: list[str],
    final_answer: str,
    thinking_preview: str = "",
) -> str:
    gold_typed = {tuple(item) for item in metrics["gold_typed_edges"]}
    pred_typed = {tuple(item) for item in metrics["predicted_typed_edges"]}
    gold_by_pair: dict[tuple[str, str], set[str]] = {}
    pred_by_pair: dict[tuple[str, str], set[str]] = {}
    for source, target, relation in gold_typed:
        gold_by_pair.setdefault((source, target), set()).add(relation)
    for source, target, relation in pred_typed:
        pred_by_pair.setdefault((source, target), set()).add(relation)
    confidence_lookup = {
        (edge["source_event_id"], edge["target_event_id"], edge["relation_type"]): edge["confidence"]
        for edge in predicted_edges
    }

    lines = [
        "=" * 100,
        f"SAMPLE {sample.sample_id}",
        f"title: {sample.documents[0].title}",
        f"events={len(sample.events)} | gold_edges={len(gold_typed)} | predicted_edges={len(pred_typed)}",
        "",
        "EVENTS",
    ]
    for event in sample.events:
        lines.append(
            f"[{event.event_id}] sent={event.sentence_index} | trigger={event_trigger(event)} | mention={event_mention(event)}"
        )

    lines.extend(["", "PREDICTED EDGE REVIEW"])
    if not pred_typed:
        lines.append("(no valid predicted edges)")
    for edge in sorted(pred_typed):
        source, target, relation = edge
        if edge in gold_typed:
            tag = "MATCH"
        elif (source, target) in gold_by_pair:
            expected = ",".join(sorted(gold_by_pair[(source, target)]))
            tag = f"MIS-TYPE expected={expected}"
        else:
            tag = "EXTRA"
        lines.append(f"[{tag}] {edge_text(edge)} | confidence={confidence_lookup.get(edge, 0.0):.3f}")

    lines.extend(["", "MISSED GOLD EDGES"])
    missing_count = 0
    for edge in sorted(gold_typed - pred_typed):
        source, target, relation = edge
        if (source, target) in pred_by_pair:
            predicted = ",".join(sorted(pred_by_pair[(source, target)]))
            lines.append(f"[MIS-TYPE] {edge_text(edge)} | predicted={predicted}")
        else:
            lines.append(f"[MISS] {edge_text(edge)}")
        missing_count += 1
    if missing_count == 0:
        lines.append("(none)")

    typed = metrics["typed"]
    pair = metrics["pair"]
    lines.extend(
        [
            "",
            "METRICS",
            (
                f"typed_edge: precision={typed['precision']:.4f} recall={typed['recall']:.4f} "
                f"f1={typed['f1']:.4f} tp={typed['tp']} fp={typed['fp']} fn={typed['fn']}"
            ),
            (
                f"edge_pair:  precision={pair['precision']:.4f} recall={pair['recall']:.4f} "
                f"f1={pair['f1']:.4f} accuracy={pair['accuracy']:.4f}"
            ),
        ]
    )
    if warnings:
        lines.extend(["", "PARSE WARNINGS", *[f"- {warning}" for warning in warnings]])
    if thinking_preview:
        lines.extend(["", "THINKING PREVIEW", thinking_preview])
    lines.extend(["", "FINAL ANSWER", final_answer.strip() or "(empty)"])
    return "\n".join(lines)


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate_section(section: str) -> dict[str, float | int]:
        tp = sum(int(row["metrics"][section]["tp"]) for row in rows)
        fp = sum(int(row["metrics"][section]["fp"]) for row in rows)
        fn = sum(int(row["metrics"][section]["fn"]) for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    pair = aggregate_section("pair")
    pair_tn = sum(int(row["metrics"]["pair"]["tn"]) for row in rows)
    pair_total = int(pair["tp"]) + int(pair["fp"]) + int(pair["fn"]) + pair_tn
    pair["tn"] = pair_tn
    pair["accuracy"] = (int(pair["tp"]) + pair_tn) / pair_total if pair_total else 0.0
    per_relation = {}
    for relation in ALLOWED_RELATIONS:
        tp = sum(int(row["metrics"]["per_relation"][relation]["tp"]) for row in rows)
        fp = sum(int(row["metrics"]["per_relation"][relation]["fp"]) for row in rows)
        fn = sum(int(row["metrics"]["per_relation"][relation]["fn"]) for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_relation[relation] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return {
        "samples": len(rows),
        "json_parse_rate": sum(bool(row["parsed_json"]) for row in rows) / len(rows) if rows else 0.0,
        "typed": aggregate_section("typed"),
        "pair": pair,
        "per_relation": per_relation,
        "average_gold_edges": sum(len(row["metrics"]["gold_typed_edges"]) for row in rows) / len(rows) if rows else 0.0,
        "average_predicted_edges": sum(len(row["metrics"]["predicted_typed_edges"]) for row in rows) / len(rows) if rows else 0.0,
        "average_generated_tokens": sum(int(row["generated_tokens"]) for row in rows) / len(rows) if rows else 0.0,
    }


def select_samples(args: argparse.Namespace) -> list[DocumentGraphSample]:
    samples = load_maven_document_graph_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=None,
        max_events=args.max_events,
    )
    samples = [
        sample
        for sample in samples
        if sample.gold_graph is not None
        and any(gold_edge_in_scope(edge, args.gold_scope) for edge in sample.gold_graph.edges)
    ]
    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
        selected = samples[: args.limit]
    else:
        selected = samples[args.sample_offset : args.sample_offset + args.limit]
    if not selected:
        raise RuntimeError(
            f"No MAVEN graph samples found for split={args.split!r}, offset={args.sample_offset}, limit={args.limit}."
        )
    return selected


def main() -> None:
    args = parse_args()
    samples = select_samples(args)
    prompts = [build_graph_prompt(sample, max_document_chars=args.max_document_chars) for sample in samples]

    if args.dry_run:
        sample = samples[0]
        print(prompts[0])
        print("\n" + "=" * 100)
        print("GOLD EDGES")
        for edge in sample.gold_graph.edges if sample.gold_graph is not None else []:
            if gold_edge_in_scope(edge, args.gold_scope):
                print(f"- {edge.source_event_id} --{edge.relation_type}--> {edge.target_event_id}")
        return

    print(f"loading Qwen3 Thinking model from {args.model_path} ...", flush=True)
    chat = Qwen3ThinkingChat(
        model_path=args.model_path,
        allow_download=args.allow_download,
        device=args.device,
    )
    print(
        f"loaded | source={chat.model_source} | device={chat.device_summary} "
        f"| samples={len(samples)}",
        flush=True,
    )

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    readable_path = output_dir / "readable.log"
    metrics_path = output_dir / "metrics.json"
    predictions_path.write_text("", encoding="utf-8")
    readable_path.write_text("", encoding="utf-8")

    started = time.time()
    rows: list[dict[str, Any]] = []
    system_prompt = (
        "You are an event graph analyst. Reason carefully about temporal and causal relations. "
        "Your final answer must follow the requested JSON schema exactly."
    )
    for index, (sample, prompt) in enumerate(zip(samples, prompts), start=1):
        response = chat.generate(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed + index - 1,
        )
        valid_event_ids = {event.event_id for event in sample.events}
        predicted_edges, warnings, parsed_json = parse_predicted_edges(response.answer, valid_event_ids)
        metrics = evaluate_edges(sample, predicted_edges, gold_scope=args.gold_scope)
        thinking_preview = ""
        if args.show_thinking:
            thinking_preview = compact_text(response.thinking, args.thinking_preview_chars)
        readable = render_readable_sample(
            sample=sample,
            predicted_edges=predicted_edges,
            metrics=metrics,
            warnings=warnings,
            final_answer=response.answer,
            thinking_preview=thinking_preview,
        )
        print(readable, flush=True)
        with readable_path.open("a", encoding="utf-8") as handle:
            handle.write(readable + "\n\n")

        row = {
            "sample_id": sample.sample_id,
            "title": sample.documents[0].title,
            "events": [event.to_dict() for event in sample.events],
            "gold_edges": [edge.to_dict() for edge in sample.gold_graph.edges if gold_edge_in_scope(edge, args.gold_scope)] if sample.gold_graph is not None else [],
            "predicted_edges": predicted_edges,
            "parsed_json": parsed_json,
            "parse_warnings": warnings,
            "metrics": metrics,
            "final_answer": response.answer,
            "generated_tokens": response.generated_tokens,
        }
        if args.save_thinking:
            row["thinking"] = response.thinking
        rows.append(row)
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"progress {index}/{len(samples)} | typed_f1={metrics['typed']['f1']:.4f} "
            f"| pair_f1={metrics['pair']['f1']:.4f} | tokens={response.generated_tokens}",
            flush=True,
        )

    aggregate = aggregate_metrics(rows)
    output = {
        "config": {
            **vars(args),
            "dataset": str(resolve_repo_path(args.dataset)),
            "model_source": chat.model_source,
            "device": chat.device_summary,
            "allowed_relations": list(ALLOWED_RELATIONS),
            "gold_mapping_note": "CAUSE -> causes; current project maps MAVEN temporal/precondition relations to precedes.",
        },
        **aggregate,
        "elapsed_seconds": round(time.time() - started, 3),
        "outputs": {
            "predictions": str(predictions_path),
            "readable": str(readable_path),
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"saved | typed_f1={output['typed']['f1']:.4f} | pair_f1={output['pair']['f1']:.4f} "
        f"| pair_accuracy={output['pair']['accuracy']:.4f} | path={metrics_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, Qwen3ThinkingUnavailable, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
