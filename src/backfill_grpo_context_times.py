from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EVENT_LINE = re.compile(r"(^\s*-\s*H\d+\s*\|\s*)(.*)$", re.MULTILINE)


def _event_time(event: dict[str, Any]) -> str:
    metadata = event.get("metadata", {})
    value = event.get("event_time") or event.get("date") or event.get("publish_time")
    if not value and isinstance(metadata, dict):
        value = metadata.get("event_time") or metadata.get("date") or metadata.get("publish_time")
    return str(value or "-").strip()


def _load_event_times(paths: list[Path]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, str]] = {}
    metadata_by_query: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = str(row.get("query_id", "")).strip()
                if not query_id:
                    continue
                result[query_id] = {
                    str(event.get("event_id", "")).strip(): _event_time(event)
                    for event in row.get("events", [])
                    if isinstance(event, dict) and str(event.get("event_id", "")).strip()
                }
                metadata_by_query[query_id] = dict(row.get("metadata", {})) if isinstance(row.get("metadata"), dict) else {}
    return result, metadata_by_query


def _backfill_prompt(prompt: str, ref_to_id: dict[str, str], event_times: dict[str, str]) -> tuple[str, int]:
    added = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal added
        prefix, rest = match.group(1), match.group(2)
        ref_match = re.match(r"\s*-\s*(H\d+)\s*\|", prefix + rest)
        if "time=" in rest or ref_match is None:
            return match.group(0)
        ref = ref_match.group(1)
        event_id = ref_to_id.get(ref, "")
        timestamp = event_times.get(event_id, "-")
        added += int(timestamp != "-")
        return prefix + f"time={timestamp} | " + rest

    updated = EVENT_LINE.sub(replace, str(prompt))
    updated = updated.replace(
        '"event": {"trigger": "deploy", "mention": "security forces deploy near the capital", "actors": ["security forces"], "relative_time": "t-1"}',
        '"event": {"trigger": "deploy", "mention": "security forces deploy near the capital", "actors": ["security forces"], "event_time": "YYYY-MM-DD", "relative_time": "t-1"}',
    )
    updated = updated.replace(
        "In this MIRAI task, relative_time is measured from the cutoff/observation date: use t+1 for the next day, t+2 for two days later, and so on. You may also provide the equivalent absolute event_time (YYYY-MM-DD).",
        "In this MIRAI task, relative_time is measured from the target answer date: use t-1 for the day before the answer, t-2 for two days before, and t for the answer date. Prefer the equivalent absolute event_time (YYYY-MM-DD).",
    )
    updated = updated.replace(
        "Intermediate trace events may be new future hypotheses before the target time, but their support must point to visible events/edges.",
        "Intermediate forecast events must be after the cutoff/observation date and before the answer event when an answer-time estimate is available; their support must point to visible events/edges.",
    )
    anchor_line = (
        "In this MIRAI task, relative_time is measured from the target answer date: use t-1 for the day before the answer, "
        "t-2 for two days before, and t for the answer date. Prefer the equivalent absolute event_time (YYYY-MM-DD)."
    )
    if anchor_line not in updated:
        marker = "Intermediate forecast events must be after the cutoff/observation date and before the answer event when an answer-time estimate is available; their support must point to visible events/edges."
        if marker in updated:
            updated = updated.replace(marker, marker + "\n" + anchor_line)
    updated = updated.replace(
        '"expected_effect": "why this raises or lowers a candidate outcome"',
        '"expected_effect": "specific mechanism: how this event changes the likelihood of the answer"',
    )
    return updated, added


def _add_target_line(prompt: str, target_time: str) -> str:
    if not target_time or "Target answer date:" in prompt:
        return prompt
    marker = re.search(r"(?m)^Target/Cutoff date:.*$", prompt)
    if not marker:
        return prompt
    return prompt[: marker.end()] + f"\nTarget answer date: {target_time}" + prompt[marker.end() :]


def backfill(
    rows: list[dict[str, Any]],
    event_times: dict[str, dict[str, str]],
    input_metadata: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {"rows": 0, "rows_with_event_times": 0, "prompt_events_added": 0, "graph_events_updated": 0}
    output: list[dict[str, Any]] = []
    for row in rows:
        stats["rows"] += 1
        query_id = str(row.get("query_id", "")).strip()
        times = event_times.get(query_id, {})
        source_metadata = input_metadata.get(query_id, {})
        trajectory = row.get("trajectory", {})
        metadata = trajectory.get("metadata", {}) if isinstance(trajectory, dict) else {}
        ref_to_id = metadata.get("event_ref_to_id", {}) if isinstance(metadata, dict) else {}
        ref_to_id = ref_to_id if isinstance(ref_to_id, dict) else {}
        graph = metadata.get("refined_graph", {}) if isinstance(metadata, dict) else {}
        graph = graph if isinstance(graph, dict) else {}
        graph_events = graph.get("events", [])
        for event in graph_events if isinstance(graph_events, list) else []:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id", "")).strip()
            if event_id in times and times[event_id] != "-":
                event["event_time"] = times[event_id]
                stats["graph_events_updated"] += 1
        if times:
            stats["rows_with_event_times"] += 1
        prompt, added = _backfill_prompt(row.get("forecast_prompt", ""), ref_to_id, times)
        stats["prompt_events_added"] += added
        row["forecast_prompt"] = prompt
        row["temporal_anchor"] = "observation_to_target"
        mirai_query = row.get("mirai_query")
        if not isinstance(mirai_query, dict):
            mirai_query = {}
            row["mirai_query"] = mirai_query
        target_events = source_metadata.get("target_events", []) if isinstance(source_metadata, dict) else []
        if isinstance(target_events, list) and target_events:
            mirai_query["target_events"] = target_events
            dates = sorted(
                str(item.get("date", "")).strip()
                for item in target_events
                if isinstance(item, dict) and str(item.get("date", "")).strip()
            )
            if dates:
                mirai_query["target_time"] = dates[0]
        for key in ("target_label_start", "target_label_end", "target_horizon_days"):
            if key in source_metadata:
                mirai_query[key] = source_metadata[key]
        prompt = _add_target_line(prompt, str(mirai_query.get("target_time", "")).strip())
        row["forecast_prompt"] = prompt
        if isinstance(metadata, dict):
            metadata["temporal_anchor"] = "observation_to_target"
            metadata["refined_graph"] = graph
            query = metadata.get("query", {})
            if isinstance(query, dict):
                query["observation_time"] = str(mirai_query.get("date_str", "")).strip()
                if mirai_query.get("target_time"):
                    query["target_time"] = mirai_query["target_time"]
        for step in trajectory.get("steps", []) if isinstance(trajectory, dict) else []:
            if isinstance(step, dict) and step.get("name") == "forecast" and isinstance(step.get("metadata"), dict):
                step["metadata"]["temporal_anchor"] = "observation_to_target"
                step["metadata"]["refined_graph"] = graph
        output.append(row)
    return output, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill event times into an existing prompt-only GRPO context JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Existing grpo_context.jsonl")
    parser.add_argument("--event-input", type=Path, nargs="+", required=True, help="Existing event-input JSONL files or directories")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL path; input is never overwritten")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_paths: list[Path] = []
    for path in args.event_input:
        event_paths.extend(sorted(path.glob("*.jsonl")) if path.is_dir() else [path])
    event_times, input_metadata = _load_event_times(event_paths)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    output, stats = backfill(rows, event_times, input_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input": str(args.input), "output": str(args.output), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
