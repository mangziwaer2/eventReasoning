from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causal_graph import CoarseCausalGraph
from causal_graph import EventNode
from causal_graph import NewsDocument
from causal_graph import QuerySpec


FORECAST_TRACE_SYSTEM_PROMPT = "You output a grounded forecast_trace followed by a final_answer JSON only."


@dataclass(slots=True)
class ForecastPromptBundle:
    prompt: str
    event_ref_to_id: dict[str, str]
    edge_ref_to_id: dict[str, str]
    choices: list[dict[str, Any]]


def compact_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def append_no_think(prompt: str) -> str:
    text = str(prompt).rstrip()
    if text.endswith("/no_think"):
        return text
    return f"{text}\n/no_think"


def _event_trigger(event: EventNode) -> str:
    trigger = str(event.metadata.get("trigger", "")).strip()
    if trigger:
        return trigger
    text = event.text.replace("trigger=", "")
    return text.split(";", 1)[0].strip()[:48]


def _render_documents(documents: list[NewsDocument], max_document_chars: int) -> str:
    lines: list[str] = []
    for document in documents:
        lines.append(
            "\n".join(
                [
                    f"[Document {document.document_id}]",
                    f"Date: {document.publish_time or '-'}",
                    f"Title: {compact_text(document.title, 180)}",
                    f"Text: {compact_text(document.text, max_document_chars)}",
                ]
            )
        )
    return "\n\n".join(lines) if lines else "- none"


def _render_events(graph: CoarseCausalGraph, max_events: int) -> tuple[str, dict[str, str], dict[str, str]]:
    event_ref_to_id: dict[str, str] = {}
    id_to_event_ref: dict[str, str] = {}
    lines: list[str] = []
    for index, event in enumerate(graph.events[:max_events], start=1):
        event_ref = f"H{index:02d}"
        event_ref_to_id[event_ref] = event.event_id
        id_to_event_ref[event.event_id] = event_ref
        participants = ", ".join(event.participants) if event.participants else "-"
        lines.append(
            " | ".join(
                [
                    event_ref,
                    f"event_id={event.event_id}",
                    f"doc={event.document_id}",
                    f"sent={event.sentence_index}",
                    f"trigger={_event_trigger(event)}",
                    f"event={event.text}",
                    f"participants={participants}",
                ]
            )
        )
    return ("\n".join(f"- {line}" for line in lines) if lines else "- none", event_ref_to_id, id_to_event_ref)


def _render_edges(
    graph: CoarseCausalGraph,
    max_edges: int,
    id_to_event_ref: dict[str, str],
) -> tuple[str, dict[str, str]]:
    edge_ref_to_id: dict[str, str] = {}
    lines: list[str] = []
    visible_edges = sorted(graph.edges, key=lambda item: float(item.score), reverse=True)[:max_edges]
    for index, edge in enumerate(visible_edges, start=1):
        edge_ref = f"R{index:02d}"
        edge_ref_to_id[edge_ref] = edge.edge_id
        source_ref = id_to_event_ref.get(edge.source_event_id, edge.source_event_id)
        target_ref = id_to_event_ref.get(edge.target_event_id, edge.target_event_id)
        lines.append(
            " | ".join(
                [
                    edge_ref,
                    f"edge_id={edge.edge_id}",
                    f"{source_ref} -> {target_ref}",
                    f"relation={edge.relation_type}",
                    f"confidence={float(edge.score):.3f}",
                    f"source_event_id={edge.source_event_id}",
                    f"target_event_id={edge.target_event_id}",
                ]
            )
        )
    return "\n".join(f"- {line}" for line in lines) if lines else "- none", edge_ref_to_id


def build_structured_forecast_prompt(
    query: QuerySpec,
    documents: list[NewsDocument],
    refined_graph: CoarseCausalGraph,
    choices: list[dict[str, Any]] | None = None,
    max_graph_events: int = 24,
    max_graph_edges: int = 48,
    max_document_chars: int = 700,
) -> ForecastPromptBundle:
    # The no-refinement research path deliberately does not expose the global
    # event-code inventory. The finite answer space is learned from labels and
    # enforced by decoding/reward outside the prompt.
    del choices
    normalized_choices: list[dict[str, Any]] = []
    event_block, event_ref_to_id, id_to_event_ref = _render_events(refined_graph, max_graph_events)
    edge_block, edge_ref_to_id = _render_edges(refined_graph, max_graph_edges, id_to_event_ref)
    prompt = (
        "You are a future event forecasting model.\n"
        "Input includes query, cutoff-before documents, observed events, and a coarse causal graph.\n"
        "First output a structured forecast_trace, then output a final_answer with one event_code.\n"
        "No candidate choices are provided. Do not output choice_id or copy a candidate list.\n"
        "The event_code is the closed-set target learned during training; output the valid code directly.\n"
        "Use only visible historical events and coarse-graph edges as support. Do not invent historical support.\n"
        "Intermediate trace events may be new future hypotheses before the target time, but their support must point to visible events/edges.\n"
        "Keep the trace compact, concrete, and grounded; prefer a few well-supported steps over verbose speculation.\n"
        "The final trace event should explain why the final event_code is likely.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "forecast_trace": {\n'
        '    "intermediate_events": [\n'
        "      {\n"
        '        "trace_event_id": "ft_1",\n'
        '        "event": {"trigger": "deploy", "mention": "security forces deploy near the capital", "actors": ["security forces"], "relative_time": "t-1"},\n'
        '        "supporting_events": [{"event_ref": "H01", "event": "copy a visible historical event"}],\n'
        '        "supporting_edge_refs": ["R01"],\n'
        '        "expected_effect": "why this raises or lowers a candidate outcome",\n'
        '        "confidence": 0.0\n'
        "      }\n"
        "    ],\n"
        '    "trace_edges": [\n'
        '      {"source_ref": "H01", "target_ref": "ft_1", "relation_type": "causes", "confidence": 0.0},\n'
        '      {"source_ref": "ft_1", "target_ref": "answer_173", "relation_type": "raises_likelihood", "confidence": 0.0}\n'
        "    ]\n"
        "  },\n"
        '  "final_answer": {"event_code": "000", "event": "event description", "confidence": 0.0}\n'
        "}\n\n"
        "Invalid outputs: nonexistent event_ref/edge_ref, generic events like 'tensions rise', invalid event_code, or cutoff-after facts as observed history.\n\n"
        f"QueryId: {query.query_id}\n"
        f"Query: {query.text}\n"
        f"Target/Cutoff date: {query.cutoff_time or '-'}\n"
        f"Focus actors: {', '.join(query.focus_entities) or '-'}\n\n"
        "Documents:\n"
        f"{_render_documents(documents, max_document_chars)}\n\n"
        "Visible historical events:\n"
        f"{event_block}\n\n"
        "Coarse causal edges:\n"
        f"{edge_block}\n"
    )
    return ForecastPromptBundle(
        prompt=prompt,
        event_ref_to_id=event_ref_to_id,
        edge_ref_to_id=edge_ref_to_id,
        choices=normalized_choices,
    )
