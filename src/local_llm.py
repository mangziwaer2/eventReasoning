from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal_graph import ForecastCandidate
from causal_graph import ForecastResult
from causal_graph import LocalCausalGraph


class LocalGenerationUnavailable(RuntimeError):
    pass


def _import_transformers() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise LocalGenerationUnavailable(
            "Local generation requires `torch` and `transformers`. "
            "Install them before running the local model validation path."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def render_graph_context(graph: LocalCausalGraph, max_events: int = 8, max_edges: int = 10) -> str:
    event_lines = []
    for event in graph.events[:max_events]:
        participants = ", ".join(event.participants) if event.participants else "-"
        event_lines.append(
            f"- {event.event_id} [{event.node_type}] {event.text} | participants={participants} | conf={event.confidence:.2f}"
        )

    edge_lines = []
    for edge in graph.edges[:max_edges]:
        edge_lines.append(
            f"- {edge.edge_id}: {edge.source_event_id} --{edge.relation_type}/{edge.confidence:.2f}--> {edge.target_event_id}"
        )

    return "\n".join(
        [
            "Query:",
            graph.query.text,
            "",
            "Retrieved evidence documents:",
            *[
                f"- {doc.document_id} | {doc.publish_time or '-'} | {doc.title}"
                for doc in graph.documents
            ],
            "",
            "Local events:",
            *event_lines,
            "",
            "Local edges:",
            *edge_lines,
        ]
    ).strip()


def build_forecast_prompt(graph: LocalCausalGraph) -> str:
    graph_context = render_graph_context(graph)
    return (
        "You are helping with event forecasting from a query-conditioned local causal graph.\n"
        "Use only the provided graph and evidence summary.\n"
        "If the graph is too weak, abstain.\n\n"
        f"{graph_context}\n\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "abstain": true or false,\n'
        '  "forecast_event": "one short future event hypothesis",\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "rationale": "brief explanation grounded in the graph",\n'
        '  "support_event_ids": ["e1", "e2"]\n'
        "}\n"
    )


def parse_forecast_response(query_id: str, prompt: str, raw_response: str, gold: dict[str, Any]) -> ForecastResult:
    response_text = raw_response.strip()
    payload: dict[str, Any] | None = None

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(response_text[start : end + 1])
            except json.JSONDecodeError:
                payload = None

    candidates: list[ForecastCandidate] = []
    metadata: dict[str, Any] = {}
    if payload is None:
        candidates.append(ForecastCandidate(text=response_text, confidence=0.0))
        metadata["parsed_json"] = False
    else:
        metadata["parsed_json"] = True
        metadata["abstain"] = bool(payload.get("abstain", False))
        forecast_event = str(payload.get("forecast_event", "")).strip()
        confidence = payload.get("confidence", 0.0)
        rationale = str(payload.get("rationale", "")).strip()
        support_event_ids = payload.get("support_event_ids", [])
        if not isinstance(support_event_ids, list):
            support_event_ids = []
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        if forecast_event:
            candidates.append(
                ForecastCandidate(
                    text=forecast_event,
                    confidence=max(0.0, min(confidence_value, 1.0)),
                    rationale=rationale,
                    support_event_ids=[str(item) for item in support_event_ids],
                )
            )

    return ForecastResult(
        query_id=query_id,
        prompt=prompt,
        raw_response=raw_response,
        candidates=candidates,
        gold=gold,
        metadata=metadata,
    )


class LocalQwenGenerator:
    def __init__(self, model_path: Path, max_new_tokens: int = 160) -> None:
        torch, auto_model_cls, auto_tokenizer_cls = _import_transformers()
        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = auto_tokenizer_cls.from_pretrained(model_path, trust_remote_code=False)
        self.model = auto_model_cls.from_pretrained(model_path, trust_remote_code=False)

        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to("cuda")
        else:
            self.device = "cpu"
        self.model.eval()

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": "You forecast events from a local causal graph and must follow JSON output exactly."},
                {"role": "user", "content": prompt},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        input_ids = input_ids.to(self.device)
        attention_mask = self._torch.ones_like(input_ids)

        with self._torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
