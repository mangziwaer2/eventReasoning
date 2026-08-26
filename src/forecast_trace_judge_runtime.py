from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from forecast_trace_judge import (
    JUDGE_SYSTEM_PROMPT,
    build_description_judge_prompt,
    build_judge_prompt,
)
from forecast_trace_schema import extract_first_json_object
from local_qwen_lora import load_qwen_for_inference
from path_utils import resolve_repo_path


DESCRIPTION_JUDGE_SYSTEM_PROMPT = (
    "You evaluate semantic equivalence between a CAMEO event code's canonical description and "
    "a model-generated description. Do not require exact wording. Do not judge whether the code "
    "is correct for a query. Return JSON only."
)


def normalize_context_prompt(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                payload = json.loads(text)
                if isinstance(payload, list):
                    parts = [
                        str(item.get("content", ""))
                        for item in payload
                        if isinstance(item, dict) and item.get("content")
                    ]
                    if parts:
                        return "\n\n".join(parts)
            except json.JSONDecodeError:
                pass
        return text
    return str(value or "").strip()


def graph_from_forecast_context(context: str) -> dict[str, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for line in str(context).splitlines():
        event = re.match(r"\s*-\s*(H\d+)\s*\|.*?mention=(.*?)\s*\|\s*participants=", line)
        if event:
            events.append({"event_id": event.group(1), "text": event.group(2).strip()})
            continue
        edge = re.match(
            r"\s*-\s*(R\d+)\s*\|.*?\s(H\d+)\s*->\s*([^| ]+)\s*\|\s*relation=([^|]+)\|\s*confidence=([0-9.]+)",
            line,
        )
        if edge:
            edges.append(
                {
                    "edge_id": edge.group(1),
                    "source_event_id": edge.group(2),
                    "target_event_id": edge.group(3),
                    "relation_type": edge.group(4).strip(),
                    "score": float(edge.group(5)),
                }
            )
    return {"events": events, "edges": edges}


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def parse_judge_response_robust(raw_response: str) -> dict[str, Any]:
    text = str(raw_response or "").strip()
    payload = extract_first_json_object(text) or {}
    keys = ("support", "causal", "temporal", "answer_link", "hallucination", "overall")
    scores = {key: _clamp(payload.get(key, 0.0)) for key in keys}
    partial = not bool(payload)
    found_score = False
    if partial:
        for key in keys:
            match = re.search(rf"[\"']{re.escape(key)}[\"']\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))", text)
            if match:
                scores[key] = _clamp(match.group(1))
                found_score = True
    if "overall" not in payload and not found_score and not any(scores.values()):
        scores["overall"] = (
            0.3 * scores["support"]
            + 0.25 * scores["causal"]
            + 0.2 * scores["temporal"]
            + 0.25 * scores["answer_link"]
        ) * (1.0 - scores["hallucination"])
    elif "overall" not in payload and not found_score:
        scores["overall"] = (
            0.3 * scores["support"]
            + 0.25 * scores["causal"]
            + 0.2 * scores["temporal"]
            + 0.25 * scores["answer_link"]
        ) * (1.0 - scores["hallucination"])
    return {
        "parsed_json": bool(payload) or found_score,
        "partial_json": partial and extract_first_json_object(text) is None,
        **scores,
        "reason": str(payload.get("reason", "")).strip(),
        "raw_response": text,
    }


def parse_description_judge_response(raw_response: str) -> dict[str, Any]:
    text = str(raw_response or "").strip()
    payload = extract_first_json_object(text) or {}
    match = re.search(r"[\"']match[\"']\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))", text)
    parsed = bool(payload) or match is not None
    score = _clamp(payload.get("match", match.group(1) if match else 0.0))
    return {
        "parsed_json": parsed,
        "partial_json": not bool(payload) and match is not None,
        "match": score,
        "reason": str(payload.get("reason", "")).strip(),
        "raw_response": text,
    }


class FrozenQwenTraceJudge:
    """One frozen Qwen runtime for trace and semantic code-description rewards."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_new_tokens: int = 384,
        description_max_new_tokens: int = 96,
        thinking: bool = False,
        cache_path: str | Path | None = None,
        max_context_chars: int = 12000,
    ) -> None:
        self.model_path = resolve_repo_path(str(model_path))
        self.max_new_tokens = max(64, int(max_new_tokens))
        self.description_max_new_tokens = max(32, int(description_max_new_tokens))
        self.thinking = bool(thinking)
        self.max_context_chars = max(1000, int(max_context_chars))
        self.cache_path = resolve_repo_path(str(cache_path)) if cache_path else None
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.cache = {str(key): value for key, value in loaded.items() if isinstance(value, dict)}
            except (OSError, json.JSONDecodeError):
                self.cache = {}
        self.model = None
        self.tokenizer = None
        self.torch = None

    def _load(self) -> None:
        if self.model is not None:
            return
        self.model, self.tokenizer, self.torch = load_qwen_for_inference(self.model_path)
        self.model.eval()

    def _encode(self, prompt: str, system_prompt: str):
        self._load()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"] if isinstance(encoded, dict) else encoded
        else:
            text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            encoded = self.tokenizer(text, return_tensors="pt")
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if not hasattr(input_ids, "to"):
            input_ids = self.torch.tensor(input_ids, dtype=self.torch.long)
        return input_ids.to(next(self.model.parameters()).device)

    def _generate(self, prompt: str, system_prompt: str, max_new_tokens: int) -> str:
        input_ids = self._encode(prompt, system_prompt)
        attention_mask = self.torch.ones_like(input_ids)
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()

    def _cached_generate(self, key_prefix: str, prompt: str, system_prompt: str, max_new_tokens: int) -> dict[str, Any] | None:
        key = hashlib.sha256((key_prefix + "\n" + prompt).encode("utf-8")).hexdigest()
        if key in self.cache:
            return {**self.cache[key], "cached": True}
        try:
            raw = self._generate(
                prompt if self.thinking else prompt + "\n/no_think",
                system_prompt,
                max_new_tokens,
            )
            result = parse_judge_response_robust(raw) if key_prefix == "trace" else parse_description_judge_response(raw)
        except Exception as exc:
            result = {
                "parsed_json": False,
                "partial_json": False,
                "match": 0.0,
                "reason": f"judge_error: {exc}",
                "raw_response": "",
            } if key_prefix == "description" else {
                "parsed_json": False,
                "partial_json": False,
                "support": 0.0,
                "causal": 0.0,
                "temporal": 0.0,
                "answer_link": 0.0,
                "hallucination": 1.0,
                "overall": 0.0,
                "reason": f"judge_error: {exc}",
                "raw_response": "",
            }
        self.cache[key] = result
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def score(self, prediction: dict[str, Any], *, query: Any = None, graph: Any = None, context_prompt: str = "") -> dict[str, Any]:
        normalized_context = normalize_context_prompt(context_prompt)
        prompt_graph = graph_from_forecast_context(normalized_context) if normalized_context else {}
        if prompt_graph.get("events") or prompt_graph.get("edges"):
            graph = prompt_graph
        prompt = build_judge_prompt(
            prediction,
            query=query,
            graph=graph,
            context_prompt=normalized_context,
            max_context_chars=self.max_context_chars,
        )
        return self._cached_generate("trace", prompt, JUDGE_SYSTEM_PROMPT, self.max_new_tokens) or {}

    def score_description(self, code: str, canonical_description: str, generated_description: str) -> dict[str, Any]:
        prompt = build_description_judge_prompt(code, canonical_description, generated_description)
        return self._cached_generate(
            "description",
            prompt,
            DESCRIPTION_JUDGE_SYSTEM_PROMPT,
            self.description_max_new_tokens,
        ) or {}
