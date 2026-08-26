from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from forecast_trace_judge import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from forecast_trace_schema import extract_first_json_object
from local_qwen_lora import load_qwen_for_inference
from path_utils import resolve_repo_path


def normalize_context_prompt(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                payload = json.loads(text)
                if isinstance(payload, list):
                    parts = [str(item.get("content", "")) for item in payload if isinstance(item, dict) and item.get("content")]
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


def parse_judge_response_robust(raw_response: str) -> dict[str, Any]:
    text = str(raw_response or "").strip()
    payload = extract_first_json_object(text) or {}
    keys = ("support", "causal", "temporal", "answer_link", "hallucination", "overall")

    def clamp(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    scores = {key: clamp(payload.get(key, 0.0)) for key in keys}
    partial = not bool(payload)
    if partial:
        for key in keys:
            match = re.search(rf"[\"']{re.escape(key)}[\"']\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))", text)
            if match:
                scores[key] = clamp(match.group(1))
        partial = True
    if "overall" not in payload and not any(key == "overall" and scores[key] for key in keys):
        scores["overall"] = (
            0.3 * scores["support"]
            + 0.25 * scores["causal"]
            + 0.2 * scores["temporal"]
            + 0.25 * scores["answer_link"]
        ) * (1.0 - scores["hallucination"])
    reason = str(payload.get("reason", "")).strip()
    return {
        "parsed_json": bool(payload) or any(scores[key] for key in keys),
        "partial_json": partial and extract_first_json_object(text) is None,
        **scores,
        "reason": reason,
        "raw_response": text,
    }


class FrozenQwenTraceJudge:
    """Single frozen Qwen runtime used by the only supported judge-GRPO entrypoint."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_new_tokens: int = 384,
        thinking: bool = False,
        cache_path: str | Path | None = None,
        max_context_chars: int = 12000,
    ) -> None:
        self.model_path = resolve_repo_path(str(model_path))
        self.max_new_tokens = max(64, int(max_new_tokens))
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

    def _encode(self, prompt: str):
        self._load()
        messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"] if isinstance(encoded, dict) else encoded
        else:
            text = f"<|im_start|>system\n{JUDGE_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            encoded = self.tokenizer(text, return_tensors="pt")
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if not hasattr(input_ids, "to"):
            input_ids = self.torch.tensor(input_ids, dtype=self.torch.long)
        return input_ids.to(next(self.model.parameters()).device)

    def _generate(self, prompt: str) -> str:
        input_ids = self._encode(prompt)
        attention_mask = self.torch.ones_like(input_ids)
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()

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
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key in self.cache:
            return {**self.cache[key], "cached": True}
        try:
            raw = self._generate(prompt if self.thinking else prompt + "\n/no_think")
            result = parse_judge_response_robust(raw)
        except Exception as exc:
            result = {
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
