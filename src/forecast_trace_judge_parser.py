from __future__ import annotations

import re
from typing import Any

from forecast_trace_judge import parse_judge_response
from forecast_trace_schema import extract_first_json_object


_SCORE_KEYS = ("support", "causal", "temporal", "answer_link", "hallucination", "overall")


def parse_judge_response_robust(raw_response: str) -> dict[str, Any]:
    """Recover scalar judge scores when generation was cut off before closing JSON."""
    parsed = parse_judge_response(raw_response)
    if parsed.get("parsed_json"):
        return parsed
    text = str(raw_response or "")
    recovered: dict[str, Any] = {}
    for key in _SCORE_KEYS:
        match = re.search(rf"[\"']{re.escape(key)}[\"']\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))", text)
        if match:
            recovered[key] = max(0.0, min(float(match.group(1)), 1.0))
    reason_match = re.search(r"[\"']reason[\"']\s*:\s*[\"'](.*?)(?:[\"']\s*[,}]|$)", text, re.DOTALL)
    if reason_match:
        recovered["reason"] = reason_match.group(1).strip()
    if not recovered:
        return parsed
    if "overall" not in recovered:
        recovered["overall"] = (
            0.3 * recovered.get("support", 0.0)
            + 0.25 * recovered.get("causal", 0.0)
            + 0.2 * recovered.get("temporal", 0.0)
            + 0.25 * recovered.get("answer_link", 0.0)
        ) * (1.0 - recovered.get("hallucination", 0.0))
    return {
        **parsed,
        **recovered,
        "parsed_json": True,
        "partial_json": extract_first_json_object(text) is None,
        "raw_response": text.strip(),
    }
