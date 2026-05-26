from __future__ import annotations

import re
from dataclasses import dataclass


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}

QUESTION_WORDS = {"as", "how", "what", "when", "where", "which", "who", "why"}

NON_EVENT_TOKENS = STOPWORDS | QUESTION_WORDS | {
    "according",
    "afp",
    "ani",
    "author",
    "day",
    "figure",
    "list",
    "monday",
    "month",
    "more",
    "read",
    "recently",
    "reported",
    "reports",
    "reuters",
    "said",
    "says",
    "source",
    "summary",
    "sunday",
    "thursday",
    "today",
    "tuesday",
    "update",
    "wednesday",
    "week",
    "year",
    "yesterday",
}

EVENT_VERB_HINTS = {
    "accuse",
    "acknowledge",
    "agree",
    "announce",
    "approve",
    "attack",
    "ban",
    "bomb",
    "call",
    "clash",
    "close",
    "condemn",
    "criticize",
    "delay",
    "denounce",
    "deploy",
    "displace",
    "evacuate",
    "expand",
    "force",
    "halt",
    "impose",
    "increase",
    "invade",
    "kill",
    "launch",
    "meet",
    "open",
    "plan",
    "propose",
    "protest",
    "raise",
    "reject",
    "respond",
    "resume",
    "sanction",
    "strike",
    "support",
    "threaten",
    "transfer",
    "urge",
    "visit",
    "warn",
    "worsen",
    "withdraw",
}

EVENT_NOUN_HINTS = {
    "attack",
    "bombing",
    "ceasefire",
    "condemnation",
    "conflict",
    "crisis",
    "displacement",
    "evacuation",
    "invasion",
    "plan",
    "proposal",
    "protest",
    "rejection",
    "response",
    "sanction",
    "strike",
    "visit",
    "warning",
    "war",
}

CLAUSE_SPLIT_PATTERN = re.compile(
    r"\s*(?:,?\s+(?:and|but|while|after|because|since|as|which|who|that)\s+|[;:])\s*",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class AtomicEvent:
    text: str
    normalized_text: str
    trigger: str
    participants: list[str]
    score: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def normalize_text(text: str) -> str:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    return " ".join(tokens)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def split_event_clauses(sentence: str) -> list[str]:
    raw_parts = CLAUSE_SPLIT_PATTERN.split(sentence.strip())
    clauses: list[str] = []
    for part in raw_parts:
        clean = part.strip(" ,.-")
        if len(clean) < 12:
            continue
        clauses.append(clean)
    return clauses or [sentence.strip()]


def extract_titlecase_entities(text: str) -> list[str]:
    matches = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    entities: list[str] = []
    for item in matches:
        lowered = item.lower()
        if lowered in QUESTION_WORDS:
            continue
        if item not in entities:
            entities.append(item)
    return entities[:6]


def clean_document_text(title: str, text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_lines: list[str] = []
    seen_normalized: set[str] = set()
    normalized_title = normalize_text(title)

    for line in lines:
        normalized_line = normalize_text(line)
        if not normalized_line:
            continue
        if normalized_line == normalized_title:
            continue
        if normalized_line in seen_normalized:
            continue
        if line.lower().startswith("by ") and len(line.split()) <= 8:
            continue
        if line.lower().startswith("source:"):
            continue
        seen_normalized.add(normalized_line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _simple_stem(token: str) -> str:
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def infer_trigger(text: str) -> str:
    tokens = tokenize(text)
    for token in tokens:
        stem = _simple_stem(token)
        if token in NON_EVENT_TOKENS or stem in NON_EVENT_TOKENS:
            continue
        if token in EVENT_VERB_HINTS or stem in EVENT_VERB_HINTS:
            return stem
    for token in tokens:
        stem = _simple_stem(token)
        if token in NON_EVENT_TOKENS or stem in NON_EVENT_TOKENS:
            continue
        if token in EVENT_NOUN_HINTS or stem in EVENT_NOUN_HINTS:
            return stem
    return ""


def lexical_overlap(left: str, right: str) -> tuple[float, list[str]]:
    left_tokens = {token for token in tokenize(left) if token not in STOPWORDS}
    right_tokens = {token for token in tokenize(right) if token not in STOPWORDS}
    if not left_tokens or not right_tokens:
        return 0.0, []
    matched = sorted(left_tokens & right_tokens)
    union = left_tokens | right_tokens
    return len(matched) / len(union), matched


def score_clause(query_text: str, clause: str, is_title: bool = False) -> float:
    overlap_score, _ = lexical_overlap(query_text, clause)
    tokens = tokenize(clause)
    trigger = infer_trigger(clause)
    trigger_bonus = 0.18 if trigger else 0.0
    entity_bonus = 0.1 if extract_titlecase_entities(clause) else 0.0
    event_word_bonus = 0.18 if any(
        _simple_stem(token) in EVENT_VERB_HINTS or _simple_stem(token) in EVENT_NOUN_HINTS for token in tokens
    ) else 0.0
    title_bonus = 0.15 if is_title else 0.0
    length_penalty = 0.0 if len(tokens) <= 26 else -0.08
    return overlap_score + trigger_bonus + entity_bonus + event_word_bonus + title_bonus + length_penalty


def extract_atomic_events(query_text: str, sentence: str, is_title: bool = False) -> list[AtomicEvent]:
    events: list[AtomicEvent] = []
    seen_normalized: set[str] = set()
    for clause in split_event_clauses(sentence):
        normalized = normalize_text(clause)
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)

        trigger = infer_trigger(clause)
        if not trigger:
            continue
        score = score_clause(query_text, clause, is_title=is_title)
        if score < 0.24:
            continue
        events.append(
            AtomicEvent(
                text=clause,
                normalized_text=normalized,
                trigger=trigger,
                participants=extract_titlecase_entities(clause),
                score=round(min(score, 0.98), 4),
            )
        )
    return events
