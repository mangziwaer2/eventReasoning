from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:
    OpenAI = None  # type: ignore


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
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
    "will",
    "with",
}

TEMPORAL_CUES = {
    "after",
    "afterward",
    "before",
    "following",
    "immediately",
    "later",
    "overnight",
    "shortly",
    "then",
}

CAUSAL_CUES = {
    "because",
    "caused",
    "causing",
    "due",
    "led",
    "made",
    "prompted",
    "resulted",
    "triggered",
}

CONSEQUENCE_TOKENS = {
    "cut",
    "declined",
    "delayed",
    "dropped",
    "fell",
    "increased",
    "jumped",
    "plunged",
    "rose",
    "slowed",
    "weakened",
}

MULTIWORD_ENTITY_PATTERNS = {
    "battery units",
    "borrowing costs",
    "central bank",
    "consumer demand",
    "expansion plans",
    "interest rates",
    "investor confidence",
    "main server",
    "product recall",
    "stock prices",
    "supply chain",
}

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "financial_policy": {"bank", "central", "interest", "rate", "rates"},
    "financing_cost": {"borrowing", "cost", "costs", "financing"},
    "business_delay": {"delay", "delayed", "expansion", "late"},
    "weather": {"flood", "flooded", "rain", "rainfall", "storm"},
    "traffic_disruption": {"roads", "slowed", "traffic"},
    "product_crisis": {"battery", "recall", "units"},
    "management": {"ceo", "resigned", "stepped"},
    "market_drop": {"fell", "plunged", "price", "prices", "stock"},
    "confidence": {"confidence", "investor", "trust", "weakened"},
    "service_outage": {"offline", "website"},
}

CAUSAL_COMPATIBILITY: dict[str, set[str]] = {
    "financial_policy": {"business_delay", "financing_cost"},
    "financing_cost": {"business_delay"},
    "product_crisis": {"confidence", "management", "market_drop"},
    "confidence": {"market_drop"},
    "service_outage": {"business_delay"},
    "weather": {"traffic_disruption"},
}


@dataclass
class LiteDocument:
    document_id: str
    title: str
    text: str
    publish_time: str | None = None
    source: str | None = None


@dataclass
class LiteEvent:
    event_id: str
    document_id: str
    text: str
    normalized_text: str
    trigger: str
    entities: list[str]
    concept_id: str
    order_index: int


@dataclass
class LiteConcept:
    concept_id: str
    canonical_name: str


@dataclass
class LiteEntity:
    entity_id: str
    canonical_name: str


@dataclass
class LiteEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    evidence_text: str


@dataclass
class LiteMemory:
    documents: list[LiteDocument]
    events: list[LiteEvent]
    concepts: list[LiteConcept]
    entities: list[LiteEntity]
    edges: list[LiteEdge]

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": [asdict(document) for document in self.documents],
            "events": [asdict(event) for event in self.events],
            "concepts": [asdict(concept) for concept in self.concepts],
            "entities": [asdict(entity) for entity in self.entities],
            "edges": [asdict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiteMemory":
        return cls(
            documents=[LiteDocument(**item) for item in payload.get("documents", [])],
            events=[LiteEvent(**item) for item in payload.get("events", [])],
            concepts=[LiteConcept(**item) for item in payload.get("concepts", [])],
            entities=[LiteEntity(**item) for item in payload.get("entities", [])],
            edges=[LiteEdge(**item) for item in payload.get("edges", [])],
        )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def slugify(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or fallback


def lexical_overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return len(set(left) & set(right))


def detect_categories(text: str) -> set[str]:
    token_set = set(content_tokens(text))
    categories: set[str] = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if token_set & keywords:
            categories.add(category)
    return categories


def pair_rule_score(left_text: str, right_text: str) -> int:
    left_categories = detect_categories(left_text)
    right_categories = detect_categories(right_text)
    score = 0
    for category in left_categories:
        if right_categories & CAUSAL_COMPATIBILITY.get(category, set()):
            score += 1
    return score


def split_event_clauses(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|;\s+", normalize_whitespace(text))
    events = []
    for chunk in chunks:
        cleaned = chunk.strip(" -")
        if len(content_tokens(cleaned)) >= 3:
            events.append(cleaned.rstrip("."))
    return events


def dedupe_event_clauses(clauses: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen_token_sets: list[set[str]] = []
    for clause in clauses:
        clause_tokens = set(content_tokens(clause))
        if not clause_tokens:
            continue

        is_duplicate = False
        for token_set in seen_token_sets:
            overlap = len(clause_tokens & token_set)
            union = len(clause_tokens | token_set)
            if union and overlap / union >= 0.80:
                is_duplicate = True
                break

        if not is_duplicate:
            deduped.append(clause)
            seen_token_sets.append(clause_tokens)

    return deduped


def infer_trigger(event_text: str) -> str:
    tokens = content_tokens(event_text)
    return tokens[0] if tokens else "event"


def extract_entities(text: str) -> list[str]:
    lowered = text.lower()
    entities = {match.strip() for match in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)}
    for pattern in MULTIWORD_ENTITY_PATTERNS:
        if pattern in lowered:
            entities.add(pattern)
    return sorted(entity for entity in entities if entity)


def infer_concept_id(event_text: str) -> str:
    lowered = event_text.lower()
    if "central bank" in lowered and ("interest rate" in lowered or "rate hike" in lowered or "rates" in lowered):
        return "concept_interest_rate_hike"
    if ("borrowing" in lowered and "cost" in lowered) or "financing pressure" in lowered:
        return "concept_financing_cost_up"
    if "delay" in lowered and ("expansion" in lowered or "hiring" in lowered):
        return "concept_business_delay"
    if "recall" in lowered and ("battery" in lowered or "product" in lowered):
        return "concept_product_recall"
    if "confidence" in lowered and ("weaken" in lowered or "down" in lowered):
        return "concept_investor_confidence_down"
    if ("stock" in lowered or "price" in lowered) and any(token in lowered for token in ("fell", "fall", "drop", "plung")):
        return "concept_stock_price_drop"

    tokens = content_tokens(event_text)[:4]
    label = "_".join(tokens) if tokens else "event"
    return f"concept_{slugify(label, 'event')}"


def has_causal_signal(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in TEMPORAL_CUES | CAUSAL_CUES)


def structure_document(raw_document: dict[str, Any]) -> tuple[LiteDocument, list[LiteEvent], list[LiteEdge], list[LiteConcept], list[LiteEntity]]:
    document = LiteDocument(
        document_id=str(raw_document.get("document_id") or raw_document.get("id") or f"doc_{abs(hash(raw_document.get('text', '')))}"),
        title=normalize_whitespace(str(raw_document.get("title") or "")),
        text=normalize_whitespace(str(raw_document.get("text") or raw_document.get("news_document") or "")),
        publish_time=raw_document.get("publish_time"),
        source=raw_document.get("source"),
    )

    event_texts = raw_document.get("events")
    if isinstance(event_texts, list) and event_texts:
        clauses = [normalize_whitespace(str(item)) for item in event_texts if str(item).strip()]
    else:
        combined = ". ".join(part for part in [document.title, document.text] if part)
        clauses = split_event_clauses(combined)
    clauses = dedupe_event_clauses(clauses)

    events: list[LiteEvent] = []
    concepts: dict[str, LiteConcept] = {}
    entities: dict[str, LiteEntity] = {}
    for index, clause in enumerate(clauses, start=1):
        concept_id = infer_concept_id(clause)
        event_entities = extract_entities(clause)
        event = LiteEvent(
            event_id=f"{document.document_id}:e{index}",
            document_id=document.document_id,
            text=clause,
            normalized_text=" ".join(content_tokens(clause)),
            trigger=infer_trigger(clause),
            entities=[f"entity_{slugify(entity, 'entity')}" for entity in event_entities],
            concept_id=concept_id,
            order_index=index,
        )
        events.append(event)
        concepts.setdefault(concept_id, LiteConcept(concept_id=concept_id, canonical_name=clause.lower()))
        for entity_name in event_entities:
            entity_id = f"entity_{slugify(entity_name, 'entity')}"
            entities.setdefault(entity_id, LiteEntity(entity_id=entity_id, canonical_name=entity_name.lower()))

    edges = infer_edges(document, events)
    return document, events, edges, list(concepts.values()), list(entities.values())


def infer_edges(document: LiteDocument, events: Sequence[LiteEvent]) -> list[LiteEdge]:
    edges: list[LiteEdge] = []
    lowered_doc = f"{document.title} {document.text}".lower()
    edge_index = 1

    for index, source in enumerate(events):
        for target in events[index + 1:]:
            temporal_confidence = 0.55 if target.order_index == source.order_index + 1 else 0.35
            edges.append(
                LiteEdge(
                    edge_id=f"{document.document_id}:temporal:{edge_index}",
                    source_id=source.event_id,
                    target_id=target.event_id,
                    relation_type="temporal_before",
                    confidence=round(min(0.95, temporal_confidence), 3),
                    evidence_text=f"{source.text} -> {target.text}",
                )
            )
            edge_index += 1

            overlap = lexical_overlap(content_tokens(source.text), content_tokens(target.text))
            compatibility = pair_rule_score(source.text, target.text)
            shared_entities = len(set(source.entities) & set(target.entities))
            effect_has_consequence = bool(set(content_tokens(target.text)) & CONSEQUENCE_TOKENS)
            local_signal = has_causal_signal(f"{source.text} {target.text}") or has_causal_signal(lowered_doc)

            confidence = 0.0
            if target.order_index == source.order_index + 1:
                confidence += 0.35
            if local_signal:
                confidence += 0.15
            confidence += 0.15 * compatibility
            confidence += 0.10 if effect_has_consequence else 0.0
            confidence += 0.10 if overlap > 0 else 0.0
            confidence += 0.10 if shared_entities > 0 else 0.0

            if confidence < 0.60:
                continue

            relation_type = "precondition" if effect_has_consequence and compatibility > 0 else "direct_cause"
            edges.append(
                LiteEdge(
                    edge_id=f"{document.document_id}:causal:{edge_index}",
                    source_id=source.event_id,
                    target_id=target.event_id,
                    relation_type=relation_type,
                    confidence=round(min(0.95, confidence), 3),
                    evidence_text=f"{source.text} -> {target.text}",
                )
            )
            edge_index += 1

    return edges


class LiteCausalEventRAG:
    def __init__(
        self,
        use_llm_reranker: bool = False,
        model_name: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.use_llm_reranker = use_llm_reranker
        self.model_name = model_name or os.getenv("LITECERF_MODEL_NAME") or os.getenv("DSPY_MODEL_NAME") or "gpt-4.1-mini"
        self.api_base = api_base or os.getenv("OPENAI_API_BASE") or None
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or None
        self.memory: LiteMemory | None = None

    def build_memory_from_records(self, raw_documents: Sequence[dict[str, Any]]) -> LiteMemory:
        documents: list[LiteDocument] = []
        events: list[LiteEvent] = []
        concepts: dict[str, LiteConcept] = {}
        entities: dict[str, LiteEntity] = {}
        edges: list[LiteEdge] = []

        for raw_document in raw_documents:
            document, doc_events, doc_edges, doc_concepts, doc_entities = structure_document(raw_document)
            documents.append(document)
            events.extend(doc_events)
            edges.extend(doc_edges)
            for concept in doc_concepts:
                concepts.setdefault(concept.concept_id, concept)
            for entity in doc_entities:
                entities.setdefault(entity.entity_id, entity)

        self.memory = LiteMemory(
            documents=documents,
            events=events,
            concepts=list(concepts.values()),
            entities=list(entities.values()),
            edges=edges,
        )
        return self.memory

    def build_memory_from_jsonl(self, path: str | Path) -> LiteMemory:
        return self.build_memory_from_records(load_jsonl(path))

    def save_memory(self, path: str | Path) -> None:
        if self.memory is None:
            raise ValueError("Memory is empty. Build or load memory before saving.")
        save_json(path, self.memory.to_dict())

    def load_memory(self, path: str | Path) -> LiteMemory:
        self.memory = LiteMemory.from_dict(load_json(path))
        return self.memory

    def forecast(
        self,
        query_text: str,
        focus_entities: Sequence[str] | None = None,
        top_k: int = 3,
        max_hops: int = 2,
    ) -> dict[str, Any]:
        if self.memory is None:
            raise ValueError("Memory is empty. Build or load memory before forecasting.")

        retrieval = self._retrieve(query_text=query_text, focus_entities=focus_entities or [], max_hops=max_hops)
        reranked = self._rerank(query_text=query_text, retrieval=retrieval)
        decision = self._decide(reranked)

        return {
            "query_text": query_text,
            "decision": decision["decision"],
            "predictions": reranked["predictions"][:top_k],
            "retrieved_seed_events": retrieval["seed_events"],
            "candidate_paths": retrieval["candidate_paths"][: top_k * 3],
            "abstention_reason": decision.get("abstention_reason"),
        }

    def _retrieve(self, query_text: str, focus_entities: Sequence[str], max_hops: int) -> dict[str, Any]:
        assert self.memory is not None
        query_tokens = content_tokens(query_text)
        normalized_focus_entities = [entity.lower() for entity in focus_entities]
        event_by_id = {event.event_id: event for event in self.memory.events}
        outgoing: dict[str, list[LiteEdge]] = defaultdict(list)
        for edge in self.memory.edges:
            if edge.relation_type in {"direct_cause", "precondition"}:
                outgoing[edge.source_id].append(edge)

        seed_scores: list[tuple[LiteEvent, float]] = []
        for event in self.memory.events:
            event_tokens = content_tokens(event.text)
            overlap = lexical_overlap(query_tokens, event_tokens)
            if overlap == 0:
                continue

            overlap_ratio = overlap / max(1, len(set(query_tokens)))
            query_norm = " ".join(query_tokens)
            exact_bonus = 0.35 if query_norm and (query_norm in event.normalized_text or event.normalized_text in query_norm) else 0.0
            entity_bonus = 0.0
            if normalized_focus_entities:
                entity_bonus = 0.20 * sum(
                    1 for entity in event.entities if any(term in entity for term in normalized_focus_entities)
                )
            score = overlap_ratio * 1.1 + exact_bonus + entity_bonus
            if score >= 0.35:
                seed_scores.append((event, round(score, 3)))

        seed_scores.sort(key=lambda item: item[1], reverse=True)
        seed_scores = seed_scores[:5]

        candidate_paths: list[dict[str, Any]] = []
        candidate_scores: dict[str, dict[str, Any]] = {}

        for seed_event, seed_score in seed_scores:
            for edge in outgoing.get(seed_event.event_id, []):
                target = event_by_id.get(edge.target_id)
                if target is None or target.concept_id == seed_event.concept_id:
                    continue
                target_tokens = set(content_tokens(target.text))
                if target_tokens and query_tokens:
                    overlap = len(target_tokens & set(query_tokens))
                    union = len(target_tokens | set(query_tokens))
                    if union and overlap / union >= 0.80:
                        continue

                path_score = round(seed_score * 0.55 + edge.confidence * 0.90, 3)
                self._register_candidate(
                    candidate_scores,
                    candidate_paths,
                    candidate_event=target,
                    source_event=seed_event,
                    path_event_ids=[seed_event.event_id, target.event_id],
                    path_texts=[seed_event.text, target.text],
                    path_score=path_score,
                    relation_path=[edge.relation_type],
                )

                if max_hops < 2:
                    continue

                for second_edge in outgoing.get(target.event_id, []):
                    target2 = event_by_id.get(second_edge.target_id)
                    if target2 is None or target2.concept_id in {seed_event.concept_id, target.concept_id}:
                        continue
                    target2_tokens = set(content_tokens(target2.text))
                    if target2_tokens and query_tokens:
                        overlap2 = len(target2_tokens & set(query_tokens))
                        union2 = len(target2_tokens | set(query_tokens))
                        if union2 and overlap2 / union2 >= 0.80:
                            continue

                    path_score2 = round(
                        seed_score * 0.40 + edge.confidence * 0.55 + second_edge.confidence * 0.55 - 0.10,
                        3,
                    )
                    self._register_candidate(
                        candidate_scores,
                        candidate_paths,
                        candidate_event=target2,
                        source_event=seed_event,
                        path_event_ids=[seed_event.event_id, target.event_id, target2.event_id],
                        path_texts=[seed_event.text, target.text, target2.text],
                        path_score=path_score2,
                        relation_path=[edge.relation_type, second_edge.relation_type],
                    )

        predictions = sorted(candidate_scores.values(), key=lambda item: item["heuristic_score"], reverse=True)

        return {
            "seed_events": [
                {
                    "event_id": event.event_id,
                    "text": event.text,
                    "score": score,
                }
                for event, score in seed_scores
            ],
            "candidate_paths": sorted(candidate_paths, key=lambda item: item["path_score"], reverse=True),
            "predictions": predictions,
        }

    def _register_candidate(
        self,
        candidate_scores: dict[str, dict[str, Any]],
        candidate_paths: list[dict[str, Any]],
        candidate_event: LiteEvent,
        source_event: LiteEvent,
        path_event_ids: list[str],
        path_texts: list[str],
        path_score: float,
        relation_path: list[str],
    ) -> None:
        candidate_paths.append(
            {
                "candidate_event_id": candidate_event.event_id,
                "candidate_text": candidate_event.text,
                "source_event_id": source_event.event_id,
                "path_event_ids": path_event_ids,
                "path_texts": path_texts,
                "relation_path": relation_path,
                "path_score": path_score,
            }
        )

        existing = candidate_scores.get(candidate_event.concept_id)
        payload = {
            "candidate_id": candidate_event.concept_id,
            "text": candidate_event.text,
            "concept_id": candidate_event.concept_id,
            "heuristic_score": path_score,
            "support_path": path_texts,
            "path_event_ids": path_event_ids,
            "relation_path": relation_path,
        }
        if existing is None or path_score > existing["heuristic_score"]:
            candidate_scores[candidate_event.concept_id] = payload
        else:
            existing["heuristic_score"] = round(existing["heuristic_score"] + 0.05, 3)

    def _rerank(self, query_text: str, retrieval: dict[str, Any]) -> dict[str, Any]:
        predictions = [dict(item) for item in retrieval["predictions"]]
        if not predictions:
            return {"predictions": []}

        if self.use_llm_reranker and self.api_key and OpenAI is not None:
            llm_order = self._llm_rerank(query_text, predictions)
            if llm_order:
                rank_bonus = {candidate_id: 0.20 * (len(llm_order) - index) for index, candidate_id in enumerate(llm_order)}
                for prediction in predictions:
                    prediction["heuristic_score"] = round(
                        prediction["heuristic_score"] + rank_bonus.get(prediction["candidate_id"], 0.0),
                        3,
                    )

        predictions.sort(key=lambda item: item["heuristic_score"], reverse=True)
        for prediction in predictions:
            prediction["verification"] = {
                "temporal_consistent": True,
                "support_path_length": len(prediction["support_path"]),
                "score": prediction["heuristic_score"],
            }
        return {"predictions": predictions}

    def _llm_rerank(self, query_text: str, predictions: Sequence[dict[str, Any]]) -> list[str]:
        if OpenAI is None or not self.api_key:
            return []

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        candidate_lines = []
        for prediction in predictions[:5]:
            candidate_lines.append(
                f"{prediction['candidate_id']}: {prediction['text']} | support={' -> '.join(prediction['support_path'])} | score={prediction['heuristic_score']}"
            )

        prompt = (
            "You are ranking event forecast candidates.\n"
            f"Query: {query_text}\n"
            "Candidates:\n"
            + "\n".join(candidate_lines)
            + "\nReturn JSON with ranked_candidate_ids as a list from best to worst. Use only the provided ids."
        )

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Rank forecast candidates conservatively and prefer causal support."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            match = re.search(r"\{.*\}", content, flags=re.S)
            if not match:
                return []
            payload = json.loads(match.group(0))
            ranked_ids = payload.get("ranked_candidate_ids", [])
            if not isinstance(ranked_ids, list):
                return []
            return [str(item) for item in ranked_ids]
        except Exception:
            return []

    def _decide(self, reranked: dict[str, Any]) -> dict[str, Any]:
        predictions = reranked["predictions"]
        if not predictions:
            return {"decision": "abstain", "abstention_reason": "No supported forecast candidates were retrieved."}

        top_score = predictions[0]["heuristic_score"]
        second_score = predictions[1]["heuristic_score"] if len(predictions) > 1 else 0.0
        score_gap = top_score - second_score
        if top_score < 1.00:
            return {"decision": "abstain", "abstention_reason": "Top forecast candidate has weak support."}
        if score_gap < 0.05 and top_score < 1.20:
            return {"decision": "abstain", "abstention_reason": "Forecast candidates are too close to separate confidently."}
        return {"decision": "forecast"}


def _print_forecast_result(result: dict[str, Any]) -> None:
    print(f"Query: {result['query_text']}")
    print(f"Decision: {result['decision']}")
    if result["abstention_reason"]:
        print(f"Reason: {result['abstention_reason']}")
    if result["retrieved_seed_events"]:
        print("Seed events:")
        for seed in result["retrieved_seed_events"]:
            print(f"  - {seed['event_id']} | score={seed['score']:.3f} | {seed['text']}")
    if result["predictions"]:
        print("Predictions:")
        for prediction in result["predictions"]:
            path = " -> ".join(prediction["support_path"])
            print(
                f"  - score={prediction['heuristic_score']:.3f} | {prediction['text']} | support={path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lite Causal EventRAG prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-memory", help="Build a small event memory from a JSONL document file.")
    build_parser.add_argument("--input", required=True, help="Input JSONL path.")
    build_parser.add_argument("--output", required=True, help="Output memory JSON path.")

    forecast_parser = subparsers.add_parser("forecast", help="Run event forecasting from an existing memory file.")
    forecast_parser.add_argument("--memory", required=True, help="Memory JSON path.")
    forecast_parser.add_argument("--query", required=True, help="Forecast query text.")
    forecast_parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to print.")
    forecast_parser.add_argument(
        "--focus-entity",
        action="append",
        default=[],
        help="Optional entity string to bias retrieval. Can be repeated.",
    )
    forecast_parser.add_argument("--use-llm", action="store_true", help="Use an OpenAI-compatible reranker if configured.")

    args = parser.parse_args()

    if args.command == "build-memory":
        cerf = LiteCausalEventRAG()
        memory = cerf.build_memory_from_jsonl(args.input)
        cerf.save_memory(args.output)
        print(
            f"Built memory with {len(memory.documents)} documents, {len(memory.events)} events, and {len(memory.edges)} edges."
        )
        return

    if args.command == "forecast":
        cerf = LiteCausalEventRAG(use_llm_reranker=args.use_llm)
        cerf.load_memory(args.memory)
        result = cerf.forecast(args.query, focus_entities=args.focus_entity, top_k=args.top_k)
        _print_forecast_result(result)


if __name__ == "__main__":
    main()
