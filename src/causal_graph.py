from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QuerySpec:
    query_id: str
    text: str
    cutoff_time: str | None = None
    focus_entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NewsDocument:
    document_id: str
    title: str
    text: str
    publish_time: str | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceSpan:
    document_id: str
    sentence_index: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EventNode:
    event_id: str
    text: str
    normalized_text: str
    document_id: str
    sentence_index: int
    participants: list[str] = field(default_factory=list)
    node_type: str = "observed"
    confidence: float = 0.0
    evidence: list[EvidenceSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


@dataclass(slots=True)
class CausalEdge:
    edge_id: str
    source_event_id: str
    target_event_id: str
    relation_type: str
    confidence: float
    evidence: list[EvidenceSpan] = field(default_factory=list)
    is_hypothesis: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


@dataclass(slots=True)
class RetrievalHit:
    document_id: str
    score: float
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphBuildTrace:
    retrieval_hits: list[RetrievalHit] = field(default_factory=list)
    event_notes: list[str] = field(default_factory=list)
    bridge_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_hits": [item.to_dict() for item in self.retrieval_hits],
            "event_notes": self.event_notes,
            "bridge_notes": self.bridge_notes,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class LocalCausalGraph:
    query: QuerySpec
    documents: list[NewsDocument]
    events: list[EventNode]
    edges: list[CausalEdge]
    trace: GraphBuildTrace = field(default_factory=GraphBuildTrace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "documents": [item.to_dict() for item in self.documents],
            "events": [item.to_dict() for item in self.events],
            "edges": [item.to_dict() for item in self.edges],
            "trace": self.trace.to_dict(),
        }


@dataclass(slots=True)
class CoarseCausalEdge:
    edge_id: str
    source_event_id: str
    target_event_id: str
    relation_type: str
    score: float
    evidence: list[EvidenceSpan] = field(default_factory=list)
    feature_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "relation_type": self.relation_type,
            "score": self.score,
            "evidence": [item.to_dict() for item in self.evidence],
            "feature_scores": self.feature_scores,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CoarseCausalGraph:
    query: QuerySpec
    documents: list[NewsDocument]
    events: list[EventNode]
    edges: list[CoarseCausalEdge]
    trace: GraphBuildTrace = field(default_factory=GraphBuildTrace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "documents": [item.to_dict() for item in self.documents],
            "events": [item.to_dict() for item in self.events],
            "edges": [item.to_dict() for item in self.edges],
            "trace": self.trace.to_dict(),
        }


@dataclass(slots=True)
class ForecastCandidate:
    text: str
    confidence: float = 0.0
    rationale: str = ""
    support_event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ForecastResult:
    query_id: str
    prompt: str
    raw_response: str
    candidates: list[ForecastCandidate] = field(default_factory=list)
    gold: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "candidates": [item.to_dict() for item in self.candidates],
            "gold": self.gold,
            "metadata": self.metadata,
        }
