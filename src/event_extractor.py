from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field

from causal_graph import NewsDocument
from causal_graph import QuerySpec
from event_extraction import AtomicEvent
from event_extraction import extract_atomic_events
from event_extraction import split_sentences


@dataclass(slots=True)
class SentenceEventExtraction:
    sentence_index: int
    sentence_text: str
    is_title: bool
    events: list[AtomicEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "sentence_index": self.sentence_index,
            "sentence_text": self.sentence_text,
            "is_title": self.is_title,
            "events": [
                {
                    "text": event.text,
                    "normalized_text": event.normalized_text,
                    "trigger": event.trigger,
                    "participants": event.participants,
                    "score": event.score,
                }
                for event in self.events
            ],
        }


@dataclass(slots=True)
class DocumentEventExtraction:
    document_id: str
    title: str
    extractor_name: str
    sentence_extractions: list[SentenceEventExtraction] = field(default_factory=list)

    def iter_events(self) -> list[tuple[int, AtomicEvent]]:
        rows: list[tuple[int, AtomicEvent]] = []
        for sentence_extraction in self.sentence_extractions:
            for event in sentence_extraction.events:
                rows.append((sentence_extraction.sentence_index, event))
        return rows

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "extractor_name": self.extractor_name,
            "sentence_extractions": [item.to_dict() for item in self.sentence_extractions],
        }


class BaseEventExtractor(ABC):
    name = "base"

    def split_document(self, document: NewsDocument) -> list[str]:
        return [document.title] + split_sentences(document.text)

    def extract_document(self, query: QuerySpec, document: NewsDocument) -> DocumentEventExtraction:
        sentence_extractions: list[SentenceEventExtraction] = []
        for sentence_index, sentence in enumerate(self.split_document(document)):
            events = self.extract_sentence(
                query_text=query.text,
                sentence=sentence,
                is_title=sentence_index == 0,
            )
            if not events:
                continue
            sentence_extractions.append(
                SentenceEventExtraction(
                    sentence_index=sentence_index,
                    sentence_text=sentence,
                    is_title=sentence_index == 0,
                    events=events,
                )
            )
        return DocumentEventExtraction(
            document_id=document.document_id,
            title=document.title,
            extractor_name=self.name,
            sentence_extractions=sentence_extractions,
        )

    @abstractmethod
    def extract_sentence(self, query_text: str, sentence: str, is_title: bool = False) -> list[AtomicEvent]:
        raise NotImplementedError


class RuleBasedEventExtractor(BaseEventExtractor):
    name = "rule"

    def extract_sentence(self, query_text: str, sentence: str, is_title: bool = False) -> list[AtomicEvent]:
        return extract_atomic_events(query_text=query_text, sentence=sentence, is_title=is_title)


def build_event_extractor(name: str = "rule") -> BaseEventExtractor:
    normalized = name.strip().lower()
    if normalized == "rule":
        return RuleBasedEventExtractor()
    raise ValueError(f"Unsupported event extractor: {name}")
