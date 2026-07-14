from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from event_input import EventInputValidationError
from event_input import load_event_input_index
from event_input import materialize_event_input
from event_input import parse_event_input_record
from coarse_graph_dataset import build_event_pair_inference_samples
from coarse_graph_dataset import load_preextracted_document_graph_sample


def valid_payload() -> dict:
    return {
        "schema_version": "event-input-v1",
        "sample_id": "sample_1",
        "query_id": "q1",
        "query": {
            "query_id": "q1",
            "text": "What happens next?",
            "cutoff_time": "2025-01-01",
            "focus_entities": [],
            "metadata": {},
        },
        "documents": [
            {
                "document_id": "d1",
                "title": "Example",
                "text": "A attacked B. B retreated.",
            }
        ],
        "events": [
            {
                "event_id": "e1",
                "trigger": "attacked",
                "mention": "A attacked B",
                "document_id": "d1",
                "sentence_index": 0,
                "confidence": 1.0,
                "evidence": "A attacked B.",
            },
            {
                "event_id": "e2",
                "trigger": "retreated",
                "mention": "B retreated",
                "document_id": "d1",
                "sentence_index": 1,
                "confidence": 0.9,
            },
        ],
    }


class EventInputTests(unittest.TestCase):
    def test_parse_and_materialize_strict_events(self) -> None:
        record = parse_event_input_record(valid_payload())
        query, documents, events = materialize_event_input(record)
        self.assertEqual(query.query_id, "q1")
        self.assertEqual(documents[0].document_id, "d1")
        self.assertEqual(events[0].text, "trigger=attacked; mention=A attacked B")
        self.assertEqual(events[0].metadata["trigger"], "attacked")
        self.assertEqual(events[0].evidence[0].text, "A attacked B.")

    def test_duplicate_event_ids_are_rejected(self) -> None:
        payload = valid_payload()
        payload["events"][1]["event_id"] = "e1"
        with self.assertRaises(EventInputValidationError):
            parse_event_input_record(payload)

    def test_unknown_document_reference_is_rejected(self) -> None:
        payload = valid_payload()
        payload["events"][0]["document_id"] = "missing"
        with self.assertRaises(EventInputValidationError):
            parse_event_input_record(payload)

    def test_relation_edges_are_rejected(self) -> None:
        payload = valid_payload()
        payload["edges"] = [{"source_event_id": "e1", "target_event_id": "e2"}]
        with self.assertRaises(EventInputValidationError):
            parse_event_input_record(payload)

    def test_self_contained_input_builds_document_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "events.json"
            path.write_text(json.dumps(valid_payload()), encoding="utf-8")
            sample = load_preextracted_document_graph_sample(path, max_events=2)
        self.assertEqual(sample.query.query_id, "q1")
        self.assertEqual(len(sample.events), 2)
        self.assertEqual(sample.metadata["event_source"], "precomputed")
        pairs = build_event_pair_inference_samples(sample, max_sentence_gap=3, max_pairs=8)
        self.assertEqual(len(pairs), 2)

    def test_jsonl_is_indexed_by_query_id(self) -> None:
        second = valid_payload()
        second["sample_id"] = "sample_2"
        second["query_id"] = "q2"
        second["query"]["query_id"] = "q2"
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "events.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in (valid_payload(), second)),
                encoding="utf-8",
            )
            index = load_event_input_index(path)
        self.assertEqual(set(index), {"q1", "q2"})


if __name__ == "__main__":
    unittest.main()
