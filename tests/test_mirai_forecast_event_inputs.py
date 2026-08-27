from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_mirai_forecast_event_inputs import convert_row
from event_input import parse_event_input_record


class MiraiForecastEventInputTests(unittest.TestCase):
    def test_convert_row_extracts_rule_events_and_keeps_future_labels_in_metadata(self) -> None:
        row = {
            "schema_version": "mirai-forecast-v1",
            "sample_id": "2023-01-02_USA_CAN",
            "query": {
                "query_id": "2023-01-02_USA_CAN",
                "text": "As of 2023-01-02, what important event may happen next between United States and Canada?",
                "cutoff_time": "2023-01-02",
                "focus_entities": ["United States", "Canada"],
                "metadata": {},
            },
            "documents": [
                {
                    "document_id": "d1",
                    "title": "Officials meet after talks",
                    "text": "Officials met in Ottawa. The governments announced a new agreement.",
                    "publish_time": "2023-01-01",
                    "source": "MIRAI",
                    "metadata": {},
                }
            ],
            "targets": {
                "horizon_days": 7,
                "label_start": "2023-01-03",
                "label_end": "2023-01-09",
                "event_codes": ["036"],
                "events": [{"date": "2023-01-03", "event_code": "036", "relation_name": "Arrest", "docids": []}],
            },
            "metadata": {"history_start": "2022-12-03"},
        }

        payload, skipped = convert_row(row, source_split="train", max_events=8, max_events_per_doc=8, min_events=1)

        self.assertIsNone(skipped)
        assert payload is not None
        record = parse_event_input_record(payload)
        self.assertEqual(record.schema_version, "event-input-v1")
        self.assertEqual(record.query.query_id, "2023-01-02_USA_CAN")
        self.assertGreaterEqual(len(record.events), 1)
        self.assertEqual(record.metadata["event_source"], "rule_offline_extractor")
        self.assertEqual(record.metadata["answer_list"], ["036"])
        self.assertEqual(record.metadata["target_events"][0]["relation_name"], "Arrest")

    def test_convert_row_reports_samples_without_enough_rule_events(self) -> None:
        row = {
            "sample_id": "q-empty",
            "query": {"query_id": "q-empty", "text": "What happens next?"},
            "documents": [{"document_id": "d1", "title": "Weather", "text": "A sunny day."}],
            "targets": {"event_codes": ["010"]},
        }
        payload, skipped = convert_row(row, source_split="dev", min_events=2)
        self.assertIsNone(payload)
        self.assertIsNotNone(skipped)
        assert skipped is not None
        self.assertEqual(skipped["reason"], "too_few_events")


if __name__ == "__main__":
    unittest.main()
