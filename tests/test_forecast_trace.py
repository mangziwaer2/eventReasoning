from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecast_trace_schema import parse_structured_forecast
from coarse_graph_dataset import parse_pair_payload
from rl_pipeline_hooks import ForecastTraceReward
from rl_pipeline_hooks import PipelineTrajectory


class ForecastTraceTests(unittest.TestCase):
    def test_structured_forecast_parser_accepts_refs_and_choice_id(self) -> None:
        raw = """
        {
          "forecast_trace": {
            "intermediate_events": [
              {
                "trace_event_id": "ft_1",
                "event": {"trigger": "deploy", "mention": "security forces deploy near capital", "actors": ["security forces"], "relative_time": "t-1"},
                "supporting_events": [{"event_ref": "H01", "event": "police warned organizers"}],
                "supporting_edge_refs": ["R01"],
                "expected_effect": "raises likelihood of arrests",
                "confidence": 0.7
              }
            ],
            "trace_edges": [
              {"source_ref": "H01", "target_ref": "ft_1", "relation_type": "causes", "confidence": 0.8},
              {"source_ref": "ft_1", "target_ref": "answer_C001", "relation_type": "raises_likelihood", "confidence": 0.9}
            ]
          },
          "final_answer": {"choice_id": "C001", "confidence": 0.75}
        }
        """
        prediction = parse_structured_forecast(raw, choices=[{"choice_id": "C001", "event_code": "036"}])
        self.assertTrue(prediction["parsed_json"])
        self.assertEqual(prediction["predicted_event_base_code"], "036")
        self.assertEqual(prediction["forecast_trace"]["intermediate_events"][0]["supporting_event_ids"], ["H01"])
        self.assertEqual(prediction["forecast_trace"]["intermediate_events"][0]["supporting_edge_ids"], ["R01"])

    def test_reward_scores_grounded_trace_with_bridge(self) -> None:
        prediction = parse_structured_forecast(
            """
            {
              "forecast_trace": {
                "intermediate_events": [
                  {
                    "trace_event_id": "ft_1",
                    "event": "security forces deploy near capital",
                    "actors": ["security forces"],
                    "relative_time": "t-1",
                    "supporting_event_ids": ["H01"],
                    "supporting_edge_ids": ["R01"],
                    "expected_effect": "raises likelihood of arrests",
                    "confidence": 0.7
                  }
                ],
                "trace_edges": [
                  {"source_id": "H01", "target_id": "ft_1", "relation_type": "causes", "confidence": 0.8},
                  {"source_id": "ft_1", "target_id": "answer_C001", "relation_type": "raises_likelihood", "confidence": 0.9}
                ]
              },
              "final_answer": {"choice_id": "C001", "event_code": "036", "confidence": 0.8}
            }
            """,
            choices=[{"choice_id": "C001", "event_code": "036"}],
        )
        graph = {
            "events": [{"event_id": "e1", "text": "police warned organizers"}],
            "edges": [{"edge_id": "r1", "source_event_id": "e1", "target_event_id": "e1", "score": 0.9}],
        }
        trajectory = PipelineTrajectory(
            sample_id="sample_1",
            metadata={
                "refined_graph": graph,
                "choices": [{"choice_id": "C001", "event_code": "036"}],
                "event_ref_to_id": {"H01": "e1"},
                "edge_ref_to_id": {"R01": "r1"},
            },
        )
        reward = ForecastTraceReward()(prediction, {"answer_list": ["036"]}, trajectory)
        self.assertEqual(reward["answer"], 1.0)
        self.assertEqual(reward["valid_event_ref_ratio"], 1.0)
        self.assertEqual(reward["valid_edge_ref_ratio"], 1.0)
        self.assertGreater(reward["graph_bridge"], 0.0)
        self.assertGreater(reward["total"], 1.0)

    def test_coarse_pair_parser_prefers_confidence_and_accepts_score(self) -> None:
        parsed = parse_pair_payload('{"relation_type": "none", "confidence": 1.0}')
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["relation_type"], "none")
        self.assertEqual(parsed["confidence"], 1.0)

        legacy = parse_pair_payload('{"relation_type": "causes", "score": 0.74}')
        self.assertIsNotNone(legacy)
        self.assertEqual(legacy["relation_type"], "causes")
        self.assertEqual(legacy["confidence"], 0.74)


if __name__ == "__main__":
    unittest.main()
