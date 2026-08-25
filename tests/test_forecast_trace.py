from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecast_trace_schema import parse_structured_forecast
from forecast_trace_prompt import build_structured_forecast_prompt
from causal_graph import CoarseCausalEdge
from causal_graph import CoarseCausalGraph
from causal_graph import EventNode
from causal_graph import NewsDocument
from causal_graph import QuerySpec
from coarse_graph_dataset import parse_pair_payload
from forecast_trace_grpo_rewards import ForecastTraceGRPOReward
from forecast_trace_grpo_rewards import completion_to_text
from forecast_trace_grpo_rewards import rollout_row_to_grpo_sample
from train_forecast_trace_grpo import filter_rollout_rows_by_edge_count
from rl_pipeline_hooks import ForecastTraceReward
from rl_pipeline_hooks import PipelineTrajectory


class ForecastTraceTests(unittest.TestCase):
    def test_no_refinement_prompt_uses_direct_event_code_without_choices(self) -> None:
        query = QuerySpec(query_id="q1", text="What happens next?", cutoff_time="2024-01-01")
        documents = [NewsDocument(document_id="d1", title="Observed", text="Police warned organizers.")]
        events = [
            EventNode(
                event_id="e1",
                text="trigger=warn; police warned organizers",
                normalized_text="police warned organizers",
                document_id="d1",
                sentence_index=0,
                metadata={"trigger": "warn"},
            )
        ]
        graph = CoarseCausalGraph(
            query=query,
            documents=documents,
            events=events,
            edges=[
                CoarseCausalEdge(
                    edge_id="r1",
                    source_event_id="e1",
                    target_event_id="e1",
                    relation_type="causes",
                    score=0.8,
                )
            ],
        )

        bundle = build_structured_forecast_prompt(
            query=query,
            documents=documents,
            refined_graph=graph,
            choices=[
                {"choice_id": "C001", "event_code": "036", "description": "arrest or detain"},
            ],
        )

        self.assertEqual(bundle.choices, [])
        self.assertNotIn("Choices:\n", bundle.prompt)
        self.assertIn('"event_code": "000"', bundle.prompt)
        self.assertIn("No candidate choices are provided", bundle.prompt)

        compact_bundle = build_structured_forecast_prompt(
            query=query,
            documents=documents,
            refined_graph=graph,
            choices=[],
            context_mode="events-graph",
            max_event_chars=24,
        )
        self.assertNotIn("Documents:\n", compact_bundle.prompt)
        self.assertIn("mention=", compact_bundle.prompt)


        prediction = parse_structured_forecast(
            '{"forecast_trace": {"intermediate_events": [], "trace_edges": []}, '
            '"final_answer": {"event_code": "036", "confidence": 0.8}}'
        )
        self.assertEqual(prediction["predicted_event_base_code"], "036")
        self.assertEqual(prediction["final_answer"]["choice_id"], "")

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

    def test_grpo_reward_callable_scores_batch_completion(self) -> None:
        choices = [{"choice_id": "C001", "event_code": "036"}]
        completion = """
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
              {"source_id": "H01", "target_id": "ft_1", "relation_type": "causes", "confidence": 0.8}
            ]
          },
          "final_answer": {"choice_id": "C001", "event_code": "036", "confidence": 0.8}
        }
        """
        trajectory = PipelineTrajectory(
            sample_id="sample_1",
            metadata={
                "refined_graph": {
                    "events": [{"event_id": "e1", "text": "police warned organizers"}],
                    "edges": [{"edge_id": "r1", "source_event_id": "e1", "target_event_id": "e1", "score": 0.9}],
                },
                "event_ref_to_id": {"H01": "e1"},
                "edge_ref_to_id": {"R01": "r1"},
            },
        )
        reward_fn = ForecastTraceGRPOReward()
        rewards = reward_fn(
            prompts=["prompt text"],
            completions=[completion],
            mirai_query=[{"answer_list": ["036"]}],
            trajectory=[trajectory.to_dict()],
            choices=[choices],
        )
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 1.0)
        self.assertEqual(reward_fn.last_breakdowns[0]["answer"], 1.0)
        self.assertEqual(reward_fn.last_breakdowns[0]["valid_event_ref_ratio"], 1.0)

    def test_grpo_reward_accepts_chat_completion_and_rollout_row_context(self) -> None:
        choices = [{"choice_id": "C001", "event_code": "036"}]
        completion = '{"final_answer": {"choice_id": "C001", "event_code": "036", "confidence": 0.8}}'
        row = {
            "query_id": "sample_1",
            "forecast_prompt": "Forecast the target event.",
            "forecast_system_prompt": "Return JSON.",
            "mirai_query": {"answer_list": ["036"]},
            "choices": choices,
            "trajectory": PipelineTrajectory(sample_id="sample_1").to_dict(),
        }
        sample = rollout_row_to_grpo_sample(row)
        self.assertEqual(sample["prompt"][0]["content"], "Return JSON.")
        self.assertEqual(sample["query_id"], "sample_1")
        self.assertEqual(__import__("json").loads(sample["reward_context"])["choices"], choices)

        reward_fn = ForecastTraceGRPOReward()
        rewards = reward_fn(
            completions=[[{"role": "assistant", "content": completion}]],
            reward_contexts=[row],
        )
        self.assertEqual(completion_to_text([{"role": "assistant", "content": completion}]), completion)
        self.assertEqual(rewards, [1.05])

    def test_grpo_reward_uses_direct_event_code_without_choice_inventory(self) -> None:
        completion = '{"final_answer": {"event_code": "036", "confidence": 0.8}}'
        row = {
            "query_id": "sample_no_choices",
            "forecast_prompt": "Forecast the target event directly.",
            "forecast_system_prompt": "Return forecast_trace and event_code JSON.",
            "mirai_query": {"answer_list": ["036"]},
            "choices": [],
            "trajectory": PipelineTrajectory(sample_id="sample_no_choices").to_dict(),
        }
        reward_fn = ForecastTraceGRPOReward()
        rewards = reward_fn(
            completions=[completion],
            reward_contexts=[row],
        )
        self.assertEqual(rewards, [1.05])
        self.assertEqual(reward_fn.last_breakdowns[0]["answer"], 1.0)

    def test_grpo_low_edge_filter_is_optional(self) -> None:
        rows = [
            {"query_id": "zero", "trajectory": {"metadata": {"refined_graph": {"edges": []}}}},
            {"query_id": "one", "trajectory": {"metadata": {"refined_graph": {"edges": [{"edge_id": "r1"}]}}}},
            {"query_id": "two", "trajectory": {"metadata": {"refined_graph": {"edges": [{"edge_id": "r1"}, {"edge_id": "r2"}]}}}},
        ]
        kept, dropped = filter_rollout_rows_by_edge_count(rows, min_coarse_edges=2)
        self.assertEqual([row["query_id"] for row in kept], ["two"])
        self.assertEqual(dropped, 2)
        kept, dropped = filter_rollout_rows_by_edge_count(rows, min_coarse_edges=0)
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, 0)

        fallback_rows = [{"trajectory": {"metadata": {}}, "coarse": {"edge_count": 2}}]
        kept, dropped = filter_rollout_rows_by_edge_count(fallback_rows, min_coarse_edges=2)
        self.assertEqual(kept, fallback_rows)
        self.assertEqual(dropped, 0)

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
