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
from forecast_trace_graph import graph_bridge_score
from evaluate_local_qwen_pipeline import example_from_event_input
from event_input import parse_event_input_record
from train_forecast_trace_grpo import filter_rollout_rows_by_edge_count
from rl_pipeline_hooks import ForecastTraceReward
from rl_pipeline_hooks import PipelineTrajectory
from rl_pipeline_hooks import _valid_answer_format_score


class ForecastTraceTests(unittest.TestCase):
    def test_answer_format_audit_requires_every_code_to_be_three_digits(self) -> None:
        valid = parse_structured_forecast(
            '{"answers":[{"event_code":"036"},{"event_code":"042"}]}'
        )
        invalid = parse_structured_forecast(
            '{"answers":[{"event_code":"036"},{"event_code":"F21"}]}'
        )
        empty = parse_structured_forecast('{"answers":[]}')

        self.assertEqual(_valid_answer_format_score(valid), 1.0)
        self.assertEqual(_valid_answer_format_score(invalid), 0.0)
        self.assertEqual(_valid_answer_format_score(empty), 0.0)

    def test_mirai_forecast_event_input_supplies_query_and_documents(self) -> None:
        payload = {
            "schema_version": "event-input-v1",
            "sample_id": "mirai_2023-01-01_USA_CAN",
            "query_id": "2023-01-01_USA_CAN",
            "query": {
                "query_id": "2023-01-01_USA_CAN",
                "text": "As of 2023-01-01, what important event may happen next between United States and Canada?",
                "cutoff_time": "2023-01-01",
                "focus_entities": ["United States", "Canada"],
                "metadata": {"actor1_country_code": "USA", "actor2_country_code": "CAN"},
            },
            "documents": [{"document_id": "d1", "title": "Talks", "text": "Officials met."}],
            "events": [
                {
                    "event_id": "e1",
                    "trigger": "meet",
                    "mention": "Officials met",
                    "document_id": "d1",
                    "sentence_index": 0,
                    "confidence": 0.8,
                },
                {
                    "event_id": "e2",
                    "trigger": "announce",
                    "mention": "Officials announced a plan",
                    "document_id": "d1",
                    "sentence_index": 0,
                    "confidence": 0.8,
                },
            ],
            "metadata": {"answer_list": ["036"]},
        }
        example = example_from_event_input(parse_event_input_record(payload))
        self.assertEqual(example.query_id, "2023-01-01_USA_CAN")
        self.assertEqual(example.answer_list, ["036"])
        self.assertEqual(example.docids, ["d1"])

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
        )

        self.assertNotIn("Choices:\n", bundle.prompt)
        self.assertNotIn("QueryId:", bundle.prompt)
        self.assertNotIn("doc=", bundle.prompt)
        self.assertNotIn("sent=", bundle.prompt)
        self.assertNotIn("event_id=", bundle.prompt)
        self.assertNotIn("edge_id=", bundle.prompt)
        self.assertIn('"answers": [{"event_code": "<3-digit-event-code>"', bundle.prompt)
        self.assertIn("Predict every likely closed-set event_code", bundle.prompt)

        compact_bundle = build_structured_forecast_prompt(
            query=query,
            documents=documents,
            refined_graph=graph,
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
        self.assertNotIn("choice_id", prediction["final_answer"])

    def test_structured_forecast_parser_keeps_multi_label_answers(self) -> None:
        prediction = parse_structured_forecast(
            '{"answers":[{"event_code":"036","event_description":"arrest or detain"},'
            '{"event_code":"190","event_description":"use conventional military force"}]}'
        )
        self.assertEqual(prediction["predicted_event_base_codes"], ["036", "190"])
        self.assertEqual(prediction["answers"][1]["event"], "use conventional military force")
        reward = ForecastTraceReward()(prediction, {"answer_list": ["036", "190"]}, PipelineTrajectory("q"))
        self.assertEqual(reward["answer"], 1.0)

    def test_multi_label_answers_support_the_trace_bridge(self) -> None:
        prediction = parse_structured_forecast(
            """
            {
              "forecast_trace": {
                "intermediate_events": [{"trace_event_id": "ft_1", "supporting_event_ids": ["e1"]}],
                "trace_edges": [
                  {"source_id": "e1", "target_id": "ft_1", "confidence": 1.0},
                  {"source_id": "ft_1", "target_id": "answers", "confidence": 0.8}
                ]
              },
              "answers": [
                {"event_code": "036", "event_description": "arrest or detain"},
                {"event_code": "190", "event_description": "use conventional military force"}
              ]
            }
            """
        )
        score = graph_bridge_score({"events": [{"event_id": "e1"}], "edges": []}, prediction)
        self.assertEqual(prediction["predicted_event_base_codes"], ["036", "190"])
        self.assertGreater(score, 0.0)

    def test_structured_forecast_parser_accepts_refs_and_event_code(self) -> None:
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
          "final_answer": {"event_code": "036", "confidence": 0.75}
        }
        """
        prediction = parse_structured_forecast(raw)
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
                  {"source_id": "ft_1", "target_id": "answer_036", "relation_type": "raises_likelihood", "confidence": 0.9}
                ]
              },
              "final_answer": {"event_code": "036", "confidence": 0.8}
            }
            """,
        )
        graph = {
            "events": [{"event_id": "e1", "text": "police warned organizers"}],
            "edges": [{"edge_id": "r1", "source_event_id": "e1", "target_event_id": "e1", "score": 0.9}],
        }
        trajectory = PipelineTrajectory(
            sample_id="sample_1",
            metadata={
                "refined_graph": graph,
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
          "final_answer": {"event_code": "036", "confidence": 0.8}
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
        )
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 1.0)
        self.assertEqual(reward_fn.last_breakdowns[0]["answer"], 1.0)
        self.assertEqual(reward_fn.last_breakdowns[0]["valid_event_ref_ratio"], 1.0)

    def test_grpo_reward_accepts_chat_completion_and_rollout_row_context(self) -> None:
        completion = '{"final_answer": {"event_code": "036", "confidence": 0.8}}'
        row = {
            "query_id": "sample_1",
            "forecast_prompt": "Forecast the target event.",
            "forecast_system_prompt": "Return JSON.",
            "mirai_query": {"answer_list": ["036"]},
            "trajectory": PipelineTrajectory(sample_id="sample_1").to_dict(),
        }
        sample = rollout_row_to_grpo_sample(row)
        self.assertEqual(sample["prompt"][0]["content"], "Return JSON.")
        self.assertEqual(sample["query_id"], "sample_1")
        self.assertNotIn("choices", __import__("json").loads(sample["reward_context"]))

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

    def test_wrong_answer_trace_reward_retains_trace_ranking(self) -> None:
        graph = {
            "events": [{"event_id": "e1", "text": "police warned organizers"}],
            "edges": [{"edge_id": "r1", "source_event_id": "e1", "target_event_id": "e1", "score": 0.9}],
        }
        trajectory = PipelineTrajectory(
            sample_id="sample_1",
            metadata={
                "refined_graph": graph,
                "event_ref_to_id": {"H01": "e1"},
                "edge_ref_to_id": {"R01": "r1"},
            },
        )
        grounded = parse_structured_forecast(
            "\n".join(
                [
                    "{",
                    '  "forecast_trace": {"intermediate_events": [{"trace_event_id": "ft_1", "event": "security forces deploy near capital", "relative_time": "t-1", "supporting_event_ids": ["H01"], "supporting_edge_ids": ["R01"]}],',
                    '                     "trace_edges": [{"source_id": "H01", "target_id": "ft_1", "relation_type": "causes", "confidence": 0.8}, {"source_id": "ft_1", "target_id": "answer_999", "relation_type": "raises_likelihood", "confidence": 0.9}]},',
                    '  "final_answer": {"event_code": "999", "confidence": 0.8}',
                    "}",
                ]
            )
        )
        ungrounded = parse_structured_forecast(
            "\n".join(
                [
                    "{",
                    '  "forecast_trace": {"intermediate_events": [{"trace_event_id": "ft_1", "event": "security forces deploy near capital", "relative_time": "t-1"}],',
                    '                     "trace_edges": [{"source_id": "H99", "target_id": "ft_1", "relation_type": "causes", "confidence": 0.8}, {"source_id": "ft_1", "target_id": "answer_999", "relation_type": "raises_likelihood", "confidence": 0.9}]},',
                    '  "final_answer": {"event_code": "999", "confidence": 0.8}',
                    "}",
                ]
            )
        )
        reward_fn = ForecastTraceReward()
        grounded_reward = reward_fn(grounded, {"answer_list": ["036"]}, trajectory)
        ungrounded_reward = reward_fn(ungrounded, {"answer_list": ["036"]}, trajectory)
        self.assertEqual(grounded_reward["answer"], 0.0)
        self.assertEqual(ungrounded_reward["answer"], 0.0)
        self.assertGreater(grounded_reward["total"], ungrounded_reward["total"])
        self.assertLess(grounded_reward["trace"], grounded_reward["trace_unscaled"])

    def test_reward_penalizes_restatement_of_historical_event(self) -> None:
        graph = {
            "events": [{"event_id": "e1", "text": "police warned organizers"}],
            "edges": [{"edge_id": "r1", "source_event_id": "e1", "target_event_id": "e1", "score": 0.9}],
        }
        trajectory = PipelineTrajectory(
            sample_id="sample_copy",
            metadata={
                "refined_graph": graph,
                "event_ref_to_id": {"H01": "e1"},
                "edge_ref_to_id": {"R01": "r1"},
            },
        )
        copied = parse_structured_forecast(
            '{"forecast_trace":{"intermediate_events":[{"event":"police warned organizers","relative_time":"t-1","supporting_event_ids":["H01"],"supporting_edge_ids":["R01"]}],"trace_edges":[{"source_id":"H01","target_id":"ft_1","relation_type":"causes"}]},"final_answer":{"event_code":"036"}}'
        )
        distinct = parse_structured_forecast(
            '{"forecast_trace":{"intermediate_events":[{"event":"police may arrest organizers","relative_time":"t-1","supporting_event_ids":["H01"],"supporting_edge_ids":["R01"]}],"trace_edges":[{"source_id":"H01","target_id":"ft_1","relation_type":"causes"}]},"final_answer":{"event_code":"036"}}'
        )
        reward_fn = ForecastTraceReward()
        copied_reward = reward_fn(copied, {"answer_list": ["036"]}, trajectory)
        distinct_reward = reward_fn(distinct, {"answer_list": ["036"]}, trajectory)
        self.assertEqual(copied_reward["historical_copy_penalty"], 1.0)
        self.assertEqual(distinct_reward["historical_copy_penalty"], 0.0)
        self.assertLess(copied_reward["total"], distinct_reward["total"])

    def test_temporal_reward_uses_observation_to_target_interval(self) -> None:
        graph = {
            "events": [{"event_id": "e1", "text": "observed trigger", "event_time": "2023-02-28"}],
            "edges": [],
        }
        trajectory = PipelineTrajectory(
            sample_id="time_interval",
            metadata={
                "query": {"cutoff_time": "2023-02-28"},
                "refined_graph": graph,
                "event_ref_to_id": {"H01": "e1"},
            },
        )
        gold = {"answer_list": ["043"], "target_events": [{"date": "2023-03-02", "event_code": "043"}]}

        def score(event_time: str, relative_time: str) -> dict[str, float]:
            prediction = parse_structured_forecast(
                '{"forecast_trace":{"intermediate_events":[{"event":"next step",'
                f'"event_time":"{event_time}","relative_time":"{relative_time}",'
                '"supporting_event_ids":["H01"],"expected_effect":"because the trigger causes the next step"}]},'
                '"final_answer":{"event_code":"043"}}'
            )
            return ForecastTraceReward()(prediction, gold, trajectory)

        self.assertEqual(score("2023-03-01", "t-1")["temporal"], 1.0)
        self.assertEqual(score("2023-02-28", "t-2")["temporal"], 0.0)
        self.assertEqual(score("2023-03-03", "t+1")["temporal"], 0.0)

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
