from __future__ import annotations

import unittest

from forecast_trace_judge import build_description_judge_prompt
from forecast_trace_judge_runtime import parse_description_judge_response
from forecast_trace_schema import parse_structured_forecast


class DescriptionJudgeTests(unittest.TestCase):
    def test_description_is_preserved_by_forecast_parser(self) -> None:
        prediction = parse_structured_forecast(
            '{"answers":[{"event_code":"042","event_description":"a diplomatic visit"}]}'
        )
        self.assertEqual(prediction["final_answer"]["event_description"], "a diplomatic visit")

    def test_description_judge_accepts_semantic_score(self) -> None:
        result = parse_description_judge_response('{"match":0.82,"reason":"same event type"}')
        self.assertTrue(result["parsed_json"])
        self.assertAlmostEqual(result["match"], 0.82)

    def test_description_prompt_does_not_require_exact_wording(self) -> None:
        prompt = build_description_judge_prompt("042", "make a diplomatic visit", "visit diplomatically")
        self.assertIn("semantic equivalence", prompt)
        self.assertIn("not exact wording", prompt)


if __name__ == "__main__":
    unittest.main()
