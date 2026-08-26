from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from forecast_trace_judge_parser import parse_judge_response_robust
from merge_grpo_context_shards import load_rows


class JudgeAdditionTests(unittest.TestCase):
    def test_truncated_judge_json_recovers_scores(self) -> None:
        result = parse_judge_response_robust(
            '{"support":0.8,"causal":0.6,"temporal":0.5,"answer_link":0.7,"hallucination":0.1,"overall":0.65'
        )
        self.assertTrue(result["parsed_json"])
        self.assertTrue(result["partial_json"])
        self.assertAlmostEqual(result["overall"], 0.65)

    def test_context_shards_deduplicate_query_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(json.dumps({"query_id": "a"}) + "\n", encoding="utf-8")
            second.write_text(json.dumps({"query_id": "a"}) + "\n" + json.dumps({"query_id": "b"}) + "\n", encoding="utf-8")
            self.assertEqual([row["query_id"] for row in load_rows([first, second])], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
