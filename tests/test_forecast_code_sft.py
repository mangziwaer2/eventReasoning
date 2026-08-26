from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_forecast_code_sft import Example
from train_forecast_code_sft import encode
from train_forecast_code_sft import message_ids


class DictChatTemplateTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        if add_generation_prompt:
            return {"input_ids": [[10, 11, 12]], "attention_mask": [[1, 1, 1]]}
        return {"input_ids": [[10, 11, 12, 13, 14]], "attention_mask": [[1, 1, 1, 1, 1]]}

    def encode(self, text, *, add_special_tokens):
        return [20, 21]


class ForecastCodeSftTests(unittest.TestCase):
    def test_chat_template_batch_encoding_is_normalized_to_token_ids(self) -> None:
        tokenizer = DictChatTemplateTokenizer()
        messages = [{"role": "user", "content": "hello"}]
        self.assertEqual(message_ids(tokenizer, messages, generation=True), [10, 11, 12])
        self.assertEqual(message_ids(tokenizer, messages, generation=False), [10, 11, 12, 13, 14])

    def test_encode_with_dict_chat_template_contains_only_integer_tokens(self) -> None:
        item = Example(sample_id="codebook:010", prompt="Event base code: 010", completion='{"event_code":"010"}')
        encoded = encode(
            DictChatTemplateTokenizer(),
            item,
            "Return JSON only.",
            max_prompt=32,
            max_completion=32,
            max_sequence=64,
        )
        self.assertEqual(encoded["input_ids"], [10, 11, 12, 13, 14])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 13, 14])
        self.assertTrue(all(isinstance(token_id, int) for token_id in encoded["input_ids"]))


if __name__ == "__main__":
    unittest.main()
