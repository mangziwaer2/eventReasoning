from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from train_forecast_trace_grpo import CollapseGuardCallback
from train_forecast_trace_grpo import build_training_config
from local_qwen_lora import _load_tokenizer_compat


class _Config:
    def __init__(
        self,
        *,
        learning_rate=0.0,
        num_train_epochs=0.0,
        max_steps=-1,
        beta=0.0,
        max_grad_norm=0.0,
        generation_batch_size=None,
        loss_type="",
        repetition_penalty=1.0,
        **kwargs,
    ) -> None:
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.generation_batch_size = generation_batch_size
        self.loss_type = loss_type
        self.repetition_penalty = repetition_penalty
        self.extra = kwargs


class _Control:
    should_training_stop = False


class _TokenizerLoader:
    calls: list[str] = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append(str(source))
        if str(source) == "adapter":
            raise AttributeError("'list' object has no attribute 'keys'")
        return object()


class GrpoStabilityTests(unittest.TestCase):
    def test_training_config_has_kl_and_hard_step_cap(self) -> None:
        args = argparse.Namespace(
            output_dir="outputs/test",
            learning_rate=1e-6,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            generation_batch_size=4,
            num_train_epochs=1.0,
            max_steps=120,
            warmup_ratio=0.1,
            max_grad_norm=0.5,
            logging_steps=10,
            save_steps=100,
            num_generations=4,
            max_prompt_length=2048,
            max_completion_length=512,
            seed=42,
            beta=0.04,
        )
        config = build_training_config(_Config, args)
        self.assertEqual(config.max_steps, 120)
        self.assertAlmostEqual(config.beta, 0.04)
        self.assertAlmostEqual(config.max_grad_norm, 0.5)
        self.assertEqual(config.loss_type, "grpo")
        self.assertAlmostEqual(config.repetition_penalty, 1.05)
        self.assertEqual(config.generation_batch_size, 4)

    def test_collapse_guard_stops_after_patience(self) -> None:
        guard = CollapseGuardCallback(patience=2, min_entropy=0.03, max_zero_std_ratio=0.95)
        control = _Control()
        guard.on_log(None, None, control, logs={"entropy": 0.01, "frac_reward_zero_std": 1.0})
        self.assertFalse(control.should_training_stop)
        guard.on_log(None, None, control, logs={"entropy": 0.01, "frac_reward_zero_std": 1.0})
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.triggered)

    def test_tokenizer_falls_back_for_old_adapter_metadata(self) -> None:
        _TokenizerLoader.calls = []
        tokenizer = _load_tokenizer_compat(_TokenizerLoader, "adapter", fallback_source="base")
        self.assertIsNotNone(tokenizer)
        self.assertEqual(_TokenizerLoader.calls, ["adapter", "base"])


if __name__ == "__main__":
    unittest.main()
