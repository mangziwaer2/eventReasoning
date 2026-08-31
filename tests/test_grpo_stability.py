from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from train_forecast_trace_grpo import CollapseGuardCallback
from train_forecast_trace_grpo import STAGE_MANIFEST_NAME
from train_forecast_trace_grpo import build_training_config
from train_forecast_trace_grpo import reference_policy_mode
from train_forecast_trace_grpo import summarize_stage_health
from train_forecast_trace_grpo import validate_two_stage_args
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
        loss_type="",
        repetition_penalty=1.0,
        **kwargs,
    ) -> None:
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.beta = beta
        self.max_grad_norm = max_grad_norm
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

    @staticmethod
    def _stage_args(**overrides):
        values = {
            "grpo_stage": "bootstrap",
            "beta": 0.0,
            "adapter_path": "adapter",
            "allow_unverified_bootstrap_adapter": False,
            "stage_min_reward_groups": 2,
            "stage_min_valid_answer_rate": 0.5,
            "stage_min_format_score": 0.75,
            "stage_min_answer_hit_group_rate": 0.05,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_bootstrap_stage_requires_zero_beta(self) -> None:
        validate_two_stage_args(self._stage_args(beta=0.0))
        with self.assertRaisesRegex(ValueError, "requires --beta 0"):
            validate_two_stage_args(self._stage_args(beta=0.04))
        with self.assertRaisesRegex(ValueError, "preserve main's --wrong-answer-trace-scale 0.05"):
            validate_two_stage_args(self._stage_args(wrong_answer_trace_scale=0.2))

    def test_kl_stage_requires_passed_bootstrap_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "bootstrap"
            adapter = output_dir / "final_adapter"
            adapter.mkdir(parents=True)
            args = self._stage_args(grpo_stage="kl", beta=0.04, adapter_path=str(adapter))
            with self.assertRaisesRegex(ValueError, "requires a passed bootstrap manifest"):
                validate_two_stage_args(args)

            (output_dir / STAGE_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "stage": "bootstrap",
                        "status": "passed",
                        "reward_policy": "forecast_trace_reward",
                        "wrong_answer_trace_scale": 0.05,
                    }
                ),
                encoding="utf-8",
            )
            validate_two_stage_args(args)

            (output_dir / STAGE_MANIFEST_NAME).write_text(
                json.dumps({"stage": "bootstrap", "status": "failed"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "has not passed"):
                validate_two_stage_args(args)

    def test_reference_policy_mode_requires_ref_adapter_for_peft(self) -> None:
        class _Model:
            peft_config = {"default": object(), "ref": object()}

        class _UnsafeModel:
            peft_config = {"default": object()}

        self.assertEqual(reference_policy_mode(_Model(), 0.04), "ref_adapter")
        self.assertEqual(reference_policy_mode(_UnsafeModel(), 0.04), "base_disabled")
        self.assertEqual(reference_policy_mode(_UnsafeModel(), 0.0), "disabled")

    def test_bootstrap_health_gate_tracks_answer_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reward_history = Path(temp_dir) / "reward_history.jsonl"
            rows = [
                {
                    "batch_size": 4,
                    "breakdown_mean": {"answer": 0.1, "format": 0.9, "valid_answer_format": 0.75},
                },
                {
                    "batch_size": 4,
                    "breakdown_mean": {"answer": 0.0, "format": 0.8, "valid_answer_format": 0.5},
                },
            ]
            reward_history.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            health = summarize_stage_health(reward_history, self._stage_args())
            self.assertTrue(health["passed"])
            self.assertAlmostEqual(health["metrics"]["valid_answer_rate"], 0.625)
            self.assertAlmostEqual(health["metrics"]["answer_hit_group_rate"], 0.5)

            collapsed = summarize_stage_health(
                reward_history,
                self._stage_args(),
                collapse_triggered=True,
            )
            self.assertFalse(collapsed["passed"])
            self.assertIn("collapse_guard", collapsed["failures"])


if __name__ == "__main__":
    unittest.main()
