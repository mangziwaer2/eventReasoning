from __future__ import annotations

import argparse
import sys
from typing import Any

import train_forecast_trace_grpo as base_grpo
from forecast_trace_judge_reward import JudgeGRPOReward


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--judge-model-path", default="models/Qwen3-4B")
    parser.add_argument("--judge-weight", type=float, default=0.2)
    parser.add_argument(
        "--wrong-answer-judge-gate",
        type=float,
        default=0.2,
        help="Partial trace/judge weight before an answer-code hit; 0.2 preserves the original bootstrap signal.",
    )
    parser.add_argument("--description-weight", type=float, default=0.05)
    parser.add_argument("--description-max-new-tokens", type=int, default=96)
    parser.add_argument("--judge-max-context-chars", type=int, default=12000)
    parser.add_argument("--codebook-dataset-path", default="datasets/MIRAI_data.zip")
    parser.add_argument("--judge-max-new-tokens", type=int, default=384)
    parser.add_argument("--judge-thinking", action="store_true")
    parser.add_argument("--judge-cache-path", default=None)
    judge_args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    class ConfiguredJudgeReward(JudgeGRPOReward):
        def __init__(
            self,
            policy_name: str = "forecast_trace_reward",
            reward_key: str = "total",
            error_reward: float = -0.25,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                judge_args.judge_model_path,
                policy_name=policy_name,
                reward_key=reward_key,
                error_reward=error_reward,
                wrong_answer_trace_scale=kwargs.get("wrong_answer_trace_scale"),
                wrong_answer_judge_gate=judge_args.wrong_answer_judge_gate,
                judge_weight=judge_args.judge_weight,
                description_weight=judge_args.description_weight,
                description_max_new_tokens=judge_args.description_max_new_tokens,
                judge_max_context_chars=judge_args.judge_max_context_chars,
                codebook_dataset_path=judge_args.codebook_dataset_path,
                judge_max_new_tokens=judge_args.judge_max_new_tokens,
                judge_thinking=judge_args.judge_thinking,
                judge_cache_path=judge_args.judge_cache_path,
                audit_path=kwargs.get("audit_path"),
                audit_every=kwargs.get("audit_every", 1),
                sample_audit_path=kwargs.get("sample_audit_path"),
                sample_audit_every=kwargs.get("sample_audit_every", 1),
                sample_audit_limit=kwargs.get("sample_audit_limit", 2),
                sample_human_path=kwargs.get("sample_human_path"),
            )

    base_grpo.ForecastTraceGRPOReward = ConfiguredJudgeReward
    base_grpo.main()


if __name__ == "__main__":
    main()
