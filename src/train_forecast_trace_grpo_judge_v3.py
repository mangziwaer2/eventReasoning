from __future__ import annotations

import argparse
import sys
from typing import Any

import train_forecast_trace_grpo as base_grpo
from forecast_trace_judge_reward_v4 import RobustAuditedJudgeAugmentedGRPOReward


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--judge-model-path", default="models/Qwen3-4B")
    parser.add_argument("--judge-weight", type=float, default=0.2)
    parser.add_argument("--judge-max-new-tokens", type=int, default=384)
    parser.add_argument("--judge-thinking", action="store_true")
    parser.add_argument("--judge-cache-path", default=None)
    judge_args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    class ConfiguredReward(RobustAuditedJudgeAugmentedGRPOReward):
        def __init__(self, policy_name: str = "forecast_trace_reward", reward_key: str = "total", error_reward: float = -0.25, **kwargs: Any) -> None:
            super().__init__(
                judge_args.judge_model_path,
                policy_name=policy_name,
                reward_key=reward_key,
                error_reward=error_reward,
                judge_weight=judge_args.judge_weight,
                max_new_tokens=judge_args.judge_max_new_tokens,
                thinking=judge_args.judge_thinking,
                cache_path=judge_args.judge_cache_path,
                audit_path=kwargs.get("audit_path"),
                audit_every=kwargs.get("audit_every", 1),
                sample_audit_path=kwargs.get("sample_audit_path"),
                sample_audit_every=kwargs.get("sample_audit_every", 1),
                sample_audit_limit=kwargs.get("sample_audit_limit", 2),
            )

    base_grpo.ForecastTraceGRPOReward = ConfiguredReward
    base_grpo.main()


if __name__ == "__main__":
    main()
