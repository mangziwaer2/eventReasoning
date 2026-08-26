from __future__ import annotations

import argparse
import sys

import train_forecast_trace_grpo as base_grpo
from forecast_trace_judge_reward_v2 import JudgeAugmentedGRPOReward


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--judge-model-path", default="models/Qwen3-4B")
    parser.add_argument("--judge-weight", type=float, default=0.2)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--judge-thinking", action="store_true")
    parser.add_argument("--judge-cache-path", default=None)
    judge_args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    class ConfiguredReward(JudgeAugmentedGRPOReward):
        def __init__(self, policy_name="forecast_trace_reward", reward_key="total", error_reward=-0.25, **kwargs):
            del policy_name, reward_key, error_reward, kwargs
            super().__init__(
                judge_args.judge_model_path,
                judge_weight=judge_args.judge_weight,
                max_new_tokens=judge_args.judge_max_new_tokens,
                thinking=judge_args.judge_thinking,
                cache_path=judge_args.judge_cache_path,
            )

    base_grpo.ForecastTraceGRPOReward = ConfiguredReward
    base_grpo.main()


if __name__ == "__main__":
    main()
