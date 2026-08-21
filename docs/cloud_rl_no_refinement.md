# Cloud RL no-refinement runbook

This runbook uses the fixed MIRAI pseudo split event inputs built under:

datasets/mirai_event_inputs_rule/
Files:

mirai_event_input_train.jsonl
mirai_event_input_dev.jsonl
mirai_event_input_test.jsonl
mirai_event_input_all.jsonl
mirai_pseudo_splits_seed42.json
build_summary.json
## 1. Generate train rollouts on cloud

Use frozen Qwen4B for coarse graph construction, skip refinement, and save forecast prompts for RL.

python src/evaluate_local_qwen_pipeline.py
--split test
--event-source precomputed
--precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_train.jsonl
--queries-from-precomputed-events
--model-path models/Qwen3-4B
--coarse-base-model-path models/Qwen3-4B 
--forecast-base-model-path models/Qwen3-4B
--policy forecast_trace_reward
--prediction-mode forecast-trace
--forecast-temperature 0.7
--forecast-max-new-tokens 512
--output-dir outputs/rollouts_mirai_rule_train_no_refine

Optional: if you already have a supervised LoRA-B adapter, add:

--forecast-adapter-path outputs/forecast_trace_sft_lora/best_adapter
## 2. Recompute rewards

python src/score_forecast_trace_rewards.py \
  --input outputs/rollouts_mirai_rule_train_no_refine/predictions.jsonl \
  --output outputs/rollouts_mirai_rule_train_no_refine/predictions.rescored.jsonl \
  --metrics-output outputs/rollouts_mirai_rule_train_no_refine/reward_metrics.json \
  --policy forecast_trace_reward
## 3. Train LoRA-B with offline RL

python src/train_forecast_trace_rl.py \
  --input outputs/rollouts_mirai_rule_train_no_refine/predictions.rescored.jsonl \
  --model-path models/Qwen3-4B \
  --output-dir outputs/forecast_trace_rl_mirai_rule_no_refine_lora \
  --completion-source raw \
  --reward-source recompute \
  --weighting exp \
  --reward-baseline mean \
  --reward-temperature 1.0 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 5e-5 \
  --max-length 2048
Optional: continue from an SFT adapter:

--adapter-path outputs/forecast_trace_sft_lora/best_adapter
## 4. Evaluate on dev

python src/evaluate_local_qwen_pipeline.py \
  --split test \
  --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_dev.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B \
  --coarse-base-model-path models/Qwen3-4B \
  --skip-refinement \
  --forecast-base-model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_rl_mirai_rule_no_refine_lora/best_adapter \
  --policy forecast_trace_reward \
  --prediction-mode forecast-trace \
  --forecast-temperature 0.0 \
  --forecast-max-new-tokens 512 \
  --output-dir outputs/eval_mirai_rule_dev_no_refine_rl
## 5. Final held-out pseudo-test evaluation

Run this only after tuning on dev.

python src/evaluate_local_qwen_pipeline.py \
  --split test \
  --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_test.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B \
  --coarse-base-model-path models/Qwen3-4B \
  --skip-refinement \
  --forecast-base-model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_rl_mirai_rule_no_refine_lora/best_adapter \
  --policy forecast_trace_reward \
  --prediction-mode forecast-trace \
  --forecast-temperature 0.0 \
  --forecast-max-new-tokens 512 \
  --output-dir outputs/eval_mirai_rule_test_no_refine_rl
## Metrics to compare

Use metrics.json and reward_metrics.json:

forecast_parse_rate
code_hit_rate
code_hit_with_alternatives_rate
average_reward
average_reward_breakdown.valid_event_ref_ratio
average_reward_breakdown.valid_edge_ref_ratio
average_reward_breakdown.graph_bridge
average_reward_breakdown.generic_penalty
