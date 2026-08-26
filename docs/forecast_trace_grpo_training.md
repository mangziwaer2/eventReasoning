# Forecast Trace GRPO Training

`train_forecast_trace_grpo.py` is the online training entry for the forecasting model. TRL samples `num_generations` completions from the current LoRA B policy inside each GRPO step, calls `ForecastTraceGRPOReward` on those completions, computes group-relative advantages, and updates the adapter.

## Input contract

Use the prompt-only file produced by:

```bash
python src/evaluate_local_qwen_pipeline.py --stage prepare-grpo-context ...
```

The input file is `grpo_context.jsonl`. A row contains:

- `forecast_prompt`
- `forecast_system_prompt`
- `mirai_query`
- `trajectory` with the graph and reference mappings used by the reward
- `schema_version` and `stage`

It must not contain a precomputed forecast completion or a precomputed reward. `choices` remains an empty compatibility field for the no-refinement path.

## Training command

```powershell
python -u src/train_forecast_trace_grpo.py `
  --input outputs/grpo_context_mirai_rule_train_no_refine/grpo_context.jsonl `
  --model-path models/Qwen3-4B `
  --output-dir outputs/forecast_trace_grpo `
  --num-generations 4 `
  --per-device-train-batch-size 4 `
  --gradient-accumulation-steps 8 `
  --learning-rate 5e-6 `
  --num-train-epochs 1 `
  --logging-steps 1 `
  --reward-log-every 1
```

The trainer writes `training.log`, `reward_history.jsonl`, `run_config.json`, `sample_summary.json`, `train_history.json`, `metrics.json`, and `final_adapter/`.

Use `--adapter-path` to continue an existing trainable LoRA adapter. Use `--no-chat-prompt` only when the base model expects plain text prompts.

## Reward behavior

The reward combines final event-code correctness, JSON/schema format, support-event and support-edge grounding, temporal ordering, graph bridge quality, and penalties for generic or over-dense traces. Malformed completions receive the finite fallback reward from `--error-reward` so one bad generation cannot crash the trainer or produce NaN advantages.

When observing a cloud run, inspect these fields in `reward_history.jsonl`:

- `reward_mean`, `reward_min`, `reward_max`
- `breakdown_mean.answer`
- `breakdown_mean.format`
- `breakdown_mean.grounding`
- `breakdown_mean.graph_bridge`
- `error_count`

A useful first signal is that `format` and `answer` become nonzero while `error_count` stays low. A rising total reward with a collapsed answer component is not sufficient evidence of useful forecasting.