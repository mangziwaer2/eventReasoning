# Forecast Trace GRPO Training

The project now exposes a TRL-compatible reward callable in
`src/forecast_trace_grpo_rewards.py`. It parses every generated completion with
the existing structured forecast parser and reuses `forecast_trace_reward`, so
GRPO receives one deterministic scalar per completion while the full reward
breakdown remains available in `reward_fn.last_breakdowns` for debugging.

## Input

First generate rollouts with prompts saved:

```powershell
python src/evaluate_local_qwen_pipeline.py --prediction-mode forecast-trace --output-dir outputs/local_qwen_pipeline_eval
```

The GRPO dataset adapter keeps `forecast_prompt`, `mirai_query`, and the
serialized pipeline trajectory. `choices` remains an empty compatibility field
and is not serialized into the active forecast prompt. Rows without
`forecast_prompt` are skipped.

## Training

Install the dependencies from `requirements.txt`, then run:

```powershell
python src/train_forecast_trace_grpo.py `
  --input outputs/local_qwen_pipeline_eval/predictions.jsonl `
  --model-path models/Qwen2.5-0.5B `
  --output-dir outputs/forecast_trace_grpo `
  --num-generations 4 `
  --per-device-train-batch-size 4 `
  --gradient-accumulation-steps 8
```

`--num-generations` controls the group size used by GRPO. In a single-process
run, `--per-device-train-batch-size` must be at least that value and divisible
by it. The default reward is the `total` field from `forecast_trace_reward`.

Use `--no-chat-prompt` when the base model was trained with plain text prompts.
Use `--adapter-path` to continue an existing trainable LoRA adapter.

## Reward behavior

The reward combines final answer correctness, JSON/schema format, support-event
and support-edge grounding, temporal ordering, graph bridge quality, and
penalties for generic or over-dense traces. Malformed completions receive a
finite fallback reward (`--error-reward`, default `-0.25`) so one bad generation
cannot crash the trainer or produce NaN advantages.
