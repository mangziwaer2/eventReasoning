# Frozen Qwen Trace Judge

The judge is an optional process reward for forecast traces. It is frozen,
lazy-loaded, and cached; the deterministic reward remains the primary signal.

## Offline pilot

```bash
PYTHONPATH=src conda run -n toolkit python src/run_trace_judge_v4.py \
  --input outputs/forecast_trace_grpo_mirai_rule_no_refine_logged/rollout_samples.jsonl \
  --output outputs/trace_judge_rollouts.jsonl \
  --metrics-output outputs/trace_judge_rollouts.metrics.json \
  --model-path models/Qwen3-4B \
  --cache-path outputs/trace_judge.cache.json \
  --max-samples 0
```

`run_trace_judge_v4.py` accepts both pipeline prediction rows and GRPO audit
rows. It normalizes chat prompts, recovers prompt-visible `Hxx`/`Rxx`
references when the audit row has no serialized graph, and writes the raw judge
response together with component scores.

## GRPO

Use the judge-aware entrypoint only after the offline score has useful variance:

```bash
PYTHONPATH=src conda run -n toolkit python src/train_forecast_trace_grpo_judge.py \
  --input outputs/grpo_context_mirai_rule_train_no_refine_420/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft/best_adapter \
  --judge-model-path models/Qwen3-4B \
  --judge-weight 0.2 \
  --judge-cache-path outputs/trace_judge_grpo.cache.json \
  --output-dir outputs/forecast_trace_grpo_judge \
  --num-generations 4 \
  --per-device-train-batch-size 4 \
  --num-train-epochs 1
```

The policy and judge each load a Qwen model. Run the offline pilot first and
check GPU memory before starting online GRPO. The judge contribution is gated
for wrong final answers so a fluent but incorrect trace cannot dominate the
answer objective.
