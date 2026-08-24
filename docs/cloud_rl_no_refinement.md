# Cloud Online GRPO Runbook

This is the no-refinement research path for the forecasting stage.

The loop is split into two parts for engineering reasons:

```text
events -> LoRA A coarse graph -> fixed GRPO context
                                      |
                         LoRA B samples completions
                                      |
                         reward -> GRPO update
```

`grpo_context.jsonl` is not an offline prediction file. It contains prompts and the fixed graph environment only. It must not contain `forecast_prediction`, `raw_forecast`, or a precomputed reward. LoRA B generates the completion inside `GRPOTrainer`, the reward callable scores that completion immediately, and the optimizer updates LoRA B.

## 1. Train LoRA A

LoRA A is trained separately as an event-pair relation classifier. Start with the existing supervised command:

```bash
python src/train_coarse_graph_qwen.py \
  --model-path models/Qwen2.5-0.5B \
  --train-limit 8192 \
  --validation-limit 512 \
  --max-events 16 \
  --negative-ratio 1.5 \
  --epochs 3 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --lr 2e-4 \
  --document-mode title \
  --debug-samples 2 \
  --log-every 25 \
  --output-dir outputs/coarse_graph_qwen_lora_run1
```

Use `outputs/coarse_graph_qwen_lora_run1/best_adapter` as `--coarse-adapter-path` below. If an existing A adapter is not available, omit that option and use the frozen base model for a baseline.

## 2. Prepare prompt-only GRPO contexts

This command runs event input, coarse graph construction, optional refinement, and prompt construction. It deliberately does not load LoRA B and does not generate a forecast.

```bash
python -u src/evaluate_local_qwen_pipeline.py \
  --stage prepare-grpo-context \
  --split train \
  --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_train.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen2.5-0.5B \
  --coarse-base-model-path models/Qwen2.5-0.5B \
  --coarse-adapter-path outputs/coarse_graph_qwen_lora_run1/best_adapter \
  --skip-refinement \
  --prediction-mode forecast-trace \
  --output-dir outputs/grpo_context_mirai_rule_train_no_refine
```

The output is:

```text
outputs/grpo_context_mirai_rule_train_no_refine/grpo_context.jsonl
outputs/grpo_context_mirai_rule_train_no_refine/metrics.json
```

This stage can still take time because LoRA A classifies candidate event pairs. That cost is paid once per context and is not multiplied by GRPO generations or epochs.

## 3. Start online GRPO for LoRA B

```bash
python -u src/train_forecast_trace_grpo.py \
  --input outputs/grpo_context_mirai_rule_train_no_refine/grpo_context.jsonl \
  --model-path models/Qwen2.5-0.5B \
  --output-dir outputs/forecast_trace_grpo_mirai_rule_no_refine \
  --num-generations 4 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --logging-steps 1 \
  --save-steps 100 \
  --reward-log-every 1
```

For a first cloud run, keep `--num-train-epochs 1`. The generated completions are not written back into the context file. The trainer writes:

```text
training.log
reward_history.jsonl
run_config.json
sample_summary.json
train_history.json
metrics.json
final_adapter/
```

`reward_history.jsonl` is the main file to send back for diagnosis. Each row is one reward callback batch and includes reward mean/min/max, component means, and fallback-error count.

## 4. Evaluate dev

Evaluation is the only stage that generates a final forecast for every sample:

```bash
python -u src/evaluate_local_qwen_pipeline.py \
  --split dev \
  --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_dev.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen2.5-0.5B \
  --coarse-base-model-path models/Qwen2.5-0.5B \
  --coarse-adapter-path outputs/coarse_graph_qwen_lora_run1/best_adapter \
  --skip-refinement \
  --forecast-base-model-path models/Qwen2.5-0.5B \
  --forecast-adapter-path outputs/forecast_trace_grpo_mirai_rule_no_refine/final_adapter \
  --policy forecast_trace_reward \
  --prediction-mode forecast-trace \
  --forecast-temperature 0.0 \
  --forecast-max-new-tokens 512 \
  --output-dir outputs/eval_mirai_rule_dev_no_refine_grpo
```

Do not run `score_forecast_trace_rewards.py` before GRPO. That would only rescore stored completions and is not part of this online training path.

## Model and document policy

- LoRA A: event pairs -> coarse graph. Use compact document mode (`title` or `snippet`) and do not include full documents by default.
- LoRA B: coarse graph -> forecast trace -> final answer. Keep documents in the forecast prompt because they provide grounding evidence, but cap each document with `--forecast-max-document-chars`.
- Do not tell the model it is LoRA A or LoRA B. The active prompt identifies it as a future event forecasting model.
- Train A first, then use A as a fixed environment for B GRPO. Joint A+B RL is a separate experiment because the graph is a discrete intermediate action and needs its own graph-level reward.