# Cloud GRPO Runbook

This document contains the no-refinement baseline and the ERE-supervised refinement ablation. Run no-refinement first, then compare the refinement path with the same frozen coarse Qwen3-4B model and forecast settings.

The loop is split into two parts for engineering reasons:

```text
events -> frozen Qwen pairwise coarse graph -> fixed GRPO context
                                      |
                         LoRA B samples completions
                                      |
                         reward -> GRPO update
```

`grpo_context.jsonl` is not an offline prediction file. It contains prompts and the fixed graph environment only. It must not contain `forecast_prediction`, `raw_forecast`, or a precomputed reward. LoRA B generates the completion inside `GRPOTrainer`, the reward callable scores that completion immediately, and the optimizer updates LoRA B.

## 1. Current coarse-graph baseline

For the current LoRA B experiment, keep coarse graph generation frozen. Qwen3-4B already has useful pairwise relation ability, and the original pairwise candidate-batching path is the accepted baseline. Do not use whole-graph generation and do not train LoRA A in this run.

LoRA A remains an optional later supervised experiment. It is not needed to establish whether LoRA B can improve forecast trace generation. Therefore the commands below omit --coarse-adapter-path. Add it only after a completed coarse LoRA training run produces a directory containing adapter_config.json and adapter weights.

## 2. Prepare prompt-only GRPO contexts

This command runs event input, the frozen pairwise coarse graph, optional refinement, and prompt construction. It deliberately does not load LoRA B and does not generate a forecast. The coarse model receives candidate event pairs in batches; it does not receive the whole event list as one graph-generation prompt.

`mirai_event_input_train.jsonl` is a pseudo-training partition created from MIRAI's official `test` split. Keep `--split test`; the `train` portion in the filename identifies the local partition only.

```bash
python -u src/evaluate_local_qwen_pipeline.py \
  --stage prepare-grpo-context \
  --split test \
  --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_train.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B \
  --coarse-base-model-path models/Qwen3-4B \
  --skip-refinement \
  --prediction-mode forecast-trace \
  --coarse-topology-mode temporal-dag \
  --forecast-context-mode events-graph \
  --max-graph-events-in-prompt 14 \
  --max-graph-edges-in-prompt 24 \
  --forecast-max-event-chars 100 \
  --max-events 16 \
  --max-pairs 24 \
  --coarse-batch-size 8 \
  --coarse-max-length 1024 \
  --coarse-max-new-tokens 128 \
  --forecast-max-document-chars 500 \
  --output-dir outputs/grpo_context_mirai_rule_train_no_refine
```

The output is:

```text
outputs/grpo_context_mirai_rule_train_no_refine/grpo_context.jsonl
outputs/grpo_context_mirai_rule_train_no_refine/metrics.json
```

`grpo_context.jsonl` is not an offline prediction file. It contains prompts and the fixed graph environment only. The preparation cost is paid once per context and is not multiplied by GRPO generations or epochs.

## 3. Two-stage supervised cold start and GRPO

The GRPO adapter must continue from the forecast-stage SFT adapter. Starting GRPO directly from base Qwen3-4B is not the intended experiment.

~~~bash
python -u src/train_forecast_code_sft.py \
  --stage codebook \
  --dataset datasets/MIRAI_data.zip \
  --model-path models/Qwen3-4B \
  --output-dir outputs/mirai_code_sft_codebook \
  --num-train-epochs 8 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --max-prompt-length 256 \
  --max-completion-length 128 \
  --max-sequence-length 384 \
  --logging-steps 10
~~~

Stage one teaches all 127 MIRAI codebook entries as event_code to event_description JSON. It uses metadata from data_kg.csv, not per-query gold answers.

~~~bash
python -u src/train_forecast_code_sft.py \
  --stage forecast \
  --input outputs/grpo_context_mirai_rule_train_no_refine/grpo_context.jsonl \
  --dataset datasets/MIRAI_data.zip \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_code_sft_codebook/best_adapter \
  --output-dir outputs/mirai_forecast_sft \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --max-prompt-length 2048 \
  --max-completion-length 768 \
  --max-sequence-length 2304 \
  --logging-steps 10
~~~

Stage two trains events plus coarse graph to answers, where every answer contains both event_code and its canonical event_description. It preserves code semantics while learning the MIRAI multi-label prediction task. The trainer aborts if any target would be truncated. Use outputs/mirai_forecast_sft/best_adapter for GRPO.

~~~bash
python -u src/train_forecast_trace_grpo.py \
  --input outputs/grpo_context_mirai_rule_train_no_refine/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft/best_adapter \
  --output-dir outputs/forecast_trace_grpo_mirai_rule_no_refine_logged \
  --max-samples 32 \
  --min-coarse-edges 1 \
  --num-generations 4 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --max-prompt-length 2048 \
  --max-completion-length 512 \
  --logging-steps 1 \
  --reward-log-every 1 \
  --sample-log-every 1 \
  --sample-log-count 2
~~~

rollout_samples.jsonl records raw multi-label answers, descriptions, graph references, and reward breakdown. The terminal answer reward is set-based F1 against MIRAI AnswerList; descriptions are SFT semantic anchors and are not exact-match evaluation targets.

## 4. Evaluate dev

Evaluation is the only stage that generates a final forecast for every sample:

```bash
python -u src/evaluate_local_qwen_pipeline.py \
  --split test \
  --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_dev.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B \
  --coarse-base-model-path models/Qwen3-4B \
  --skip-refinement \
  --forecast-base-model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_grpo_mirai_rule_no_refine_logged/final_adapter \
  --policy forecast_trace_reward \
  --prediction-mode forecast-trace \
  --forecast-temperature 0.0 \
  --forecast-max-new-tokens 512 \
  --output-dir outputs/eval_mirai_rule_dev_no_refine_grpo
```

Do not run `score_forecast_trace_rewards.py` before GRPO. That would only rescore stored completions and is not part of this online training path.

## 5. ERE-supervised refinement training

Use the same frozen Qwen3-4B coarse model as the no-refinement path. First cache real Qwen coarse graphs on MAVEN-ERE:

~~~bash
python -u src/build_maven_qwen_refinement_cache.py \
  --dataset datasets/MAVEN_ERE.zip --split train --limit 0 \
  --max-events 16 --max-sentence-gap 3 --max-pairs 64 \
  --coarse-keep-threshold 0.5 --coarse-topology-mode temporal-dag \
  --base-model-path models/Qwen3-4B \
  --coarse-batch-size 1 --coarse-max-length 1024 --coarse-max-new-tokens 48 --log-every 1 \
  --output-dir outputs/maven_qwen_refinement_cache --overwrite
~~~

The cache builder and GRPO context builder use the same coarse-graph input:
the two event IDs, triggers, compact event mentions, and pair metadata. They do
not pass a query, document title, document snippet, or full document to Qwen.
The builder uses Qwen3 /no_think mode by default because each pair only needs a
compact JSON relation decision. It prints progress for every source row and
records the `events-mentions-v1` input contract plus generation/truncation
statistics in cache_manifest.json. Do not pass --coarse-thinking for normal
cache construction. That ablation requires --coarse-max-new-tokens 1024 or
higher and is materially slower.

Inspect outputs/maven_qwen_refinement_cache/cache_manifest.json. Confirm complete is true, then check coarse_parse_rate, coarse_edges, trainable_samples, and zero_candidate_samples. The refinement trainer rejects missing, incomplete, legacy, or non-events-mentions-v1 caches. Zero-candidate rows are retained for diagnosis but skipped by the refinement loader.

Train the graph refiner with ERE labels:

~~~bash
python -u src/train_refinement.py \
  --dataset-mode maven-qwen-cache \
  --qwen-refinement-cache outputs/maven_qwen_refinement_cache/samples.jsonl \
  --limit 0 --validation-ratio 0.1 --epochs 40 \
  --hidden-dim 192 --message-steps 4 --dropout 0.12 \
  --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 \
  --keep-loss-weight 1.0 \
  --strength-loss-weight 0.3 --density-loss-weight 0.08 \
  --keep-pos-weight auto --amp auto \
  --log-every 25 --debug-samples 2 \
  --output-dir outputs/refinement_graph_maven_qwen
~~~

The best checkpoint is outputs/refinement_graph_maven_qwen/refinement_model.pt. Missing ERE gold edges are not injected into candidates, so refinement does not receive leaked positive candidates. The current checkpoint format is `graph-editor-v3`: it has only keep/drop and strength heads. The Qwen candidate relation is copied unchanged. It has no relation retyping, query feature encoder, or frontier head. Checkpoints produced by the retired architecture must be retrained.

The recommended path runs the existing coarse `--coarse-topology-mode temporal-dag` pass and then refinement with `--refinement-topology-mode none`. This means Qwen decides each relation, the refiner filters and rescales those edges, and the relation string is never changed. The optional final `--refinement-topology-mode temporal-dag` decoder remains available when raw reciprocal candidates are intentionally evaluated.

## 6. With-refinement GRPO contexts

Prepare contexts with the trained graph refiner. Completion candidates stay disabled to match the real-Qwen-edge training distribution:

~~~bash
python -u src/evaluate_local_qwen_pipeline.py \
  --stage prepare-grpo-context \
  --split test \
  --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_train.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B \
  --coarse-base-model-path models/Qwen3-4B \
  --refinement-model-path outputs/refinement_graph_maven_qwen/refinement_model.pt \
  --enable-refinement \
  --no-completion-candidates \
  --refinement-topology-mode none \
  --prediction-mode forecast-trace \
  --coarse-topology-mode temporal-dag \
  --max-events 16 --max-pairs 24 --coarse-batch-size 8 \
  --coarse-max-length 1024 --coarse-max-new-tokens 128 \
  --forecast-context-mode events-graph \
  --max-graph-events-in-prompt 14 --max-graph-edges-in-prompt 24 \
  --forecast-max-event-chars 100 --forecast-max-new-tokens 512 \
  --output-dir outputs/grpo_context_mirai_rule_train_with_refinement
~~~

## 7. With-refinement: train and evaluate LoRA B

Train online GRPO with the refined contexts:

~~~bash
python -u src/train_forecast_trace_grpo.py \
  --input outputs/grpo_context_mirai_rule_train_with_refinement/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft/best_adapter \
  --output-dir outputs/forecast_trace_grpo_mirai_rule_with_refinement \
  --max-samples 0 --num-generations 4 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 --num-train-epochs 1 \
  --max-prompt-length 2048 --max-completion-length 512 \
  --logging-steps 1 --reward-log-every 1 \
  --sample-log-every 1 --sample-log-count 2
~~~

Evaluate the refinement ablation on dev:

~~~bash
python -u src/evaluate_local_qwen_pipeline.py \
  --stage evaluate --split test --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_dev.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B \
  --coarse-base-model-path models/Qwen3-4B \
  --refinement-model-path outputs/refinement_graph_maven_qwen/refinement_model.pt \
  --enable-refinement --no-completion-candidates \
  --refinement-topology-mode none \
  --forecast-base-model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_grpo_mirai_rule_with_refinement/final_adapter \
  --prediction-mode forecast-trace --forecast-temperature 0.0 \
  --forecast-max-new-tokens 512 \
  --output-dir outputs/eval_mirai_rule_dev_with_refinement_grpo
~~~

Compare outputs/eval_mirai_rule_dev_no_refine_grpo/metrics.json with outputs/eval_mirai_rule_dev_with_refinement_grpo/metrics.json. Both paths use the official test split with different local pseudo-split event files; keep event file, coarse adapter, forecast model, prompt context mode, and generation limits identical.

## 8. Cloud logs and diagnosis

For GRPO inspect training.log, reward_history.jsonl, rollout_samples.jsonl, sample_summary.json, and metrics.json. The rollout file contains representative prompts, raw completions, parsed forecasts, graph summaries, and reward components.

For refinement inspect:

~~~text
outputs/maven_qwen_refinement_cache/cache_manifest.json
outputs/maven_qwen_refinement_cache/samples.jsonl
outputs/refinement_graph_maven_qwen/train_history.json
outputs/refinement_graph_maven_qwen/debug_readable.log
~~~

A near-zero coarse_parse_rate is a coarse-model or prompt problem, not a refinement problem. Refinement cannot recover edges that never enter the coarse candidate graph.
