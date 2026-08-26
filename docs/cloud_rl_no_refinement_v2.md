# Cloud Judge-GRPO Runbook (No Refinement)

This is the current executable training path. It uses frozen Qwen3-4B for the
pairwise coarse graph, forecast SFT as the cold start, online GRPO, and a frozen
Qwen judge for trace quality. Refinement remains a separate ablation.

```text
historical events -> frozen Qwen coarse graph -> fixed prompt context
                                                    |
                                      LoRA B samples trace + answers
                                                    |
                 deterministic reward + frozen Qwen trace judge -> GRPO
```

## 1. MIRAI data boundary

The public package has no official train split:

| Data | Rows | Rule |
| --- | ---: | --- |
| `MIRAI/test/relation_query.csv` | 705 | Source of all local pseudo splits |
| `MIRAI/test_subset/relation_query.csv` | 100 | Fully overlaps `test`; never append it |
| pseudo train event inputs | 420 | SFT and GRPO |
| pseudo dev event inputs | 140 | model/threshold selection |
| pseudo holdout event inputs | 141 | final local report only |

Four of the 705 source queries were skipped because the rule extractor produced
fewer than two usable historical events. The 420/140/141 files and their fixed
seed-42 manifest already exist under `datasets/mirai_event_inputs_rule/`.

These are local pseudo-split results, not official MIRAI test generalization.
Never put a pseudo-dev or pseudo-holdout QueryId into SFT or GRPO while selecting
prompts, reward weights, epochs, or thresholds.

Rebuild the event-input split only when intentionally changing the extraction
contract:

```bash
PYTHONPATH=src conda run -n toolkit python src/build_mirai_event_inputs.py \
  --dataset datasets/MIRAI_data.zip --source-split test \
  --seed 42 --train-ratio 0.6 --dev-ratio 0.2 \
  --max-docs 4 --max-events 16 --max-events-per-doc 6 --min-events 2 \
  --output-dir datasets/mirai_event_inputs_rule
```

Keep `datasets/mirai_event_inputs_rule/mirai_pseudo_splits_seed42.json` with
every reported experiment.

## 2. Baseline settings

- Coarse graph: frozen `models/Qwen3-4B`, pairwise JSON, `/no_think`.
- Refinement: disabled with `--skip-refinement`.
- Policy: forecast LoRA initialized from forecast SFT.
- Judge: frozen `models/Qwen3-4B`, `/no_think`, weight `0.2`.
- Judge output budget: 384 tokens.
- GRPO: four generations per prompt for the full run.
- Do not enable `--coarse-thinking` or `--judge-thinking` in the baseline.

The policy and judge together used about 18.6 GiB peak GPU memory in the local
pilot. Judge reward is gated when the answer is wrong, so it cannot dominate
the event-code objective.

## 3. Build all 420 prompt contexts

The earlier 200-context output was an interrupted/limited preprocessing result,
not a dataset limit. Frozen-Qwen coarse graph generation is slow, so use
resumable 25-row shards.

```bash
PYTHONPATH=src conda run -n toolkit python src/split_event_input_shards.py \
  --input datasets/mirai_event_inputs_rule/mirai_event_input_train.jsonl \
  --output-dir outputs/mirai_event_input_train_shards_25 --shard-size 25
```

Run one shard first:

```bash
PYTHONPATH=src conda run -n toolkit python src/evaluate_local_qwen_pipeline.py \
  --stage prepare-grpo-context --split test --limit 0 \
  --event-source precomputed \
  --precomputed-events outputs/mirai_event_input_train_shards_25/events_0000.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B --coarse-base-model-path models/Qwen3-4B \
  --skip-refinement --prediction-mode forecast-trace \
  --coarse-topology-mode temporal-dag --forecast-context-mode events-graph \
  --max-graph-events-in-prompt 14 --max-graph-edges-in-prompt 24 \
  --forecast-max-event-chars 100 --forecast-max-document-chars 500 \
  --max-events 16 --max-pairs 64 --coarse-batch-size 8 \
  --coarse-max-length 1024 --coarse-max-new-tokens 128 \
  --output-dir outputs/grpo_context_shards/no_refine_0000
```

If it writes 25 valid rows, process the remaining shards. The loop skips a
completed output file:

```bash
for shard in outputs/mirai_event_input_train_shards_25/events_*.jsonl; do
  name=$(basename "$shard" .jsonl)
  out="outputs/grpo_context_shards/no_refine_${name#events_}"
  test -s "$out/grpo_context.jsonl" && continue
  PYTHONPATH=src conda run -n toolkit python src/evaluate_local_qwen_pipeline.py \
    --stage prepare-grpo-context --split test --limit 0 \
    --event-source precomputed --precomputed-events "$shard" \
    --queries-from-precomputed-events \
    --model-path models/Qwen3-4B --coarse-base-model-path models/Qwen3-4B \
    --skip-refinement --prediction-mode forecast-trace \
    --coarse-topology-mode temporal-dag --forecast-context-mode events-graph \
    --max-graph-events-in-prompt 14 --max-graph-edges-in-prompt 24 \
    --forecast-max-event-chars 100 --forecast-max-document-chars 500 \
    --max-events 16 --max-pairs 64 --coarse-batch-size 8 \
    --coarse-max-length 1024 --coarse-max-new-tokens 128 \
    --output-dir "$out"
done
```

Merge and verify:

```bash
PYTHONPATH=src conda run -n toolkit python src/merge_grpo_context_shards.py \
  --input outputs/grpo_context_shards/no_refine_*/grpo_context.jsonl \
  --output outputs/grpo_context_mirai_rule_train_no_refine_420/grpo_context.jsonl

wc -l outputs/grpo_context_mirai_rule_train_no_refine_420/grpo_context.jsonl
```

Expected: approximately 420 unique prompt rows. Never merge dev/holdout here.

## 4. SFT cold start

Stage 1 learns the 127 CAMEO code-description entries. Stage 2 learns
event/graph context to answer lists. The forecast SFT target is `answers`; trace
quality is learned later by GRPO reward and judge.

```bash
PYTHONPATH=src conda run -n toolkit python src/train_forecast_code_sft.py \
  --stage codebook --dataset datasets/MIRAI_data.zip \
  --model-path models/Qwen3-4B \
  --output-dir outputs/mirai_code_sft_codebook \
  --num-train-epochs 20 --per-device-train-batch-size 16 \
  --gradient-accumulation-steps 1 --learning-rate 5e-5 \
  --max-prompt-length 256 --max-completion-length 128 \
  --max-sequence-length 384 --logging-steps 10

PYTHONPATH=src conda run -n toolkit python src/train_forecast_code_sft.py \
  --stage forecast \
  --input outputs/grpo_context_mirai_rule_train_no_refine_420/grpo_context.jsonl \
  --dataset datasets/MIRAI_data.zip --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_code_sft_codebook/best_adapter \
  --output-dir outputs/mirai_forecast_sft_420 \
  --num-train-epochs 4 --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 --learning-rate 2e-5 \
  --max-prompt-length 2048 --max-completion-length 768 \
  --max-sequence-length 2304 --logging-steps 10
```

Use `outputs/mirai_forecast_sft_420/best_adapter` for GRPO. Do not start GRPO
from base Qwen or from the codebook-only adapter.

## 5. Judge-GRPO smoke run

Use `train_forecast_trace_grpo_judge_v5.py`. It combines:

- prompt-visible `Hxx/Rxx` graph references for judge grounding;
- truncation-tolerant judge JSON parsing;
- reward and rollout audit logs.

```bash
PYTHONPATH=src conda run -n toolkit python src/train_forecast_trace_grpo_judge_v5.py \
  --input outputs/grpo_context_mirai_rule_train_no_refine_420/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft_420/best_adapter \
  --judge-model-path models/Qwen3-4B --judge-weight 0.2 \
  --judge-max-new-tokens 384 \
  --judge-cache-path outputs/trace_judge_grpo_smoke.cache.json \
  --output-dir outputs/forecast_trace_grpo_judge_smoke \
  --max-samples 16 --num-generations 2 --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 --num-train-epochs 1 \
  --max-prompt-length 2048 --max-completion-length 384 \
  --logging-steps 1 --save-steps 8 \
  --reward-log-every 1 --sample-log-every 1 --sample-log-count 2
```

Inspect before the full run:

```bash
tail -5 outputs/forecast_trace_grpo_judge_smoke/reward_history.jsonl
tail -2 outputs/forecast_trace_grpo_judge_smoke/rollout_samples.jsonl
```

Proceed only when:

- `judge_parse_rate` is nonzero and normally close to 1;
- at least some GRPO groups have nonzero reward standard deviation;
- completions are valid JSON and are not mostly clipped;
- `error_count` remains zero;
- judge reasons refer to prompt-visible `Hxx/Rxx` evidence.

## 6. Full Judge-GRPO run

```bash
PYTHONPATH=src conda run -n toolkit python src/train_forecast_trace_grpo_judge_v5.py \
  --input outputs/grpo_context_mirai_rule_train_no_refine_420/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft_420/best_adapter \
  --judge-model-path models/Qwen3-4B --judge-weight 0.2 \
  --judge-max-new-tokens 384 \
  --judge-cache-path outputs/trace_judge_grpo_420.cache.json \
  --output-dir outputs/forecast_trace_grpo_judge_no_refine_420 \
  --max-samples 0 --min-coarse-edges 1 \
  --num-generations 4 --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 --learning-rate 5e-6 \
  --num-train-epochs 1 --max-prompt-length 2048 \
  --max-completion-length 512 --logging-steps 1 --save-steps 100 \
  --reward-log-every 1 --sample-log-every 5 --sample-log-count 2
```

The 420 prompt groups produce 1,680 sampled completions per epoch with four
generations, but this is still only 420 independent labeled queries.

Check:

```text
outputs/forecast_trace_grpo_judge_no_refine_420/training.log
outputs/forecast_trace_grpo_judge_no_refine_420/reward_history.jsonl
outputs/forecast_trace_grpo_judge_no_refine_420/rollout_samples.jsonl
outputs/forecast_trace_grpo_judge_no_refine_420/final_adapter
outputs/forecast_trace_grpo_judge_no_refine_420/metrics.json
```

## 7. Pseudo-dev and holdout

Evaluate on pseudo-dev with the same graph settings:

```bash
PYTHONPATH=src conda run -n toolkit python src/evaluate_local_qwen_pipeline.py \
  --stage evaluate --split test --limit 0 --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_dev.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B --coarse-base-model-path models/Qwen3-4B \
  --skip-refinement --forecast-base-model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_grpo_judge_no_refine_420/final_adapter \
  --policy forecast_trace_reward --prediction-mode forecast-trace \
  --forecast-temperature 0.0 --forecast-max-new-tokens 512 \
  --output-dir outputs/eval_mirai_pseudo_dev_no_refine_judge_grpo
```

After every design choice is frozen, run the identical command once with
`mirai_event_input_test.jsonl` and a separate output directory. That is a local
pseudo-holdout report, not an official benchmark test result.

## 8. Refinement decision

The current no-refinement run remains the default. A 512-sample real-Qwen /
MAVEN-ERE pilot measured candidate-edge filtering as follows:

| Metric | Keep all coarse edges | Refinement threshold 0.30 |
| --- | ---: | ---: |
| Precision | 0.5108 | 0.5420 |
| Recall | 1.0000 | 0.9726 |
| F1 | 0.6762 | 0.6961 |

Refinement has a small positive filtering effect, but it cannot recover edges
Qwen did not propose and cannot retype relations. Enable it only in a separate
ablation and require downstream pseudo-dev answer F1 or judge-score improvement.

Full refinement training/evaluation commands:

```bash
PYTHONPATH=src conda run -n toolkit python src/train_refinement.py \
  --dataset-mode maven-qwen-cache \
  --qwen-refinement-cache outputs/maven_qwen_refinement_cache/samples.jsonl \
  --limit 0 --validation-ratio 0.1 --epochs 40 \
  --hidden-dim 192 --message-steps 4 --dropout 0.12 \
  --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 \
  --keep-loss-weight 1.0 --strength-loss-weight 0.3 \
  --density-loss-weight 0.08 --keep-pos-weight auto --amp auto \
  --log-every 25 --debug-samples 2 \
  --output-dir outputs/refinement_graph_maven_qwen

PYTHONPATH=src conda run -n toolkit python src/evaluate_refinement_v2.py \
  --cache outputs/maven_qwen_refinement_cache/samples.jsonl \
  --model-path outputs/refinement_graph_maven_qwen/refinement_model.pt \
  --limit 0 --validation-ratio 0.1 --seed 42 \
  --output outputs/refinement_graph_maven_qwen/heldout_edge_metrics.json
```

For the MIRAI ablation use `--enable-refinement`,
`--refinement-keep-threshold 0.30`, `--no-completion-candidates`, and
`--refinement-topology-mode none`. Never mix refined and coarse-only contexts
inside one training run.

## 9. Increasing the dataset

There are five distinct options; only the first two are immediate.

1. **Finish all 420 pseudo-train contexts.** This replaces the incomplete
   200-context run without changing the evaluation protocol.
2. **After hyperparameters are frozen, retrain on pseudo train + dev.** This
   gives about 560 usable labeled queries while preserving the 141-query
   pseudo-holdout. Do not do this while tuning. Build and version a separate
   `train_plus_dev` event-input/context file and retrain SFT and GRPO from the
   same cold-start checkpoint.
3. **Create multiple input views for pseudo-train QueryIds.** Vary only visible
   history/graph budgets or apply bounded support-edge dropout. Keep every view
   of one QueryId in the same split. This improves robustness but does not
   create independent labels; report unique QueryIds and view count separately.
4. **Construct new temporal tasks from `MIRAI/data_final.csv`.** For cutoff
   time `t`, use only events/documents at or before `t`; label actor-pair event
   codes in `(t, t+horizon]`. Split chronologically before model training and
   deduplicate source/event IDs across windows. Call this a constructed
   pretraining set, not an official MIRAI split.
5. **Pretrain on an external CAMEO-compatible event corpus**, then fine-tune on
   MIRAI pseudo-train. Use chronological splits and keep external pretraining,
   MIRAI tuning, pseudo-dev selection, and pseudo-holdout reporting separate.

Self-generated traces may be added only as auxiliary SFT data when they have a
correct answer label, valid Hxx/Rxx references, valid JSON, and a sufficiently
high frozen-judge score. They are synthetic supervision, not new observations.

Do not increase data by duplicating `test_subset`, repeating identical contexts,
moving holdout labels into training before reporting, or counting four GRPO
rollouts as four independent examples.

## 10. Failure checks

| Symptom | Action |
| --- | --- |
| Context generation is slow | Resume from the next incomplete 25-row shard |
| Judge parse rate is zero | Use judge v5, no-think, and 384 output tokens |
| Most completions are clipped | Keep completion length 512 and improve SFT |
| Reward std is zero for all groups | Stop; inspect answer diversity and trace JSON |
| Judge says valid Hxx/Rxx refs do not exist | Confirm the v5 prompt-reference judge entrypoint |
| Refinement only lowers loss | Require held-out edge F1 and downstream dev gains |
