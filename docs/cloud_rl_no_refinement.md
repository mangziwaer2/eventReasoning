# Cloud Judge-GRPO 训练手册

本文档是当前唯一推荐的训练手册。GRPO 始终使用 frozen Qwen judge；通过生成不同的 context，比较不带 refinement 和带 refinement 的两条路线。

```text
历史事件 -> frozen Qwen3 粗因果图 -> (可选 refinement) -> GRPO context
                                                       |
                                          LoRA B 采样 trace + answers
                                                       |
                            确定性 reward + frozen Qwen judge -> GRPO 更新
```

`grpo_context.jsonl` 只保存 prompt 和固定图环境，不保存预测结果或 reward。LoRA B 在 `GRPOTrainer` 内在线采样，judge 仅以冻结推理的方式提供 process reward。

## 1. 数据边界

公开 `MIRAI_data.zip` 只有 `test` 和 `test_subset`，没有官方 train split：

| 数据 | 数量 | 用途 |
| --- | ---: | --- |
| `MIRAI/test/relation_query.csv` | 705 | 唯一公开查询来源 |
| `MIRAI/test_subset/relation_query.csv` | 100 | 完全包含在 705 条 test 中，不能当新数据 |
| 本地 pseudo train | 420 | SFT 和 GRPO |
| 本地 pseudo dev | 140 | 调参、选择 checkpoint |
| 本地 pseudo holdout | 141 | 最终本地报告 |

420/140/141 是固定 seed-42、60/20/20 划分；4 条查询因规则抽取到的历史事件不足而跳过。所有结果都必须标注为“官方 test 上构造的本地 pseudo split”，不能称作官方 train/test 泛化结果。

```bash
cd /root/autodl-tmp/eventReasoning
export PYTHONPATH=src
conda activate toolkit
```

只在需要重建事件输入时运行：

```bash
PYTHONPATH=src conda run -n toolkit python src/build_mirai_event_inputs.py \
  --dataset datasets/MIRAI_data.zip --source-split test \
  --seed 42 --train-ratio 0.6 --dev-ratio 0.2 \
  --max-docs 4 --max-events 16 --max-events-per-doc 6 --min-events 2 \
  --output-dir datasets/mirai_event_inputs_rule
```

不要把 pseudo dev/holdout 的 QueryId 放进 SFT、GRPO、judge 权重或 prompt 调参。

## 2. 生成 420 条无 refinement context

之前只有 200 条是 context 生成任务被限制或中断的结果，不是数据上限。粗图需要对每条查询进行 Qwen pairwise 推理，应分片执行并可断点续跑。

```bash
PYTHONPATH=src conda run -n toolkit python src/split_event_input_shards.py \
  --input datasets/mirai_event_inputs_rule/mirai_event_input_train.jsonl \
  --output-dir outputs/mirai_event_input_train_shards_25 --shard-size 25
```

直接一次处理全部 shard。脚本会在内部合并 query_id；若发生重复 QueryId 会直接报错。`events_*.jsonl` 由 shell 展开为 17 个输入文件。使用 `--no-capture-output` 才能实时看到 Qwen 进度：

```bash
PYTHONPATH=src conda run --no-capture-output -n toolkit python src/evaluate_local_qwen_pipeline.py \
  --stage prepare-grpo-context --split test --limit 0 \
  --event-source precomputed \
  --precomputed-events outputs/mirai_event_input_train_shards_25/events_*.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B --coarse-base-model-path models/Qwen3-4B \
  --skip-refinement --prediction-mode forecast-trace \
  --coarse-topology-mode temporal-dag --forecast-context-mode events-graph \
  --max-graph-events-in-prompt 14 --max-graph-edges-in-prompt 24 \
  --forecast-max-event-chars 100 --forecast-max-document-chars 500 \
  --max-events 16 --max-pairs 64 --coarse-batch-size 8 \
  --coarse-max-length 1024 --coarse-max-new-tokens 128 \
  --output-dir outputs/grpo_context_mirai_rule_train_no_refine_420 \
  --log-every 10
```

脚本已经直接写入正式训练 context，检查唯一 QueryId 数量：

```bash
wc -l outputs/grpo_context_mirai_rule_train_no_refine_420/grpo_context.jsonl
```

预期为约 420 个唯一 QueryId。禁止把 dev/holdout context 合并进来。

## 3. SFT 冷启动

GRPO 必须从 forecast SFT adapter 开始。先训练 127 个 CAMEO codebook 条目，再训练 event/graph 到 answers：

```bash
PYTHONPATH=src conda run -n toolkit python src/train_forecast_code_sft.py \
  --stage codebook --dataset datasets/MIRAI_data.zip \
  --model-path models/Qwen3-4B --output-dir outputs/mirai_code_sft_codebook \
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

forecast SFT 的 target 是 `answers`。trace 的格式、证据引用和因果合理性由后续 GRPO 的确定性 reward 与 judge 共同训练。

## 4. 唯一 Judge-GRPO 入口

正式训练只使用：

```text
src/train_forecast_trace_grpo_judge.py
```

该入口包含 frozen Qwen judge、prompt 可见的 `Hxx/Rxx` 引用对齐、code-description 语义一致性 judge、judge cache、截断 JSON 的标量恢复，以及 reward/rollout 审计日志。默认 trace judge 是 `/no_think`，`judge_weight=0.2`；description judge 使用 `description_weight=0.05` 和 96 token 上限，不要求 description exact match。`src/train_forecast_trace_grpo.py` 仅是被该入口复用的内部 GRPO trainer，不作为独立训练命令。

先运行 16 条 smoke test：

```bash
PYTHONPATH=src conda run -n toolkit python src/train_forecast_trace_grpo_judge.py \
  --input outputs/grpo_context_mirai_rule_train_no_refine_420/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft_420/best_adapter \
  --judge-model-path models/Qwen3-4B --judge-weight 0.2 \
  --description-weight 0.05 --description-max-new-tokens 96 \
  --codebook-dataset-path datasets/MIRAI_data.zip \
  --judge-max-new-tokens 384 \
  --judge-cache-path outputs/trace_judge_smoke.cache.json \
  --output-dir outputs/forecast_trace_grpo_judge_smoke \
  --max-samples 16 --num-generations 2 --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 --num-train-epochs 1 \
  --max-prompt-length 2048 --max-completion-length 512 \
  --logging-steps 1 --save-steps 8 \
  --reward-log-every 1 --sample-log-every 1 --sample-log-count 2
```

检查 smoke 输出：

```bash
tail -5 outputs/forecast_trace_grpo_judge_smoke/reward_history.jsonl
tail -2 outputs/forecast_trace_grpo_judge_smoke/rollout_samples.jsonl
```

仅在 judge parse rate 非零、completion 没有大量截断、部分 group 的 reward std 非零时，启动完整 GRPO。

无 refinement 的完整 judge-GRPO：

```bash
PYTHONPATH=src conda run -n toolkit python src/train_forecast_trace_grpo_judge.py \
  --input outputs/grpo_context_mirai_rule_train_no_refine_420/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft_420/best_adapter \
  --judge-model-path models/Qwen3-4B --judge-weight 0.2 \
  --description-weight 0.05 --description-max-new-tokens 96 \
  --codebook-dataset-path datasets/MIRAI_data.zip \
  --judge-max-new-tokens 384 \
  --judge-cache-path outputs/trace_judge_no_refine_420.cache.json \
  --output-dir outputs/forecast_trace_grpo_judge_no_refine_420 \
  --max-samples 0 --min-coarse-edges 1 \
  --num-generations 4 --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 --learning-rate 5e-6 \
  --num-train-epochs 1 --max-prompt-length 2048 \
  --max-completion-length 512 --logging-steps 1 --save-steps 100 \
  --reward-log-every 1 --sample-log-every 5 --sample-log-count 2
```

## 5. Refinement 训练和带 refinement 的 Judge-GRPO

当前 refinement 只过滤 Qwen 已提出的边，不能补回 Qwen 漏边，也不修改 relation 类型。已有 held-out candidate-edge pilot：

| 指标 | 保留全部粗边 | refinement threshold=0.30 |
| --- | ---: | ---: |
| Precision | 0.5108 | 0.5420 |
| Recall | 1.0000 | 0.9726 |
| F1 | 0.6762 | 0.6961 |

若 `outputs/maven_qwen_refinement_cache/samples.jsonl` 不存在，先用冻结 Qwen 构建完整 cache。MAVEN-ERE 只用于 refinement 监督，不与 MIRAI pseudo split 混合：

```bash
PYTHONPATH=src conda run -n toolkit python src/build_maven_qwen_refinement_cache.py \
  --dataset datasets/MAVEN_ERE.zip --split train --limit 0 \
  --base-model-path models/Qwen3-4B --coarse-topology-mode temporal-dag \
  --max-events 16 --max-pairs 64 --coarse-batch-size 8 \
  --coarse-max-length 1024 --coarse-max-new-tokens 128 \
  --output-dir outputs/maven_qwen_refinement_cache
```

该命令默认不启用 `--coarse-thinking`，可避免 pair JSON 因思考文本截断；cache 已存在且 `cache_manifest.json` 的 `complete=true` 时不要重复生成。

训练和评估 refiner：

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

生成带 refinement 的 context。此处直接处理 420 条也可以；若中断，复用第 2 节分片方法，把 `--skip-refinement` 替换为下面的 refinement 参数：

```bash
PYTHONPATH=src conda run --no-capture-output -n toolkit python src/evaluate_local_qwen_pipeline.py \
  --stage prepare-grpo-context --split test --limit 0 \
  --event-source precomputed \
  --precomputed-events outputs/mirai_event_input_train_shards_25/events_*.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B --coarse-base-model-path models/Qwen3-4B \
  --refinement-model-path outputs/refinement_graph_maven_qwen/refinement_model.pt \
  --enable-refinement --no-completion-candidates \
  --refinement-keep-threshold 0.30 --refinement-topology-mode none \
  --prediction-mode forecast-trace --coarse-topology-mode temporal-dag \
  --forecast-context-mode events-graph --max-graph-events-in-prompt 14 \
  --max-graph-edges-in-prompt 24 --forecast-max-event-chars 100 \
  --max-events 16 --max-pairs 64 --coarse-batch-size 8 \
  --coarse-max-length 1024 --coarse-max-new-tokens 128 \
  --output-dir outputs/grpo_context_mirai_rule_train_with_refinement
```

带 refinement 的 GRPO 仍使用同一个 judge 入口：

```bash
PYTHONPATH=src conda run -n toolkit python src/train_forecast_trace_grpo_judge.py \
  --input outputs/grpo_context_mirai_rule_train_with_refinement/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft_420/best_adapter \
  --judge-model-path models/Qwen3-4B --judge-weight 0.2 \
  --description-weight 0.05 --description-max-new-tokens 96 \
  --codebook-dataset-path datasets/MIRAI_data.zip \
  --judge-max-new-tokens 384 \
  --judge-cache-path outputs/trace_judge_with_refinement.cache.json \
  --output-dir outputs/forecast_trace_grpo_judge_with_refinement \
  --max-samples 0 --min-coarse-edges 1 \
  --num-generations 4 --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 --learning-rate 5e-6 \
  --num-train-epochs 1 --max-prompt-length 2048 \
  --max-completion-length 512 --logging-steps 1 --save-steps 100 \
  --reward-log-every 1 --sample-log-every 5 --sample-log-count 2
```

两组实验必须使用不同 output-dir、judge cache 和 final adapter，不能混合 coarse/refined context。只在相同 pseudo-dev 上比较 answer F1、judge trace score 和 reward 方差；若 refinement 没有 downstream 增益，后续训练保持无 refinement。

对 pseudo-dev 生成预测并保存 trace。GRPO 的最终 adapter 名称是 `final_adapter`：

```bash
PYTHONPATH=src conda run -n toolkit python src/evaluate_local_qwen_pipeline.py \
  --stage evaluate --split test --limit 0 --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_dev.jsonl \
  --queries-from-precomputed-events --model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_grpo_judge_no_refine_420/final_adapter \
  --skip-refinement --prediction-mode forecast-trace \
  --forecast-context-mode events-graph --output-dir outputs/eval_pseudo_dev_no_refine

PYTHONPATH=src conda run -n toolkit python src/evaluate_local_qwen_pipeline.py \
  --stage evaluate --split test --limit 0 --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_dev.jsonl \
  --queries-from-precomputed-events --model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_grpo_judge_with_refinement/final_adapter \
  --refinement-model-path outputs/refinement_graph_maven_qwen/refinement_model.pt \
  --enable-refinement --refinement-keep-threshold 0.30 \
  --prediction-mode forecast-trace --forecast-context-mode events-graph \
  --output-dir outputs/eval_pseudo_dev_with_refinement
```

评估输出中的 `predictions.jsonl` 用于离线计算 answer F1；不要用 pseudo-dev 结果反复选择 refinement threshold 后再报告 holdout。

## 6. 如何增加数据量

1. 先完成 420 条 pseudo-train context，这只是补齐之前不完整的 200 条。
2. 超参数冻结后，可合并 pseudo train 和 pseudo dev 形成约 560 条重新训练，但必须保留 141 条 pseudo holdout，并建立新的 split manifest。
3. 可对 train QueryId 做有限的事件/边预算变化或 support-edge dropout。所有 view 必须留在同一 split，并同时报告 unique QueryId 数与 view 数；它们不是独立标签。
4. 需要真实新任务时，从 `MIRAI/data_final.csv` 构造时间窗口：历史只允许 cutoff `t` 以前事件，标签来自 `(t,t+horizon]`，先按时间再按 actor pair 划分，并去重 URL/event ID。这是自构造预训练集，不是官方 MIRAI train。
5. 可先用外部 CAMEO 兼容语料预训练，再用 MIRAI pseudo-train 微调；外部预训练、MIRAI 调优、pseudo-dev 选择和 pseudo-holdout 报告必须分开。

不能通过重复 `test_subset`、复制相同 context、把 holdout 标签提前放进训练，或把 4 个 GRPO generations 当作 4 个独立查询来增加数据。

## 7. 训练检查

| 现象 | 处理 |
| --- | --- |
| context 生成慢 | 按 25 条分片续跑，不重跑已完成分片 |
| judge parse rate 为 0 | 确认使用唯一 judge 入口、`--judge-max-new-tokens 384`、不开 think |
| completion 大量截断 | 保持 `--max-completion-length 512`，先改善 SFT |
| 所有 group reward std 为 0 | 停止完整训练，检查答案多样性和 JSON 格式 |
| judge 认为 Hxx/Rxx 不存在 | 检查 context 和统一 judge 入口 |
| refinement 只降低 loss | 必须同时要求 held-out edge F1 和 pseudo-dev downstream 提升 |
