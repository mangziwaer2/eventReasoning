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

现在使用本地构建的 `mirai_forecast`，不再把官方 `MIRAI/test` 人工切分成训练集。三个 split 的职责固定如下：

| 数据 | 原始样本 | rule 输入 | 用途 |
| --- | ---: | ---: | --- |
| `datasets/mirai_forecast/train.jsonl` | 3584 | 3563 | SFT、GRPO 训练 |
| `datasets/mirai_forecast/dev.jsonl` | 713 | 708 | 选择 checkpoint 和超参数 |
| `datasets/mirai_forecast/holdout.jsonl` | 748 | 746 | 最终测试，只在实验结束后使用 |

`mirai_forecast` 的历史文档来自 cutoff 之前，标签来自之后 7 天窗口。`mirai_event_inputs_rule` 是对这三个 split 做冻结 rule 事件抽取后的 `event-input-v1` 文件；28 条样本因 rule 事件少于 2 个而跳过。这里的 `holdout` 就是最终 test，不能与 train 合并，也不能用 holdout 选择参数。

```bash
cd /root/autodl-tmp/eventReasoning
export PYTHONPATH=src
conda activate toolkit
```

如需从 `mirai_forecast` 原始 JSONL 重新做 rule 事件抽取，运行：

```bash
python src/build_mirai_forecast_event_inputs.py \
  --input datasets/mirai_forecast/train.jsonl \
          datasets/mirai_forecast/dev.jsonl \
          datasets/mirai_forecast/holdout.jsonl \
  --output-dir datasets/mirai_event_inputs_rule \
  --max-docs 4 --max-events 16 --max-events-per-doc 6 --min-events 2
```

只需抽取训练集时可只传 `train.jsonl`。不要把 dev/holdout 的 QueryId 放进 SFT、GRPO、judge 权重或 prompt 调参。

## 2. 生成 train 无 refinement context

粗图需要对 train 中的每条查询进行 Qwen pairwise 推理，应分片执行并可断点续跑。3563 条 train event-input 按每片 25 条约有 143 个 shard。

```bash
python src/split_event_input_shards.py \
  --input datasets/mirai_event_inputs_rule/mirai_forecast_event_input_train.jsonl \
  --output-dir outputs/mirai_forecast_event_input_train_shards_25 --shard-size 25
```

直接一次处理全部 shard。脚本会在内部合并 query_id；若发生重复 QueryId 会直接报错。`events_*.jsonl` 由 shell 展开为约 143 个输入文件。使用 `--no-capture-output` 才能实时看到 Qwen 进度：

```bash
python src/evaluate_local_qwen_pipeline.py \
  --stage prepare-grpo-context --split train --limit 0 \
  --event-source precomputed \
  --precomputed-events outputs/mirai_forecast_event_input_train_shards_25/events_*.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B --coarse-base-model-path models/Qwen3-4B \
  --skip-refinement --prediction-mode forecast-trace \
  --coarse-topology-mode temporal-dag --forecast-context-mode events-graph \
  --max-graph-events-in-prompt 14 --max-graph-edges-in-prompt 24 \
  --forecast-max-event-chars 100 --forecast-max-document-chars 500 \
  --max-events 16 --max-pairs 64 --coarse-batch-size 8 \
  --coarse-max-length 1024 --coarse-max-new-tokens 128 \
  --output-dir outputs/grpo_context_mirai_forecast_train_no_refine \
  --log-every 10
```

脚本已经直接写入正式训练 context，检查唯一 QueryId 数量：

```bash
wc -l outputs/grpo_context_mirai_forecast_train_no_refine/grpo_context.jsonl
```

预期为 3563 个唯一 QueryId。实际 QueryId 和文档来自 `--precomputed-events`；禁止把 dev/holdout context 合并进 train。
这些 `event-input-v1` 记录已内嵌 query 和 documents，因此该阶段不会再打开 `datasets/MIRAI_data.zip`；只有使用不带上下文的旧 event-input 文件时才需要 ZIP 回退。

## 3. 单阶段 Forecast SFT

不再先训练独立的 codebook adapter。一次 SFT 同时加入 MIRAI ZIP 中实际出现的 127 个完整 code-description 映射样本，以及 train 的 query+graph→answers 样本；这样直接得到可供 GRPO 使用的 forecast adapter。ZIP 这里只读取 codebook，不读取 query/news 训练样本。

```bash
python src/train_forecast_code_sft.py \
  --input outputs/grpo_context_mirai_forecast_train_no_refine/grpo_context.jsonl \
  --dataset datasets/MIRAI_data.zip --model-path models/Qwen3-4B \
  --output-dir outputs/mirai_forecast_sft_train \
  --num-train-epochs 10 --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 --learning-rate 2e-5 \
  --max-prompt-length 2048 --max-completion-length 768 \
  --max-sequence-length 2304 --logging-steps 10
```

forecast SFT 的 target 是 `answers`；同一轮 SFT 还训练 code→description 映射。先检查 `sample_generations.txt` 和 dev 准确率，确认 code/description 输出稳定后，再启动 GRPO。trace 的格式、证据引用和因果合理性由后续 GRPO 的确定性 reward 与 judge 共同训练。

模型可见的 forecast 输入只包含 query、cutoff/focus actors、事件的 `trigger/mention/participants`，以及用 `Hxx/Rxx` 表示的粗图关系；不会显示 `QueryId`、原始 `event_id`、`doc`、`sent` 或 `edge_id`。这些 ID 只在内部 trajectory 中用于 reward 的证据对齐和审计，不会进入模型上下文。修改 prompt 后需要重新生成 train context，旧的 SFT/GRPO 日志不能继续使用。

## 4. 唯一 Judge-GRPO 入口

正式训练只使用：

```text
src/train_forecast_trace_grpo_judge.py
```

该入口包含 frozen Qwen judge、prompt 可见的 `Hxx/Rxx` 引用对齐、code-description 语义一致性 judge、judge cache、截断 JSON 的标量恢复，以及 reward/rollout 审计日志。默认 trace judge 是 `/no_think`，`judge_weight=0.2`；description judge 使用 `description_weight=0.05` 和 96 token 上限，不要求 description exact match。`src/train_forecast_trace_grpo.py` 仅是被该入口复用的内部 GRPO trainer，不作为独立训练命令。


无 refinement 的完整 judge-GRPO：

```bash
python src/train_forecast_trace_grpo_judge.py \
  --input outputs/grpo_context_mirai_forecast_train_no_refine/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft_train/best_adapter \
  --judge-model-path models/Qwen3-4B --judge-weight 0.2 \
  --description-weight 0.05 --description-max-new-tokens 96 \
  --codebook-dataset-path datasets/MIRAI_data.zip \
  --judge-max-new-tokens 384 \
  --judge-cache-path outputs/trace_judge_mirai_forecast_train_no_refine.cache.json \
  --output-dir outputs/forecast_trace_grpo_judge_mirai_forecast_train_no_refine \
  --max-samples 0 --min-coarse-edges 1 \
  --num-generations 4 --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 --learning-rate 5e-6 \
  --num-train-epochs 1 --max-prompt-length 2048 \
  --max-completion-length 512 --logging-steps 1 --save-steps 100 \
  --reward-log-every 1 --sample-log-every 5 --sample-log-count 2
```

## 5. Refinement 训练和带 refinement 的 Judge-GRPO

当前 refinement 使用 graph-editor-v4：对候选边联合预测 keep/drop、关系类型和 strength，并可对 completion candidates 补边。旧的 graph-editor-v3 checkpoint 和 v2 cache 不兼容，必须重建。

| 指标 | 保留全部粗边 | refinement threshold=0.30 |
| --- | ---: | ---: |
| Precision | 0.5108 | 0.5420 |
| Recall | 1.0000 | 0.9726 |
| F1 | 0.6762 | 0.6961 |

若 `outputs/maven_qwen_refinement_cache_v3/samples.jsonl` 不存在，先用冻结 Qwen 构建完整 v3 cache。MAVEN-ERE 只用于 refinement 监督，不与 `mirai_forecast` 的 train/dev/holdout 混合：

```bash
python src/build_maven_qwen_refinement_cache.py \
  --dataset datasets/MAVEN_ERE.zip --split train --limit 0 \
  --base-model-path models/Qwen3-4B --coarse-topology-mode temporal-dag \
  --max-events 16 --max-pairs 64 --coarse-batch-size 8 \
  --coarse-max-length 1024 --coarse-max-new-tokens 128 \
  --negative-completion-ratio 0.75 --max-completion-edges 128 \
  --output-dir outputs/maven_qwen_refinement_cache_v3 \
  --overwrite
```

该命令默认不启用 `--coarse-thinking`，可避免 pair JSON 因思考文本截断；cache 已存在且 `cache_manifest.json` 的 `complete=true` 时不要重复生成。

训练和评估 refiner：

```bash
python src/train_refinement.py \
  --dataset-mode maven-qwen-cache \
  --qwen-refinement-cache outputs/maven_qwen_refinement_cache_v3/samples.jsonl \
  --limit 0 --validation-ratio 0.1 --epochs 40 \
  --hidden-dim 192 --message-steps 4 --dropout 0.12 \
  --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 \
  --keep-loss-weight 1.0 --strength-loss-weight 0.3 \
  --relation-loss-weight 0.5 --density-loss-weight 0.08 \
  --keep-pos-weight auto --amp auto \
  --log-every 25 --debug-samples 2 \
  --output-dir outputs/refinement_graph_maven_qwen_v4

python src/evaluate_refinement_v2.py \
  --cache outputs/maven_qwen_refinement_cache_v3/samples.jsonl \
  --model-path outputs/refinement_graph_maven_qwen_v4/refinement_model.pt \
  --limit 0 --validation-ratio 0.1 --seed 42 \
  --output outputs/refinement_graph_maven_qwen_v4/heldout_edge_metrics.json
```

生成 train 的带 refinement context。此处处理 3563 条 train；若中断，复用第 2 节分片方法，把 `--skip-refinement` 替换为下面的 refinement 参数：

```bash
python src/evaluate_local_qwen_pipeline.py \
  --stage prepare-grpo-context --split train --limit 0 \
  --event-source precomputed \
  --precomputed-events outputs/mirai_forecast_event_input_train_shards_25/events_*.jsonl \
  --queries-from-precomputed-events \
  --model-path models/Qwen3-4B --coarse-base-model-path models/Qwen3-4B \
  --refinement-model-path outputs/refinement_graph_maven_qwen_v4/refinement_model.pt \
  --enable-refinement --include-completion-candidates \
  --max-completion-edges 128 \
  --refinement-keep-threshold 0.50 --refinement-topology-mode temporal-dag \
  --prediction-mode forecast-trace --coarse-topology-mode temporal-dag \
  --forecast-context-mode events-graph --max-graph-events-in-prompt 14 \
  --max-graph-edges-in-prompt 24 --forecast-max-event-chars 100 \
  --max-events 16 --max-pairs 64 --coarse-batch-size 8 \
  --coarse-max-length 1024 --coarse-max-new-tokens 128 \
  --output-dir outputs/grpo_context_mirai_forecast_train_with_refinement
```

带 refinement 的 GRPO 仍使用同一个 judge 入口：

```bash
python src/train_forecast_trace_grpo_judge.py \
  --input outputs/grpo_context_mirai_forecast_train_with_refinement/grpo_context.jsonl \
  --model-path models/Qwen3-4B \
  --adapter-path outputs/mirai_forecast_sft_train/best_adapter \
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

两组实验必须使用不同 output-dir、judge cache 和 final adapter，不能混合 coarse/refined context。先在 dev 上比较 answer F1、judge trace score 和 reward 方差；参数冻结后只在 holdout（最终 test）上报告一次。若 refinement 没有 downstream 增益，后续训练保持无 refinement。

先在 dev 上生成预测并保存 trace，用于选择最终 adapter。GRPO 的最终 adapter 名称是 `final_adapter`：

```bash
python src/evaluate_local_qwen_pipeline.py \
  --stage evaluate --split dev --limit 0 --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_forecast_event_input_dev.jsonl \
  --queries-from-precomputed-events --model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_grpo_judge_mirai_forecast_train_no_refine/final_adapter \
  --skip-refinement --prediction-mode forecast-trace \
  --forecast-context-mode events-graph --output-dir outputs/eval_mirai_forecast_dev_no_refine

python src/evaluate_local_qwen_pipeline.py \
  --stage evaluate --split dev --limit 0 --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_forecast_event_input_dev.jsonl \
  --queries-from-precomputed-events --model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_grpo_judge_with_refinement/final_adapter \
  --refinement-model-path outputs/refinement_graph_maven_qwen_v4/refinement_model.pt \
  --enable-refinement --include-completion-candidates \
  --max-completion-edges 128 --refinement-keep-threshold 0.50 \
  --refinement-topology-mode temporal-dag \
  --prediction-mode forecast-trace --forecast-context-mode events-graph \
  --output-dir outputs/eval_mirai_forecast_dev_with_refinement
```

评估输出中的 `predictions.jsonl` 用于离线计算 answer F1。确认 dev 配置后，在 holdout（最终 test）上只运行一次：

```bash
python src/evaluate_local_qwen_pipeline.py \
  --stage evaluate --split holdout --limit 0 --event-source precomputed \
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_forecast_event_input_holdout.jsonl \
  --queries-from-precomputed-events --model-path models/Qwen3-4B \
  --forecast-adapter-path outputs/forecast_trace_grpo_judge_mirai_forecast_train_no_refine/final_adapter \
  --skip-refinement --prediction-mode forecast-trace \
  --forecast-context-mode events-graph --output-dir outputs/eval_mirai_forecast_holdout_no_refine
```

不要用 holdout 结果选择超参数、checkpoint 或 refinement threshold。

## 6. 如何增加数据量

1. 先完成 train context 和训练；dev 只用于调参，holdout 只用于最终测试。
2. 如需扩大 train，只能从新的时间窗口重新构建 train，并重新生成 split manifest；不能把 dev/holdout 追加到训练集。
3. 可对 train QueryId 做有限的事件/边预算变化或 support-edge dropout。所有 view 必须留在同一 split，并同时报告 unique QueryId 数与 view 数；它们不是独立标签。
4. 可先用外部 CAMEO 兼容语料预训练，再用 `mirai_forecast/train.jsonl` 微调；外部预训练、train 调优、dev 选择和 holdout 报告必须分开。

不能通过重复样本、复制相同 context、把 dev/holdout 标签提前放进训练，或把 4 个 GRPO generations 当作 4 个独立查询来增加数据。

## 7. 训练检查

| 现象 | 处理 |
| --- | --- |
| context 生成慢 | 按 25 条分片续跑，不重跑已完成分片 |
| judge parse rate 为 0 | 确认使用唯一 judge 入口、`--judge-max-new-tokens 384`、不开 think |
| completion 大量截断 | 保持 `--max-completion-length 512`，先改善 SFT |
| 所有 group reward std 为 0 | 停止完整训练，检查答案多样性和 JSON 格式 |
| judge 认为 Hxx/Rxx 不存在 | 检查 context 和统一 judge 入口 |
| trace 复述可见历史事件 | 查看 \`historical_copy_penalty\`；历史事件只能作为 \`Hxx\` 证据，\`intermediate_events\` 必须是不同的未来假设 |
| refinement 只降低 loss | 必须同时要求 held-out edge F1 和 dev downstream 提升 |
