# Refinement 模型结构说明

本文档说明当前 `refinement` 阶段的任务定义、模型结构、计算流程、训练参数，以及日志中常见指标的含义。

## 1. 任务定位

当前项目分成两个图建模阶段：

1. 粗图生成阶段：Qwen 或其他粗图生成器对多个事件对进行关系判断，经过多轮事件对预测后合成一张完整的 coarse causal graph。
2. 图级 refinement 阶段：输入完整 coarse graph，输出 refined causal graph，用于后续 LLM 事件预测。

因此 refinement 不是单独的事件对分类任务。它接收整张图，在图上下文中同时判断所有候选边：

- 保留还是删除 coarse edge
- 是否补充 coarse graph 漏掉的 candidate edge
- 修正边的 relation type
- 重估边强度 strength
- 评估 frontier node，供后续预测阶段使用

## 2. 日志指标说明

### AMP

`amp=True` 表示启用了 Automatic Mixed Precision，即自动混合精度训练。

在 4090 这类 GPU 上，AMP 会让大部分矩阵计算使用 FP16，部分数值敏感计算仍使用 FP32。好处是：

- 显存占用更低
- 训练速度更快
- 可以容纳更大的图或更大的 hidden_dim

训练脚本默认 `--amp auto`，当设备是 CUDA 时自动开启。若遇到 CUDA 半精度相关错误，可以临时使用：

```bash
--amp off
```

当前模型中的 segmented softmax 已做 dtype 兼容处理，可以在 AMP 下运行。

### pos_ratio

`pos_ratio` 是 keep/drop 任务中的正样本比例：

```text
pos_ratio = gold_keep_edges / candidate_edges
```

其中 `candidate_edges` 不只包含 coarse graph 原有边，也包含 refinement 数据管线构造出的 completion candidates 和负例候选。

例如日志中：

```text
train_edges=331888 | pos_ratio=0.505
```

表示训练集中共有 331888 条候选边，其中约 50.5% 是 gold refined graph 中应该保留或补充的边。它不是 relation type 分布；relation type 分布看 `relation_counts`。

`pos_ratio` 会影响 `keep_pos_weight=auto` 的结果。若正样本很少，BCE loss 会自动提高正样本权重，避免模型倾向于全部预测 drop。

## 3. 输入格式

模型类为：

```python
TemporalRelationalEdgeRefiner
```

一次 forward 接收一张图：

```python
outputs = model(
    node_features=node_features,
    edge_index=edge_index,
    edge_features=edge_features,
    query_features=query_features,
)
```

### node_features: `[N, 10]`

每个事件节点 10 维：

1. participants_count
2. token_count
3. sentence_index
4. event confidence
5. query_overlap
6. trigger_length
7. normalized event time
8. event time observed flag
9. is_bridge_hypothesis
10. is_title_event

### edge_index: `[E, 2]`

每行是候选边：

```text
[source_node_index, target_node_index]
```

候选边来源包括：

- coarse graph 原有边
- gold 中存在但 coarse graph 漏掉的 completion candidate
- 非 gold 的 completion negative candidate

### edge_features: `[E, 20]`

每条候选边 20 维：

1. coarse_score
2. coarse_relation_id
3. temporal_score
4. entity_overlap_score
5. lexical_support_score
6. marker_score
7. query_alignment_score
8. source_time_value
9. target_time_value
10. delta_time_value
11. abs_delta_time_value
12. sentence_gap
13. is_cross_document
14. source_out_degree
15. source_in_degree
16. target_out_degree
17. target_in_degree
18. same_sentence
19. is_coarse_edge
20. is_completion_candidate

第 19、20 维很重要：模型可以区分“粗图已有边”和“为了补漏边加入的候选边”。

### query_features: `[6]`

1. query focus entity count
2. document count
3. event count
4. candidate edge count
5. cutoff time value
6. cutoff time present flag

## 4. 模型结构

默认配置：

```text
node_dim=10
edge_dim=20
query_dim=6
hidden_dim=192
num_message_passing_steps=4
num_relations=4
dropout=0.12
```

relation type 当前为：

```text
precedes, causes, escalates, mitigates
```

### 4.1 编码层

模型先把节点、边、query 编码到统一 hidden space。

节点编码：

```text
node_features [N, 10] -> node_states [N, H]
```

Query 编码：

```text
query_features [6] -> query_state [H]
```

边编码分三部分：

1. 去掉 relation_id 后的 edge scalar features
2. relation embedding
3. continuous time encoding

时间编码器使用：

```text
linear(time_features) + sin(periodic(time_features))
```

这样同时表达线性时间差和周期性时间模式。

最后拼接：

```text
edge_scalar_state + time_state + relation_embedding -> edge_state [E, H]
```

### 4.2 Graph Context

每轮 message passing 前后都会构造一个图级状态：

```text
graph_state = MLP(mean(node_states), mean(edge_states), query_state)
```

它让每条边和每个节点都能看到整张图的整体背景，而不是只看局部事件对。

### 4.3 节点消息传递

每轮 message passing 同时做正向和反向传播。

正向消息：

```text
source -> target
```

反向消息：

```text
target -> source
```

每条边根据 coarse relation id 使用 relation-specific linear transform。也就是说 `causes`、`precedes` 等关系会有不同的变换参数。

注意力权重按目标节点分组做 softmax：

```text
同一个 target node 的入边竞争注意力权重
```

反向消息同理，按 source node 分组。

聚合后使用：

```text
GRUCell + residual + LayerNorm
```

更新节点状态。

### 4.4 边状态更新

每轮节点更新之后，模型也更新边状态。

边更新输入包括：

- source node state
- target node state
- abs(source - target)
- old edge state
- time state
- query state
- graph state

然后通过：

```text
edge_update_input MLP -> GRUCell -> residual LayerNorm
```

得到新的 edge_state。

这一步是当前 refinement 和普通事件对分类的关键区别：边不是孤立判断，而是在整张图的节点和其他边更新后再被重估。

### 4.5 输出头

每条边最终构造：

```text
source_state
target_state
abs(source_state - target_state)
source_state * target_state
edge_state
time_state
query_state
graph_state
```

拼接后进入三个边级 head：

1. `keep_head`: 输出 keep logit
2. `type_head`: 输出 relation type logits
3. `strength_head`: 输出 0 到 1 的 edge strength

keep logit 还加入 coarse prior：

```text
edge_keep_logit =
    keep_head(edge_context)
    + 0.5 * logit(coarse_score)
    + source_bias
```

其中：

```text
source_bias = 0.35 * is_coarse_edge - 0.25 * is_completion_candidate
```

这样做的原因是：

- coarse graph 原有边应作为较可信先验，而不是完全从零判断
- completion candidate 应该更谨慎，避免模型过度补边
- 最终仍由 GNN 输出修正该先验

模型还输出：

```text
frontier_scores [N]
```

用于后续 graph-conditioned forecasting 阶段选择更可能指向未来发展的节点。

## 5. 训练目标

训练 loss 由四部分组成：

```text
loss =
  1.0 * keep_loss
  + 0.7 * type_loss
  + 0.3 * strength_loss
  + 0.08 * density_loss
```

### keep_loss

使用 `BCEWithLogitsLoss`，监督每条候选边是否属于 refined graph。

若 `--keep-pos-weight auto`，脚本会根据正负样本比例自动设置正样本权重。

### type_loss

只在 gold positive edges 上计算 `CrossEntropyLoss`，监督 relation type。

默认：

```text
type_label_smoothing=0.02
type_class_weighting=auto
```

这样可以缓解 relation type 分布不均衡。当前 MAVEN 小样本里常见 `precedes` 远多于 `causes`，`escalates/mitigates` 可能为 0。

### strength_loss

只在 gold positive edges 上计算 `SmoothL1Loss(beta=0.08)`，监督边强度。

SmoothL1 比 MSE 对异常 score 更稳。

### density_loss

图密度约束：

```text
density_loss = (mean(pred_keep_prob) - mean(gold_keep_label))^2
```

它约束 refined graph 的边密度接近 gold graph，避免模型把候选边全部保留或全部删除。

## 6. 默认训练参数及理由

推荐 4090 起步命令：

```bash
python src/train_refinement.py \
  --dataset-mode maven \
  --limit 8192 \
  --max-events 16 \
  --epochs 40 \
  --output-dir outputs/refinement_graph_4090
```

主要默认参数：

```text
hidden_dim=192
message_steps=4
dropout=0.12
lr=3e-4
weight_decay=1e-4
grad_clip=1.0
negative_completion_ratio=0.75
max_completion_edges=0
validation_ratio=0.1
amp=auto
```

### hidden_dim=192

当前模型约 3.67M 参数，足够表达图级关系模式，同时不会对 4090 造成明显压力。

若训练集扩大很多或加入文本 embedding，可以提高到 256。若显存紧张，可降到 128。

### message_steps=4

4 轮消息传递可以覆盖多跳局部因果链。例如：

```text
A -> B -> C -> D
```

边 `A -> D` 的判断可以间接受到中间链路影响。少于 2 轮容易退化成局部边判断；太多轮在当前小图上可能过平滑。

### dropout=0.12

MAVEN 转换出的 graph supervision 有噪声，且 relation 分布不均衡。适度 dropout 可以减少对 coarse score 或局部模式的过拟合。

### lr=3e-4 + AdamW

图模型不是大语言模型微调，参数量较小。`3e-4` 对 AdamW 是较稳的起点；训练脚本还有 ReduceLROnPlateau，会在验证 loss 不再改善时自动降学习率。

### grad_clip=1.0

图大小和边数差异较大，梯度范数可能波动。clip 到 1.0 可以避免偶发 batch 导致训练不稳定。

### negative_completion_ratio=0.75

每张图除了 coarse edges 和 missing gold edges，还加入约 `0.75 * gold_edges` 的非 gold completion candidates。

这样模型不仅学习删错边，还学习“什么时候不要补边”。

### max_events=16

当前 refinement 是整图训练，候选边数量随事件数增长较快。`max_events=16` 是初期实验的折中：图足够有结构，又能保持训练速度和日志可读性。

## 7. 粗图到精图的例子

假设输入 coarse graph 有 4 个事件：

```text
E1: Protesters gather in capital
E2: Police deploy to the square
E3: Clashes break out overnight
E4: Government announces curfew
```

粗图边：

```text
E1 -> E2 | precedes | score=0.72
E2 -> E3 | causes   | score=0.64
E1 -> E4 | causes   | score=0.31
```

数据管线还可能加入 completion candidate：

```text
E3 -> E4 | candidate | score=0.44
```

模型输出：

```text
E1 -> E2 | keep=0.88 | pred=precedes | strength=0.80
E2 -> E3 | keep=0.91 | pred=causes   | strength=0.86
E1 -> E4 | keep=0.19 | pred=causes   | strength=0.35
E3 -> E4 | keep=0.67 | pred=causes   | strength=0.62
```

最终 refined graph：

```text
KEEP: E1 -> E2 | precedes | 0.80
KEEP: E2 -> E3 | causes   | 0.86
DROP: E1 -> E4
ADD : E3 -> E4 | causes   | 0.62
```

这就是 refinement 的核心作用：不是重新做单个事件对分类，而是在整张 coarse graph 中重估结构，删除低可信边，修正关系，补充可能漏掉的因果链。

## 8. Epoch 结束时的可读日志

训练脚本会在每个 epoch 后随机抽样验证集样本，打印粗图到 refinement 的变化过程，并保存到：

```text
outputs/.../debug_readable.log
```

示例：

```text
[epoch 001] refinement debug sample=maven_...
query: ...
graph: events=12 coarse_edges=35 completion_candidates=8 gold_edges=24 -> refined_keep=21 refined_add=3 drop=14 reject=5
edges:
  #004 DROP source=coarse keep=0.172 pred=precedes:0.418 coarse=causes:0.530 gold=DROP:none
       src: ...
       tgt: ...
  #031 ADD source=completion keep=0.704 pred=causes:0.661 coarse=precedes:0.440 gold=KEEP:causes
       src: ...
       tgt: ...
```

相关参数：

```bash
--debug-samples 1
--debug-max-edges 12
--debug-keep-threshold 0.5
```

如果只想安静训练：

```bash
--debug-samples 0
```

## 9. 当前限制

当前 refinement 模型只使用结构化特征，没有直接使用事件文本 embedding 或 query-text embedding。

后续可增强方向：

1. 为 event text / query text 加入 frozen encoder embedding
2. 增加 graph-text contrastive loss
3. 引入更细的 temporal relation 类型，而不是把大量 MAVEN temporal relation 合并成 `precedes`
4. 在 MIRAI forecast 目标上做下游验证，确认 refined graph 是否真正提升预测质量
