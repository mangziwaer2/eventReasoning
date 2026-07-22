# Forecast Trace Pipeline

版本：v0.1
日期：2026-07-22
状态：新主线的实现规格书

## 1. 目标

本项目最终要做的是：

> 输入 cutoff 前的文档和事件，构建 refined causal graph，再让 LLM 在图约束下输出结构化未来事件轨迹，并完成 `t` 时刻闭集事件预测。

核心输出不只是答案，而是：

```text
G_refined
-> forecast_trace
-> final_answer
```

其中：

- `G_refined`：只包含 cutoff 前已经观察到的历史事件。
- `forecast_trace`：模型预测的 `t` 之前若干中间事件，是 prospective graph expansion。
- `final_answer`：`t` 时刻闭集选择答案，是主评测对象。

`forecast_trace` 可以看成临时扩展图 `G_augmented` 的新增节点和边，但不能写回历史图 `G_refined`，否则会污染已观察事实。

## 2. 完整流程

```text
[Input]
query + target_time t + candidate choices
cutoff 前 documents
strict historical event mentions
        |
        v
[1] candidate event-pair construction
根据时间、句距、实体重叠、query 相关性构造稀疏候选事件对
        |
        v
[2] coarse graph generation
Qwen LoRA 对每个候选事件对输出 none / precedes / causes
合并得到 G_coarse
        |
        v
[3] graph refinement
GNN refiner 对完整图执行 keep/drop/add/retype/strength/frontier
得到 G_refined
        |
        v
[4] structured forecast trace generation
冻结或轻量 adapter 的 Qwen 读取 documents + events + G_refined + choices
输出 t 前中间事件和支撑边
        |
        v
[5] closed-set final prediction
从候选答案中选择 t 时刻事件
        |
        v
[6] evaluation / RL reward
主指标评估 final_answer
辅助指标评估 forecast_trace 的结构合法性、图支撑和干预敏感性
```

## 3. 数据结构

### 3.1 历史事件节点

历史事件必须来自数据集 gold event mentions 或冻结外部抽取器：

```json
{
  "event_id": "e1",
  "trigger": "protested",
  "mention": "opposition supporters protested in the capital",
  "normalized_event": "opposition supporters protest in capital",
  "participants": ["opposition supporters", "capital"],
  "document_id": "doc_001",
  "sentence_index": 3,
  "event_time": "t-5",
  "confidence": 1.0,
  "evidence": "Opposition supporters protested in the capital on Monday."
}
```

历史节点约束：

- 只能来自 cutoff 前文档。
- 不能包含未来答案或 gold label。
- 不能把整句直接当作事件，必须有 trigger 和 mention。

### 3.2 refined graph

```json
{
  "nodes": ["e1", "e2", "e3"],
  "edges": [
    {
      "edge_id": "r1",
      "source_event_id": "e1",
      "target_event_id": "e2",
      "relation_type": "causes",
      "confidence": 0.82
    }
  ],
  "frontier_nodes": [
    {"event_id": "e2", "frontier_score": 0.77}
  ]
}
```

`frontier_nodes` 表示最可能延伸到未来的历史事件，用于压缩 prompt 和引导 trace。

### 3.3 forecast trace

```json
{
  "intermediate_events": [
    {
      "trace_event_id": "ft_1",
      "node_source": "forecast_trace",
      "observed": false,
      "relative_time": "t-2",
      "event": "security forces deploy near the capital",
      "actors": ["security forces", "capital"],
      "supporting_event_ids": ["e2"],
      "supporting_edge_ids": ["r1"],
      "expected_effect": "raises likelihood of arrests or confrontation",
      "confidence": 0.68
    }
  ],
  "trace_edges": [
    {
      "source_id": "e2",
      "target_id": "ft_1",
      "relation_type": "causes",
      "confidence": 0.61
    },
    {
      "source_id": "ft_1",
      "target_id": "answer_B",
      "relation_type": "raises_likelihood",
      "confidence": 0.69
    }
  ]
}
```

`trace_event_id` 使用 `ft_*`，明确表示这是预测节点，不是历史观测节点。

### 3.4 final answer

```json
{
  "choice_id": "B",
  "event_code": "180",
  "event": "government arrests opposition members",
  "confidence": 0.74,
  "supporting_trace_event_ids": ["ft_1"],
  "supporting_event_ids": ["e2"],
  "supporting_edge_ids": ["r1"]
}
```

最终评测只根据 `choice_id` 或数据集要求的 `event_code` 计算主指标。

## 4. 样本级例子

### 4.1 输入

```json
{
  "query": {
    "subject": "Government of Country X",
    "object": "opposition movement",
    "target_time": "2024-05-10",
    "question": "What event is most likely to happen on the target date?"
  },
  "choices": [
    {"choice_id": "A", "event_code": "010", "description": "make a public statement"},
    {"choice_id": "B", "event_code": "173", "description": "arrest or detain opposition members"},
    {"choice_id": "C", "event_code": "030", "description": "provide aid"},
    {"choice_id": "D", "event_code": "190", "description": "use conventional military force"}
  ],
  "documents": [
    {
      "document_id": "doc_1",
      "published_at": "2024-05-06",
      "text": "Opposition supporters protested in the capital after the disputed election. Police warned organizers that unauthorized marches would be dispersed."
    },
    {
      "document_id": "doc_2",
      "published_at": "2024-05-08",
      "text": "The interior ministry accused opposition leaders of inciting unrest. Security forces were placed on alert near government buildings."
    }
  ],
  "events": [
    {
      "event_id": "e1",
      "trigger": "protested",
      "mention": "opposition supporters protested in the capital",
      "participants": ["opposition supporters", "capital"],
      "document_id": "doc_1",
      "sentence_index": 0
    },
    {
      "event_id": "e2",
      "trigger": "warned",
      "mention": "police warned organizers",
      "participants": ["police", "organizers"],
      "document_id": "doc_1",
      "sentence_index": 1
    },
    {
      "event_id": "e3",
      "trigger": "accused",
      "mention": "interior ministry accused opposition leaders",
      "participants": ["interior ministry", "opposition leaders"],
      "document_id": "doc_2",
      "sentence_index": 0
    },
    {
      "event_id": "e4",
      "trigger": "alert",
      "mention": "security forces were placed on alert",
      "participants": ["security forces"],
      "document_id": "doc_2",
      "sentence_index": 1
    }
  ]
}
```

### 4.2 粗图生成

候选事件对：

```text
e1 -> e2
e1 -> e3
e2 -> e4
e3 -> e4
```

Qwen coarse 输出：

```json
{
  "edges": [
    {"source": "e1", "target": "e2", "relation": "precedes", "score": 0.72},
    {"source": "e1", "target": "e3", "relation": "causes", "score": 0.58},
    {"source": "e2", "target": "e4", "relation": "causes", "score": 0.64},
    {"source": "e3", "target": "e4", "relation": "causes", "score": 0.81}
  ]
}
```

### 4.3 refinement

refinement 发现 `e1 -> e3` 证据弱，保留 `e3 -> e4`，并补充 `e1 -> e4`：

```json
{
  "edges": [
    {"edge_id": "r1", "source": "e1", "target": "e2", "relation": "precedes", "confidence": 0.69},
    {"edge_id": "r2", "source": "e2", "target": "e4", "relation": "causes", "confidence": 0.66},
    {"edge_id": "r3", "source": "e3", "target": "e4", "relation": "causes", "confidence": 0.86},
    {"edge_id": "r4", "source": "e1", "target": "e4", "relation": "causes", "confidence": 0.57}
  ],
  "frontier_nodes": [
    {"event_id": "e4", "frontier_score": 0.82},
    {"event_id": "e3", "frontier_score": 0.71}
  ]
}
```

### 4.4 forecast trace + final answer

预测 LLM 输出：

```json
{
  "forecast_trace": {
    "intermediate_events": [
      {
        "trace_event_id": "ft_1",
        "relative_time": "t-1",
        "event": "security forces move to detain protest organizers",
        "actors": ["security forces", "protest organizers"],
        "supporting_event_ids": ["e3", "e4"],
        "supporting_edge_ids": ["r3"],
        "expected_effect": "makes arrests of opposition members more likely",
        "confidence": 0.72
      }
    ],
    "trace_edges": [
      {
        "source_id": "e4",
        "target_id": "ft_1",
        "relation_type": "causes",
        "confidence": 0.67
      },
      {
        "source_id": "ft_1",
        "target_id": "answer_B",
        "relation_type": "raises_likelihood",
        "confidence": 0.74
      }
    ]
  },
  "final_answer": {
    "choice_id": "B",
    "event_code": "173",
    "event": "arrest or detain opposition members",
    "confidence": 0.76,
    "supporting_trace_event_ids": ["ft_1"],
    "supporting_event_ids": ["e3", "e4"],
    "supporting_edge_ids": ["r3"]
  },
  "open_answer": "Security forces are likely to move from alert status to detaining opposition organizers, so the most likely target-day event is arrest or detention."
}
```

### 4.5 插入到临时扩展图

评测和 RL 时构造：

```text
G_augmented = G_refined + ft_1 + answer_B
```

新增路径：

```text
e3 -> e4 -> ft_1 -> answer_B
```

这条路径可以被打分、删除、反转或与错误答案路径比较。

## 5. 训练方法

### 5.1 Phase A：监督训练粗图

数据：MAVEN-ERE gold event mentions 和 gold relations。

输入：

```text
document context + source event mention + target event mention + optional query
```

输出：

```json
{"relation_type": "none|precedes|causes", "score": 0.0}
```

训练目标：

- relation cross entropy。
- score calibration。
- 类别不均衡处理，负采样由 `--negative-ratio` 控制。

指标：

- candidate recall@K。
- typed edge precision / recall / F1。
- relation macro-F1。
- causal-only F1。
- ECE / Brier score。
- JSON parse rate。

### 5.2 Phase B：监督训练 refinement

数据：

```text
G_coarse + gold graph
```

优先使用 out-of-fold coarse prediction 生成真实训练分布；人工扰动 gold graph 只作为增强。

输出：

- keep/drop。
- add completion edge。
- relation retype。
- edge strength。
- frontier node score。

训练目标：

```text
L = keep CE
  + type CE
  + strength regression
  + density regularization
  + frontier auxiliary loss
```

指标：

- refined typed edge F1。
- refined 相对 coarse 的 Delta F1。
- add/delete/retype 分项准确率。
- cycle rate。
- temporal violation rate。
- graph density。
- frontier ranking MAP/NDCG。

### 5.3 Phase C：闭集预测 SFT

数据：MIRAI query、cutoff 前 documents、预抽取 events、构图结果、候选答案、gold final answer。

输入：

```text
query + documents + events + G_refined + choices
```

输出：

```text
<forecast_trace>...</forecast_trace>
<final_answer>...</final_answer>
```

如果没有人工 trace 标注，先采用两种可选伪标签：

- 规则 trace：从 frontier nodes 到 gold answer 候选构造最短或最高分路径。
- teacher trace：使用冻结强模型生成 trace，再由 schema validator 过滤。

SFT 目标：

- final answer token loss。
- trace schema token loss。
- choice id 强约束。

指标：

- answer accuracy。
- macro-F1。
- Hit@K。
- trace parse rate。
- support id validity。
- cutoff leakage check。

### 5.4 Phase D：RL 强化

RL 不要求人工标注 `t-1` 事件。模型生成 trace 后，把 trace 插入临时扩展图 `G_augmented`，再计算奖励。

可训练模块按风险从低到高：

```text
1. forecast small adapter / head
2. refinement threshold or policy head
3. coarse LoRA policy
4. 两阶段联合策略
```

默认冻结：

- 事件抽取器。
- 最终预测 LLM 主干。
- reward evaluator。

## 6. RL 奖励

总奖励：

```text
R = 1.00 * R_answer
  + 0.20 * R_format
  + 0.20 * R_grounding
  + 0.20 * R_temporal
  + 0.30 * R_graph_bridge
  + 0.30 * R_intervention
  - 0.15 * R_generic
  - 0.15 * R_density
```

权重是初始建议，必须在 validation 上调，不在 test 上调。

### 6.1 R_answer

```text
final_answer 正确：1.0
final_answer 错误：0.0
```

主奖励。若 `final_answer` 错误，trace 相关奖励应设置上限，例如最高只能拿到 `0.2`，防止模型生成漂亮但无效的中间事件。

### 6.2 R_format

```text
标签完整：0.4
JSON 可解析：0.4
字段齐全：0.2
```

### 6.3 R_grounding

检查 trace 是否被历史输入支撑：

```text
supporting_event_ids 存在比例
supporting_edge_ids 存在比例
actors 是否来自 query / 历史事件 / 文档实体
event 是否是具体动作，而不是空泛趋势
```

建议初始：

```text
R_grounding =
  0.35 * valid_event_id_ratio
+ 0.35 * valid_edge_id_ratio
+ 0.20 * actor_grounding_ratio
+ 0.10 * concreteness_score
```

`concreteness_score` 可先用规则近似：包含动词 trigger、参与者非空、长度不过短不过长。

### 6.4 R_temporal

```text
relative_time 必须早于 t
trace 事件顺序必须从更早到更晚
历史事件不能被写成 cutoff 后已经发生
```

### 6.5 R_graph_bridge

把 trace 插入 `G_augmented` 后，检查是否形成支持正确答案的路径：

```text
historical frontier -> trace events -> final answer
```

定义：

```text
path_score(answer) = mean(edge_confidence on best path to answer)
R_graph_bridge = path_score(gold_answer) - max path_score(wrong_answer)
```

若没有可达路径，记为负分或 0。

### 6.6 R_intervention

对 trace 引用的关键边做干预：

```text
G_augmented
G_minus = 删除 trace 支撑边
G_reverse = 反转 trace 支撑边
```

冻结预测模型重新计算正确答案概率：

```text
R_intervention =
  P_gold(G_augmented)
- max(P_gold(G_minus), P_gold(G_reverse))
```

如果删除关键边后模型答案不变且置信度不降，说明 trace 没有真正参与预测，奖励应低。

### 6.7 R_generic

惩罚空泛事件：

```text
"situation worsens"
"tensions rise"
"further developments occur"
```

这些描述没有明确 trigger、actor、effect，不能作为高质量中间事件。

### 6.8 R_density

惩罚过多 trace 节点和边：

```text
R_density = max(0, trace_event_count - K) + max(0, trace_edge_count - M)
```

初始建议：

```text
K = 3
M = 5
```

## 7. 训练与评测指标

### 7.1 图模块指标

- candidate recall@K。
- coarse typed edge F1。
- refined typed edge F1。
- refined Delta F1。
- graph density。
- cycle rate。
- temporal violation rate。

### 7.2 trace 指标

- trace parse rate。
- valid supporting event id ratio。
- valid supporting edge id ratio。
- trace temporal violation rate。
- average trace event count。
- generic event rate。
- graph bridge success rate。
- intervention sensitivity。

### 7.3 最终预测指标

- Accuracy。
- Macro-F1。
- Micro-F1。
- Hit@K。
- MRR / MAP。
- calibration ECE。
- Brier score。
- abstain coverage-risk。

最终论文主表以最终预测指标为中心，图指标和 trace 指标作为机制分析。

## 8. 必做对照

```text
Documents only
Events only
Coarse graph
Refined graph
Refined graph + forecast_trace
Shuffled graph + forecast_trace
Refined graph + random trace
Oracle / gold graph upper bound
```

判断标准：

```text
Refined graph + forecast_trace
必须显著优于 refined graph without trace
也必须优于 shuffled graph + forecast_trace
```

否则不能说明提升来自因果图约束的未来轨迹。

## 9. 后续代码任务

P0：

- 新增 `forecast_trace_schema.py`：parse、validate、pretty log。
- 扩展 `evaluate_local_qwen_pipeline.py`：支持 `--prediction-mode forecast-trace`。
- 输出 `trace_metrics.json` 或写入统一 `metrics.json`。

P1：

- 新增 `forecast_trace_rewards.py`：实现 reward breakdown。
- 新增 `forecast_trace_graph.py`：构造 `G_augmented`，支持路径和干预。
- 在 readable log 中展示 `G_refined -> forecast_trace -> final_answer`。

P2：

- 新增 MIRAI 闭集预测 SFT 样本构造。
- 支持 teacher trace / rule trace 两种伪标签。
- 实现离线 reward 计算，不先上 PPO。

P3：

- refinement policy 轻量 RL。
- coarse + refinement 联合 RL。
- 完整消融、显著性检验和错误分析。
