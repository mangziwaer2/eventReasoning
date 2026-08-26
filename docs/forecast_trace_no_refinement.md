# No-refinement Forecast Trace

版本：v0.1

这份文档定义当前可验证的最小主线。`refinement` 暂不进入默认实验，等 no-refinement 版本验证可行后再接入。

## 当前流程

```text
event-input-v1 / MIRAI snapshot
        |
        v
历史事件输入（Gold event 或冻结抽取器）
        |
        v
事件对候选构造
        |
        v
Qwen coarse relation -> coarse causal graph
        |
        v
Qwen forecast_trace -> final_answer.event_code
        |
        v
确定性 reward / reward-weighted RL / GRPO
```

预测 prompt 只接收 cutoff 以前的 documents、历史 events 和 coarse graph。它不再把 MIRAI 的全部候选 choices 序列化到上下文中；模型直接输出数据集标签空间中的 `event_code`。数据集的 `answer_list` 只作为监督目标和 reward 输入，不能作为历史证据。

JSON 记录中暂时保留 `choices: []` 字段，是为了兼容既有 predictions、trajectory 和 reward 读取器；该字段不会进入 active forecast prompt。

## 运行入口

### 无模型 dry run

```powershell
python src/debug_no_model_pipeline.py --output outputs/debug_no_refinement.json
```

该入口使用规则 mock 完成事件对分类和预测，验证输入、图、trace、answer、trajectory 与 reward 的数据契约，不加载模型。

### 本地 Qwen 推理

```powershell
python src/evaluate_local_qwen_pipeline.py `
  --event-source precomputed `
  --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_train.jsonl `
  --coarse-base-model-path models/Qwen3-4B `
  --forecast-base-model-path models/Qwen3-4B `
  --prediction-mode forecast-trace
```

no-refinement 是默认设置。只有显式传入 `--enable-refinement` 才会启用旧的 refinement 分支；该分支目前不属于验证主线。

### 后续训练入口

- `src/train_forecast_trace_rl.py`：离线 rollout 的 reward-weighted LoRA continuation。
- `src/train_forecast_trace_grpo.py`：TRL GRPO 入口，使用同一份 forecast prompt 和 reward context。

两者都训练完整的 forecast completion，不能把 trace 和 answer 拆成两个独立样本。answer 对齐 `answer_list`，trace 通过结构、证据引用、因果连接和与 answer 的一致性获得 reward。

## 输出契约

模型输出一个 JSON 对象：

```json
{
  "forecast_trace": {
    "intermediate_events": [],
    "trace_edges": []
  },
  "final_answer": {
    "event_code": "173",
    "event": "event description",
    "confidence": 0.72
  }
}
```

`forecast_trace` 是开放式预测的主要研究对象；`final_answer.event_code` 是当前闭集训练与评测的可拟合目标。解析器仍兼容旧的 `choice_id`，但 active prompt 不再要求或提供它。

## 文件边界

| 作用 | 当前文件 |
| --- | --- |
| 预测 prompt 与引用编号 | `src/forecast_trace_prompt.py` |
| forecast JSON 解析 | `src/forecast_trace_schema.py` |
| no-refinement 本地推理 | `src/evaluate_local_qwen_pipeline.py` |
| no-model 契约检查 | `src/debug_no_model_pipeline.py` |
| reward 与 trajectory hook | `src/rl_pipeline_hooks.py`, `src/forecast_trace_grpo_rewards.py` |
| 离线 RL | `src/train_forecast_trace_rl.py` |
| GRPO | `src/train_forecast_trace_grpo.py` |
| 可选、暂不默认 | `src/refinement_model.py`, `src/run_refinement.py`, 以及评估脚本中的 refinement 分支 |

所有 active prompt 的静态模板、输入变量和输出格式见 [Prompt 模板清单](forecast_trace_prompt_templates.md)。
