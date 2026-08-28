# Forecast Trace Prompt 模板清单

本文档只记录当前代码实际使用的 prompt。动态部分（文档正文、事件和边）由运行时填充；模板中的 `{...}` 表示变量，不是模型需要原样输出的内容。

## 1. 事件抽取

来源：`src/evaluate_local_qwen_pipeline.py:build_event_extraction_prompt`。

system：

```text
You extract structured event mentions and return strict JSON only.
```

user 模板：

```text
Extract concrete event mentions that are useful for forecasting the query.
Use only the provided documents. Do not invent events.
Return strict JSON only with this schema:
{
  "events": [
    {
      "document_id": "same id as the source document",
      "trigger": "short event trigger word or phrase",
      "event": "one concrete event mention",
      "evidence": "short source sentence or clause",
      "participants": ["entity"],
      "confidence": 0.0
    }
  ]
}
Return at most {max_events} events.

Query: {query_text}

Documents:
[Document {document_id}]
Title: {title}
Date: {publish_time}
Text: {document_text}
```

该模板只用于 `--event-source qwen`。验证主线默认使用预计算事件，因此建议先使用 Gold event 或冻结抽取器排除抽取误差。

## 2. Coarse 事件对关系

来源：`src/coarse_graph_dataset.py:EventPairSample.to_instruction_example`，由 `src/evaluate_local_qwen_pipeline.py:format_pair_prompt` 包装。

system：

```text
You classify directed relations between event pairs.
```

user 模板：

```text
You classify the relation between two candidate events.
Return strict JSON with the schema {"relation_type": ..., "confidence": ...} only.
Allowed relation_type values: none, precedes, causes, escalates, mitigates.
Use none when there is no supported directed relation from source_event to target_event.
confidence is your certainty in the selected relation_type; confident none should have high confidence.

Source Event:
id={source_event_id}
doc={document_id}
sent={sentence_index}
trigger={trigger}
event={event_text}
participants={participants}

Target Event:
id={target_event_id}
doc={document_id}
sent={sentence_index}
trigger={trigger}
event={event_text}
participants={participants}

Document Context:
{title_or_document_text}

Metadata:
{pair_metadata}
```

输出被合并为 `CoarseCausalGraph`。在 no-refinement 版本中，该 graph 直接提供给 forecast prompt。

## 3. Active Forecast Trace（no-refinement）

来源：`src/forecast_trace_prompt.py:build_structured_forecast_prompt`。

system：

```text
You output a grounded forecast_trace followed by a final_answer JSON only.
```

user 模板：

```text
You are a future event forecasting model.
Input includes query, cutoff-before documents, observed events, and a coarse causal graph.
First output a structured forecast_trace, then output a final_answer with one event_code.
No candidate choices are provided. Do not output choice_id or copy a candidate list.
The event_code is the closed-set target learned during training; output the valid code directly. Preserve actor direction: 042 means the subject travels to visit, while 043 means the subject hosts/receives the visitor; these are not interchangeable.
Use only visible historical events and coarse-graph edges as support. Do not invent historical support.
Intermediate trace events must occur strictly after the observation/cutoff date and before the target answer date; their support must point to visible events/edges. When using relative_time, measure from the target answer date: t-1 is the day before the answer; prefer absolute event_time.
Keep the trace compact, concrete, and grounded; prefer a few well-supported steps over verbose speculation.
The final trace event should explain why the final event_code is likely.
Return strict JSON only with this schema:
{
  "forecast_trace": {
    "intermediate_events": [
      {
        "trace_event_id": "ft_1",
        "event": {"trigger": "deploy", "mention": "...", "actors": [], "event_time": "YYYY-MM-DD", "relative_time": "t-1"},
        "supporting_events": [{"event_ref": "H01", "event": "copy a visible historical event"}],
        "supporting_edge_refs": ["R01"],
        "expected_effect": "specific mechanism: how this event changes the likelihood of the answer",
        "confidence": 0.0
      }
    ],
    "trace_edges": [
      {"source_ref": "H01", "target_ref": "ft_1", "relation_type": "causes", "confidence": 0.0},
      {"source_ref": "ft_1", "target_ref": "answer_<event_code>", "relation_type": "raises_likelihood", "confidence": 0.0}
    ]
  },
  "final_answer": {"event_code": "000", "event": "event description", "confidence": 0.0}
}

Invalid outputs: nonexistent event_ref/edge_ref, generic events like 'tensions rise', invalid event_code, events at/before the cutoff or at/after the answer date.

QueryId: {query_id}
Query: {query_text}
Target/Cutoff date: {cutoff_time}
Target answer date: {target_time}
Focus actors: {focus_entities}

Documents:
{documents}

Visible historical events:
- H01 | event_id={event_id} | doc={document_id} | sent={sentence_index} | trigger={trigger} | event={event_text} | participants={participants}

Coarse causal edges:
- R01 | edge_id={edge_id} | H01 -> H02 | relation={relation_type} | confidence={score}
```

关键设计是：choices 不进入上下文，answer 直接是 `event_code`；trace 与 answer 在同一个 completion 中生成，便于后续对整段 rollout 计算 reward。`Hxx` 和 `Rxx` 是 prompt 内的短引用，解析后再映射回真实 event/edge id。

## 4. Legacy Forecast（非 active）

来源：`src/evaluate_local_qwen_pipeline.py:render_refined_graph_prompt` 及其旧分支。

该模板仍保留用于回归比较，内容包含候选 choices 和扁平 `forecast_event`。它不是 no-refinement 默认路径，不应作为新实验的训练 prompt；只有传入 `--prediction-mode legacy` 时才会使用。

## 5. 训练时的使用方式

- `src/train_forecast_trace_rl.py` 读取 predictions JSONL 中保存的 `forecast_prompt` 和 `forecast_system_prompt`，训练完整 forecast completion。
- `src/evaluate_local_qwen_pipeline.py --stage prepare-grpo-context` 只生成 prompt 和固定图环境，写入 `grpo_context.jsonl`。`src/train_forecast_trace_grpo.py` 在训练时采样 completion，并将 `reward_context` 传给 `ForecastTraceGRPOReward`。
- reward 使用 `answer_list` 检查 `event_code`，同时检查 trace 的 JSON 合法性、历史事件/边引用、时间约束和 trace 到最终答案的连接。

因此训练数据可以带有标准答案，但标准答案只约束闭集 `event_code`；开放式 `forecast_trace` 不与 choices 做字符串匹配，而由结构化 reward 和后续 RL 探索约束。
