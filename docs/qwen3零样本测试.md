# Qwen3-4B-Thinking 零样本测试

本测试用于判断 `Qwen3-4B-Thinking-2507` 是否具备免微调完成粗图构建的能力，不改变现有 Qwen2.5 LoRA 和 refinement 训练代码。

## 环境

官方模型要求：

```text
transformers >= 4.51.0
```

安装依赖：

```bash
pip install -r requirements.txt
```

推荐把模型下载到：

```text
models/Qwen3-4B-Thinking-2507
```

也可以直接使用 Hugging Face 模型 ID，脚本只有在显式传入 `--allow-download` 时才允许下载：

```text
Qwen/Qwen3-4B-Thinking-2507
```

## 1. 自由问答

本地模型交互模式：

```bash
python src/chat_qwen3_thinking.py \
  --model-path models/Qwen3-4B-Thinking-2507 \
  --show-thinking
```

单次问答：

```bash
python src/chat_qwen3_thinking.py \
  --model-path models/Qwen3-4B-Thinking-2507 \
  --prompt "Explain why causal direction is not the same as temporal order." \
  --show-thinking
```

允许自动下载：

```bash
python src/chat_qwen3_thinking.py \
  --model-path Qwen/Qwen3-4B-Thinking-2507 \
  --allow-download
```

默认使用官方推荐的 Thinking 采样参数：

```text
temperature=0.6
top_p=0.95
top_k=20
```

Thinking 模型不使用贪心解码。交互历史只保存 final answer，不保存 thinking 内容。

## 2. 粗图零样本测试

先检查输入 prompt 和 gold graph，不加载模型：

```bash
python src/evaluate_qwen3_coarse_graph.py \
  --split valid \
  --limit 1 \
  --max-events 16 \
  --dry-run
```

正式测试 10 个样本：

```bash
python src/evaluate_qwen3_coarse_graph.py \
  --model-path models/Qwen3-4B-Thinking-2507 \
  --split valid \
  --limit 10 \
  --max-events 16 \
  --gold-scope causal \
  --output-dir outputs/qwen3_zero_shot_coarse_valid10
```

随机抽样：

```bash
python src/evaluate_qwen3_coarse_graph.py \
  --model-path models/Qwen3-4B-Thinking-2507 \
  --split valid \
  --limit 20 \
  --shuffle \
  --seed 42 \
  --gold-scope causal \
  --output-dir outputs/qwen3_zero_shot_coarse_random20
```

`--gold-scope causal` 只比较 MAVEN `CAUSE/PRECONDITION`，这是默认设置。`--gold-scope all` 还会比较当前项目映射为 `precedes` 的时间关系，任务更接近完整事件关系图而不是纯因果图。

## 输出

```text
outputs/<run>/metrics.json
outputs/<run>/predictions.jsonl
outputs/<run>/readable.log
```

`readable.log` 对每条预测边标记：

```text
[MATCH]     方向和类型正确
[MIS-TYPE]  事件方向匹配但关系类型错误
[EXTRA]     gold 中不存在
[MISS]      gold 有但模型未输出
```

主要指标：

- `typed.f1`：方向和关系类型都正确。
- `pair.f1`：只要求方向事件对正确，不要求类型正确。
- `pair.accuracy`：所有有向事件对上的存在性准确率，容易受大量 none 影响，只作辅助。
- `per_relation`：`precedes` 和 `causes` 分项结果。
- `json_parse_rate`：最终答案能否稳定解析为指定 JSON。

## 是否可以免微调

不能只看少量可读样本。建议至少在固定的 MAVEN valid 子集上比较：

1. Qwen3 zero-shot 文档级整图。
2. 当前 Qwen2.5-0.5B LoRA 逐事件对粗图。
3. 两者经过相同 refinement 后的图指标。
4. 两者最终用于未来事件预测的 MIRAI 指标。

只有 Qwen3 在 `typed F1、candidate recall、JSON 稳定性、推理成本` 和最终预测上达到可接受水平，才能决定取消粗图微调。即使免去 coarse 微调，refinement 和后续预测效用训练是否保留仍需单独实验。
