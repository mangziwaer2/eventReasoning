# Causal Graph Enhanced LLM Event Forecasting

本项目研究一个明确问题：

> 给定截止时间前的新闻文档和上游已抽取事件，构建并修正事件因果图，再让 LLM 基于该图预测未来事件。

本项目不研究或训练事件抽取器。MAVEN 使用数据集 gold events，MIRAI 使用冻结外部抽取器预计算的事件。最终评价标准是未来事件预测效果，不是事件抽取准确率或单独的图边分类准确率。

## 方法主线

```text
gold/frozen upstream extractor (outside our method)
-> query + cutoff time + documents + strict event mentions
-> candidate event pairs
-> Qwen LoRA coarse graph
-> graph-level refinement
-> refined causal graph
-> frozen LLM future event prediction
```

- 粗图模型：默认 `Qwen2.5-0.5B + LoRA`，负责高频事件对关系评分。
- Refinement：time-aware relational GNN，整图执行删边、补边、改类和强度重估。
- 最终预测模型：建议使用原生 Qwen 4B 或更大模型，读取 refined graph 后输出 Top-K 未来事件。
- 主数据集：MAVEN-ERE 用于图监督，MIRAI 用于最终未来事件预测评测。
- 对外输入：统一使用 [`event-input-v1`](docs/事件输入规范.md)，其他使用者可替换自己的事件抽取器。

完整研究设计、输入输出、创新点和论文路线见 [项目书](docs/项目书.md)。

## 仓库结构

```text
docs/
  项目书.md                         唯一研究总纲
  事件输入规范.md                   预抽取事件 schema 与实验边界
  训练操作手册.md                   训练、续训和评测命令
  coarse_graph_qwen_training.md    粗图训练细节
  refinement模型结构说明.md         Refinement 结构和参数

src/
  causal_graph.py                  图与预测数据结构
  event_input.py                   预抽取事件加载与严格校验
  event_extraction.py              事件文本处理
  event_extractor.py               兼容用可选抽取基线
  mirai_dataset.py                 MIRAI 数据读取
  coarse_graph_dataset.py          MAVEN 图、事件对和候选边数据
  train_coarse_graph_qwen.py       粗图 Qwen LoRA 训练
  run_coarse_graph_qwen.py         粗图推理与组图
  refinement_dataset.py            整图 refinement 数据
  refinement_model.py              图 refinement 模型
  train_refinement.py              Refinement 训练
  run_refinement.py                Refinement 推理
  evaluate_maven_pipeline.py       粗图与 refinement 图指标
  evaluate_local_qwen_pipeline.py  无 API 的端到端未来预测
  rl_pipeline_hooks.py             后续 RL 接口
```

`datasets/`、`models/` 和 `outputs/` 是本地大文件目录，不纳入 Git。

## 环境

Python 依赖：

```bash
pip install -r requirements.txt
```

CUDA 版本的 PyTorch 应根据云端 CUDA 环境单独安装。若 `torchao` 与 `peft` 不兼容，应升级到兼容版本或卸载未使用的 `torchao`，不要让可选量化扩展阻塞 LoRA。

## 事件输入

校验标准样例：

```bash
python src/event_input.py --input examples/event_input.example.json
```

直接从预抽取事件生成粗图：

```bash
python src/run_coarse_graph_qwen.py \
  --input-mode events \
  --input examples/event_input.example.json \
  --base-model-path models/Qwen2.5-0.5B \
  --adapter-path outputs/coarse_graph_qwen_lora_4090_full/best_adapter \
  --output outputs/example_coarse_graph.json
```

## 训练

从仓库根目录执行。

粗图 Qwen LoRA：

```bash
python src/train_coarse_graph_qwen.py \
  --train-limit 0 \
  --validation-limit 0 \
  --max-events 16 \
  --negative-ratio 1.5 \
  --epochs 10 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --output-dir outputs/coarse_graph_qwen_lora_4090_full
```

Refinement：

```bash
python src/train_refinement.py \
  --dataset-mode maven \
  --limit 0 \
  --max-events 16 \
  --epochs 40 \
  --hidden-dim 192 \
  --message-steps 4 \
  --output-dir outputs/refinement_graph_4090_full
```

云端完整参数和续训命令见 [cloud_training.txt](cloud_training.txt)。

## 评测

MAVEN 两阶段图指标：

```bash
python src/evaluate_maven_pipeline.py \
  --split test \
  --limit 0 \
  --base-model-path models/Qwen2.5-0.5B \
  --coarse-adapter-path outputs/coarse_graph_qwen_lora_4090_full/best_adapter \
  --refinement-model-path outputs/refinement_graph_4090_full/refinement_model.pt
```

无 API 的 MIRAI 未来预测小样本评测：

```bash
python src/evaluate_local_qwen_pipeline.py \
  --limit 8 \
  --event-source precomputed \
  --precomputed-events datasets/mirai_events_test.jsonl \
  --model-path models/Qwen2.5-4B \
  --coarse-base-model-path models/Qwen2.5-0.5B \
  --coarse-adapter-path outputs/coarse_graph_qwen_lora_4090_full/best_adapter \
  --refinement-model-path outputs/refinement_graph_4090_full/refinement_model.pt \
  --output-dir outputs/local_qwen_pipeline_eval
```

当前端到端入口已打通，但论文所需的 documents-only、events-only、coarse、refined、shuffled 和 oracle 对照仍需按 [项目书](docs/项目书.md) 的路线补齐。

## 当前结论

- 已完成：严格预抽取事件输入、Qwen 粗图训练/续训、整图 refinement、MAVEN 评测入口、本地 Qwen 未来预测入口。
- 尚未证明：refined graph 能显著提升 MIRAI 未来事件预测。
- 下一步 P0：运行现有 checkpoint 的完整图指标，并把 refinement 训练数据改为真实 out-of-fold coarse predictions。
