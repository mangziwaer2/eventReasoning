# 文档索引

## 总纲

- [项目书](项目书.md)：研究目标、完整流程、输入输出、创新点、评测和论文路线。
- [结构化未来事件轨迹方案](结构化未来事件轨迹方案.md)：新的主线方案，说明 `forecast_trace`、闭集预测、RL 奖励和现有模块如何保留与改造。
- [Forecast Trace Pipeline](forecast_trace_pipeline.md)：具体流程、样本级输入输出、训练方法、RL 奖励和评测指标。

## 操作

- [训练操作手册](训练操作手册.md)：本地检查、云端训练、续训和评测命令。
- [事件输入规范](事件输入规范.md)：`event-input-v1`、Gold-event/Frozen-extractor 设置和防泄漏要求。
- [Coarse Graph Qwen Training](coarse_graph_qwen_training.md)：粗图 Qwen LoRA 的数据与日志说明。

## 模型细节

- [Refinement 模型结构说明](refinement模型结构说明.md)：图特征、消息传递、损失、参数和日志字段。

`项目书.md` 是唯一研究总纲。专项文档解释实现细节和阶段方案，不单独改变项目目标或创新点。