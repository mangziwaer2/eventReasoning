# Forecast Trace RL 训练流程与公式

本文记录端到端训练、推理、评估和离线 RL 继续训练流程。当前默认实验跳过 refinement，预测 prompt 不包含 choices，LoRA B 在同一个 completion 中输出 `forecast_trace` 和 `final_answer.event_code`。

## 1. 当前代码完整性

已具备：

- 事件输入：event-input-v1，输入显式事件、提及、证据和文档，见 src/event_input.py。
- 粗图推理：src/run_coarse_graph_qwen.py / src/evaluate_local_qwen_pipeline.py，输入候选事件对，输出单对关系 JSON，多次迭代组装 G_coarse。
- Refinement 训练和推理：src/train_refinement.py、src/run_refinement.py，基于 MAVEN-ERE 训练图级 keep/drop/retype/strength。
- 端到端 MIRAI 评估：src/evaluate_local_qwen_pipeline.py，默认串联 event input -> coarse -> LoRA B forecast。
- Reward 重算：src/score_forecast_trace_rewards.py。
- RL 继续训练：src/train_forecast_trace_rl.py，从 pipeline rollout 的 predictions.jsonl 训练 LoRA B。

仍未完成：

- forecastQA.zip 已在 datasets/，但当前没有 ForecastQA loader、prompt builder 或评估入口，因此还不能说已经接入 ForecastQA。
- `src/train_forecast_trace_rl.py` 是离线 reward-weighted policy optimization；在线组内采样使用 `src/train_forecast_trace_grpo.py`。

## 2. 数据集是否满足

当前最小闭环已经满足：

- MAVEN_ERE.zip：用于训练/评估 coarse/refinement 的事件关系和图结构。
- MIRAI_data.zip：用于端到端未来事件闭集预测评估和 RL rollout。
- 预抽取 MIRAI events：端到端评估使用 --event-source precomputed --precomputed-events ...，该文件需要是 event-input-v1 JSONL。

如果要做“第 t 天 trace 事件与 t 前真实事件重合给分”，还需要一个按时间对齐的未来/中间事件标注数据集。当前 MAVEN-ERE 是历史文档内事件关系数据，MIRAI 主标签是目标日期 event code，不直接提供完整的 t 前中间事件轨迹 gold。因此当前 reward 只能评价：

- final answer 是否命中 MIRAI answer list；
- trace 是否引用合法历史事件/边；
- trace 是否避免泛化事件；
- trace 是否形成从历史图到答案的桥接路径。

## 3. 输入输出格式

### 3.1 Coarse 阶段

当前按“不训练 coarse，先用 4B/frozen coarse”的方式跑。输入不是一次把所有候选事件直接给模型让它输出整图，而是：

    prompt: event-pair relation classification
    documents/title/snippet
    source_event: trigger + mention + doc + sent + context
    target_event: trigger + mention + doc + sent + context
    metadata: sentence gap / candidate prior

输出：

    {"relation_type": "none|precedes|causes|escalates|mitigates", "confidence": 0.0}

注意这里必须叫 confidence，语义是模型对所选 relation_type 的置信度。确定是 none 时也应该输出高 confidence，例如：

    {"relation_type": "none", "confidence": 0.93}

推理时对多个候选事件对重复执行这个分类器，然后保留 relation_type != none 且 confidence >= threshold 的边，得到 G_coarse。

### 3.2 Refinement 阶段（当前跳过）

输入是完整粗图 G_coarse 和可选补边候选。模型输出每条候选边的：

    keep probability
    relation type logits
    strength

推理时按 --refinement-keep-threshold 保留边，得到 G_refined。

### 3.3 LoRA B forecast 阶段

输入：

    query + observation/cutoff date + target answer date + cutoff-before documents + observed events + coarseGraph

输出严格 JSON：

    {
      "forecast_trace": {
        "intermediate_events": [
          {
            "trace_event_id": "ft_1",
            "event": {
              "trigger": "deploy",
              "mention": "security forces deploy near the capital",
              "actors": ["security forces"],
              "event_time": "2023-02-20",
              "relative_time": "t-1"
            },
            "supporting_event_ids": ["H01"],
            "supporting_edge_ids": ["R01"],
            "expected_effect": "the deployment increases pressure on organizers, making arrests more likely",
            "confidence": 0.71
          }
        ],
        "trace_edges": [
          {"source_id": "H01", "target_id": "ft_1", "relation_type": "causes", "confidence": 0.77},
          {"source_id": "ft_1", "target_id": "answer_036", "relation_type": "raises_likelihood", "confidence": 0.73}
        ]
      },
      "final_answer": {
        "event_code": "036",
        "confidence": 0.76
      }
    }

## 4. 训练与评估命令

### 4.1 端到端生成 rollout

如果 coarse 暂时不训练，直接使用 4B frozen coarse：

    python src/evaluate_local_qwen_pipeline.py ^
      --limit 8 ^
      --event-source precomputed ^
      --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_test.jsonl ^
      --model-path models/Qwen3-4B ^
      --coarse-base-model-path models/Qwen3-4B ^
      --forecast-base-model-path models/Qwen3-4B ^
      --policy forecast_trace_reward ^
      --prediction-mode forecast-trace ^
      --output-dir outputs/local_qwen_pipeline_eval

输出：

    outputs/local_qwen_pipeline_eval/predictions.jsonl
    outputs/local_qwen_pipeline_eval/metrics.json

predictions.jsonl 现在默认保存 forecast_prompt 和 forecast_system_prompt，这是 RL 训练必须的输入。如果使用 --no-save-forecast-prompts，之后不能直接做 RL。

### 4.2 重算 reward

    python src/score_forecast_trace_rewards.py ^
      --input outputs/local_qwen_pipeline_eval/predictions.jsonl ^
      --output outputs/local_qwen_pipeline_eval/predictions.rescored.jsonl ^
      --metrics-output outputs/local_qwen_pipeline_eval/reward_metrics.json ^
      --policy forecast_trace_reward

### 4.3 RL 继续训练 LoRA B

    python src/train_forecast_trace_rl.py ^
      --input outputs/local_qwen_pipeline_eval/predictions.rescored.jsonl ^
      --model-path models/Qwen3-4B ^
      --adapter-path outputs/forecast_trace_sft_lora/best_adapter ^
      --output-dir outputs/forecast_trace_rl_lora ^
      --completion-source raw ^
      --reward-source recompute ^
      --min-reward 0.0 ^
      --weighting exp ^
      --reward-baseline mean ^
      --reward-temperature 1.0 ^
      --epochs 1 ^
      --batch-size 1 ^
      --gradient-accumulation-steps 8 ^
      --lr 5e-5 ^
      --max-length 2048

如果还没有 LoRA B SFT adapter，可以去掉 --adapter-path，脚本会从 base model 新建 LoRA。

输出：

    outputs/forecast_trace_rl_lora/
      best_adapter/
      latest_adapter/
      train_config.json
      rollout_summary.json
      debug_rollout_samples.jsonl
      train_history.json
      metrics.json
      latest_training_state.pt

### 4.4 使用 RL 后的 LoRA B 重新评估

    python src/evaluate_local_qwen_pipeline.py ^
      --limit 8 ^
      --event-source precomputed ^
      --precomputed-events datasets/mirai_event_inputs_rule/mirai_event_input_test.jsonl ^
      --model-path models/Qwen3-4B ^
      --coarse-base-model-path models/Qwen3-4B ^
      --forecast-base-model-path models/Qwen3-4B ^
      --forecast-adapter-path outputs/forecast_trace_rl_lora/best_adapter ^
      --policy forecast_trace_reward ^
      --prediction-mode forecast-trace ^
      --output-dir outputs/local_qwen_pipeline_eval_rl

## 5. Reward 公式

对样本 i，模型输出为 y_i，gold 为 g_i，trajectory 为 tau_i。

总 reward：

    R_i = w_answer * A_i + T_i

其中 trace component：

    T_i =
      w_format * F_i
    + w_grounding * G_i
    + w_temporal * U_i
    + w_bridge * B_i
    - w_generic * P_generic_i
    - w_density * P_density_i

当前默认权重在 src/rl_pipeline_hooks.py 的 ForecastTraceReward：

    w_answer    = 1.0
    w_format    = 0.2
    w_grounding = 0.2
    w_temporal  = 0.2
    w_bridge    = 0.3
    w_generic   = 0.15
    w_density   = 0.15
    wrong_answer_trace_scale = 0.2

各项定义：

- A_i：final answer reward。主答案命中 gold answer list 得 1；alternative 命中得 0.5；否则 0。
- F_i：格式分。由 parsed JSON、trace 字段完整性和 `final_answer.event_code` 完整性组成。
- G_i：grounding 分。0.65 * valid_event_ref_ratio + 0.35 * valid_edge_ref_ratio。
- U_i：时间分。MIRAI 中 trace 必须位于 observation/cutoff 与 target answer date 之间；可直接输出绝对 `event_time`，或用 target-relative `t-1/t-2/...` 表示答案日前若干天。
- B_i：bridge 分。在 G_coarse + forecast_trace 上，从支持历史事件到 `answer_<event_code>` 的最佳路径置信度乘积。
- P_generic_i：泛化事件惩罚，例如 tensions rise、situation worsens。
- P_density_i：过密 trace 惩罚，防止输出很多无用中间事件和边。

如果 A_i = 0，trace component 会被截断：

    T_i = wrong_answer_trace_scale * T_i

这样可以避免模型答错最终事件时，仅靠看似漂亮的 trace 拿到过高 reward。

## 6. RL 损失公式

训练脚本对每个 rollout 样本计算 reward 权重 alpha_i。默认 --weighting exp：

    b = mean(R)
    alpha_i_raw = exp((R_i - b) / tau)
    alpha_i = clip(alpha_i_raw, alpha_min, alpha_max)
    alpha_i = alpha_i / mean(alpha)

其中：

    tau = --reward-temperature
    alpha_min = --min-weight
    alpha_max = --max-weight

LoRA B 的训练目标是加权 token NLL：

    L_i = - (1 / |y_i|) * sum_t log pi_theta(y_i,t | x_i, y_i,<t)
    L = mean_i alpha_i * L_i

也可以理解为离线 policy gradient 的稳定近似：

    maximize mean_i alpha_i * log pi_theta(y_i | x_i)

其中 x_i 是 query + documents + observed events + coarseGraph prompt，y_i 是包含 trace 和 answer 的完整 forecast JSON。

## 7. 下一步建议

建议顺序：

1. 先用 frozen 4B coarse 跑小样本端到端，确认 predictions.jsonl 中 forecast prompt、trace、reward 都正常。
2. 用 score_forecast_trace_rewards.py 看 reward breakdown，重点检查 invalid refs、generic events、答案越界。
3. 用 train_forecast_trace_rl.py 对 LoRA B 做 1 epoch 小规模继续训练。
4. 用 RL adapter 重跑 MIRAI eval，对比 code_hit_rate、average_reward、valid_event_ref_ratio、generic_penalty。
5. 如果 RL/GRPO 后格式稳定且 reward 提升，再接入 refinement 并做严格增量对照。
