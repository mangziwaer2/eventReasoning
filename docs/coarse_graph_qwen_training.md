# Coarse Graph Qwen Training

This stage trains a Qwen LoRA event-pair relation classifier. It is the coarse graph generator stage:

```text
event mentions -> event-pair relation classification -> coarse causal graph
```

Event extraction is outside this method. MAVEN-ERE training reads gold event mentions from the dataset. Inference accepts externally extracted events through `event-input-v1`.

The model predicts strict JSON:

```json
{"relation_type": "causes", "score": 0.82}
```

Allowed relation values:

```text
none, precedes, causes, escalates, mitigates
```

## Event Nodes

Event nodes are strict event mentions, not arbitrary sentences.

MAVEN event nodes are built from the annotated event mention trigger and local context:

```text
trigger=attacked; mention=... forces attacked the village ...
```

The full sentence is retained in metadata and evidence, but the graph node text used by prompts and graph output is the event mention string.

## Recommended Cloud Command

For an initial 4090 run:

```bash
python src/train_coarse_graph_qwen.py \
  --model-path models/Qwen2.5-0.5B \
  --train-limit 8192 \
  --validation-limit 512 \
  --max-events 16 \
  --negative-ratio 1.5 \
  --epochs 3 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --lr 2e-4 \
  --document-mode title \
  --debug-samples 2 \
  --log-every 25 \
  --output-dir outputs/coarse_graph_qwen_lora_run1
```

If memory allows, raise `batch-size` to 8. If memory is tight, keep batch size at 2 or 4 and increase gradient accumulation.

## Outputs

The script saves:

```text
outputs/.../train_config.json
outputs/.../train_history.json
outputs/.../debug_predictions.jsonl
outputs/.../debug_readable.log
outputs/.../best_adapter/
outputs/.../latest_adapter/
outputs/.../adapter_model.safetensors
```

`best_adapter/` is selected by validation loss. `latest_adapter/` is the most recent epoch.

## Logs

The script prints:

```text
coarse qwen training | device=cuda | train_samples=... | val_samples=... | pos_ratio=...
train_relation_counts=...
epoch 001/003 batch 0025/...
epoch 001/003 done best | loss=... | val_loss=...
```

The readable debug file shows validation examples:

```text
[epoch 001] coarse qwen debug sample=...
gold=causes:1.000 parsed={'relation_type': 'causes', 'score': 0.81}
source_event: trigger=attacked; mention=...
target_event: trigger=fled; mention=...
```

Use this file to inspect whether Qwen is learning relation JSON and whether event mentions are concrete enough.

## Inference

After training:

```bash
python src/run_coarse_graph_qwen.py \
  --input-mode mirai \
  --query-id 1 \
  --base-model-path models/Qwen2.5-0.5B \
  --adapter-path outputs/coarse_graph_qwen_lora_run1/best_adapter \
  --max-events 16 \
  --max-pairs 128 \
  --keep-threshold 0.5 \
  --output outputs/coarse_graph_qwen_query1.json
```

The output coarse graph can then be passed to refinement:

```bash
python src/run_refinement.py \
  --coarse-graph outputs/coarse_graph_qwen_query1.json \
  --model-path outputs/refinement_graph_4090/refinement_model.pt \
  --include-completion-candidates \
  --output outputs/refined_graph_query1.json
```
