from __future__ import annotations

import argparse
import json
from pathlib import Path

from coarse_graph_dataset import load_maven_pair_samples
from local_qwen_lora import LoraUnavailable
from local_qwen_lora import load_qwen_with_lora
from path_utils import REPO_ROOT
from path_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen LoRA coarse-graph proposer inference on MAVEN-ERE event-pair samples.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "outputs" / "coarse_graph_qwen_lora"), help="LoRA model directory.")
    parser.add_argument("--split", default="valid", help="MAVEN split name.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of MAVEN rows.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative to positive pair ratio.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def _format_prompt(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou infer causal relations between structured events.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    args = parse_args()
    samples = load_maven_pair_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit,
        negative_ratio=args.negative_ratio,
    )
    instruction_samples = [sample.to_instruction_example() for sample in samples[:8]]

    try:
        model, tokenizer, torch = load_qwen_with_lora(resolve_repo_path(args.model_path))
    except LoraUnavailable as exc:
        error_payload = {"error": str(exc)}
        print(json.dumps(error_payload, ensure_ascii=False))
        return

    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for item in instruction_samples:
            prompt = _format_prompt(item["prompt"])
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
            predictions.append(
                {
                    "sample_id": item["sample_id"],
                    "prompt": item["prompt"],
                    "target": item["target"],
                    "prediction": generated,
                }
            )

    output_text = json.dumps({"predictions": predictions}, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
