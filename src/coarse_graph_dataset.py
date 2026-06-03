from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
from path_utils import resolve_repo_path


RELATION_TO_ID = {
    "none": 0,
    "precedes": 1,
    "causes": 2,
    "escalates": 3,
    "mitigates": 4,
}

ID_TO_RELATION = {value: key for key, value in RELATION_TO_ID.items()}


@dataclass(slots=True)
class CoarseGraphPairSample:
    sample_id: str
    query_text: str
    event_a_text: str
    event_b_text: str
    features: list[float]
    relation_label: int
    edge_score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query_text": self.query_text,
            "event_a_text": self.event_a_text,
            "event_b_text": self.event_b_text,
            "features": self.features,
            "relation_label": self.relation_label,
            "edge_score": self.edge_score,
            "metadata": self.metadata,
        }

    def to_instruction_example(self) -> dict[str, Any]:
        prompt = (
            "You are given a query and two structured events.\n"
            "Predict the relation from event A to event B and a confidence score between 0 and 1.\n"
            "Allowed relations: none, precedes, causes, escalates, mitigates.\n\n"
            f"Query: {self.query_text}\n"
            f"Event A: {self.event_a_text}\n"
            f"Event B: {self.event_b_text}\n"
            "Return strict JSON with keys relation and score."
        )
        target = json.dumps(
            {
                "relation": ID_TO_RELATION.get(self.relation_label, "none"),
                "score": round(float(self.edge_score), 4),
            },
            ensure_ascii=False,
        )
        return {
            "sample_id": self.sample_id,
            "prompt": prompt,
            "target": target,
            "metadata": self.metadata,
        }


class CoarseGraphPairDataset(Dataset):
    def __init__(self, samples: list[CoarseGraphPairSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "features": torch.tensor(sample.features, dtype=torch.float32),
            "relation_label": torch.tensor(sample.relation_label, dtype=torch.long),
            "edge_score": torch.tensor(sample.edge_score, dtype=torch.float32),
            "metadata": sample.metadata,
        }


def _read_maven_rows(zip_path: Path, split: str) -> list[dict[str, Any]]:
    member_name = f"MAVEN_ERE/{split}.jsonl"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            text_stream = io.TextIOWrapper(handle, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _normalize_sentence(sentence) -> str:
    if isinstance(sentence, list):
        return " ".join(sentence)
    return str(sentence)


def _simple_tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.replace(",", " ").replace(".", " ").split() if token.strip()]


def _overlap_score(left: str, right: str) -> float:
    left_tokens = set(_simple_tokenize(left))
    right_tokens = set(_simple_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _maven_relation_to_type(relation_name: str) -> str:
    if relation_name == "CAUSE":
        return "causes"
    if relation_name == "PRECONDITION":
        return "precedes"
    if relation_name == "BEFORE":
        return "precedes"
    if relation_name in {"OVERLAP", "SIMULTANEOUS", "CONTAINS", "ENDS-ON", "BEGINS-ON"}:
        return "precedes"
    return "none"


def _pair_features(
    query_text: str,
    event_a_text: str,
    event_b_text: str,
    sent_a: int,
    sent_b: int,
    event_type_a: int,
    event_type_b: int,
) -> list[float]:
    query_event_a = _overlap_score(query_text, event_a_text)
    query_event_b = _overlap_score(query_text, event_b_text)
    event_overlap = _overlap_score(event_a_text, event_b_text)
    sentence_gap = float(max(0, sent_b - sent_a))
    same_sentence = 1.0 if sent_a == sent_b else 0.0
    event_type_gap = float(abs(event_type_a - event_type_b))
    return [
        query_event_a,
        query_event_b,
        event_overlap,
        sentence_gap,
        same_sentence,
        float(event_type_a),
        float(event_type_b),
        event_type_gap,
    ]


def load_maven_pair_samples(
    dataset_path: Path,
    split: str = "train",
    limit: int | None = None,
    negative_ratio: float = 1.0,
    seed: int = 7,
) -> list[CoarseGraphPairSample]:
    random.seed(seed)
    rows = _read_maven_rows(dataset_path, split=split)
    if limit is not None:
        rows = rows[:limit]

    samples: list[CoarseGraphPairSample] = []

    for row in rows:
        query_text = row.get("title", "MAVEN-ERE sample")
        sentences = [_normalize_sentence(sentence) for sentence in row.get("sentences", [])]
        events = row.get("events", [])
        if len(events) < 2:
            continue

        event_map: dict[str, dict[str, Any]] = {}
        for event in events:
            mentions = event.get("mention", [])
            if not mentions:
                continue
            mention = mentions[0]
            sent_id = int(mention.get("sent_id", 0))
            event_map[event["id"]] = {
                "text": sentences[sent_id],
                "sent_id": sent_id,
                "event_type_id": int(event.get("type_id", -1)),
            }

        positive_pairs: dict[tuple[str, str], tuple[str, float]] = {}
        for relation_name, pairs in row.get("causal_relations", {}).items():
            for source_id, target_id in pairs:
                if source_id not in event_map or target_id not in event_map:
                    continue
                positive_pairs[(source_id, target_id)] = (_maven_relation_to_type(relation_name), 1.0)

        for relation_name, pairs in row.get("temporal_relations", {}).items():
            for source_id, target_id in pairs:
                if source_id not in event_map or target_id not in event_map:
                    continue
                positive_pairs.setdefault((source_id, target_id), (_maven_relation_to_type(relation_name), 0.85))

        for pair_index, ((source_id, target_id), (relation_type, edge_score)) in enumerate(positive_pairs.items()):
            source = event_map[source_id]
            target = event_map[target_id]
            samples.append(
                CoarseGraphPairSample(
                    sample_id=f"{row['id']}_pos_{pair_index}",
                    query_text=query_text,
                    event_a_text=source["text"],
                    event_b_text=target["text"],
                    features=_pair_features(
                        query_text=query_text,
                        event_a_text=source["text"],
                        event_b_text=target["text"],
                        sent_a=source["sent_id"],
                        sent_b=target["sent_id"],
                        event_type_a=source["event_type_id"],
                        event_type_b=target["event_type_id"],
                    ),
                    relation_label=RELATION_TO_ID.get(relation_type, 0),
                    edge_score=edge_score,
                    metadata={"positive": True, "relation_type": relation_type, "row_id": row["id"]},
                )
            )

        negatives_target = max(1, int(len(positive_pairs) * negative_ratio))
        event_ids = list(event_map.keys())
        negative_count = 0
        seen_negative_pairs: set[tuple[str, str]] = set()
        while negative_count < negatives_target and len(event_ids) >= 2:
            source_id, target_id = random.sample(event_ids, 2)
            pair = (source_id, target_id)
            if pair in positive_pairs or pair in seen_negative_pairs:
                continue
            seen_negative_pairs.add(pair)
            source = event_map[source_id]
            target = event_map[target_id]
            samples.append(
                CoarseGraphPairSample(
                    sample_id=f"{row['id']}_neg_{negative_count}",
                    query_text=query_text,
                    event_a_text=source["text"],
                    event_b_text=target["text"],
                    features=_pair_features(
                        query_text=query_text,
                        event_a_text=source["text"],
                        event_b_text=target["text"],
                        sent_a=source["sent_id"],
                        sent_b=target["sent_id"],
                        event_type_a=source["event_type_id"],
                        event_type_b=target["event_type_id"],
                    ),
                    relation_label=RELATION_TO_ID["none"],
                    edge_score=0.0,
                    metadata={"positive": False, "relation_type": "none", "row_id": row["id"]},
                )
            )
            negative_count += 1

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MAVEN-ERE event-pair samples for coarse graph training.")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "datasets" / "MAVEN_ERE.zip"), help="Path to MAVEN-ERE zip file.")
    parser.add_argument("--split", default="train", help="MAVEN split name.")
    parser.add_argument("--limit", type=int, default=2, help="Maximum number of MAVEN rows.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative to positive pair ratio.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_maven_pair_samples(
        dataset_path=resolve_repo_path(args.dataset),
        split=args.split,
        limit=args.limit,
        negative_ratio=args.negative_ratio,
    )
    payload = {
        "samples": [sample.to_dict() for sample in samples],
        "instruction_samples": [sample.to_instruction_example() for sample in samples[:16]],
    }
    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        resolve_repo_path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
