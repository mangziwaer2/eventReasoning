from __future__ import annotations

import ast
import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from causal_graph import NewsDocument
from causal_graph import QuerySpec
from event_extraction import clean_document_text


@dataclass(slots=True)
class MiraiQueryExample:
    query_id: str
    date_str: str
    actor1_country_code: str
    actor2_country_code: str
    actor1_country_name: str
    actor2_country_name: str
    relation_name: str
    event_base_code: str
    docids: list[str]
    answer_list: list[str]
    answer_dict: dict[str, int]
    raw_row: dict[str, str]

    def build_query_text(self) -> str:
        return (
            f"As of {self.date_str}, what important event may happen next between "
            f"{self.actor1_country_name} and {self.actor2_country_name}?"
        )

    def build_query_spec(self) -> QuerySpec:
        return QuerySpec(
            query_id=self.query_id,
            text=self.build_query_text(),
            cutoff_time=self.date_str,
            focus_entities=[self.actor1_country_name, self.actor2_country_name],
            metadata={
                "dataset": "MIRAI",
                "actor1_country_code": self.actor1_country_code,
                "actor2_country_code": self.actor2_country_code,
                "relation_name": self.relation_name,
                "event_base_code": self.event_base_code,
                "gold_answer_list": self.answer_list,
            },
        )

    def gold_summary(self) -> dict[str, object]:
        return {
            "date_str": self.date_str,
            "actor1_country_name": self.actor1_country_name,
            "actor2_country_name": self.actor2_country_name,
            "relation_name": self.relation_name,
            "event_base_code": self.event_base_code,
            "answer_list": self.answer_list,
            "answer_dict": self.answer_dict,
        }


def _read_tsv_rows(zip_path: Path, member_name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            reader = csv.DictReader(
                io.TextIOWrapper(handle, encoding="utf-8", errors="replace"),
                delimiter="\t",
            )
            return list(reader)


def _parse_literal_list(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _parse_literal_dict(raw_value: str) -> dict[str, int]:
    if not raw_value:
        return {}
    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return {}
    if isinstance(value, dict):
        parsed: dict[str, int] = {}
        for key, item in value.items():
            try:
                parsed[str(key)] = int(item)
            except (TypeError, ValueError):
                continue
        return parsed
    return {}


def load_mirai_queries(zip_path: Path, split: str = "test", limit: int = 0) -> list[MiraiQueryExample]:
    member_name = f"MIRAI/{split}/relation_query.csv"
    rows = _read_tsv_rows(zip_path, member_name)
    if limit >0:
        rows = rows[:limit]

    examples: list[MiraiQueryExample] = []
    for row in rows:
        examples.append(
            MiraiQueryExample(
                query_id=row["QueryId"],
                date_str=row["DateStr"],
                actor1_country_code=row["Actor1CountryCode"],
                actor2_country_code=row["Actor2CountryCode"],
                actor1_country_name=row["Actor1CountryName"],
                actor2_country_name=row["Actor2CountryName"],
                relation_name=row["RelName"],
                event_base_code=row["EventBaseCode"],
                docids=_parse_literal_list(row.get("Docids", "")),
                answer_list=_parse_literal_list(row.get("AnswerList", "")),
                answer_dict=_parse_literal_dict(row.get("AnswerDict", "")),
                raw_row=row,
            )
        )
    return examples


def get_mirai_query_by_id(zip_path: Path, query_id: str, split: str = "test") -> MiraiQueryExample:
    for example in load_mirai_queries(zip_path, split=split):
        if example.query_id == str(query_id):
            return example
    raise KeyError(f"QueryId {query_id} was not found in MIRAI/{split}/relation_query.csv")


def load_mirai_news_for_docids(zip_path: Path, docids: list[str]) -> list[NewsDocument]:
    docid_set = set(docids)
    if not docid_set:
        return []

    selected_rows: dict[str, NewsDocument] = {}
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("MIRAI/data_news.csv") as handle:
            reader = csv.DictReader(
                io.TextIOWrapper(handle, encoding="utf-8", errors="replace"),
                delimiter="\t",
            )
            for row in reader:
                docid = row.get("Docid", "")
                if docid not in docid_set or docid in selected_rows:
                    continue

                title = row.get("Title", "").strip()
                text = row.get("Text", "").strip()
                abstract = row.get("Abstract", "").strip()
                parts: list[str] = []
                if abstract:
                    parts.append(abstract)
                if text:
                    parts.append(text)
                combined_text = clean_document_text(title=title, text="\n".join(parts))

                selected_rows[docid] = NewsDocument(
                    document_id=docid,
                    title=title,
                    text=combined_text,
                    publish_time=row.get("Date"),
                    source="MIRAI",
                    metadata={
                        "url": row.get("URL", ""),
                        "md5": row.get("MD5", ""),
                    },
                )

                if len(selected_rows) == len(docid_set):
                    break

    ordered_documents: list[NewsDocument] = []
    seen: set[str] = set()
    for docid in docids:
        if docid in seen:
            continue
        if docid in selected_rows:
            ordered_documents.append(selected_rows[docid])
            seen.add(docid)
    return ordered_documents


def export_mirai_query_snapshot(example: MiraiQueryExample) -> str:
    payload = {
        "query_id": example.query_id,
        "date_str": example.date_str,
        "actor1_country_name": example.actor1_country_name,
        "actor2_country_name": example.actor2_country_name,
        "relation_name": example.relation_name,
        "event_base_code": example.event_base_code,
        "docids": example.docids[:10],
        "answer_list": example.answer_list,
        "answer_dict": example.answer_dict,
        "query_text": example.build_query_text(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
