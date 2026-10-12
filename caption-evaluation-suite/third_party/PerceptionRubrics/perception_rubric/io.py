"""JSONL helpers for the minimal Perception Rubric evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


CAPTION_FIELDS = ("caption", "prediction", "output", "model_output", "generated_caption")


def load_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            yield record


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_caption(record: Dict[str, Any]) -> Optional[str]:
    for field in CAPTION_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def load_predictions(path: str | Path) -> Dict[str, str]:
    """Load predictions from JSONL. Rows with empty/missing captions are included as empty string."""
    predictions: Dict[str, str] = {}
    for record in load_jsonl(path):
        sample_id = record.get("id")
        if sample_id is None:
            raise ValueError("Each prediction row must include an `id` field")
        caption = get_caption(record) or ""
        predictions[str(sample_id)] = caption
    return predictions


def get_rubrics(record: Dict[str, Any]) -> Any:
    if "rubrics" in record:
        return record["rubrics"]
    grouped = {}
    if "must_right" in record:
        grouped["must_right"] = record["must_right"]
    if "group_a" in record:
        grouped["group_a"] = record["group_a"]
    if "easy_wrong" in record:
        grouped["easy_wrong"] = record["easy_wrong"]
    if "group_b" in record:
        grouped["group_b"] = record["group_b"]
    return grouped if grouped else None
