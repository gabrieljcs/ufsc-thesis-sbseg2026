from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from thesis_eval.io import read_jsonl, write_jsonl


STRONGREJECT_GITHUB_URL = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"
STRONGREJECT_SMALL_GITHUB_URL = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_small_dataset.csv"
STRONGREJECT_HF_ID = "walledai/StrongREJECT"


def import_strongreject(
    source: str = "github",
    input_path: Path | None = None,
    limit: int | None = None,
    small: bool = False,
) -> list[dict[str, Any]]:
    if source == "local":
        if input_path is None:
            raise ValueError("--input is required when --source local")
        rows = _read_csv_rows(input_path.read_text(encoding="utf-8"))
    elif source == "github":
        rows = _download_github_rows(small=small)
    elif source == "hf":
        rows = _download_hf_rows()
    else:
        raise ValueError(f"Unknown StrongREJECT source {source!r}")

    records = [_normalize_row(row, index) for index, row in enumerate(rows, start=1)]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("StrongREJECT import produced no records")
    return records


def load_prompt_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    output = rows[:limit] if limit is not None else rows
    for row in output:
        if "prompt_id" not in row or "text" not in row:
            raise ValueError(f"Prompt row must contain prompt_id and text: {row}")
    return output


def write_prompt_records(path: Path, records: list[dict[str, Any]]) -> None:
    write_jsonl(path, records)


def _download_github_rows(small: bool = False) -> list[dict[str, str]]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("GitHub StrongREJECT import requires requests.") from exc
    url = STRONGREJECT_SMALL_GITHUB_URL if small else STRONGREJECT_GITHUB_URL
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return _read_csv_rows(response.text)


def _download_hf_rows() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face StrongREJECT import requires datasets.") from exc
    dataset = load_dataset(STRONGREJECT_HF_ID, split="train")
    return [dict(row) for row in dataset]


def _read_csv_rows(text: str) -> list[dict[str, str]]:
    with io.StringIO(text) as handle:
        return list(csv.DictReader(handle))


def _normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    prompt = row.get("forbidden_prompt") or row.get("prompt") or row.get("text")
    if not prompt:
        raise ValueError(f"StrongREJECT row {index} has no prompt/forbidden_prompt/text column")
    category = str(row.get("category", "")).strip()
    source = str(row.get("source", "")).strip()
    return {
        "prompt_id": f"strongreject_{index:04d}",
        "text": str(prompt),
        "category": category,
        "source": source,
        "harmful_goal": "pending_manual_intent_spec",
        "expected_output_form": "other",
        "fixed_constraints": "pending_manual_intent_spec",
        "allowed_adaptation": "fluency_only_no_semantic_change",
        "benchmark": "strongreject",
        "original_index": index,
    }
