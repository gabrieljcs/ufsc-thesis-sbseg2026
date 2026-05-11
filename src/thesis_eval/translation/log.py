from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from thesis_eval.io import read_jsonl, write_jsonl
from thesis_eval.schema import validate_translation_log


@dataclass
class TranslationLog:
    prompt_id: str
    source_language: str
    target_language: str
    harmful_goal: str
    expected_output_form: str
    fixed_constraints: str
    allowed_adaptation: str
    translator_config: dict[str, str]
    blaser_qe_score: float | None
    blaser_status: str
    human_audit_status: str
    reference_subset_score: float | None
    harm_preservation: str
    revision_note: str | None
    source_text: str
    translated_text: str
    roundtrip_text: str | None = None
    roundtrip_drift_score: float | None = None
    roundtrip_status: str = "pending"
    xsts_semantic_equivalence: str = "pending"
    xsts_attack_strength: str = "pending"
    xsts_constraint_preservation: str = "pending"
    xsts_naturalness: str = "pending"
    xsts_no_harmfulness_drift: str = "pending"
    translation_revised: bool = False
    translation_excluded: bool = False

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        validate_translation_log(record)
        return record


def load_translation_logs(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        validate_translation_log(row)
    return rows


def save_translation_logs(path: Path, logs: list[TranslationLog | dict[str, Any]]) -> None:
    rows = [log.to_dict() if isinstance(log, TranslationLog) else log for log in logs]
    for row in rows:
        validate_translation_log(row)
    write_jsonl(path, rows)
