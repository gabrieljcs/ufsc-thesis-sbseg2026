from __future__ import annotations

from pathlib import Path
from typing import Any

from thesis_eval.benchmarks.uriel import load_distance_matrix
from thesis_eval.benchmarks.strongreject import load_prompt_records
from thesis_eval.config import load_config
from thesis_eval.evaluation.strongreject import score_generation_rows
from thesis_eval.io import write_jsonl
from thesis_eval.models.generation import attach_outputs, build_generation_rows
from thesis_eval.paths import DATA_DIR, OUTPUT_DIR, ensure_dir
from thesis_eval.translation.log import save_translation_logs
from thesis_eval.translation.pipeline import PromptRecord, apply_translation_qc, run_translation


PILOT_PROMPTS = [
    PromptRecord(
        prompt_id="pilot_strongreject_001",
        text="[StrongREJECT prompt placeholder: replace with the real forbidden prompt before research use.]",
        harmful_goal="pending_real_strongreject_import",
    ),
    PromptRecord(
        prompt_id="pilot_strongreject_002",
        text="[StrongREJECT prompt placeholder: replace with the real forbidden prompt before research use.]",
        harmful_goal="pending_real_strongreject_import",
    ),
]


def enrich_generation_rows(rows: list[dict[str, Any]], distance_csv: Path) -> list[dict[str, Any]]:
    matrix = load_distance_matrix(distance_csv)
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        aligned = str(row["aligned_language"])
        attack = str(row["attack_language"])
        enriched.update(
            {
                "distance": matrix[aligned][attack],
                "if_score": None,
                "cons_score": None,
                "spec_score": None,
                "input_tokens": None,
                "tokens_per_char": None,
                "token_inflation": None,
                "excluded": False,
                "exclusion_reason": None,
            }
        )
        output.append(enriched)
    return output


def _pilot_prompts_from_file(path: Path, limit: int = 2) -> list[PromptRecord]:
    rows = load_prompt_records(path, limit=limit)
    return [
        PromptRecord(
            prompt_id=str(row["prompt_id"]),
            text=str(row["text"]),
            harmful_goal=str(row.get("harmful_goal", "pending_manual_intent_spec")),
            expected_output_form=str(row.get("expected_output_form", "other")),
            fixed_constraints=str(row.get("fixed_constraints", "pending_manual_intent_spec")),
            allowed_adaptation=str(row.get("allowed_adaptation", "fluency_only_no_semantic_change")),
        )
        for row in rows
    ]


def run_mock_pilot(output_dir: Path = OUTPUT_DIR / "pilot", prompts_path: Path | None = None, prompt_limit: int = 2) -> dict[str, Path]:
    cfg = load_config()
    ensure_dir(output_dir)
    distance_csv = DATA_DIR / "uriel_plus" / "distance_matrix.csv"
    if not distance_csv.exists():
        raise FileNotFoundError("Run thesis-eval prepare-uriel before run-pilot")

    prompts = _pilot_prompts_from_file(prompts_path, limit=prompt_limit) if prompts_path else PILOT_PROMPTS[:prompt_limit]
    translation_logs = []
    for language in ("por", "ara"):
        translation_logs.extend(run_translation(prompts, language, engine="placeholder"))
    translations_path = output_dir / "translations.jsonl"
    save_translation_logs(translations_path, translation_logs)

    qc_logs = apply_translation_qc([log.to_dict() for log in translation_logs])
    qc_path = output_dir / "translations_qc.jsonl"
    write_jsonl(qc_path, qc_logs)

    generation_rows = build_generation_rows(qc_logs, model="sagui_7b", aligned_language=cfg.aligned_language["sagui_7b"])
    mock_outputs = ["[mock target response: replace with vLLM output before research use.]" for _ in generation_rows]
    generations = enrich_generation_rows(attach_outputs(generation_rows, mock_outputs), distance_csv)
    generations_path = output_dir / "target_generations.jsonl"
    write_jsonl(generations_path, generations)

    scored = score_generation_rows(generations, evaluator="mock")
    scored_path = output_dir / "strongreject_scores.jsonl"
    write_jsonl(scored_path, scored)

    return {
        "translations": translations_path,
        "translations_qc": qc_path,
        "generations": generations_path,
        "strongreject_scores": scored_path,
    }
