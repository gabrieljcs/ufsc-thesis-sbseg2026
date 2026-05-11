from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

from thesis_eval.io import read_jsonl
from thesis_eval.metrics.spec import add_spec_scores


OPTIONS = {"A", "B", "C", "D"}
OPTION_ORDER = ("A", "B", "C", "D")
BELEBELE_LANGUAGE_CODES = {
    "ara": "arb_Arab",
    "bul": "bul_Cyrl",
    "dan": "dan_Latn",
    "eng": "eng_Latn",
    "fin": "fin_Latn",
    "ita": "ita_Latn",
    "nor": "nob_Latn",
    "por": "por_Latn",
    "rus": "rus_Cyrl",
    "spa": "spa_Latn",
    "swa": "swh_Latn",
    "swe": "swe_Latn",
    "ukr": "ukr_Cyrl",
}
ANSWER_BY_NUMBER = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
}


def parse_choice(text: str) -> str | None:
    value = text.strip().upper()
    if value in OPTIONS:
        return value
    matches = re.findall(r"\b([ABCD])\b", value)
    if matches:
        return matches[-1]
    return None


def _normalize_choice_text(text: str) -> str:
    value = text.strip().casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\"'“”‘’«».,;:!?()\[\]{}]+", "", value)
    return value.strip()


def parse_choice_from_options(text: str, options: dict[str, str]) -> str | None:
    direct = parse_choice(text)
    if direct is not None:
        return direct
    normalized_text = _normalize_choice_text(text)
    if not normalized_text:
        return None
    exact_matches = [
        label
        for label, option_text in options.items()
        if _normalize_choice_text(option_text) == normalized_text
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    contains_matches = [
        label
        for label, option_text in options.items()
        if (normalized_option := _normalize_choice_text(option_text)) and (
            normalized_option in normalized_text or normalized_text in normalized_option
        )
    ]
    if len(contains_matches) == 1:
        return contains_matches[0]
    return None


def _row_predicted_choice(row: dict[str, Any]) -> str | None:
    predicted_choice = row.get("predicted_choice")
    if predicted_choice is not None:
        parsed = parse_choice(str(predicted_choice))
        if parsed is not None:
            return parsed
    return parse_choice(str(row.get("prediction", "")))


def compute_if_cons(predictions: list[dict[str, Any]], aligned_language: dict[str, str]) -> list[dict[str, Any]]:
    by_model_lang: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model_item_lang: dict[tuple[str, str, str], str | None] = {}
    for row in predictions:
        model = str(row["model"])
        language = str(row["language"])
        item_id = str(row["item_id"])
        predicted = _row_predicted_choice(row)
        normalized = dict(row)
        normalized["predicted_choice"] = predicted
        by_model_lang[(model, language)].append(normalized)
        by_model_item_lang[(model, item_id, language)] = predicted

    output: list[dict[str, Any]] = []
    for (model, language), rows in sorted(by_model_lang.items()):
        correct = 0
        comparable = 0
        consistent = 0
        aligned = aligned_language[model]
        for row in rows:
            predicted = row["predicted_choice"]
            gold = parse_choice(str(row.get("gold", "")))
            if predicted is not None and gold is not None and predicted == gold:
                correct += 1
            aligned_prediction = by_model_item_lang.get((model, str(row["item_id"]), aligned))
            if predicted is not None and aligned_prediction is not None:
                comparable += 1
                if predicted == aligned_prediction:
                    consistent += 1
        total = len(rows)
        output.append(
            {
                "model": model,
                "attack_language": language,
                "if_score": correct / total if total else None,
                "cons_score": consistent / comparable if comparable else None,
                "n_items": total,
                "n_cons_comparable": comparable,
            }
        )
    complete = [row for row in output if row["if_score"] is not None and row["cons_score"] is not None]
    enriched = add_spec_scores(complete)
    spec_by_key = {(row["model"], row["attack_language"]): row["spec_score"] for row in enriched}
    for row in output:
        row["spec_score"] = spec_by_key.get((row["model"], row["attack_language"]))
    return output


def resolve_belebele_language_code(language: str) -> str:
    if language not in BELEBELE_LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported BELEBELE language {language!r}; expected one of {sorted(BELEBELE_LANGUAGE_CODES)}"
        )
    return BELEBELE_LANGUAGE_CODES[language]


def load_belebele_rows(dataset_dir: Path, language: str, limit: int | None = None) -> list[dict[str, Any]]:
    data_dir = dataset_dir / "data" if (dataset_dir / "data").is_dir() else dataset_dir
    dialect = resolve_belebele_language_code(language)
    path = data_dir / f"{dialect}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing BELEBELE file for {language}: {path}")
    rows = read_jsonl(path)
    if limit is not None:
        return rows[:limit]
    return rows


def build_belebele_prompt(row: dict[str, Any]) -> str:
    return (
        "Passage:\n"
        f"{row['flores_passage']}\n\n"
        "Question:\n"
        f"{row['question']}\n\n"
        "Choices:\n"
        f"A. {row['mc_answer1']}\n"
        f"B. {row['mc_answer2']}\n"
        f"C. {row['mc_answer3']}\n"
        f"D. {row['mc_answer4']}\n\n"
        "Choose the correct option. Reply with only one letter: A, B, C, or D.\n"
        "Answer:"
    )


def build_belebele_prediction_rows(
    dataset_rows: list[dict[str, Any]],
    *,
    model: str,
    language: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row_index, row in enumerate(dataset_rows, start=1):
        answer_number = str(row["correct_answer_num"])
        if answer_number not in ANSWER_BY_NUMBER:
            raise ValueError(f"Unexpected BELEBELE correct_answer_num={answer_number!r}")
        item_id = f"belebele_{row_index:04d}"
        output.append(
            {
                "model": model,
                "language": language,
                "dataset_dialect": str(row["dialect"]),
                "item_id": item_id,
                "belebele_row_index": row_index,
                "question_number": int(row["question_number"]),
                "gold": ANSWER_BY_NUMBER[answer_number],
                "choice_a": str(row["mc_answer1"]),
                "choice_b": str(row["mc_answer2"]),
                "choice_c": str(row["mc_answer3"]),
                "choice_d": str(row["mc_answer4"]),
            }
        )
    return output


def attach_belebele_predictions(rows: list[dict[str, Any]], outputs: list[Any]) -> list[dict[str, Any]]:
    if len(rows) != len(outputs):
        raise ValueError("BELEBELE row count and output count differ")
    enriched: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs, strict=True):
        text = str(output.get("text", "")) if isinstance(output, dict) else str(output)
        predicted = parse_choice_from_options(
            text,
            {
                "A": str(row.get("choice_a", "")),
                "B": str(row.get("choice_b", "")),
                "C": str(row.get("choice_c", "")),
                "D": str(row.get("choice_d", "")),
            },
        )
        enriched_row = dict(row)
        enriched_row["raw_output"] = text
        enriched_row["prediction"] = predicted if predicted is not None else text.strip()
        enriched_row["predicted_choice"] = predicted
        if isinstance(output, dict):
            for key in (
                "generation_completed",
                "api_http_status",
                "api_failure",
                "api_failure_reason",
                "provider_block",
                "provider_block_reason",
            ):
                if key in output:
                    enriched_row[key] = output[key]
        else:
            enriched_row["generation_completed"] = True
        enriched.append(enriched_row)
    return enriched


def repair_belebele_predictions(
    predictions: list[dict[str, Any]],
    *,
    dataset_dir: Path,
) -> list[dict[str, Any]]:
    by_language: dict[str, list[dict[str, Any]]] = {}
    for language in sorted({str(row["language"]) for row in predictions}):
        dataset_rows = load_belebele_rows(dataset_dir, language)
        by_language[language] = dataset_rows
    language_offsets: dict[str, int] = defaultdict(int)
    repaired: list[dict[str, Any]] = []
    for row in predictions:
        language = str(row["language"])
        if row.get("belebele_row_index") is not None:
            row_index = int(row["belebele_row_index"])
        else:
            language_offsets[language] += 1
            row_index = language_offsets[language]
        dataset_rows = by_language[language]
        if row_index < 1 or row_index > len(dataset_rows):
            raise ValueError(
                f"BELEBELE row index {row_index} out of range for language={language} "
                f"(n={len(dataset_rows)})"
            )
        dataset_row = dataset_rows[row_index - 1]
        choice_map = {
            "A": str(dataset_row["mc_answer1"]),
            "B": str(dataset_row["mc_answer2"]),
            "C": str(dataset_row["mc_answer3"]),
            "D": str(dataset_row["mc_answer4"]),
        }
        raw_output = str(row.get("raw_output", row.get("prediction", "")))
        predicted = parse_choice_from_options(raw_output, choice_map)
        updated = dict(row)
        updated["item_id"] = f"belebele_{row_index:04d}"
        updated["belebele_row_index"] = row_index
        updated["question_number"] = int(dataset_row["question_number"])
        updated["gold"] = ANSWER_BY_NUMBER[str(dataset_row["correct_answer_num"])]
        updated["dataset_dialect"] = str(dataset_row["dialect"])
        updated["choice_a"] = choice_map["A"]
        updated["choice_b"] = choice_map["B"]
        updated["choice_c"] = choice_map["C"]
        updated["choice_d"] = choice_map["D"]
        updated["prediction"] = predicted if predicted is not None else raw_output.strip()
        updated["predicted_choice"] = predicted
        repaired.append(updated)
    return repaired
