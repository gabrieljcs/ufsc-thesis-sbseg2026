from __future__ import annotations

from typing import Any

from thesis_eval.translation.pipeline import NLLB_CODES, NllbTranslator


def simple_drift_score(source_text: str, roundtrip_text: str) -> float:
    if not source_text.strip() and not roundtrip_text.strip():
        return 1.0
    if not source_text.strip() or not roundtrip_text.strip():
        return 0.0
    source_words = set(source_text.lower().split())
    roundtrip_words = set(roundtrip_text.lower().split())
    if not source_words and not roundtrip_words:
        return 1.0
    return len(source_words.intersection(roundtrip_words)) / max(len(source_words.union(roundtrip_words)), 1)


def attach_prompt_roundtrip(
    logs: list[dict[str, Any]],
    engine: str = "placeholder",
    checkpoint: str = "facebook/nllb-200-distilled-600M",
    device: str = "auto",
    dtype: str = "auto",
    pass_threshold: float = 0.60,
    fail_threshold: float = 0.35,
    translator: NllbTranslator | None = None,
) -> list[dict[str, Any]]:
    if translator is None:
        if engine == "nllb":
            translator = NllbTranslator(checkpoint=checkpoint, device=device, dtype=dtype)
        elif engine != "placeholder":
            raise ValueError(f"Unknown back-translation engine {engine!r}")

    output: list[dict[str, Any]] = []
    for log in logs:
        row = dict(log)
        target_language = str(row["target_language"])
        translated_text = str(row["translated_text"])
        if target_language == "eng":
            roundtrip = translated_text
        elif translator is not None:
            roundtrip = translator.translate(translated_text, "eng", source_language=target_language)
        else:
            roundtrip = translated_text
        score = simple_drift_score(str(row["source_text"]), roundtrip)
        row["roundtrip_text"] = roundtrip
        row["roundtrip_drift_score"] = round(float(score), 4)
        if score >= pass_threshold:
            row["roundtrip_status"] = "pass"
        elif score < fail_threshold:
            row["roundtrip_status"] = "fail"
        else:
            row["roundtrip_status"] = "flag"
        output.append(row)
    return output


def attach_response_backtranslations(
    rows: list[dict[str, Any]],
    engine: str = "placeholder",
    checkpoint: str = "facebook/nllb-200-distilled-600M",
    device: str = "auto",
    dtype: str = "auto",
    translator: NllbTranslator | None = None,
) -> list[dict[str, Any]]:
    if translator is None:
        if engine == "nllb" and any(str(row.get("attack_language")) != "eng" for row in rows):
            translator = NllbTranslator(checkpoint=checkpoint, device=device, dtype=dtype)
        elif engine != "placeholder":
            raise ValueError(f"Unknown response back-translation engine {engine!r}")

    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        existing_backtranslation = enriched.get("model_output_backtranslated")
        if existing_backtranslation is not None:
            enriched["model_output_backtranslated"] = str(existing_backtranslation)
            output.append(enriched)
            continue
        attack_language = str(enriched.get("attack_language", "eng"))
        model_output = str(enriched.get("model_output") or "")
        if not model_output:
            enriched["model_output_backtranslated"] = None
        elif attack_language == "eng" or attack_language not in NLLB_CODES:
            enriched["model_output_backtranslated"] = model_output
        elif translator is not None:
            translated = translator.translate(model_output, "eng", source_language=attack_language)
            if not translated.strip() and _is_nonlinguistic_output(model_output):
                translated = model_output
            enriched["model_output_backtranslated"] = translated
        else:
            enriched["model_output_backtranslated"] = model_output
        output.append(enriched)
    return output


def _is_nonlinguistic_output(text: str) -> bool:
    return bool(text.strip()) and not any(character.isalpha() for character in text)
