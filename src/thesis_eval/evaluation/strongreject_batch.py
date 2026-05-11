from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_eval.evaluation.strongreject import (
    apply_strongreject_result,
    build_strongreject_rubric_messages,
    parse_strongreject_rubric_judge_response,
    prepare_row_for_strongreject,
    scoring_prompt_text,
)
from thesis_eval.translation.pipeline import NLLB_CODES, NllbTranslator


def prepare_strongreject_batch_entries(
    rows: list[dict[str, Any]],
    *,
    input_path: Path,
    output_path: Path,
    judge_model: str,
    custom_id_prefix: str,
    next_request_index: int,
    backtranslate_engine: str = "reuse",
    translator: NllbTranslator | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    request_index = next_request_index
    for row_index, row in enumerate(rows):
        enriched = _resolve_backtranslation(
            row,
            input_path=input_path,
            row_index=row_index,
            backtranslate_engine=backtranslate_engine,
            translator=translator,
        )
        prepared_row, response_text = prepare_row_for_strongreject(enriched)
        custom_id: str | None = None
        if response_text is not None:
            custom_id = f"{custom_id_prefix}-{request_index:08d}"
            request_index += 1
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": judge_model,
                        "messages": build_strongreject_rubric_messages(
                            scoring_prompt_text(prepared_row),
                            response_text,
                        ),
                        "temperature": 0,
                    },
                }
            )
        manifest.append(
            {
                "custom_id": custom_id,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "row_index": row_index,
                "row_sha256": _row_sha256(prepared_row),
                "row": prepared_row,
            }
        )
    return requests, manifest, request_index


def ingest_strongreject_batch_results(
    manifest_rows: list[dict[str, Any]],
    batch_output_rows: list[dict[str, Any]],
    batch_error_rows: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    results_by_custom_id = {
        str(row["custom_id"]): row for row in batch_output_rows if row.get("custom_id") is not None
    }
    if batch_error_rows:
        for row in batch_error_rows:
            custom_id = row.get("custom_id")
            if custom_id is not None:
                results_by_custom_id[str(custom_id)] = row

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest_rows:
        output_path = str(entry["output_path"])
        grouped.setdefault(output_path, [])
        grouped[output_path].append(
            _apply_batch_result_to_manifest_row(entry, results_by_custom_id.get(str(entry.get("custom_id"))))
        )
    return grouped


def extract_batch_response_text(response_body: dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if output_text:
        return str(output_text)

    choices = response_body.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        text = _coerce_message_content_to_text(content)
        if text.strip():
            return text

    outputs = response_body.get("output")
    if isinstance(outputs, list):
        fragments: list[str] = []
        for output in outputs:
            if not isinstance(output, dict):
                continue
            for item in output.get("content") or []:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"output_text", "text"} and item.get("text"):
                    fragments.append(str(item["text"]))
        if fragments:
            return "\n".join(fragments)

    raise ValueError(f"Could not extract judge text from batch response body: {response_body}")


def _apply_batch_result_to_manifest_row(
    manifest_entry: dict[str, Any],
    batch_row: dict[str, Any] | None,
) -> dict[str, Any]:
    row = dict(manifest_entry["row"])
    custom_id = manifest_entry.get("custom_id")
    if custom_id is None:
        return row
    if batch_row is None:
        return _mark_batch_failure(row, f"missing_batch_result:{custom_id}")

    error = batch_row.get("error")
    if error:
        return _mark_batch_failure(row, f"strongreject_error: {error}")

    response = batch_row.get("response")
    if not isinstance(response, dict):
        return _mark_batch_failure(row, f"strongreject_error: missing response payload for {custom_id}")

    status_code = int(response.get("status_code") or 0)
    if status_code != 200:
        return _mark_batch_failure(
            row,
            f"strongreject_error: batch status {status_code} for {custom_id}",
        )

    body = response.get("body")
    if not isinstance(body, dict):
        return _mark_batch_failure(row, f"strongreject_error: malformed batch body for {custom_id}")

    try:
        judge_response = extract_batch_response_text(body)
        result = parse_strongreject_rubric_judge_response(judge_response)
    except Exception as exc:
        return _mark_batch_failure(row, f"strongreject_error: {exc}")

    return apply_strongreject_result(row, result)


def _resolve_backtranslation(
    row: dict[str, Any],
    *,
    input_path: Path,
    row_index: int,
    backtranslate_engine: str,
    translator: NllbTranslator | None,
) -> dict[str, Any]:
    enriched = dict(row)
    if enriched.get("api_failure") or enriched.get("provider_block") or not enriched.get("generation_completed", True):
        return enriched

    existing_backtranslation = enriched.get("model_output_backtranslated")
    if existing_backtranslation is not None:
        enriched["model_output_backtranslated"] = str(existing_backtranslation)
        return enriched

    model_output = str(enriched.get("model_output") or "")
    if not model_output.strip():
        enriched["model_output_backtranslated"] = None
        return enriched

    attack_language = str(enriched.get("attack_language", "eng"))
    if attack_language == "eng" or attack_language not in NLLB_CODES:
        enriched["model_output_backtranslated"] = model_output
        return enriched

    if backtranslate_engine == "reuse":
        raise ValueError(
            f"{input_path.name} row {row_index + 1} is missing model_output_backtranslated "
            f"for attack_language={attack_language}. Run backtranslate-responses first "
            "or use --backtranslate-engine nllb."
        )
    if backtranslate_engine == "placeholder":
        enriched["model_output_backtranslated"] = model_output
        return enriched
    if backtranslate_engine != "nllb":
        raise ValueError(f"Unknown back-translation engine {backtranslate_engine!r}")
    if translator is None:
        raise RuntimeError(
            "prepare_strongreject_batch_entries received --backtranslate-engine nllb "
            "without an initialized translator."
        )
    enriched["model_output_backtranslated"] = translator.translate(
        model_output,
        "eng",
        source_language=attack_language,
    )
    return enriched


def _coerce_message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict) and item.get("text"):
                fragments.append(str(item["text"]))
        return "\n".join(fragment for fragment in fragments if fragment)
    return ""


def _mark_batch_failure(row: dict[str, Any], reason: str) -> dict[str, Any]:
    failed = dict(row)
    failed["strongreject_score"] = None
    failed["unsafe_binary"] = None
    failed["refusal_binary"] = None
    failed["specificity_score"] = None
    failed["convincingness_score"] = None
    failed["scoring_failed"] = True
    failed["scoring_failure_reason"] = reason
    failed["excluded"] = True
    failed["exclusion_reason"] = reason
    return failed


def _row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
