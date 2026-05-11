from __future__ import annotations

from typing import Any

from thesis_eval.config import ConfigBundle
from thesis_eval.schema import ANALYSIS_COLUMNS, validate_analysis_rows


def build_frozen_rows(
    scored_rows: list[dict[str, Any]],
    cfg: ConfigBundle,
    translations: list[dict[str, Any]] | None = None,
    spec_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    translation_by_key = {
        (str(row["prompt_id"]), str(row["target_language"])): row for row in (translations or [])
    }
    spec_by_key = {
        (str(row["model"]), str(row["attack_language"])): row for row in (spec_rows or [])
    }
    output: list[dict[str, Any]] = []
    for row in scored_rows:
        model = str(row["model"])
        attack_language = str(row["attack_language"])
        model_cfg = cfg.models[model]
        translation = translation_by_key.get((str(row["prompt_id"]), attack_language), {})
        spec = spec_by_key.get((model, attack_language), {})
        frozen = {column: None for column in ANALYSIS_COLUMNS}
        frozen.update(
            {
                "model": model,
                "model_display_name": model_cfg.get("display_name", model),
                "model_pair_language": model_cfg.get("pair_language", model_cfg["aligned_language"]),
                "model_alignment_pole": model_cfg.get("alignment_pole"),
                "model_access_mode": model_cfg.get("access_mode"),
                "provider_model_id": model_cfg.get("provider_model_id") or model_cfg.get("hf_id"),
                "provider_model_version": row.get("provider_model_version") or model_cfg.get("provider_model_id") or model_cfg.get("hf_id"),
                "aligned_language": row.get("aligned_language") or model_cfg["aligned_language"],
                "attack_language": attack_language,
                "prompt_id": row["prompt_id"],
                "run_id": row["run_id"],
                "benchmark": row.get("benchmark", "strongreject"),
                "prompt_text": row.get("prompt_text"),
                "source_prompt_text": row.get("source_prompt_text"),
                "model_output": row.get("model_output"),
                "model_output_backtranslated": row.get("model_output_backtranslated"),
                "generation_completed": bool(row.get("generation_completed", False)),
                "api_http_status": row.get("api_http_status"),
                "api_failure": bool(row.get("api_failure", False)),
                "api_failure_reason": row.get("api_failure_reason"),
                "provider_block": bool(row.get("provider_block", False)),
                "provider_block_reason": row.get("provider_block_reason"),
                "strongreject_score": row.get("strongreject_score"),
                "unsafe_binary": row.get("unsafe_binary"),
                "refusal_binary": row.get("refusal_binary"),
                "specificity_score": row.get("specificity_score"),
                "convincingness_score": row.get("convincingness_score"),
                "scoring_failed": bool(row.get("scoring_failed", False)),
                "scoring_failure_reason": row.get("scoring_failure_reason"),
                "distance": row.get("distance"),
                "if_score": spec.get("if_score", row.get("if_score")),
                "cons_score": spec.get("cons_score", row.get("cons_score")),
                "spec_score": spec.get("spec_score", row.get("spec_score")),
                "input_tokens": row.get("input_tokens"),
                "tokens_per_char": row.get("tokens_per_char"),
                "token_inflation": row.get("token_inflation"),
                "context_usage_ratio": row.get("context_usage_ratio"),
                "truncation_risk": bool(row.get("truncation_risk", False)),
                "benign_control_id": row.get("benign_control_id"),
                "benign_relevance": row.get("benign_relevance"),
                "benign_completeness": row.get("benign_completeness"),
                "benign_coherence": row.get("benign_coherence"),
                "benign_task_completion": row.get("benign_task_completion"),
                "translation_blaser_qe_score": translation.get("blaser_qe_score"),
                "translation_blaser_status": translation.get("blaser_status"),
                "roundtrip_drift_score": translation.get("roundtrip_drift_score"),
                "roundtrip_status": translation.get("roundtrip_status"),
                "human_audit_status": translation.get("human_audit_status"),
                "harm_preservation": translation.get("harm_preservation"),
                "translation_revised": bool(translation.get("translation_revised", False)),
                "translation_excluded": bool(translation.get("translation_excluded", False)),
                "excluded": bool(row.get("excluded", False) or translation.get("translation_excluded", False)),
                "exclusion_reason": row.get("exclusion_reason") or ("translation_rejected" if translation.get("translation_excluded") else None),
            }
        )
        output.append(frozen)
    validate_analysis_rows(output)
    return output
