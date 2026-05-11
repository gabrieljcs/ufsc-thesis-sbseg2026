from __future__ import annotations

from collections.abc import Iterable, Mapping

ANALYSIS_COLUMNS = [
    "model",
    "model_display_name",
    "model_pair_language",
    "model_alignment_pole",
    "model_access_mode",
    "provider_model_id",
    "provider_model_version",
    "aligned_language",
    "attack_language",
    "prompt_id",
    "run_id",
    "benchmark",
    "prompt_text",
    "source_prompt_text",
    "model_output",
    "model_output_backtranslated",
    "generation_completed",
    "api_http_status",
    "api_failure",
    "api_failure_reason",
    "provider_block",
    "provider_block_reason",
    "strongreject_score",
    "unsafe_binary",
    "refusal_binary",
    "specificity_score",
    "convincingness_score",
    "scoring_failed",
    "scoring_failure_reason",
    "distance",
    "if_score",
    "cons_score",
    "spec_score",
    "input_tokens",
    "tokens_per_char",
    "token_inflation",
    "context_usage_ratio",
    "truncation_risk",
    "benign_control_id",
    "benign_relevance",
    "benign_completeness",
    "benign_coherence",
    "benign_task_completion",
    "translation_blaser_qe_score",
    "translation_blaser_status",
    "roundtrip_drift_score",
    "roundtrip_status",
    "human_audit_status",
    "harm_preservation",
    "translation_revised",
    "translation_excluded",
    "excluded",
    "exclusion_reason",
]

TRANSLATION_LOG_FIELDS = [
    "prompt_id",
    "source_language",
    "target_language",
    "harmful_goal",
    "expected_output_form",
    "fixed_constraints",
    "allowed_adaptation",
    "translator_config",
    "blaser_qe_score",
    "blaser_status",
    "human_audit_status",
    "reference_subset_score",
    "harm_preservation",
    "revision_note",
    "source_text",
    "translated_text",
    "roundtrip_text",
    "roundtrip_drift_score",
    "roundtrip_status",
    "xsts_semantic_equivalence",
    "xsts_attack_strength",
    "xsts_constraint_preservation",
    "xsts_naturalness",
    "xsts_no_harmfulness_drift",
    "translation_revised",
    "translation_excluded",
]


def validate_analysis_rows(rows: Iterable[Mapping[str, object]]) -> None:
    required = set(ANALYSIS_COLUMNS)
    for index, row in enumerate(rows, start=1):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Row {index} missing columns: {sorted(missing)}")
        score = row.get("strongreject_score")
        if score is not None and float(score) < 0:
            raise ValueError(f"Row {index} has negative strongreject_score")


def validate_translation_log(record: Mapping[str, object]) -> None:
    missing = set(TRANSLATION_LOG_FIELDS).difference(record)
    if missing:
        raise ValueError(f"Translation log missing fields: {sorted(missing)}")
    if record["source_language"] != "eng":
        raise ValueError("StrongREJECT translation logs must use source_language='eng'")
    if record["blaser_status"] not in {"pass", "flag", "fail", "pending"}:
        raise ValueError("blaser_status must be pass, flag, fail, or pending")
    if record["human_audit_status"] not in {"not_audited", "audited_pass", "audited_revised", "audited_excluded"}:
        raise ValueError("human_audit_status has an unexpected value")
    if record["harm_preservation"] not in {"preserved", "attenuated", "amplified", "pending"}:
        raise ValueError("harm_preservation has an unexpected value")
    if record["roundtrip_status"] not in {"pass", "flag", "fail", "pending"}:
        raise ValueError("roundtrip_status must be pass, flag, fail, or pending")
