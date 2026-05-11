from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from thesis_eval.assets import AssetSpec, asset_status, download_asset, load_assets, select_assets, verify_asset
from thesis_eval.benchmarks.strongreject import import_strongreject, load_prompt_records, write_prompt_records
from thesis_eval.benchmarks.uriel import (
    compute_urielplus_matrix,
    compute_urielplus_matrix_uv,
    load_distance_matrix,
    subset_matrix,
    write_distance_matrix,
)
from thesis_eval.cli import main
from thesis_eval.config import load_config
from thesis_eval.analysis.dataset import build_frozen_rows
from thesis_eval.analysis.reporting import (
    counterfactual_safety_by_aligned_language,
    prereg_distance_slope_retention,
    prereg_falsification_summary,
    reference_distance_curve,
)
from thesis_eval.benchmarks.belebele import (
    build_belebele_prediction_rows,
    build_belebele_prompt,
    compute_if_cons,
    load_belebele_rows,
    parse_choice,
    parse_choice_from_options,
    repair_belebele_predictions,
    resolve_belebele_language_code,
)
from thesis_eval.evaluation.strongreject import parse_strongreject_rubric_judge_response, score_generation_rows
from thesis_eval.io import read_jsonl, write_jsonl
from thesis_eval.metrics.asr import summarize_labels
from thesis_eval.metrics.spec import add_spec_scores, z_scores
from thesis_eval.metrics.tokenizer import attach_tokenizer_metrics
from thesis_eval.models.generation import (
    _maybe_wrap_vllm_attention_backend_error,
    _model_attention_softcap,
    _parse_vllm_estimated_max_model_len,
    _retry_vllm_max_model_len_from_error,
    _select_vllm_attention_backend_override,
    build_generation_runner,
    build_prompt_messages,
    describe_prompt_format,
    estimate_model_asset_state,
    format_generation_prompt,
    generate_outputs,
)
from thesis_eval.openai_batch import retrieve_batch
from thesis_eval.runtime import load_runtime_profiles
from thesis_eval.schema import validate_analysis_rows, validate_translation_log
from thesis_eval.translation.backtranslate import attach_response_backtranslations
from thesis_eval.translation.audit import apply_audit_decisions, build_audit_queue
from thesis_eval.translation.calibration import calibrate_blaser_thresholds
from thesis_eval.translation.llm_audit import build_xsts_prompt, judge_audit_rows
from thesis_eval.translation.log import TranslationLog
from thesis_eval.translation.pipeline import PromptRecord, apply_translation_qc, run_translation


class DatasetBuilderTests(unittest.TestCase):
    def test_build_frozen_rows_from_scored_generation(self) -> None:
        cfg = load_config()
        scored = score_generation_rows(
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "por",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "prompt",
                    "source_prompt_text": "source",
                    "model_output": "safe refusal",
                    "model_output_backtranslated": None,
                    "generation_completed": True,
                    "api_http_status": None,
                    "api_failure": False,
                    "api_failure_reason": None,
                    "provider_block": False,
                    "provider_block_reason": None,
                    "benchmark": "strongreject",
                    "distance": 0.0,
                    "input_tokens": 10,
                    "tokens_per_char": 1.0,
                    "token_inflation": 1.0,
                    "excluded": False,
                    "exclusion_reason": None,
                }
            ],
            evaluator="mock",
        )
        translations = [
            TranslationLog(
                prompt_id="p1",
                source_language="eng",
                target_language="por",
                harmful_goal="goal",
                expected_output_form="other",
                fixed_constraints="none",
                allowed_adaptation="fluency",
                translator_config={"engine": "placeholder"},
                blaser_qe_score=None,
                blaser_status="pending",
                human_audit_status="not_audited",
                reference_subset_score=None,
                harm_preservation="pending",
                revision_note=None,
                source_text="source",
                translated_text="prompt",
            ).to_dict()
        ]
        rows = build_frozen_rows(scored, cfg, translations=translations)
        validate_analysis_rows(rows)
        self.assertEqual(rows[0]["model_alignment_pole"], "weak")
        self.assertEqual(rows[0]["translation_blaser_status"], "pending")



if __name__ == "__main__":
    unittest.main()
