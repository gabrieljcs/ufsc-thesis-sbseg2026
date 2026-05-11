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


class AuditTests(unittest.TestCase):
    def test_audit_queue_and_import(self) -> None:
        log = TranslationLog(
            prompt_id="p1",
            source_language="eng",
            target_language="ara",
            harmful_goal="goal",
            expected_output_form="other",
            fixed_constraints="none",
            allowed_adaptation="fluency",
            translator_config={"engine": "placeholder"},
            blaser_qe_score=0.5,
            blaser_status="flag",
            human_audit_status="not_audited",
            reference_subset_score=None,
            harm_preservation="pending",
            revision_note=None,
            source_text="source",
            translated_text="translation",
        ).to_dict()
        queue = build_audit_queue([log])
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["audit_plan"], "plan_b")
        self.assertEqual(queue[0]["audit_method"], "uncalibrated_llm_xsts")
        updated = apply_audit_decisions(
            [log],
            [
                {
                    "prompt_id": "p1",
                    "target_language": "ara",
                    "audit_decision": "pass",
                    "harm_preservation": "preserved",
                    "translated_text": "translation",
                }
            ],
        )
        self.assertEqual(updated[0]["human_audit_status"], "audited_pass")
        self.assertEqual(updated[0]["harm_preservation"], "preserved")
        self.assertFalse(updated[0]["translation_revised"])

    def test_plan_a_audit_queue_marks_tiers_and_exclusions(self) -> None:
        logs = [
            self._audit_log("p1", "ara"),
            self._audit_log("p2", "fin"),
        ]
        queue = build_audit_queue(logs, audit_plan="plan_a")
        by_lang = {row["target_language"]: row for row in queue}
        self.assertEqual(by_lang["ara"]["audit_tier"], "tier_a")
        self.assertEqual(by_lang["ara"]["audit_method"], "human_xsts")
        self.assertEqual(by_lang["fin"]["audit_tier"], "tier_b")
        self.assertTrue(by_lang["fin"]["caveat_language"])
        updated = apply_audit_decisions(
            logs,
            [{"prompt_id": "p2", "target_language": "fin", "audit_decision": "exclude"}],
        )
        self.assertTrue(updated[1]["translation_excluded"])
        self.assertEqual(updated[1]["human_audit_status"], "audited_excluded")

    def test_llm_audit_mock_fills_decision_columns(self) -> None:
        queue = build_audit_queue([self._audit_log("p1", "ara")])
        judged = judge_audit_rows(queue, provider="mock", model="mock-xsts")
        self.assertEqual(judged[0]["xsts_score"], 4)
        self.assertEqual(judged[0]["xsts_judge_model"], "mock-xsts")
        self.assertEqual(judged[0]["audit_decision"], "pass")
        self.assertTrue(judged[0]["xsts_note"])
        prompt = build_xsts_prompt(queue[0])
        self.assertIn("XSTS", prompt)
        self.assertIn("Source (English)", prompt)

    def _audit_log(self, prompt_id: str, target_language: str) -> dict[str, object]:
        return TranslationLog(
            prompt_id=prompt_id,
            source_language="eng",
            target_language=target_language,
            harmful_goal="goal",
            expected_output_form="other",
            fixed_constraints="none",
            allowed_adaptation="fluency",
            translator_config={"engine": "placeholder"},
            blaser_qe_score=0.5,
            blaser_status="flag",
            human_audit_status="not_audited",
            reference_subset_score=None,
            harm_preservation="pending",
            revision_note=None,
            source_text="source",
            translated_text="translation",
        ).to_dict()

    def test_blaser_calibration(self) -> None:
        result = calibrate_blaser_thresholds(
            [
                {"blaser_qe_score": "4.2", "audit_decision": "pass"},
                {"blaser_qe_score": "3.1", "audit_decision": "revise"},
            ]
        )
        self.assertEqual(result["audited_pass"], 1)
        self.assertLessEqual(result["fail_threshold"], result["pass_threshold"])



if __name__ == "__main__":
    unittest.main()
