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


class StrongRejectImportTests(unittest.TestCase):
    def test_import_local_csv_and_pilot_records(self) -> None:
        csv_text = "category,source,forbidden_prompt\ncat,src,benign fixture prompt\n"
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "strongreject.csv"
            out_path = Path(tmp) / "prompts.jsonl"
            csv_path.write_text(csv_text, encoding="utf-8")
            records = import_strongreject(source="local", input_path=csv_path)
            self.assertEqual(records[0]["prompt_id"], "strongreject_0001")
            self.assertEqual(records[0]["text"], "benign fixture prompt")
            write_prompt_records(out_path, records)
            loaded = load_prompt_records(out_path)
            self.assertEqual(loaded[0]["benchmark"], "strongreject")

    def test_verify_strongreject_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prompts.jsonl"
            pilot = Path(tmp) / "pilot_prompts.jsonl"
            records = [
                {
                    "prompt_id": "strongreject_0001",
                    "text": "benign fixture prompt",
                    "harmful_goal": "pending",
                    "expected_output_form": "other",
                    "fixed_constraints": "pending",
                    "allowed_adaptation": "fluency",
                    "benchmark": "strongreject",
                }
            ]
            write_prompt_records(output, records)
            write_prompt_records(pilot, records)
            spec = AssetSpec(
                name="strongreject_fixture",
                kind="strongreject",
                group="datasets",
                groups=("datasets", "jailbreaks"),
                description="fixture",
                output=output,
                pilot_output=pilot,
                required_files=(output, pilot),
                required_any_files=((output,),),
                required_columns=("prompt_id", "text", "benchmark"),
            )
            result = verify_asset(spec)
            self.assertTrue(result["ok"])



if __name__ == "__main__":
    unittest.main()
