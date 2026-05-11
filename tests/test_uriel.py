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


class UrielTests(unittest.TestCase):
    def test_compute_urielplus_matrix_with_injected_client(self) -> None:
        class FakeUriel:
            def new_distance(self, distance_type: str, source: str, target: str) -> float:
                self.last_distance_type = distance_type
                return 0.25 if {source, target} == {"por", "eng"} else 0.5

            def get_distance_metric(self) -> str:
                return "angular"

        matrix, metadata = compute_urielplus_matrix(["por", "eng"], uriel=FakeUriel())
        self.assertEqual(matrix["por"]["eng"], 0.25)
        self.assertEqual(matrix["eng"]["por"], 0.25)
        self.assertEqual(metadata["distance_metric"], "angular")

    def test_compute_urielplus_matrix_uv_uses_self_contained_backend(self) -> None:
        completed = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": (
                    '{"matrix":{"por":{"por":0.0,"eng":0.25},"eng":{"por":0.25,"eng":0.0}},'
                    '"metadata":{"source":"urielplus"}}'
                ),
                "stderr": "",
            },
        )()
        with patch("thesis_eval.benchmarks.uriel.subprocess.run", return_value=completed) as run:
            matrix, metadata = compute_urielplus_matrix_uv(["por", "eng"])
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["uv", "run", "--quiet", "--script"])
        script_path = Path(command[4])
        self.assertEqual(script_path.name, "compute_urielplus.py")
        self.assertEqual(script_path.parent.name, "scripts")
        self.assertEqual(matrix["por"]["eng"], 0.25)
        self.assertEqual(metadata["execution_backend"], "uv")

    def test_matrix_round_trip_and_subset(self) -> None:
        matrix = {
            "por": {"por": 0.0, "ara": 0.5},
            "ara": {"por": 0.5, "ara": 0.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "distance.csv"
            write_distance_matrix(path, matrix, ["por", "ara"])
            loaded = load_distance_matrix(path)
            self.assertEqual(loaded["por"]["ara"], 0.5)
            self.assertEqual(set(subset_matrix(loaded, ["ara", "por"])), {"ara", "por"})



if __name__ == "__main__":
    unittest.main()
