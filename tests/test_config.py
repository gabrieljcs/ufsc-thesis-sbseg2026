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


class ConfigTests(unittest.TestCase):
    def test_load_config(self) -> None:
        cfg = load_config()
        self.assertEqual(len(cfg.models), 9)
        self.assertEqual(len(cfg.attack_languages), 13)
        self.assertEqual(cfg.aligned_language["sagui_7b"], "por")
        self.assertEqual(cfg.aligned_language["llamantino_2_ultrachat_7b"], "ita")
        self.assertEqual(cfg.models["sabia_3"]["access_mode"], "api")
        self.assertEqual(cfg.models["llama3_1_8b_reference"]["alignment_pole"], "reference")
        self.assertEqual(cfg.models["llama3_1_8b_reference"]["analysis_role"], "reference_baseline")

    def test_runtime_profiles(self) -> None:
        profiles = load_runtime_profiles()
        self.assertEqual(profiles["macbook"].target_backend, "transformers")
        self.assertEqual(profiles["rtx_5070_ti"].target_backend, "vllm")
        self.assertEqual(profiles["runpod_a100"].target_backend, "vllm")
        self.assertEqual(profiles["runpod_a100"].device, "cuda")
        self.assertEqual(profiles["tpu"].device, "xla")
        self.assertEqual(profiles["trc_tpu"].device, "xla")

    def test_assets_manifest(self) -> None:
        assets = load_assets()
        self.assertIn("strongreject", assets)
        self.assertIn("nllb_200", assets)
        self.assertIn("sagui_7b", assets)
        self.assertIn("sabia_3", assets)
        self.assertIn("llamantino_2_ultrachat_7b", assets)
        self.assertIn("llama3_1_8b_reference", assets)
        self.assertEqual(select_assets(groups=["translation"])[0].group, "translation")
        self.assertEqual({asset.name for asset in select_assets(groups=["jailbreaks"])}, {"strongreject", "multijail"})
        self.assertEqual({asset.name for asset in select_assets(groups=["reference_targets"])}, {"llama3_1_8b_reference"})
        status = asset_status(assets["nllb_200"])
        self.assertEqual(status["name"], "nllb_200")

    def test_incomplete_hf_asset_resumes_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "partial-model"
            local_dir.mkdir()
            (local_dir / ".cache" / "huggingface").mkdir(parents=True)
            (local_dir / "config.json").write_text("{}", encoding="utf-8")
            asset = AssetSpec(
                name="partial",
                kind="hf_model",
                group="targets",
                groups=("targets",),
                description="partial fixture",
                repo_id="org/model",
                local_dir=local_dir,
                required_files=(local_dir / "config.json",),
            )
            with patch("thesis_eval.assets._snapshot_download") as snapshot_download:
                result = download_asset(asset)
            snapshot_download.assert_called_once()
            self.assertEqual(result["action"], "downloaded")

    def test_sharded_hf_asset_requires_all_indexed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "partial-sharded-model"
            (local_dir / ".cache" / "huggingface").mkdir(parents=True)
            (local_dir / "config.json").write_text("{}", encoding="utf-8")
            (local_dir / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}',
                encoding="utf-8",
            )
            (local_dir / "model-00001-of-00002.safetensors").write_text("partial", encoding="utf-8")
            asset = AssetSpec(
                name="partial_sharded",
                kind="hf_model",
                group="targets",
                groups=("targets",),
                description="partial sharded fixture",
                repo_id="org/model",
                local_dir=local_dir,
                required_files=(local_dir / "config.json",),
                required_any_files=((local_dir / "model.safetensors.index.json",),),
            )
            result = verify_asset(asset)
            self.assertFalse(result["ok"])
            weights_check = next(check for check in result["checks"] if check["name"] == "model weights")
            self.assertEqual(weights_check["missing_shard_count"], 1)



if __name__ == "__main__":
    unittest.main()
