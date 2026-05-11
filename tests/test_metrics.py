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


class MetricsTests(unittest.TestCase):
    def test_asr_summary_maps_invalid_out_of_success(self) -> None:
        summary = summarize_labels(["unsafe", "safe", "invalid", "unsafe"])
        self.assertEqual(summary["unsafe"], 2)
        self.assertEqual(summary["invalid"], 1)
        self.assertAlmostEqual(summary["asr"], 0.5)

    def test_spec_scores(self) -> None:
        rows = [
            {"if_score": 0.8, "cons_score": 0.7},
            {"if_score": 0.6, "cons_score": 0.9},
        ]
        enriched = add_spec_scores(rows)
        self.assertIn("spec_score", enriched[0])
        self.assertEqual(z_scores([1.0, 1.0]), [0.0, 0.0])

    def test_tokenizer_metrics_use_aligned_language_baseline(self) -> None:
        class FakeTokenizer:
            chat_template = "fake"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                assert tokenize
                assert add_generation_prompt
                return [0 for _ in messages[0]["content"].split()]

        rows = [
            {
                "model": "sagui_7b",
                "benchmark": "strongreject",
                "prompt_id": "p1",
                "aligned_language": "por",
                "attack_language": "por",
                "prompt_text": "um dois",
            },
            {
                "model": "sagui_7b",
                "benchmark": "strongreject",
                "prompt_id": "p1",
                "aligned_language": "por",
                "attack_language": "ara",
                "prompt_text": "um dois tres quatro",
            },
        ]
        enriched = attach_tokenizer_metrics(rows, FakeTokenizer())
        self.assertEqual(enriched[0]["input_tokens"], 2)
        self.assertEqual(enriched[0]["token_inflation"], 1.0)
        self.assertEqual(enriched[1]["input_tokens"], 4)
        self.assertEqual(enriched[1]["token_inflation"], 2.0)

    def test_attach_tokenizer_metrics_skips_api_model_without_tokenizer_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generations = tmp_path / "sabia_rows.jsonl"
            output = tmp_path / "sabia_rows.with_metrics.jsonl"
            write_jsonl(
                generations,
                [
                    {
                        "model": "sabia_3",
                        "benchmark": "strongreject",
                        "prompt_id": "p1",
                        "aligned_language": "por",
                        "attack_language": "ara",
                        "prompt_text": "prompt",
                        "model_output": "response",
                    }
                ],
            )
            main(
                [
                    "attach-tokenizer-metrics",
                    "--generations",
                    str(generations),
                    "--output",
                    str(output),
                ]
            )
            rows = read_jsonl(output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "sabia_3")
            self.assertNotIn("input_tokens", rows[0])

    def test_tokenizer_metrics_use_manual_prompt_fallback_without_chat_template(self) -> None:
        class FakeTokenizer:
            name_or_path = "swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA"
            chat_template = None

            def __call__(self, text, add_special_tokens):
                self.last_text = text
                self.last_add_special_tokens = add_special_tokens
                return {"input_ids": [0 for _ in text.split()]}

        tokenizer = FakeTokenizer()
        rows = [
            {
                "model": "llamantino_2_ultrachat_7b",
                "benchmark": "strongreject",
                "prompt_id": "p1",
                "aligned_language": "ita",
                "attack_language": "ita",
                "prompt_text": "Ciao mondo",
            }
        ]
        attach_tokenizer_metrics(rows, tokenizer)
        self.assertTrue(tokenizer.last_add_special_tokens)
        self.assertIn("[INST]", tokenizer.last_text)
        self.assertIn("Sei un assistente disponibile", tokenizer.last_text)

    def test_prereg_distance_slope_retention(self) -> None:
        rows = [
            {
                "model": "llamantino_2_ultrachat_7b",
                "model_pair_language": "ita",
                "model_alignment_pole": "weak",
                "distance": distance,
                "unsafe_binary": unsafe,
                "excluded": False,
            }
            for distance, unsafe in [(0.0, 0), (0.0, 0), (1.0, 1), (1.0, 1)]
        ] + [
            {
                "model": "llamantino_anita_8b",
                "model_pair_language": "ita",
                "model_alignment_pole": "strong",
                "distance": distance,
                "unsafe_binary": unsafe,
                "excluded": False,
            }
            for distance, unsafe in [(0.0, 0), (0.0, 0), (1.0, 1), (1.0, 1)]
        ]
        pair_rows = prereg_distance_slope_retention(rows)
        self.assertTrue(pair_rows[0]["slope_sign_retained"])
        summary = prereg_falsification_summary(rows)
        self.assertEqual(summary[0]["pairs_with_slope_sign_retained"], 1)

    def test_reference_baseline_descriptive_tables(self) -> None:
        rows = [
            {
                "model": "llama3_1_8b_reference",
                "model_alignment_pole": "reference",
                "aligned_language": "eng",
                "attack_language": "por",
                "unsafe_binary": unsafe,
                "strongreject_score": float(unsafe),
                "refusal_binary": 1 - unsafe,
                "distance": 0.25,
                "excluded": False,
            }
            for unsafe in [1, 0]
        ] + [
            {
                "model": "sagui_7b",
                "model_alignment_pole": "weak",
                "aligned_language": "por",
                "attack_language": "por",
                "unsafe_binary": unsafe,
                "strongreject_score": float(unsafe),
                "refusal_binary": 1 - unsafe,
                "distance": 0.0,
                "excluded": False,
            }
            for unsafe in [0, 0]
        ]
        curve = reference_distance_curve(rows)
        self.assertEqual(curve[0]["attack_language"], "por")
        self.assertEqual(curve[0]["distance_from_english"], 0.25)
        safety = counterfactual_safety_by_aligned_language(rows)
        self.assertEqual(safety[0]["model"], "sagui_7b")
        self.assertEqual(safety[0]["asr_gap_model_minus_reference"], -0.5)



if __name__ == "__main__":
    unittest.main()
