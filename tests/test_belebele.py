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


class BelebeleTests(unittest.TestCase):
    def test_parse_choice_and_compute_spec(self) -> None:
        self.assertEqual(parse_choice("Answer: B"), "B")
        self.assertEqual(
            parse_choice_from_options(
                "Alessandro Safina è italiano",
                {"A": "Un cantante francese", "B": "Alessandro Safina è italiano", "C": "Un atleta", "D": "Uno scrittore"},
            ),
            "B",
        )
        predictions = [
            {"model": "sagui_7b", "language": "por", "item_id": "i1", "prediction": "A", "gold": "A"},
            {"model": "sagui_7b", "language": "ara", "item_id": "i1", "prediction": "B", "gold": "A"},
            {"model": "sagui_7b", "language": "por", "item_id": "i2", "prediction": "C", "gold": "C"},
            {"model": "sagui_7b", "language": "ara", "item_id": "i2", "prediction": "C", "gold": "C"},
        ]
        rows = compute_if_cons(predictions, {"sagui_7b": "por"})
        by_lang = {row["attack_language"]: row for row in rows}
        self.assertEqual(by_lang["por"]["if_score"], 1.0)
        self.assertEqual(by_lang["ara"]["cons_score"], 0.5)
        self.assertIn("spec_score", by_lang["ara"])

    def test_load_belebele_rows_and_prompt(self) -> None:
        self.assertEqual(resolve_belebele_language_code("por"), "por_Latn")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            sample = {
                "question_number": 1,
                "flores_passage": "Trecho",
                "question": "Pergunta?",
                "mc_answer1": "A1",
                "mc_answer2": "A2",
                "mc_answer3": "A3",
                "mc_answer4": "A4",
                "correct_answer_num": "2",
                "dialect": "por_Latn",
            }
            write_jsonl(data_dir / "por_Latn.jsonl", [sample])
            rows = load_belebele_rows(tmp_path, "por")
            prompt = build_belebele_prompt(rows[0])
            predictions = build_belebele_prediction_rows(rows, model="sagui_7b", language="por")
        self.assertEqual(len(rows), 1)
        self.assertIn("Passage:", prompt)
        self.assertIn("Pergunta?", prompt)
        self.assertIn("Answer:", prompt)
        self.assertEqual(predictions[0]["gold"], "B")
        self.assertEqual(predictions[0]["item_id"], "belebele_0001")
        self.assertEqual(predictions[0]["belebele_row_index"], 1)
        self.assertEqual(predictions[0]["choice_b"], "A2")

    def test_cli_predict_belebele_writes_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "belebele" / "data"
            data_dir.mkdir(parents=True)
            write_jsonl(
                data_dir / "por_Latn.jsonl",
                [
                    {
                        "question_number": 1,
                        "flores_passage": "Trecho",
                        "question": "Pergunta?",
                        "mc_answer1": "A1",
                        "mc_answer2": "A2",
                        "mc_answer3": "A3",
                        "mc_answer4": "A4",
                        "correct_answer_num": "2",
                        "dialect": "por_Latn",
                    }
                ],
            )
            output = tmp_path / "predictions.jsonl"
            main(
                [
                    "predict-belebele",
                    "--dataset-dir",
                    str(tmp_path / "belebele"),
                    "--language",
                    "por",
                    "--model",
                    "sagui_7b",
                    "--backend",
                    "mock",
                    "--output",
                    str(output),
                ]
            )
            rows = read_jsonl(output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prediction"], "A")
        self.assertEqual(rows[0]["gold"], "B")
        self.assertEqual(rows[0]["language"], "por")

    def test_cli_predict_belebele_sabia_api_writes_prediction_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "belebele" / "data"
            data_dir.mkdir(parents=True)
            write_jsonl(
                data_dir / "por_Latn.jsonl",
                [
                    {
                        "question_number": 1,
                        "flores_passage": "Trecho",
                        "question": "Pergunta?",
                        "mc_answer1": "A1",
                        "mc_answer2": "A2",
                        "mc_answer3": "A3",
                        "mc_answer4": "A4",
                        "correct_answer_num": "2",
                        "dialect": "por_Latn",
                    }
                ],
            )
            output = tmp_path / "sabia_predictions.jsonl"
            with patch(
                "thesis_eval.cli.iter_generate_with_maritaca",
                return_value=iter(
                    [
                        {
                            "text": "B",
                            "generation_completed": True,
                            "api_http_status": 200,
                            "api_failure": False,
                            "api_failure_reason": None,
                            "provider_block": False,
                            "provider_block_reason": None,
                        }
                    ]
                ),
            ):
                main(
                    [
                        "predict-belebele",
                        "--dataset-dir",
                        str(tmp_path / "belebele"),
                        "--language",
                        "por",
                        "--model",
                        "sabia_3",
                        "--backend",
                        "maritaca",
                        "--output",
                        str(output),
                    ]
                )
            rows = read_jsonl(output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prediction"], "B")
        self.assertEqual(rows[0]["api_http_status"], 200)
        self.assertFalse(rows[0]["api_failure"])
        self.assertTrue(rows[0]["generation_completed"])

    def test_cli_predict_belebele_sabia_api_resumes_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "belebele" / "data"
            data_dir.mkdir(parents=True)
            dataset_rows = [
                {
                    "question_number": 1,
                    "flores_passage": "Trecho 1",
                    "question": "Pergunta 1?",
                    "mc_answer1": "A1",
                    "mc_answer2": "A2",
                    "mc_answer3": "A3",
                    "mc_answer4": "A4",
                    "correct_answer_num": "2",
                    "dialect": "por_Latn",
                },
                {
                    "question_number": 2,
                    "flores_passage": "Trecho 2",
                    "question": "Pergunta 2?",
                    "mc_answer1": "A1",
                    "mc_answer2": "A2",
                    "mc_answer3": "A3",
                    "mc_answer4": "A4",
                    "correct_answer_num": "3",
                    "dialect": "por_Latn",
                },
            ]
            write_jsonl(
                data_dir / "por_Latn.jsonl",
                dataset_rows,
            )
            expected_prompt = build_belebele_prompt(dataset_rows[1])
            partial = tmp_path / "sabia_predictions.partial.jsonl"
            write_jsonl(
                partial,
                [
                    {
                        "model": "sabia_3",
                        "language": "por",
                        "dataset_dialect": "por_Latn",
                        "item_id": "belebele_0001",
                        "question_number": 1,
                        "gold": "B",
                        "raw_output": "B",
                        "prediction": "B",
                        "predicted_choice": "B",
                        "generation_completed": True,
                        "api_http_status": 200,
                        "api_failure": False,
                        "api_failure_reason": None,
                        "provider_block": False,
                        "provider_block_reason": None,
                    }
                ],
            )
            output = tmp_path / "sabia_predictions.jsonl"
            calls: list[list[str]] = []

            def fake_iter(prompts, **kwargs):
                calls.append(list(prompts))
                return iter(
                    [
                        {
                            "text": "C",
                            "generation_completed": True,
                            "api_http_status": 200,
                            "api_failure": False,
                            "api_failure_reason": None,
                            "provider_block": False,
                            "provider_block_reason": None,
                        }
                    ]
                )

            with patch("thesis_eval.cli.iter_generate_with_maritaca", side_effect=fake_iter):
                main(
                    [
                        "predict-belebele",
                        "--dataset-dir",
                        str(tmp_path / "belebele"),
                        "--language",
                        "por",
                        "--model",
                        "sabia_3",
                        "--backend",
                        "maritaca",
                        "--output",
                        str(output),
                    ]
                )
            rows = read_jsonl(output)
        self.assertEqual(calls, [[expected_prompt]])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prediction"], "B")
        self.assertEqual(rows[1]["prediction"], "C")

    def test_repair_belebele_predictions_recovers_option_text_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            dataset_rows = [
                {
                    "question_number": 1,
                    "flores_passage": "Passage",
                    "question": "Question?",
                    "mc_answer1": "Paris",
                    "mc_answer2": "Rome",
                    "mc_answer3": "Madrid",
                    "mc_answer4": "Berlin",
                    "correct_answer_num": "1",
                    "dialect": "eng_Latn",
                },
                {
                    "question_number": 1,
                    "flores_passage": "Second passage",
                    "question": "Second question?",
                    "mc_answer1": "Lisbon",
                    "mc_answer2": "Dublin",
                    "mc_answer3": "Vienna",
                    "mc_answer4": "Prague",
                    "correct_answer_num": "2",
                    "dialect": "eng_Latn",
                },
            ]
            write_jsonl(data_dir / "eng_Latn.jsonl", dataset_rows)
            repaired = repair_belebele_predictions(
                [
                    {
                        "model": "llamantino_2_ultrachat_7b",
                        "language": "eng",
                        "item_id": "belebele_0001",
                        "question_number": 1,
                        "gold": "A",
                        "raw_output": "Rome",
                        "prediction": "Rome",
                    },
                    {
                        "model": "llamantino_2_ultrachat_7b",
                        "language": "eng",
                        "item_id": "belebele_0001",
                        "question_number": 1,
                        "gold": "A",
                        "raw_output": "Dublin",
                        "prediction": "Dublin",
                    }
                ],
                dataset_dir=tmp_path,
            )
        self.assertEqual(repaired[0]["prediction"], "B")
        self.assertEqual(repaired[0]["predicted_choice"], "B")
        self.assertEqual(repaired[0]["choice_b"], "Rome")
        self.assertEqual(repaired[1]["item_id"], "belebele_0002")
        self.assertEqual(repaired[1]["gold"], "B")
        self.assertEqual(repaired[1]["predicted_choice"], "B")

    def test_cli_repair_belebele_predictions_writes_reparsed_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "belebele" / "data"
            data_dir.mkdir(parents=True)
            write_jsonl(
                data_dir / "eng_Latn.jsonl",
                [
                    {
                        "question_number": 1,
                        "flores_passage": "Passage",
                        "question": "Question?",
                        "mc_answer1": "Paris",
                        "mc_answer2": "Rome",
                        "mc_answer3": "Madrid",
                        "mc_answer4": "Berlin",
                        "correct_answer_num": "1",
                        "dialect": "eng_Latn",
                    }
                ],
            )
            predictions = tmp_path / "dirty.jsonl"
            write_jsonl(
                predictions,
                [
                    {
                        "model": "llamantino_2_ultrachat_7b",
                        "language": "eng",
                        "item_id": "belebele_0001",
                        "question_number": 1,
                        "gold": "A",
                        "raw_output": "Rome",
                        "prediction": "Rome",
                    }
                ],
            )
            output = tmp_path / "clean.jsonl"
            main(
                [
                    "repair-belebele-predictions",
                    "--predictions",
                    str(predictions),
                    "--dataset-dir",
                    str(tmp_path / "belebele"),
                    "--output",
                    str(output),
                ]
            )
            rows = read_jsonl(output)
        self.assertEqual(rows[0]["prediction"], "B")
        self.assertEqual(rows[0]["predicted_choice"], "B")



if __name__ == "__main__":
    unittest.main()
