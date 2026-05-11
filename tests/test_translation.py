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


class TranslationTests(unittest.TestCase):
    def test_split_sentences_handles_multi_sentence_prompts(self) -> None:
        from thesis_eval.translation.pipeline import _split_sentences

        self.assertEqual(_split_sentences(""), [])
        self.assertEqual(_split_sentences("Hello world."), ["Hello world."])
        self.assertEqual(
            _split_sentences("First sentence. Second sentence! Third? Fourth."),
            ["First sentence.", "Second sentence!", "Third?", "Fourth."],
        )
        self.assertEqual(
            _split_sentences("How do I make X? I already have Y. Give me Z."),
            ["How do I make X?", "I already have Y.", "Give me Z."],
        )

    def test_split_sentences_handles_arabic_question_mark(self) -> None:
        from thesis_eval.translation.pipeline import _split_sentences

        # Arabic question mark U+061F separates two sentences; comma U+060C must NOT split.
        text = "هل أنت هنا؟ أنا هنا، وأنت هناك."
        sentences = _split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].endswith("؟"))
        self.assertIn("،", sentences[1])

    def test_translation_log_is_complete(self) -> None:
        prompt = PromptRecord(prompt_id="p1", text="placeholder")
        logs = run_translation([prompt], "por", engine="placeholder")
        record = logs[0].to_dict()
        validate_translation_log(record)
        self.assertEqual(record["target_language"], "por")

    def test_translation_qc_sets_blaser_fields(self) -> None:
        log = TranslationLog(
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
            source_text="same text",
            translated_text="same text",
        ).to_dict()
        qc = apply_translation_qc([log])
        self.assertEqual(qc[0]["blaser_status"], "pass")
        self.assertEqual(qc[0]["harm_preservation"], "preserved")

    def test_translation_qc_accepts_blaser_scores(self) -> None:
        log = TranslationLog(
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
            translated_text="translation",
        ).to_dict()
        with patch("thesis_eval.translation.blaser.score_blaser_qe", return_value=[4.2]):
            qc = apply_translation_qc([log], scorer="blaser", pass_threshold=4.0, fail_threshold=3.0)
        self.assertEqual(qc[0]["blaser_qe_score"], 4.2)
        self.assertEqual(qc[0]["blaser_status"], "pass")

    def test_unknown_qc_scorer_fails(self) -> None:
        with self.assertRaises(ValueError):
            apply_translation_qc([], scorer="not-real")

    def test_cli_translate_writes_multiple_language_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prompts = tmp_path / "prompts.jsonl"
            outputs = tmp_path / "translations"
            prompts.write_text('{"prompt_id":"p1","text":"hello"}\n', encoding="utf-8")
            main(
                [
                    "translate",
                    "--prompts",
                    str(prompts),
                    "--target-language",
                    "por",
                    "--target-language",
                    "ara",
                    "--engine",
                    "placeholder",
                    "--output-dir",
                    str(outputs),
                    "--output-template",
                    "{lang}.jsonl",
                ]
            )
            self.assertTrue((outputs / "por.jsonl").exists())
            self.assertTrue((outputs / "ara.jsonl").exists())

    def test_cli_qc_writes_multiple_language_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            ara = tmp_path / "ara.jsonl"
            outputs = tmp_path / "qc"
            write_jsonl(por, [self._translation_log("p1", "por")])
            write_jsonl(ara, [self._translation_log("p1", "ara")])
            main(
                [
                    "translation-qc",
                    "--translations",
                    str(por),
                    "--translations",
                    str(ara),
                    "--scorer",
                    "heuristic",
                    "--output-dir",
                    str(outputs),
                    "--output-template",
                    "{lang}.qc.jsonl",
                ]
            )
            self.assertTrue((outputs / "por.qc.jsonl").exists())
            self.assertTrue((outputs / "ara.qc.jsonl").exists())

    def test_cli_generate_targets_batches_multiple_translation_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            ara = tmp_path / "ara.jsonl"
            output = tmp_path / "generations.jsonl"
            write_jsonl(por, [self._translation_log("p1", "por")])
            write_jsonl(ara, [self._translation_log("p1", "ara")])
            main(
                [
                    "generate-targets",
                    "--translations",
                    str(por),
                    "--translations",
                    str(ara),
                    "--model",
                    "sagui_7b",
                    "--backend",
                    "mock",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(len(output.read_text(encoding="utf-8").strip().splitlines()), 2)

    def test_cli_generate_targets_max_records_limits_retained_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output = tmp_path / "generations.jsonl"
            skipped = dict(self._translation_log("p0", "por"))
            skipped["translation_excluded"] = True
            write_jsonl(
                por,
                [
                    skipped,
                    self._translation_log("p1", "por"),
                    self._translation_log("p2", "por"),
                ],
            )
            main(
                [
                    "generate-targets",
                    "--translations",
                    str(por),
                    "--model",
                    "sagui_7b",
                    "--backend",
                    "mock",
                    "--output",
                    str(output),
                    "--max-records",
                    "1",
                ]
            )
            rows = read_jsonl(output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt_id"], "p1")

    def test_cli_generate_targets_output_dir_writes_per_language_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            ara = tmp_path / "ara.jsonl"
            output_dir = tmp_path / "generations"
            write_jsonl(por, [self._translation_log("p1", "por")])
            write_jsonl(ara, [self._translation_log("p1", "ara")])
            main(
                [
                    "generate-targets",
                    "--translations",
                    str(por),
                    "--translations",
                    str(ara),
                    "--model",
                    "sagui_7b",
                    "--backend",
                    "mock",
                    "--output-dir",
                    str(output_dir),
                    "--output-template",
                    "{model}_{lang}.jsonl",
                ]
            )
            por_rows = read_jsonl(output_dir / "sagui_7b_por.jsonl")
            ara_rows = read_jsonl(output_dir / "sagui_7b_ara.jsonl")
            self.assertEqual(len(por_rows), 1)
            self.assertEqual(len(ara_rows), 1)
            self.assertEqual(por_rows[0]["generation_backend"], "mock")
            self.assertEqual(por_rows[0]["generation_temperature"], 0.0)
            self.assertEqual(por_rows[0]["generation_prompt_strategy"], "raw_prompt")
            self.assertIsNone(por_rows[0]["generation_system_prompt_id"])

    def test_cli_generate_targets_output_dir_reuses_local_generation_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            ara = tmp_path / "ara.jsonl"
            output_dir = tmp_path / "generations"
            write_jsonl(por, [self._translation_log("p1", "por")])
            write_jsonl(ara, [self._translation_log("p1", "ara")])

            def fake_runner(prompts):
                return [f"OUT:{prompt}" for prompt in prompts]

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner) as build_runner, patch(
                "thesis_eval.cli.generate_outputs"
            ) as generate_outputs_mock:
                main(
                    [
                        "generate-targets",
                        "--translations",
                        str(por),
                        "--translations",
                        str(ara),
                        "--model",
                        "sagui_7b",
                        "--backend",
                        "transformers",
                        "--output-dir",
                        str(output_dir),
                        "--output-template",
                        "{model}_{lang}.jsonl",
                        "--skip-tokenizer-metrics",
                    ]
                )

            build_runner.assert_called_once()
            build_kwargs = build_runner.call_args.kwargs
            self.assertEqual(build_kwargs["backend"], "transformers")
            self.assertEqual(len(build_kwargs["prompt_texts"]), 2)
            generate_outputs_mock.assert_not_called()
            por_rows = read_jsonl(output_dir / "sagui_7b_por.jsonl")
            ara_rows = read_jsonl(output_dir / "sagui_7b_ara.jsonl")
            self.assertEqual(por_rows[0]["model_output"], "OUT:same text")
            self.assertEqual(ara_rows[0]["model_output"], "OUT:same text")

    def test_cli_generate_targets_output_dir_keeps_local_partial_checkpoint_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output_dir = tmp_path / "generations"
            write_jsonl(
                por,
                [
                    self._translation_log("p1", "por", translated_text="primeiro"),
                    self._translation_log("p2", "por", translated_text="segundo"),
                ],
            )
            calls: list[list[str]] = []

            def fake_runner(prompts):
                calls.append(list(prompts))
                if len(calls) == 1:
                    return [f"OUT:{prompt}" for prompt in prompts]
                raise RuntimeError("boom")

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    main(
                        [
                            "generate-targets",
                            "--translations",
                            str(por),
                            "--model",
                            "sagui_7b",
                            "--backend",
                            "transformers",
                            "--output-dir",
                            str(output_dir),
                            "--output-template",
                            "{model}_{lang}.jsonl",
                            "--local-batch-size",
                            "1",
                            "--skip-tokenizer-metrics",
                        ]
                    )

            partial_rows = read_jsonl(output_dir / "sagui_7b_por.partial.jsonl")
            self.assertEqual(calls, [["primeiro"], ["segundo"]])
            self.assertEqual(len(partial_rows), 1)
            self.assertEqual(partial_rows[0]["model_output"], "OUT:primeiro")

    def test_cli_generate_targets_output_dir_resumes_local_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output_dir = tmp_path / "generations"
            write_jsonl(
                por,
                [
                    self._translation_log("p1", "por", translated_text="primeiro"),
                    self._translation_log("p2", "por", translated_text="segundo"),
                ],
            )
            partial_path = output_dir / "sagui_7b_por.partial.jsonl"
            write_jsonl(
                partial_path,
                [
                    {
                        "model": "sagui_7b",
                        "aligned_language": "por",
                        "attack_language": "por",
                        "prompt_id": "p1",
                        "run_id": "sagui_7b:por:p1:pilot-1",
                        "prompt_text": "primeiro",
                        "source_prompt_text": "same text",
                        "model_output": "OUT:primeiro",
                    }
                ],
            )
            calls: list[list[str]] = []

            def fake_runner(prompts):
                calls.append(list(prompts))
                return [f"OUT:{prompt}" for prompt in prompts]

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner):
                main(
                    [
                        "generate-targets",
                        "--translations",
                        str(por),
                        "--model",
                        "sagui_7b",
                        "--backend",
                        "transformers",
                        "--output-dir",
                        str(output_dir),
                        "--output-template",
                        "{model}_{lang}.jsonl",
                        "--local-batch-size",
                        "1",
                        "--skip-tokenizer-metrics",
                    ]
                )

            final_rows = read_jsonl(output_dir / "sagui_7b_por.jsonl")
            self.assertEqual(calls, [["segundo"]])
            self.assertEqual(len(final_rows), 2)
            self.assertEqual(final_rows[0]["model_output"], "OUT:primeiro")
            self.assertEqual(final_rows[1]["model_output"], "OUT:segundo")

    def test_cli_generate_targets_single_output_keeps_local_partial_checkpoint_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output = tmp_path / "sagui_7b_por.jsonl"
            write_jsonl(
                por,
                [
                    self._translation_log("p1", "por", translated_text="primeiro"),
                    self._translation_log("p2", "por", translated_text="segundo"),
                ],
            )
            calls: list[list[str]] = []

            def fake_runner(prompts):
                calls.append(list(prompts))
                if len(calls) == 1:
                    return [f"OUT:{prompt}" for prompt in prompts]
                raise RuntimeError("boom")

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    main(
                        [
                            "generate-targets",
                            "--translations",
                            str(por),
                            "--model",
                            "sagui_7b",
                            "--backend",
                            "transformers",
                            "--output",
                            str(output),
                            "--local-batch-size",
                            "1",
                            "--skip-tokenizer-metrics",
                        ]
                    )

            partial_rows = read_jsonl(tmp_path / "sagui_7b_por.partial.jsonl")
            self.assertEqual(calls, [["primeiro"], ["segundo"]])
            self.assertEqual(len(partial_rows), 1)
            self.assertEqual(partial_rows[0]["model_output"], "OUT:primeiro")

    def test_cli_generate_targets_single_output_resumes_local_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output = tmp_path / "sagui_7b_por.jsonl"
            write_jsonl(
                por,
                [
                    self._translation_log("p1", "por", translated_text="primeiro"),
                    self._translation_log("p2", "por", translated_text="segundo"),
                ],
            )
            write_jsonl(
                tmp_path / "sagui_7b_por.partial.jsonl",
                [
                    {
                        "model": "sagui_7b",
                        "aligned_language": "por",
                        "attack_language": "por",
                        "prompt_id": "p1",
                        "run_id": "sagui_7b:por:p1:pilot-1",
                        "prompt_text": "primeiro",
                        "source_prompt_text": "same text",
                        "model_output": "OUT:primeiro",
                    }
                ],
            )
            calls: list[list[str]] = []

            def fake_runner(prompts):
                calls.append(list(prompts))
                return [f"OUT:{prompt}" for prompt in prompts]

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner):
                main(
                    [
                        "generate-targets",
                        "--translations",
                        str(por),
                        "--model",
                        "sagui_7b",
                        "--backend",
                        "transformers",
                        "--output",
                        str(output),
                        "--local-batch-size",
                        "1",
                        "--skip-tokenizer-metrics",
                    ]
                )

            final_rows = read_jsonl(output)
            self.assertEqual(calls, [["segundo"]])
            self.assertEqual(len(final_rows), 2)
            self.assertEqual(final_rows[0]["model_output"], "OUT:primeiro")
            self.assertEqual(final_rows[1]["model_output"], "OUT:segundo")

    def _translation_log(self, prompt_id: str, target_language: str, translated_text: str = "same text") -> dict[str, object]:
        return TranslationLog(
            prompt_id=prompt_id,
            source_language="eng",
            target_language=target_language,
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
            source_text="same text",
            translated_text=translated_text,
        ).to_dict()



if __name__ == "__main__":
    unittest.main()
