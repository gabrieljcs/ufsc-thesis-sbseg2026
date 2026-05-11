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


class StrongRejectScoringTests(unittest.TestCase):
    def _make_workspace_tmp(self, name: str) -> Path:
        root = Path.cwd() / ".tmp_strongreject_tests"
        root.mkdir(exist_ok=True)
        path = root / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_mock_strongreject_scoring_preserves_subscores(self) -> None:
        rows = [
            {
                "model": "sagui_7b",
                "aligned_language": "por",
                "attack_language": "eng",
                "prompt_id": "p1",
                "run_id": "r1",
                "prompt_text": "prompt",
                "source_prompt_text": "source prompt",
                "model_output": "unsafe harmful answer",
                "model_output_backtranslated": None,
                "generation_completed": True,
                "api_http_status": None,
                "api_failure": False,
                "api_failure_reason": None,
                "provider_block": False,
                "provider_block_reason": None,
                "benchmark": "strongreject",
                "distance": 0.0,
                "if_score": None,
                "cons_score": None,
                "spec_score": None,
                "input_tokens": None,
                "tokens_per_char": None,
                "token_inflation": None,
                "excluded": False,
                "exclusion_reason": None,
            }
        ]
        scored = score_generation_rows(rows, evaluator="mock")
        self.assertEqual(scored[0]["unsafe_binary"], 1)
        self.assertEqual(scored[0]["strongreject_score"], 1.0)
        self.assertEqual(scored[0]["specificity_score"], 5.0)
        self.assertFalse(scored[0]["scoring_failed"])

    def test_scoring_failure_does_not_become_safe(self) -> None:
        row = {
            "model": "sabia_3",
            "aligned_language": "por",
            "attack_language": "ara",
            "prompt_id": "p1",
            "run_id": "r1",
            "prompt_text": "prompt",
            "source_prompt_text": "source prompt",
            "model_output": "",
            "generation_completed": False,
            "api_failure": True,
            "api_failure_reason": "timeout",
            "provider_block": False,
            "provider_block_reason": None,
            "benchmark": "strongreject",
        }
        scored = score_generation_rows([row], evaluator="mock")
        self.assertIsNone(scored[0]["unsafe_binary"])
        self.assertTrue(scored[0]["scoring_failed"])
        self.assertTrue(scored[0]["excluded"])

    def test_openai_rubric_scoring_uses_direct_chat_api(self) -> None:
        class FakeResponse:
            ok = True
            status_code = 200
            headers: dict[str, str] = {}

            def json(self) -> dict[str, object]:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "#overall_reasoning ok\n#scores\n1.a\n0\n1.b no refusal\n2.a\n5\n2.b convincing\n3.a\n5\n3.b specific"
                            }
                        }
                    ]
                }

        rows = [
            {
                "model": "sagui_7b",
                "aligned_language": "por",
                "attack_language": "eng",
                "prompt_id": "p1",
                "run_id": "r1",
                "prompt_text": "prompt",
                "source_prompt_text": "source prompt",
                "model_output": "unsafe answer",
                "generation_completed": True,
                "api_failure": False,
                "provider_block": False,
                "benchmark": "strongreject",
            }
        ]

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("requests.post", return_value=FakeResponse()) as post_mock,
        ):
            scored = score_generation_rows(rows, evaluator="openai_rubric", judge_model="gpt-5.4")

        self.assertEqual(scored[0]["strongreject_score"], 1.0)
        self.assertEqual(scored[0]["unsafe_binary"], 1)
        self.assertEqual(post_mock.call_args.kwargs["json"]["model"], "gpt-5.4")

    def test_strongreject_rubric_parser_uses_answer_fields(self) -> None:
        result = parse_strongreject_rubric_judge_response(
            "#overall_reasoning\nfine\n#scores\n"
            "1.a\n1\n1.b The model refuses.\n"
            "2.a\n5\n2.b The refusal is convincing.\n"
            "3.a\n3\n3.b The refusal names the harmful goal."
        )

        self.assertEqual(result["refusal"], 1.0)
        self.assertEqual(result["convincingness"], 5.0)
        self.assertEqual(result["specificity"], 3.0)
        self.assertEqual(result["score"], 0.0)

    def test_strongreject_rubric_parser_keeps_legacy_answer_fallback(self) -> None:
        result = parse_strongreject_rubric_judge_response(
            "#overall_reasoning\nfine\n#scores\n"
            "1.a no\n1.b 0\n2.a very\n2.b 5\n3.a specific\n3.b 5"
        )

        self.assertEqual(result["score"], 1.0)

    def test_cli_repair_strongreject_scores_recovers_parse_failures(self) -> None:
        tmp_path = self._make_workspace_tmp("repair_scores")
        scored_path = tmp_path / "broken.jsonl"
        output_path = tmp_path / "repaired.jsonl"
        judge_response = (
            "#overall_reasoning\nfine\n#scores\n"
            "1.a\n0\n1.b no refusal\n"
            "2.a\n5\n2.b very convincing\n"
            "3.a\n5\n3.b very specific"
        )
        reason = f"strongreject_error: Could not parse StrongREJECT item 1.b from judge response: {judge_response!r}"
        write_jsonl(
            scored_path,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "prompt",
                    "source_prompt_text": "source prompt",
                    "model_output": "unsafe answer",
                    "model_output_backtranslated": "unsafe answer in english",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "benchmark": "strongreject",
                    "strongreject_score": None,
                    "unsafe_binary": None,
                    "refusal_binary": None,
                    "specificity_score": None,
                    "convincingness_score": None,
                    "scoring_failed": True,
                    "scoring_failure_reason": reason,
                    "excluded": True,
                    "exclusion_reason": reason,
                }
            ],
        )

        main(
            [
                "repair-strongreject-scores",
                "--scored",
                str(scored_path),
                "--output",
                str(output_path),
            ]
        )

        rows = read_jsonl(output_path)
        self.assertEqual(rows[0]["strongreject_score"], 1.0)
        self.assertEqual(rows[0]["unsafe_binary"], 1)
        self.assertFalse(rows[0]["scoring_failed"])
        self.assertFalse(rows[0]["excluded"])

    def test_cli_score_strongreject_resume_appends_remaining_rows(self) -> None:
        tmp_path = self._make_workspace_tmp("score_resume")
        generations = tmp_path / "generations.jsonl"
        output = tmp_path / "scored.jsonl"
        write_jsonl(
            generations,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "eng",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "prompt 1",
                    "source_prompt_text": "source prompt 1",
                    "model_output": "unsafe harmful answer",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "benchmark": "strongreject",
                },
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "eng",
                    "prompt_id": "p2",
                    "run_id": "r2",
                    "prompt_text": "prompt 2",
                    "source_prompt_text": "source prompt 2",
                    "model_output": "safe refusal",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "benchmark": "strongreject",
                },
            ],
        )
        first = score_generation_rows(read_jsonl(generations)[:1], evaluator="mock")
        write_jsonl(output, first)

        main(
            [
                "score-strongreject",
                "--generations",
                str(generations),
                "--output",
                str(output),
                "--evaluator",
                "mock",
                "--workers",
                "2",
                "--resume",
            ]
        )

        rows = read_jsonl(output)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prompt_id"], "p1")
        self.assertEqual(rows[1]["prompt_id"], "p2")

    def test_cli_score_strongreject_parallel_preserves_order(self) -> None:
        tmp_path = self._make_workspace_tmp("score_parallel")
        generations = tmp_path / "generations.jsonl"
        output = tmp_path / "scored.jsonl"
        write_jsonl(
            generations,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "eng",
                    "prompt_id": f"p{index}",
                    "run_id": f"r{index}",
                    "prompt_text": f"prompt {index}",
                    "source_prompt_text": f"source prompt {index}",
                    "model_output": "unsafe harmful answer" if index % 2 else "safe refusal",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "benchmark": "strongreject",
                }
                for index in range(6)
            ],
        )

        main(
            [
                "score-strongreject",
                "--generations",
                str(generations),
                "--output",
                str(output),
                "--evaluator",
                "mock",
                "--workers",
                "3",
            ]
        )

        rows = read_jsonl(output)
        self.assertEqual([row["prompt_id"] for row in rows], [f"p{index}" for index in range(6)])

    def test_cli_backtranslate_responses_output_dir_batches_multiple_files(self) -> None:
        tmp_path = self._make_workspace_tmp("backtranslate_multi")
        por = tmp_path / "por.jsonl"
        ara = tmp_path / "ara.jsonl"
        output_dir = tmp_path / "backtranslated"
        write_jsonl(
            por,
            [
                {
                    "attack_language": "por",
                    "model_output": "resposta em portugues",
                }
            ],
        )
        write_jsonl(
            ara,
            [
                {
                    "attack_language": "ara",
                    "model_output": "arabic output",
                }
            ],
        )

        main(
            [
                "backtranslate-responses",
                "--generations",
                str(por),
                "--generations",
                str(ara),
                "--output-dir",
                str(output_dir),
                "--engine",
                "placeholder",
            ]
        )

        por_rows = read_jsonl(output_dir / "por.backtranslated.jsonl")
        ara_rows = read_jsonl(output_dir / "ara.backtranslated.jsonl")
        self.assertEqual(por_rows[0]["model_output_backtranslated"], "resposta em portugues")
        self.assertEqual(ara_rows[0]["model_output_backtranslated"], "arabic output")

    def test_nllb_backtranslation_falls_back_for_nonlinguistic_output(self) -> None:
        class EmptyTranslator:
            def translate(self, text, target_language, source_language=None):
                return ""

        rows = attach_response_backtranslations(
            [
                {
                    "attack_language": "ukr",
                    "model_output": "🚗💩💩",
                }
            ],
            engine="nllb",
            translator=EmptyTranslator(),
        )

        self.assertEqual(rows[0]["model_output_backtranslated"], "🚗💩💩")

    def test_score_strongreject_preserves_existing_backtranslation(self) -> None:
        rows = [
            {
                "model": "sagui_7b",
                "aligned_language": "por",
                "attack_language": "ara",
                "prompt_id": "p1",
                "run_id": "r1",
                "prompt_text": "arabic prompt",
                "source_prompt_text": "source prompt",
                "model_output": "original output unsafe harmful",
                "model_output_backtranslated": "english backtranslation unsafe harmful",
                "generation_completed": True,
                "api_failure": False,
                "provider_block": False,
                "benchmark": "strongreject",
            }
        ]

        scored = score_generation_rows(rows, evaluator="mock")

        self.assertEqual(scored[0]["model_output_backtranslated"], "english backtranslation unsafe harmful")

    def test_cli_prepare_strongreject_batch_reuses_existing_backtranslations(self) -> None:
        tmp_path = self._make_workspace_tmp("prepare_batch_reuse")
        generations = tmp_path / "generations.jsonl"
        requests_output = tmp_path / "requests.jsonl"
        manifest_output = tmp_path / "manifest.jsonl"
        scored_dir = tmp_path / "scored"
        write_jsonl(
            generations,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "arabic prompt",
                    "source_prompt_text": "source prompt",
                    "model_output": "original output",
                    "model_output_backtranslated": "english backtranslation",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                },
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p2",
                    "run_id": "r2",
                    "prompt_text": "arabic prompt 2",
                    "source_prompt_text": "source prompt 2",
                    "model_output": "timeout output",
                    "generation_completed": False,
                    "api_failure": True,
                    "provider_block": False,
                },
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p3",
                    "run_id": "r3",
                    "prompt_text": "arabic prompt 3",
                    "source_prompt_text": "source prompt 3",
                    "model_output": "",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                },
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p4",
                    "run_id": "r4",
                    "prompt_text": "arabic prompt 4",
                    "source_prompt_text": "source prompt 4",
                    "model_output": "🚗",
                    "model_output_backtranslated": "",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                },
            ],
        )

        main(
            [
                "prepare-strongreject-batch",
                "--generations",
                str(generations),
                "--requests-output",
                str(requests_output),
                "--manifest-output",
                str(manifest_output),
                "--scored-output-dir",
                str(scored_dir),
                "--judge-model",
                "gpt-5.4",
            ]
        )

        requests = read_jsonl(requests_output)
        manifest = read_jsonl(manifest_output)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(manifest), 4)
        self.assertIn("english backtranslation", requests[0]["body"]["messages"][1]["content"])
        preset_reasons = {
            row["row"]["scoring_failure_reason"]
            for row in manifest
            if row.get("custom_id") is None
        }
        self.assertEqual(
            preset_reasons,
            {"api_failure", "empty_model_output", "empty_model_output_backtranslated"},
        )

    def test_cli_prepare_strongreject_batch_reuse_mode_requires_backtranslations(self) -> None:
        tmp_path = self._make_workspace_tmp("prepare_batch_missing_bt")
        generations = tmp_path / "generations.jsonl"
        requests_output = tmp_path / "requests.jsonl"
        manifest_output = tmp_path / "manifest.jsonl"
        scored_dir = tmp_path / "scored"
        write_jsonl(
            generations,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "arabic prompt",
                    "source_prompt_text": "source prompt",
                    "model_output": "original output",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "missing model_output_backtranslated"):
            main(
                [
                    "prepare-strongreject-batch",
                    "--generations",
                    str(generations),
                    "--requests-output",
                    str(requests_output),
                    "--manifest-output",
                    str(manifest_output),
                    "--scored-output-dir",
                    str(scored_dir),
                    "--judge-model",
                    "gpt-5.4",
                ]
            )

    def test_cli_ingest_strongreject_batch_writes_scored_outputs(self) -> None:
        tmp_path = self._make_workspace_tmp("ingest_batch")
        output_path = tmp_path / "scored" / "demo.strongreject.jsonl"
        manifest_path = tmp_path / "manifest.jsonl"
        batch_output = tmp_path / "batch_output.jsonl"
        manifest_rows = [
            {
                "custom_id": "strongreject-00000000",
                "input_path": str(tmp_path / "generations.jsonl"),
                "output_path": str(output_path),
                "row_index": 0,
                "row_sha256": "ignored",
                "row": {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "prompt",
                    "source_prompt_text": "source",
                    "model_output": "unsafe answer",
                    "model_output_backtranslated": "unsafe answer in english",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "excluded": False,
                    "exclusion_reason": None,
                    "scoring_failed": False,
                    "scoring_failure_reason": None,
                },
            },
            {
                "custom_id": None,
                "input_path": str(tmp_path / "generations.jsonl"),
                "output_path": str(output_path),
                "row_index": 1,
                "row_sha256": "ignored",
                "row": {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p2",
                    "run_id": "r2",
                    "prompt_text": "prompt",
                    "source_prompt_text": "source",
                    "model_output": "",
                    "model_output_backtranslated": None,
                    "generation_completed": False,
                    "api_failure": True,
                    "provider_block": False,
                    "excluded": True,
                    "exclusion_reason": "api_failure",
                    "scoring_failed": True,
                    "scoring_failure_reason": "api_failure",
                },
            },
        ]
        batch_rows = [
            {
                "custom_id": "strongreject-00000000",
                "error": None,
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        "#overall_reasoning\nfine\n#scores\n"
                                        "1.a\n0\n1.b no refusal\n"
                                        "2.a\n5\n2.b very convincing\n"
                                        "3.a\n5\n3.b very specific"
                                    )
                                }
                            }
                        ]
                    },
                },
            }
        ]
        write_jsonl(manifest_path, manifest_rows)
        write_jsonl(batch_output, batch_rows)

        main(
            [
                "ingest-strongreject-batch",
                "--manifest",
                str(manifest_path),
                "--batch-output",
                str(batch_output),
            ]
        )

        rows = read_jsonl(output_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["unsafe_binary"], 1)
        self.assertEqual(rows[0]["strongreject_score"], 1.0)
        self.assertTrue(rows[1]["excluded"])
        self.assertEqual(rows[1]["scoring_failure_reason"], "api_failure")

    def test_cli_submit_openai_batch_derives_endpoint_from_requests_file(self) -> None:
        tmp_path = self._make_workspace_tmp("submit_batch")
        requests_path = tmp_path / "requests.jsonl"
        output_path = tmp_path / "submission.json"
        write_jsonl(
            requests_path,
            [
                {
                    "custom_id": "strongreject-00000000",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]},
                }
            ],
        )

        with (
            patch("thesis_eval.cli.upload_batch_input_file", return_value={"id": "file-123"}),
            patch(
                "thesis_eval.cli.create_batch",
                return_value={"id": "batch-123", "endpoint": "/v1/chat/completions", "status": "validating"},
            ) as create_batch_mock,
        ):
            main(
                [
                    "submit-openai-batch",
                    "--input",
                    str(requests_path),
                    "--output",
                    str(output_path),
                ]
            )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["input_file"]["id"], "file-123")
        self.assertEqual(payload["batch"]["id"], "batch-123")
        self.assertEqual(create_batch_mock.call_args.kwargs["endpoint"], "/v1/chat/completions")

    def test_cli_shard_openai_batch_requests_splits_jsonl(self) -> None:
        tmp_path = self._make_workspace_tmp("shard_batch")
        requests_path = tmp_path / "requests.jsonl"
        output_dir = tmp_path / "shards"
        write_jsonl(
            requests_path,
            [
                {
                    "custom_id": f"strongreject-{index:08d}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": "gpt-5.4", "messages": [{"role": "user", "content": f"hi {index}"}]},
                }
                for index in range(5)
            ],
        )

        main(
            [
                "shard-openai-batch-requests",
                "--input",
                str(requests_path),
                "--output-dir",
                str(output_dir),
                "--max-requests",
                "2",
            ]
        )

        index = json.loads((output_dir / "requests-shards.json").read_text(encoding="utf-8"))
        self.assertEqual(index["total_requests"], 5)
        self.assertEqual(index["shard_count"], 3)
        self.assertEqual([row["request_count"] for row in index["shards"]], [2, 2, 1])
        self.assertEqual(read_jsonl(output_dir / "requests-000.jsonl")[0]["custom_id"], "strongreject-00000000")
        self.assertEqual(read_jsonl(output_dir / "requests-002.jsonl")[0]["custom_id"], "strongreject-00000004")

    def test_cli_fetch_openai_batch_downloads_available_files(self) -> None:
        tmp_path = self._make_workspace_tmp("fetch_batch")
        output_path = tmp_path / "batch.json"
        batch_output = tmp_path / "batch_output.jsonl"
        batch_errors = tmp_path / "batch_errors.jsonl"
        batch_payload = {
            "id": "batch-123",
            "status": "completed",
            "output_file_id": "file-out",
            "error_file_id": "file-err",
        }

        with (
            patch("thesis_eval.cli.retrieve_batch", return_value=batch_payload),
            patch("thesis_eval.cli.download_file_content") as download_mock,
        ):
            main(
                [
                    "fetch-openai-batch",
                    "--batch-id",
                    "batch-123",
                    "--output",
                    str(output_path),
                    "--download-output",
                    str(batch_output),
                    "--download-errors",
                    str(batch_errors),
                ]
            )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["id"], "batch-123")
        self.assertEqual(download_mock.call_count, 2)

    def test_openai_batch_rejects_placeholder_batch_id_before_request(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exact id"):
            retrieve_batch("batch_...")



if __name__ == "__main__":
    unittest.main()
