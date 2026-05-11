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


class GenerationTests(unittest.TestCase):
    def test_mock_backend_does_not_load_model(self) -> None:
        outputs = generate_outputs(["prompt"], "unused/model", backend="mock")
        self.assertEqual(len(outputs), 1)
        self.assertIn("mock target response", outputs[0])

    def test_build_generation_runner_mock_is_reusable(self) -> None:
        runner = build_generation_runner("unused/model", backend="mock")
        self.assertEqual(runner(["one"]), ["[mock target response: replace with local model output before research use.]"])
        self.assertEqual(len(runner(["one", "two"])), 2)

    def test_local_generation_batch_size_defaults_by_backend(self) -> None:
        from thesis_eval.cli import _resolve_local_generation_batch_size

        self.assertEqual(_resolve_local_generation_batch_size(None, "vllm"), 8)
        self.assertEqual(_resolve_local_generation_batch_size(None, "transformers"), 32)
        self.assertEqual(_resolve_local_generation_batch_size(3, "vllm"), 3)

    def test_parse_vllm_estimated_max_model_len(self) -> None:
        message = "Based on the available memory, the estimated maximum model length is 1184."
        self.assertEqual(_parse_vllm_estimated_max_model_len(message), 1184)
        self.assertIsNone(_parse_vllm_estimated_max_model_len("some other failure"))

    def test_retry_vllm_max_model_len_from_error(self) -> None:
        error = RuntimeError("Based on the available memory, the estimated maximum model length is 1184.")
        retry_len = _retry_vllm_max_model_len_from_error(
            error,
            longest_prompt_tokens=744,
            requested_max_model_len=1256,
        )
        self.assertEqual(retry_len, 1120)

    def test_retry_vllm_max_model_len_from_nested_error_message(self) -> None:
        inner = ValueError("Based on the available memory, the estimated maximum model length is 1104.")
        outer = RuntimeError("Engine core initialization failed. See root cause above. Failed core proc(s): {}")
        outer.__cause__ = inner
        retry_len = _retry_vllm_max_model_len_from_error(
            outer,
            longest_prompt_tokens=744,
            requested_max_model_len=1256,
        )
        self.assertEqual(retry_len, 1040)

    def test_retry_vllm_max_model_len_from_generic_engine_failure(self) -> None:
        error = RuntimeError("Engine core initialization failed. See root cause above. Failed core proc(s): {}")
        retry_len = _retry_vllm_max_model_len_from_error(
            error,
            longest_prompt_tokens=744,
            requested_max_model_len=1256,
        )
        self.assertEqual(retry_len, 1000)

    def test_retry_vllm_max_model_len_raises_if_prompt_alone_does_not_fit(self) -> None:
        error = RuntimeError("Based on the available memory, the estimated maximum model length is 1184.")
        with self.assertRaisesRegex(RuntimeError, "cannot fit even the prompt alone"):
            _retry_vllm_max_model_len_from_error(
                error,
                longest_prompt_tokens=1184,
                requested_max_model_len=1256,
            )

    def test_model_attention_softcap_reads_nested_text_config(self) -> None:
        cfg = SimpleNamespace(text_config=SimpleNamespace(attn_logit_softcapping=50.0))
        self.assertEqual(_model_attention_softcap(cfg), 50.0)

    def test_select_vllm_attention_backend_override_prefers_flashinfer(self) -> None:
        cfg = SimpleNamespace(attn_logit_softcapping=50.0)
        with patch("thesis_eval.models.generation._flashinfer_available", return_value=True):
            backend, note = _select_vllm_attention_backend_override(cfg, explicit_backend=None)
        self.assertEqual(backend, "FLASHINFER")
        self.assertIsNotNone(note)
        self.assertIn("forcing VLLM_ATTENTION_BACKEND=FLASHINFER", str(note))

    def test_select_vllm_attention_backend_override_respects_explicit_env(self) -> None:
        cfg = SimpleNamespace(attn_logit_softcapping=50.0)
        with patch("thesis_eval.models.generation._flashinfer_available", return_value=True):
            backend, note = _select_vllm_attention_backend_override(cfg, explicit_backend="FLASH_ATTN")
        self.assertIsNone(backend)
        self.assertIsNotNone(note)
        self.assertIn("respecting pre-set VLLM_ATTENTION_BACKEND=FLASH_ATTN", str(note))

    def test_wrap_vllm_attention_backend_error_adds_actionable_guidance(self) -> None:
        error = RuntimeError("This flash attention build does not support tanh softcapping.")
        with patch("thesis_eval.models.generation._flashinfer_available", return_value=True):
            wrapped = _maybe_wrap_vllm_attention_backend_error(
                error,
                explicit_backend="FLASH_ATTN",
                auto_backend_override=None,
            )
        self.assertIsNotNone(wrapped)
        self.assertIn("FLASHINFER", str(wrapped))
        self.assertIn("transformers backend", str(wrapped))

    def test_model_status_reports_chat_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text('{"model_type":"llama","architectures":["LlamaForCausalLM"]}', encoding="utf-8")
            (path / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}', encoding="utf-8")
            result = estimate_model_asset_state("fixture/model", local_dir=str(path))
            self.assertTrue(result["has_chat_template"])
            self.assertEqual(result["model_type"], "llama")

    def test_generation_rows_skip_translation_exclusions(self) -> None:
        from thesis_eval.models.generation import build_generation_rows

        translations = [
            {
                "prompt_id": "p1",
                "target_language": "por",
                "translated_text": "keep",
                "source_text": "source",
                "translation_excluded": False,
            },
            {
                "prompt_id": "p2",
                "target_language": "por",
                "translated_text": "drop",
                "source_text": "source",
                "translation_excluded": True,
            },
        ]
        rows = build_generation_rows(translations, model="sagui_7b", aligned_language="por")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt_id"], "p1")

    def test_format_generation_prompt_wraps_llamantino2_without_chat_template(self) -> None:
        class FakeTokenizer:
            name_or_path = "swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA"
            chat_template = None

        formatted = format_generation_prompt(FakeTokenizer(), "Ciao?")
        self.assertTrue(formatted.startswith("[INST]"))
        self.assertIn("Sei un assistente disponibile", formatted)
        self.assertTrue(formatted.endswith("Ciao? [/INST]"))

    def test_format_generation_prompt_uses_native_sagui_user_only_chat_template(self) -> None:
        class FakeTokenizer:
            name_or_path = "OliveiraJLT/Sagui-7B-Instruct-v0.1"
            chat_template = "{{ messages }}"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                self.tokenize = tokenize
                self.add_generation_prompt = add_generation_prompt
                return "FORMATTED"

        tokenizer = FakeTokenizer()
        formatted = format_generation_prompt(tokenizer, "Ola?")
        self.assertEqual(formatted, "FORMATTED")
        self.assertFalse(tokenizer.tokenize)
        self.assertTrue(tokenizer.add_generation_prompt)
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Ola?", "text": ""}])

    def test_build_prompt_messages_include_gpt_sw3_text_field(self) -> None:
        class FakeTokenizer:
            name_or_path = "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct"
            chat_template = "fake"

        messages = build_prompt_messages(FakeTokenizer(), "Hej")
        self.assertEqual(messages, [{"role": "user", "content": "Hej", "text": ""}])

    def test_describe_prompt_format_reports_local_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sagui_path = tmp_path / "sagui-7b-instruct-v0.1"
            sagui_path.mkdir()
            (sagui_path / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}', encoding="utf-8")
            sagui_meta = describe_prompt_format(str(sagui_path))
            self.assertEqual(sagui_meta["strategy"], "chat_template_user_only")
            self.assertIsNone(sagui_meta["system_prompt_id"])

            llamantino_path = tmp_path / "llamantino-2-chat-7b-hf-ultrachat-ita"
            llamantino_path.mkdir()
            (llamantino_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            llamantino_meta = describe_prompt_format(str(llamantino_path))
            self.assertEqual(llamantino_meta["strategy"], "manual_llama2_inst")
            self.assertFalse(llamantino_meta["uses_chat_template"])

    def test_longest_tokenized_prompt_length_tracks_longest_prompt(self) -> None:
        from thesis_eval.models.generation import _longest_tokenized_prompt_length

        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=True):
                extra = 1 if add_special_tokens else 0
                return {"input_ids": [0 for _ in range(len(text.split()) + extra)]}

        longest = _longest_tokenized_prompt_length(FakeTokenizer(), ["um dois", "um dois tres quatro"])
        self.assertEqual(longest, 5)



if __name__ == "__main__":
    unittest.main()
