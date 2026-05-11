from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from thesis_eval.progress import info, step
from thesis_eval.translation.log import TranslationLog


# Sentence terminators across the attack-language inventory.
#
# .!? cover the Latin and Cyrillic targets (eng, bul, dan, fin, ita, nor,
# por, rus, spa, swa, swe, ukr); Cyrillic typesetting uses ASCII punctuation.
# ؟ (U+061F) is the Arabic question mark; without it, multi-question Arabic
# inputs round-trip as a single chunk and trigger NLLB's premature-EOS
# failure mode the same way English multi-sentence inputs do.
#
# The Arabic comma ، (U+060C) and semicolon ؛ (U+061B) are excluded: both are
# used inside sentences and would cause spurious splits.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟])\s+")


def _split_sentences(text: str) -> list[str]:
    # NLLB-200 was trained on single-sentence pairs and emits EOS prematurely
    # on multi-sentence inputs, dropping all but the first (or sometimes the
    # last) sentence. Splitting and translating each sentence individually
    # avoids that failure mode. The splitter runs in both directions, so it
    # must recognize target-language punctuation too.
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


NLLB_CODES = {
    "ara": "arb_Arab",
    "bul": "bul_Cyrl",
    "dan": "dan_Latn",
    "eng": "eng_Latn",
    "fin": "fin_Latn",
    "ita": "ita_Latn",
    "nor": "nob_Latn",
    "por": "por_Latn",
    "rus": "rus_Cyrl",
    "spa": "spa_Latn",
    "swa": "swh_Latn",
    "swe": "swe_Latn",
    "ukr": "ukr_Cyrl",
}


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    text: str
    harmful_goal: str = "pending_manual_intent_spec"
    expected_output_form: str = "other"
    fixed_constraints: str = "pending_manual_intent_spec"
    allowed_adaptation: str = "fluency_only_no_semantic_change"


def _resolve_torch_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NLLB translation was requested with --device cuda, but CUDA is not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("NLLB translation was requested with --device mps, but MPS is not available.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(torch: Any, requested: str, device: str) -> Any | None:
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    if requested != "auto":
        raise ValueError(f"Unknown dtype: {requested}")
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return None


@dataclass
class NllbTranslator:
    checkpoint: str
    device: str = "auto"
    dtype: str = "auto"
    source_language: str = "eng"

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("NLLB translation requires transformers. Install the GPU/translation dependencies first.") from exc
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("NLLB translation requires torch. Install `uv sync --extra mac` or the RTX/Docker environment.") from exc

        self.resolved_device = _resolve_torch_device(torch, self.device)
        self.resolved_dtype = _resolve_torch_dtype(torch, self.dtype, self.resolved_device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, src_lang=NLLB_CODES[self.source_language])
        model_kwargs: dict[str, Any] = {}
        if self.resolved_dtype is not None:
            model_kwargs["dtype"] = self.resolved_dtype
        info(f"Loading NLLB checkpoint={self.checkpoint} device={self.resolved_device} dtype={self.dtype}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint, **model_kwargs)
        self.model.to(self.resolved_device)
        self.model.eval()
        self.torch = torch

    def translate(self, text: str, target_language: str, source_language: str | None = None) -> str:
        if source_language is not None:
            self.tokenizer.src_lang = NLLB_CODES[source_language]
        sentences = _split_sentences(text)
        if not sentences:
            return ""
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(NLLB_CODES[target_language])
        encoded = self.tokenizer(sentences, return_tensors="pt", padding=True).to(self.resolved_device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=512,
            )
        chunks = [str(d) for d in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]
        return " ".join(chunks).strip()


def translate_text(
    text: str,
    target_language: str,
    engine: str = "placeholder",
    checkpoint: str = "facebook/nllb-200-distilled-600M",
    device: str = "auto",
    dtype: str = "auto",
) -> str:
    if engine == "placeholder":
        return text
    if engine != "nllb":
        raise ValueError(f"Unknown translation engine: {engine}")
    return NllbTranslator(checkpoint=checkpoint, device=device, dtype=dtype).translate(text, target_language)


def make_translation_log(
    prompt: PromptRecord,
    target_language: str,
    translated_text: str,
    engine: str,
    checkpoint: str,
    device: str = "auto",
    dtype: str = "auto",
) -> TranslationLog:
    return TranslationLog(
        prompt_id=prompt.prompt_id,
        source_language="eng",
        target_language=target_language,
        harmful_goal=prompt.harmful_goal,
        expected_output_form=prompt.expected_output_form,
        fixed_constraints=prompt.fixed_constraints,
        allowed_adaptation=prompt.allowed_adaptation,
        translator_config={
            "engine": engine,
            "checkpoint": checkpoint,
            "source_code": NLLB_CODES["eng"],
            "target_code": NLLB_CODES[target_language],
            "device": device,
            "dtype": dtype,
        },
        blaser_qe_score=None,
        blaser_status="pending",
        human_audit_status="not_audited",
        reference_subset_score=None,
        harm_preservation="pending",
        revision_note=None,
        source_text=prompt.text,
        translated_text=translated_text,
        roundtrip_text=None,
        roundtrip_drift_score=None,
        roundtrip_status="pending",
        xsts_semantic_equivalence="pending",
        xsts_attack_strength="pending",
        xsts_constraint_preservation="pending",
        xsts_naturalness="pending",
        xsts_no_harmfulness_drift="pending",
        translation_revised=False,
        translation_excluded=False,
    )


def run_translation(
    prompts: list[PromptRecord],
    target_language: str,
    engine: str = "placeholder",
    checkpoint: str = "facebook/nllb-200-distilled-600M",
    device: str = "auto",
    dtype: str = "auto",
    stream: bool = False,
    translator: NllbTranslator | None = None,
) -> list[TranslationLog]:
    logs: list[TranslationLog] = []
    info(f"Translating {len(prompts)} prompts to {target_language} with engine={engine} device={device} dtype={dtype}")
    active_translator = translator
    if active_translator is None and engine == "nllb" and target_language != "eng":
        active_translator = NllbTranslator(checkpoint=checkpoint, device=device, dtype=dtype)
    for index, prompt in enumerate(prompts, start=1):
        with step(f"translate {index}/{len(prompts)} prompt_id={prompt.prompt_id} target={target_language}"):
            if target_language == "eng":
                translated = prompt.text
            elif active_translator is not None:
                translated = active_translator.translate(prompt.text, target_language, source_language="eng")
            else:
                translated = translate_text(prompt.text, target_language, engine, checkpoint, device=device, dtype=dtype)
            logged_device = active_translator.resolved_device if active_translator is not None else device
            logs.append(make_translation_log(prompt, target_language, translated, engine, checkpoint, device=logged_device, dtype=dtype))
            if stream:
                info(f"  src: {prompt.text}")
                info(f"  {target_language}: {translated}")
    return logs


def heuristic_blaser_score(source_text: str, translated_text: str) -> float:
    if not source_text.strip() or not translated_text.strip():
        return 0.0
    ratio = len(translated_text) / max(len(source_text), 1)
    length_score = max(0.0, 1.0 - abs(1.0 - ratio))
    return round(min(1.0, max(0.0, 0.55 + 0.45 * length_score)), 4)


def apply_translation_qc(
    logs: list[dict[str, Any]],
    pass_threshold: float = 0.72,
    fail_threshold: float = 0.55,
    scorer: str = "heuristic",
    device: str | None = None,
    blaser_scorer: "Any | None" = None,
) -> list[dict[str, Any]]:
    if scorer == "heuristic":
        scores = [heuristic_blaser_score(str(log["source_text"]), str(log["translated_text"])) for log in logs]
    elif scorer == "blaser":
        from thesis_eval.translation.blaser import score_blaser_qe

        scores = score_blaser_qe(logs, device=device, scorer=blaser_scorer)
    else:
        raise ValueError(f"Unknown translation QC scorer {scorer!r}")
    output: list[dict[str, Any]] = []
    for log, score in zip(logs, scores, strict=True):
        row = dict(log)
        row["blaser_qe_score"] = round(float(score), 4)
        if score >= pass_threshold:
            row["blaser_status"] = "pass"
        elif score < fail_threshold:
            row["blaser_status"] = "fail"
        else:
            row["blaser_status"] = "flag"
        row["harm_preservation"] = "preserved" if row["blaser_status"] == "pass" else "pending"
        output.append(row)
    return output
