from __future__ import annotations

from pathlib import Path
from typing import Any

from thesis_eval.models.generation import build_prompt_messages, format_generation_prompt


def load_tokenizer(model_ref: str, trust_remote_code: bool = False) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Tokenizer metrics require transformers. Install the mac or rtx extras.") from exc
    return AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        local_files_only=Path(model_ref).exists(),
    )


def count_chat_template_tokens(tokenizer: Any, prompt: str) -> int:
    if getattr(tokenizer, "chat_template", None):
        tokenized = tokenizer.apply_chat_template(
            build_prompt_messages(tokenizer, prompt),
            tokenize=True,
            add_generation_prompt=True,
        )
        return _token_count(tokenized)
    tokenized = tokenizer(format_generation_prompt(tokenizer, prompt), add_special_tokens=True)
    return _token_count(tokenized["input_ids"])


def attach_tokenizer_metrics(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        key = _count_key(row)
        counts[key] = count_chat_template_tokens(tokenizer, str(row.get("prompt_text", "")))

    enriched: list[dict[str, Any]] = []
    for row in rows:
        output_row = dict(row)
        input_tokens = counts[_count_key(row)]
        prompt_text = str(row.get("prompt_text", ""))
        baseline_key = (
            str(row["model"]),
            str(row.get("benchmark", "")),
            str(row["prompt_id"]),
            str(row["aligned_language"]),
        )
        baseline_tokens = counts.get(baseline_key)
        output_row["input_tokens"] = input_tokens
        output_row["tokens_per_char"] = input_tokens / len(prompt_text) if prompt_text else None
        output_row["token_inflation"] = input_tokens / baseline_tokens if baseline_tokens else None
        enriched.append(output_row)
    return enriched


def _count_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["model"]),
        str(row.get("benchmark", "")),
        str(row["prompt_id"]),
        str(row["attack_language"]),
    )


def _token_count(tokenized: Any) -> int:
    if hasattr(tokenized, "shape"):
        return int(tokenized.shape[-1])
    if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
        return len(tokenized[0])
    return len(tokenized)
