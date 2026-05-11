from __future__ import annotations

from typing import Any


def compute_spbleu(hypotheses: list[str], references: list[str]) -> float:
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    if not hypotheses:
        raise ValueError("Cannot compute spBLEU for an empty set")
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError("spBLEU requires sacrebleu. Install the GPU/translation dependencies first.") from exc
    return float(sacrebleu.corpus_bleu(hypotheses, [references], tokenize="flores200").score)


def attach_reference_scores(logs: list[dict[str, Any]], references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = {(row["prompt_id"], row["target_language"]): row["reference_text"] for row in references}
    output: list[dict[str, Any]] = []
    for log in logs:
        row = dict(log)
        key = (row["prompt_id"], row["target_language"])
        if key in refs:
            row["reference_subset_score"] = compute_spbleu([str(row["translated_text"])], [str(refs[key])])
        output.append(row)
    return output
