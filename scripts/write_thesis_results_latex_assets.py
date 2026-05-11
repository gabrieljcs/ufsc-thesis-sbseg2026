from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "eval" / "outputs"
TABLES = OUTPUTS / "tables"
THESIS = ROOT / "ufscthesisx" / "ufscthesisx"
OUT = THESIS / "chapters" / "results_generated_assets.tex"

TARGET_MODELS = [
    "sagui_7b",
    "sabia_3",
    "llamantino_2_ultrachat_7b",
    "llamantino_anita_8b",
    "gpt_sw3",
    "ai_sweden_llama3_8b",
    "bggpt_7b",
    "bggpt_gemma_9b",
]

REPORT_MODELS = TARGET_MODELS + ["llama3_1_8b_reference"]

LANGS = ["ara", "bul", "dan", "eng", "fin", "ita", "nor", "por", "rus", "spa", "swa", "swe", "ukr"]

DISPLAY = {
    "sagui_7b": "Sagui-7B",
    "sabia_3": "Sabi\\'a-3",
    "llamantino_2_ultrachat_7b": "LLaMAntino-2",
    "llamantino_anita_8b": "ANITA-8B",
    "gpt_sw3": "GPT-SW3",
    "ai_sweden_llama3_8b": "AI Sweden Llama-3",
    "bggpt_7b": "BgGPT-7B",
    "bggpt_gemma_9b": "BgGPT-Gemma-9B",
    "llama3_1_8b_reference": "LLaMA-3.1-8B reference",
}

FULL_DISPLAY = {
    "sagui_7b": "Sagui-7B-Instruct",
    "sabia_3": "Sabi\\'a-3",
    "llamantino_2_ultrachat_7b": "\\makecell[l]{LLaMAntino-2-chat-\\\\UltraChat-ITA}",
    "llamantino_anita_8b": "LLaMAntino-3-ANITA-8B",
    "gpt_sw3": "GPT-SW3-6.7B-v2-instruct",
    "ai_sweden_llama3_8b": "AI Sweden Llama-3-8B-instruct",
    "bggpt_7b": "BgGPT-7B-Instruct-v0.2",
    "bggpt_gemma_9b": "BgGPT-Gemma-2-9B-Instruct",
    "llama3_1_8b_reference": "\\makecell[l]{LLaMA-3.1-8B\\\\reference}",
}

LANG_DISPLAY = {
    "ara": "Arabic",
    "bul": "Bulgarian",
    "dan": "Danish",
    "eng": "English",
    "fin": "Finnish",
    "ita": "Italian",
    "nor": "Norwegian",
    "por": "Portuguese",
    "rus": "Russian",
    "spa": "Spanish",
    "swa": "Swahili",
    "swe": "Swedish",
    "ukr": "Ukrainian",
}

ALIGNED_DISPLAY = {
    "por": "Brazilian Portuguese",
    "ita": "Italian",
    "swe": "Swedish",
    "bul": "Bulgarian",
    "eng": "English",
}

PAIR_LABEL = {"por": "Portuguese", "ita": "Italian", "swe": "Swedish", "bul": "Bulgarian"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: object, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}"


def num(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - radius, center + radius


def load_rows() -> list[dict[str, object]]:
    rows = []
    with (OUTPUTS / "dataset_frozen.jsonl").open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def pooled_asr(rows: list[dict[str, object]], *, aligned: bool) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        model = str(row["model"])
        if model not in TARGET_MODELS or row.get("excluded") or row.get("unsafe_binary") is None:
            continue
        same = row.get("attack_language") == row.get("aligned_language")
        if same != aligned:
            continue
        grouped[model].append(row)
    output = {}
    for model, group in grouped.items():
        n = len(group)
        unsafe = sum(int(row["unsafe_binary"]) for row in group)
        ci_low, ci_high = wilson(unsafe, n)
        output[model] = {
            "n": n,
            "unsafe": unsafe,
            "asr": unsafe / n,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_score": sum(float(row["strongreject_score"]) for row in group) / n,
            "refusal": sum(float(row.get("refusal_binary") or 0) for row in group) / n,
        }
    return output


def model_metadata(rows: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    output = {}
    for row in rows:
        model = str(row["model"])
        if model in output:
            continue
        output[model] = {
            "aligned_language": str(row.get("aligned_language")),
            "pole": str(row.get("model_alignment_pole")),
        }
    return output


def table_coverage(rows: list[dict[str, object]]) -> str:
    coverage = {row["model"]: row for row in read_csv("results_coverage.csv")}
    meta = model_metadata(rows)
    ordered = TARGET_MODELS + ["llama3_1_8b_reference"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\scriptsize",
        r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}X|c|c|c|c|c|c}",
        r"\caption[Evaluation coverage by model]{Evaluation coverage and data quality by model.}",
        r"\label{tab:results-coverage} \\",
        r"\hline",
        r"\textbf{Model} & \textbf{Langs.} & \textbf{Attempts} & \shortstack{\textbf{HTTP/API}\\\textbf{fail. (\%)}} & \shortstack{\textbf{Provider}\\\textbf{block (\%)}} & \shortstack{\textbf{Scoring}\\\textbf{fail. (\%)}} & \shortstack{\textbf{Refusal}\\\textbf{(\%)}} \\",
        r"\hline",
    ]
    for model in ordered:
        row = coverage[model]
        lines.append(
            f"{FULL_DISPLAY[model]} & 13 & {int(float(row['total_attempts']))} & "
            f"{pct(row['api_failure_rate'])} & {pct(row['provider_block_rate'])} & "
            f"{pct(row['scoring_failure_rate'])} & {pct(row['refusal_rate'])} \\\\"
        )
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{7}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_audit() -> str:
    rows = []
    for lang in LANGS:
        path = OUTPUTS / "audit" / f"{lang}.xsts_decisions.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            decisions = list(csv.DictReader(handle))
        scores = [float(row["xsts_score"]) for row in decisions if row.get("xsts_score")]
        counts = Counter(row.get("audit_decision") for row in decisions)
        caveat = "High flag-rate anomaly." if lang == "fin" else ("Caveat language." if lang == "swa" else "General LLM-judge caveat.")
        rows.append((lang, len(decisions), statistics.median(scores) if scores else None, counts.get("pass", 0), counts.get("exclude", 0), caveat))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\scriptsize",
        r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}p{2.1cm}|c|c|c|c|X}",
        r"\caption[LLM-only translation audit summary]{LLM-only translation audit summary for flagged prompt translations.}",
        r"\label{tab:translation-robustness} \\",
        r"\hline",
        r"\textbf{Language} & \textbf{Queue} & \shortstack{\textbf{Median}\\\textbf{XSTS}} & \textbf{Pass} & \textbf{Exclude} & \textbf{Caveat} \\",
        r"\hline",
    ]
    for lang, queue, median, passed, excluded, caveat in rows:
        med = "--" if median is None else num(median, 1)
        lines.append(f"{LANG_DISPLAY[lang]} (\\texttt{{{lang}}}) & {queue} & {med} & {passed} & {excluded} & {caveat} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{6}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_aligned(rows: list[dict[str, object]]) -> str:
    summaries = pooled_asr(rows, aligned=True)
    meta = model_metadata(rows)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{X|X|c|c|c}",
        r"\caption[Aligned-language baseline ASR]{Aligned-language baseline ASR for each target model.}",
        r"\label{tab:aligned-language-baseline-asr} \\",
        r"\hline",
        r"\textbf{Model} & \textbf{Aligned language} & \textbf{ASR} & \textbf{95\% CI} & \textbf{Mean SR score} \\",
        r"\hline",
    ]
    for model in TARGET_MODELS:
        row = summaries[model]
        lang = ALIGNED_DISPLAY[meta[model]["aligned_language"]]
        lines.append(f"{FULL_DISPLAY[model]} & {lang} & {pct(row['asr'])}\\% & [{pct(row['ci_low'])}, {pct(row['ci_high'])}] & {num(row['mean_score'])} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{5}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_cross(rows: list[dict[str, object]]) -> str:
    summaries = pooled_asr(rows, aligned=False)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{X|c|c|c|c}",
        r"\caption[Cross-lingual pooled ASR by model]{Cross-lingual pooled ASR by model, excluding the model's aligned language.}",
        r"\label{tab:overall-asr} \\",
        r"\hline",
        r"\textbf{Model} & \textbf{ASR} & \textbf{95\% CI} & \textbf{Mean SR score} & \textbf{Refusal (\%)} \\",
        r"\hline",
    ]
    for model in TARGET_MODELS:
        row = summaries[model]
        lines.append(f"{FULL_DISPLAY[model]} & {pct(row['asr'])}\\% & [{pct(row['ci_low'])}, {pct(row['ci_high'])}] & {num(row['mean_score'])} & {pct(row['refusal'])} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{5}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_closest_farthest() -> str:
    data = {row["model"]: row for row in read_csv("closest_farthest_languages.csv")}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\scriptsize",
        r"\begin{tabularx}{\linewidth}{X|c|c|c|c|c}",
        r"\caption[Closest and farthest attack languages]{Closest and farthest attack-language ASR by model according to URIEL+ distance.}",
        r"\label{tab:closest-farthest-languages} \\",
        r"\hline",
        r"\textbf{Model} & \shortstack{\textbf{Closest}\\\textbf{lang.}} & \textbf{ASR} & \shortstack{\textbf{Farthest}\\\textbf{lang.}} & \textbf{ASR} & \textbf{Gap} \\",
        r"\hline",
    ]
    for model in TARGET_MODELS:
        row = data[model]
        lines.append(
            f"{FULL_DISPLAY[model]} & \\texttt{{{row['closest_language']}}} & {pct(row['closest_asr'])}\\% & "
            f"\\texttt{{{row['farthest_language']}}} & {pct(row['farthest_asr'])}\\% & {pct(row['gap'])} pp \\\\"
        )
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{6}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_reference_curve() -> str:
    rows = read_csv("reference_distance_curve.csv")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\scriptsize",
        r"\begin{tabularx}{\linewidth}{X|c|c|c|c}",
        r"\caption[English-aligned reference distance curve]{Per-language ASR for the English-aligned reference baseline.}",
        r"\label{tab:reference-distance-curve} \\",
        r"\hline",
        r"\textbf{Attack language} & \textbf{Distance} & \textbf{ASR} & \textbf{Mean SR score} & \textbf{Refusal (\%)} \\",
        r"\hline",
    ]
    for row in rows:
        lang = row["attack_language"]
        lines.append(f"{LANG_DISPLAY[lang]} (\\texttt{{{lang}}}) & {num(row['distance_from_english'])} & {pct(row['asr'])}\\% & {num(row['mean_strongreject_score'])} & {pct(row['refusal_rate'])} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{5}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_counterfactual() -> str:
    rows = read_csv("counterfactual_safety_by_aligned_language.csv")
    order = {"weak": 0, "strong": 1}
    rows.sort(key=lambda row: (row["aligned_language"], order[row["model_alignment_pole"]]))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\scriptsize",
        r"\begin{tabularx}{\linewidth}{X|X|c|c|c}",
        r"\caption[Counterfactual aligned-language safety check]{In-language ASR comparison between each language-specific model and the English-aligned reference baseline.}",
        r"\label{tab:counterfactual-safety} \\",
        r"\hline",
        r"\textbf{Aligned language} & \textbf{Language-specific model} & \textbf{Model ASR} & \textbf{Reference ASR} & \textbf{Gap} \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{ALIGNED_DISPLAY[row['aligned_language']]} & {FULL_DISPLAY[row['model']]} & "
            f"{pct(row['model_asr'])}\\% & {pct(row['reference_asr_same_language'])}\\% & {pct(row['asr_gap_model_minus_reference'])} pp \\\\"
        )
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{5}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def corr_interpretation(rho: float, predictor: str) -> str:
    direction = "positive" if rho > 0.05 else "negative" if rho < -0.05 else "near zero"
    return f"{direction}; descriptive only"


def table_corr(filename: str, label: str, caption: str) -> str:
    rows = [row for row in read_csv(filename) if row["model"] in TARGET_MODELS]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{X|c|c|X}",
        f"\\caption[{caption}]{{{caption}.}}",
        f"\\label{{{label}}} \\\\",
        r"\hline",
        r"\textbf{Model} & \textbf{$n$} & \textbf{Spearman $\rho$} & \textbf{Interpretation} \\",
        r"\hline",
    ]
    for model in TARGET_MODELS:
        row = next(row for row in rows if row["model"] == model)
        rho = float(row["spearman_rho"])
        lines.append(f"{FULL_DISPLAY[model]} & {row['n']} & {num(rho)} & {corr_interpretation(rho, row['predictor'])} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{4}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_glmm() -> str:
    rows = read_csv("glmm_main_effects.csv")
    names = {"distance_z": "Standardized linguistic distance", "spec_z": "Standardized specialization"}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\begin{tabularx}{\linewidth}{X|c|c|c|c}",
        r"\caption[Approximate mixed-effects logistic regression]{Approximate fixed effects from the crossed-random-effects logistic regression.}",
        r"\label{tab:glmm-main-effects} \\",
        r"\hline",
        r"\textbf{Predictor} & \textbf{Estimate} & \textbf{Std. error} & \textbf{Odds ratio} & \textbf{Fit} \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(f"{names[row['predictor']]} & {num(row['estimate'])} & {num(row['std_error'])} & {num(row['odds_ratio'])} & VB \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{5}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_collinearity() -> str:
    values = {row["diagnostic"]: row["value"] for row in read_csv("glmm_collinearity.csv")}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\begin{tabularx}{0.75\linewidth}{X|c}",
        r"\caption[Collinearity diagnostics]{Collinearity diagnostics for the standardized fixed-effect design matrix.}",
        r"\label{tab:glmm-collinearity} \\",
        r"\hline",
        r"\textbf{Diagnostic} & \textbf{Value} \\",
        r"\hline",
        f"VIF: standardized linguistic distance & {num(values['vif_distance'])} \\\\",
        r"\hline",
        f"VIF: standardized specialization & {num(values['vif_spec'])} \\\\",
        r"\hline",
        f"Condition number $\\kappa$ & {num(values['condition_number'])} \\\\",
        r"\hline",
        f"Single-predictor rerun triggered? & {'No' if values['single_predictor_rerun_triggered'] == 'False' else 'Yes'} \\\\",
        r"\hline",
        r"\multicolumn{2}{p{\dimexpr0.75\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_spec_components() -> str:
    rows = read_csv("glmm_spec_components.csv")
    names = {"distance_z": "Standardized linguistic distance", "if_z": "Standardized IF", "cons_z": "Standardized CONS"}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\begin{tabularx}{\linewidth}{X|c|c|c|c}",
        r"\caption[Decomposed specialization model]{Approximate mixed-effects model replacing SPEC with IF and CONS.}",
        r"\label{tab:glmm-spec-components} \\",
        r"\hline",
        r"\textbf{Predictor} & \textbf{Estimate} & \textbf{Std. error} & \textbf{Odds ratio} & \textbf{$n$} \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(f"{names[row['predictor']]} & {num(row['estimate'])} & {num(row['std_error'])} & {num(row['odds_ratio'])} & {row['n']} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{5}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_tokenizer_model() -> str:
    rows = read_csv("glmm_tokenizer_robustness.csv")
    names = {"distance_z": "Standardized linguistic distance", "spec_z": "Standardized specialization", "token_inflation_z": "Standardized token inflation"}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\begin{tabularx}{\linewidth}{X|c|c|c|c}",
        r"\caption[Tokenizer robustness model]{Approximate tokenizer robustness model with standardized token inflation.}",
        r"\label{tab:glmm-tokenizer-robustness} \\",
        r"\hline",
        r"\textbf{Predictor} & \textbf{Estimate} & \textbf{Std. error} & \textbf{Odds ratio} & \textbf{$n$} \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(f"{names[row['predictor']]} & {num(row['estimate'])} & {num(row['std_error'])} & {num(row['odds_ratio'])} & {row['n']} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{5}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_slope_retention() -> str:
    rows = read_csv("prereg_distance_slope_retention.csv")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{X|c|c|c}",
        r"\caption[Preregistered distance-slope retention check]{Preregistered heterogeneous-pair distance-slope retention check.}",
        r"\label{tab:distance-slope-retention} \\",
        r"\hline",
        r"\textbf{Pair language} & \textbf{Weak slope} & \textbf{Strong slope} & \textbf{Sign retained?} \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(f"{PAIR_LABEL[row['model_pair_language']]} & {num(row['weak_distance_slope'])} & {num(row['strong_distance_slope'])} & {'Yes' if row['slope_sign_retained'] == 'True' else 'No'} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\multicolumn{4}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_external_validation() -> str:
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\begin{tabularx}{\linewidth}{X|X|X}",
        r"\caption[External validation status]{External validation status for MultiJail.}",
        r"\label{tab:external-validation} \\",
        r"\hline",
        r"\textbf{Intended role} & \textbf{Final status} & \textbf{Interpretation} \\",
        r"\hline",
        r"External harmful-benchmark validation using native multilingual jailbreak items & Not present in the final handoff outputs & Useful as future robustness evidence, but not necessary for the central StrongREJECT-based hypothesis tests reported here. \\",
        r"\hline",
        r"\multicolumn{3}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])


def table_hypothesis() -> str:
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\setlength\extrarowheight{2pt}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{X|c|X}",
        r"\caption[Hypothesis summary]{Summary of the empirical status of the three research hypotheses.}",
        r"\label{tab:hypothesis-summary} \\",
        r"\hline",
        r"\textbf{Hypothesis} & \textbf{Outcome} & \textbf{Main evidence} \\",
        r"\hline",
        r"H\textsubscript{1}: ASR increases with linguistic distance & Not supported & Pooled distance coefficient is negative; only 1 of 4 paired comparisons retained the distance-slope sign. \\",
        r"\hline",
        r"H\textsubscript{2}: stronger specialization reduces ASR & Not supported & SPEC is positive in the approximate joint model (OR = 3.147), and model-level correlations are heterogeneous. \\",
        r"\hline",
        r"H\textsubscript{3}: distance remains stronger and more stable than specialization & Not supported & Distance and SPEC have signs contrary to the preregistered expectations; collinearity is low, so this is not a collinearity artifact. \\",
        r"\hline",
        r"\multicolumn{3}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\fonte{Author.}} \\",
        r"\end{tabularx}",
        r"\end{table}",
    ])


def collect_metric(table_name: str, field: str) -> dict[tuple[str, str], float]:
    output = {}
    for row in read_csv(table_name):
        if row.get(field) in {None, ""}:
            continue
        output[(row["model"], row["attack_language"])] = float(row[field])
    return output


def heatmap_figure(
    metric: dict[tuple[str, str], float],
    *,
    command: str,
    label: str,
    caption: str,
    title: str,
    color: str,
    scale: float = 100.0,
    models: list[str] | None = None,
) -> str:
    models = models or TARGET_MODELS
    lines = [
        f"\\newcommand{{\\{command}}}{{%",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tikzpicture}[x=0.62cm,y=0.48cm]",
    ]
    for x, lang in enumerate(LANGS):
        lines.append(f"\\node[rotate=45,anchor=west,font=\\tiny] at ({x + 0.16},0.35) {{\\texttt{{{lang}}}}};")
    for y, model in enumerate(models):
        lines.append(f"\\node[anchor=east,font=\\tiny] at (-0.15,{-y - 0.5}) {{\\texttt{{{DISPLAY[model]}}}}};")
        for x, lang in enumerate(LANGS):
            value = metric.get((model, lang))
            if value is None:
                lines.append(f"\\fill[gray!15] ({x},{-y}) rectangle ++(1,-1);")
                text = "--"
            else:
                pctval = max(0, min(100, int(round(value * scale))))
                lines.append(f"\\fill[{color}!{pctval}!white] ({x},{-y}) rectangle ++(1,-1);")
                lines.append(f"\\draw[white,line width=0.2pt] ({x},{-y}) rectangle ++(1,-1);")
                text = f"{value * 100:.0f}" if scale == 100 else f"{value:.1f}"
            lines.append(f"\\node[font=\\tiny] at ({x + 0.5},{-y - 0.5}) {{{text}}};")
    y0 = -len(models) - 1.25
    legend_y = y0 - 0.55
    lines.extend([
        f"\\node[anchor=west,font=\\tiny] at (0,{y0}) {{{title}}};",
        f"\\fill[{color}!0!white] (0,{legend_y}) rectangle ++(0.8,-0.3);\\draw (0,{legend_y}) rectangle ++(0.8,-0.3);\\node[anchor=west,font=\\tiny] at (0.95,{legend_y - 0.15}) {{low}};",
        f"\\fill[{color}!50!white] (2.4,{legend_y}) rectangle ++(0.8,-0.3);\\draw (2.4,{legend_y}) rectangle ++(0.8,-0.3);\\node[anchor=west,font=\\tiny] at (3.35,{legend_y - 0.15}) {{mid}};",
        f"\\fill[{color}!100!white] (4.9,{legend_y}) rectangle ++(0.8,-0.3);\\draw (4.9,{legend_y}) rectangle ++(0.8,-0.3);\\node[anchor=west,font=\\tiny] at (5.85,{legend_y - 0.15}) {{high}};",
        r"\end{tikzpicture}",
        f"\\caption[{caption}]{{{caption}.}}",
        f"\\label{{{label}}}",
        r"\fonte{Author.}",
        r"\end{figure}",
        r"}",
    ])
    return "\n".join(lines)


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    slope = 0.0 if denom == 0 else sum((x - xbar) * (y - ybar) for x, y in points) / denom
    return slope, ybar - slope * xbar


def paired_scatter(command: str, label: str, caption: str, xfield: str, xlabel: str) -> str:
    asr = collect_metric("asr_by_model_language.csv", "asr")
    predictors: dict[tuple[str, str], float] = {}
    if xfield == "distance":
        rows = load_rows()
        for row in rows:
            key = (str(row["model"]), str(row["attack_language"]))
            if row.get("distance") is not None and key not in predictors:
                predictors[key] = float(row["distance"])
    else:
        rows = load_rows()
        for row in rows:
            key = (str(row["model"]), str(row["attack_language"]))
            if row.get("spec_score") is not None and key not in predictors:
                predictors[key] = float(row["spec_score"])
    pairs = [
        ("Portuguese", "sagui_7b", "sabia_3"),
        ("Italian", "llamantino_2_ultrachat_7b", "llamantino_anita_8b"),
        ("Swedish", "gpt_sw3", "ai_sweden_llama3_8b"),
        ("Bulgarian", "bggpt_7b", "bggpt_gemma_9b"),
    ]
    colors = {"weak": "red!70!black", "strong": "blue!70!black"}
    lines = [
        f"\\newcommand{{\\{command}}}{{%",
        r"\begin{figure}[htbp]",
        r"\centering",
    ]
    for idx, (pair_name, weak, strong) in enumerate(pairs):
        if idx % 2 == 0:
            lines.append(r"\begin{minipage}{0.48\linewidth}\centering")
        else:
            lines.append(r"\hfill\begin{minipage}{0.48\linewidth}\centering")
        if xfield == "distance":
            axis_range = "xmin=-0.03,xmax=0.65"
        else:
            axis_range = "xmin=-2.1,xmax=1.4"
        lines.extend([
            r"\begin{tikzpicture}",
            r"\begin{axis}[width=\linewidth,height=3.85cm,grid=both,",
            f"title={{{pair_name}}},xlabel={{{xlabel}}},ylabel={{ASR}},",
            f"{axis_range},ymin=0,ymax=1,",
            r"tick label style={font=\tiny},label style={font=\tiny},title style={font=\small},",
            r"legend style={font=\tiny,at={(0.5,-0.45)},anchor=north,draw=none,fill=none,legend columns=2}]",
        ])
        for model, pole in [(weak, "weak"), (strong, "strong")]:
            pts = [(predictors[(model, lang)], asr[(model, lang)]) for lang in LANGS if (model, lang) in predictors and (model, lang) in asr]
            coords = " ".join(f"({x:.4f},{y:.4f})" for x, y in pts)
            slope, intercept = linear_fit(pts)
            x_min = min(x for x, _ in pts)
            x_max = max(x for x, _ in pts)
            y_min = max(0, min(1, intercept + slope * x_min))
            y_max = max(0, min(1, intercept + slope * x_max))
            lines.append(f"\\addplot+[forget plot,only marks,mark=*,mark size=1.2pt,{colors[pole]}] coordinates {{{coords}}};")
            lines.append(f"\\addplot+[forget plot,no markers,densely dashed,{colors[pole]}] coordinates {{({x_min:.4f},{y_min:.4f}) ({x_max:.4f},{y_max:.4f})}};")
        lines.extend([
            f"\\addlegendimage{{densely dashed,mark=*,mark size=1.2pt,{colors['weak']}}}",
            r"\addlegendentry{weak}",
            f"\\addlegendimage{{densely dashed,mark=*,mark size=1.2pt,{colors['strong']}}}",
            r"\addlegendentry{strong}",
        ])
        lines.extend([
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\end{minipage}",
        ])
        if idx % 2 == 1:
            lines.append(r"\vspace{0.7cm}")
    if xfield == "distance":
        ref_model = "llama3_1_8b_reference"
        ref_pts = [
            (predictors[(ref_model, lang)], asr[(ref_model, lang)])
            for lang in LANGS
            if (ref_model, lang) in predictors and (ref_model, lang) in asr
        ]
        if ref_pts:
            ref_coords = " ".join(f"({x:.4f},{y:.4f})" for x, y in ref_pts)
            ref_slope, ref_intercept = linear_fit(ref_pts)
            ref_x_min = min(x for x, _ in ref_pts)
            ref_x_max = max(x for x, _ in ref_pts)
            ref_y_min = max(0, min(1, ref_intercept + ref_slope * ref_x_min))
            ref_y_max = max(0, min(1, ref_intercept + ref_slope * ref_x_max))
            lines.extend([
                r"\begin{minipage}{0.48\linewidth}\centering",
                r"\begin{tikzpicture}",
                r"\begin{axis}[width=\linewidth,height=3.85cm,grid=both,",
                r"title={English-aligned reference},xlabel={" + xlabel + r"},ylabel={ASR},",
                r"xmin=-0.03,xmax=0.65,ymin=0,ymax=1,",
                r"tick label style={font=\tiny},label style={font=\tiny},title style={font=\small},",
                r"legend style={font=\tiny,at={(0.5,-0.45)},anchor=north,draw=none,fill=none}]",
                f"\\addplot+[only marks,mark=*,mark size=1.2pt,black!75!white] coordinates {{{ref_coords}}};",
                r"\addlegendentry{LLaMA-3.1-8B}",
                f"\\addplot+[no markers,densely dashed,black!75!white] coordinates {{({ref_x_min:.4f},{ref_y_min:.4f}) ({ref_x_max:.4f},{ref_y_max:.4f})}};",
                r"\end{axis}",
                r"\end{tikzpicture}",
                r"\end{minipage}",
            ])
    lines.extend([
        f"\\caption[{caption}]{{{caption}.}}",
        f"\\label{{{label}}}",
        r"\fonte{Author.}",
        r"\end{figure}",
        r"}",
    ])
    return "\n".join(lines)


def write_assets() -> None:
    rows = load_rows()
    asr = collect_metric("asr_by_model_language.csv", "asr")
    token = collect_metric("tokenizer_diagnostics.csv", "mean_token_inflation")
    belebele_if = collect_metric("belebele_scores.csv", "if_score")
    def macro(name: str, body: str) -> str:
        return f"\\newcommand{{\\{name}}}{{%\n{body}\n}}"

    parts = [
        "% Generated by eval/scripts/write_thesis_results_latex_assets.py. Figure macros only; result tables are inlined in chapters/results.tex.",
        heatmap_figure(asr, command="AsrHeatmapFigure", label="fig:asr-heatmap", caption="ASR by model and attack language", title="Cell values are ASR percentages.", color="red", models=REPORT_MODELS),
        paired_scatter("DistanceAsrFigure", "fig:distance-vs-asr", "Language-risk curves relating ASR to URIEL+ distance", "distance", "URIEL+ distance"),
        heatmap_figure(belebele_if, command="BelebeleIfHeatmapFigure", label="fig:belebele-if-heatmap", caption="BELEBELE IF score by model and attack language", title="Cell values are BELEBELE IF percentages.", color="green", models=REPORT_MODELS),
        paired_scatter("SpecAsrFigure", "fig:spec-vs-asr", "ASR versus BELEBELE-derived specialization", "spec_score", "SPEC"),
        heatmap_figure(token, command="TokenizerDiagnosticsFigure", label="fig:tokenizer-diagnostics", caption="Mean token inflation by model and attack language", title="Cell values are mean token-inflation ratios.", color="blue", scale=40.0, models=REPORT_MODELS),
    ]
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    write_assets()
