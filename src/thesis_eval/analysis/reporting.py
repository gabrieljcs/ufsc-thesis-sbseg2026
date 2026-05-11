from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from thesis_eval.metrics.asr import wilson_ci


def coverage_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        total = len(model_rows)
        output.append(
            {
                "model": model,
                "total_attempts": total,
                "api_failure_rate": _mean_bool(model_rows, "api_failure"),
                "provider_block_rate": _mean_bool(model_rows, "provider_block"),
                "scoring_failure_rate": _mean_bool(model_rows, "scoring_failed"),
                "refusal_rate": _mean_numeric(model_rows, "refusal_binary"),
            }
        )
    return output


def paired_design_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("model_alignment_pole") in {"weak", "strong"}
        and row.get("analysis_role", "paired") != "reference_baseline"
    ]


def asr_by_model_language(rows: list[dict[str, Any]], exclude_aligned: bool = False) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("excluded") or row.get("unsafe_binary") is None:
            continue
        if exclude_aligned and row.get("attack_language") == row.get("aligned_language"):
            continue
        grouped[(str(row["model"]), str(row["attack_language"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, language), group_rows in sorted(grouped.items()):
        total = len(group_rows)
        successes = sum(1 for row in group_rows if int(row["unsafe_binary"]) == 1)
        low, high = wilson_ci(successes, total)
        output.append(
            {
                "model": model,
                "attack_language": language,
                "n": total,
                "unsafe": successes,
                "asr": successes / total,
                "ci_low": low,
                "ci_high": high,
                "mean_strongreject_score": _mean_numeric(group_rows, "strongreject_score"),
                "refusal_rate": _mean_numeric(group_rows, "refusal_binary"),
            }
        )
    return output


def reference_distance_curve(rows: list[dict[str, Any]], reference_model: str = "llama3_1_8b_reference") -> list[dict[str, Any]]:
    summaries = asr_by_model_language([row for row in rows if row.get("model") == reference_model])
    distances: dict[str, float] = {}
    for row in rows:
        if row.get("model") == reference_model and row.get("distance") is not None:
            distances[str(row["attack_language"])] = float(row["distance"])
    output: list[dict[str, Any]] = []
    for summary in summaries:
        language = str(summary["attack_language"])
        output.append(
            {
                "reference_model": reference_model,
                "attack_language": language,
                "distance_from_english": distances.get(language),
                "n": summary["n"],
                "unsafe": summary["unsafe"],
                "asr": summary["asr"],
                "mean_strongreject_score": summary["mean_strongreject_score"],
                "refusal_rate": summary["refusal_rate"],
            }
        )
    return sorted(output, key=lambda row: (row["distance_from_english"] is None, row["distance_from_english"] or 0.0, row["attack_language"]))


def counterfactual_safety_by_aligned_language(
    rows: list[dict[str, Any]],
    reference_model: str = "llama3_1_8b_reference",
) -> list[dict[str, Any]]:
    summaries = {
        (str(row["model"]), str(row["attack_language"])): row
        for row in asr_by_model_language(rows)
    }
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        model = str(row["model"])
        if model == reference_model or row.get("model_alignment_pole") not in {"weak", "strong"}:
            continue
        aligned_language = str(row.get("aligned_language"))
        key = (model, aligned_language)
        if key in seen:
            continue
        seen.add(key)
        model_summary = summaries.get(key)
        reference_summary = summaries.get((reference_model, aligned_language))
        if model_summary is None or reference_summary is None:
            continue
        output.append(
            {
                "aligned_language": aligned_language,
                "model": model,
                "model_alignment_pole": row.get("model_alignment_pole"),
                "model_asr": model_summary["asr"],
                "reference_model": reference_model,
                "reference_asr_same_language": reference_summary["asr"],
                "asr_gap_model_minus_reference": model_summary["asr"] - reference_summary["asr"],
                "model_n": model_summary["n"],
                "reference_n": reference_summary["n"],
            }
        )
    return sorted(output, key=lambda row: (row["aligned_language"], row["model_alignment_pole"], row["model"]))


def closest_farthest_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = asr_by_model_language(rows)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    distances: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("distance") is not None:
            distances[(str(row["model"]), str(row["attack_language"]))] = float(row["distance"])
    for row in summaries:
        by_model[str(row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        with_distance = [row for row in model_rows if (model, str(row["attack_language"])) in distances]
        if not with_distance:
            continue
        closest = min(with_distance, key=lambda row: distances[(model, str(row["attack_language"]))])
        farthest = max(with_distance, key=lambda row: distances[(model, str(row["attack_language"]))])
        output.append(
            {
                "model": model,
                "closest_language": closest["attack_language"],
                "closest_distance": distances[(model, str(closest["attack_language"]))],
                "closest_asr": closest["asr"],
                "farthest_language": farthest["attack_language"],
                "farthest_distance": distances[(model, str(farthest["attack_language"]))],
                "farthest_asr": farthest["asr"],
                "gap": farthest["asr"] - closest["asr"],
            }
        )
    return output


def spearman_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = asr_by_model_language(rows)
    distance_rows: list[dict[str, Any]] = []
    spec_rows: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictors: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["model"]), str(row["attack_language"]))
        if row.get("distance") is not None:
            predictors[key]["distance"] = float(row["distance"])
        if row.get("spec_score") is not None:
            predictors[key]["spec_score"] = float(row["spec_score"])
    for row in summaries:
        by_model[str(row["model"])].append(row)
    for model, model_rows in sorted(by_model.items()):
        distance_pairs = [
            (predictors[(model, str(row["attack_language"]))]["distance"], float(row["asr"]))
            for row in model_rows
            if "distance" in predictors[(model, str(row["attack_language"]))]
        ]
        spec_pairs = [
            (predictors[(model, str(row["attack_language"]))]["spec_score"], float(row["asr"]))
            for row in model_rows
            if "spec_score" in predictors[(model, str(row["attack_language"]))]
        ]
        distance_rows.append(_correlation_row(model, "distance", distance_pairs))
        spec_rows.append(_correlation_row(model, "spec_score", spec_pairs))
    return distance_rows, spec_rows


def tokenizer_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("excluded"):
            continue
        grouped[(str(row["model"]), str(row["attack_language"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, language), group_rows in sorted(grouped.items()):
        output.append(
            {
                "model": model,
                "attack_language": language,
                "n": len(group_rows),
                "mean_token_inflation": _mean_numeric(group_rows, "token_inflation"),
                "mean_tokens_per_char": _mean_numeric(group_rows, "tokens_per_char"),
                "truncation_risk_rate": _mean_bool(group_rows, "truncation_risk"),
            }
        )
    return output


def belebele_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("if_score") is None and row.get("cons_score") is None and row.get("spec_score") is None:
            continue
        key = (str(row["model"]), str(row["attack_language"]))
        if key in by_key:
            continue
        by_key[key] = {
            "model": key[0],
            "attack_language": key[1],
            "if_score": row.get("if_score"),
            "cons_score": row.get("cons_score"),
            "spec_score": row.get("spec_score"),
        }
    return [by_key[key] for key in sorted(by_key)]


def collinearity_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = paired_design_rows(rows)
    pairs = [
        (float(row["distance"]), float(row["spec_score"]))
        for row in rows
        if not row.get("excluded") and row.get("distance") is not None and row.get("spec_score") is not None
    ]
    if len(pairs) < 2:
        return [
            {"diagnostic": "vif_distance", "value": None},
            {"diagnostic": "vif_spec", "value": None},
            {"diagnostic": "condition_number", "value": None},
            {"diagnostic": "single_predictor_rerun_triggered", "value": None},
        ]
    r = _pearson([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    vif = 1 / max(1e-12, 1 - r * r)
    condition = math.sqrt((1 + abs(r)) / max(1e-12, 1 - abs(r)))
    return [
        {"diagnostic": "vif_distance", "value": vif},
        {"diagnostic": "vif_spec", "value": vif},
        {"diagnostic": "condition_number", "value": condition},
        {"diagnostic": "single_predictor_rerun_triggered", "value": bool(vif > 5)},
    ]


def prereg_distance_slope_retention(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = paired_design_rows(rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("excluded") or row.get("unsafe_binary") is None or row.get("distance") is None:
            continue
        pair_language = row.get("model_pair_language") or row.get("aligned_language")
        pole = row.get("model_alignment_pole")
        if pair_language is None or pole not in {"weak", "strong"}:
            continue
        grouped[(str(pair_language), str(row["model"]), str(pole))].append(row)

    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (pair_language, model, pole), group_rows in grouped.items():
        xs = [float(row["distance"]) for row in group_rows]
        ys = [int(row["unsafe_binary"]) for row in group_rows]
        slope = _logistic_slope(xs, ys)
        by_pair[pair_language][pole] = {
            "model": model,
            "slope": slope,
            "sign": _sign(slope),
            "n": len(group_rows),
        }

    output: list[dict[str, Any]] = []
    for pair_language, poles in sorted(by_pair.items()):
        weak = poles.get("weak", {})
        strong = poles.get("strong", {})
        weak_sign = weak.get("sign")
        strong_sign = strong.get("sign")
        retained = weak_sign is not None and weak_sign != 0 and weak_sign == strong_sign
        output.append(
            {
                "model_pair_language": pair_language,
                "weak_model": weak.get("model"),
                "weak_distance_slope": weak.get("slope"),
                "weak_slope_sign": weak_sign,
                "weak_n": weak.get("n", 0),
                "strong_model": strong.get("model"),
                "strong_distance_slope": strong.get("slope"),
                "strong_slope_sign": strong_sign,
                "strong_n": strong.get("n", 0),
                "slope_sign_retained": retained,
            }
        )
    return output


def prereg_falsification_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_rows = prereg_distance_slope_retention(rows)
    retained = sum(1 for row in pair_rows if row["slope_sign_retained"])
    panel_complete = len(pair_rows) == 4
    return [
        {
            "pairs_evaluated": len(pair_rows),
            "pairs_with_slope_sign_retained": retained,
            "panel_complete": panel_complete,
            "interpretation": _prereg_interpretation(retained, panel_complete),
        }
    ]


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _logistic_slope(xs: list[float], ys: list[int]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    scale = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / len(xs))
    if scale == 0:
        return None
    zs = [(x - mean_x) / scale for x in xs]
    intercept = 0.0
    slope = 0.0
    ridge = 1e-6
    for _ in range(50):
        g0 = -ridge * intercept
        g1 = -ridge * slope
        h00 = -ridge
        h01 = 0.0
        h11 = -ridge
        for z, y in zip(zs, ys, strict=True):
            eta = max(-35.0, min(35.0, intercept + slope * z))
            p = 1 / (1 + math.exp(-eta))
            w = p * (1 - p)
            residual = y - p
            g0 += residual
            g1 += residual * z
            h00 -= w
            h01 -= w * z
            h11 -= w * z * z
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        delta0 = (h11 * g0 - h01 * g1) / det
        delta1 = (-h01 * g0 + h00 * g1) / det
        intercept -= delta0
        slope -= delta1
        if max(abs(delta0), abs(delta1)) < 1e-8:
            break
    return slope


def _sign(value: float | None) -> int | None:
    if value is None:
        return None
    if abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else -1


def _prereg_interpretation(retained: int, panel_complete: bool) -> str:
    if not panel_complete:
        return "h1_panel_incomplete"
    if retained >= 4:
        return "h1_supported_in_all_four_pairs"
    if retained == 3:
        return "h1_supported_in_three_of_four_pairs"
    if retained == 2:
        return "h1_supported_in_two_of_four_pairs"
    return "h1_not_supported"


def _correlation_row(model: str, predictor: str, pairs: list[tuple[float, float]]) -> dict[str, Any]:
    return {
        "model": model,
        "predictor": predictor,
        "n": len(pairs),
        "spearman_rho": _spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs]) if len(pairs) >= 2 else None,
        "p_value": None,
        "interpretation": "descriptive_only",
    }


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _ranks(values: list[float]) -> list[float]:
    sorted_values = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    position = 0
    while position < len(sorted_values):
        end = position
        while end + 1 < len(sorted_values) and sorted_values[end + 1][0] == sorted_values[position][0]:
            end += 1
        rank = (position + end + 2) / 2
        for _, index in sorted_values[position : end + 1]:
            ranks[index] = rank
        position = end + 1
    return ranks
