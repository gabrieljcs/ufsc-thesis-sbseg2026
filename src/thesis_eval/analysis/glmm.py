from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

from thesis_eval.analysis.reporting import paired_design_rows


@dataclass(frozen=True)
class Predictor:
    source: str
    standardized: str


MAIN_PREDICTORS = (
    Predictor("distance", "distance_z"),
    Predictor("spec_score", "spec_z"),
)
SPEC_COMPONENT_PREDICTORS = (
    Predictor("distance", "distance_z"),
    Predictor("if_score", "if_z"),
    Predictor("cons_score", "cons_z"),
)
TOKENIZER_PREDICTORS = (
    Predictor("distance", "distance_z"),
    Predictor("spec_score", "spec_z"),
    Predictor("token_inflation", "token_inflation_z"),
)

EFFECT_FIELDS = [
    "fit",
    "predictor",
    "estimate",
    "std_error",
    "ci_lower",
    "ci_upper",
    "odds_ratio",
    "odds_ratio_ci_lower",
    "odds_ratio_ci_upper",
    "posterior_probability_positive",
    "p_value",
    "interval_type",
    "n",
    "fit_method",
]

DIAGNOSTIC_FIELDS = [
    "fit",
    "converged",
    "optimizer",
    "optimizer_attempts",
    "random_seed",
    "optimizer_message",
    "iterations",
    "gradient_norm",
    "objective",
    "n",
    "n_prompts",
    "n_attack_languages",
    "prompt_random_intercept_sd",
    "language_random_intercept_sd",
    "covariance_min_eigenvalue",
    "hessian_positive_definite",
    "fixed_effect_prior_sd",
    "log_random_sd_prior_sd",
    "warnings",
    "prior_attempt_warnings",
    "fit_method",
]


def fit_main_glmm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible entry point returning the converged pooled main fit."""
    data = _prepare_data(rows, MAIN_PREDICTORS)
    effects, _ = _fit_laplace_map(data, MAIN_PREDICTORS, fit="pooled")
    return effects


def fit_glmm_suite(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Fit the complete pre-specified model suite and convergent sensitivity models."""
    outputs: dict[str, list[dict[str, Any]]] = {}
    diagnostics: list[dict[str, Any]] = []

    pooled = _prepare_data(rows, MAIN_PREDICTORS)
    main_effects, main_diagnostics = _fit_laplace_map(pooled, MAIN_PREDICTORS, fit="pooled")
    outputs["glmm_main_effects.csv"] = main_effects
    diagnostics.append(main_diagnostics)

    strata_effects: list[dict[str, Any]] = []
    for stratum in ("weak", "strong"):
        data = _prepare_data(rows, MAIN_PREDICTORS, stratum=stratum)
        effects, fit_diagnostics = _fit_laplace_map(data, MAIN_PREDICTORS, fit=stratum)
        strata_effects.extend(effects)
        diagnostics.append(fit_diagnostics)
    outputs["glmm_strata_effects.csv"] = strata_effects

    components = _prepare_data(rows, SPEC_COMPONENT_PREDICTORS)
    component_effects, component_diagnostics = _fit_laplace_map(
        components,
        SPEC_COMPONENT_PREDICTORS,
        fit="spec_components",
    )
    outputs["glmm_spec_components.csv"] = component_effects
    diagnostics.append(component_diagnostics)

    tokenizer = _prepare_data(rows, TOKENIZER_PREDICTORS)
    tokenizer_effects, tokenizer_diagnostics = _fit_laplace_map(
        tokenizer,
        TOKENIZER_PREDICTORS,
        fit="tokenizer_robustness",
    )
    outputs["glmm_tokenizer_robustness.csv"] = tokenizer_effects
    diagnostics.append(tokenizer_diagnostics)

    outputs["glmm_diagnostics.csv"] = diagnostics
    outputs["glmm_gee_sensitivity.csv"] = _fit_prompt_clustered_gee(pooled, MAIN_PREDICTORS)

    prior_effects, prior_diagnostics = _fit_laplace_map(
        pooled,
        MAIN_PREDICTORS,
        fit="pooled_tighter_fixed_effect_prior",
        fixed_effect_prior_sd=2.0,
    )
    outputs["glmm_prior_sensitivity.csv"] = prior_effects
    outputs["glmm_prior_sensitivity_diagnostics.csv"] = [prior_diagnostics]

    postreview_effects: list[dict[str, Any]] = []
    postreview_diagnostics: list[dict[str, Any]] = []
    postreview_specs = (
        ("excluding_aligned_language", {"exclude_aligned": True}),
        (
            "excluding_translation_caveat_languages",
            {"exclude_attack_languages": frozenset({"fin", "swa", "ukr"})},
        ),
        ("excluding_blaser_flagged", {"exclude_blaser_flagged": True}),
    )
    prepared_postreview: dict[str, Any] = {}
    for fit, prepare_kwargs in postreview_specs:
        sensitivity_data = _prepare_data(rows, MAIN_PREDICTORS, **prepare_kwargs)
        prepared_postreview[fit] = sensitivity_data
        effects, fit_diagnostics = _fit_laplace_map(
            sensitivity_data,
            MAIN_PREDICTORS,
            fit=fit,
        )
        postreview_effects.extend(effects)
        postreview_diagnostics.append(fit_diagnostics)
    outputs["glmm_postreview_sensitivity.csv"] = postreview_effects
    outputs["glmm_postreview_sensitivity_diagnostics.csv"] = postreview_diagnostics
    outputs["glmm_postreview_gee_sensitivity.csv"] = _fit_prompt_clustered_gee(
        prepared_postreview["excluding_aligned_language"],
        MAIN_PREDICTORS,
        fit="prompt_clustered_gee_excluding_aligned_language",
    )
    return outputs


def _prepare_data(
    rows: list[dict[str, Any]],
    predictors: tuple[Predictor, ...],
    *,
    stratum: str | None = None,
    exclude_aligned: bool = False,
    exclude_attack_languages: frozenset[str] = frozenset(),
    exclude_blaser_flagged: bool = False,
) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("GLMM fitting requires the analysis extra: `uv sync --extra analysis`.") from exc

    selected: list[dict[str, Any]] = []
    for row in paired_design_rows(rows):
        if row.get("excluded") or row.get("unsafe_binary") is None:
            continue
        if stratum is not None and row.get("model_alignment_pole") != stratum:
            continue
        if exclude_aligned and row.get("attack_language") == row.get("aligned_language"):
            continue
        if row.get("attack_language") in exclude_attack_languages:
            continue
        if exclude_blaser_flagged and row.get("translation_blaser_status") == "flag":
            continue
        if any(row.get(predictor.source) is None for predictor in predictors):
            continue
        selected.append(row)

    data = pd.DataFrame(selected)
    if data.empty:
        raise ValueError("No complete rows available for GLMM fitting")
    data["unsafe_binary"] = data["unsafe_binary"].astype(int)
    for predictor in predictors:
        data[predictor.standardized] = _z(data[predictor.source].astype(float))
    return data


def _fit_laplace_map(
    data: Any,
    predictors: tuple[Predictor, ...],
    *,
    fit: str,
    fixed_effect_prior_sd: float = 10.0,
    log_random_sd_prior_sd: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import numpy as np
        from scipy.stats import norm
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as exc:
        raise RuntimeError("GLMM fitting requires the analysis extra: `uv sync --extra analysis`.") from exc

    predictor_names = [predictor.standardized for predictor in predictors]
    formula = "unsafe_binary ~ " + " + ".join([*predictor_names, "C(model)"])
    model = BinomialBayesMixedGLM.from_formula(
        formula,
        {
            "prompt": "0 + C(prompt_id)",
            "attack_language": "0 + C(attack_language)",
        },
        data,
        fe_p=fixed_effect_prior_sd,
        vcp_p=log_random_sd_prior_sd,
    )
    result = None
    selected_warning_messages: list[str] = []
    prior_attempt_warning_messages: list[str] = []
    optimizer_attempts = (
        ("BFGS", {"gtol": 1e-5, "maxiter": 2000, "disp": False}),
        ("BFGS", {"gtol": 1e-5, "maxiter": 2000, "disp": False}),
        ("BFGS", {"gtol": 1e-5, "maxiter": 2000, "disp": False}),
        ("L-BFGS-B", {"gtol": 1e-6, "ftol": 1e-12, "maxiter": 4000, "disp": False}),
    )
    random_seed = 20260712 + sum((index + 1) * ord(char) for index, char in enumerate(fit))
    for attempt_index, (optimizer_method, optimizer_options) in enumerate(optimizer_attempts):
        random_state = np.random.get_state()
        np.random.seed(random_seed + attempt_index)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                candidate = model.fit_map(
                    method=optimizer_method,
                    minim_opts=optimizer_options,
                    scale_fe=False,
                )
            finally:
                np.random.set_state(random_state)
        current_warning_messages = [str(item.message) for item in caught]
        candidate_gradient_norm = float(
            np.linalg.norm(np.asarray(candidate.optim_retvals.jac, dtype=float))
        )
        result = candidate
        if candidate.optim_retvals.success and candidate_gradient_norm <= 1e-4:
            selected_warning_messages = current_warning_messages
            break
        prior_attempt_warning_messages.extend(current_warning_messages)
    assert result is not None

    optimizer = result.optim_retvals
    gradient_norm = float(np.linalg.norm(np.asarray(optimizer.jac, dtype=float)))
    covariance = np.asarray(result.cov_params(), dtype=float)
    covariance_min_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    finite_diagnostics = all(
        math.isfinite(value)
        for value in (gradient_norm, float(optimizer.fun), covariance_min_eigenvalue)
    )
    converged = bool(optimizer.success and finite_diagnostics and gradient_norm <= 1e-4)
    if not converged:
        raise RuntimeError(
            f"GLMM fit {fit!r} did not satisfy convergence diagnostics: "
            f"success={optimizer.success}, gradient_norm={gradient_norm:.6g}, "
            f"message={optimizer.message}"
        )

    fixed_means = dict(zip(result.model.exog_names, result.fe_mean, strict=True))
    fixed_sds = dict(zip(result.model.exog_names, result.fe_sd, strict=True))
    effects: list[dict[str, Any]] = []
    for name in predictor_names:
        estimate = float(fixed_means[name])
        std_error = float(fixed_sds[name])
        ci_lower = estimate - 1.96 * std_error
        ci_upper = estimate + 1.96 * std_error
        effects.append(
            {
                "fit": fit,
                "predictor": name,
                "estimate": estimate,
                "std_error": std_error,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "odds_ratio": math.exp(estimate),
                "odds_ratio_ci_lower": math.exp(ci_lower),
                "odds_ratio_ci_upper": math.exp(ci_upper),
                "posterior_probability_positive": float(norm.cdf(estimate / std_error)),
                "p_value": None,
                "interval_type": "laplace_approx_95_credible",
                "n": int(len(data)),
                "fit_method": "laplace_map_crossed_logistic_glmm",
            }
        )

    random_sds = np.exp(np.asarray(result.vcp_mean, dtype=float))
    diagnostics = {
        "fit": fit,
        "converged": converged,
        "optimizer": optimizer_method,
        "optimizer_attempts": attempt_index + 1,
        "random_seed": random_seed + attempt_index,
        "optimizer_message": str(optimizer.message),
        "iterations": int(optimizer.nit),
        "gradient_norm": gradient_norm,
        "objective": float(optimizer.fun),
        "n": int(len(data)),
        "n_prompts": int(data["prompt_id"].nunique()),
        "n_attack_languages": int(data["attack_language"].nunique()),
        "prompt_random_intercept_sd": float(random_sds[0]),
        "language_random_intercept_sd": float(random_sds[1]),
        "covariance_min_eigenvalue": covariance_min_eigenvalue,
        "hessian_positive_definite": bool(covariance_min_eigenvalue > 0),
        "fixed_effect_prior_sd": fixed_effect_prior_sd,
        "log_random_sd_prior_sd": log_random_sd_prior_sd,
        "warnings": " | ".join(selected_warning_messages),
        "prior_attempt_warnings": " | ".join(prior_attempt_warning_messages),
        "fit_method": "laplace_map_crossed_logistic_glmm",
    }
    return effects, diagnostics


def _fit_prompt_clustered_gee(
    data: Any,
    predictors: tuple[Predictor, ...],
    *,
    fit: str = "prompt_clustered_gee_language_fixed_effects",
) -> list[dict[str, Any]]:
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Independence
        from statsmodels.genmod.generalized_estimating_equations import GEE
    except ImportError as exc:
        raise RuntimeError("GEE fitting requires the analysis extra: `uv sync --extra analysis`.") from exc

    predictor_names = [predictor.standardized for predictor in predictors]
    formula = "unsafe_binary ~ " + " + ".join(
        [*predictor_names, "C(model)", "C(attack_language)"]
    )
    model = GEE.from_formula(
        formula,
        groups="prompt_id",
        data=data,
        family=sm.families.Binomial(),
        cov_struct=Independence(),
    )
    result = model.fit(maxiter=500, ctol=1e-8)
    if not result.converged:
        raise RuntimeError("Prompt-clustered GEE sensitivity model did not converge")

    effects: list[dict[str, Any]] = []
    for name in predictor_names:
        estimate = float(result.params[name])
        std_error = float(result.bse[name])
        ci_lower = estimate - 1.96 * std_error
        ci_upper = estimate + 1.96 * std_error
        effects.append(
            {
                "fit": fit,
                "predictor": name,
                "estimate": estimate,
                "std_error": std_error,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "odds_ratio": math.exp(estimate),
                "odds_ratio_ci_lower": math.exp(ci_lower),
                "odds_ratio_ci_upper": math.exp(ci_upper),
                "posterior_probability_positive": None,
                "p_value": float(result.pvalues[name]),
                "interval_type": "robust_95_confidence",
                "n": int(len(data)),
                "fit_method": "prompt_clustered_gee_attack_language_fixed_effects",
            }
        )
    return effects


def _z(series: Any) -> Any:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0
    return (series - series.mean()) / std
