from __future__ import annotations

from typing import Any

from thesis_eval.analysis.reporting import paired_design_rows


def fit_main_glmm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as exc:
        raise RuntimeError("GLMM fitting requires the analysis extra: `uv sync --extra analysis`.") from exc

    data = pd.DataFrame(
        [
            row
            for row in paired_design_rows(rows)
            if not row.get("excluded")
            and row.get("unsafe_binary") is not None
            and row.get("distance") is not None
            and row.get("spec_score") is not None
        ]
    )
    if data.empty:
        raise ValueError("No complete rows available for GLMM fitting")
    data["unsafe_binary"] = data["unsafe_binary"].astype(int)
    data["distance_z"] = _z(data["distance"].astype(float))
    data["spec_z"] = _z(data["spec_score"].astype(float))
    model = BinomialBayesMixedGLM.from_formula(
        "unsafe_binary ~ distance_z + spec_z + C(model)",
        {
            "prompt": "0 + C(prompt_id)",
            "attack_language_re": "0 + C(attack_language)",
        },
        data,
    )
    result = model.fit_vb()
    params = dict(zip(result.model.exog_names, result.fe_mean, strict=True))
    ses = dict(zip(result.model.exog_names, result.fe_sd, strict=True))
    output: list[dict[str, Any]] = []
    for name in ("distance_z", "spec_z"):
        estimate = float(params[name])
        se = float(ses[name])
        output.append(
            {
                "predictor": name,
                "estimate": estimate,
                "std_error": se,
                "odds_ratio": float(__import__("math").exp(estimate)),
                "p_value": None,
                "fit_method": "statsmodels_binomial_bayes_mixed_glm_vb",
            }
        )
    return output


def _z(series: Any) -> Any:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0
    return (series - series.mean()) / std
