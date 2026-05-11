from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_eval.paths import CONFIG_DIR, read_yaml


@dataclass(frozen=True)
class ConfigBundle:
    attack_languages: list[str]
    aligned_language: dict[str, str]
    models: dict[str, dict[str, Any]]


PAIRED_ALIGNMENT_POLES = {"weak", "strong"}
REFERENCE_ALIGNMENT_POLES = {"reference"}


def load_config(config_dir: Path = CONFIG_DIR) -> ConfigBundle:
    languages = read_yaml(config_dir / "languages.yaml")
    models = read_yaml(config_dir / "models.yaml")

    attack_languages = list(languages.get("attack_languages", []))
    aligned_language = dict(languages.get("aligned_language", {}))
    if not attack_languages:
        raise ValueError("configs/languages.yaml must define attack_languages")
    if len(set(attack_languages)) != len(attack_languages):
        raise ValueError("attack_languages contains duplicates")
    if not models:
        raise ValueError("configs/models.yaml must define at least one model")
    if set(models) != set(aligned_language):
        raise ValueError("Model keys must match aligned_language keys")
    for model_name, model_cfg in models.items():
        model_lang = model_cfg.get("aligned_language")
        if model_lang != aligned_language[model_name]:
            raise ValueError(f"{model_name} aligned_language mismatch")
        if model_lang not in attack_languages:
            raise ValueError(f"{model_name} aligned language {model_lang} missing from inventory")
        access_mode = model_cfg.get("access_mode")
        if access_mode not in {"open_weight", "api"}:
            raise ValueError(f"{model_name} access_mode must be open_weight or api")
        pole = model_cfg.get("alignment_pole")
        if pole not in PAIRED_ALIGNMENT_POLES | REFERENCE_ALIGNMENT_POLES:
            raise ValueError(f"{model_name} alignment_pole must be weak, strong, or reference")
        analysis_role = model_cfg.get("analysis_role", "paired")
        if pole in REFERENCE_ALIGNMENT_POLES and analysis_role != "reference_baseline":
            raise ValueError(f"{model_name} reference models must set analysis_role=reference_baseline")
        if pole in PAIRED_ALIGNMENT_POLES and analysis_role not in {"paired", None}:
            raise ValueError(f"{model_name} paired models must not use analysis_role={analysis_role!r}")
        if not model_cfg.get("provider_model_id"):
            raise ValueError(f"{model_name} must define provider_model_id")
    return ConfigBundle(
        attack_languages=attack_languages,
        aligned_language=aligned_language,
        models=models,
    )
