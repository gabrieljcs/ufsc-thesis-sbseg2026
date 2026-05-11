from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_eval.paths import CONFIG_DIR, read_yaml


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    target_backend: str
    device: str
    dtype: str
    notes: str


def load_runtime_profiles(path: Path = CONFIG_DIR / "runtime.yaml") -> dict[str, RuntimeProfile]:
    data = read_yaml(path)
    raw_profiles = data.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError("runtime.yaml must contain a profiles mapping")
    profiles: dict[str, RuntimeProfile] = {}
    for name, raw in raw_profiles.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Runtime profile {name} must be a mapping")
        profiles[str(name)] = RuntimeProfile(
            name=str(name),
            target_backend=str(raw["target_backend"]),
            device=str(raw["device"]),
            dtype=str(raw["dtype"]),
            notes=str(raw.get("notes", "")),
        )
    return profiles


def resolve_runtime_profile(name: str, path: Path = CONFIG_DIR / "runtime.yaml") -> RuntimeProfile:
    profiles = load_runtime_profiles(path)
    if name not in profiles:
        raise ValueError(f"Unknown runtime profile {name!r}; expected one of {sorted(profiles)}")
    return profiles[name]
