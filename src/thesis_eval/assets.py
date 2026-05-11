from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

from thesis_eval.benchmarks.strongreject import import_strongreject, write_prompt_records
from thesis_eval.io import read_jsonl
from thesis_eval.paths import CONFIG_DIR, EVAL_ROOT, ensure_dir, read_yaml
from thesis_eval.progress import info, step


@dataclass(frozen=True)
class AssetSpec:
    name: str
    kind: str
    group: str
    groups: tuple[str, ...]
    description: str
    repo_id: str | None = None
    local_dir: Path | None = None
    output: Path | None = None
    pilot_output: Path | None = None
    pilot_size: int = 2
    source: str | None = None
    allow_patterns: list[str] | None = None
    required_files: tuple[Path, ...] = ()
    required_any_files: tuple[tuple[Path, ...], ...] = ()
    required_columns: tuple[str, ...] = ()


def load_assets(path: Path = CONFIG_DIR / "assets.yaml") -> dict[str, AssetSpec]:
    data = read_yaml(path)
    raw_assets = data.get("assets", {})
    if not isinstance(raw_assets, dict):
        raise ValueError("assets.yaml must contain an assets mapping")
    assets: dict[str, AssetSpec] = {}
    for name, raw in raw_assets.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Asset {name} must be a mapping")
        assets[str(name)] = AssetSpec(
            name=str(name),
            kind=str(raw["kind"]),
            group=str(raw["group"]),
            groups=_asset_groups(raw),
            description=str(raw.get("description", "")),
            repo_id=str(raw["repo_id"]) if raw.get("repo_id") else None,
            local_dir=_resolve_path(raw.get("local_dir")),
            output=_resolve_path(raw.get("output")),
            pilot_output=_resolve_path(raw.get("pilot_output")),
            pilot_size=int(raw.get("pilot_size", 2)),
            source=str(raw["source"]) if raw.get("source") else None,
            allow_patterns=list(raw["allow_patterns"]) if isinstance(raw.get("allow_patterns"), list) else None,
            required_files=tuple(_resolve_path(value) for value in raw.get("required_files", []) if _resolve_path(value) is not None),
            required_any_files=_resolve_any_file_groups(raw.get("required_any_files", [])),
            required_columns=tuple(str(value) for value in raw.get("required_columns", [])),
        )
    return assets


def select_assets(names: list[str] | None = None, groups: list[str] | None = None) -> list[AssetSpec]:
    assets = load_assets()
    selected: list[AssetSpec] = []
    if names:
        missing = [name for name in names if name not in assets]
        if missing:
            raise ValueError(f"Unknown asset(s): {missing}. Available: {sorted(assets)}")
        selected.extend(assets[name] for name in names)
    if groups:
        known_groups = {group for asset in assets.values() for group in asset.groups}
        missing_groups = [group for group in groups if group not in known_groups]
        if missing_groups:
            raise ValueError(f"Unknown group(s): {missing_groups}. Available: {sorted(known_groups)}")
        selected.extend(asset for asset in assets.values() if any(group in asset.groups for group in groups))
    if not names and not groups:
        selected = list(assets.values())

    deduped: dict[str, AssetSpec] = {}
    for asset in selected:
        deduped[asset.name] = asset
    return list(deduped.values())


def asset_status(asset: AssetSpec) -> dict[str, Any]:
    path = asset.output or asset.local_dir
    exists = bool(path and path.exists())
    file_count = 0
    byte_count = 0
    if exists and path:
        if path.is_file():
            file_count = 1
            byte_count = path.stat().st_size
        else:
            for child in path.rglob("*"):
                if child.is_file():
                    file_count += 1
                    byte_count += child.stat().st_size
    return {
        "name": asset.name,
        "kind": asset.kind,
        "group": asset.group,
        "groups": list(asset.groups),
        "repo_id": asset.repo_id,
        "path": str(path) if path else None,
        "exists": exists,
        "files": file_count,
        "bytes": byte_count,
        "description": asset.description,
    }


def download_asset(asset: AssetSpec, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    status_before = asset_status(asset)
    if dry_run:
        return {**status_before, "action": "dry_run"}
    if status_before["exists"] and not force:
        verification = verify_asset(asset)
        if verification["ok"]:
            return {**verification, "action": "skip_verified"}
        info(f"Found incomplete asset {asset.name}; resuming into {status_before['path']}")

    if asset.kind == "strongreject":
        if asset.output is None:
            raise ValueError("StrongREJECT asset requires output")
        with step(f"download {asset.name} from {asset.source or 'github'}"):
            records = import_strongreject(source=asset.source or "github")
        write_prompt_records(asset.output, records)
        if asset.pilot_output is not None:
            write_prompt_records(asset.pilot_output, records[: asset.pilot_size])
        return {**asset_status(asset), "action": "downloaded", "records": len(records)}

    if asset.kind == "api_model":
        return {**asset_status(asset), "action": "api_model_no_download"}

    if asset.kind in {"hf_model", "hf_dataset"}:
        if asset.repo_id is None or asset.local_dir is None:
            raise ValueError(f"{asset.name} requires repo_id and local_dir")
        info(f"Downloading {asset.name} ({asset.repo_id}) to {asset.local_dir}")
        if asset.allow_patterns:
            info(f"Allow patterns: {', '.join(asset.allow_patterns)}")
        with step(f"snapshot_download {asset.name}"):
            _snapshot_download(asset, force=force)
        return {**asset_status(asset), "action": "downloaded"}

    raise ValueError(f"Unknown asset kind {asset.kind!r}")


def verify_asset(asset: AssetSpec, deep: bool = False) -> dict[str, Any]:
    status = asset_status(asset)
    checks: list[dict[str, Any]] = []
    if asset.kind != "api_model":
        checks.append(_check_exists(status["path"]))
    for path in asset.required_files:
        checks.append(_check_required_file(path))
    for paths in asset.required_any_files:
        checks.append(_check_required_any_file(paths))
    if asset.kind == "strongreject" and asset.output:
        checks.extend(_verify_strongreject(asset))
    elif asset.kind == "api_model":
        checks.append({"name": "api model configured", "ok": bool(asset.repo_id), "model_id": asset.repo_id})
    elif asset.kind == "hf_model" and asset.local_dir:
        checks.extend(_verify_hf_snapshot(asset.local_dir, expect_config=True))
    elif asset.kind == "hf_dataset" and asset.local_dir:
        checks.extend(_verify_hf_snapshot(asset.local_dir, expect_config=False))
    if deep and asset.local_dir and asset.local_dir.exists():
        checks.append(_hash_summary(asset.local_dir))
    ok = all(check["ok"] for check in checks)
    return {**status, "ok": ok, "checks": checks}


def verify_assets(assets: list[AssetSpec], deep: bool = False) -> list[dict[str, Any]]:
    return [verify_asset(asset, deep=deep) for asset in assets]


def _snapshot_download(asset: AssetSpec, force: bool = False) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Asset download requires huggingface_hub. Run `uv sync` first.") from exc
    assert asset.local_dir is not None
    ensure_dir(asset.local_dir)
    kwargs: dict[str, Any] = {
        "repo_id": asset.repo_id,
        "repo_type": "dataset" if asset.kind == "hf_dataset" else "model",
        "local_dir": str(asset.local_dir),
        "resume_download": True,
    }
    if asset.allow_patterns:
        kwargs["allow_patterns"] = asset.allow_patterns
    if force:
        kwargs["force_download"] = True
    snapshot_download(**kwargs)


def _check_exists(path_value: str | None) -> dict[str, Any]:
    if path_value is None:
        return {"name": "path configured", "ok": False, "message": "asset has no path"}
    path = Path(path_value)
    return {"name": "path exists", "ok": path.exists(), "path": str(path)}


def _check_required_file(path: Path) -> dict[str, Any]:
    return {
        "name": "required file",
        "ok": path.exists() and path.is_file(),
        "path": str(path),
    }


def _check_required_any_file(paths: tuple[Path, ...]) -> dict[str, Any]:
    existing = [path for path in paths if path.exists() and path.is_file()]
    return {
        "name": "required any file",
        "ok": bool(existing),
        "paths": [str(path) for path in paths],
        "matched": [str(path) for path in existing],
    }


def _verify_strongreject(asset: AssetSpec) -> list[dict[str, Any]]:
    assert asset.output is not None
    checks: list[dict[str, Any]] = []
    if not asset.output.exists():
        return checks
    try:
        rows = read_jsonl(asset.output)
    except Exception as exc:
        return [{"name": "jsonl readable", "ok": False, "message": str(exc), "path": str(asset.output)}]
    checks.append({"name": "jsonl readable", "ok": True, "rows": len(rows), "path": str(asset.output)})
    checks.append({"name": "nonempty", "ok": len(rows) > 0, "rows": len(rows)})
    missing_columns = sorted({column for row in rows for column in asset.required_columns if column not in row})
    checks.append({"name": "required columns", "ok": not missing_columns, "missing": missing_columns})
    duplicate_ids = len({row.get("prompt_id") for row in rows}) != len(rows)
    checks.append({"name": "unique prompt_id", "ok": not duplicate_ids})
    if asset.pilot_output:
        checks.append({"name": "pilot prompt file", "ok": asset.pilot_output.exists(), "path": str(asset.pilot_output)})
    return checks


def _verify_hf_snapshot(local_dir: Path, expect_config: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not local_dir.exists():
        return checks
    refs = local_dir / ".cache" / "huggingface"
    checks.append({"name": "hf cache metadata", "ok": refs.exists(), "path": str(refs)})
    config = local_dir / "config.json"
    adapter_config = local_dir / "adapter_config.json"
    if expect_config:
        config_path = config if config.exists() else adapter_config
        checks.append({"name": "model config", "ok": config_path.exists(), "path": str(config_path)})
        if config_path.exists():
            try:
                json.loads(config_path.read_text(encoding="utf-8"))
                checks.append({"name": "model config parseable", "ok": True, "path": str(config_path)})
            except Exception as exc:
                checks.append({"name": "model config parseable", "ok": False, "path": str(config_path), "message": str(exc)})
    file_count = sum(1 for path in local_dir.rglob("*") if path.is_file())
    checks.append({"name": "snapshot nonempty", "ok": file_count > 0, "files": file_count})
    if expect_config:
        checks.append(_verify_hf_model_weights(local_dir))
    return checks


def _verify_hf_model_weights(local_dir: Path) -> dict[str, Any]:
    single_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "flax_model.msgpack",
        "model.pt",
        "adapter_model.bin",
    ]
    present_single = [name for name in single_files if (local_dir / name).is_file()]
    if present_single:
        return {"name": "model weights", "ok": True, "files": present_single, "format": "single"}

    index_files = [
        local_dir / "model.safetensors.index.json",
        local_dir / "pytorch_model.bin.index.json",
    ]
    for index_path in index_files:
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"name": "model weights", "ok": False, "index": str(index_path), "message": f"could not parse index: {exc}"}
        weight_map = index.get("weight_map", {})
        expected = sorted({str(filename) for filename in weight_map.values()})
        missing = [filename for filename in expected if not (local_dir / filename).is_file()]
        return {
            "name": "model weights",
            "ok": bool(expected) and not missing,
            "format": "sharded",
            "index": str(index_path),
            "expected_shards": len(expected),
            "missing_shards": missing[:20],
            "missing_shard_count": len(missing),
        }

    shard_patterns = ("model-*.safetensors", "pytorch_model-*.bin")
    present_shards = sorted(str(path.name) for pattern in shard_patterns for path in local_dir.glob(pattern))
    return {
        "name": "model weights",
        "ok": bool(present_shards),
        "format": "shards_without_index" if present_shards else "missing",
        "files": present_shards[:20],
        "file_count": len(present_shards),
    }


def _hash_summary(local_dir: Path) -> dict[str, Any]:
    import hashlib

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        file_count += 1
        size = path.stat().st_size
        byte_count += size
        digest.update(str(path.relative_to(local_dir)).encode("utf-8"))
        digest.update(size.to_bytes(8, "big", signed=False))
    return {
        "name": "deep hash summary",
        "ok": file_count > 0,
        "files": file_count,
        "bytes": byte_count,
        "sha256_manifest": digest.hexdigest(),
    }


def _resolve_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else EVAL_ROOT / path


def _asset_groups(raw: dict[str, Any]) -> tuple[str, ...]:
    values = raw.get("groups")
    if isinstance(values, list):
        groups = [str(value) for value in values]
    else:
        groups = [str(raw["group"])]
    if str(raw["group"]) not in groups:
        groups.insert(0, str(raw["group"]))
    return tuple(dict.fromkeys(groups))


def _resolve_any_file_groups(value: Any) -> tuple[tuple[Path, ...], ...]:
    groups: list[tuple[Path, ...]] = []
    if not isinstance(value, list):
        return ()
    for group in value:
        if not isinstance(group, list):
            continue
        paths = tuple(path for item in group if (path := _resolve_path(item)) is not None)
        if paths:
            groups.append(paths)
    return tuple(groups)
