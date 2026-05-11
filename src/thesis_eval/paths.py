from __future__ import annotations

from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = EVAL_ROOT.parent
CONFIG_DIR = EVAL_ROOT / "configs"
DATA_DIR = EVAL_ROOT / "data"
OUTPUT_DIR = EVAL_ROOT / "outputs"
THESIS_ROOT = WORKSPACE_ROOT / "ufscthesisx"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _read_simple_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip('"').strip("'") for item in inner.split(",")]
    return value.strip('"').strip("'")


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    # Fallback parser for the simple YAML shapes used by this repo's configs,
    # active when PyYAML is not importable.
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            stripped = line.strip()
            if stripped.startswith("- "):
                if not isinstance(parent, list):
                    raise ValueError(f"List item without list parent in {path}: {raw_line!r}")
                item = stripped[2:].strip()
                if item.startswith("[") and item.endswith("]"):
                    parent.append(_parse_scalar(item))
                else:
                    parent.append(_parse_scalar(item))
                continue
            key, sep, value = stripped.partition(":")
            if not sep:
                raise ValueError(f"Unsupported YAML line in {path}: {raw_line!r}")
            if not isinstance(parent, dict):
                raise ValueError(f"Mapping item without mapping parent in {path}: {raw_line!r}")
            value = value.strip()
            if value:
                parent[key.strip()] = _parse_scalar(value)
            else:
                child: dict[str, Any] | list[Any]
                child = [] if _next_content_line_is_list(path, raw_line) else {}
                parent[key.strip()] = child
                stack.append((indent, child))
    return root


def _next_content_line_is_list(path: Path, current_raw_line: str) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        current_index = lines.index(current_raw_line.rstrip("\n"))
    except ValueError:
        return False
    current_indent = len(current_raw_line) - len(current_raw_line.lstrip(" "))
    for line in lines[current_index + 1 :]:
        candidate = line.split("#", 1)[0].rstrip()
        if not candidate.strip():
            continue
        indent = len(candidate) - len(candidate.lstrip(" "))
        return indent > current_indent and candidate.strip().startswith("- ")
    return False
