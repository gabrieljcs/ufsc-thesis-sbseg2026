from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import subprocess
from typing import Any


def load_distance_matrix(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        languages = header[1:]
        matrix: dict[str, dict[str, float]] = {}
        for row in reader:
            source = row[0]
            matrix[source] = {language: float(value) for language, value in zip(languages, row[1:], strict=True)}
    validate_distance_matrix(matrix, languages)
    return matrix


def validate_distance_matrix(matrix: dict[str, dict[str, float]], languages: list[str] | None = None) -> None:
    languages = languages or list(matrix)
    if set(matrix) != set(languages):
        raise ValueError("Distance matrix rows do not match requested language inventory")
    for src in languages:
        if set(matrix[src]) != set(languages):
            raise ValueError(f"Distance matrix columns for {src} do not match inventory")
        if abs(matrix[src][src]) > 1e-9:
            raise ValueError(f"Distance matrix diagonal must be zero for {src}")
        for tgt in languages:
            value = matrix[src][tgt]
            if not 0 <= value <= 1:
                raise ValueError(f"Distance {src}->{tgt}={value} outside [0, 1]")
            if abs(value - matrix[tgt][src]) > 1e-9:
                raise ValueError(f"Distance matrix is not symmetric for {src}, {tgt}")


def subset_matrix(matrix: dict[str, dict[str, float]], languages: list[str]) -> dict[str, dict[str, float]]:
    missing = [language for language in languages if language not in matrix]
    if missing:
        raise ValueError(f"Distance matrix missing languages: {missing}")
    subset = {src: {tgt: matrix[src][tgt] for tgt in languages} for src in languages}
    validate_distance_matrix(subset, languages)
    return subset


def compute_urielplus_matrix(
    languages: list[str],
    distance_type: str = "featural",
    uriel: Any | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    if uriel is None:
        try:
            from urielplus.urielplus import URIELPlus
        except ImportError as exc:
            raise RuntimeError(
                "URIEL+ recomputation requires the optional `urielplus` dependency. "
                "Use the default `prepare-uriel` uv backend, or install urielplus in the "
                "active environment before using `--uriel-backend in-process`."
            ) from exc
        uriel = URIELPlus()

    matrix: dict[str, dict[str, float]] = {
        src: {tgt: 0.0 for tgt in languages} for src in languages
    }
    failed_pairs: list[dict[str, str]] = []
    for i, src in enumerate(languages):
        for tgt in languages[i + 1 :]:
            try:
                value = float(uriel.new_distance(distance_type, src, tgt))
            except Exception as exc:
                failed_pairs.append({"source": src, "target": tgt, "error": str(exc)})
                continue
            matrix[src][tgt] = value
            matrix[tgt][src] = value

    if failed_pairs:
        examples = ", ".join(f"{pair['source']}-{pair['target']}" for pair in failed_pairs[:5])
        raise RuntimeError(f"URIEL+ failed to compute {len(failed_pairs)} pair(s): {examples}")

    validate_distance_matrix(matrix, languages)
    return matrix, _urielplus_metadata(uriel, languages, distance_type)


def compute_urielplus_matrix_external(
    python_executable: Path,
    languages: list[str],
    distance_type: str = "featural",
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    return _run_urielplus_subprocess(
        [str(python_executable), str(_script_path())],
        languages,
        distance_type,
        backend_metadata={"execution_backend": "python_executable"},
    )


def compute_urielplus_matrix_uv(
    languages: list[str],
    distance_type: str = "featural",
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    return _run_urielplus_subprocess(
        ["uv", "run", "--quiet", "--script", str(_script_path())],
        languages,
        distance_type,
        backend_metadata={
            "execution_backend": "uv",
            "urielplus_script": str(_script_path()),
        },
    )


def _run_urielplus_subprocess(
    command: list[str],
    languages: list[str],
    distance_type: str,
    backend_metadata: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    result = subprocess.run(
        [*command, "--languages-json", json.dumps(languages), "--distance-type", distance_type],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "External URIEL+ recomputation failed with "
            f"{command[0]}: {result.stderr.strip() or result.stdout.strip()}"
        )
    payload = json.loads(result.stdout)
    matrix = payload["matrix"]
    validate_distance_matrix(matrix, languages)
    metadata = payload["metadata"]
    metadata.update(backend_metadata)
    return matrix, metadata


def _script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "compute_urielplus.py"


def write_distance_matrix(path: Path, matrix: dict[str, dict[str, float]], languages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + languages)
        for src in languages:
            writer.writerow([src] + [f"{matrix[src][tgt]:.6f}" for tgt in languages])


def write_run_config(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)


def write_latex_table(path: Path, matrix: dict[str, dict[str, float]], languages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = "l" + ("r" * len(languages))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by thesis-eval prepare-uriel. Do not edit manually.\n")
        handle.write(f"\\begin{{tabular}}{{{columns}}}\n")
        handle.write("\\hline\n")
        handle.write(" & " + " & ".join(f"\\texttt{{{lang}}}" for lang in languages) + " \\\\\n")
        handle.write("\\hline\n")
        for src in languages:
            values = " & ".join(f"{matrix[src][tgt]:.3f}" for tgt in languages)
            handle.write(f"\\texttt{{{src}}} & {values} \\\\\n")
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def compare_matrices(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]) -> dict[str, float]:
    max_abs_delta = 0.0
    compared = 0
    for src, row in left.items():
        for tgt, value in row.items():
            if src in right and tgt in right[src]:
                compared += 1
                max_abs_delta = max(max_abs_delta, abs(value - right[src][tgt]))
    return {"compared_cells": compared, "max_abs_delta": max_abs_delta}


def _urielplus_metadata(uriel: Any, languages: list[str], distance_type: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "generator": "thesis-eval prepare-uriel",
        "source": "urielplus",
        "urielplus_version": _package_version("urielplus"),
        "distance_type": distance_type,
        "languages_iso": languages,
    }
    for key, getter in {
        "codes": "get_codes",
        "distance_metric": "get_distance_metric",
        "aggregation": "get_aggregation",
        "fill_with_base_lang": "get_fill_with_base_lang",
        "cache": "get_cache",
    }.items():
        if hasattr(uriel, getter):
            try:
                metadata[key] = getattr(uriel, getter)()
            except Exception as exc:
                metadata[f"{key}_error"] = str(exc)

    coverage_getter = f"get_languages_with_{distance_type}_data"
    if hasattr(uriel, coverage_getter):
        try:
            available = set(getattr(uriel, coverage_getter)())
            metadata["coverage_count"] = len(available)
            metadata["missing_languages"] = [language for language in languages if language not in available]
        except Exception as exc:
            metadata["coverage_error"] = str(exc)
    return metadata


def _package_version(package_name: str) -> str:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return "unknown"
