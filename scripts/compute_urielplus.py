# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "urielplus==1.1",
# ]
# ///
from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version as package_version

from urielplus.urielplus import URIELPlus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages-json", required=True)
    parser.add_argument("--distance-type", default="featural")
    args = parser.parse_args()

    languages = json.loads(args.languages_json)
    uriel = URIELPlus()
    matrix = {source: {target: 0.0 for target in languages} for source in languages}
    for index, source in enumerate(languages):
        for target in languages[index + 1 :]:
            value = float(uriel.new_distance(args.distance_type, source, target))
            matrix[source][target] = value
            matrix[target][source] = value

    try:
        urielplus_version = package_version("urielplus")
    except PackageNotFoundError:
        urielplus_version = "unknown"

    metadata = {
        "generator": "thesis-eval prepare-uriel",
        "source": "urielplus",
        "urielplus_version": urielplus_version,
        "distance_type": args.distance_type,
        "languages_iso": languages,
        "python_executable": sys.executable,
        "codes": _safe_get(uriel, "get_codes"),
        "distance_metric": _safe_get(uriel, "get_distance_metric"),
        "aggregation": _safe_get(uriel, "get_aggregation"),
        "fill_with_base_lang": _safe_get(uriel, "get_fill_with_base_lang"),
        "cache": _safe_get(uriel, "get_cache"),
    }
    coverage_getter = f"get_languages_with_{args.distance_type}_data"
    if hasattr(uriel, coverage_getter):
        available = set(getattr(uriel, coverage_getter)())
        metadata["coverage_count"] = len(available)
        metadata["missing_languages"] = [language for language in languages if language not in available]

    print(json.dumps({"matrix": matrix, "metadata": metadata}))


def _safe_get(uriel: URIELPlus, name: str) -> object:
    if not hasattr(uriel, name):
        return None
    try:
        return getattr(uriel, name)()
    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    main()
