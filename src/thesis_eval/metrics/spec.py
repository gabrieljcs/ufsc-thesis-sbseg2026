from __future__ import annotations

import math
from typing import Any


def z_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def add_spec_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if_z = z_scores([float(row["if_score"]) for row in rows])
    cons_z = z_scores([float(row["cons_score"]) for row in rows])
    output: list[dict[str, Any]] = []
    for row, if_component, cons_component in zip(rows, if_z, cons_z, strict=True):
        enriched = dict(row)
        enriched["spec_score"] = 0.5 * if_component + 0.5 * cons_component
        output.append(enriched)
    return output
