from __future__ import annotations

from typing import Any


def calibrate_blaser_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed: list[float] = []
    revised: list[float] = []
    for row in rows:
        score_value = row.get("blaser_qe_score")
        decision = str(row.get("audit_decision") or row.get("human_audit_status") or "").strip().lower()
        if score_value in {None, ""}:
            continue
        score = float(score_value)
        if decision in {"pass", "audited_pass"}:
            passed.append(score)
        elif decision in {"revise", "fail", "audited_revised"}:
            revised.append(score)
    if not passed and not revised:
        raise ValueError("No audited rows with BLASER scores were available for calibration")
    pass_threshold = min(passed) if passed else max(revised)
    fail_threshold = max(revised) if revised else min(passed)
    if fail_threshold > pass_threshold:
        midpoint = (fail_threshold + pass_threshold) / 2
        fail_threshold = midpoint
        pass_threshold = midpoint
    return {
        "audited_pass": len(passed),
        "audited_revised": len(revised),
        "pass_threshold": round(pass_threshold, 4),
        "fail_threshold": round(fail_threshold, 4),
    }
