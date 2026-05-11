from __future__ import annotations

from typing import Any


PLAN_A_HUMAN_LANGUAGES = {"ara", "bul", "ita", "por", "spa"}
AUDIT_CAVEAT_LANGUAGES = {"fin", "swa"}


AUDIT_COLUMNS = [
    "audit_plan",
    "audit_tier",
    "audit_method",
    "caveat_language",
    "prompt_id",
    "target_language",
    "source_text",
    "translated_text",
    "roundtrip_text",
    "roundtrip_drift_score",
    "roundtrip_status",
    "blaser_qe_score",
    "blaser_status",
    "harm_preservation",
    "xsts_semantic_equivalence",
    "xsts_attack_strength",
    "xsts_constraint_preservation",
    "xsts_naturalness",
    "xsts_no_harmfulness_drift",
    "xsts_score",
    "xsts_judge_model",
    "human_reviewer_id",
    "audit_decision",
    "xsts_note",
    "revision_note",
]


def audit_assignment(target_language: str, audit_plan: str) -> dict[str, Any]:
    if audit_plan not in {"plan_a", "plan_b"}:
        raise ValueError("audit_plan must be plan_a or plan_b")
    if target_language == "eng":
        return {
            "audit_plan": audit_plan,
            "audit_tier": "source",
            "audit_method": "none",
            "caveat_language": False,
        }
    if audit_plan == "plan_a" and target_language in PLAN_A_HUMAN_LANGUAGES:
        return {
            "audit_plan": audit_plan,
            "audit_tier": "tier_a",
            "audit_method": "human_xsts",
            "caveat_language": False,
        }
    return {
        "audit_plan": audit_plan,
        "audit_tier": "tier_b" if audit_plan == "plan_a" else "llm_only",
        "audit_method": "calibrated_llm_xsts" if audit_plan == "plan_a" else "uncalibrated_llm_xsts",
        "caveat_language": target_language in AUDIT_CAVEAT_LANGUAGES,
    }


def build_audit_queue(
    logs: list[dict[str, Any]],
    include_passed: bool = False,
    audit_plan: str = "plan_b",
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for log in logs:
        needs_audit = include_passed or log.get("blaser_status") in {"flag", "fail"} or log.get("harm_preservation") != "preserved"
        if not needs_audit:
            continue
        assignment = audit_assignment(str(log["target_language"]), audit_plan)
        queue.append(
            {
                **assignment,
                "prompt_id": log["prompt_id"],
                "target_language": log["target_language"],
                "source_text": log["source_text"],
                "translated_text": log["translated_text"],
                "roundtrip_text": log.get("roundtrip_text"),
                "roundtrip_drift_score": log.get("roundtrip_drift_score"),
                "roundtrip_status": log.get("roundtrip_status"),
                "blaser_qe_score": log.get("blaser_qe_score"),
                "blaser_status": log.get("blaser_status"),
                "harm_preservation": log.get("harm_preservation"),
                "xsts_semantic_equivalence": log.get("xsts_semantic_equivalence", "pending"),
                "xsts_attack_strength": log.get("xsts_attack_strength", "pending"),
                "xsts_constraint_preservation": log.get("xsts_constraint_preservation", "pending"),
                "xsts_naturalness": log.get("xsts_naturalness", "pending"),
                "xsts_no_harmfulness_drift": log.get("xsts_no_harmfulness_drift", "pending"),
                "xsts_score": log.get("xsts_score", ""),
                "xsts_judge_model": log.get("xsts_judge_model", ""),
                "human_reviewer_id": log.get("human_reviewer_id", ""),
                "audit_decision": "",
                "xsts_note": "",
                "revision_note": "",
            }
        )
    return queue


def apply_audit_decisions(logs: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decision_by_key = {(row["prompt_id"], row["target_language"]): row for row in decisions}
    output: list[dict[str, Any]] = []
    for log in logs:
        row = dict(log)
        decision = decision_by_key.get((row["prompt_id"], row["target_language"]))
        if decision:
            audit_decision = str(decision.get("audit_decision", "")).strip().lower()
            if audit_decision not in {"pass", "revise", "exclude", "reject"}:
                raise ValueError(f"audit_decision must be pass, revise, exclude, or reject for {row['prompt_id']}:{row['target_language']}")
            if audit_decision == "pass":
                row["human_audit_status"] = "audited_pass"
            elif audit_decision == "revise":
                row["human_audit_status"] = "audited_revised"
            else:
                row["human_audit_status"] = "audited_excluded"
                row["translation_excluded"] = True
                row["exclusion_reason"] = "translation_audit_rejected"
            if decision.get("revision_note"):
                row["revision_note"] = decision["revision_note"]
            if decision.get("harm_preservation") in {"preserved", "attenuated", "amplified"}:
                row["harm_preservation"] = decision["harm_preservation"]
            if decision.get("translated_text") and str(decision["translated_text"]) != str(row.get("translated_text")):
                row["translated_text"] = decision["translated_text"]
                row["translation_revised"] = True
            for key in (
                "audit_plan",
                "audit_tier",
                "audit_method",
                "caveat_language",
                "xsts_score",
                "xsts_judge_model",
                "human_reviewer_id",
                "xsts_semantic_equivalence",
                "xsts_attack_strength",
                "xsts_constraint_preservation",
                "xsts_naturalness",
                "xsts_no_harmfulness_drift",
            ):
                if decision.get(key):
                    row[key] = decision[key]
            if audit_decision == "revise":
                row["translation_revised"] = True
        output.append(row)
    return output
