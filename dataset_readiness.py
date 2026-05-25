#!/usr/bin/env python3
"""Sprint 5 full-dataset readiness validators.

These checks are deterministic infrastructure gates. They intentionally do not
perform semantic embedding, NLI, hosted-model, GPU, or network work.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from persona_eval import (
    PERSONA_SCHEMA_PATH,
    REPO_ROOT,
    PersonaValidationError,
    load_jsonl,
    validate_behavior_labels,
    validate_personas,
    validate_schema_file,
)


READINESS_VERSION = "sprint5"
REVIEW_MANIFEST_SCHEMA_PATH = REPO_ROOT / "schemas" / "review_manifest.schema.json"
DEFAULT_SAMPLE_PATH = REPO_ROOT / "data" / "personas.sample.jsonl"
DEFAULT_REVIEW_MANIFEST_PATH = REPO_ROOT / "reviews" / "personas.sample.review.jsonl"
ALLOWED_DATA_FILENAMES = {"personas.sample.jsonl"}
PASSING_REVIEW_STATUSES = {"approved", "sample_reviewed"}
PASSING_GATE_STATUSES = {"passed", "manual_pass"}
INCOMPLETE_GATE_STATUSES = {"manual_required", "blocked", "not_run"}
READINESS_CHECKS = (
    "sample_schema_tests_pass",
    "variant_validation_tests_pass",
    "source_license_checks_pass",
    "pii_and_real_person_filters_exist",
    "restricted_role_filters_exist",
    "semantic_equivalence_validation_exists",
    "nli_contradiction_equivalence_checks_exist",
    "gold_label_preview_checks_exist",
    "dedupe_checks_exist",
    "human_review_manifest_or_metadata_exists",
    "human_review_coverage_sufficient",
    "unvalidated_candidates_outside_data",
)

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    (
        "phone_number",
        re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    ),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "street_address",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][A-Za-z0-9.'-]*\s+"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
            re.IGNORECASE,
        ),
    ),
)
KNOWN_REAL_PERSON_NAMES = {
    "barack obama",
    "beyonce",
    "donald trump",
    "elon musk",
    "joe biden",
    "mark zuckerberg",
    "oprah winfrey",
    "sam altman",
    "taylor swift",
}
REAL_PERSON_TRIGGER = re.compile(
    r"\b(?:act as|as|be|become|impersonate|pretend to be|roleplay as)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\b"
)
RESTRICTED_PATTERNS: dict[str, tuple[str, ...]] = {
    "medical_decision_role": (
        r"\bdiagnose\b",
        r"\bprescribe\b",
        r"\btreatment directive\b",
        r"\bmedical decision\b",
        r"\bdecide (?:a )?patient(?:'s)? treatment\b",
    ),
    "legal_guarantee_role": (
        r"\blegal guarantee\b",
        r"\bguarantee (?:a |the )?(?:legal )?(?:outcome|result|win)\b",
        r"\bguarantee (?:a |the )?(?:case|appeal|lawsuit) (?:will|shall) (?:win|succeed)\b",
        r"\bpromise (?:a |the )?(?:case|appeal|lawsuit) (?:will|shall) (?:win|succeed)\b",
    ),
    "self_harm_role": (
        r"\bself[-\s]?harm\b",
        r"\bsuicide\b",
        r"\bhelp (?:me|someone) die\b",
        r"\bkill myself\b",
    ),
    "extremist_role": (
        r"\bterrorist\b",
        r"\bextremist\b",
        r"\bisis\b",
        r"\bnazi organizer\b",
        r"\bradicali[sz]ation\b",
    ),
    "fraud_deception_role": (
        r"\bphishing\b",
        r"\bscam\b",
        r"\bfake invoice\b",
        r"\bidentity theft\b",
        r"\bdeceive\b",
        r"\bforge\b",
    ),
    "credential_impersonation_role": (
        r"\bfake credential",
        r"\binvent credential",
        r"\bimpersonate credential",
        r"\bpretend (?:i am|to be) (?:licensed|certified|board[-\s]?certified)\b",
        r"\bclaim (?:i am|to be) (?:licensed|certified|board[-\s]?certified) without\b",
    ),
}
MITIGATING_PREFIXES = (
    "avoid",
    "do not",
    "don't",
    "never",
    "refuse",
    "without",
    "forbidden",
    "prevent",
)


@dataclass(frozen=True)
class TextField:
    path: str
    text: str
    restricted_role_relevant: bool


def status_payload(status: str, reason_code: str | None, evidence: str | None = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "reason_code": reason_code,
        "evidence": evidence,
    }
    payload.update(extra)
    return payload


def finding(
    *,
    persona_id: str,
    category: str,
    field: str,
    matched_text: str,
    reason_code: str,
) -> dict[str, str]:
    return {
        "persona_id": persona_id,
        "category": category,
        "field": field,
        "matched_text": matched_text,
        "reason_code": reason_code,
    }


def _iter_string_fields(value: Any, *, path: str = "", restricted_role_relevant: bool = True) -> list[TextField]:
    fields: list[TextField] = []
    if isinstance(value, str):
        fields.append(TextField(path=path or "<root>", text=value, restricted_role_relevant=restricted_role_relevant))
    elif isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            child_relevant = restricted_role_relevant and key not in {"source", "forbidden_behaviors"}
            fields.extend(
                _iter_string_fields(
                    child,
                    path=child_path,
                    restricted_role_relevant=child_relevant,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            fields.extend(
                _iter_string_fields(
                    child,
                    path=child_path,
                    restricted_role_relevant=restricted_role_relevant,
                )
            )
    return fields


def row_text_fields(row: dict[str, Any]) -> list[TextField]:
    return _iter_string_fields(row)


def normalize_text_for_dedupe(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def persona_dedupe_text(row: dict[str, Any]) -> str:
    fields: list[str] = []
    persona_spec = row.get("persona_spec", {})
    seed_prompt = row.get("seed_prompt", {})
    annotation = row.get("annotation", {})
    for value in (
        persona_spec.get("domain", "") if isinstance(persona_spec, dict) else "",
        seed_prompt.get("text", "") if isinstance(seed_prompt, dict) else "",
        row.get("expected_behavior", {}),
        annotation.get("gold_labels", {}) if isinstance(annotation, dict) else {},
    ):
        fields.append(json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value))
    if isinstance(persona_spec, dict):
        for key in ("facts", "traits", "forbidden_behaviors"):
            fields.append(json.dumps(persona_spec.get(key, []), sort_keys=True))
    variants = row.get("variants", [])
    if isinstance(variants, list):
        for variant in variants:
            if isinstance(variant, dict):
                fields.append(str(variant.get("text", "")))
    return "\n".join(fields)


def _matched_known_real_person(text: str) -> str | None:
    lowered = text.lower()
    for name in sorted(KNOWN_REAL_PERSON_NAMES):
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return name
    trigger = REAL_PERSON_TRIGGER.search(text)
    if trigger:
        candidate = normalize_text_for_dedupe(trigger.group(1))
        if candidate in KNOWN_REAL_PERSON_NAMES:
            return candidate
    return None


def pii_real_person_findings(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for row in rows:
        persona_id = str(row.get("persona_id", "<missing>"))
        for field in row_text_fields(row):
            for category, pattern in PII_PATTERNS:
                match = pattern.search(field.text)
                if match:
                    findings.append(
                        finding(
                            persona_id=persona_id,
                            category=category,
                            field=field.path,
                            matched_text=match.group(0),
                            reason_code="pii_indicator_detected",
                        )
                    )
            real_person = _matched_known_real_person(field.text)
            if real_person is not None:
                findings.append(
                    finding(
                        persona_id=persona_id,
                        category="real_person_indicator",
                        field=field.path,
                        matched_text=real_person,
                        reason_code="known_real_person_indicator_detected",
                    )
                )
    return findings


def pii_real_person_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    findings = pii_real_person_findings(rows)
    if findings:
        return {
            "status": "blocked",
            "reason_code": "pii_or_real_person_findings_present",
            "row_count": len(rows),
            "finding_count": len(findings),
            "findings": findings,
        }
    return {
        "status": "pass",
        "reason_code": None,
        "row_count": len(rows),
        "finding_count": 0,
        "findings": [],
        "evidence": "Deterministic PII and known-real-person indicators returned no findings.",
    }


def _has_mitigating_prefix(text: str, start: int) -> bool:
    prefix = text[max(0, start - 42):start].lower()
    return any(token in prefix for token in MITIGATING_PREFIXES)


def restricted_role_findings(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for row in rows:
        persona_id = str(row.get("persona_id", "<missing>"))
        for field in row_text_fields(row):
            if not field.restricted_role_relevant:
                continue
            for category, patterns in RESTRICTED_PATTERNS.items():
                for raw_pattern in patterns:
                    pattern = re.compile(raw_pattern, re.IGNORECASE)
                    match = pattern.search(field.text)
                    if not match or _has_mitigating_prefix(field.text, match.start()):
                        continue
                    findings.append(
                        finding(
                            persona_id=persona_id,
                            category=category,
                            field=field.path,
                            matched_text=match.group(0),
                            reason_code="restricted_role_indicator_detected",
                        )
                    )
                    break
    return findings


def restricted_role_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    findings = restricted_role_findings(rows)
    if findings:
        return {
            "status": "blocked",
            "reason_code": "restricted_role_findings_present",
            "row_count": len(rows),
            "finding_count": len(findings),
            "findings": findings,
        }
    return {
        "status": "pass",
        "reason_code": None,
        "row_count": len(rows),
        "finding_count": 0,
        "findings": [],
        "evidence": "Deterministic restricted-role indicators returned no findings.",
    }


def exact_duplicate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text_for_dedupe(persona_dedupe_text(row))].append(str(row.get("persona_id", "<missing>")))
    return [
        {
            "normalized_text_hash": f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}",
            "persona_ids": sorted(persona_ids),
        }
        for text, persona_ids in sorted(grouped.items())
        if len(persona_ids) > 1
    ]


def deterministic_text_similarity(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    token_overlap = 0.0
    if left_tokens and right_tokens:
        token_overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    sequence_ratio = difflib.SequenceMatcher(None, left, right).ratio()
    return max(token_overlap, sequence_ratio)


def near_duplicate_pairs(rows: list[dict[str, Any]], *, threshold: float = 0.92) -> list[dict[str, Any]]:
    if threshold <= 0.0 or threshold > 1.0:
        raise PersonaValidationError("near-duplicate threshold must be in (0, 1]")
    normalized = [
        (str(row.get("persona_id", "<missing>")), normalize_text_for_dedupe(persona_dedupe_text(row)))
        for row in rows
    ]
    pairs: list[dict[str, Any]] = []
    for left_index in range(len(normalized)):
        left_id, left_text = normalized[left_index]
        for right_index in range(left_index + 1, len(normalized)):
            right_id, right_text = normalized[right_index]
            if left_text == right_text:
                continue
            ratio = deterministic_text_similarity(left_text, right_text)
            if ratio >= threshold:
                pairs.append(
                    {
                        "persona_ids": [left_id, right_id],
                        "similarity": ratio,
                        "threshold": threshold,
                    }
                )
    return pairs


def dedupe_report(rows: list[dict[str, Any]], *, near_threshold: float = 0.92) -> dict[str, Any]:
    exact_groups = exact_duplicate_groups(rows)
    near_pairs = near_duplicate_pairs(rows, threshold=near_threshold)
    status = "blocked" if exact_groups or near_pairs else "manual_required"
    reason_code = None
    if exact_groups:
        reason_code = "exact_duplicate_persona_text"
    elif near_pairs:
        reason_code = "near_duplicate_persona_text"
    else:
        reason_code = "embedding_cluster_review_manual_required"
    return {
        "status": status,
        "reason_code": reason_code,
        "row_count": len(rows),
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_threshold": near_threshold,
        "embedding_cluster_review": {
            "status": "manual_required",
            "reason_code": "real_embedding_cluster_review_not_implemented",
            "evidence": (
                "Sprint 5 includes deterministic exact and near-duplicate checks only; "
                "embedding-cluster review must be recorded before full readiness."
            ),
        },
    }


def candidate_location_report(root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root_path = Path(root)
    data_dir = root_path / "data"
    unexpected: list[str] = []
    if data_dir.exists():
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(data_dir).as_posix()
            if rel not in ALLOWED_DATA_FILENAMES:
                unexpected.append(rel)
    if unexpected:
        return {
            "status": "blocked",
            "reason_code": "unexpected_candidate_or_full_files_under_data",
            "unexpected_files": unexpected,
            "evidence": "Only data/personas.sample.jsonl is allowed before the full-dataset gate passes.",
        }
    return {
        "status": "pass",
        "reason_code": None,
        "unexpected_files": [],
        "evidence": "No unvalidated candidate or full dataset files are under data/.",
    }


def load_review_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for index, row in enumerate(rows, start=1):
        try:
            validate_schema_file(row, REVIEW_MANIFEST_SCHEMA_PATH)
        except PersonaValidationError as exc:
            raise PersonaValidationError(f"review manifest row {index}: {exc}") from exc
    return rows


def validate_review_manifest_for_personas(
    review_rows: list[dict[str, Any]],
    persona_rows: list[dict[str, Any]],
) -> None:
    persona_ids = {row["persona_id"] for row in persona_rows}
    review_ids = [row["persona_id"] for row in review_rows]
    duplicates = sorted(persona_id for persona_id, count in Counter(review_ids).items() if count > 1)
    if duplicates:
        raise PersonaValidationError("duplicate review manifest persona_id values: " + ", ".join(duplicates))
    unknown = sorted(set(review_ids) - persona_ids)
    if unknown:
        raise PersonaValidationError("review manifest contains unknown persona_id values: " + ", ".join(unknown))


def review_manifest_report(
    persona_rows: list[dict[str, Any]],
    *,
    review_manifest_path: str | Path | None = DEFAULT_REVIEW_MANIFEST_PATH,
) -> dict[str, Any]:
    if review_manifest_path is None:
        return status_payload(
            "blocked",
            "review_manifest_not_configured",
            "No review manifest path was provided.",
            review_rows=[],
        )
    path = Path(review_manifest_path)
    if not path.exists():
        return status_payload(
            "blocked",
            "review_manifest_missing",
            f"{path} does not exist.",
            review_rows=[],
        )
    try:
        review_rows = load_review_manifest(path)
        validate_review_manifest_for_personas(review_rows, persona_rows)
    except PersonaValidationError as exc:
        return status_payload(
            "blocked",
            "review_manifest_invalid",
            str(exc),
            review_rows=[],
        )
    return status_payload(
        "pass",
        None,
        f"{path} validates against schemas/review_manifest.schema.json.",
        review_rows=review_rows,
        reviewed_persona_count=len({row["persona_id"] for row in review_rows}),
    )


def _review_row_has_evidence(row: dict[str, Any]) -> bool:
    if not str(row.get("review_reason", "")).strip():
        return False
    for field in (
        "semantic_equivalence_status",
        "nli_equivalence_status",
        "contradiction_status",
        "safety_review_status",
        "gold_label_review_status",
    ):
        status = row.get(field, {})
        evidence = status.get("evidence") if isinstance(status, dict) else None
        if isinstance(evidence, list) and any(str(item).strip() for item in evidence):
            return True
    return False


def review_coverage_report(
    persona_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    *,
    low_confidence_persona_ids: set[str] | None = None,
    minimum_fraction: float = 0.10,
    require_full_dataset_scope: bool = False,
) -> dict[str, Any]:
    if minimum_fraction < 0.0 or minimum_fraction > 1.0:
        raise PersonaValidationError("minimum review fraction must be in [0, 1]")
    persona_ids = {row["persona_id"] for row in persona_rows}
    low_confidence_persona_ids = low_confidence_persona_ids or set()
    review_by_id = {row["persona_id"]: row for row in review_rows}
    reviewed_with_evidence = {
        persona_id
        for persona_id, row in review_by_id.items()
        if row.get("review_status") in PASSING_REVIEW_STATUSES and _review_row_has_evidence(row)
    }
    missing_low_confidence = sorted(low_confidence_persona_ids - reviewed_with_evidence)
    minimum_required = math.ceil(len(persona_ids) * minimum_fraction) if persona_ids else 0
    if missing_low_confidence:
        return {
            "status": "blocked",
            "reason_code": "low_confidence_rows_missing_review_evidence",
            "missing_low_confidence_persona_ids": missing_low_confidence,
            "reviewed_with_evidence_count": len(reviewed_with_evidence),
            "minimum_required_review_count": minimum_required,
        }
    if len(reviewed_with_evidence) < minimum_required:
        return {
            "status": "blocked",
            "reason_code": "minimum_review_coverage_not_met",
            "missing_low_confidence_persona_ids": [],
            "reviewed_with_evidence_count": len(reviewed_with_evidence),
            "minimum_required_review_count": minimum_required,
        }
    if require_full_dataset_scope and len(persona_ids) < 200:
        return {
            "status": "blocked",
            "reason_code": "full_dataset_review_scope_missing",
            "missing_low_confidence_persona_ids": [],
            "reviewed_with_evidence_count": len(reviewed_with_evidence),
            "minimum_required_review_count": minimum_required,
            "persona_count": len(persona_ids),
            "evidence": "Review workflow exists, but current evidence covers only the fixture/sample scope.",
        }
    return {
        "status": "pass",
        "reason_code": None,
        "missing_low_confidence_persona_ids": [],
        "reviewed_with_evidence_count": len(reviewed_with_evidence),
        "minimum_required_review_count": minimum_required,
        "evidence": "Review manifest covers all low-confidence rows and the configured minimum fraction.",
    }


def gold_label_preview_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, str]] = []
    for row in rows:
        try:
            validate_behavior_labels(row)
        except PersonaValidationError as exc:
            mismatches.append(
                {
                    "persona_id": str(row.get("persona_id", "<missing>")),
                    "reason_code": "behavior_label_shape_invalid",
                    "detail": str(exc),
                }
            )
            continue
        if row["expected_behavior"] != row["annotation"]["gold_labels"]:
            mismatches.append(
                {
                    "persona_id": row["persona_id"],
                    "reason_code": "expected_behavior_gold_label_mismatch",
                    "detail": "expected_behavior must match annotation.gold_labels for preview fixtures",
                }
            )
    if mismatches:
        return {
            "status": "blocked",
            "reason_code": "gold_label_preview_mismatches",
            "mismatches": mismatches,
        }
    return {
        "status": "pass",
        "reason_code": None,
        "mismatches": [],
        "evidence": "Expected behavior and gold-label preview fields are structurally consistent.",
    }


def manual_gate_report(
    review_rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    ready_reason_code: str,
    blocked_reason_code: str,
) -> dict[str, Any]:
    incomplete: list[dict[str, str]] = []
    for row in review_rows:
        for field in fields:
            gate = row.get(field, {})
            status = gate.get("status") if isinstance(gate, dict) else None
            evidence = gate.get("evidence") if isinstance(gate, dict) else None
            if status not in PASSING_GATE_STATUSES or not isinstance(evidence, list) or not evidence:
                incomplete.append(
                    {
                        "persona_id": str(row.get("persona_id", "<missing>")),
                        "field": field,
                        "status": str(status),
                    }
                )
    if incomplete:
        return {
            "status": "blocked",
            "reason_code": blocked_reason_code,
            "incomplete": incomplete,
        }
    return {
        "status": "pass",
        "reason_code": None,
        "evidence": ready_reason_code,
    }


def build_low_confidence_ids(
    *,
    safety_findings: list[dict[str, str]],
    restricted_findings: list[dict[str, str]],
    exact_duplicate_groups_value: list[dict[str, Any]],
    near_duplicate_pairs_value: list[dict[str, Any]],
) -> set[str]:
    ids = {finding["persona_id"] for finding in safety_findings}
    ids.update(finding["persona_id"] for finding in restricted_findings)
    for group in exact_duplicate_groups_value:
        ids.update(group["persona_ids"])
    for pair in near_duplicate_pairs_value:
        ids.update(pair["persona_ids"])
    return ids


def full_dataset_readiness_report(
    *,
    root: str | Path = REPO_ROOT,
    persona_path: str | Path = DEFAULT_SAMPLE_PATH,
    review_manifest_path: str | Path | None = DEFAULT_REVIEW_MANIFEST_PATH,
) -> dict[str, Any]:
    root_path = Path(root)
    checks: dict[str, dict[str, Any]] = {}
    try:
        rows = validate_personas(persona_path)
        sample_error = None
    except Exception as exc:  # pragma: no cover - defensive reporting path.
        rows = []
        sample_error = str(exc)

    sample_ok = sample_error is None
    checks["sample_schema_tests_pass"] = (
        status_payload("pass", None, "data/personas.sample.jsonl validates against schemas.")
        if sample_ok and (root_path / "tests" / "test_schema.py").exists()
        else status_payload("blocked", "sample_schema_validation_missing_or_failing", sample_error)
    )
    checks["variant_validation_tests_pass"] = (
        status_payload("pass", None, "Variant-count/type validation is implemented and sample rows pass.")
        if sample_ok and (root_path / "tests" / "test_variants.py").exists()
        else status_payload("blocked", "variant_validation_missing_or_failing", sample_error)
    )
    checks["source_license_checks_pass"] = (
        status_payload("pass", None, "Source/license metadata is schema-validated for current rows.")
        if sample_ok and PERSONA_SCHEMA_PATH.exists()
        else status_payload("blocked", "source_license_validation_missing_or_failing", sample_error)
    )

    pii_report = pii_real_person_report(rows) if rows else {"status": "blocked", "reason_code": "no_rows"}
    checks["pii_and_real_person_filters_exist"] = (
        status_payload("pass", None, pii_report.get("evidence"), details=pii_report)
        if pii_report["status"] == "pass"
        else status_payload("blocked", pii_report.get("reason_code"), "PII/real-person findings must be resolved.", details=pii_report)
    )

    restricted_report = restricted_role_report(rows) if rows else {"status": "blocked", "reason_code": "no_rows"}
    checks["restricted_role_filters_exist"] = (
        status_payload("pass", None, restricted_report.get("evidence"), details=restricted_report)
        if restricted_report["status"] == "pass"
        else status_payload("blocked", restricted_report.get("reason_code"), "Restricted-role findings must be resolved.", details=restricted_report)
    )

    gold_report = gold_label_preview_report(rows) if rows else {"status": "blocked", "reason_code": "no_rows"}
    checks["gold_label_preview_checks_exist"] = (
        status_payload("pass", None, gold_report.get("evidence"), details=gold_report)
        if gold_report["status"] == "pass"
        else status_payload("blocked", gold_report.get("reason_code"), "Gold-label preview mismatches must be resolved.", details=gold_report)
    )

    dedupe = dedupe_report(rows) if rows else {"status": "blocked", "reason_code": "no_rows"}
    checks["dedupe_checks_exist"] = (
        status_payload("pass", None, "Exact, near-duplicate, and embedding-cluster dedupe evidence is complete.", details=dedupe)
        if dedupe["status"] == "pass"
        else status_payload(
            "blocked",
            dedupe.get("reason_code"),
            "Exact/near duplicate validators exist, but embedding-cluster review remains manual-required.",
            details=dedupe,
        )
    )

    review_report = review_manifest_report(rows, review_manifest_path=review_manifest_path) if rows else {
        "status": "blocked",
        "reason_code": "no_rows",
        "review_rows": [],
    }
    review_rows = review_report.get("review_rows", [])
    checks["human_review_manifest_or_metadata_exists"] = (
        status_payload("pass", None, review_report.get("evidence"), details={k: v for k, v in review_report.items() if k != "review_rows"})
        if review_report["status"] == "pass"
        else status_payload("blocked", review_report.get("reason_code"), review_report.get("evidence"), details=review_report)
    )

    semantic_report = manual_gate_report(
        review_rows,
        fields=("semantic_equivalence_status",),
        ready_reason_code="Semantic equivalence status has review evidence for every row.",
        blocked_reason_code="semantic_equivalence_manual_required_or_missing",
    )
    checks["semantic_equivalence_validation_exists"] = (
        status_payload("pass", None, semantic_report.get("evidence"), details=semantic_report)
        if semantic_report["status"] == "pass"
        else status_payload("blocked", semantic_report.get("reason_code"), "Semantic equivalence requires real validator or complete review evidence.", details=semantic_report)
    )

    nli_report = manual_gate_report(
        review_rows,
        fields=("nli_equivalence_status", "contradiction_status"),
        ready_reason_code="NLI equivalence and contradiction statuses have review evidence for every row.",
        blocked_reason_code="nli_or_contradiction_manual_required_or_missing",
    )
    checks["nli_contradiction_equivalence_checks_exist"] = (
        status_payload("pass", None, nli_report.get("evidence"), details=nli_report)
        if nli_report["status"] == "pass"
        else status_payload("blocked", nli_report.get("reason_code"), "NLI/contradiction checks require real validator or complete review evidence.", details=nli_report)
    )

    low_confidence_ids = build_low_confidence_ids(
        safety_findings=pii_report.get("findings", []),
        restricted_findings=restricted_report.get("findings", []),
        exact_duplicate_groups_value=dedupe.get("exact_duplicate_groups", []),
        near_duplicate_pairs_value=dedupe.get("near_duplicate_pairs", []),
    )
    coverage = review_coverage_report(
        rows,
        review_rows,
        low_confidence_persona_ids=low_confidence_ids,
        require_full_dataset_scope=True,
    ) if review_rows else {
        "status": "blocked",
        "reason_code": "review_manifest_missing",
    }
    checks["human_review_coverage_sufficient"] = (
        status_payload("pass", None, coverage.get("evidence"), details=coverage)
        if coverage["status"] == "pass"
        else status_payload("blocked", coverage.get("reason_code"), coverage.get("evidence"), details=coverage)
    )

    candidate_report = candidate_location_report(root_path)
    checks["unvalidated_candidates_outside_data"] = (
        status_payload("pass", None, candidate_report.get("evidence"), details=candidate_report)
        if candidate_report["status"] == "pass"
        else status_payload("blocked", candidate_report.get("reason_code"), candidate_report.get("evidence"), details=candidate_report)
    )

    overall = "ready" if all(check["status"] == "pass" for check in checks.values()) else "blocked"
    blocking = [name for name in READINESS_CHECKS if checks[name]["status"] != "pass"]
    return {
        "status": overall,
        "readiness_version": READINESS_VERSION,
        "checks": {name: checks[name] for name in READINESS_CHECKS},
        "blocking_checks": blocking,
        "note": "Full dataset generation must stay blocked until every check passes with real evidence.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic full-dataset readiness gates")
    parser.add_argument("--persona-path", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    parser.add_argument("--json", action="store_true", help="print the full readiness report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    review_path: str | None = args.review_manifest
    if review_path == "":
        review_path = None
    try:
        report = full_dataset_readiness_report(
            persona_path=args.persona_path,
            review_manifest_path=review_path,
        )
    except (OSError, json.JSONDecodeError, PersonaValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        print(f"dataset_readiness={report['status']}")
        print("blocking_checks=" + ",".join(report["blocking_checks"]))
    return 0 if report["status"] in {"ready", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
