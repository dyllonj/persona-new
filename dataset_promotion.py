#!/usr/bin/env python3
"""Sprint 7 dataset promotion gates.

This module prepares the promotion path without creating the full dataset by
default. It validates a proposed promotion set and reports blockers; writes are
allowed only for a non-dry run whose gates all pass.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys
from typing import Any

import dataset_readiness
from persona_eval import (
    PERSONA_SCHEMA_PATH,
    REPO_ROOT,
    PersonaValidationError,
    hash_file_bytes,
    load_jsonl,
    load_schema,
    validate_persona_row,
)


PROMOTION_VERSION = "sprint7_dry_run"
TARGET_PROMOTION_ROWS = 200
DEFAULT_FULL_DATASET_PATH = REPO_ROOT / "data" / "personas.full.jsonl"
DEFAULT_PROMOTION_MANIFEST_PATH = REPO_ROOT / "reports" / "dataset_promotion_manifest.json"
PERSONA_TOP_LEVEL_FIELDS = set(load_schema(PERSONA_SCHEMA_PATH)["properties"])
PASSING_PROMOTION_REVIEW_STATUSES = {"approved"}
PROMOTION_REVIEW_GATE_FIELDS = (
    "semantic_equivalence_status",
    "nli_equivalence_status",
    "contradiction_status",
    "safety_review_status",
    "gold_label_review_status",
)


def _relative_or_absolute(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _empty_report(
    *,
    candidate_path: str | Path,
    review_path: str | Path,
    out_path: str | Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "promotion_version": PROMOTION_VERSION,
        "status": "blocked",
        "dry_run": dry_run,
        "target_count": TARGET_PROMOTION_ROWS,
        "candidate_path": _relative_or_absolute(candidate_path),
        "review_path": _relative_or_absolute(review_path),
        "out_path": _relative_or_absolute(out_path) if out_path is not None else None,
        "candidate_input_hash": None,
        "review_manifest_hash": None,
        "promoted_output_hash": None,
        "candidate_row_count": 0,
        "valid_candidate_count": 0,
        "approved_candidate_count": 0,
        "write_permitted": False,
        "write_performed": False,
        "rejection_counts": {},
        "rejections": [],
    }


def _add_rejection(
    report: dict[str, Any],
    *,
    reason_code: str,
    persona_id: str | None = None,
    detail: str | None = None,
    category: str | None = None,
) -> None:
    rejection: dict[str, Any] = {"reason_code": reason_code}
    if persona_id is not None:
        rejection["persona_id"] = persona_id
    if category is not None:
        rejection["category"] = category
    if detail is not None:
        rejection["detail"] = detail
    report["rejections"].append(rejection)


def _finalize_rejection_counts(report: dict[str, Any]) -> None:
    report["rejection_counts"] = dict(sorted(Counter(row["reason_code"] for row in report["rejections"]).items()))


def _candidate_low_confidence_flags(raw_row: dict[str, Any]) -> list[str]:
    candidates = [
        raw_row.get("low_confidence_flags"),
        raw_row.get("candidate_meta", {}).get("low_confidence_flags")
        if isinstance(raw_row.get("candidate_meta"), dict)
        else None,
        raw_row.get("metadata", {}).get("low_confidence_flags")
        if isinstance(raw_row.get("metadata"), dict)
        else None,
    ]
    for value in candidates:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
    return []


def _candidate_to_persona_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    """Return the persona payload from a candidate row.

    Sprint 6 may wrap the eventual persona item with candidate metadata. The
    promotion gate accepts either a bare persona row or a row with a nested
    `persona` object. Bare rows may include candidate metadata, which is ignored
    for the promoted output.
    """
    if isinstance(raw_row.get("persona"), dict):
        return copy.deepcopy(raw_row["persona"])
    return {field: copy.deepcopy(raw_row[field]) for field in PERSONA_TOP_LEVEL_FIELDS if field in raw_row}


def load_candidate_personas(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    raw_rows = load_jsonl(path)
    persona_rows: list[dict[str, Any]] = []
    low_confidence_by_id: dict[str, list[str]] = {}
    invalid_rows: list[dict[str, Any]] = []
    schema = load_schema(PERSONA_SCHEMA_PATH)
    seen_ids: set[str] = set()

    for index, raw_row in enumerate(raw_rows, start=1):
        persona_row = _candidate_to_persona_row(raw_row)
        persona_id = str(persona_row.get("persona_id", f"<row-{index}>"))
        try:
            validate_persona_row(persona_row, schema)
            if persona_id in seen_ids:
                raise PersonaValidationError(f"duplicate persona_id: {persona_id}")
        except PersonaValidationError as exc:
            invalid_rows.append(
                {
                    "row_number": index,
                    "persona_id": persona_id,
                    "reason_code": "candidate_schema_invalid",
                    "detail": str(exc),
                }
            )
            continue
        seen_ids.add(persona_id)
        persona_rows.append(persona_row)
        low_confidence_by_id[persona_id] = _candidate_low_confidence_flags(raw_row)

    return persona_rows, low_confidence_by_id, invalid_rows


def _review_has_text(row: dict[str, Any], field: str) -> bool:
    return isinstance(row.get(field), str) and bool(row[field].strip())


def _review_gate_passes(row: dict[str, Any], field: str) -> bool:
    gate = row.get(field)
    if not isinstance(gate, dict):
        return False
    evidence = gate.get("evidence")
    return (
        gate.get("status") in dataset_readiness.PASSING_GATE_STATUSES
        and isinstance(evidence, list)
        and any(str(item).strip() for item in evidence)
    )


def _review_has_required_evidence(row: dict[str, Any]) -> bool:
    if row.get("review_status") not in PASSING_PROMOTION_REVIEW_STATUSES:
        return False
    if not _review_has_text(row, "reviewer"):
        return False
    if not _review_has_text(row, "reviewed_at"):
        return False
    if not _review_has_text(row, "review_reason"):
        return False
    if "unresolved_concerns" not in row:
        return False
    unresolved = row.get("unresolved_concerns")
    if not isinstance(unresolved, list) or unresolved:
        return False
    return all(_review_gate_passes(row, field) for field in PROMOTION_REVIEW_GATE_FIELDS)


def _review_failure_reason(row: dict[str, Any] | None) -> str:
    if row is None:
        return "review_evidence_missing"
    if row.get("review_status") not in PASSING_PROMOTION_REVIEW_STATUSES:
        return "review_status_not_approved"
    if not all(_review_has_text(row, field) for field in ("reviewer", "reviewed_at", "review_reason")):
        return "review_metadata_missing"
    if "unresolved_concerns" not in row:
        return "unresolved_concerns_missing"
    unresolved = row.get("unresolved_concerns")
    if not isinstance(unresolved, list) or unresolved:
        return "unresolved_review_concerns"
    return "review_gate_evidence_incomplete"


def _load_review_rows(path: str | Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return dataset_readiness.load_review_manifest(path)
    except PersonaValidationError as exc:
        _add_rejection(
            report,
            reason_code="review_manifest_invalid",
            detail=str(exc),
        )
        return []


def _add_safety_rejections(report: dict[str, Any], persona_rows: list[dict[str, Any]]) -> None:
    for finding in dataset_readiness.pii_real_person_findings(persona_rows):
        _add_rejection(
            report,
            reason_code="safety_filter_blocked",
            persona_id=finding["persona_id"],
            category=finding["category"],
            detail=f"{finding['field']}: {finding['matched_text']}",
        )
    for finding in dataset_readiness.restricted_role_findings(persona_rows):
        _add_rejection(
            report,
            reason_code="restricted_category_blocked",
            persona_id=finding["persona_id"],
            category=finding["category"],
            detail=f"{finding['field']}: {finding['matched_text']}",
        )


def _add_dedupe_rejections(report: dict[str, Any], persona_rows: list[dict[str, Any]]) -> None:
    for group in dataset_readiness.exact_duplicate_groups(persona_rows):
        _add_rejection(
            report,
            reason_code="duplicate_candidate_blocked",
            detail="exact duplicate persona text: " + ", ".join(group["persona_ids"]),
        )
    for pair in dataset_readiness.near_duplicate_pairs(persona_rows):
        _add_rejection(
            report,
            reason_code="duplicate_candidate_blocked",
            detail=(
                "near duplicate persona text: "
                + ", ".join(pair["persona_ids"])
                + f" similarity={pair['similarity']:.3f}"
            ),
        )


def _add_gold_label_rejections(report: dict[str, Any], persona_rows: list[dict[str, Any]]) -> None:
    gold_report = dataset_readiness.gold_label_preview_report(persona_rows)
    for mismatch in gold_report.get("mismatches", []):
        _add_rejection(
            report,
            reason_code="gold_label_preview_blocked",
            persona_id=mismatch.get("persona_id"),
            detail=mismatch.get("detail"),
        )


def _add_review_rejections(
    report: dict[str, Any],
    persona_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    low_confidence_by_id: dict[str, list[str]],
) -> None:
    try:
        dataset_readiness.validate_review_manifest_for_personas(review_rows, persona_rows)
    except PersonaValidationError as exc:
        _add_rejection(
            report,
            reason_code="review_manifest_invalid",
            detail=str(exc),
        )
        return

    review_by_id = {row["persona_id"]: row for row in review_rows}
    reviewed_with_evidence = 0
    for persona_row in persona_rows:
        persona_id = persona_row["persona_id"]
        review_row = review_by_id.get(persona_id)
        if _review_has_required_evidence(review_row or {}):
            reviewed_with_evidence += 1
            continue
        reason = _review_failure_reason(review_row)
        _add_rejection(report, reason_code=reason, persona_id=persona_id)
        if low_confidence_by_id.get(persona_id):
            _add_rejection(
                report,
                reason_code="low_confidence_review_required",
                persona_id=persona_id,
                detail=", ".join(low_confidence_by_id[persona_id]),
            )

    minimum_required = (len(persona_rows) + 9) // 10
    if reviewed_with_evidence < minimum_required:
        _add_rejection(
            report,
            reason_code="minimum_review_coverage_not_met",
            detail=f"reviewed_with_evidence={reviewed_with_evidence}; minimum_required={minimum_required}",
        )
    report["approved_candidate_count"] = reviewed_with_evidence


def evaluate_promotion(
    *,
    candidate_path: str | Path,
    review_path: str | Path,
    out_path: str | Path | None = DEFAULT_FULL_DATASET_PATH,
    dry_run: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = _empty_report(candidate_path=candidate_path, review_path=review_path, out_path=out_path, dry_run=dry_run)
    candidate_path = Path(candidate_path)
    review_path = Path(review_path)
    report["candidate_input_hash"] = hash_file_bytes(candidate_path)
    report["review_manifest_hash"] = hash_file_bytes(review_path)

    persona_rows, low_confidence_by_id, invalid_rows = load_candidate_personas(candidate_path)
    report["candidate_row_count"] = len(load_jsonl(candidate_path))
    report["valid_candidate_count"] = len(persona_rows)
    for invalid in invalid_rows:
        _add_rejection(
            report,
            reason_code=invalid["reason_code"],
            persona_id=invalid["persona_id"],
            detail=invalid["detail"],
        )

    if len(persona_rows) != TARGET_PROMOTION_ROWS:
        _add_rejection(
            report,
            reason_code="promotion_requires_exactly_200_valid_rows",
            detail=f"valid_candidate_count={len(persona_rows)}; target_count={TARGET_PROMOTION_ROWS}",
        )

    _add_safety_rejections(report, persona_rows)
    _add_dedupe_rejections(report, persona_rows)
    _add_gold_label_rejections(report, persona_rows)
    review_rows = _load_review_rows(review_path, report)
    review_manifest_invalid = any(row["reason_code"] == "review_manifest_invalid" for row in report["rejections"])
    if review_rows or not review_manifest_invalid:
        _add_review_rejections(report, persona_rows, review_rows, low_confidence_by_id)

    _finalize_rejection_counts(report)
    if not report["rejections"] and len(persona_rows) == TARGET_PROMOTION_ROWS:
        report["status"] = "ready"
        report["write_permitted"] = not dry_run
    return report, persona_rows


def write_promotion_outputs(
    *,
    persona_rows: list[dict[str, Any]],
    report: dict[str, Any],
    out_path: str | Path,
    manifest_path: str | Path,
) -> None:
    if report.get("status") != "ready" or not report.get("write_permitted"):
        raise PersonaValidationError("promotion output write is blocked until all gates pass in a non-dry run")
    if len(persona_rows) != TARGET_PROMOTION_ROWS:
        raise PersonaValidationError("promotion output write requires exactly 200 persona rows")
    out = Path(out_path)
    manifest = Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in persona_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    report["promoted_output_hash"] = hash_file_bytes(out)
    report["write_performed"] = True
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _print_summary(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, sort_keys=True, indent=2))
        return
    print(f"promotion_status={report['status']}")
    print(f"dry_run={str(report['dry_run']).lower()}")
    print(f"target_count={report['target_count']}")
    print(f"candidate_row_count={report['candidate_row_count']}")
    print(f"valid_candidate_count={report['valid_candidate_count']}")
    print(f"approved_candidate_count={report['approved_candidate_count']}")
    print("rejection_counts=" + json.dumps(report["rejection_counts"], sort_keys=True))
    if report["status"] != "ready":
        print("write_blocked=true")
    elif report["dry_run"]:
        print("write_blocked=true")
    else:
        print("write_blocked=false")


def cmd_validate_candidates(args: argparse.Namespace) -> int:
    report, _ = evaluate_promotion(
        candidate_path=args.candidate_path,
        review_path=args.review_path,
        out_path=None,
        dry_run=True,
    )
    _print_summary(report, json_output=args.json)
    return 0 if report["status"] == "ready" else 1


def cmd_promote(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    manifest_path = Path(args.manifest_out) if args.manifest_out else DEFAULT_PROMOTION_MANIFEST_PATH
    report, persona_rows = evaluate_promotion(
        candidate_path=args.candidate_path,
        review_path=args.review_path,
        out_path=out_path,
        dry_run=args.dry_run,
    )
    if report["status"] == "ready" and not args.dry_run:
        write_promotion_outputs(
            persona_rows=persona_rows,
            report=report,
            out_path=out_path,
            manifest_path=manifest_path,
        )
    _print_summary(report, json_output=args.json)
    return 0 if report["status"] == "ready" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and dry-run full-dataset promotion gates")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-candidates", help="validate a proposed 200-row promotion set")
    validate_parser.add_argument("--candidate-path", required=True)
    validate_parser.add_argument("--review-path", required=True)
    validate_parser.add_argument("--json", action="store_true", help="print full promotion report JSON")
    validate_parser.set_defaults(func=cmd_validate_candidates)

    promote_parser = subparsers.add_parser("promote", help="promote rows only when all gates pass")
    promote_parser.add_argument("--candidate-path", required=True)
    promote_parser.add_argument("--review-path", required=True)
    promote_parser.add_argument("--out", default=str(DEFAULT_FULL_DATASET_PATH))
    promote_parser.add_argument("--manifest-out")
    promote_parser.add_argument("--dry-run", action="store_true")
    promote_parser.add_argument("--json", action="store_true", help="print full promotion report JSON")
    promote_parser.set_defaults(func=cmd_promote)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, json.JSONDecodeError, PersonaValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
