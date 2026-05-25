#!/usr/bin/env python3
"""Build Sprint 8 smoke evidence from completed vLLM smoke artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import aggregate
from persona_eval import (
    RUN_MODEL_COUNT,
    RUN_VARIANTS_PER_PERSONA,
    SMOKE_RUN_CALL_COUNT,
    SMOKE_RUN_PERSONA_COUNT,
    SMOKE_RUN_SEED_COUNT,
    PersonaValidationError,
    expected_call_count,
    hash_file_bytes,
    utc_now,
    validate_run_evidence_file,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "results" / "vllm_smoke_20" / "manifest.json"
DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "vllm_smoke_20" / "results.jsonl"
DEFAULT_AGGREGATE_REPORT_PATH = REPO_ROOT / "reports" / "vllm_smoke_20" / "aggregate_report.json"
DEFAULT_OUT_PATH = REPO_ROOT / "reports" / "vllm_smoke_20" / "smoke_evidence.json"


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise PersonaValidationError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PersonaValidationError(f"{label} must be a JSON object")
    return payload


def validate_smoke_shape(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        raise PersonaValidationError("smoke results must contain at least one result row")

    variants_by_persona: dict[str, set[str]] = defaultdict(set)
    seeds_by_persona: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(rows, start=1):
        persona_id = row.get("persona_id")
        variant_id = row.get("variant_id")
        seed = row.get("seed")
        if not isinstance(persona_id, str) or not persona_id.strip():
            raise PersonaValidationError(f"result row {index}: persona_id is required")
        if not isinstance(variant_id, str) or not variant_id.strip():
            raise PersonaValidationError(f"result row {index}: variant_id is required")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PersonaValidationError(f"result row {index}: seed must be an integer")
        if "base" not in row or "tuned" not in row:
            raise PersonaValidationError(f"result row {index}: base and tuned outputs are required")
        variants_by_persona[persona_id].add(variant_id)
        seeds_by_persona[persona_id].add(seed)

    persona_count = len(variants_by_persona)
    variant_counts = {len(values) for values in variants_by_persona.values()}
    seed_counts = {len(values) for values in seeds_by_persona.values()}
    if persona_count != SMOKE_RUN_PERSONA_COUNT:
        raise PersonaValidationError(
            f"smoke evidence requires persona_count={SMOKE_RUN_PERSONA_COUNT}; got {persona_count}"
        )
    if variant_counts != {RUN_VARIANTS_PER_PERSONA}:
        raise PersonaValidationError(
            f"smoke evidence requires variants_per_persona={RUN_VARIANTS_PER_PERSONA}; got {sorted(variant_counts)}"
        )
    if seed_counts != {SMOKE_RUN_SEED_COUNT}:
        raise PersonaValidationError(
            f"smoke evidence requires seed_count={SMOKE_RUN_SEED_COUNT}; got {sorted(seed_counts)}"
        )

    planned_calls = expected_call_count(
        persona_count,
        RUN_VARIANTS_PER_PERSONA,
        RUN_MODEL_COUNT,
        SMOKE_RUN_SEED_COUNT,
    )
    if planned_calls != SMOKE_RUN_CALL_COUNT:
        raise PersonaValidationError(
            f"smoke evidence planned calls must equal {SMOKE_RUN_CALL_COUNT}; got {planned_calls}"
        )

    expected_result_rows = persona_count * RUN_VARIANTS_PER_PERSONA * SMOKE_RUN_SEED_COUNT
    if len(rows) != expected_result_rows:
        raise PersonaValidationError(
            f"smoke results must contain {expected_result_rows} matched base/tuned rows; got {len(rows)}"
        )

    return {
        "persona_count": persona_count,
        "variants_per_persona": RUN_VARIANTS_PER_PERSONA,
        "model_count": RUN_MODEL_COUNT,
        "seed_count": SMOKE_RUN_SEED_COUNT,
        "planned_generation_calls": planned_calls,
        "matched_result_rows": len(rows),
    }


def validate_aggregate_binding(
    *,
    aggregate_report: dict[str, Any],
    manifest_path: Path,
    results_path: Path,
    shape: dict[str, int],
) -> None:
    aggregate.validate_aggregate_report(aggregate_report)
    expected_manifest_hash = hash_file_bytes(manifest_path)
    expected_results_hash = hash_file_bytes(results_path)
    if aggregate_report.get("source_manifest_hash") != expected_manifest_hash:
        raise PersonaValidationError("aggregate report source_manifest_hash does not match manifest.json")
    if aggregate_report.get("source_results_hash") != expected_results_hash:
        raise PersonaValidationError("aggregate report source_results_hash does not match results.jsonl")

    counts = aggregate_report.get("counts")
    if not isinstance(counts, dict):
        raise PersonaValidationError("aggregate report counts object is required for smoke evidence")
    if counts.get("persona_count") != shape["persona_count"]:
        raise PersonaValidationError("aggregate report persona_count does not match smoke results")
    if counts.get("row_count") != shape["matched_result_rows"]:
        raise PersonaValidationError("aggregate report row_count does not match smoke results")


def build_smoke_evidence(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    aggregate_report_path: str | Path = DEFAULT_AGGREGATE_REPORT_PATH,
    out_path: str | Path = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    manifest_path = resolve_path(manifest_path)
    results_path = resolve_path(results_path)
    aggregate_report_path = resolve_path(aggregate_report_path)
    out_path = resolve_path(out_path)

    manifest = aggregate.load_manifest(manifest_path)
    rows = aggregate.load_result_rows(results_path)
    aggregate.assert_manifest_matches_rows(manifest, rows)
    aggregate_report = load_json_object(aggregate_report_path, "aggregate report")
    shape = validate_smoke_shape(rows)
    validate_aggregate_binding(
        aggregate_report=aggregate_report,
        manifest_path=manifest_path,
        results_path=results_path,
        shape=shape,
    )

    evidence = {
        "evidence_type": "smoke_run_evidence",
        "stage": "smoke",
        "status": "pass",
        "created_at": utc_now(),
        "run_id": manifest["run_id"],
        **shape,
        "manifest_path": display_path(manifest_path),
        "results_path": display_path(results_path),
        "aggregate_report_path": display_path(aggregate_report_path),
        "manifest_hash": hash_file_bytes(manifest_path),
        "results_hash": hash_file_bytes(results_path),
        "aggregate_report_hash": hash_file_bytes(aggregate_report_path),
        "adapter": manifest.get("adapter"),
        "serving_stack": manifest.get("serving_stack"),
        "serving_stack_version": manifest.get("serving_stack_version"),
        "model_base": manifest.get("model_base"),
        "model_tuned": manifest.get("model_tuned"),
        "model_endpoints": manifest.get("model_endpoints"),
        "aggregate_report_status": "valid",
        "validation_notes": [
            "Validated manifest schema, result-row schema, aggregate report schema, and smoke run shape.",
            "Evidence proves artifact completeness only; it does not rerun vLLM or judge model quality.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(out_path)
    validate_run_evidence_file(
        out_path,
        expected_stage="smoke",
        expected_persona_count=SMOKE_RUN_PERSONA_COUNT,
        expected_seed_count=SMOKE_RUN_SEED_COUNT,
        expected_call_count_value=SMOKE_RUN_CALL_COUNT,
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Sprint 8 vLLM smoke evidence")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Smoke run manifest JSON")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH), help="Smoke run result rows JSONL")
    parser.add_argument(
        "--aggregate-report",
        default=str(DEFAULT_AGGREGATE_REPORT_PATH),
        help="Aggregate report JSON for the smoke run",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Smoke evidence JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        evidence = build_smoke_evidence(
            manifest_path=args.manifest,
            results_path=args.results,
            aggregate_report_path=args.aggregate_report,
            out_path=args.out,
        )
    except (OSError, json.JSONDecodeError, PersonaValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"smoke_evidence_path={args.out}")
    print(f"planned_generation_calls={evidence['planned_generation_calls']}")
    print("smoke_evidence_status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
