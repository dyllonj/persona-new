#!/usr/bin/env python3
"""Build Sprint 8 smoke evidence from completed vLLM smoke artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

import aggregate
from persona_eval import (
    RUN_MODEL_COUNT,
    RUN_VARIANTS_PER_PERSONA,
    SMOKE_RUN_CALL_COUNT,
    SMOKE_RUN_PERSONA_COUNT,
    SMOKE_RUN_SEED_COUNT,
    PersonaValidationError,
    REAL_HTTP_ADAPTERS,
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
REQUIRED_RUNTIME_METADATA_FIELDS = (
    "model_base",
    "model_tuned",
    "model_base_revision_or_hash",
    "model_tuned_revision_or_hash",
    "tokenizer_name",
    "tokenizer_hash",
    "chat_template_hash",
    "serving_stack_version",
    "gpu_cuda_driver",
)
LOCAL_ENDPOINT_HOSTS = {"localhost", "127.0.0.1", "::1"}
PLACEHOLDER_MARKERS = ("<", ">", "fill-from", "placeholder")


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


def is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    lowered = stripped.lower()
    return (
        not stripped
        or lowered in {"not_available", "todo", "tbd", "placeholder", "unknown"}
        or any(marker in lowered for marker in PLACEHOLDER_MARKERS)
    )


def require_non_placeholder(payload: dict[str, Any], field: str, label: str) -> None:
    if is_placeholder(payload.get(field)):
        raise PersonaValidationError(f"{label}.{field} must be filled with non-placeholder runtime metadata")


def require_current_hash(*, payload: dict[str, Any], path_field: str, hash_field: str, label: str) -> Path:
    raw_path = payload.get(path_field)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PersonaValidationError(f"{label}.{path_field} is required")
    resolved = resolve_path(raw_path)
    if not resolved.exists():
        raise PersonaValidationError(f"{label}.{path_field} does not exist: {resolved}")
    expected_hash = payload.get(hash_field)
    if not isinstance(expected_hash, str) or not expected_hash.strip():
        raise PersonaValidationError(f"{label}.{hash_field} is required")
    actual_hash = hash_file_bytes(resolved)
    if expected_hash != actual_hash:
        raise PersonaValidationError(
            f"{label}.{hash_field} does not match {path_field}: expected {expected_hash}, got {actual_hash}"
        )
    return resolved


def is_local_endpoint(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    return host in LOCAL_ENDPOINT_HOSTS


def validate_real_smoke_manifest(manifest: dict[str, Any]) -> None:
    adapter = manifest.get("adapter")
    if adapter not in REAL_HTTP_ADAPTERS:
        raise PersonaValidationError("smoke evidence requires a real vllm/openai-compatible adapter")
    if manifest.get("serving_stack") != "vllm":
        raise PersonaValidationError("smoke evidence requires serving_stack=vllm")
    if manifest.get("score_mode") != "disabled":
        raise PersonaValidationError("Sprint 8 smoke evidence requires score_mode=disabled")
    if manifest.get("raw_request_response_logging_status") != "enabled":
        raise PersonaValidationError("smoke evidence requires raw_request_response_logging_status=enabled")
    logging = manifest.get("raw_request_response_logging")
    if not isinstance(logging, dict) or not logging.get("raw_request_logged") or not logging.get("raw_response_logged"):
        raise PersonaValidationError("smoke evidence requires raw request and raw response logging")
    for field in REQUIRED_RUNTIME_METADATA_FIELDS:
        require_non_placeholder(manifest, field, "smoke manifest")

    expected_persona_path = (REPO_ROOT / "data" / "personas.full.jsonl").resolve()
    actual_persona_path = Path(str(manifest.get("persona_path", ""))).resolve()
    if actual_persona_path != expected_persona_path:
        raise PersonaValidationError("smoke evidence requires persona_path=data/personas.full.jsonl")

    require_current_hash(
        payload=manifest,
        path_field="promotion_manifest_path",
        hash_field="promotion_manifest_hash",
        label="smoke manifest",
    )

    model_endpoints = manifest.get("model_endpoints")
    if not isinstance(model_endpoints, dict):
        raise PersonaValidationError("smoke manifest.model_endpoints is required")
    for side in ("base", "tuned"):
        endpoint = model_endpoints.get(side)
        if not isinstance(endpoint, dict):
            raise PersonaValidationError(f"smoke manifest.model_endpoints.{side} is required")
        provider = endpoint.get("provider_or_endpoint")
        if not is_local_endpoint(provider):
            raise PersonaValidationError(
                f"smoke manifest.model_endpoints.{side}.provider_or_endpoint must be a local endpoint"
            )
        for field in ("model_id", "model_revision_or_hash", "tokenizer_name", "tokenizer_hash", "chat_template_hash"):
            require_non_placeholder(endpoint, field, f"smoke manifest.model_endpoints.{side}")


def validate_result_runtime_metadata(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    model_endpoints = manifest.get("model_endpoints")
    if not isinstance(model_endpoints, dict):
        raise PersonaValidationError("smoke manifest.model_endpoints is required")
    expected_endpoints = {
        side: model_endpoints[side]["provider_or_endpoint"]
        for side in ("base", "tuned")
    }
    expected_adapter = manifest.get("adapter")
    for index, row in enumerate(rows, start=1):
        for side in ("base", "tuned"):
            raw_request = row.get(side, {}).get("raw_request")
            if not isinstance(raw_request, dict):
                raise PersonaValidationError(f"result row {index}: {side}.raw_request is required")
            if raw_request.get("adapter") != expected_adapter:
                raise PersonaValidationError(f"result row {index}: {side}.raw_request.adapter must match manifest")
            if raw_request.get("provider_or_endpoint") != expected_endpoints[side]:
                raise PersonaValidationError(
                    f"result row {index}: {side}.raw_request.provider_or_endpoint must match manifest endpoint"
                )
            endpoint = model_endpoints[side]
            for field in ("model_id", "model_revision_or_hash", "tokenizer_name", "tokenizer_hash", "chat_template_hash"):
                if raw_request.get(field) != endpoint.get(field):
                    raise PersonaValidationError(
                        f"result row {index}: {side}.raw_request.{field} must match manifest endpoint metadata"
                    )
            if not is_local_endpoint(raw_request.get("provider_or_endpoint")):
                raise PersonaValidationError(f"result row {index}: {side}.raw_request endpoint must be local")
            metric = row.get("metrics", {}).get("token_kl", {})
            if isinstance(metric, dict) and metric.get("status") == "ok":
                raise PersonaValidationError("Sprint 8 smoke evidence must not include canonical Token-KL rows")


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
    validate_real_smoke_manifest(manifest)
    validate_result_runtime_metadata(rows, manifest)
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
        "runtime_metadata_policy": manifest.get("runtime_metadata_policy"),
        "gpu_cuda_driver": manifest.get("gpu_cuda_driver"),
        "score_mode": manifest.get("score_mode"),
        "promotion_manifest_path": manifest.get("promotion_manifest_path"),
        "promotion_manifest_hash": manifest.get("promotion_manifest_hash"),
        "aggregate_report_status": "valid",
        "validation_notes": [
            "Validated manifest schema, result-row schema, aggregate report schema, smoke run shape, and real-runtime metadata.",
            "Evidence proves artifact completeness and runtime metadata binding only; it does not rerun vLLM or judge model quality.",
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
