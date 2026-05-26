#!/usr/bin/env python3
"""Production/open model matrix validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
import datetime as dt
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover - covered only in stripped environments.
    jsonschema = None


REPO_ROOT = Path(__file__).resolve().parent
MODEL_MATRIX_SCHEMA_PATH = REPO_ROOT / "schemas" / "model_matrix.schema.json"

MODEL_ROLES = {
    "base",
    "instruct",
    "standalone_instruct",
    "judge",
    "embedding",
    "nli",
}
TOKEN_KL_APPLICABILITY_VALUES = {
    "canonical_possible",
    "diagnostic_only",
    "not_applicable",
}
PLACEHOLDER_RUNTIME_VALUES = {
    "",
    "not_available",
    "todo",
    "tbd",
    "placeholder",
    "unknown",
}
MODEL_RUNTIME_PLACEHOLDER_FIELDS = (
    "provider_or_endpoint",
    "required_revision_or_hash",
)
LICENSE_REVIEW_REQUIRED_FIELDS = (
    "license_reviewed_by",
    "license_reviewed_at",
    "redistribution_or_usage_notes",
)
LICENSE_REVIEW_SOURCE_FIELDS = (
    "license_terms_url",
    "license_source_url",
)
SAME_FAMILY_ALIGNMENT_REQUIREMENTS = (
    "same_tokenizer_family_required",
    "same_vocabulary_required",
    "prompt_rendering_hash_required",
    "chat_template_policy_match_required",
    "fixed_continuation_scoring_required",
    "comparable_next_token_logprobs_required",
)


class ModelMatrixValidationError(ValueError):
    """Raised when a production/open model matrix violates guardrails."""


def load_model_matrix(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ModelMatrixValidationError("model matrix must be a JSON object")
    return payload


def _schema_payload() -> dict[str, Any]:
    with MODEL_MATRIX_SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ModelMatrixValidationError("model matrix schema must be a JSON object")
    return schema


def validate_model_matrix_schema(matrix: dict[str, Any]) -> None:
    if jsonschema is None:
        raise ModelMatrixValidationError("jsonschema is required to validate model matrix configs")
    try:
        jsonschema.Draft202012Validator(_schema_payload()).validate(matrix)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path)
        prefix = f"model matrix schema violation at {path}: " if path else "model matrix schema violation: "
        raise ModelMatrixValidationError(prefix + exc.message) from exc


def _is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    lowered = stripped.lower()
    return (
        lowered in PLACEHOLDER_RUNTIME_VALUES
        or (stripped.startswith("<") and stripped.endswith(">"))
        or "fill-from" in lowered
        or "placeholder" in lowered
    )


def _is_review_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or _is_placeholder(value):
        return False
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt.datetime.fromisoformat(normalized)
        return True
    except ValueError:
        pass
    try:
        dt.date.fromisoformat(value.strip())
        return True
    except ValueError:
        return False


def _is_http_url(value: Any) -> bool:
    if not isinstance(value, str) or _is_placeholder(value):
        return False
    stripped = value.strip()
    return stripped.startswith("https://") or stripped.startswith("http://")


def _model_entries(matrix: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for index, pair in enumerate(matrix.get("drift_pairs", [])):
        entries.append((f"drift_pairs[{index}].base_model", pair.get("base_model", {})))
        entries.append((f"drift_pairs[{index}].instruct_model", pair.get("instruct_model", {})))
    for index, model in enumerate(matrix.get("standalone_instruct_models", [])):
        entries.append((f"standalone_instruct_models[{index}]", model))
    for index, comparison in enumerate(matrix.get("cross_family_comparisons", [])):
        entries.append((f"cross_family_comparisons[{index}].left_model", comparison.get("left_model", {})))
        entries.append((f"cross_family_comparisons[{index}].right_model", comparison.get("right_model", {})))
    return entries


def _validate_model_entry(path: str, model: dict[str, Any]) -> None:
    role = model.get("role")
    if role not in MODEL_ROLES:
        raise ModelMatrixValidationError(f"{path}.role must be one of: {', '.join(sorted(MODEL_ROLES))}")
    applicability = model.get("token_kl_applicability")
    if applicability not in TOKEN_KL_APPLICABILITY_VALUES:
        allowed = ", ".join(sorted(TOKEN_KL_APPLICABILITY_VALUES))
        raise ModelMatrixValidationError(f"{path}.token_kl_applicability must be one of: {allowed}")
    if role in {"judge", "embedding", "nli"} and applicability == "canonical_possible":
        raise ModelMatrixValidationError(f"{path} is an evaluator model and cannot declare canonical Token-KL")
    if role == "standalone_instruct" and not model.get("paired_baseline_model_id"):
        if applicability != "not_applicable":
            raise ModelMatrixValidationError(
                f"{path} standalone instruct model must mark Token-KL not_applicable without an explicit paired baseline"
            )


def _validate_canonical_possible_pair(path: str, pair: dict[str, Any]) -> None:
    if pair.get("comparison_type") != "same_family_base_instruct":
        raise ModelMatrixValidationError(f"{path} canonical_possible is limited to same_family_base_instruct pairs")
    base = pair.get("base_model", {})
    instruct = pair.get("instruct_model", {})
    if base.get("role") != "base" or instruct.get("role") != "instruct":
        raise ModelMatrixValidationError(f"{path} canonical_possible requires base and instruct roles")
    if base.get("expected_tokenizer_family") != instruct.get("expected_tokenizer_family"):
        raise ModelMatrixValidationError(
            f"{path} canonical_possible requires matching expected_tokenizer_family"
        )
    if base.get("expected_chat_template_policy") != instruct.get("expected_chat_template_policy"):
        raise ModelMatrixValidationError(
            f"{path} canonical_possible requires a shared expected_chat_template_policy"
        )

    contract = pair.get("alignment_contract")
    if not isinstance(contract, dict):
        raise ModelMatrixValidationError(f"{path}.alignment_contract is required for canonical_possible")
    missing = [field for field in SAME_FAMILY_ALIGNMENT_REQUIREMENTS if contract.get(field) is not True]
    if missing:
        raise ModelMatrixValidationError(
            f"{path}.alignment_contract must require: {', '.join(missing)}"
        )
    if contract.get("canonical_token_kl_status_before_proof") != "not_applicable":
        raise ModelMatrixValidationError(
            f"{path}.alignment_contract must keep canonical Token-KL not_applicable before proof"
        )


def _validate_drift_pairs(matrix: dict[str, Any]) -> None:
    for index, pair in enumerate(matrix.get("drift_pairs", [])):
        path = f"drift_pairs[{index}]"
        applicability = pair.get("token_kl_applicability")
        if applicability not in TOKEN_KL_APPLICABILITY_VALUES:
            allowed = ", ".join(sorted(TOKEN_KL_APPLICABILITY_VALUES))
            raise ModelMatrixValidationError(f"{path}.token_kl_applicability must be one of: {allowed}")
        if applicability == "canonical_possible":
            _validate_canonical_possible_pair(path, pair)


def _validate_cross_family_comparisons(matrix: dict[str, Any]) -> None:
    for index, comparison in enumerate(matrix.get("cross_family_comparisons", [])):
        path = f"cross_family_comparisons[{index}]"
        if comparison.get("comparison_type") != "cross_family":
            raise ModelMatrixValidationError(f"{path}.comparison_type must be cross_family")
        if comparison.get("token_kl_applicability") == "canonical_possible":
            raise ModelMatrixValidationError(f"{path} cross-family comparison cannot declare canonical Token-KL possible")


def _validate_metric_applicability(matrix: dict[str, Any]) -> None:
    metric_applicability = matrix.get("metric_applicability", {})
    for category in ("standalone_instruct_models", "cross_family_comparisons"):
        policy = metric_applicability.get(category, {}).get("token_kl")
        if isinstance(policy, str) and "canonical" in policy.lower():
            raise ModelMatrixValidationError(
                f"metric_applicability.{category}.token_kl must not claim canonical Token-KL"
            )


def _runtime_placeholder_paths(matrix: dict[str, Any]) -> list[str]:
    placeholders: list[str] = []
    for path, model in _model_entries(matrix):
        for field in MODEL_RUNTIME_PLACEHOLDER_FIELDS:
            if _is_placeholder(model.get(field)):
                placeholders.append(f"{path}.{field}")
    return sorted(placeholders)


def _license_review_blocking_paths(matrix: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for path, model in _model_entries(matrix):
        if model.get("license_review_required") is not True:
            continue
        status = model.get("license_review_status")
        if status != "approved":
            blockers.append(f"{path}.license_review_status")
        for field in LICENSE_REVIEW_REQUIRED_FIELDS:
            value = model.get(field)
            if field == "license_reviewed_at":
                if not _is_review_timestamp(value):
                    blockers.append(f"{path}.{field}")
                continue
            if not isinstance(value, str) or _is_placeholder(value):
                blockers.append(f"{path}.{field}")
        if not any(_is_http_url(model.get(field)) for field in LICENSE_REVIEW_SOURCE_FIELDS):
            blockers.append(f"{path}.license_terms_url_or_license_source_url")
    return sorted(blockers)


def resolve_model_matrix_entry(matrix: dict[str, Any], entry_id: str) -> dict[str, Any]:
    """Resolve a matrix entry into the runtime policy stored on manifests/results."""
    validate_model_matrix(matrix)
    for pair in matrix.get("drift_pairs", []):
        if pair.get("pair_id") == entry_id:
            return {
                "entry_id": entry_id,
                "entry_type": "drift_pair",
                "comparison_type": pair.get("comparison_type"),
                "token_kl_applicability": pair.get("token_kl_applicability"),
                "token_kl_reason_code": pair.get("token_kl_reason_code"),
                "model_ids": {
                    "base": pair.get("base_model", {}).get("model_id"),
                    "tuned": pair.get("instruct_model", {}).get("model_id"),
                },
                "metric_applicability": matrix.get("metric_applicability", {}).get("drift_pairs", {}),
            }
    for model in matrix.get("standalone_instruct_models", []):
        model_id = model.get("model_id")
        entry_key = model.get("entry_id") or model_id
        if entry_key == entry_id:
            return {
                "entry_id": entry_id,
                "entry_type": "standalone_instruct_model",
                "comparison_type": "standalone_instruct",
                "token_kl_applicability": model.get("token_kl_applicability"),
                "token_kl_reason_code": model.get("token_kl_reason_code"),
                "model_ids": {"standalone": model_id},
                "metric_applicability": matrix.get("metric_applicability", {}).get("standalone_instruct_models", {}),
            }
    for comparison in matrix.get("cross_family_comparisons", []):
        if comparison.get("comparison_id") == entry_id:
            return {
                "entry_id": entry_id,
                "entry_type": "cross_family_comparison",
                "comparison_type": comparison.get("comparison_type"),
                "token_kl_applicability": comparison.get("token_kl_applicability"),
                "token_kl_reason_code": comparison.get("token_kl_reason_code"),
                "model_ids": {
                    "left": comparison.get("left_model", {}).get("model_id"),
                    "right": comparison.get("right_model", {}).get("model_id"),
                },
                "metric_applicability": matrix.get("metric_applicability", {}).get("cross_family_comparisons", {}),
            }
    raise ModelMatrixValidationError(f"model matrix entry not found: {entry_id}")


def validate_model_matrix(
    matrix: dict[str, Any],
    *,
    require_real_run_ready: bool = False,
) -> dict[str, Any]:
    """Validate model-matrix structure and return readiness metadata.

    Template validation allows placeholder revisions/endpoints only when the
    matrix is explicitly marked as a template. Real-run readiness requires every
    runtime placeholder to be replaced before execution.
    """
    validate_model_matrix_schema(matrix)
    for path, model in _model_entries(matrix):
        _validate_model_entry(path, model)
    _validate_drift_pairs(matrix)
    _validate_cross_family_comparisons(matrix)
    _validate_metric_applicability(matrix)

    placeholder_paths = _runtime_placeholder_paths(matrix)
    license_review_blockers = _license_review_blocking_paths(matrix)
    declared_ready = matrix.get("template_status") == "real_run_ready" and matrix.get("real_run_ready") is True
    real_run_ready = declared_ready and len(placeholder_paths) == 0 and len(license_review_blockers) == 0
    template_status = matrix.get("template_status")
    if placeholder_paths and template_status == "real_run_ready":
        raise ModelMatrixValidationError(
            "model matrix template_status=real_run_ready but runtime placeholders remain"
        )
    if placeholder_paths and matrix.get("real_run_ready") is True:
        raise ModelMatrixValidationError(
            "model matrix real_run_ready=true but runtime placeholders remain"
        )
    if license_review_blockers and template_status == "real_run_ready":
        raise ModelMatrixValidationError(
            "model matrix template_status=real_run_ready but license review evidence is incomplete: "
            + ", ".join(license_review_blockers)
        )
    if license_review_blockers and matrix.get("real_run_ready") is True:
        raise ModelMatrixValidationError(
            "model matrix real_run_ready=true but license review evidence is incomplete: "
            + ", ".join(license_review_blockers)
        )
    if require_real_run_ready and template_status != "real_run_ready":
        raise ModelMatrixValidationError("model matrix is not real-run ready; template_status must be real_run_ready")
    if require_real_run_ready and matrix.get("real_run_ready") is not True:
        raise ModelMatrixValidationError("model matrix is not real-run ready; real_run_ready must be true")
    if require_real_run_ready and placeholder_paths:
        raise ModelMatrixValidationError(
            "model matrix is not real-run ready; replace placeholders: "
            + ", ".join(placeholder_paths)
        )
    if require_real_run_ready and license_review_blockers:
        raise ModelMatrixValidationError(
            "model matrix is not real-run ready; complete license review evidence: "
            + ", ".join(license_review_blockers)
        )

    return {
        "status": "real_run_ready" if real_run_ready else "template_valid",
        "real_run_ready": real_run_ready,
        "placeholder_paths": placeholder_paths,
        "placeholder_count": len(placeholder_paths),
        "license_review_blocker_paths": license_review_blockers,
        "license_review_blocker_count": len(license_review_blockers),
        "drift_pair_count": len(matrix.get("drift_pairs", [])),
        "standalone_instruct_model_count": len(matrix.get("standalone_instruct_models", [])),
        "cross_family_comparison_count": len(matrix.get("cross_family_comparisons", [])),
    }
