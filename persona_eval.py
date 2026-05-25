#!/usr/bin/env python3
"""Sprint 0 validation and planning CLI for persona drift fixtures."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover - covered only in stripped environments.
    jsonschema = None


REPO_ROOT = Path(__file__).resolve().parent
PERSONA_SCHEMA_PATH = REPO_ROOT / "schemas" / "persona_item.schema.json"

REQUIRED_VARIANT_TYPES = {
    "canonical",
    "paraphrase",
    "negation_preserving",
    "distractor",
    "instruction_prefix",
    "temperature_robust",
}
REQUIRED_SOURCE_FIELDS = {
    "dataset",
    "source_url",
    "license",
    "license_url",
    "split",
    "source_persona_id",
    "retrieved_at",
    "revision_or_hash",
    "modification_notes",
    "redistribution_notes",
}
BEHAVIOR_LABEL_FIELDS = {
    "stance",
    "primary_action",
    "secondary_modifiers",
}


class PersonaValidationError(ValueError):
    """Raised when a persona fixture violates the Sprint 0 contract."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return decoded objects."""
    jsonl_path = Path(path)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PersonaValidationError(
                    f"{jsonl_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise PersonaValidationError(f"{jsonl_path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def load_schema(path: str | Path = PERSONA_SCHEMA_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_with_schema(row: dict[str, Any], schema: dict[str, Any]) -> None:
    if jsonschema is None:
        raise PersonaValidationError("jsonschema is required for Sprint 0 schema validation")
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.path) or "<root>"
        raise PersonaValidationError(f"schema violation at {location}: {error.message}")


def validate_source_metadata(row: dict[str, Any]) -> None:
    source = row.get("source")
    if not isinstance(source, dict):
        raise PersonaValidationError("source must be an object")

    missing = sorted(REQUIRED_SOURCE_FIELDS - set(source))
    if missing:
        raise PersonaValidationError(f"missing source fields: {', '.join(missing)}")

    for field in sorted(REQUIRED_SOURCE_FIELDS - {"revision_or_hash"}):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PersonaValidationError(f"source.{field} must be a non-empty string")

    try:
        dt.date.fromisoformat(source["retrieved_at"])
    except ValueError as exc:
        raise PersonaValidationError("source.retrieved_at must be an ISO date") from exc

    revision = source.get("revision_or_hash")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise PersonaValidationError("source.revision_or_hash must be null or a non-empty string")

    if revision is None and "no stable" not in source["modification_notes"].lower():
        raise PersonaValidationError(
            "source.modification_notes must explain null revision_or_hash provenance"
        )


def _validate_behavior_object(value: Any, label_path: str) -> None:
    if not isinstance(value, dict):
        raise PersonaValidationError(f"{label_path} must be an object")

    keys = set(value)
    missing = sorted(BEHAVIOR_LABEL_FIELDS - keys)
    extras = sorted(keys - BEHAVIOR_LABEL_FIELDS)
    if missing:
        raise PersonaValidationError(f"{label_path} missing fields: {', '.join(missing)}")
    if extras:
        raise PersonaValidationError(f"{label_path} has unsupported fields: {', '.join(extras)}")

    for field in ("stance", "primary_action"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise PersonaValidationError(f"{label_path}.{field} must be a non-empty string")

    modifiers = value["secondary_modifiers"]
    if not isinstance(modifiers, list):
        raise PersonaValidationError(f"{label_path}.secondary_modifiers must be an array")
    for index, modifier in enumerate(modifiers):
        if not isinstance(modifier, str) or not modifier.strip():
            raise PersonaValidationError(
                f"{label_path}.secondary_modifiers[{index}] must be a non-empty string"
            )


def validate_behavior_labels(row: dict[str, Any]) -> None:
    _validate_behavior_object(row.get("expected_behavior"), "expected_behavior")
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        raise PersonaValidationError("annotation must be an object")
    _validate_behavior_object(annotation.get("gold_labels"), "annotation.gold_labels")


def validate_variants(row: dict[str, Any]) -> None:
    variants = row.get("variants")
    if not isinstance(variants, list):
        raise PersonaValidationError("variants must be an array")
    if len(variants) != len(REQUIRED_VARIANT_TYPES):
        raise PersonaValidationError("variants must contain exactly six executable variants")

    seen: list[str] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise PersonaValidationError(f"variants[{index}] must be an object")
        variant_type = variant.get("type")
        if variant_type not in REQUIRED_VARIANT_TYPES:
            raise PersonaValidationError(f"variants[{index}].type is unsupported: {variant_type!r}")
        seen.append(variant_type)

    seen_set = set(seen)
    missing = sorted(REQUIRED_VARIANT_TYPES - seen_set)
    duplicates = sorted({variant_type for variant_type in seen if seen.count(variant_type) > 1})
    if missing:
        raise PersonaValidationError(f"missing variant types: {', '.join(missing)}")
    if duplicates:
        raise PersonaValidationError(f"duplicate variant types: {', '.join(duplicates)}")


def validate_persona_row(row: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    schema = load_schema() if schema is None else schema
    validate_with_schema(row, schema)
    validate_source_metadata(row)
    validate_behavior_labels(row)
    validate_variants(row)


def validate_personas(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    schema = load_schema()
    for index, row in enumerate(rows, start=1):
        try:
            validate_persona_row(row, schema)
        except PersonaValidationError as exc:
            raise PersonaValidationError(f"row {index}: {exc}") from exc
    return rows


def _count(value: int | list[Any] | tuple[Any, ...], name: str) -> int:
    if isinstance(value, int):
        count = value
    else:
        count = len(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def expected_call_count(
    personas: int | list[Any] | tuple[Any, ...],
    variants: int | list[Any] | tuple[Any, ...],
    model_count: int,
    seed_count: int,
) -> int:
    """Return personas * variants * models * seeds after positive-count checks."""
    return (
        _count(personas, "personas")
        * _count(variants, "variants")
        * _count(model_count, "model_count")
        * _count(seed_count, "seed_count")
    )


def _seed_count_from_args(args: argparse.Namespace) -> int:
    if args.seed_count is not None:
        return args.seed_count
    if args.seeds:
        return len(args.seeds)
    raise PersonaValidationError("plan requires --seed-count or --seeds")


def _model_count_from_args(args: argparse.Namespace) -> int:
    if args.model_count is not None:
        return args.model_count
    models = [model for model in (args.model_base, args.model_tuned) if model]
    if models:
        return len(models)
    raise PersonaValidationError("plan requires --model-count or --model-base/--model-tuned")


def cmd_validate(args: argparse.Namespace) -> int:
    rows = validate_personas(args.persona_path)
    print(f"valid_persona_rows={len(rows)}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    if args.persona_path:
        rows = validate_personas(args.persona_path)
        persona_count = len(rows)
        variant_counts = {len(row["variants"]) for row in rows}
        if len(variant_counts) != 1:
            raise PersonaValidationError("all persona rows must have the same variant count")
        variants_per_persona = variant_counts.pop()
    else:
        if args.persona_count is None or args.variants_per_persona is None:
            raise PersonaValidationError(
                "plan requires --persona-path or both --persona-count and --variants-per-persona"
            )
        persona_count = args.persona_count
        variants_per_persona = args.variants_per_persona

    total = expected_call_count(
        persona_count,
        variants_per_persona,
        _model_count_from_args(args),
        _seed_count_from_args(args),
    )
    print(f"planned_generation_calls={total}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persona drift Sprint 0 harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate persona JSONL fixtures")
    validate_parser.add_argument("--persona-path", required=True)
    validate_parser.set_defaults(func=cmd_validate)

    plan_parser = subparsers.add_parser("plan", help="print planned generation-call count")
    plan_parser.add_argument("--persona-path")
    plan_parser.add_argument("--persona-count", type=int)
    plan_parser.add_argument("--variants-per-persona", type=int)
    plan_parser.add_argument("--model-count", type=int)
    plan_parser.add_argument("--seed-count", type=int)
    plan_parser.add_argument("--model-base")
    plan_parser.add_argument("--model-tuned")
    plan_parser.add_argument("--seeds", nargs="+")
    plan_parser.set_defaults(func=cmd_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, PersonaValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
