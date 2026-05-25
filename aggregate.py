#!/usr/bin/env python3
"""Sprint 4 persona-level aggregation and readiness reporting."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

from dataset_readiness import READINESS_CHECKS, full_dataset_readiness_report
from persona_eval import (
    PERSONA_SCHEMA_PATH,
    PersonaValidationError,
    hash_file_bytes,
    load_jsonl,
    validate_schema_file,
    validate_result_row,
    validate_run_manifest,
)


REPO_ROOT = Path(__file__).resolve().parent
AGGREGATE_REPORT_SCHEMA_PATH = REPO_ROOT / "schemas" / "aggregate_report.schema.json"
AGGREGATION_VERSION = "sprint5"
REPORT_SCHEMA_VERSION = "aggregate_report_v2"
BC_F1_FIELDS = (
    "stance_exact",
    "primary_action_exact",
    "secondary_modifiers_f1",
    "combined_score",
)
OUTPUT_DELTA_FIELDS = (
    ("completion_tokens", ("usage", "completion_tokens")),
    ("total_tokens", ("usage", "total_tokens")),
    ("latency_s", ("latency_s",)),
)

def not_applicable(reason_code: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "not_applicable", "reason_code": reason_code}
    payload.update(extra)
    return payload


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise PersonaValidationError("manifest must be a JSON object")
    validate_run_manifest(manifest)
    return manifest


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for index, row in enumerate(rows, start=1):
        try:
            validate_result_row(row)
            validate_base_tuned_match(row)
        except PersonaValidationError as exc:
            raise PersonaValidationError(f"result row {index}: {exc}") from exc
    return rows


def validate_base_tuned_match(row: dict[str, Any]) -> None:
    """Confirm the paired base/tuned output in a result row shares audit axes."""
    base_request = row.get("base", {}).get("raw_request", {})
    tuned_request = row.get("tuned", {}).get("raw_request", {})
    if not isinstance(base_request, dict) or not isinstance(tuned_request, dict):
        raise PersonaValidationError("base/tuned raw_request must be objects")
    row_axes = ("run_id", "persona_id", "variant_id", "variant_type", "seed", "prompt_hash")
    for axis in row_axes:
        if base_request.get(axis) != row.get(axis):
            raise PersonaValidationError(f"base raw_request {axis} does not match result row")
        if tuned_request.get(axis) != row.get(axis):
            raise PersonaValidationError(f"tuned raw_request {axis} does not match result row")
    for axis in ("system_prompt_hash", "prompt_template_hash", "prompt_template_version"):
        if base_request.get(axis) != tuned_request.get(axis):
            raise PersonaValidationError(f"base/tuned raw_request {axis} mismatch")


def assert_manifest_matches_rows(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise PersonaValidationError("results JSONL must contain at least one row")
    run_id = manifest["run_id"]
    mismatched = sorted({row.get("run_id") for row in rows if row.get("run_id") != run_id})
    if mismatched:
        raise PersonaValidationError(
            "result rows contain run_id values not matching manifest: " + ", ".join(map(str, mismatched))
        )


def mean(values: list[float]) -> float:
    if not values:
        raise PersonaValidationError("mean requires at least one value")
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        raise PersonaValidationError("sample_sd requires at least two values")
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise PersonaValidationError("percentile requires at least one value")
    if p < 0.0 or p > 1.0:
        raise PersonaValidationError("percentile p must be in [0, 1]")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = p * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_ci(
    persona_values: list[float],
    *,
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    if len(persona_values) < 2:
        return not_applicable(
            "insufficient_personas_for_bootstrap",
            lower=None,
            upper=None,
            method="paired_bootstrap_over_persona_values",
            inference_unit="persona_id",
            inference_unit_count=len(persona_values),
        )
    rng = random.Random(seed)
    boot_means: list[float] = []
    n = len(persona_values)
    for _ in range(iterations):
        sample = [persona_values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(mean(sample))
    return {
        "status": "ok",
        "method": "paired_bootstrap_over_persona_values",
        "iterations": iterations,
        "seed": seed,
        "lower": percentile(boot_means, 0.025),
        "upper": percentile(boot_means, 0.975),
        "inference_unit": "persona_id",
        "inference_unit_count": len(persona_values),
    }


def effect_size_from_deltas(deltas: list[float]) -> dict[str, Any]:
    if len(deltas) < 2:
        return not_applicable("insufficient_personas_for_effect_size", value=None)
    sd = sample_sd(deltas)
    if sd == 0.0:
        return not_applicable("zero_delta_variance", value=None)
    return {
        "status": "ok",
        "method": "cohen_style_mean_delta_over_sample_sd",
        "value": mean(deltas) / sd,
    }


def paired_permutation_p_value(deltas: list[float]) -> dict[str, Any]:
    nonzero = [delta for delta in deltas if delta != 0.0]
    if len(nonzero) < 2:
        return not_applicable("insufficient_nonzero_persona_deltas", value=None)
    observed = abs(mean(nonzero))
    if len(nonzero) <= 20:
        extreme = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            total += 1
            permuted = [delta * sign for delta, sign in zip(nonzero, signs, strict=True)]
            if abs(mean(permuted)) >= observed - 1e-12:
                extreme += 1
        return {
            "status": "ok",
            "method": "exact_sign_flip_paired_permutation",
            "value": extreme / total,
            "nonzero_delta_count": len(nonzero),
        }
    rng = random.Random(0)
    iterations = 10000
    extreme = 0
    for _ in range(iterations):
        permuted = [delta * rng.choice((-1.0, 1.0)) for delta in nonzero]
        if abs(mean(permuted)) >= observed - 1e-12:
            extreme += 1
    return {
        "status": "ok",
        "method": "monte_carlo_sign_flip_paired_permutation",
        "value": extreme / iterations,
        "iterations": iterations,
        "seed": 0,
        "nonzero_delta_count": len(nonzero),
    }


def mcnemar_exact(base_passes: list[bool], tuned_passes: list[bool]) -> dict[str, Any]:
    if len(base_passes) != len(tuned_passes):
        raise PersonaValidationError("McNemar inputs must have the same length")
    if not base_passes:
        return not_applicable("no_paired_pass_fail_values", value=None)
    base_only = sum(1 for base, tuned in zip(base_passes, tuned_passes, strict=True) if base and not tuned)
    tuned_only = sum(1 for base, tuned in zip(base_passes, tuned_passes, strict=True) if tuned and not base)
    discordant = base_only + tuned_only
    if discordant == 0:
        return not_applicable(
            "no_discordant_pairs",
            value=None,
            base_only=base_only,
            tuned_only=tuned_only,
            discordant_count=discordant,
        )
    tail = min(base_only, tuned_only)
    probability = sum(math.comb(discordant, k) * (0.5**discordant) for k in range(tail + 1))
    return {
        "status": "ok",
        "method": "exact_mcnemar_binomial",
        "value": min(1.0, 2.0 * probability),
        "base_only": base_only,
        "tuned_only": tuned_only,
        "discordant_count": discordant,
    }


def numeric_summary(persona_values: dict[str, float]) -> dict[str, Any]:
    values = list(persona_values.values())
    if not values:
        return not_applicable(
            "no_available_persona_level_values",
            mean=None,
            ci95=not_applicable("no_available_persona_level_values", lower=None, upper=None),
            inference_unit="persona_id",
            inference_unit_count=0,
        )
    return {
        "status": "ok",
        "mean": mean(values),
        "ci95": paired_bootstrap_ci(values),
        "inference_unit": "persona_id",
        "inference_unit_count": len(values),
        "persona_values": dict(sorted(persona_values.items())),
    }


def paired_delta_statistics(pairs_by_persona: dict[str, list[tuple[float, float]]]) -> dict[str, Any]:
    baseline_by_persona: dict[str, float] = {}
    comparator_by_persona: dict[str, float] = {}
    delta_by_persona: dict[str, float] = {}
    for persona_id, pairs in pairs_by_persona.items():
        if not pairs:
            continue
        baseline_values = [pair[0] for pair in pairs]
        comparator_values = [pair[1] for pair in pairs]
        baseline = mean(baseline_values)
        comparator = mean(comparator_values)
        baseline_by_persona[persona_id] = baseline
        comparator_by_persona[persona_id] = comparator
        delta_by_persona[persona_id] = comparator - baseline

    if not delta_by_persona:
        return not_applicable(
            "no_comparable_baseline_fields",
            baseline_mean=None,
            comparator_mean=None,
            mean_delta=None,
            ci95=not_applicable("no_comparable_baseline_fields", lower=None, upper=None),
            effect_size=not_applicable("no_comparable_baseline_fields", value=None),
            p_value=not_applicable("no_comparable_baseline_fields", value=None),
        )

    deltas = list(delta_by_persona.values())
    return {
        "status": "ok",
        "baseline_mean": mean(list(baseline_by_persona.values())),
        "comparator_mean": mean(list(comparator_by_persona.values())),
        "mean_delta": mean(deltas),
        "absolute_delta_vs_baseline": mean(deltas),
        "ci95": paired_bootstrap_ci(deltas),
        "effect_size": effect_size_from_deltas(deltas),
        "p_value": paired_permutation_p_value(deltas),
        "inference_unit": "persona_id",
        "inference_unit_count": len(delta_by_persona),
        "persona_deltas": dict(sorted(delta_by_persona.items())),
    }


def group_rows_by_persona(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["persona_id"]].append(row)
    return dict(sorted(grouped.items()))


def aggregate_numeric_metric(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    field: str,
    status: str = "ok",
) -> dict[str, Any]:
    values_by_persona: dict[str, list[float]] = defaultdict(list)
    available_rows = 0
    for row in rows:
        metric = row["metrics"][metric_name]
        if metric.get("status") != status:
            continue
        value = metric.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values_by_persona[row["persona_id"]].append(float(value))
            available_rows += 1
    persona_values = {
        persona_id: mean(values)
        for persona_id, values in values_by_persona.items()
        if values
    }
    summary = numeric_summary(persona_values)
    summary["available_row_count"] = available_rows
    summary["available_persona_count"] = len(persona_values)
    return summary


def availability_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_metric: dict[str, Any] = {}
    for metric_name in ("behavioral_consistency_f1", "persona_adherence", "token_kl"):
        statuses = Counter(row["metrics"][metric_name].get("status") for row in rows)
        reason_codes = Counter(
            row["metrics"][metric_name].get("reason_code")
            for row in rows
            if row["metrics"][metric_name].get("reason_code") is not None
        )
        by_metric[metric_name] = {
            "status_counts": dict(sorted(statuses.items())),
            "reason_code_counts": dict(sorted(reason_codes.items())),
        }
    return by_metric


def summarize_behavioral_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_summaries = {
        field: aggregate_numeric_metric(
            rows,
            metric_name="behavioral_consistency_f1",
            field=field,
            status="ok",
        )
        for field in BC_F1_FIELDS
    }
    return {
        "status": "ok" if any(summary["status"] == "ok" for summary in field_summaries.values()) else "not_applicable",
        "fields": field_summaries,
        "note": "BC-F1 is aggregated over persona-level means after variants and seeds are averaged inside each persona.",
    }


def summarize_persona_adherence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mock_fields = ("mock_score", "semantic_similarity", "fact_contradiction_rate")
    mock_summaries = {
        field: aggregate_numeric_metric(
            rows,
            metric_name="persona_adherence",
            field=field,
            status="mock_only",
        )
        for field in mock_fields
    }
    any_mock = any(summary["status"] == "ok" for summary in mock_summaries.values())
    return {
        "real_persona_adherence": not_applicable(
            "real_pa_backends_and_calibration_not_pinned",
            mean=None,
            ci95=not_applicable("real_pa_backends_and_calibration_not_pinned", lower=None, upper=None),
        ),
        "mock_plumbing": {
            "status": "mock_only" if any_mock else "not_applicable",
            "reason_code": None if any_mock else "no_mock_only_pa_rows",
            "label": "mock_only_plumbing_not_real_persona_adherence",
            "fields": mock_summaries,
        },
        "warning": "Mock-only PA values are plumbing checks and are not reported as real Persona Adherence.",
    }


def summarize_token_kl(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row["metrics"]["token_kl"].get("status") for row in rows)
    reason_counts = Counter(
        row["metrics"]["token_kl"].get("reason_code")
        for row in rows
        if row["metrics"]["token_kl"].get("reason_code") is not None
    )
    values_by_persona: dict[str, list[float]] = defaultdict(list)
    ok_row_count = 0
    for row in rows:
        metric = row["metrics"]["token_kl"]
        if metric.get("status") != "ok":
            continue
        value = metric.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values_by_persona[row["persona_id"]].append(float(value))
            ok_row_count += 1
    persona_values = {
        persona_id: mean(values)
        for persona_id, values in values_by_persona.items()
        if values
    }
    return {
        "canonical_ok": numeric_summary(persona_values),
        "ok_row_count": ok_row_count,
        "status_counts": dict(sorted(status_counts.items())),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "diagnostic_only_count": status_counts.get("diagnostic_only", 0),
        "not_applicable_count": status_counts.get("not_applicable", 0),
        "note": "Only token_kl.status=ok rows are averaged. Unavailable and diagnostic-only rows are counted, never coerced to zero.",
    }


def variant_type_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant_type"]].append(row)
    breakdown: dict[str, Any] = {}
    for variant_type, variant_rows in sorted(grouped.items()):
        breakdown[variant_type] = {
            "row_count": len(variant_rows),
            "persona_count": len({row["persona_id"] for row in variant_rows}),
            "seed_count": len({row["seed"] for row in variant_rows}),
            "metric_availability": availability_counts(variant_rows),
        }
    return breakdown


def persona_shape_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = group_rows_by_persona(rows)
    return {
        "variant_count_by_persona": {
            persona_id: len({row["variant_id"] for row in persona_rows})
            for persona_id, persona_rows in grouped.items()
        },
        "seed_count_by_persona": {
            persona_id: len({row["seed"] for row in persona_rows})
            for persona_id, persona_rows in grouped.items()
        },
        "row_count_by_persona": {
            persona_id: len(persona_rows)
            for persona_id, persona_rows in grouped.items()
        },
    }


def summarize_flags(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = group_rows_by_persona(rows)
    flagged_personas = sorted(
        persona_id
        for persona_id, persona_rows in grouped.items()
        if any(row.get("flags") for row in persona_rows)
    )
    flag_counts = Counter(flag for row in rows for flag in row.get("flags", []))
    persona_count = len(grouped)
    return {
        "flagged_persona_count": len(flagged_personas),
        "flagged_persona_percentage": (len(flagged_personas) / persona_count * 100.0) if persona_count else 0.0,
        "flagged_personas": flagged_personas,
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def nested_numeric(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return float(current)
    return None


def summarize_output_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_field: dict[str, dict[str, list[tuple[float, float]]]] = {
        name: defaultdict(list)
        for name, _ in OUTPUT_DELTA_FIELDS
    }
    base_passes_by_persona: dict[str, list[bool]] = defaultdict(list)
    tuned_passes_by_persona: dict[str, list[bool]] = defaultdict(list)

    for row in rows:
        persona_id = row["persona_id"]
        for field_name, path in OUTPUT_DELTA_FIELDS:
            base_value = nested_numeric(row["base"], path)
            tuned_value = nested_numeric(row["tuned"], path)
            if base_value is not None and tuned_value is not None:
                by_field[field_name][persona_id].append((base_value, tuned_value))
        base_passes_by_persona[persona_id].append(not bool(row["base"].get("truncation_flag")))
        tuned_passes_by_persona[persona_id].append(not bool(row["tuned"].get("truncation_flag")))

    pass_pairs: list[tuple[bool, bool]] = []
    for persona_id in sorted(base_passes_by_persona):
        base_values = base_passes_by_persona[persona_id]
        tuned_values = tuned_passes_by_persona[persona_id]
        if not base_values or not tuned_values:
            continue
        pass_pairs.append((all(base_values), all(tuned_values)))

    return {
        "numeric_deltas": {
            field_name: paired_delta_statistics(dict(persona_pairs))
            for field_name, persona_pairs in by_field.items()
        },
        "truncation_pass_fail_mcnemar": mcnemar_exact(
            [pair[0] for pair in pass_pairs],
            [pair[1] for pair in pass_pairs],
        ),
        "note": "Base/tuned output fields are matched within each result row, averaged inside persona_id, then compared across personas.",
    }


def readiness_status(status: str, reason_code: str | None, evidence: str | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason_code,
        "evidence": evidence,
    }


def full_dataset_readiness() -> dict[str, Any]:
    return full_dataset_readiness_report()


def statistical_method_notes() -> dict[str, Any]:
    return {
        "inference_unit": "persona_id",
        "within_persona_aggregation": "variants and seeds are averaged inside each persona before cross-persona statistics",
        "confidence_interval": "95 percent paired bootstrap over persona-level values or deltas",
        "effect_size": "Cohen-style mean paired delta divided by sample SD of persona-level deltas when applicable",
        "p_value": "paired sign-flip permutation over persona-level deltas when applicable",
        "mcnemar": "exact binomial McNemar support is used for paired pass/fail discordance when applicable",
        "multiple_comparison_note": (
            "Primary summaries are persona-level overall metrics. Variant-type and secondary field cuts are "
            "exploratory and require multiple-comparison handling before inferential claims."
        ),
    }


def build_report(
    *,
    manifest_path: str | Path,
    results_path: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    rows = load_result_rows(results_path)
    assert_manifest_matches_rows(manifest, rows)
    grouped = group_rows_by_persona(rows)

    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "run_id": manifest["run_id"],
        "source_manifest_hash": hash_file_bytes(manifest_path),
        "source_results_hash": hash_file_bytes(results_path),
        "source_manifest_path": str(Path(manifest_path)),
        "source_results_path": str(Path(results_path)),
        "source_manifest": {
            "harness_version": manifest.get("harness_version"),
            "metric_version": manifest.get("metric_version"),
            "extractor_version": manifest.get("extractor_version"),
            "adapter": manifest.get("adapter"),
            "serving_stack": manifest.get("serving_stack"),
            "scoring_capability": manifest.get("scoring_capability"),
        },
        "counts": {
            "persona_count": len(grouped),
            "row_count": len(rows),
            "inference_unit": "persona_id",
            **persona_shape_summary(rows),
        },
        "matched_pair_summary": {
            "matched_result_rows": len(rows),
            "matched_persona_count": len(grouped),
            "match_axes": ["run_id", "persona_id", "variant_id", "variant_type", "seed", "prompt_hash"],
            "note": "Each result row contains one matched base/tuned pair; statistics are computed after persona-level aggregation.",
        },
        "availability_summaries": availability_counts(rows),
        "flagged_personas": summarize_flags(rows),
        "variant_type_breakdowns": variant_type_breakdown(rows),
        "metric_summaries": {
            "behavioral_consistency_f1": summarize_behavioral_consistency(rows),
            "persona_adherence": summarize_persona_adherence(rows),
            "token_kl": summarize_token_kl(rows),
        },
        "paired_output_deltas": summarize_output_deltas(rows),
        "statistical_method_notes": statistical_method_notes(),
        "full_dataset_readiness": full_dataset_readiness(),
        "known_limitations": [
            "Fixture/mock aggregation does not establish real Persona Adherence.",
            "Token-KL is unavailable unless aligned continuation scoring returns status=ok.",
            "MockAdapter output is deterministic plumbing output, not a model-quality benchmark.",
            "Full dataset generation remains blocked until readiness checks pass.",
        ],
    }
    validate_aggregate_report(report)
    return report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def validate_aggregate_report(report: dict[str, Any]) -> None:
    validate_schema_file(report, AGGREGATE_REPORT_SCHEMA_PATH)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_chart_data(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    chart_dir = out_dir / "chart_data"
    variant_rows = [
        {
            "variant_type": variant_type,
            "row_count": summary["row_count"],
            "persona_count": summary["persona_count"],
            "seed_count": summary["seed_count"],
        }
        for variant_type, summary in sorted(report["variant_type_breakdowns"].items())
    ]
    availability_rows: list[dict[str, Any]] = []
    for metric_name, summary in sorted(report["availability_summaries"].items()):
        for status, count in sorted(summary["status_counts"].items()):
            availability_rows.append({"metric": metric_name, "status": status, "count": count})
    persona_rows = [
        {
            "persona_id": persona_id,
            "row_count": report["counts"]["row_count_by_persona"][persona_id],
            "variant_count": report["counts"]["variant_count_by_persona"][persona_id],
            "seed_count": report["counts"]["seed_count_by_persona"][persona_id],
        }
        for persona_id in sorted(report["counts"]["row_count_by_persona"])
    ]

    variant_path = chart_dir / "variant_type_breakdown.csv"
    availability_path = chart_dir / "metric_availability.csv"
    persona_path = chart_dir / "persona_shape.csv"
    write_csv(
        variant_path,
        variant_rows,
        ["variant_type", "row_count", "persona_count", "seed_count"],
    )
    write_csv(availability_path, availability_rows, ["metric", "status", "count"])
    write_csv(persona_path, persona_rows, ["persona_id", "row_count", "variant_count", "seed_count"])
    return {
        "variant_type_breakdown_csv": str(variant_path),
        "metric_availability_csv": str(availability_path),
        "persona_shape_csv": str(persona_path),
    }


def write_report(report: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    validate_aggregate_report(report)
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    chart_paths = write_chart_data(report, out_path)
    report = dict(report)
    report["chart_data_paths"] = chart_paths
    report_path = out_path / "aggregate_report.json"
    write_json(report_path, report)
    return {"aggregate_report_json": str(report_path), **chart_paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate persona drift mock or fixture results")
    parser.add_argument("--manifest", required=True, help="Path to run manifest JSON")
    parser.add_argument("--results", required=True, help="Path to result rows JSONL")
    parser.add_argument("--out", required=True, help="Output directory for aggregate report and chart data")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_report(manifest_path=args.manifest, results_path=args.results)
        written = write_report(report, args.out)
    except (OSError, json.JSONDecodeError, PersonaValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"aggregate_report_path={written['aggregate_report_json']}")
    print(f"variant_type_breakdown_csv={written['variant_type_breakdown_csv']}")
    print(f"metric_availability_csv={written['metric_availability_csv']}")
    print(f"persona_shape_csv={written['persona_shape_csv']}")
    print(f"full_dataset_readiness={report['full_dataset_readiness']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
