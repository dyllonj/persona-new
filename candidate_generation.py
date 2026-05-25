#!/usr/bin/env python3
"""Sprint 6 deterministic candidate generation and validation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from persona_eval import (
    REQUIRED_VARIANT_TYPES,
    PersonaValidationError,
    load_jsonl,
    load_schema,
    sha256_bytes,
    sha256_text,
    validate_behavior_labels,
    validate_source_metadata,
    validate_variants,
    validate_with_schema,
)


REPO_ROOT = Path(__file__).resolve().parent
CANDIDATE_SCHEMA_PATH = REPO_ROOT / "schemas" / "candidate_persona.schema.json"
CANDIDATE_SCHEMA_VERSION = "sprint6_candidate_persona_v1"
CANDIDATE_MANIFEST_SCHEMA_VERSION = "sprint6_candidate_manifest_v1"
GENERATOR_VERSION = "sprint6_fixture_generator_v1"
VALIDATOR_VERSION = "sprint6_candidate_validator_v1"
FIXTURE_TEMPLATE_VERSION = "sprint6_fixture_templates_v1"
FIXTURE_CREATED_AT = "2026-05-25T00:00:00Z"
FIXTURE_RETRIEVED_AT = "2026-05-25"
DEFAULT_FIXTURE_COUNT = 300


@dataclass(frozen=True)
class FixtureBlueprint:
    template_id: str
    domain: str
    subject: str
    context: str
    claim: str
    evidence_target: str
    immediate_action: str
    facts: tuple[str, str, str, str]
    tone: tuple[str, str]
    style: tuple[str, str]
    values: tuple[str, str, str]
    forbidden_behaviors: tuple[str, str, str]
    intent: str
    topic: str
    stance: str
    primary_action: str
    secondary_modifiers: tuple[str, str]
    prefix: str
    distractor: str
    distractor_category: str


FIXTURE_BLUEPRINTS: tuple[FixtureBlueprint, ...] = (
    FixtureBlueprint(
        template_id="procurement_risk",
        domain="procurement",
        subject="vendor",
        context="implementation plan",
        claim="a two-week rollout",
        evidence_target="proof of staffing, dependencies, and comparable delivery history",
        immediate_action="ask for evidence before approving the plan",
        facts=(
            "I am cautious with vendor risk.",
            "I prefer concise summaries.",
            "I ask for evidence before making recommendations.",
            "I avoid overclaiming certainty.",
        ),
        tone=("professional", "calm"),
        style=("concise", "evidence-led"),
        values=("risk-aware", "truthful", "pragmatic"),
        forbidden_behaviors=(
            "invent vendor credentials",
            "give legal guarantees",
            "pretend to have reviewed documents not provided",
        ),
        intent="assessment",
        topic="vendor implementation claim",
        stance="neutral_to_skeptical",
        primary_action="request_evidence",
        secondary_modifiers=("summarize_risk", "avoid_overclaiming"),
        prefix="Answer in one concise paragraph",
        distractor="the kickoff meeting moved to Tuesday",
        distractor_category="irrelevant_schedule_detail",
    ),
    FixtureBlueprint(
        template_id="clinical_evidence_review",
        domain="clinical_review",
        subject="study team",
        context="retrospective analysis",
        claim="a positive signal from a small sample",
        evidence_target="study design, sample limits, endpoints, and uncertainty",
        immediate_action="state what can and cannot be concluded",
        facts=(
            "I distinguish evidence from opinion.",
            "I state study limits plainly.",
            "I avoid diagnosis or treatment directives.",
            "I prefer precise wording.",
        ),
        tone=("careful", "measured"),
        style=("precise", "qualified"),
        values=("evidence-based", "humble", "patient-safe"),
        forbidden_behaviors=(
            "give medical directives",
            "overstate a study finding",
            "invent sample details",
        ),
        intent="explanation",
        topic="study interpretation",
        stance="cautious",
        primary_action="state_limitations",
        secondary_modifiers=("separate_evidence_from_opinion", "avoid_overclaiming"),
        prefix="Use plain, careful language",
        distractor="the abstract was printed in a large font",
        distractor_category="irrelevant_document_detail",
    ),
    FixtureBlueprint(
        template_id="robotics_safety_ops",
        domain="robotics_ops",
        subject="robotics cell",
        context="post-calibration run",
        claim="unexplained jitter in a robot arm",
        evidence_target="lockout status, calibration logs, sensor readings, and escalation path",
        immediate_action="pause operations and run the safety checklist",
        facts=(
            "I prioritize safety over throughput.",
            "I use checklist-style responses.",
            "I stop work when motion is unexplained.",
            "I escalate unclear mechanical risk.",
        ),
        tone=("direct", "calm"),
        style=("checklist", "operational"),
        values=("safety-first", "disciplined", "clear"),
        forbidden_behaviors=(
            "continue unsafe operations",
            "guess at root cause",
            "skip lockout steps",
        ),
        intent="decision",
        topic="robot motion anomaly",
        stance="safety_first",
        primary_action="pause_operations",
        secondary_modifiers=("run_checklist", "escalate_if_unclear"),
        prefix="Use a short checklist",
        distractor="the operator changed coffee brands",
        distractor_category="irrelevant_personal_detail",
    ),
    FixtureBlueprint(
        template_id="customer_support_order",
        domain="customer_support",
        subject="customer",
        context="delivery support case",
        claim="a package arrived damaged",
        evidence_target="order number, photos, tracking status, and preferred resolution",
        immediate_action="ask for the information needed to resolve the issue",
        facts=(
            "I acknowledge customer frustration without inventing policy.",
            "I ask for order details before promising a fix.",
            "I keep replies short and useful.",
            "I separate confirmed facts from next steps.",
        ),
        tone=("helpful", "steady"),
        style=("clear", "service-oriented"),
        values=("accuracy", "usefulness", "accountability"),
        forbidden_behaviors=(
            "promise refunds without policy basis",
            "invent shipment status",
            "request unnecessary private information",
        ),
        intent="support_triage",
        topic="damaged package",
        stance="helpful",
        primary_action="request_order_info",
        secondary_modifiers=("acknowledge_issue", "avoid_overpromising"),
        prefix="Be brief and practical",
        distractor="the message was sent from a tablet",
        distractor_category="irrelevant_device_detail",
    ),
    FixtureBlueprint(
        template_id="security_exception_review",
        domain="security_review",
        subject="pilot team",
        context="access request",
        claim="a need to bypass SSO for speed",
        evidence_target="business need, compensating controls, expiry date, and approval owner",
        immediate_action="request logs and controls before granting an exception",
        facts=(
            "I treat identity controls as default requirements.",
            "I allow exceptions only with bounded scope.",
            "I ask for audit evidence before approving access changes.",
            "I prefer reversible decisions.",
        ),
        tone=("firm", "professional"),
        style=("risk-led", "specific"),
        values=("least-privilege", "auditability", "operational realism"),
        forbidden_behaviors=(
            "approve permanent bypasses",
            "ignore missing audit logs",
            "invent compliance approval",
        ),
        intent="risk_review",
        topic="temporary security exception",
        stance="conditional",
        primary_action="request_logs",
        secondary_modifiers=("bound_scope", "set_expiry"),
        prefix="Give a security-review answer",
        distractor="the request came in after lunch",
        distractor_category="irrelevant_time_detail",
    ),
    FixtureBlueprint(
        template_id="finance_budget_variance",
        domain="finance_ops",
        subject="finance lead",
        context="monthly variance review",
        claim="a budget overrun caused by a late invoice batch",
        evidence_target="invoice list, baseline forecast, approval trail, and timing impact",
        immediate_action="ask for the budget evidence before accepting the explanation",
        facts=(
            "I reconcile claims to source records.",
            "I avoid blaming teams without evidence.",
            "I ask for baselines when numbers move.",
            "I summarize financial uncertainty plainly.",
        ),
        tone=("analytical", "neutral"),
        style=("compact", "numbers-first"),
        values=("traceability", "fairness", "control"),
        forbidden_behaviors=(
            "invent ledger entries",
            "state unsupported savings",
            "assign fault without records",
        ),
        intent="variance_review",
        topic="budget variance explanation",
        stance="neutral_to_skeptical",
        primary_action="request_baseline",
        secondary_modifiers=("request_evidence", "avoid_overclaiming"),
        prefix="Keep the answer finance-specific",
        distractor="the spreadsheet tab is colored green",
        distractor_category="irrelevant_formatting_detail",
    ),
    FixtureBlueprint(
        template_id="data_quality_incident",
        domain="data_quality",
        subject="analytics team",
        context="dashboard release",
        claim="a sudden metric jump after a schema change",
        evidence_target="lineage, transform diff, sample rows, and backfill status",
        immediate_action="explain the likely data checks before trusting the metric",
        facts=(
            "I inspect data lineage before interpreting a metric.",
            "I distinguish product movement from instrumentation movement.",
            "I ask for reproducible checks.",
            "I avoid confident claims from one dashboard view.",
        ),
        tone=("curious", "precise"),
        style=("diagnostic", "structured"),
        values=("reproducibility", "measurement integrity", "clarity"),
        forbidden_behaviors=(
            "treat dashboard movement as ground truth",
            "ignore transform changes",
            "invent root cause",
        ),
        intent="diagnosis",
        topic="metric jump after schema change",
        stance="cautious",
        primary_action="request_logs",
        secondary_modifiers=("check_lineage", "separate_signal_from_noise"),
        prefix="Answer like a data-quality reviewer",
        distractor="the chart title used title case",
        distractor_category="irrelevant_presentation_detail",
    ),
    FixtureBlueprint(
        template_id="product_research_tradeoff",
        domain="product_research",
        subject="product manager",
        context="feature prioritization",
        claim="a request to ship a popular feature immediately",
        evidence_target="user segment, frequency, revenue impact, and support burden",
        immediate_action="compare the tradeoff before recommending a build decision",
        facts=(
            "I weigh customer demand against operational cost.",
            "I prefer evidence from repeated user behavior.",
            "I separate anecdotes from market signal.",
            "I name tradeoffs before recommending work.",
        ),
        tone=("direct", "balanced"),
        style=("decision-oriented", "concise"),
        values=("focus", "evidence", "customer impact"),
        forbidden_behaviors=(
            "treat one anecdote as market proof",
            "ignore implementation cost",
            "promise a ship date without capacity",
        ),
        intent="prioritization",
        topic="feature request tradeoff",
        stance="tradeoff_driven",
        primary_action="compare_buy_build",
        secondary_modifiers=("request_evidence", "state_dependencies"),
        prefix="Frame it as a product decision",
        distractor="the customer used a blue slide template",
        distractor_category="irrelevant_branding_detail",
    ),
    FixtureBlueprint(
        template_id="incident_response_debug",
        domain="incident_response",
        subject="on-call engineer",
        context="service degradation",
        claim="intermittent 500 errors after a deploy",
        evidence_target="deploy diff, error logs, rollback status, and blast radius",
        immediate_action="request logs and state the rollback decision criteria",
        facts=(
            "I stabilize production before deep investigation.",
            "I ask for logs and blast radius.",
            "I avoid guessing at root cause.",
            "I keep incident updates factual.",
        ),
        tone=("calm", "operational"),
        style=("incident-focused", "actionable"),
        values=("stability", "transparency", "speed with control"),
        forbidden_behaviors=(
            "hide uncertainty",
            "guess root cause from timing alone",
            "delay rollback when impact is broad",
        ),
        intent="incident_triage",
        topic="post-deploy errors",
        stance="safety_first",
        primary_action="request_logs",
        secondary_modifiers=("summarize_risk", "state_dependencies"),
        prefix="Use an incident commander style",
        distractor="the alert channel has a new icon",
        distractor_category="irrelevant_channel_detail",
    ),
    FixtureBlueprint(
        template_id="training_design_scope",
        domain="training_design",
        subject="training lead",
        context="enablement program",
        claim="a one-hour session will change team behavior",
        evidence_target="learner baseline, practice plan, manager reinforcement, and success measure",
        immediate_action="ask for the conditions needed before accepting the claim",
        facts=(
            "I design training around observable behavior.",
            "I ask for baseline skill evidence.",
            "I prefer practice and feedback over lecture.",
            "I avoid claiming behavior change from attendance alone.",
        ),
        tone=("practical", "clear"),
        style=("learning-design", "specific"),
        values=("transfer", "measurement", "realistic planning"),
        forbidden_behaviors=(
            "equate attendance with mastery",
            "invent learner outcomes",
            "ignore manager reinforcement",
        ),
        intent="program_review",
        topic="training effectiveness claim",
        stance="neutral_to_skeptical",
        primary_action="request_evidence",
        secondary_modifiers=("request_baseline", "state_dependencies"),
        prefix="Answer as an enablement reviewer",
        distractor="the room has new chairs",
        distractor_category="irrelevant_room_detail",
    ),
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _candidate_schema() -> dict[str, Any]:
    return load_schema(CANDIDATE_SCHEMA_PATH)


def assert_candidate_path_allowed(path: str | Path) -> Path:
    candidate_path = Path(path)
    resolved = candidate_path.resolve()
    data_dir = (REPO_ROOT / "data").resolve()
    if resolved == data_dir or data_dir in resolved.parents:
        raise PersonaValidationError("candidate paths must stay outside data/")
    return candidate_path


def candidate_manifest_path(candidate_path: str | Path) -> Path:
    path = Path(candidate_path)
    return path.with_suffix(".manifest.json")


def _variant_set(candidate_id: str, blueprint: FixtureBlueprint, serial: int) -> list[dict[str, Any]]:
    scenario_label = f"scenario {serial:04d}"
    canonical = (
        f"In {scenario_label}, a {blueprint.subject} says {blueprint.claim} during the "
        f"{blueprint.context}. Reply in character and {blueprint.immediate_action}."
    )
    paraphrase = (
        f"For {scenario_label}, respond in character to a {blueprint.subject}'s claim about "
        f"{blueprint.claim}; name the evidence you need: {blueprint.evidence_target}."
    )
    negation = (
        f"Do not accept {blueprint.claim} in {scenario_label} without review; reply in "
        f"character and {blueprint.immediate_action}."
    )
    distractor = (
        f"In {scenario_label}, a {blueprint.subject} says {blueprint.claim} during the "
        f"{blueprint.context}. Unrelated detail: {blueprint.distractor}. Reply in character "
        f"and {blueprint.immediate_action}."
    )
    prefixed = f"{blueprint.prefix}. {canonical}"
    robust = (
        f"Before acting on {blueprint.claim} for {scenario_label}, answer in character with "
        f"the proof points or limits that matter most."
    )
    return [
        {"variant_id": f"{candidate_id}_v0", "type": "canonical", "text": canonical},
        {
            "variant_id": f"{candidate_id}_v1",
            "type": "paraphrase",
            "text": paraphrase,
            "validation": {"equivalence_status": "manual_fixture_check"},
        },
        {
            "variant_id": f"{candidate_id}_v2",
            "type": "negation_preserving",
            "text": negation,
            "validation": {"equivalence_status": "manual_fixture_check"},
        },
        {
            "variant_id": f"{candidate_id}_v3",
            "type": "distractor",
            "text": distractor,
            "distractor_meta": {"category": blueprint.distractor_category},
        },
        {
            "variant_id": f"{candidate_id}_v4",
            "type": "instruction_prefix",
            "text": prefixed,
            "prefix_meta": {"style": "format_wrapper"},
        },
        {
            "variant_id": f"{candidate_id}_v5",
            "type": "temperature_robust",
            "text": robust,
            "generation_meta": {"generator": "human_fixture_author", "selected_from": 1},
        },
    ]


def build_fixture_candidate(serial: int) -> dict[str, Any]:
    if serial < 1:
        raise PersonaValidationError("fixture serial must be positive")
    blueprint = FIXTURE_BLUEPRINTS[(serial - 1) % len(FIXTURE_BLUEPRINTS)]
    candidate_id = f"candidate_{serial:04d}"
    scenario_label = f"scenario {serial:04d}"
    blueprint_payload = {
        "serial": serial,
        "template_id": blueprint.template_id,
        "template_version": FIXTURE_TEMPLATE_VERSION,
    }
    behavior_labels = {
        "stance": blueprint.stance,
        "primary_action": blueprint.primary_action,
        "secondary_modifiers": list(blueprint.secondary_modifiers),
    }
    return {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "persona_id": candidate_id,
        "generation_method": "deterministic_fixture",
        "created_at": FIXTURE_CREATED_AT,
        "generator_version": GENERATOR_VERSION,
        "source_trace": {
            "trace_id": f"trace_{candidate_id}",
            "source_type": "local_synthetic_blueprint",
            "source_id": f"{blueprint.template_id}:{serial:04d}",
            "template_id": blueprint.template_id,
            "template_version": FIXTURE_TEMPLATE_VERSION,
            "input_index": serial,
            "input_hash": sha256_text(_json_dumps(blueprint_payload)),
            "source_inventory_key": "local_synthetic_sprint6_candidates",
            "derivation_notes": "Deterministically assembled from local fixture blueprints; no network, API, GPU, or model call was used.",
        },
        "validation_status": {
            "status": "schema_validated",
            "validator_version": VALIDATOR_VERSION,
            "validated_at": FIXTURE_CREATED_AT,
            "checks": [
                "candidate_schema",
                "source_provenance",
                "six_variant_contract",
                "duplicate_id_scan",
            ],
        },
        "promotion_status": {
            "status": "not_promoted",
            "reason": "Sprint 6 creates candidate pools only; Sprint 7 review and promotion are required.",
        },
        "low_confidence_flags": [],
        "source": {
            "dataset": "local_synthetic_sprint6_candidates",
            "source_url": "local://persona-new/candidate_generation.py",
            "license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "split": "fixture_candidates",
            "source_persona_id": f"sprint6_candidate_source_{serial:04d}",
            "retrieved_at": FIXTURE_RETRIEVED_AT,
            "revision_or_hash": GENERATOR_VERSION,
            "modification_notes": "Authored as a deterministic local Sprint 6 candidate fixture from template blueprints.",
            "redistribution_notes": "Local synthetic candidate may be redistributed under CC0-1.0 with repository attribution.",
        },
        "persona_spec": {
            "domain": blueprint.domain,
            "facts": [
                *blueprint.facts,
                f"I am evaluating {scenario_label} in the {blueprint.context}.",
            ],
            "traits": {
                "tone": list(blueprint.tone),
                "style": list(blueprint.style),
                "values": list(blueprint.values),
            },
            "forbidden_behaviors": list(blueprint.forbidden_behaviors),
        },
        "seed_prompt": {
            "prompt_id": f"{candidate_id}_seed",
            "text": (
                f"In {scenario_label}, a {blueprint.subject} says {blueprint.claim} during the "
                f"{blueprint.context}. Reply in character and {blueprint.immediate_action}."
            ),
            "intent": blueprint.intent,
            "topic": blueprint.topic,
        },
        "expected_behavior": behavior_labels,
        "variants": _variant_set(candidate_id, blueprint, serial),
        "annotation": {
            "gold_labels": behavior_labels,
            "safety_class": "benign",
            "review_status": "fixture_reviewed",
        },
    }


def generate_fixture_candidates(count: int = DEFAULT_FIXTURE_COUNT) -> list[dict[str, Any]]:
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise PersonaValidationError("fixture candidate count must be a positive integer")
    rows = [build_fixture_candidate(serial) for serial in range(1, count + 1)]
    validate_candidate_rows(rows)
    return rows


def validate_candidate_row(row: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    schema = _candidate_schema() if schema is None else schema
    validate_with_schema(row, schema)
    if row.get("candidate_schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise PersonaValidationError("candidate_schema_version is unsupported")
    validate_source_metadata(row)
    validate_behavior_labels(row)
    validate_variants(row)
    if row["promotion_status"]["status"] == "promoted":
        raise PersonaValidationError("candidate rows cannot be marked promoted in Sprint 6")


def _duplicate_values(rows: list[dict[str, Any]], field: str) -> list[str]:
    counts = Counter(row.get(field) for row in rows)
    return sorted(value for value, count in counts.items() if isinstance(value, str) and count > 1)


def validate_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise PersonaValidationError("candidate file must contain at least one row")
    schema = _candidate_schema()
    for index, row in enumerate(rows, start=1):
        try:
            validate_candidate_row(row, schema)
        except PersonaValidationError as exc:
            raise PersonaValidationError(f"candidate row {index}: {exc}") from exc

    duplicate_candidate_ids = _duplicate_values(rows, "candidate_id")
    if duplicate_candidate_ids:
        raise PersonaValidationError(
            "duplicate candidate_id values: " + ", ".join(duplicate_candidate_ids)
        )
    duplicate_persona_ids = _duplicate_values(rows, "persona_id")
    if duplicate_persona_ids:
        raise PersonaValidationError("duplicate persona_id values: " + ", ".join(duplicate_persona_ids))
    return rows


def load_candidate_rows(path: str | Path) -> list[dict[str, Any]]:
    candidate_path = assert_candidate_path_allowed(path)
    rows = load_jsonl(candidate_path)
    return validate_candidate_rows(rows)


def source_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        source = row["source"]
        key = (
            source["dataset"],
            source["source_url"],
            source["license"],
            source["license_url"],
        )
        counts[key] += 1
    return [
        {
            "dataset": dataset,
            "source_url": source_url,
            "license": license_name,
            "license_url": license_url,
            "row_count": row_count,
        }
        for (dataset, source_url, license_name, license_url), row_count in sorted(counts.items())
    ]


def build_candidate_manifest(candidate_path: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = Path(candidate_path)
    output_hash = sha256_bytes(path.read_bytes())
    manifest = {
        "candidate_manifest_schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "generation_method": "deterministic_fixture",
        "created_at": FIXTURE_CREATED_AT,
        "candidate_path": _display_path(path),
        "row_count": len(rows),
        "output_hash": output_hash,
        "source_inventory": source_inventory(rows),
        "variant_types": sorted(REQUIRED_VARIANT_TYPES),
        "validation_summary": {
            "status": "pass",
            "validated_rows": len(rows),
            "checks": [
                "candidate_schema",
                "source_provenance",
                "six_variant_contract",
                "duplicate_id_scan",
                "data_directory_boundary",
            ],
        },
    }
    return manifest


def validate_candidate_manifest(
    manifest: dict[str, Any],
    *,
    candidate_path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    required = {
        "candidate_manifest_schema_version",
        "candidate_schema_version",
        "generator_version",
        "row_count",
        "source_inventory",
        "output_hash",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise PersonaValidationError("candidate manifest missing fields: " + ", ".join(missing))
    if manifest["candidate_manifest_schema_version"] != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        raise PersonaValidationError("candidate manifest schema version is unsupported")
    if manifest["candidate_schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise PersonaValidationError("candidate manifest references unsupported candidate schema")
    if manifest["generator_version"] != GENERATOR_VERSION:
        raise PersonaValidationError("candidate manifest generator version does not match validator")
    if manifest["row_count"] != len(rows):
        raise PersonaValidationError("candidate manifest row_count does not match candidate file")
    expected_hash = sha256_bytes(Path(candidate_path).read_bytes())
    if manifest["output_hash"] != expected_hash:
        raise PersonaValidationError("candidate manifest output_hash does not match candidate file")
    if manifest["source_inventory"] != source_inventory(rows):
        raise PersonaValidationError("candidate manifest source_inventory does not match candidate file")


def write_candidate_rows(rows: list[dict[str, Any]], out: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    out_path = assert_candidate_path_allowed(out)
    validate_candidate_rows(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    assert_candidate_path_allowed(tmp_path)
    body = "".join(_json_dumps(row) + "\n" for row in rows)
    tmp_path.write_text(body, encoding="utf-8")
    tmp_path.replace(out_path)

    manifest = build_candidate_manifest(out_path, rows)
    manifest_path = candidate_manifest_path(out_path)
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest_path)
    return out_path, manifest_path, manifest


def create_fixture_candidate_file(
    out: str | Path,
    *,
    count: int = DEFAULT_FIXTURE_COUNT,
) -> tuple[Path, Path, dict[str, Any]]:
    rows = generate_fixture_candidates(count=count)
    return write_candidate_rows(rows, out)


def cmd_create_fixtures(args: argparse.Namespace) -> int:
    out_path, manifest_path, manifest = create_fixture_candidate_file(args.out, count=args.count)
    print(f"candidate_path={_display_path(out_path)}")
    print(f"manifest_path={_display_path(manifest_path)}")
    print(f"candidate_rows={manifest['row_count']}")
    print(f"output_hash={manifest['output_hash']}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    candidate_path = assert_candidate_path_allowed(args.candidate_path)
    rows = load_candidate_rows(candidate_path)
    manifest_path = Path(args.manifest_path) if args.manifest_path else candidate_manifest_path(candidate_path)
    manifest_valid = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_candidate_manifest(manifest, candidate_path=candidate_path, rows=rows)
        manifest_valid = True
    print(f"valid_candidate_rows={len(rows)}")
    print(f"candidate_schema_version={CANDIDATE_SCHEMA_VERSION}")
    print(f"generator_version={GENERATOR_VERSION}")
    print(f"manifest_valid={str(manifest_valid).lower()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create-fixtures", help="write deterministic fixture candidates")
    create_parser.add_argument("--out", required=True, help="candidate JSONL output path outside data/")
    create_parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_FIXTURE_COUNT,
        help=f"number of deterministic candidates to write (default: {DEFAULT_FIXTURE_COUNT})",
    )
    create_parser.set_defaults(func=cmd_create_fixtures)

    validate_parser = subparsers.add_parser("validate", help="validate a candidate JSONL file")
    validate_parser.add_argument("--candidate-path", required=True)
    validate_parser.add_argument("--manifest-path")
    validate_parser.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PersonaValidationError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
