#!/usr/bin/env python3
"""Fixture-first persona drift evaluation harness."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover - covered only in stripped environments.
    jsonschema = None


REPO_ROOT = Path(__file__).resolve().parent
PERSONA_SCHEMA_PATH = REPO_ROOT / "schemas" / "persona_item.schema.json"
BEHAVIOR_TAGS_SCHEMA_PATH = REPO_ROOT / "schemas" / "behavior_tags.schema.json"
RUN_MANIFEST_SCHEMA_PATH = REPO_ROOT / "schemas" / "run_manifest.schema.json"
RESULT_ROW_SCHEMA_PATH = REPO_ROOT / "schemas" / "result_row.schema.json"

PROMPT_TEMPLATE_VERSION = "v1"
EVALUATOR_VERSION = "sprint2"
DEFAULT_SYSTEM_PROMPT = "\n".join(
    [
        "You are participating in a persona-drift evaluation.",
        "Answer the user task while following the persona contract.",
        "Do not invent persona facts, credentials, sources, or hidden context.",
    ]
)
PROMPT_TEMPLATE_SOURCE = "\n".join(
    [
        "prompt_template_version={prompt_template_version}",
        "Persona Facts:",
        "{persona_facts}",
        "Persona Traits:",
        "{persona_traits}",
        "Persona Values:",
        "{persona_values}",
        "Forbidden Behaviors:",
        "{forbidden_behaviors}",
        "User Task / Question:",
        "{user_task}",
    ]
)

TOKEN_KL_STATUS_FIELDS = (
    "status",
    "value",
    "reason_code",
    "scoring_path",
    "fixed_continuation_id",
    "fixed_continuation_hash",
    "tokenizer_hash_match",
    "vocabulary_match",
    "chat_template_hash_match",
    "k",
    "endpoint_cap",
    "diagnostic_only",
)
TOKEN_KL_STATUS_VALUES = {"ok", "not_applicable", "diagnostic_only", "error"}
TOKEN_KL_SCORING_PATHS = {
    "local_forward",
    "vllm_prompt_logprobs",
    "one_token_loop",
    "hosted_top_logprobs",
    "none",
}
CACHE_KEY_FIELDS = (
    "prompt_hash",
    "system_prompt_hash",
    "model_id",
    "model_revision_or_hash",
    "tokenizer_hash",
    "chat_template_hash",
    "decoding_params",
    "seed",
    "evaluator_version",
    "adapter",
    "provider_or_endpoint",
    "serving_stack",
    "serving_stack_version",
    "scoring_capability",
)

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
BEHAVIOR_TAG_STATUSES = {"not_run", "ok", "ambiguous", "error"}
CONTROLLED_STANCES = {
    "support",
    "oppose",
    "neutral",
    "conditional",
    "uncertain",
    "neutral_to_skeptical",
    "cautious",
    "safety_first",
    "helpful",
    "skeptical",
    "supportive",
    "tradeoff_driven",
}
CONTROLLED_PRIMARY_ACTIONS = {
    "recommend",
    "request_evidence",
    "refuse",
    "ask_followup",
    "escalate",
    "summarize_risk",
    "state_limitations",
    "pause_operations",
    "request_order_info",
    "request_logs",
    "ask_budget",
    "state_dependencies",
    "explain_error",
    "request_baseline",
    "compare_buy_build",
}
LOGICALLY_INVALID_BEHAVIOR_PAIRS = {
    ("support", "refuse"),
}
METRIC_STATUS_VALUES = {"not_run", "ok", "not_applicable", "mock_only", "diagnostic_only", "error"}


class PersonaValidationError(ValueError):
    """Raised when a persona fixture violates the Sprint 0 contract."""


class TokenKLUnavailableError(ValueError):
    """Raised when canonical Token-KL is requested from invalid inputs."""


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_template_version: str
    prompt_template_hash: str
    prompt_text: str
    system_prompt: str
    prompt_hash: str
    system_prompt_hash: str


@dataclass(frozen=True)
class GenerationRequest:
    run_id: str
    persona_id: str
    variant_id: str
    variant_type: str
    model_alias: str
    model_id: str
    model_revision_or_hash: str
    tokenizer_name: str
    tokenizer_hash: str
    chat_template_hash: str
    prompt_template_version: str
    prompt_template_hash: str
    prompt_text: str
    system_prompt: str
    seed: int
    decoding_params: dict[str, Any]
    stop_sequences: list[str]
    prompt_hash: str
    system_prompt_hash: str
    adapter: str = "mock"
    provider_or_endpoint: str = "local_mock"
    serving_stack: str = "mock"
    serving_stack_version: str = "sprint1"
    scoring_capability: str = "none"

    def to_raw_request(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    status: str
    reason_code: str | None
    response_text: str
    stop_reason: str
    truncation_flag: bool
    usage: dict[str, int]
    latency_s: float
    raw_response: dict[str, Any]

    def to_model_output(self, request: GenerationRequest) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "response_text": self.response_text,
            "raw_request": request.to_raw_request(),
            "stop_reason": self.stop_reason,
            "truncation_flag": self.truncation_flag,
            "usage": self.usage,
            "latency_s": self.latency_s,
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class ScoreContinuationRequest:
    run_id: str
    persona_id: str
    variant_id: str
    variant_type: str
    model_alias: str
    model_id: str
    model_revision_or_hash: str
    tokenizer_name: str
    tokenizer_hash: str
    chat_template_hash: str
    prompt_template_version: str
    prompt_template_hash: str
    prompt_text: str
    system_prompt: str
    seed: int
    decoding_params: dict[str, Any]
    stop_sequences: list[str]
    fixed_continuation: str
    fixed_continuation_id: str
    k: int
    prompt_hash: str
    system_prompt_hash: str
    adapter: str = "mock"
    provider_or_endpoint: str = "local_mock"
    serving_stack: str = "mock"
    serving_stack_version: str = "sprint1"
    scoring_capability: str = "none"

    def to_raw_request(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreContinuationResult:
    status: str
    value: float | None
    reason_code: str | None
    scoring_path: str
    fixed_continuation_id: str | None = None
    fixed_continuation_hash: str | None = None
    tokenizer_hash_match: bool | None = None
    vocabulary_match: bool | None = None
    chat_template_hash_match: bool | None = None
    k: int | None = None
    endpoint_cap: int | None = None
    diagnostic_only: bool = False

    def to_token_kl_status(self) -> dict[str, Any]:
        status = asdict(self)
        validate_token_kl_status(status)
        return status


class BaseAdapter(ABC):
    """Adapter contract for model generation and aligned continuation scoring."""

    adapter_name = "base"
    provider_or_endpoint = "not_available"
    serving_stack = "not_available"
    serving_stack_version = "not_available"
    scoring_capability = "none"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a model generation result for the rendered prompt."""

    @abstractmethod
    def score_continuation(self, request: ScoreContinuationRequest) -> ScoreContinuationResult:
        """Score a fixed continuation, or return a structured unavailable status."""


class MockAdapter(BaseAdapter):
    """Deterministic local adapter for Sprint 1 tests and mock runs."""

    adapter_name = "mock"
    provider_or_endpoint = "local_mock"
    serving_stack = "mock"
    serving_stack_version = "sprint1"
    scoring_capability = "none"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = {
            "run_id": request.run_id,
            "persona_id": request.persona_id,
            "variant_id": request.variant_id,
            "variant_type": request.variant_type,
            "model_alias": request.model_alias,
            "model_id": request.model_id,
            "model_revision_or_hash": request.model_revision_or_hash,
            "tokenizer_name": request.tokenizer_name,
            "tokenizer_hash": request.tokenizer_hash,
            "chat_template_hash": request.chat_template_hash,
            "prompt_template_version": request.prompt_template_version,
            "prompt_template_hash": request.prompt_template_hash,
            "prompt_hash": request.prompt_hash,
            "system_prompt_hash": request.system_prompt_hash,
            "seed": request.seed,
            "decoding_params": request.decoding_params,
            "stop_sequences": request.stop_sequences,
            "adapter": self.adapter_name,
        }
        digest = sha256_json(payload)
        response_text = (
            f"mock_response model={request.model_id} seed={request.seed} "
            f"prompt={request.prompt_hash} output={digest}"
        )
        usage = {
            "prompt_tokens": token_count(request.system_prompt) + token_count(request.prompt_text),
            "completion_tokens": token_count(response_text),
            "total_tokens": token_count(request.system_prompt)
            + token_count(request.prompt_text)
            + token_count(response_text),
        }
        raw_response = {
            "adapter": self.adapter_name,
            "run_id": request.run_id,
            "persona_id": request.persona_id,
            "variant_id": request.variant_id,
            "variant_type": request.variant_type,
            "model_alias": request.model_alias,
            "model_id": request.model_id,
            "seed": request.seed,
            "prompt_hash": request.prompt_hash,
            "output_hash": digest,
            "response_text": response_text,
        }
        return GenerationResult(
            status="ok",
            reason_code=None,
            response_text=response_text,
            stop_reason="mock_complete",
            truncation_flag=False,
            usage=usage,
            latency_s=0.0,
            raw_response=raw_response,
        )

    def score_continuation(self, request: ScoreContinuationRequest) -> ScoreContinuationResult:
        del request
        return ScoreContinuationResult(
            status="not_applicable",
            value=None,
            reason_code="aligned_scoring_unavailable",
            scoring_path="none",
            fixed_continuation_id=None,
            fixed_continuation_hash=None,
            tokenizer_hash_match=None,
            vocabulary_match=None,
            chat_template_hash_match=None,
            k=None,
            endpoint_cap=None,
            diagnostic_only=False,
        )


class BaseEmbeddingBackend(ABC):
    """Interface for Persona Adherence semantic backends."""

    backend_name = "base"
    model_revision = "not_pinned"

    @abstractmethod
    def embed(self, text: str) -> dict[str, float]:
        """Return a deterministic sparse embedding for local fixture tests."""


class MockEmbeddingBackend(BaseEmbeddingBackend):
    """Deterministic token-overlap embedding for plumbing tests only."""

    backend_name = "mock_embedding"
    model_revision = "mock_embedding_sprint2"

    def embed(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize_for_mock_metrics(text))
        total = math.sqrt(sum(count * count for count in counts.values()))
        if total == 0.0:
            return {}
        return {token: count / total for token, count in sorted(counts.items())}


class BaseNLIJudge(ABC):
    """Interface for fact contradiction checks used by Persona Adherence."""

    backend_name = "base"
    model_revision = "not_pinned"

    @abstractmethod
    def judge(self, response_text: str, fact: str) -> str:
        """Return entailment, neutral, or contradiction for local fixture tests."""


class MockNLIJudge(BaseNLIJudge):
    """Deterministic heuristic NLI judge for plumbing tests only."""

    backend_name = "mock_nli"
    model_revision = "mock_nli_sprint2"

    def judge(self, response_text: str, fact: str) -> str:
        response_tokens = set(tokenize_for_mock_metrics(response_text))
        fact_tokens = set(tokenize_for_mock_metrics(fact))
        if not fact_tokens:
            return "neutral"
        if {"not", "never", "ignore"} & response_tokens and len(response_tokens & fact_tokens) >= 2:
            return "contradiction"
        if len(response_tokens & fact_tokens) >= max(2, min(4, len(fact_tokens) // 2)):
            return "entailment"
        return "neutral"


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


def validate_schema_file(value: dict[str, Any], schema_path: str | Path) -> None:
    validate_with_schema(value, load_schema(schema_path))


def validate_run_manifest(manifest: dict[str, Any]) -> None:
    validate_schema_file(manifest, RUN_MANIFEST_SCHEMA_PATH)


def validate_result_row(result_row: dict[str, Any]) -> None:
    validate_schema_file(result_row, RESULT_ROW_SCHEMA_PATH)
    validate_behavior_tags(result_row.get("behavior_tags", {}))
    metrics = result_row.get("metrics", {})
    if not isinstance(metrics, dict):
        raise PersonaValidationError("result_row.metrics must be an object")
    validate_token_kl_status(metrics.get("token_kl", {}))
    validate_persona_adherence_status(metrics.get("persona_adherence", {}))
    validate_behavioral_consistency_status(metrics.get("behavioral_consistency_f1", {}))


def _validate_metric_base(metric: Any, metric_name: str) -> dict[str, Any]:
    if not isinstance(metric, dict):
        raise PersonaValidationError(f"metrics.{metric_name} must be an object")
    status = metric.get("status")
    if status not in METRIC_STATUS_VALUES:
        raise PersonaValidationError(f"metrics.{metric_name}.status is unsupported: {status!r}")
    if status != "ok" and status != "not_run" and not metric.get("reason_code"):
        raise PersonaValidationError(f"metrics.{metric_name}.reason_code is required")
    return metric


def validate_behavioral_consistency_status(metric: Any) -> None:
    metric = _validate_metric_base(metric, "behavioral_consistency_f1")
    status = metric["status"]
    if status == "not_run":
        return
    if status != "ok":
        return
    for field in (
        "stance_exact",
        "primary_action_exact",
        "secondary_modifiers_precision",
        "secondary_modifiers_recall",
        "secondary_modifiers_f1",
        "combined_score",
    ):
        value = metric.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PersonaValidationError(f"metrics.behavioral_consistency_f1.{field} must be numeric")
        if value < 0.0 or value > 1.0:
            raise PersonaValidationError(f"metrics.behavioral_consistency_f1.{field} must be in [0, 1]")
    if metric.get("reason_code") is not None:
        raise PersonaValidationError("metrics.behavioral_consistency_f1.reason_code must be null when ok")


def validate_persona_adherence_status(metric: Any) -> None:
    metric = _validate_metric_base(metric, "persona_adherence")
    status = metric["status"]
    if status in {"not_run", "not_applicable", "error"}:
        if metric.get("value") is not None:
            raise PersonaValidationError("metrics.persona_adherence.value must be null unless real scoring is pinned")
        return
    if status != "mock_only":
        raise PersonaValidationError("metrics.persona_adherence cannot report real ok scores in Sprint 2")
    required_fields = (
        "reason_code",
        "value",
        "mock_score",
        "semantic_similarity",
        "threshold",
        "pass_at_threshold",
        "fact_contradiction_rate",
        "persona_serialization_hash",
        "calibration_id",
        "calibration_hash",
        "embedding_model_revision",
        "nli_or_judge_model_revision",
        "mock_only",
    )
    missing = [field for field in required_fields if field not in metric]
    if missing:
        raise PersonaValidationError(
            "metrics.persona_adherence mock_only missing fields: " + ", ".join(missing)
        )
    if metric.get("value") is not None:
        raise PersonaValidationError("metrics.persona_adherence.value must remain null in mock_only mode")
    if metric.get("mock_only") is not True:
        raise PersonaValidationError("metrics.persona_adherence.mock_only must be true in mock_only mode")
    if not isinstance(metric.get("reason_code"), str) or not metric["reason_code"].strip():
        raise PersonaValidationError("metrics.persona_adherence.reason_code is required in mock_only mode")
    mock_score = metric.get("mock_score")
    if not isinstance(mock_score, (int, float)) or isinstance(mock_score, bool):
        raise PersonaValidationError("metrics.persona_adherence.mock_score must be numeric in mock_only mode")
    if mock_score < 0.0 or mock_score > 1.0:
        raise PersonaValidationError("metrics.persona_adherence.mock_score must be in [0, 1]")
    semantic_similarity = metric.get("semantic_similarity")
    if not isinstance(semantic_similarity, (int, float)) or isinstance(semantic_similarity, bool):
        raise PersonaValidationError(
            "metrics.persona_adherence.semantic_similarity must be numeric in mock_only mode"
        )
    if semantic_similarity < 0.0 or semantic_similarity > 1.0:
        raise PersonaValidationError("metrics.persona_adherence.semantic_similarity must be in [0, 1]")
    contradiction_rate = metric.get("fact_contradiction_rate")
    if not isinstance(contradiction_rate, (int, float)) or isinstance(contradiction_rate, bool):
        raise PersonaValidationError(
            "metrics.persona_adherence.fact_contradiction_rate must be numeric in mock_only mode"
        )
    if contradiction_rate < 0.0 or contradiction_rate > 1.0:
        raise PersonaValidationError("metrics.persona_adherence.fact_contradiction_rate must be in [0, 1]")
    if metric.get("threshold") is not None or metric.get("pass_at_threshold") is not None:
        raise PersonaValidationError(
            "metrics.persona_adherence threshold fields must remain null in mock_only mode"
        )
    for field in ("persona_serialization_hash", "embedding_model_revision", "nli_or_judge_model_revision"):
        if not isinstance(metric.get(field), str) or not metric[field].strip():
            raise PersonaValidationError(f"metrics.persona_adherence.{field} is required in mock_only mode")
    for field in ("calibration_id", "calibration_hash"):
        value = metric.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise PersonaValidationError(
                f"metrics.persona_adherence.{field} must be null or a non-empty string"
            )


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


def normalize_behavior_labels(value: dict[str, Any]) -> dict[str, Any]:
    labels = value.get("labels") if isinstance(value, dict) and "labels" in value else value
    if not isinstance(labels, dict):
        raise PersonaValidationError("behavior labels must be an object")
    _validate_behavior_object(labels, "behavior_labels")
    return {
        "stance": labels["stance"],
        "primary_action": labels["primary_action"],
        "secondary_modifiers": sorted(set(labels["secondary_modifiers"])),
    }


def validate_behavior_tags(tags: dict[str, Any]) -> None:
    validate_schema_file(tags, BEHAVIOR_TAGS_SCHEMA_PATH)
    status = tags.get("status")
    if status not in BEHAVIOR_TAG_STATUSES:
        raise PersonaValidationError(f"behavior_tags.status is unsupported: {status!r}")
    if status == "not_run":
        return

    reason_code = tags.get("reason_code")
    labels = tags.get("labels")
    if status == "ok":
        if reason_code is not None:
            raise PersonaValidationError("behavior_tags.reason_code must be null when status is ok")
        normalized = normalize_behavior_labels(labels)
        pair = (normalized["stance"], normalized["primary_action"])
        if pair in LOGICALLY_INVALID_BEHAVIOR_PAIRS:
            raise PersonaValidationError(
                f"behavior_tags has logically invalid stance/action pair: {pair[0]} + {pair[1]}"
            )
    else:
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise PersonaValidationError("behavior_tags.reason_code is required unless status is ok")
        if labels is not None:
            raise PersonaValidationError("behavior_tags.labels must be null unless status is ok")


def behavior_tags_not_run() -> dict[str, Any]:
    return {"status": "not_run"}


def behavior_tags_ok(
    labels: dict[str, Any],
    *,
    parser: str = "rule_first",
    parser_fallback: bool = False,
    json_repair_used: bool = False,
    human_review_required: bool = False,
    llm_tagger_status: str = "disabled",
) -> dict[str, Any]:
    modifiers = labels.get("secondary_modifiers") if isinstance(labels, dict) else None
    if isinstance(modifiers, list) and len(modifiers) != len(set(modifiers)):
        raise PersonaValidationError("behavior_tags.labels.secondary_modifiers must not contain duplicates")
    tags = {
        "status": "ok",
        "reason_code": None,
        "parser": parser,
        "labels": normalize_behavior_labels(labels),
        "parser_fallback": parser_fallback,
        "json_repair_used": json_repair_used,
        "human_review_required": human_review_required,
        "llm_tagger_status": llm_tagger_status,
    }
    validate_behavior_tags(tags)
    return tags


def extract_behavior_tags_rule_first(text: str, *, llm_tagger_enabled: bool = False) -> dict[str, Any]:
    lowered = text.lower()
    labels: dict[str, Any] | None = None
    if any(token in lowered for token in ("evidence", "proof", "supporting docs")):
        labels = {
            "stance": "neutral_to_skeptical",
            "primary_action": "request_evidence",
            "secondary_modifiers": ["avoid_overclaiming"],
        }
    elif any(token in lowered for token in ("cannot conclude", "limitations", "limited study")):
        labels = {
            "stance": "cautious",
            "primary_action": "state_limitations",
            "secondary_modifiers": ["avoid_overclaiming"],
        }
    elif any(token in lowered for token in ("pause", "halt", "stop operations", "stop work")):
        labels = {
            "stance": "safety_first",
            "primary_action": "pause_operations",
            "secondary_modifiers": ["escalate_if_unclear"],
        }
    elif any(token in lowered for token in ("order", "package")) and any(
        token in lowered for token in ("arrived", "order number", "tracking")
    ):
        labels = {
            "stance": "helpful",
            "primary_action": "request_order_info",
            "secondary_modifiers": ["acknowledge_issue"],
        }
    elif "refuse" in lowered or "can't help with that" in lowered:
        labels = {"stance": "oppose", "primary_action": "refuse", "secondary_modifiers": []}
    elif "recommend" in lowered:
        labels = {"stance": "support", "primary_action": "recommend", "secondary_modifiers": []}

    if labels is not None:
        return behavior_tags_ok(labels)

    tags = {
        "status": "ambiguous",
        "reason_code": "rule_parser_no_match",
        "parser": "rule_first",
        "labels": None,
        "parser_fallback": False,
        "json_repair_used": False,
        "human_review_required": True,
        "llm_tagger_status": "disabled" if not llm_tagger_enabled else "disabled_no_api_calls",
    }
    validate_behavior_tags(tags)
    return tags


def _modifier_prf(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def behavior_consistency_f1(predicted: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    predicted_labels = normalize_behavior_labels(predicted)
    expected_labels = normalize_behavior_labels(expected)
    stance_exact = 1.0 if predicted_labels["stance"] == expected_labels["stance"] else 0.0
    primary_action_exact = (
        1.0 if predicted_labels["primary_action"] == expected_labels["primary_action"] else 0.0
    )
    precision, recall, modifier_f1 = _modifier_prf(
        set(predicted_labels["secondary_modifiers"]),
        set(expected_labels["secondary_modifiers"]),
    )
    combined_score = stance_exact * primary_action_exact * modifier_f1
    metric = {
        "status": "ok",
        "reason_code": None,
        "stance_exact": stance_exact,
        "primary_action_exact": primary_action_exact,
        "secondary_modifiers_precision": precision,
        "secondary_modifiers_recall": recall,
        "secondary_modifiers_f1": modifier_f1,
        "combined_score": combined_score,
    }
    validate_behavioral_consistency_status(metric)
    return metric


def _majority_string(values: list[str]) -> str:
    counts = Counter(values)
    highest = max(counts.values())
    return sorted(value for value, count in counts.items() if count == highest)[0]


def leave_one_out_majority_behavior_labels(
    behaviors: list[dict[str, Any]],
    target_index: int,
) -> dict[str, Any]:
    if target_index < 0 or target_index >= len(behaviors):
        raise PersonaValidationError("target_index is out of range")
    others = [normalize_behavior_labels(item) for index, item in enumerate(behaviors) if index != target_index]
    if not others:
        return {
            "status": "not_applicable",
            "reason_code": "leave_one_out_requires_another_variant",
            "labels": None,
            "excluded_index": target_index,
            "source_count": 0,
        }

    threshold = (len(others) // 2) + 1
    modifier_counts: Counter[str] = Counter()
    for labels in others:
        modifier_counts.update(set(labels["secondary_modifiers"]))
    labels = {
        "stance": _majority_string([item["stance"] for item in others]),
        "primary_action": _majority_string([item["primary_action"] for item in others]),
        "secondary_modifiers": sorted(
            modifier for modifier, count in modifier_counts.items() if count >= threshold
        ),
    }
    return {
        "status": "ok",
        "reason_code": None,
        "labels": labels,
        "excluded_index": target_index,
        "source_count": len(others),
    }


def pairwise_behavior_agreement(behaviors: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_behavior_labels(item) for item in behaviors]
    if len(normalized) < 2:
        return {
            "status": "not_applicable",
            "reason_code": "pairwise_agreement_requires_two_variants",
            "pair_count": 0,
        }
    pair_metrics: list[dict[str, Any]] = []
    for left_index in range(len(normalized)):
        for right_index in range(left_index + 1, len(normalized)):
            pair_metrics.append(behavior_consistency_f1(normalized[left_index], normalized[right_index]))
    pair_count = len(pair_metrics)
    return {
        "status": "ok",
        "reason_code": None,
        "pair_count": pair_count,
        "stance_exact_mean": sum(metric["stance_exact"] for metric in pair_metrics) / pair_count,
        "primary_action_exact_mean": sum(metric["primary_action_exact"] for metric in pair_metrics) / pair_count,
        "secondary_modifiers_f1_mean": sum(
            metric["secondary_modifiers_f1"] for metric in pair_metrics
        )
        / pair_count,
        "combined_score_mean": sum(metric["combined_score"] for metric in pair_metrics) / pair_count,
    }


def validate_variants(row: dict[str, Any]) -> None:
    variants = row.get("variants")
    if not isinstance(variants, list):
        raise PersonaValidationError("variants must be an array")
    if len(variants) != len(REQUIRED_VARIANT_TYPES):
        raise PersonaValidationError("variants must contain exactly six executable variants")

    seen: list[str] = []
    variant_ids: list[str] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise PersonaValidationError(f"variants[{index}] must be an object")
        variant_id = variant.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id.strip():
            raise PersonaValidationError(f"variants[{index}].variant_id must be a non-empty string")
        variant_ids.append(variant_id)
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

    duplicate_ids = sorted({variant_id for variant_id in variant_ids if variant_ids.count(variant_id) > 1})
    if duplicate_ids:
        raise PersonaValidationError(f"duplicate variant_ids: {', '.join(duplicate_ids)}")


def validate_persona_row(row: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    schema = load_schema() if schema is None else schema
    validate_with_schema(row, schema)
    validate_source_metadata(row)
    validate_behavior_labels(row)
    validate_variants(row)


def validate_personas(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    schema = load_schema()
    persona_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        try:
            validate_persona_row(row, schema)
        except PersonaValidationError as exc:
            raise PersonaValidationError(f"row {index}: {exc}") from exc
        persona_id = row["persona_id"]
        if persona_id in persona_ids:
            raise PersonaValidationError(f"row {index}: duplicate persona_id: {persona_id}")
        persona_ids.add(persona_id)
    return rows


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def hash_file_bytes(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def token_count(text: str) -> int:
    return len(text.split())


def tokenize_for_mock_metrics(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def render_list(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "- [none]"
    return "\n".join(f"- {str(value)}" for value in values)


def render_traits(traits: dict[str, Any]) -> str:
    lines: list[str] = []
    for field in ("tone", "style"):
        values = traits.get(field) if isinstance(traits, dict) else None
        if not isinstance(values, list) or not values:
            lines.append(f"- {field}: [none]")
            continue
        for value in values:
            lines.append(f"- {field}: {value}")
    return "\n".join(lines) if lines else "- [none]"


def prompt_template_hash(prompt_template_version: str = PROMPT_TEMPLATE_VERSION) -> str:
    return sha256_text(PROMPT_TEMPLATE_SOURCE.format(
        prompt_template_version=prompt_template_version,
        persona_facts="{persona_facts}",
        persona_traits="{persona_traits}",
        persona_values="{persona_values}",
        forbidden_behaviors="{forbidden_behaviors}",
        user_task="{user_task}",
    ))


def render_prompt(
    persona_row: dict[str, Any],
    variant: dict[str, Any],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
) -> RenderedPrompt:
    persona_spec = persona_row.get("persona_spec", {})
    traits = persona_spec.get("traits", {}) if isinstance(persona_spec, dict) else {}
    prompt_text = PROMPT_TEMPLATE_SOURCE.format(
        prompt_template_version=prompt_template_version,
        persona_facts=render_list(persona_spec.get("facts")),
        persona_traits=render_traits(traits),
        persona_values=render_list(traits.get("values") if isinstance(traits, dict) else None),
        forbidden_behaviors=render_list(persona_spec.get("forbidden_behaviors")),
        user_task=str(variant.get("text", "")),
    )
    return RenderedPrompt(
        prompt_template_version=prompt_template_version,
        prompt_template_hash=prompt_template_hash(prompt_template_version),
        prompt_text=prompt_text,
        system_prompt=system_prompt,
        prompt_hash=sha256_text(prompt_text),
        system_prompt_hash=sha256_text(system_prompt),
    )


def serialize_persona_for_adherence(persona_row: dict[str, Any]) -> str:
    persona_spec = persona_row.get("persona_spec", {})
    traits = persona_spec.get("traits", {}) if isinstance(persona_spec, dict) else {}
    return "\n".join(
        [
            "Persona Domain:",
            str(persona_spec.get("domain", "")),
            "Persona Facts:",
            render_list(persona_spec.get("facts")),
            "Persona Traits:",
            render_traits(traits),
            "Persona Values:",
            render_list(traits.get("values") if isinstance(traits, dict) else None),
            "Forbidden Behaviors:",
            render_list(persona_spec.get("forbidden_behaviors")),
        ]
    )


def persona_adherence_hash(persona_row: dict[str, Any]) -> str:
    return sha256_text(serialize_persona_for_adherence(persona_row))


def sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    return sum(left[token] * right[token] for token in shared)


def persona_adherence_status_not_applicable(
    *,
    reason_code: str = "real_pa_backends_not_pinned",
    persona_serialization_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "reason_code": reason_code,
        "value": None,
        "mock_score": None,
        "semantic_similarity": None,
        "threshold": None,
        "pass_at_threshold": None,
        "fact_contradiction_rate": None,
        "persona_serialization_hash": persona_serialization_hash,
        "calibration_id": None,
        "calibration_hash": None,
        "embedding_model_revision": "not_pinned",
        "nli_or_judge_model_revision": "not_pinned",
        "mock_only": False,
    }


def score_persona_adherence_mock(
    persona_row: dict[str, Any],
    response_text: str,
    *,
    embedding_backend: BaseEmbeddingBackend | None = None,
    nli_judge: BaseNLIJudge | None = None,
    calibration_id: str | None = None,
    calibration_hash: str | None = None,
) -> dict[str, Any]:
    embedding_backend = embedding_backend or MockEmbeddingBackend()
    nli_judge = nli_judge or MockNLIJudge()
    serialized = serialize_persona_for_adherence(persona_row)
    persona_hash = sha256_text(serialized)
    semantic_similarity = sparse_cosine(embedding_backend.embed(serialized), embedding_backend.embed(response_text))
    facts = persona_row.get("persona_spec", {}).get("facts", [])
    contradiction_count = 0
    fact_count = 0
    if isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                fact_count += 1
                if nli_judge.judge(response_text, fact) == "contradiction":
                    contradiction_count += 1
    contradiction_rate = contradiction_count / fact_count if fact_count else 0.0
    mock_score = max(0.0, min(1.0, semantic_similarity * (1.0 - contradiction_rate)))
    metric = {
        "status": "mock_only",
        "reason_code": "mock_backends_not_real_pa",
        "value": None,
        "mock_score": mock_score,
        "semantic_similarity": semantic_similarity,
        "threshold": None,
        "pass_at_threshold": None,
        "fact_contradiction_rate": contradiction_rate,
        "persona_serialization_hash": persona_hash,
        "calibration_id": calibration_id,
        "calibration_hash": calibration_hash,
        "embedding_model_revision": embedding_backend.model_revision,
        "nli_or_judge_model_revision": nli_judge.model_revision,
        "mock_only": True,
    }
    validate_persona_adherence_status(metric)
    return metric


def persona_adherence_calibration_fixtures(
    persona_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(persona_rows) < 2:
        raise PersonaValidationError("persona adherence calibration fixtures require at least two rows")
    fixtures: list[dict[str, Any]] = []
    for index, row in enumerate(persona_rows):
        swapped = persona_rows[(index + 1) % len(persona_rows)]
        own_facts = row.get("persona_spec", {}).get("facts", [])
        swapped_facts = swapped.get("persona_spec", {}).get("facts", [])
        fixtures.append(
            {
                "kind": "positive",
                "persona_id": row["persona_id"],
                "persona_row": row,
                "response_text": " ".join(str(fact) for fact in own_facts[:2]),
            }
        )
        fixtures.append(
            {
                "kind": "persona_swapped_negative",
                "persona_id": row["persona_id"],
                "negative_persona_id": swapped["persona_id"],
                "persona_row": row,
                "response_text": " ".join(str(fact) for fact in swapped_facts[:2]),
            }
        )
    return fixtures


def calibrate_persona_adherence_mock(
    fixtures: list[dict[str, Any]],
    *,
    embedding_backend: BaseEmbeddingBackend | None = None,
    nli_judge: BaseNLIJudge | None = None,
) -> dict[str, Any]:
    if not fixtures:
        raise PersonaValidationError("persona adherence calibration fixtures cannot be empty")
    scores_by_kind: dict[str, list[float]] = {"positive": [], "persona_swapped_negative": []}
    for fixture in fixtures:
        status = score_persona_adherence_mock(
            fixture["persona_row"],
            fixture["response_text"],
            embedding_backend=embedding_backend,
            nli_judge=nli_judge,
        )
        kind = fixture["kind"]
        if kind in scores_by_kind:
            scores_by_kind[kind].append(status["mock_score"])
    if not scores_by_kind["positive"] or not scores_by_kind["persona_swapped_negative"]:
        raise PersonaValidationError("calibration requires positive and persona-swapped negative fixtures")
    positive_mean = sum(scores_by_kind["positive"]) / len(scores_by_kind["positive"])
    negative_mean = sum(scores_by_kind["persona_swapped_negative"]) / len(
        scores_by_kind["persona_swapped_negative"]
    )
    payload = {
        "status": "mock_only",
        "reason_code": "mock_calibration_not_real_pa_threshold",
        "calibration_id": "mock_pa_calibration_sprint2",
        "positive_mean": positive_mean,
        "persona_swapped_negative_mean": negative_mean,
        "separates_positive_from_negative": positive_mean > negative_mean,
        "threshold": None,
    }
    payload["calibration_hash"] = sha256_json(payload)
    return payload


def git_value(args: list[str], fallback: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback
    return completed.stdout.strip() or fallback


def current_commit() -> str:
    return git_value(["rev-parse", "HEAD"], "not_available")


def is_dirty_worktree() -> bool:
    return bool(git_value(["status", "--porcelain"], ""))


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_run_manifest(
    *,
    persona_path: str | Path,
    model_base: str,
    model_tuned: str,
    seeds: list[int],
    run_id: str | None = None,
    timestamp_utc: str | None = None,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
    chat_template_hash: str = "not_available",
    tokenizer_name: str = "not_available",
    tokenizer_hash: str = "not_available",
    model_base_revision_or_hash: str = "not_available",
    model_tuned_revision_or_hash: str = "not_available",
    adapter: str = "mock",
    provider_or_endpoint: str = "local_mock",
    serving_stack: str = "mock",
    serving_stack_version: str = "sprint1",
    scoring_capability: str = "none",
    gpu_cuda_driver: str = "not_available",
    decoding_params: dict[str, Any] | None = None,
    stop_sequences: list[str] | None = None,
    extractor_version: str = "not_available",
    embedding_model_revision: str = "not_available",
    nli_or_judge_model_revision: str = "not_available",
    code_commit: str | None = None,
    dirty_worktree: bool | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id or f"run-{utc_now()}",
        "timestamp_utc": timestamp_utc or utc_now(),
        "code_commit": code_commit if code_commit is not None else current_commit(),
        "dirty_worktree": dirty_worktree if dirty_worktree is not None else is_dirty_worktree(),
        "persona_jsonl_hash": hash_file_bytes(persona_path),
        "prompt_template_version": prompt_template_version,
        "prompt_template_hash": prompt_template_hash(prompt_template_version),
        "chat_template_hash": chat_template_hash,
        "tokenizer_name": tokenizer_name,
        "tokenizer_hash": tokenizer_hash,
        "model_base": model_base,
        "model_tuned": model_tuned,
        "model_base_revision_or_hash": model_base_revision_or_hash,
        "model_tuned_revision_or_hash": model_tuned_revision_or_hash,
        "adapter": adapter,
        "provider_or_endpoint": provider_or_endpoint,
        "serving_stack": serving_stack,
        "serving_stack_version": serving_stack_version,
        "scoring_capability": scoring_capability,
        "gpu_cuda_driver": gpu_cuda_driver,
        "decoding_params": decoding_params or {},
        "seeds": seeds,
        "stop_sequences": stop_sequences or [],
        "extractor_version": extractor_version,
        "embedding_model_revision": embedding_model_revision,
        "nli_or_judge_model_revision": nli_or_judge_model_revision,
    }
    validate_run_manifest(manifest)
    return manifest


def validate_token_kl_status(status: dict[str, Any]) -> None:
    if not isinstance(status, dict):
        raise PersonaValidationError("token_kl must be an object")
    missing = [field for field in TOKEN_KL_STATUS_FIELDS if field not in status]
    if missing:
        raise PersonaValidationError(f"token_kl missing fields: {', '.join(missing)}")
    if status["status"] not in TOKEN_KL_STATUS_VALUES:
        raise PersonaValidationError(f"unsupported token_kl.status: {status['status']!r}")
    if status["scoring_path"] not in TOKEN_KL_SCORING_PATHS:
        raise PersonaValidationError(
            f"unsupported token_kl.scoring_path: {status['scoring_path']!r}"
        )
    if not isinstance(status["diagnostic_only"], bool):
        raise PersonaValidationError("token_kl.diagnostic_only must be a boolean")
    if status["k"] is not None and (
        not isinstance(status["k"], int) or isinstance(status["k"], bool) or status["k"] <= 0
    ):
        raise PersonaValidationError("token_kl.k must be null or a positive integer")
    if status["endpoint_cap"] is not None and (
        not isinstance(status["endpoint_cap"], int)
        or isinstance(status["endpoint_cap"], bool)
        or status["endpoint_cap"] <= 0
    ):
        raise PersonaValidationError("token_kl.endpoint_cap must be null or a positive integer")

    is_number = isinstance(status["value"], (int, float)) and not isinstance(status["value"], bool)
    if status["value"] is not None and not is_number:
        raise PersonaValidationError("token_kl.value must be null or numeric")

    if status["status"] != "ok" and not status["reason_code"]:
        raise PersonaValidationError("token_kl.reason_code is required unless status is ok")
    if status["status"] == "ok":
        if not is_number:
            raise PersonaValidationError("token_kl.value is required when status is ok")
        if status["reason_code"] is not None:
            raise PersonaValidationError("token_kl.reason_code must be null when status is ok")
        if status["scoring_path"] == "none":
            raise PersonaValidationError("token_kl.scoring_path cannot be none when status is ok")
        if status["diagnostic_only"]:
            raise PersonaValidationError("token_kl.diagnostic_only must be false when status is ok")
        if status["k"] is None:
            raise PersonaValidationError("token_kl.k is required when status is ok")
        for field in (
            "fixed_continuation_id",
            "fixed_continuation_hash",
        ):
            if not isinstance(status[field], str) or not status[field].strip():
                raise PersonaValidationError(f"token_kl.{field} is required when status is ok")
        for field in (
            "tokenizer_hash_match",
            "vocabulary_match",
            "chat_template_hash_match",
        ):
            if status[field] is not True:
                raise PersonaValidationError(f"token_kl.{field} must be true when status is ok")
        if status["endpoint_cap"] is not None and status["endpoint_cap"] < status["k"]:
            raise PersonaValidationError(
                "token_kl.endpoint_cap below k cannot be canonical ok output"
            )
    elif status["status"] == "diagnostic_only":
        if not status["diagnostic_only"]:
            raise PersonaValidationError(
                "token_kl.diagnostic_only must be true when status is diagnostic_only"
            )
        if status["scoring_path"] == "none":
            raise PersonaValidationError(
                "token_kl.scoring_path cannot be none when status is diagnostic_only"
            )
        if status["k"] is None:
            raise PersonaValidationError("token_kl.k is required when status is diagnostic_only")
        if status["endpoint_cap"] is None:
            raise PersonaValidationError(
                "token_kl.endpoint_cap is required for diagnostic_only endpoint output"
            )
    else:
        if status["value"] is not None:
            raise PersonaValidationError("token_kl.value must be null unless status is ok")
        if status["diagnostic_only"]:
            raise PersonaValidationError(
                "token_kl.diagnostic_only must be false unless status is diagnostic_only"
            )


def token_kl_not_applicable(
    *,
    reason_code: str = "aligned_scoring_unavailable",
    scoring_path: str = "none",
    fixed_continuation_id: str | None = None,
    fixed_continuation_hash: str | None = None,
    tokenizer_hash_match: bool | None = None,
    vocabulary_match: bool | None = None,
    chat_template_hash_match: bool | None = None,
    k: int | None = None,
    endpoint_cap: int | None = None,
) -> dict[str, Any]:
    return ScoreContinuationResult(
        status="not_applicable",
        value=None,
        reason_code=reason_code,
        scoring_path=scoring_path,
        fixed_continuation_id=fixed_continuation_id,
        fixed_continuation_hash=fixed_continuation_hash,
        tokenizer_hash_match=tokenizer_hash_match,
        vocabulary_match=vocabulary_match,
        chat_template_hash_match=chat_template_hash_match,
        k=k,
        endpoint_cap=endpoint_cap,
        diagnostic_only=False,
    ).to_token_kl_status()


def endpoint_capped_token_kl_status(
    *,
    fixed_continuation_id: str,
    fixed_continuation: str,
    k: int,
    endpoint_cap: int,
) -> dict[str, Any]:
    return ScoreContinuationResult(
        status="diagnostic_only",
        value=None,
        reason_code="endpoint_top_logprobs_cap_below_k",
        scoring_path="hosted_top_logprobs",
        fixed_continuation_id=fixed_continuation_id,
        fixed_continuation_hash=sha256_text(fixed_continuation),
        tokenizer_hash_match=True,
        vocabulary_match=True,
        chat_template_hash_match=True,
        k=k,
        endpoint_cap=endpoint_cap,
        diagnostic_only=True,
    ).to_token_kl_status()


def normalize_topk_distribution(
    distribution: dict[str, float],
    *,
    input_type: str = "probability",
) -> dict[str, float]:
    if not isinstance(distribution, dict) or not distribution:
        raise PersonaValidationError("top-k distribution must be a non-empty object")
    values: dict[str, float] = {}
    if input_type not in {"probability", "logprob"}:
        raise PersonaValidationError("input_type must be probability or logprob")
    for token, value in distribution.items():
        if not isinstance(token, str) or not token:
            raise PersonaValidationError("top-k distribution token keys must be non-empty strings")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise PersonaValidationError("top-k distribution values must be finite numbers")
        values[token] = math.exp(float(value)) if input_type == "logprob" else float(value)
        if values[token] < 0.0:
            raise PersonaValidationError("probability values must be non-negative")
    total = sum(values.values())
    if total <= 0.0:
        raise PersonaValidationError("top-k distribution probability mass must be positive")
    return {token: probability / total for token, probability in sorted(values.items())}


def kl_divergence(
    left: dict[str, float],
    right: dict[str, float],
    *,
    epsilon: float = 1e-12,
) -> float:
    if epsilon <= 0.0:
        raise PersonaValidationError("epsilon must be positive")
    support = sorted(set(left) | set(right))
    left_smoothed = {token: left.get(token, 0.0) + epsilon for token in support}
    right_smoothed = {token: right.get(token, 0.0) + epsilon for token in support}
    left_total = sum(left_smoothed.values())
    right_total = sum(right_smoothed.values())
    return sum(
        (left_smoothed[token] / left_total)
        * math.log((left_smoothed[token] / left_total) / (right_smoothed[token] / right_total))
        for token in support
    )


def mean_token_kl(
    base_steps: list[dict[str, float]],
    tuned_steps: list[dict[str, float]],
    *,
    input_type: str = "probability",
    epsilon: float = 1e-12,
) -> float:
    if len(base_steps) != len(tuned_steps) or not base_steps:
        raise PersonaValidationError("Token-KL requires aligned non-empty distribution steps")
    divergences: list[float] = []
    for base_step, tuned_step in zip(base_steps, tuned_steps, strict=True):
        base_distribution = normalize_topk_distribution(base_step, input_type=input_type)
        tuned_distribution = normalize_topk_distribution(tuned_step, input_type=input_type)
        divergences.append(kl_divergence(base_distribution, tuned_distribution, epsilon=epsilon))
    return sum(divergences) / len(divergences)


def token_kl_from_aligned_topk(
    base_steps: list[dict[str, float]],
    tuned_steps: list[dict[str, float]],
    *,
    fixed_continuation_id: str | None,
    fixed_continuation: str | None,
    tokenizer_hash_match: bool,
    vocabulary_match: bool,
    chat_template_hash_match: bool,
    k: int = 50,
    endpoint_cap: int | None = None,
    scoring_path: str = "local_forward",
    input_type: str = "probability",
) -> dict[str, Any]:
    fixed_continuation_hash = sha256_text(fixed_continuation) if fixed_continuation else None
    if not fixed_continuation_id or not fixed_continuation:
        return token_kl_not_applicable(
            reason_code="missing_fixed_continuation",
            fixed_continuation_id=fixed_continuation_id,
            fixed_continuation_hash=fixed_continuation_hash,
            tokenizer_hash_match=tokenizer_hash_match,
            vocabulary_match=vocabulary_match,
            chat_template_hash_match=chat_template_hash_match,
            k=k,
            endpoint_cap=endpoint_cap,
        )
    if not tokenizer_hash_match:
        return token_kl_not_applicable(
            reason_code="tokenizer_hash_mismatch",
            fixed_continuation_id=fixed_continuation_id,
            fixed_continuation_hash=fixed_continuation_hash,
            tokenizer_hash_match=False,
            vocabulary_match=vocabulary_match,
            chat_template_hash_match=chat_template_hash_match,
            k=k,
            endpoint_cap=endpoint_cap,
        )
    if not vocabulary_match:
        return token_kl_not_applicable(
            reason_code="vocabulary_mismatch",
            fixed_continuation_id=fixed_continuation_id,
            fixed_continuation_hash=fixed_continuation_hash,
            tokenizer_hash_match=tokenizer_hash_match,
            vocabulary_match=False,
            chat_template_hash_match=chat_template_hash_match,
            k=k,
            endpoint_cap=endpoint_cap,
        )
    if not chat_template_hash_match:
        return token_kl_not_applicable(
            reason_code="chat_template_hash_mismatch",
            fixed_continuation_id=fixed_continuation_id,
            fixed_continuation_hash=fixed_continuation_hash,
            tokenizer_hash_match=tokenizer_hash_match,
            vocabulary_match=vocabulary_match,
            chat_template_hash_match=False,
            k=k,
            endpoint_cap=endpoint_cap,
        )
    if scoring_path == "hosted_top_logprobs" and endpoint_cap is not None and endpoint_cap < k:
        return endpoint_capped_token_kl_status(
            fixed_continuation_id=fixed_continuation_id,
            fixed_continuation=fixed_continuation,
            k=k,
            endpoint_cap=endpoint_cap,
        )
    if scoring_path not in {"local_forward", "vllm_prompt_logprobs", "one_token_loop"}:
        return token_kl_not_applicable(
            reason_code="aligned_scoring_unavailable",
            scoring_path="none",
            fixed_continuation_id=fixed_continuation_id,
            fixed_continuation_hash=fixed_continuation_hash,
            tokenizer_hash_match=tokenizer_hash_match,
            vocabulary_match=vocabulary_match,
            chat_template_hash_match=chat_template_hash_match,
            k=k,
            endpoint_cap=endpoint_cap,
        )

    value = mean_token_kl(base_steps, tuned_steps, input_type=input_type)
    return ScoreContinuationResult(
        status="ok",
        value=value,
        reason_code=None,
        scoring_path=scoring_path,
        fixed_continuation_id=fixed_continuation_id,
        fixed_continuation_hash=fixed_continuation_hash,
        tokenizer_hash_match=True,
        vocabulary_match=True,
        chat_template_hash_match=True,
        k=k,
        endpoint_cap=endpoint_cap,
        diagnostic_only=False,
    ).to_token_kl_status()


def canonical_token_kl_from_generation_outputs(
    base_generation: GenerationResult,
    tuned_generation: GenerationResult,
) -> float:
    del base_generation, tuned_generation
    raise TokenKLUnavailableError(
        "canonical Token-KL requires aligned continuation scoring; free-running "
        "generation outputs are not valid inputs"
    )


def build_cache_key_payload(**values: Any) -> dict[str, Any]:
    missing = [field for field in CACHE_KEY_FIELDS if field not in values]
    if missing:
        raise PersonaValidationError(f"missing cache key fields: {', '.join(missing)}")
    return {field: values[field] for field in CACHE_KEY_FIELDS}


def cache_key_from_payload(payload: dict[str, Any]) -> str:
    canonical_payload = build_cache_key_payload(**payload)
    return "sha256:" + hashlib.sha256(canonical_json(canonical_payload).encode("utf-8")).hexdigest()


def build_generation_request(
    *,
    rendered: RenderedPrompt,
    run_id: str,
    persona_id: str,
    variant_id: str,
    variant_type: str,
    model_alias: str,
    model_id: str,
    seed: int,
    model_revision_or_hash: str = "not_available",
    tokenizer_name: str = "not_available",
    tokenizer_hash: str = "not_available",
    chat_template_hash: str = "not_available",
    decoding_params: dict[str, Any] | None = None,
    stop_sequences: list[str] | None = None,
    adapter: BaseAdapter | None = None,
) -> GenerationRequest:
    selected_adapter = adapter or MockAdapter()
    return GenerationRequest(
        run_id=run_id,
        persona_id=persona_id,
        variant_id=variant_id,
        variant_type=variant_type,
        model_alias=model_alias,
        model_id=model_id,
        model_revision_or_hash=model_revision_or_hash,
        tokenizer_name=tokenizer_name,
        tokenizer_hash=tokenizer_hash,
        chat_template_hash=chat_template_hash,
        prompt_template_version=rendered.prompt_template_version,
        prompt_template_hash=rendered.prompt_template_hash,
        prompt_text=rendered.prompt_text,
        system_prompt=rendered.system_prompt,
        seed=seed,
        decoding_params=decoding_params or {},
        stop_sequences=stop_sequences or [],
        prompt_hash=rendered.prompt_hash,
        system_prompt_hash=rendered.system_prompt_hash,
        adapter=selected_adapter.adapter_name,
        provider_or_endpoint=selected_adapter.provider_or_endpoint,
        serving_stack=selected_adapter.serving_stack,
        serving_stack_version=selected_adapter.serving_stack_version,
        scoring_capability=selected_adapter.scoring_capability,
    )


def build_score_continuation_request(
    *,
    rendered: RenderedPrompt,
    run_id: str,
    persona_id: str,
    variant_id: str,
    variant_type: str,
    model_alias: str,
    model_id: str,
    fixed_continuation: str,
    fixed_continuation_id: str,
    model_revision_or_hash: str = "not_available",
    tokenizer_name: str = "not_available",
    tokenizer_hash: str = "not_available",
    chat_template_hash: str = "not_available",
    seed: int = 0,
    decoding_params: dict[str, Any] | None = None,
    stop_sequences: list[str] | None = None,
    k: int = 50,
    adapter: BaseAdapter | None = None,
) -> ScoreContinuationRequest:
    selected_adapter = adapter or MockAdapter()
    return ScoreContinuationRequest(
        run_id=run_id,
        persona_id=persona_id,
        variant_id=variant_id,
        variant_type=variant_type,
        model_alias=model_alias,
        model_id=model_id,
        model_revision_or_hash=model_revision_or_hash,
        tokenizer_name=tokenizer_name,
        tokenizer_hash=tokenizer_hash,
        chat_template_hash=chat_template_hash,
        prompt_template_version=rendered.prompt_template_version,
        prompt_template_hash=rendered.prompt_template_hash,
        prompt_text=rendered.prompt_text,
        system_prompt=rendered.system_prompt,
        seed=seed,
        decoding_params=decoding_params or {},
        stop_sequences=stop_sequences or [],
        fixed_continuation=fixed_continuation,
        fixed_continuation_id=fixed_continuation_id,
        k=k,
        prompt_hash=rendered.prompt_hash,
        system_prompt_hash=rendered.system_prompt_hash,
        adapter=selected_adapter.adapter_name,
        provider_or_endpoint=selected_adapter.provider_or_endpoint,
        serving_stack=selected_adapter.serving_stack,
        serving_stack_version=selected_adapter.serving_stack_version,
        scoring_capability=selected_adapter.scoring_capability,
    )


def validate_generation_request_context(
    *,
    request: GenerationRequest,
    run_id: str,
    persona_id: str,
    variant_id: str,
    variant_type: str,
    seed: int,
    rendered: RenderedPrompt,
) -> None:
    expected = {
        "run_id": run_id,
        "persona_id": persona_id,
        "variant_id": variant_id,
        "variant_type": variant_type,
        "seed": seed,
        "prompt_hash": rendered.prompt_hash,
        "system_prompt_hash": rendered.system_prompt_hash,
        "prompt_template_version": rendered.prompt_template_version,
        "prompt_template_hash": rendered.prompt_template_hash,
    }
    for field, value in expected.items():
        if getattr(request, field) != value:
            raise PersonaValidationError(
                f"generation request {request.model_alias}.{field} does not match result row"
            )


def build_result_row(
    *,
    run_id: str,
    persona_row: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    rendered: RenderedPrompt,
    model_pair: dict[str, Any],
    base_request: GenerationRequest,
    base_result: GenerationResult,
    tuned_request: GenerationRequest,
    tuned_result: GenerationResult,
    score_result: ScoreContinuationResult,
    behavior_tags: dict[str, Any] | None = None,
    persona_adherence_metric: dict[str, Any] | None = None,
    behavioral_consistency_metric: dict[str, Any] | None = None,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    persona_id = persona_row["persona_id"]
    variant_id = variant["variant_id"]
    variant_type = variant["type"]
    for request in (base_request, tuned_request):
        validate_generation_request_context(
            request=request,
            run_id=run_id,
            persona_id=persona_id,
            variant_id=variant_id,
            variant_type=variant_type,
            seed=seed,
            rendered=rendered,
        )
    token_kl_status = score_result.to_token_kl_status()
    result_row = {
        "run_id": run_id,
        "persona_id": persona_id,
        "variant_id": variant_id,
        "variant_type": variant_type,
        "seed": seed,
        "prompt_text": rendered.prompt_text,
        "system_prompt": rendered.system_prompt,
        "prompt_hash": rendered.prompt_hash,
        "system_prompt_hash": rendered.system_prompt_hash,
        "prompt_template_version": rendered.prompt_template_version,
        "prompt_template_hash": rendered.prompt_template_hash,
        "model_pair": model_pair,
        "base": base_result.to_model_output(base_request),
        "tuned": tuned_result.to_model_output(tuned_request),
        "behavior_tags": behavior_tags or behavior_tags_not_run(),
        "metrics": {
            "token_kl": token_kl_status,
            "persona_adherence": persona_adherence_metric or {"status": "not_run"},
            "behavioral_consistency_f1": behavioral_consistency_metric or {"status": "not_run"},
        },
        "flags": flags or [],
    }
    validate_result_row(result_row)
    return result_row


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
    has_base = bool(args.model_base)
    has_tuned = bool(args.model_tuned)
    if has_base or has_tuned:
        if not (has_base and has_tuned):
            raise PersonaValidationError(
                "plan requires both --model-base and --model-tuned, or explicit --model-count"
            )
        return 2
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
