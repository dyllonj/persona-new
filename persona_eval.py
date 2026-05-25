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
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from model_matrix import load_model_matrix, validate_model_matrix

try:
    import jsonschema
except ImportError:  # pragma: no cover - covered only in stripped environments.
    jsonschema = None


REPO_ROOT = Path(__file__).resolve().parent
PERSONA_SCHEMA_PATH = REPO_ROOT / "schemas" / "persona_item.schema.json"
BEHAVIOR_TAGS_SCHEMA_PATH = REPO_ROOT / "schemas" / "behavior_tags.schema.json"
RUN_MANIFEST_SCHEMA_PATH = REPO_ROOT / "schemas" / "run_manifest.schema.json"
RESULT_ROW_SCHEMA_PATH = REPO_ROOT / "schemas" / "result_row.schema.json"
REVIEW_MANIFEST_SCHEMA_PATH = REPO_ROOT / "schemas" / "review_manifest.schema.json"
DEFAULT_SAMPLE_REVIEW_MANIFEST_PATH = REPO_ROOT / "reviews" / "personas.sample.review.jsonl"

PROMPT_TEMPLATE_VERSION = "v1"
EVALUATOR_VERSION = "sprint3"
METRIC_VERSION = "sprint2"
EXTRACTOR_VERSION = "sprint2"
STAGED_RUN_MAX_PERSONAS = 20
RUN_VARIANTS_PER_PERSONA = 6
RUN_MODEL_COUNT = 2
SMOKE_RUN_PERSONA_COUNT = 20
SMOKE_RUN_SEED_COUNT = 1
SMOKE_RUN_CALL_COUNT = 240
DEV_RUN_PERSONA_COUNT = 50
DEV_RUN_SEED_COUNT = 2
DEV_RUN_CALL_COUNT = 1200
FULL_RUN_PERSONA_COUNT = 200
FULL_RUN_SEED_COUNT = 2
FULL_RUN_CALL_COUNT = 4800
RUN_STAGE_PERSONA_CAPS = {
    "auto": STAGED_RUN_MAX_PERSONAS,
    "smoke": SMOKE_RUN_PERSONA_COUNT,
    "dev": DEV_RUN_PERSONA_COUNT,
    "full": FULL_RUN_PERSONA_COUNT,
}
PASSING_EVIDENCE_STATUSES = {"pass", "passed", "ok", "success", "completed", "approved"}
RUN_EVIDENCE_REQUIRED_ARTIFACT_FIELDS = (
    "manifest_path",
    "results_path",
    "aggregate_report_path",
)
RUN_EVIDENCE_ARTIFACT_HASH_FIELDS = {
    "manifest_path": "manifest_hash",
    "results_path": "results_hash",
    "aggregate_report_path": "aggregate_report_hash",
}
DEV_RUN_APPROVAL_TYPES = {"dev_run_approval", "dev_run_override"}
FULL_REVIEW_GATE_FIELDS = (
    "semantic_equivalence_status",
    "nli_equivalence_status",
    "contradiction_status",
    "safety_review_status",
    "gold_label_review_status",
)
PASSING_FULL_REVIEW_GATE_STATUSES = {"passed", "manual_pass"}
PLACEHOLDER_RUNTIME_VALUES = {"", "not_available", "todo", "tbd", "placeholder", "unknown"}
FULL_RUN_REQUIRED_RUNTIME_ARGS = (
    "model_base",
    "model_tuned",
    "model_base_revision_or_hash",
    "model_tuned_revision_or_hash",
    "tokenizer_name",
    "tokenizer_hash",
    "chat_template_hash",
    "serving_stack_version",
)
FULL_RUN_REQUIRED_MANIFEST_FIELDS = (
    "code_commit",
    "dirty_worktree",
    "persona_jsonl_hash",
    "prompt_template_hash",
    "model_base",
    "model_tuned",
    "model_base_revision_or_hash",
    "model_tuned_revision_or_hash",
    "tokenizer_name",
    "tokenizer_hash",
    "chat_template_hash",
    "serving_stack_version",
)
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
TOKEN_KL_APPLICABILITY_VALUES = {"canonical_possible", "diagnostic_only", "not_applicable"}
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
REAL_HTTP_ADAPTERS = {"vllm", "openai-compatible"}
EXPLICIT_REAL_RUN_METADATA_FIELDS = (
    "model_base_revision_or_hash",
    "model_tuned_revision_or_hash",
    "tokenizer_name",
    "tokenizer_hash",
    "chat_template_hash",
    "serving_stack_version",
    "promotion_manifest_path",
)


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


@dataclass(frozen=True)
class AdapterPair:
    """Generation adapters for the base and tuned model calls in one run."""

    base: BaseAdapter
    tuned: BaseAdapter


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


class VLLMOpenAIAdapter(BaseAdapter):
    """Minimal OpenAI-compatible HTTP adapter for explicit tiny vLLM smoke runs."""

    scoring_capability = "none"

    def __init__(
        self,
        *,
        base_url: str,
        adapter_name: str = "vllm",
        api_key_env: str | None = None,
        serving_stack: str = "vllm",
        serving_stack_version: str = "not_available",
        timeout_s: float = 60.0,
    ) -> None:
        if not base_url or not base_url.strip():
            raise PersonaValidationError("--base-url is required for vllm/openai-compatible adapters")
        self.adapter_name = adapter_name
        self.provider_or_endpoint = base_url.rstrip("/")
        self.serving_stack = serving_stack
        self.serving_stack_version = serving_stack_version
        self.timeout_s = timeout_s
        self.api_key_env = api_key_env
        self.api_key: str | None = None
        if api_key_env:
            self.api_key = os.environ.get(api_key_env)
            if not self.api_key:
                raise PersonaValidationError(f"--api-key-env {api_key_env!r} is not set")

    def _chat_completions_url(self) -> str:
        if self.provider_or_endpoint.endswith("/v1"):
            return f"{self.provider_or_endpoint}/chat/completions"
        return f"{self.provider_or_endpoint}/v1/chat/completions"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt_text},
            ],
            "temperature": request.decoding_params.get("temperature", 0.0),
            "max_tokens": request.decoding_params.get("max_tokens", 140),
            "seed": request.seed,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            self._chat_completions_url(),
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        start = time.monotonic()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise PersonaValidationError(f"{self.adapter_name} HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise PersonaValidationError(f"{self.adapter_name} request failed: {exc.reason}") from exc
        latency_s = time.monotonic() - start

        try:
            raw_response = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PersonaValidationError(f"{self.adapter_name} returned invalid JSON") from exc
        choices = raw_response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise PersonaValidationError(f"{self.adapter_name} response missing choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise PersonaValidationError(f"{self.adapter_name} response choice must be an object")
        message = first_choice.get("message")
        if isinstance(message, dict):
            response_text = message.get("content") or ""
        else:
            response_text = first_choice.get("text") or ""
        if not isinstance(response_text, str):
            raise PersonaValidationError(f"{self.adapter_name} response text must be a string")

        usage_payload = raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else {}
        prompt_tokens = int(usage_payload.get("prompt_tokens", token_count(request.system_prompt) + token_count(request.prompt_text)))
        completion_tokens = int(usage_payload.get("completion_tokens", token_count(response_text)))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage_payload.get("total_tokens", prompt_tokens + completion_tokens)),
        }
        finish_reason = first_choice.get("finish_reason") or "unknown"
        return GenerationResult(
            status="ok",
            reason_code=None,
            response_text=response_text,
            stop_reason=str(finish_reason),
            truncation_flag=finish_reason == "length",
            usage=usage,
            latency_s=latency_s,
            raw_response=raw_response,
        )

    def score_continuation(self, request: ScoreContinuationRequest) -> ScoreContinuationResult:
        return ScoreContinuationResult(
            status="not_applicable",
            value=None,
            reason_code="aligned_scoring_unavailable",
            scoring_path="none",
            fixed_continuation_id=request.fixed_continuation_id,
            fixed_continuation_hash=sha256_text(request.fixed_continuation),
            tokenizer_hash_match=None,
            vocabulary_match=None,
            chat_template_hash_match=None,
            k=request.k,
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


def _token_kl_applicability_from_result_row(result_row: dict[str, Any]) -> str | None:
    model_pair = result_row.get("model_pair")
    if not isinstance(model_pair, dict):
        return None
    direct = model_pair.get("token_kl_applicability")
    if isinstance(direct, str):
        return direct
    metric_applicability = model_pair.get("metric_applicability")
    if isinstance(metric_applicability, dict):
        value = metric_applicability.get("token_kl_applicability", metric_applicability.get("token_kl"))
        if isinstance(value, str):
            return value
    return None


def validate_result_row_token_kl_applicability(result_row: dict[str, Any]) -> None:
    applicability = _token_kl_applicability_from_result_row(result_row)
    if applicability is None:
        return
    if applicability not in TOKEN_KL_APPLICABILITY_VALUES:
        raise PersonaValidationError(f"unsupported model_pair.token_kl_applicability: {applicability!r}")
    token_kl = result_row.get("metrics", {}).get("token_kl", {})
    if not isinstance(token_kl, dict):
        raise PersonaValidationError("metrics.token_kl must be an object")
    status = token_kl.get("status")
    if applicability in {"not_applicable", "diagnostic_only"} and status == "ok":
        raise PersonaValidationError(
            f"token_kl.status=ok conflicts with model_pair.token_kl_applicability={applicability}"
        )


def validate_result_row(result_row: dict[str, Any]) -> None:
    validate_schema_file(result_row, RESULT_ROW_SCHEMA_PATH)
    validate_behavior_tags(result_row.get("behavior_tags", {}))
    metrics = result_row.get("metrics", {})
    if not isinstance(metrics, dict):
        raise PersonaValidationError("result_row.metrics must be an object")
    validate_token_kl_status(metrics.get("token_kl", {}))
    validate_result_row_token_kl_applicability(result_row)
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
    provider_or_endpoint_base: str | None = None,
    provider_or_endpoint_tuned: str | None = None,
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
    promotion_manifest_path: str | Path | None = None,
    review_manifest_path: str | Path | None = None,
    raw_request_response_logging_status: str = "enabled",
) -> dict[str, Any]:
    persona_path_obj = Path(persona_path)
    promotion_path = Path(promotion_manifest_path) if promotion_manifest_path else None
    review_path = Path(review_manifest_path) if review_manifest_path else None
    base_endpoint = provider_or_endpoint_base or provider_or_endpoint
    tuned_endpoint = provider_or_endpoint_tuned or provider_or_endpoint
    manifest = {
        "run_id": run_id or f"run-{utc_now()}",
        "timestamp_utc": timestamp_utc or utc_now(),
        "code_commit": code_commit if code_commit is not None else current_commit(),
        "dirty_worktree": dirty_worktree if dirty_worktree is not None else is_dirty_worktree(),
        "persona_path": str(persona_path_obj),
        "persona_jsonl_hash": hash_file_bytes(persona_path_obj),
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
        "runtime_metadata_policy": {
            "tokenizer_name": "shared_cli_value",
            "tokenizer_hash": "shared_cli_value",
            "chat_template_hash": "shared_cli_value",
            "note": (
                "Current run CLI records tokenizer/chat-template metadata as shared values. "
                "Use a separate model-matrix/runtime contract before treating them as independently verified per endpoint."
            ),
        },
        "model_endpoints": {
            "base": {
                "model_id": model_base,
                "model_revision_or_hash": model_base_revision_or_hash,
                "provider_or_endpoint": base_endpoint,
                "tokenizer_name": tokenizer_name,
                "tokenizer_hash": tokenizer_hash,
                "chat_template_hash": chat_template_hash,
            },
            "tuned": {
                "model_id": model_tuned,
                "model_revision_or_hash": model_tuned_revision_or_hash,
                "provider_or_endpoint": tuned_endpoint,
                "tokenizer_name": tokenizer_name,
                "tokenizer_hash": tokenizer_hash,
                "chat_template_hash": chat_template_hash,
            },
        },
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
        "review_manifest_path": str(review_path) if review_path else None,
        "review_manifest_hash": hash_file_bytes(review_path) if review_path else None,
        "promotion_manifest_path": str(promotion_path) if promotion_path else None,
        "promotion_manifest_hash": hash_file_bytes(promotion_path) if promotion_path else None,
        "raw_request_response_logging_status": raw_request_response_logging_status,
        "raw_request_response_logging": {
            "status": raw_request_response_logging_status,
            "raw_request_logged": True,
            "raw_response_logged": True,
            "storage": "results_jsonl_model_outputs",
        },
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


def parse_seeds(value: str | list[str] | tuple[str, ...] | None) -> list[int]:
    if value is None:
        raise PersonaValidationError("--seeds is required")
    raw_parts: list[str] = []
    if isinstance(value, str):
        raw_parts = value.split(",")
    else:
        for item in value:
            raw_parts.extend(str(item).split(","))
    seeds: list[int] = []
    for part in raw_parts:
        stripped = part.strip()
        if not stripped:
            raise PersonaValidationError("--seeds must be a comma-separated list of integers")
        try:
            seeds.append(int(stripped))
        except ValueError as exc:
            raise PersonaValidationError("--seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise PersonaValidationError("--seeds must include at least one integer")
    return seeds


def _seed_count_from_args(args: argparse.Namespace) -> int:
    if args.seed_count is not None:
        return args.seed_count
    if args.seeds:
        return len(parse_seeds(args.seeds))
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


def _require_model_pair(args: argparse.Namespace) -> None:
    if not args.model_base or not args.model_tuned:
        raise PersonaValidationError("run requires --model-base and --model-tuned")


def _validate_positive_int(value: int | None, name: str) -> int:
    if value is None or value <= 0:
        raise PersonaValidationError(f"{name} must be positive")
    return value


def _limited_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    _validate_positive_int(limit, "--limit-personas")
    if limit > len(rows):
        raise PersonaValidationError("--limit-personas cannot exceed validated persona row count")
    return rows[:limit]


def _run_stage_from_args(args: argparse.Namespace) -> str:
    return getattr(args, "run_stage", "auto") or "auto"


def _enforce_phase_execution_cap(args: argparse.Namespace, persona_count: int) -> None:
    stage = _run_stage_from_args(args)
    cap = RUN_STAGE_PERSONA_CAPS.get(stage, STAGED_RUN_MAX_PERSONAS)
    if persona_count <= cap:
        return
    label = "staged" if stage == "auto" else stage
    if stage == "auto":
        guidance = "use --run-stage dev/full with required evidence gates for larger runs"
    else:
        guidance = "reduce --limit-personas/--persona-count for this phase"
    raise PersonaValidationError(
        f"{label} run execution is capped at {cap} personas; {guidance}"
    )


def _promotion_manifest_arg(args: argparse.Namespace) -> str | None:
    return getattr(args, "promotion_manifest", None) or getattr(args, "promotion_manifest_path", None)


def _review_manifest_arg(args: argparse.Namespace) -> str | None:
    return getattr(args, "review_manifest", None) or getattr(args, "review_manifest_path", None)


def _dev_run_approval_arg(args: argparse.Namespace) -> str | None:
    return getattr(args, "dev_run_approval", None)


def _require_persona_path(args: argparse.Namespace, label: str) -> str:
    persona_path = getattr(args, "persona_path", None)
    if not persona_path:
        raise PersonaValidationError(f"{label} requires --persona-path")
    return persona_path


def _require_promotion_and_review_gates(
    args: argparse.Namespace,
    *,
    persona_rows: list[dict[str, Any]] | None = None,
    label: str,
) -> None:
    persona_path = _require_persona_path(args, label)
    promotion_manifest = _promotion_manifest_arg(args)
    review_manifest = _review_manifest_arg(args)
    if not promotion_manifest:
        raise PersonaValidationError(f"--promotion-manifest is required for {label}")
    if not review_manifest:
        raise PersonaValidationError(f"--review-manifest is required for {label}")
    full_rows = persona_rows if persona_rows is not None else validate_personas(persona_path)
    validate_dataset_promotion_manifest(promotion_manifest, persona_path=persona_path)
    validate_full_review_manifest(review_manifest, persona_rows=full_rows)


def _require_smoke_promoted_dataset_gate(args: argparse.Namespace) -> None:
    persona_path = _require_persona_path(args, "smoke run")
    promotion_manifest = _promotion_manifest_arg(args)
    if not promotion_manifest:
        raise PersonaValidationError("--promotion-manifest-path is required for smoke run")
    validate_dataset_promotion_manifest(promotion_manifest, persona_path=persona_path)


def _is_missing_explicit_metadata(value: str | None) -> bool:
    return value is None or not value.strip() or value.strip() == "not_available"


def _require_real_adapter_run_contract(args: argparse.Namespace) -> None:
    if args.adapter not in REAL_HTTP_ADAPTERS or args.dry_run:
        return
    expected_full_path = (REPO_ROOT / "data" / "personas.full.jsonl").resolve()
    actual_path = Path(args.persona_path).resolve() if args.persona_path else None
    if actual_path != expected_full_path:
        raise PersonaValidationError(
            "real vLLM/openai-compatible runs require "
            "--persona-path data/personas.full.jsonl after Sprint 7 promotion"
        )
    missing = [
        f"--{field.replace('_', '-')}"
        for field in EXPLICIT_REAL_RUN_METADATA_FIELDS
        if _is_missing_explicit_metadata(getattr(args, field, None))
    ]
    if missing:
        raise PersonaValidationError(
            "real vLLM/openai-compatible smoke runs require explicit runtime metadata: "
            + ", ".join(missing)
        )


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise PersonaValidationError(f"{label} does not exist: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PersonaValidationError(f"{label} must be a JSON object")
    return payload


def _require_int_field(payload: dict[str, Any], field: str, expected: int, label: str) -> None:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersonaValidationError(f"{label}.{field} must be integer {expected}")
    if value != expected:
        raise PersonaValidationError(f"{label}.{field} must equal {expected}; got {value}")


def _require_status(payload: dict[str, Any], label: str) -> None:
    status = payload.get("status")
    if status not in PASSING_EVIDENCE_STATUSES:
        allowed = ", ".join(sorted(PASSING_EVIDENCE_STATUSES))
        raise PersonaValidationError(f"{label}.status must be one of: {allowed}")


def _stage_matches(payload: dict[str, Any], expected_stage: str, label: str) -> None:
    stage = payload.get("stage") or payload.get("run_stage") or payload.get("evidence_type")
    allowed = {
        expected_stage,
        f"{expected_stage}_run",
        f"{expected_stage}_evidence",
        f"{expected_stage}_run_evidence",
    }
    if stage not in allowed:
        raise PersonaValidationError(
            f"{label} must declare stage/evidence_type for {expected_stage!r}"
        )


def _resolve_declared_path(raw_path: str, *, base_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative
    return base_path.parent / path


def validate_run_evidence_file(
    path: str | Path,
    *,
    expected_stage: str,
    expected_persona_count: int,
    expected_seed_count: int,
    expected_call_count_value: int,
) -> dict[str, Any]:
    evidence_path = Path(path)
    label = f"{expected_stage} evidence"
    payload = _load_json_object(evidence_path, label)
    _stage_matches(payload, expected_stage, label)
    _require_status(payload, label)
    _require_int_field(payload, "persona_count", expected_persona_count, label)
    _require_int_field(payload, "variants_per_persona", RUN_VARIANTS_PER_PERSONA, label)
    _require_int_field(payload, "model_count", RUN_MODEL_COUNT, label)
    _require_int_field(payload, "seed_count", expected_seed_count, label)
    _require_int_field(payload, "planned_generation_calls", expected_call_count_value, label)
    for field in RUN_EVIDENCE_REQUIRED_ARTIFACT_FIELDS:
        raw_value = payload.get(field)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise PersonaValidationError(f"{label}.{field} is required")
        declared_path = _resolve_declared_path(raw_value, base_path=evidence_path)
        if not declared_path.exists():
            raise PersonaValidationError(f"{label}.{field} does not exist: {declared_path}")
        hash_field = RUN_EVIDENCE_ARTIFACT_HASH_FIELDS[field]
        expected_hash = payload.get(hash_field)
        if not isinstance(expected_hash, str) or not expected_hash.strip():
            raise PersonaValidationError(f"{label}.{hash_field} is required")
        actual_hash = hash_file_bytes(declared_path)
        if expected_hash != actual_hash:
            raise PersonaValidationError(
                f"{label}.{hash_field} does not match {field}: expected {expected_hash}, got {actual_hash}"
            )
    return payload


def validate_dataset_promotion_manifest(
    path: str | Path,
    *,
    persona_path: str | Path,
) -> dict[str, Any]:
    payload = _load_json_object(path, "dataset promotion manifest")
    manifest_type = payload.get("manifest_type") or payload.get("artifact_type")
    if manifest_type != "dataset_promotion":
        raise PersonaValidationError("dataset promotion manifest.manifest_type must be dataset_promotion")
    if payload.get("status") != "promoted":
        raise PersonaValidationError("dataset promotion manifest.status must be promoted")
    _require_int_field(payload, "persona_count", FULL_RUN_PERSONA_COUNT, "dataset promotion manifest")
    dataset_hash = hash_file_bytes(persona_path)
    if payload.get("dataset_hash") != dataset_hash:
        raise PersonaValidationError("dataset promotion manifest.dataset_hash must match persona JSONL hash")
    return payload


def validate_full_review_manifest(
    path: str | Path,
    *,
    persona_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    review_rows = load_jsonl(path)
    persona_ids = {row["persona_id"] for row in persona_rows}
    review_ids = [row.get("persona_id") for row in review_rows]
    if len(review_rows) != FULL_RUN_PERSONA_COUNT:
        raise PersonaValidationError("full review manifest must contain exactly 200 rows")
    duplicates = sorted(persona_id for persona_id, count in Counter(review_ids).items() if count > 1)
    if duplicates:
        raise PersonaValidationError("full review manifest duplicate persona_id values: " + ", ".join(duplicates))
    if set(review_ids) != persona_ids:
        missing = sorted(persona_ids - set(review_ids))
        unknown = sorted(set(review_ids) - persona_ids)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing[:5]))
        if unknown:
            details.append("unknown=" + ",".join(str(value) for value in unknown[:5]))
        raise PersonaValidationError("full review manifest persona_id set mismatch: " + "; ".join(details))

    for index, row in enumerate(review_rows, start=1):
        try:
            validate_schema_file(row, REVIEW_MANIFEST_SCHEMA_PATH)
        except PersonaValidationError as exc:
            raise PersonaValidationError(f"full review manifest row {index}: {exc}") from exc
        if row.get("review_status") != "approved":
            raise PersonaValidationError(f"full review manifest row {index}: review_status must be approved")
        for field in FULL_REVIEW_GATE_FIELDS:
            gate = row.get(field, {})
            evidence = gate.get("evidence") if isinstance(gate, dict) else None
            if gate.get("status") not in PASSING_FULL_REVIEW_GATE_STATUSES:
                raise PersonaValidationError(
                    f"full review manifest row {index}: {field}.status must be passed/manual_pass"
                )
            if not isinstance(evidence, list) or not any(str(item).strip() for item in evidence):
                raise PersonaValidationError(
                    f"full review manifest row {index}: {field}.evidence is required"
                )
    return review_rows


def validate_full_run_approval(
    path: str | Path,
    *,
    persona_path: str | Path,
) -> dict[str, Any]:
    payload = _load_json_object(path, "full-run approval")
    if payload.get("approval_type") != "full_run_approval":
        raise PersonaValidationError("full-run approval.approval_type must be full_run_approval")
    if payload.get("status") != "approved":
        raise PersonaValidationError("full-run approval.status must be approved")
    for field in ("approved_by", "approved_at"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PersonaValidationError(f"full-run approval.{field} is required")
    _require_int_field(payload, "persona_count", FULL_RUN_PERSONA_COUNT, "full-run approval")
    _require_int_field(payload, "planned_generation_calls", FULL_RUN_CALL_COUNT, "full-run approval")
    dataset_hash = hash_file_bytes(persona_path)
    if payload.get("dataset_hash") != dataset_hash:
        raise PersonaValidationError("full-run approval.dataset_hash must match persona JSONL hash")
    return payload


def validate_dev_run_approval(
    path: str | Path,
    *,
    persona_path: str | Path,
) -> dict[str, Any]:
    payload = _load_json_object(path, "dev-run approval")
    if payload.get("approval_type") not in DEV_RUN_APPROVAL_TYPES:
        allowed = ", ".join(sorted(DEV_RUN_APPROVAL_TYPES))
        raise PersonaValidationError(f"dev-run approval.approval_type must be one of: {allowed}")
    if payload.get("status") != "approved":
        raise PersonaValidationError("dev-run approval.status must be approved")
    for field in ("approved_by", "approved_at"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PersonaValidationError(f"dev-run approval.{field} is required")
    _require_int_field(payload, "persona_count", DEV_RUN_PERSONA_COUNT, "dev-run approval")
    _require_int_field(payload, "planned_generation_calls", DEV_RUN_CALL_COUNT, "dev-run approval")
    dataset_hash = hash_file_bytes(persona_path)
    if payload.get("dataset_hash") != dataset_hash:
        raise PersonaValidationError("dev-run approval.dataset_hash must match persona JSONL hash")
    return payload


def _require_plan_shape(
    *,
    stage: str,
    persona_count: int,
    variants_per_persona: int,
    model_count: int,
    seed_count: int,
) -> int:
    expected_personas = DEV_RUN_PERSONA_COUNT if stage == "dev" else FULL_RUN_PERSONA_COUNT
    expected_seeds = DEV_RUN_SEED_COUNT if stage == "dev" else FULL_RUN_SEED_COUNT
    expected_calls = DEV_RUN_CALL_COUNT if stage == "dev" else FULL_RUN_CALL_COUNT
    if persona_count != expected_personas:
        raise PersonaValidationError(f"{stage} preflight requires persona_count={expected_personas}")
    if variants_per_persona != RUN_VARIANTS_PER_PERSONA:
        raise PersonaValidationError(f"{stage} preflight requires variants_per_persona={RUN_VARIANTS_PER_PERSONA}")
    if model_count != RUN_MODEL_COUNT:
        raise PersonaValidationError(f"{stage} preflight requires model_count={RUN_MODEL_COUNT}")
    if seed_count != expected_seeds:
        raise PersonaValidationError(f"{stage} preflight requires seed_count={expected_seeds}")
    planned_calls = expected_call_count(persona_count, variants_per_persona, model_count, seed_count)
    if planned_calls != expected_calls:
        raise PersonaValidationError(f"{stage} preflight planned call count must equal {expected_calls}")
    return planned_calls


def _runtime_value_is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    lowered = stripped.lower()
    return (
        lowered in PLACEHOLDER_RUNTIME_VALUES
        or (stripped.startswith("<") and stripped.endswith(">"))
    )


def _require_full_runtime_metadata(args: argparse.Namespace) -> None:
    missing = [
        f"--{field.replace('_', '-')}"
        for field in FULL_RUN_REQUIRED_RUNTIME_ARGS
        if _runtime_value_is_placeholder(getattr(args, field, None))
    ]
    if missing:
        raise PersonaValidationError(
            "full preflight requires explicit runtime metadata: " + ", ".join(missing)
        )


def _preflight_shape(args: argparse.Namespace) -> tuple[list[dict[str, Any]] | None, int, int]:
    if args.persona_path:
        rows = validate_personas(args.persona_path)
        variant_counts = {len(row["variants"]) for row in rows}
        if len(variant_counts) != 1:
            raise PersonaValidationError("all persona rows must have the same variant count")
        persona_count = len(rows)
        if args.limit_personas is not None:
            if args.stage == "full":
                raise PersonaValidationError("full preflight must not use --limit-personas")
            limit = _validate_positive_int(args.limit_personas, "--limit-personas")
            if limit > persona_count:
                raise PersonaValidationError("--limit-personas cannot exceed validated persona row count")
            persona_count = limit
        return rows, persona_count, variant_counts.pop()
    if args.stage == "full":
        raise PersonaValidationError("full preflight requires --persona-path")
    persona_count = _validate_positive_int(args.persona_count, "--persona-count")
    variants_per_persona = _validate_positive_int(args.variants_per_persona, "--variants-per-persona")
    return None, persona_count, variants_per_persona


def validate_dev_preflight(
    args: argparse.Namespace,
    *,
    persona_count: int,
    variants_per_persona: int,
    persona_rows: list[dict[str, Any]] | None = None,
) -> int:
    planned_calls = _require_plan_shape(
        stage="dev",
        persona_count=persona_count,
        variants_per_persona=variants_per_persona,
        model_count=args.model_count,
        seed_count=args.seed_count,
    )
    _require_promotion_and_review_gates(args, persona_rows=persona_rows, label="dev preflight")
    if not args.smoke_evidence:
        raise PersonaValidationError("--smoke-evidence is required for dev preflight")
    dev_run_approval = _dev_run_approval_arg(args)
    if not dev_run_approval:
        raise PersonaValidationError("--dev-run-approval is required for dev preflight")
    validate_run_evidence_file(
        args.smoke_evidence,
        expected_stage="smoke",
        expected_persona_count=SMOKE_RUN_PERSONA_COUNT,
        expected_seed_count=SMOKE_RUN_SEED_COUNT,
        expected_call_count_value=SMOKE_RUN_CALL_COUNT,
    )
    validate_dev_run_approval(dev_run_approval, persona_path=args.persona_path)
    return planned_calls


def validate_full_preflight(
    args: argparse.Namespace,
    *,
    persona_rows: list[dict[str, Any]],
    persona_count: int,
    variants_per_persona: int,
) -> int:
    planned_calls = _require_plan_shape(
        stage="full",
        persona_count=persona_count,
        variants_per_persona=variants_per_persona,
        model_count=args.model_count,
        seed_count=args.seed_count,
    )
    if not args.promotion_manifest:
        raise PersonaValidationError("--promotion-manifest is required for full preflight")
    if not args.review_manifest:
        raise PersonaValidationError("--review-manifest is required for full preflight")
    if not args.smoke_evidence:
        raise PersonaValidationError("--smoke-evidence is required for full preflight")
    if not args.dev_evidence:
        raise PersonaValidationError("--dev-evidence is required for full preflight")
    if not args.full_run_approval:
        raise PersonaValidationError("--full-run-approval is required for full preflight")

    validate_dataset_promotion_manifest(args.promotion_manifest, persona_path=args.persona_path)
    validate_full_review_manifest(args.review_manifest, persona_rows=persona_rows)
    validate_run_evidence_file(
        args.smoke_evidence,
        expected_stage="smoke",
        expected_persona_count=SMOKE_RUN_PERSONA_COUNT,
        expected_seed_count=SMOKE_RUN_SEED_COUNT,
        expected_call_count_value=SMOKE_RUN_CALL_COUNT,
    )
    validate_run_evidence_file(
        args.dev_evidence,
        expected_stage="dev",
        expected_persona_count=DEV_RUN_PERSONA_COUNT,
        expected_seed_count=DEV_RUN_SEED_COUNT,
        expected_call_count_value=DEV_RUN_CALL_COUNT,
    )
    validate_full_run_approval(args.full_run_approval, persona_path=args.persona_path)
    _require_full_runtime_metadata(args)
    return planned_calls


def _run_approval_stage(
    args: argparse.Namespace,
    *,
    persona_count: int,
    variants_per_persona: int,
    seed_count: int,
) -> str | None:
    explicit_stage = getattr(args, "run_stage", "auto")
    if explicit_stage in {"dev", "full"}:
        return explicit_stage
    if explicit_stage == "smoke":
        return "smoke"
    if (
        persona_count == DEV_RUN_PERSONA_COUNT
        and variants_per_persona == RUN_VARIANTS_PER_PERSONA
        and seed_count == DEV_RUN_SEED_COUNT
    ):
        return "dev"
    if (
        persona_count == FULL_RUN_PERSONA_COUNT
        and variants_per_persona == RUN_VARIANTS_PER_PERSONA
        and seed_count == FULL_RUN_SEED_COUNT
    ):
        return "full"
    return None


def _run_args_for_preflight(args: argparse.Namespace, *, seed_count: int) -> argparse.Namespace:
    values = vars(args).copy()
    values["model_count"] = RUN_MODEL_COUNT
    values["seed_count"] = seed_count
    values["promotion_manifest"] = values.get("promotion_manifest_path")
    values["review_manifest"] = values.get("review_manifest_path")
    return argparse.Namespace(**values)


def _enforce_run_approval_gates(
    args: argparse.Namespace,
    *,
    rows: list[dict[str, Any]] | None,
    persona_count: int,
    variants_per_persona: int,
    seed_count: int,
) -> None:
    stage = _run_approval_stage(
        args,
        persona_count=persona_count,
        variants_per_persona=variants_per_persona,
        seed_count=seed_count,
    )
    if stage is None:
        return
    if stage == "smoke":
        if variants_per_persona != RUN_VARIANTS_PER_PERSONA:
            raise PersonaValidationError(f"smoke run requires variants_per_persona={RUN_VARIANTS_PER_PERSONA}")
        if seed_count != SMOKE_RUN_SEED_COUNT:
            raise PersonaValidationError(f"smoke run requires seed_count={SMOKE_RUN_SEED_COUNT}")
        _require_smoke_promoted_dataset_gate(args)
        return
    preflight_args = _run_args_for_preflight(args, seed_count=seed_count)
    if stage == "dev":
        if args.persona_path:
            persona_rows = validate_personas(args.persona_path)
        else:
            persona_rows = rows
        validate_dev_preflight(
            preflight_args,
            persona_count=persona_count,
            variants_per_persona=variants_per_persona,
            persona_rows=persona_rows,
        )
        return
    if rows is None:
        raise PersonaValidationError("full run approval gates require validated persona rows")
    validate_full_preflight(
        preflight_args,
        persona_rows=rows,
        persona_count=persona_count,
        variants_per_persona=variants_per_persona,
    )


def _run_persona_shape(args: argparse.Namespace) -> tuple[list[dict[str, Any]] | None, int, int]:
    if args.persona_path:
        rows = _limited_rows(validate_personas(args.persona_path), args.limit_personas)
        variant_counts = {len(row["variants"]) for row in rows}
        if len(variant_counts) != 1:
            raise PersonaValidationError("all persona rows must have the same variant count")
        return rows, len(rows), variant_counts.pop()

    if args.dry_run and args.persona_count is not None and args.variants_per_persona is not None:
        persona_count = _validate_positive_int(args.persona_count, "--persona-count")
        variants_per_persona = _validate_positive_int(args.variants_per_persona, "--variants-per-persona")
        if args.limit_personas is not None:
            _validate_positive_int(args.limit_personas, "--limit-personas")
            if args.limit_personas > persona_count:
                raise PersonaValidationError("--limit-personas cannot exceed --persona-count")
            persona_count = args.limit_personas
        return None, persona_count, variants_per_persona

    raise PersonaValidationError(
        "run requires --persona-path, or --dry-run with --persona-count and --variants-per-persona"
    )


def _decoding_params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_tokens <= 0:
        raise PersonaValidationError("--max-tokens must be positive")
    return {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }


def _non_empty_arg(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _real_adapter_base_urls(args: argparse.Namespace) -> tuple[str, str]:
    shared = _non_empty_arg(getattr(args, "base_url", None))
    base_url = _non_empty_arg(getattr(args, "base_url_base", None)) or shared
    tuned_url = _non_empty_arg(getattr(args, "base_url_tuned", None)) or shared
    if base_url and tuned_url:
        return base_url, tuned_url
    raise PersonaValidationError(
        "--base-url is required as a shared endpoint, or provide both "
        "--base-url-base and --base-url-tuned for vllm/openai-compatible adapters"
    )


def _http_adapter_from_args(args: argparse.Namespace, *, base_url: str) -> VLLMOpenAIAdapter:
    serving_stack = args.serving_stack
    if serving_stack is None:
        serving_stack = "vllm" if args.adapter == "vllm" else "openai_compatible"
    return VLLMOpenAIAdapter(
        base_url=base_url,
        adapter_name=args.adapter,
        api_key_env=args.api_key_env,
        serving_stack=serving_stack,
        serving_stack_version=args.serving_stack_version or "not_available",
    )


def _adapters_from_args(args: argparse.Namespace) -> AdapterPair:
    if args.adapter == "mock":
        adapter = MockAdapter()
        return AdapterPair(base=adapter, tuned=adapter)
    if args.adapter in {"vllm", "openai-compatible"}:
        base_url, tuned_url = _real_adapter_base_urls(args)
        return AdapterPair(
            base=_http_adapter_from_args(args, base_url=base_url),
            tuned=_http_adapter_from_args(args, base_url=tuned_url),
        )
    raise PersonaValidationError(f"unsupported adapter: {args.adapter!r}")


def _adapter_from_args(args: argparse.Namespace) -> BaseAdapter:
    return _adapters_from_args(args).base


def _provider_or_endpoint_summary(adapters: AdapterPair) -> str:
    base_endpoint = adapters.base.provider_or_endpoint
    tuned_endpoint = adapters.tuned.provider_or_endpoint
    if base_endpoint == tuned_endpoint:
        return base_endpoint
    return "model_specific_endpoints"


def _run_output_dir(args: argparse.Namespace) -> Path:
    if not args.out:
        raise PersonaValidationError("run requires --out unless --dry-run is used")
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    protected = (REPO_ROOT / "results" / "autoresearch").resolve()
    resolved = output_dir.resolve()
    if resolved == protected or resolved.is_relative_to(protected):
        raise PersonaValidationError("persona_eval.py run must not write under results/autoresearch")
    return output_dir


def _default_review_manifest_for_persona_path(persona_path: str | None) -> str | None:
    if persona_path is None:
        return None
    try:
        if Path(persona_path).resolve() == (REPO_ROOT / "data" / "personas.sample.jsonl").resolve():
            return str(DEFAULT_SAMPLE_REVIEW_MANIFEST_PATH)
    except OSError:
        return None
    return None


def _disabled_token_kl_result() -> ScoreContinuationResult:
    return ScoreContinuationResult(
        status="not_applicable",
        value=None,
        reason_code="disabled_by_user",
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


def _score_result_for_run(
    *,
    args: argparse.Namespace,
    adapter: BaseAdapter,
    rendered: RenderedPrompt,
    run_id: str,
    persona_row: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    model_id: str,
    model_alias: str,
    decoding_params: dict[str, Any],
    stop_sequences: list[str],
) -> ScoreContinuationResult:
    if args.disable_token_kl or args.score_mode == "disabled":
        return _disabled_token_kl_result()
    score_request = build_score_continuation_request(
        rendered=rendered,
        run_id=run_id,
        persona_id=persona_row["persona_id"],
        variant_id=variant["variant_id"],
        variant_type=variant["type"],
        model_alias=model_alias,
        model_id=model_id,
        model_revision_or_hash=(
            args.model_base_revision_or_hash
            if model_alias == args.model_base_alias
            else args.model_tuned_revision_or_hash
        ),
        tokenizer_name=args.tokenizer_name,
        tokenizer_hash=args.tokenizer_hash,
        chat_template_hash=args.chat_template_hash,
        seed=seed,
        decoding_params=decoding_params,
        stop_sequences=stop_sequences,
        fixed_continuation="not_used_without_aligned_scoring",
        fixed_continuation_id="aligned-scoring-unavailable",
        adapter=adapter,
    )
    return adapter.score_continuation(score_request)


def _build_run_rows(
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    base_adapter: BaseAdapter,
    tuned_adapter: BaseAdapter,
    run_id: str,
    seeds: list[int],
    decoding_params: dict[str, Any],
    stop_sequences: list[str],
) -> list[dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    model_pair = {
        "base": args.model_base,
        "tuned": args.model_tuned,
        "base_alias": args.model_base_alias,
        "tuned_alias": args.model_tuned_alias,
    }
    for persona_row in rows:
        for variant in persona_row["variants"]:
            rendered = render_prompt(
                persona_row,
                variant,
                prompt_template_version=args.prompt_template_version,
            )
            for seed in seeds:
                base_request = build_generation_request(
                    rendered=rendered,
                    run_id=run_id,
                    persona_id=persona_row["persona_id"],
                    variant_id=variant["variant_id"],
                    variant_type=variant["type"],
                    model_alias=args.model_base_alias,
                    model_id=args.model_base,
                    model_revision_or_hash=args.model_base_revision_or_hash,
                    tokenizer_name=args.tokenizer_name,
                    tokenizer_hash=args.tokenizer_hash,
                    chat_template_hash=args.chat_template_hash,
                    seed=seed,
                    decoding_params=decoding_params,
                    stop_sequences=stop_sequences,
                    adapter=base_adapter,
                )
                tuned_request = build_generation_request(
                    rendered=rendered,
                    run_id=run_id,
                    persona_id=persona_row["persona_id"],
                    variant_id=variant["variant_id"],
                    variant_type=variant["type"],
                    model_alias=args.model_tuned_alias,
                    model_id=args.model_tuned,
                    model_revision_or_hash=args.model_tuned_revision_or_hash,
                    tokenizer_name=args.tokenizer_name,
                    tokenizer_hash=args.tokenizer_hash,
                    chat_template_hash=args.chat_template_hash,
                    seed=seed,
                    decoding_params=decoding_params,
                    stop_sequences=stop_sequences,
                    adapter=tuned_adapter,
                )
                base_result = base_adapter.generate(base_request)
                tuned_result = tuned_adapter.generate(tuned_request)
                score_result = _score_result_for_run(
                    args=args,
                    adapter=tuned_adapter,
                    rendered=rendered,
                    run_id=run_id,
                    persona_row=persona_row,
                    variant=variant,
                    seed=seed,
                    model_id=args.model_tuned,
                    model_alias=args.model_tuned_alias,
                    decoding_params=decoding_params,
                    stop_sequences=stop_sequences,
                )
                result_rows.append(
                    build_result_row(
                        run_id=run_id,
                        persona_row=persona_row,
                        variant=variant,
                        seed=seed,
                        rendered=rendered,
                        model_pair=model_pair,
                        base_request=base_request,
                        base_result=base_result,
                        tuned_request=tuned_request,
                        tuned_result=tuned_result,
                        score_result=score_result,
                    )
                )
    return result_rows


def _write_run_outputs(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.jsonl"
    manifest_tmp = output_dir / "manifest.json.tmp"
    results_tmp = output_dir / "results.jsonl.tmp"

    manifest_tmp.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with results_tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest_tmp.replace(manifest_path)
    results_tmp.replace(results_path)
    return manifest_path, results_path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def cmd_preflight(args: argparse.Namespace) -> int:
    rows, persona_count, variants_per_persona = _preflight_shape(args)
    if args.stage == "dev":
        planned_calls = validate_dev_preflight(
            args,
            persona_count=persona_count,
            variants_per_persona=variants_per_persona,
            persona_rows=rows,
        )
    else:
        if rows is None:
            raise PersonaValidationError("full preflight requires validated persona rows")
        planned_calls = validate_full_preflight(
            args,
            persona_rows=rows,
            persona_count=persona_count,
            variants_per_persona=variants_per_persona,
        )
    print(f"preflight_stage={args.stage}")
    print(f"planned_generation_calls={planned_calls}")
    print("preflight_status=pass")
    return 0


def cmd_validate_model_matrix(args: argparse.Namespace) -> int:
    matrix = load_model_matrix(args.matrix_path)
    report = validate_model_matrix(
        matrix,
        require_real_run_ready=args.require_real_run_ready,
    )
    print(f"model_matrix_status={report['status']}")
    print(f"real_run_readiness={'ready' if report['real_run_ready'] else 'blocked'}")
    print(f"placeholder_count={report['placeholder_count']}")
    print(f"drift_pairs={report['drift_pair_count']}")
    print(f"standalone_instruct_models={report['standalone_instruct_model_count']}")
    print(f"cross_family_comparisons={report['cross_family_comparison_count']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _require_model_pair(args)
    seeds = parse_seeds(args.seeds)
    decoding_params = _decoding_params_from_args(args)
    stop_sequences: list[str] = []
    adapters = _adapters_from_args(args)
    rows, persona_count, variants_per_persona = _run_persona_shape(args)
    planned_calls = expected_call_count(persona_count, variants_per_persona, 2, len(seeds))

    _enforce_phase_execution_cap(args, persona_count)
    if args.out:
        _run_output_dir(args)
    if not args.dry_run:
        if rows is None:
            raise PersonaValidationError("non-dry run requires --persona-path")
        _enforce_run_approval_gates(
            args,
            rows=rows,
            persona_count=persona_count,
            variants_per_persona=variants_per_persona,
            seed_count=len(seeds),
        )
    if args.dry_run:
        print(f"planned_generation_calls={planned_calls}")
        return 0
    _require_real_adapter_run_contract(args)

    output_dir = _run_output_dir(args)
    run_id = args.run_id or dt.datetime.now(dt.UTC).strftime("run-%Y%m%dT%H%M%SZ")
    manifest = create_run_manifest(
        persona_path=args.persona_path,
        model_base=args.model_base,
        model_tuned=args.model_tuned,
        seeds=seeds,
        run_id=run_id,
        prompt_template_version=args.prompt_template_version,
        chat_template_hash=args.chat_template_hash,
        tokenizer_name=args.tokenizer_name,
        tokenizer_hash=args.tokenizer_hash,
        model_base_revision_or_hash=args.model_base_revision_or_hash,
        model_tuned_revision_or_hash=args.model_tuned_revision_or_hash,
        adapter=adapters.base.adapter_name,
        provider_or_endpoint=_provider_or_endpoint_summary(adapters),
        provider_or_endpoint_base=adapters.base.provider_or_endpoint,
        provider_or_endpoint_tuned=adapters.tuned.provider_or_endpoint,
        serving_stack=adapters.base.serving_stack,
        serving_stack_version=adapters.base.serving_stack_version,
        scoring_capability=adapters.base.scoring_capability,
        gpu_cuda_driver=args.gpu_cuda_driver,
        decoding_params=decoding_params,
        stop_sequences=stop_sequences,
        extractor_version=EXTRACTOR_VERSION,
        embedding_model_revision="not_available",
        nli_or_judge_model_revision="not_available",
        promotion_manifest_path=args.promotion_manifest_path,
        review_manifest_path=args.review_manifest_path or _default_review_manifest_for_persona_path(args.persona_path),
    )
    manifest["harness_version"] = EVALUATOR_VERSION
    manifest["metric_version"] = METRIC_VERSION
    manifest["model_base_alias"] = args.model_base_alias
    manifest["model_tuned_alias"] = args.model_tuned_alias
    manifest["score_mode"] = "disabled" if args.disable_token_kl else args.score_mode
    validate_run_manifest(manifest)

    result_rows = _build_run_rows(
        rows=rows,
        args=args,
        base_adapter=adapters.base,
        tuned_adapter=adapters.tuned,
        run_id=run_id,
        seeds=seeds,
        decoding_params=decoding_params,
        stop_sequences=stop_sequences,
    )
    manifest_path, results_path = _write_run_outputs(
        output_dir=output_dir,
        manifest=manifest,
        rows=result_rows,
    )
    print(f"planned_generation_calls={planned_calls}")
    print(f"written_result_rows={len(result_rows)}")
    print(f"manifest_path={_display_path(manifest_path)}")
    print(f"results_path={_display_path(results_path)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persona drift evaluation harness")
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

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="validate dev/full run gates without executing model calls",
    )
    preflight_parser.add_argument("--stage", choices=["dev", "full"], required=True)
    preflight_parser.add_argument("--persona-path")
    preflight_parser.add_argument("--persona-count", type=int)
    preflight_parser.add_argument("--limit-personas", type=int)
    preflight_parser.add_argument("--variants-per-persona", type=int)
    preflight_parser.add_argument("--model-count", type=int, default=RUN_MODEL_COUNT)
    preflight_parser.add_argument("--seed-count", type=int, required=True)
    preflight_parser.add_argument("--smoke-evidence")
    preflight_parser.add_argument("--dev-evidence")
    preflight_parser.add_argument("--promotion-manifest")
    preflight_parser.add_argument("--review-manifest")
    preflight_parser.add_argument("--dev-run-approval")
    preflight_parser.add_argument("--full-run-approval")
    preflight_parser.add_argument("--model-base", default="not_available")
    preflight_parser.add_argument("--model-tuned", default="not_available")
    preflight_parser.add_argument("--model-base-revision-or-hash", default="not_available")
    preflight_parser.add_argument("--model-tuned-revision-or-hash", default="not_available")
    preflight_parser.add_argument("--tokenizer-name", default="not_available")
    preflight_parser.add_argument("--tokenizer-hash", default="not_available")
    preflight_parser.add_argument("--chat-template-hash", default="not_available")
    preflight_parser.add_argument("--serving-stack-version", default="not_available")
    preflight_parser.set_defaults(func=cmd_preflight)

    matrix_parser = subparsers.add_parser(
        "validate-model-matrix",
        help="validate production/open model matrix config without network or model calls",
    )
    matrix_parser.add_argument("--matrix-path", required=True)
    matrix_parser.add_argument(
        "--require-real-run-ready",
        action="store_true",
        help="fail if endpoint or model revision/hash placeholders remain",
    )
    matrix_parser.set_defaults(func=cmd_validate_model_matrix)

    run_parser = subparsers.add_parser("run", help="run a staged mock or explicit adapter smoke")
    run_parser.add_argument("--persona-path")
    run_parser.add_argument("--persona-count", type=int)
    run_parser.add_argument("--variants-per-persona", type=int)
    run_parser.add_argument("--out")
    run_parser.add_argument(
        "--run-stage",
        choices=["auto", "smoke", "dev", "full"],
        default="auto",
        help="approval-gate stage for non-dry run attempts",
    )
    run_parser.add_argument(
        "--adapter",
        choices=["mock", "vllm", "openai-compatible"],
        required=True,
    )
    run_parser.add_argument("--base-url", help="Shared OpenAI-compatible endpoint for both base and tuned calls")
    run_parser.add_argument("--base-url-base", help="OpenAI-compatible endpoint for base model calls")
    run_parser.add_argument("--base-url-tuned", help="OpenAI-compatible endpoint for tuned model calls")
    run_parser.add_argument("--api-key-env")
    run_parser.add_argument("--serving-stack")
    run_parser.add_argument("--serving-stack-version")
    run_parser.add_argument(
        "--score-mode",
        choices=["canonical", "disabled", "diagnostic_only"],
        default="canonical",
    )
    run_parser.add_argument("--disable-token-kl", action="store_true")
    run_parser.add_argument("--model-base", required=True)
    run_parser.add_argument("--model-tuned", required=True)
    run_parser.add_argument("--model-base-revision-or-hash", default="not_available")
    run_parser.add_argument("--model-tuned-revision-or-hash", default="not_available")
    run_parser.add_argument("--model-base-alias", default="base")
    run_parser.add_argument("--model-tuned-alias", default="tuned")
    run_parser.add_argument("--tokenizer-name", default="not_available")
    run_parser.add_argument("--tokenizer-hash", default="not_available")
    run_parser.add_argument("--chat-template-hash", default="not_available")
    run_parser.add_argument("--gpu-cuda-driver", default="not_available")
    run_parser.add_argument("--promotion-manifest-path")
    run_parser.add_argument("--review-manifest-path")
    run_parser.add_argument("--smoke-evidence")
    run_parser.add_argument("--dev-evidence")
    run_parser.add_argument("--dev-run-approval")
    run_parser.add_argument("--full-run-approval")
    run_parser.add_argument("--seeds", nargs="+", required=True)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--max-tokens", type=int, default=140)
    run_parser.add_argument("--prompt-template-version", default=PROMPT_TEMPLATE_VERSION)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--limit-personas", type=int)
    run_parser.set_defaults(func=cmd_run)

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
