# Limitations Skeleton

## Bottom Line

Status: `PENDING_SPRINT_10`. This file lists limitation categories that must be
resolved or explicitly carried into the final report. It does not assert final
benchmark limitations until Sprint 10 artifacts exist.

## Evidence Classification

### Final-run evidence

`PENDING_SPRINT_10`. Final limitations must be grounded in the frozen dataset,
run manifest, result rows, aggregate report, and model-run card.

### Diagnostic/pilot evidence

Diagnostic/pilot limitations may describe fixture, mock, smoke, or dev-run
constraints. They must remain separate from final-run limitations.

## Dataset Limitations

`PENDING_SPRINT_10`.

- Dataset hash: `PENDING_SPRINT_10:dataset_hash`
- Source/license coverage limitations: `PENDING_SPRINT_10:source_license_limitations`
- Review coverage limitations: `PENDING_SPRINT_10:review_coverage_limitations`
- Semantic-equivalence limitations: `PENDING_SPRINT_10:semantic_equivalence_limitations`
- NLI/contradiction limitations: `PENDING_SPRINT_10:nli_contradiction_limitations`
- Deduplication limitations: `PENDING_SPRINT_10:dedupe_limitations`

## Model And Runtime Limitations

`PENDING_SPRINT_10`.

- Model IDs: `PENDING_SPRINT_10:model_ids`
- Model revision uncertainty: `PENDING_SPRINT_10:model_revision_limitations`
- Tokenizer/chat-template metadata: `PENDING_SPRINT_10:tokenizer_chat_template_metadata`
- Serving stack limitations: `PENDING_SPRINT_10:serving_stack_limitations`
- Hardware/runtime limitations: `PENDING_SPRINT_10:hardware_runtime_limitations`
- Endpoint or adapter limitations: `PENDING_SPRINT_10:adapter_endpoint_limitations`

## Metric Limitations

### Persona Adherence

PA status: `PENDING_SPRINT_10:pa_status`.

If PA remains mock-only, not run, or not applicable, the final report must say so
directly and must not report it as real persona adherence.

### Token-KL

Token-KL status: `PENDING_SPRINT_10:token_kl_status`.

If aligned fixed-continuation scoring is unavailable, canonical Token-KL must be
reported as not applicable. Diagnostic endpoint-capped logprobs must not be
treated as equivalent to canonical Token-KL.

### Behavioral-Consistency F1

BC-F1 summaries: `PENDING_SPRINT_10:bc_f1_summaries`.

The final limitation text must state whether BC-F1 uses gold labels, majority
vote, pairwise agreement, or another documented comparison target.

## Statistical Limitations

`PENDING_SPRINT_10`.

- Persona-level inference limitations: `PENDING_SPRINT_10:persona_level_limitations`
- Confidence interval limitations: `PENDING_SPRINT_10:ci_limitations`
- Effect-size limitations: `PENDING_SPRINT_10:effect_size_limitations`
- P-value limitations: `PENDING_SPRINT_10:p_value_limitations`
- Multiple-comparison limitations: `PENDING_SPRINT_10:multiple_comparison_limitations`

## Operational Limitations

`PENDING_SPRINT_10`.

- Failed or retried calls: `PENDING_SPRINT_10:retry_limitations`
- Truncation and stop-reason concerns: `PENDING_SPRINT_10:truncation_limitations`
- Raw request/response completeness: `PENDING_SPRINT_10:raw_request_response_limitations`
- Cache invalidation risk: `PENDING_SPRINT_10:cache_invalidation_limitations`

## Diagnostic/Pilot Limitation Ledger

Diagnostic/pilot evidence must be labeled here and must not be merged into
final-run findings.

| Artifact | Limitation | Status |
| --- | --- | --- |
| `results/sprint5_mock/` | `PENDING_SPRINT_10:diagnostic_mock_limitations` | Diagnostic/pilot evidence |
| `reports/sprint5_mock/` | `PENDING_SPRINT_10:diagnostic_report_limitations` | Diagnostic/pilot evidence |
