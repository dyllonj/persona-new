# Methodology Skeleton

## Bottom Line

Status: `PENDING_SPRINT_10`. This methodology file describes the intended final
reporting structure and guardrails. It must be completed against frozen Sprint
10 artifacts before any final claims are made.

## Evidence Classification

### Final-run evidence

`PENDING_SPRINT_10`. Final-run evidence must be sourced from the frozen full
dataset, run manifest, result rows, aggregate report, and generated chart data.

### Diagnostic/pilot evidence

Diagnostic/pilot evidence can document harness behavior, command viability, mock
adapter behavior, smoke-run viability, or dev-run operational issues. It cannot
support final benchmark findings.

## Research Design

`PENDING_SPRINT_10`.

Planned comparison:

- Base model: `PENDING_SPRINT_10:base_model_id`
- Tuned model: `PENDING_SPRINT_10:tuned_model_id`
- Dataset: `PENDING_SPRINT_10:dataset_hash`
- Prompt template: `PENDING_SPRINT_10:prompt_template_version_and_hash`
- Seeds: `PENDING_SPRINT_10:seeds`
- Unit of inference: `persona_id`

## Dataset Construction

`PENDING_SPRINT_10`. The final dataset section must name the generation or
promotion path and must reference validation evidence, not assumptions.

Required fields:

- Dataset hash: `PENDING_SPRINT_10:dataset_hash`
- Source inventory: `PENDING_SPRINT_10:source_inventory`
- License inventory: `PENDING_SPRINT_10:license_inventory`
- Promotion manifest hash: `PENDING_SPRINT_10:promotion_manifest_hash`
- Review manifest hash: `PENDING_SPRINT_10:review_manifest_hash`
- Human-review coverage: `PENDING_SPRINT_10:human_review_coverage`

## Variant Contract

The executable set must contain exactly six variants per persona:

- `canonical`
- `paraphrase`
- `negation_preserving`
- `distractor`
- `instruction_prefix`
- `temperature_robust`

Final compliance status: `PENDING_SPRINT_10:variant_contract_status`.

## Validation And Review Workflow

`PENDING_SPRINT_10`.

The final report must summarize:

- Schema validation status: `PENDING_SPRINT_10:schema_validation_status`
- Source/provenance validation status: `PENDING_SPRINT_10:source_validation_status`
- License validation status: `PENDING_SPRINT_10:license_validation_status`
- Safety filter status: `PENDING_SPRINT_10:safety_filter_status`
- Semantic-equivalence status: `PENDING_SPRINT_10:semantic_equivalence_status`
- NLI/contradiction status: `PENDING_SPRINT_10:nli_contradiction_status`
- Gold-label preview status: `PENDING_SPRINT_10:gold_label_preview_status`
- Dedupe status: `PENDING_SPRINT_10:dedupe_status`
- Human-review status: `PENDING_SPRINT_10:human_review_status`

## Execution Method

`PENDING_SPRINT_10`.

Default serving target is local `vLLM` for the full benchmark. Hosted
OpenAI-compatible endpoints may be described only as black-box comparisons
unless they provide a verified aligned scoring path.

Required runtime fields:

- Model IDs: `PENDING_SPRINT_10:model_ids`
- Model revisions or hashes: `PENDING_SPRINT_10:model_revisions_or_hashes`
- Tokenizer/chat-template metadata: `PENDING_SPRINT_10:tokenizer_chat_template_metadata`
- Serving stack and version: `PENDING_SPRINT_10:serving_stack_version`
- Adapter: `PENDING_SPRINT_10:adapter`
- Provider or endpoint: `PENDING_SPRINT_10:provider_or_endpoint`
- Decoding parameters: `PENDING_SPRINT_10:decoding_params`
- Stop sequences: `PENDING_SPRINT_10:stop_sequences`
- Raw request/response retention status: `PENDING_SPRINT_10:raw_request_response_status`

## Metrics

### Persona Adherence

PA status: `PENDING_SPRINT_10:pa_status`.

The final report may present real PA only if the aggregate report shows pinned
semantic similarity, contradiction checks, and calibration evidence. Mock-only
values must remain diagnostic plumbing.

### Token-KL

Token-KL status: `PENDING_SPRINT_10:token_kl_status`.

Canonical Token-KL may be reported only when base and tuned models share
tokenizer/vocabulary compatibility, fixed prompt rendering, fixed continuation
scoring, and adequate comparable logprob depth.

### Behavioral-Consistency F1

BC-F1 summaries: `PENDING_SPRINT_10:bc_f1_summaries`.

Field-level scores must be reported separately:

- `stance_exact`
- `primary_action_exact`
- `secondary_modifiers_f1`
- `combined_score`, if used

## Statistical Methods

`PENDING_SPRINT_10`. Final statistics must aggregate variants and seeds inside
each persona before cross-persona inference.

Required final reporting:

- Mean: `PENDING_SPRINT_10:mean_fields`
- 95 percent confidence interval: `PENDING_SPRINT_10:ci95_fields`
- Absolute delta versus baseline: `PENDING_SPRINT_10:absolute_delta_fields`
- Effect size: `PENDING_SPRINT_10:effect_size_fields`
- P-value where applicable: `PENDING_SPRINT_10:p_value_fields`
- Flagged persona count and percentage: `PENDING_SPRINT_10:flagged_persona_summary`
- Multiple-comparison handling: `PENDING_SPRINT_10:multiple_comparison_handling`

## Artifact Freeze

`PENDING_SPRINT_10`.

The final methodology must record:

- Dataset hash: `PENDING_SPRINT_10:dataset_hash`
- Manifest hash: `PENDING_SPRINT_10:manifest_hash`
- Results hash: `PENDING_SPRINT_10:results_hash`
- Aggregate hash: `PENDING_SPRINT_10:aggregate_hash`
- Code commit: `PENDING_SPRINT_10:code_commit`
- Dirty worktree marker: `PENDING_SPRINT_10:dirty_worktree_status`
