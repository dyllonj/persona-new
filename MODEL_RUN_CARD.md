# Model Run Card Skeleton

## Bottom Line

Status: `PENDING_SPRINT_10`. This model-run card is a scaffold for the final
full benchmark run. It must be filled from the final run manifest, result rows,
and aggregate report only.

## Evidence Classification

### Final-run evidence

`PENDING_SPRINT_10`. Final model-run evidence must come from
`results/full_200/manifest.json`, `results/full_200/results.jsonl`, and
`reports/full_200/aggregate_report.json`.

### Diagnostic/pilot evidence

Diagnostic/pilot evidence can describe mock, smoke, or dev-run behavior only. It
must not be presented as final benchmark behavior.

## Run Identity

| Field | Value |
| --- | --- |
| Run ID | `PENDING_SPRINT_10:run_id` |
| Timestamp UTC | `PENDING_SPRINT_10:timestamp_utc` |
| Code commit | `PENDING_SPRINT_10:code_commit` |
| Dirty worktree marker | `PENDING_SPRINT_10:dirty_worktree_status` |
| Dataset hash | `PENDING_SPRINT_10:dataset_hash` |
| Manifest hash | `PENDING_SPRINT_10:manifest_hash` |
| Results hash | `PENDING_SPRINT_10:results_hash` |
| Aggregate hash | `PENDING_SPRINT_10:aggregate_hash` |

## Model Identity

| Field | Base | Tuned |
| --- | --- | --- |
| Model IDs | `PENDING_SPRINT_10:base_model_id` | `PENDING_SPRINT_10:tuned_model_id` |
| Model revision/hash | `PENDING_SPRINT_10:base_model_revision_or_hash` | `PENDING_SPRINT_10:tuned_model_revision_or_hash` |
| Tokenizer name | `PENDING_SPRINT_10:base_tokenizer_name` | `PENDING_SPRINT_10:tuned_tokenizer_name` |
| Tokenizer hash | `PENDING_SPRINT_10:base_tokenizer_hash` | `PENDING_SPRINT_10:tuned_tokenizer_hash` |
| Chat-template hash | `PENDING_SPRINT_10:base_chat_template_hash` | `PENDING_SPRINT_10:tuned_chat_template_hash` |

Tokenizer/chat-template metadata:
`PENDING_SPRINT_10:tokenizer_chat_template_metadata`.

## Runtime Configuration

`PENDING_SPRINT_10`.

- Adapter: `PENDING_SPRINT_10:adapter`
- Provider or endpoint: `PENDING_SPRINT_10:provider_or_endpoint`
- Serving stack and version: `PENDING_SPRINT_10:serving_stack_version`
- GPU/CUDA/driver: `PENDING_SPRINT_10:gpu_cuda_driver`
- Decoding parameters: `PENDING_SPRINT_10:decoding_params`
- Seeds: `PENDING_SPRINT_10:seeds`
- Stop sequences: `PENDING_SPRINT_10:stop_sequences`
- Prompt template version/hash: `PENDING_SPRINT_10:prompt_template_version_hash`
- Raw request/response retention: `PENDING_SPRINT_10:raw_request_response_status`

## Scoring Capability

`PENDING_SPRINT_10`.

- PA status: `PENDING_SPRINT_10:pa_status`
- Token-KL status: `PENDING_SPRINT_10:token_kl_status`
- BC-F1 summaries: `PENDING_SPRINT_10:bc_f1_summaries`
- Scoring path: `PENDING_SPRINT_10:scoring_path`
- Fixed continuation ID/hash: `PENDING_SPRINT_10:fixed_continuation_metadata`
- Logprob depth `k`: `PENDING_SPRINT_10:token_kl_k`
- Endpoint cap: `PENDING_SPRINT_10:endpoint_cap`

## Aggregate Availability Summary

`PENDING_SPRINT_10`. Copy from the final aggregate report only.

| Metric | Availability counts | Status |
| --- | --- | --- |
| Persona Adherence | `PENDING_SPRINT_10:pa_availability_counts` | `PENDING_SPRINT_10:pa_status` |
| Token-KL | `PENDING_SPRINT_10:token_kl_availability_counts` | `PENDING_SPRINT_10:token_kl_status` |
| Behavioral-Consistency F1 | `PENDING_SPRINT_10:bc_f1_availability_counts` | `PENDING_SPRINT_10:bc_f1_summaries` |

## Run Limitations

`PENDING_SPRINT_10`. Link to `LIMITATIONS.md` after the final run is frozen.
