# Final Report Skeleton

## Bottom Line

Status: `PENDING_SPRINT_10`. This file is a publication scaffold only. It must
not contain final benchmark findings until the frozen Sprint 10 dataset, run
manifest, result rows, and aggregate report exist and their hashes are recorded.

Recommended next action: after Sprint 10, fill this report from the frozen
aggregate JSON and artifact hashes. Do not hand-enter metric values.

## Evidence Classification

### Final-run evidence

`PENDING_SPRINT_10`. Final-run evidence must come only from the frozen full run:

- Dataset: `data/personas.full.jsonl`
- Run manifest: `results/full_200/manifest.json`
- Result rows: `results/full_200/results.jsonl`
- Aggregate report: `reports/full_200/aggregate_report.json`
- Generated charts/tables: `reports/full_200/chart_data/`

### Diagnostic/pilot evidence

Diagnostic or pilot evidence may be used only to describe harness behavior,
readiness checks, or dry-run plumbing. It must not be described as a full
benchmark result. Current mock artifacts, if referenced, are diagnostic only:

- `results/sprint5_mock/`
- `reports/sprint5_mock/`

## Artifact Hash Ledger

| Artifact | Expected path or source | Hash or identifier | Evidence class | Status |
| --- | --- | --- | --- | --- |
| Dataset hash | `data/personas.full.jsonl` | `PENDING_SPRINT_10:dataset_hash` | Final-run evidence | `PENDING_SPRINT_10` |
| Run manifest hash | `results/full_200/manifest.json` | `PENDING_SPRINT_10:manifest_hash` | Final-run evidence | `PENDING_SPRINT_10` |
| Results hash | `results/full_200/results.jsonl` | `PENDING_SPRINT_10:results_hash` | Final-run evidence | `PENDING_SPRINT_10` |
| Aggregate hash | `reports/full_200/aggregate_report.json` | `PENDING_SPRINT_10:aggregate_hash` | Final-run evidence | `PENDING_SPRINT_10` |
| Code commit | `git rev-parse HEAD` at final freeze | `PENDING_SPRINT_10:code_commit` | Final-run evidence | `PENDING_SPRINT_10` |
| Dirty worktree marker | `git status --short` at final freeze | `PENDING_SPRINT_10:dirty_worktree_status` | Final-run evidence | `PENDING_SPRINT_10` |

## Research Question

`PENDING_SPRINT_10`. Fill with the final frozen comparison question before
results are interpreted. The question must name the base model, tuned model,
dataset version, prompt template version, and metrics.

## Dataset And Validation

`PENDING_SPRINT_10`.

Required final fields:

- Dataset hash: `PENDING_SPRINT_10:dataset_hash`
- Persona count: `PENDING_SPRINT_10:persona_count`
- Variant count per persona: `PENDING_SPRINT_10:variant_count`
- Source/license inventory: `PENDING_SPRINT_10:source_license_inventory`
- Review manifest hash: `PENDING_SPRINT_10:review_manifest_hash`
- Promotion manifest hash: `PENDING_SPRINT_10:promotion_manifest_hash`
- Validation gate status: `PENDING_SPRINT_10:dataset_readiness_status`

## Model And Runtime Configuration

`PENDING_SPRINT_10`.

Required final fields:

- Model IDs: `PENDING_SPRINT_10:model_ids`
- Model revisions or hashes: `PENDING_SPRINT_10:model_revisions_or_hashes`
- Tokenizer/chat-template metadata: `PENDING_SPRINT_10:tokenizer_chat_template_metadata`
- Serving stack and version: `PENDING_SPRINT_10:serving_stack_version`
- GPU/CUDA/driver metadata: `PENDING_SPRINT_10:gpu_cuda_driver`
- Decoding parameters: `PENDING_SPRINT_10:decoding_params`
- Seeds: `PENDING_SPRINT_10:seeds`
- Stop sequences: `PENDING_SPRINT_10:stop_sequences`

## Metric Availability

`PENDING_SPRINT_10`. Copy availability counts from the final aggregate report
only.

| Metric | Required summary | Final status |
| --- | --- | --- |
| PA status | `PA-mean`, `PA-pass@threshold`, `fact_contradiction_rate`, backend revisions, calibration hash | `PENDING_SPRINT_10:pa_status` |
| Token-KL status | `ok`, `diagnostic_only`, `not_applicable` counts, scoring path, `k`, endpoint cap | `PENDING_SPRINT_10:token_kl_status` |
| BC-F1 summaries | `stance_exact`, `primary_action_exact`, `secondary_modifiers_f1`, `combined_score` | `PENDING_SPRINT_10:bc_f1_summaries` |

## Results

`PENDING_SPRINT_10`. Do not fill this section until Sprint 10 artifacts exist.
Every value must be copied from `reports/full_200/aggregate_report.json` or
generated chart data.

| Result-dependent field | Value | Source |
| --- | --- | --- |
| Persona Adherence mean | `PENDING_SPRINT_10:pa_mean` | `PENDING_SPRINT_10:aggregate_report_path` |
| Persona Adherence pass rate | `PENDING_SPRINT_10:pa_pass_at_threshold` | `PENDING_SPRINT_10:aggregate_report_path` |
| Fact contradiction rate | `PENDING_SPRINT_10:fact_contradiction_rate` | `PENDING_SPRINT_10:aggregate_report_path` |
| Canonical Token-KL | `PENDING_SPRINT_10:token_kl_value_or_not_applicable` | `PENDING_SPRINT_10:aggregate_report_path` |
| BC-F1 stance exact | `PENDING_SPRINT_10:bc_f1_stance_exact` | `PENDING_SPRINT_10:aggregate_report_path` |
| BC-F1 primary action exact | `PENDING_SPRINT_10:bc_f1_primary_action_exact` | `PENDING_SPRINT_10:aggregate_report_path` |
| BC-F1 modifier F1 | `PENDING_SPRINT_10:bc_f1_secondary_modifiers_f1` | `PENDING_SPRINT_10:aggregate_report_path` |

## Statistical Reporting

`PENDING_SPRINT_10`. Final statistics must use `persona_id` as the inference
unit. Include mean, 95 percent confidence interval, absolute delta versus
baseline, effect size, p-value where applicable, flagged persona count and
percentage, and multiple-comparison handling for secondary cuts.

## Failure Analysis And Flagged Examples

`PENDING_SPRINT_10`. Include only examples traceable to final result rows.
Separate:

- Final-run flagged examples: `PENDING_SPRINT_10:final_flagged_examples`
- Diagnostic/pilot examples: `PENDING_SPRINT_10:diagnostic_examples_if_used`

## Limitations

See `LIMITATIONS.md`. Result-dependent limitations remain
`PENDING_SPRINT_10` until the final aggregate report and model-run card are
available.

## Release Notes Skeleton

`PENDING_SPRINT_10`. Fill from commit history and sprint artifacts after the
final freeze.

| Sprint range | Change summary | Evidence class | Status |
| --- | --- | --- | --- |
| Sprints 0-5 | `PENDING_SPRINT_10:sprints_0_5_summary` | Diagnostic/pilot evidence | `PENDING_SPRINT_10` |
| Sprint 6 | `PENDING_SPRINT_10:sprint_6_summary` | Diagnostic/pilot or final-run support | `PENDING_SPRINT_10` |
| Sprint 7 | `PENDING_SPRINT_10:sprint_7_summary` | Final dataset support | `PENDING_SPRINT_10` |
| Sprint 8 | `PENDING_SPRINT_10:sprint_8_summary` | Diagnostic/pilot evidence | `PENDING_SPRINT_10` |
| Sprint 9 | `PENDING_SPRINT_10:sprint_9_summary` | Diagnostic/pilot evidence | `PENDING_SPRINT_10` |
| Sprint 10 | `PENDING_SPRINT_10:sprint_10_summary` | Final-run evidence | `PENDING_SPRINT_10` |
| Sprint 11 | `PENDING_SPRINT_10:sprint_11_summary` | Reporting package | `PENDING_SPRINT_10` |

## Fill Rules

- Do not invent results.
- Do not claim full benchmark findings from mock, smoke, or dev artifacts.
- Do not report canonical Token-KL unless aligned scoring is proven and
  `token_kl.status=ok` in the final aggregate report.
- Do not report Persona Adherence as real if the final aggregate report marks it
  `mock_only`, `not_run`, or `not_applicable`.
- Do not manually edit chart values; regenerate them from aggregate JSON.
