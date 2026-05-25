# Dataset Card Skeleton

## Bottom Line

Status: `PENDING_SPRINT_10`. This dataset card is a scaffold for the final
validated benchmark dataset. It must not describe `data/personas.full.jsonl` as
ready until promotion, validation, review evidence, and final-run hashes exist.

## Evidence Classification

### Final-run evidence

`PENDING_SPRINT_10`. Final dataset claims must come from the promoted dataset,
promotion manifest, review manifest, readiness report, and final run manifest.

### Diagnostic/pilot evidence

Diagnostic/pilot evidence may describe sample fixtures or candidate plumbing
only. It must not be used as evidence that the full dataset is benchmark-ready.

## Dataset Identity

| Field | Value |
| --- | --- |
| Dataset name | `PENDING_SPRINT_10:dataset_name` |
| Dataset path | `data/personas.full.jsonl` |
| Dataset hash | `PENDING_SPRINT_10:dataset_hash` |
| Persona count | `PENDING_SPRINT_10:persona_count` |
| Variants per persona | `PENDING_SPRINT_10:variants_per_persona` |
| Promotion manifest hash | `PENDING_SPRINT_10:promotion_manifest_hash` |
| Review manifest hash | `PENDING_SPRINT_10:review_manifest_hash` |
| Code commit | `PENDING_SPRINT_10:code_commit` |

## Intended Use

`PENDING_SPRINT_10`. Fill after the final dataset is frozen. The intended use
must be limited to persona-drift evaluation under the documented metric
contracts.

## Source And License Metadata

`PENDING_SPRINT_10`.

Every row must carry:

- `source.dataset`
- `source.source_url`
- `source.license`
- `source.license_url`
- `source.split`
- `source.source_persona_id`
- `source.retrieved_at`
- `source.revision_or_hash`
- `source.modification_notes`
- `source.redistribution_notes`

Source/license inventory: `PENDING_SPRINT_10:source_license_inventory`.

## Dataset Construction

`PENDING_SPRINT_10`.

- Candidate source: `PENDING_SPRINT_10:candidate_source`
- Promotion command: `PENDING_SPRINT_10:promotion_command`
- Validation command: `PENDING_SPRINT_10:validation_command`
- Rejection counts and reasons: `PENDING_SPRINT_10:rejection_counts`
- Human-review coverage: `PENDING_SPRINT_10:human_review_coverage`

## Schema And Variant Contract

`PENDING_SPRINT_10`.

Expected executable variants:

- `canonical`
- `paraphrase`
- `negation_preserving`
- `distractor`
- `instruction_prefix`
- `temperature_robust`

Final variant validation status: `PENDING_SPRINT_10:variant_validation_status`.

## Safety And Exclusion Filters

`PENDING_SPRINT_10`.

Final status fields:

- Real-person persona filter: `PENDING_SPRINT_10:real_person_filter_status`
- Medical decision role filter: `PENDING_SPRINT_10:medical_role_filter_status`
- Legal guarantee role filter: `PENDING_SPRINT_10:legal_role_filter_status`
- Self-harm role filter: `PENDING_SPRINT_10:self_harm_filter_status`
- Extremist role filter: `PENDING_SPRINT_10:extremist_filter_status`
- Fraud/deception role filter: `PENDING_SPRINT_10:fraud_filter_status`
- Credential impersonation filter: `PENDING_SPRINT_10:credential_filter_status`

## Known Dataset Limitations

`PENDING_SPRINT_10`. Link final limitations to `LIMITATIONS.md` after Sprint 10.
