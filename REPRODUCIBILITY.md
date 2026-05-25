# Reproducibility Skeleton

## Bottom Line

Status: `PENDING_SPRINT_10`. This file defines the reproducibility record and
commands, but final hashes and model metadata are pending until the frozen Sprint
10 artifacts exist.

## Evidence Classification

### Final-run evidence

`PENDING_SPRINT_10`. Final-run evidence is limited to frozen full-run artifacts
and their exact hashes.

### Diagnostic/pilot evidence

Diagnostic/pilot evidence can show that local checks and mock aggregation run,
but it cannot be used as final benchmark evidence.

## Required Final Artifact Ledger

| Field | Command or source | Final value |
| --- | --- | --- |
| Dataset hash | `shasum -a 256 data/personas.full.jsonl` | `PENDING_SPRINT_10:dataset_hash` |
| Manifest hash | `shasum -a 256 results/full_200/manifest.json` | `PENDING_SPRINT_10:manifest_hash` |
| Results hash | `shasum -a 256 results/full_200/results.jsonl` | `PENDING_SPRINT_10:results_hash` |
| Aggregate hash | `shasum -a 256 reports/full_200/aggregate_report.json` | `PENDING_SPRINT_10:aggregate_hash` |
| Code commit | `git rev-parse HEAD` | `PENDING_SPRINT_10:code_commit` |
| Dirty worktree marker | `git status --short` | `PENDING_SPRINT_10:dirty_worktree_status` |
| Model IDs | final run manifest | `PENDING_SPRINT_10:model_ids` |
| Tokenizer/chat-template metadata | final run manifest | `PENDING_SPRINT_10:tokenizer_chat_template_metadata` |

## Environment

`PENDING_SPRINT_10`.

Required fields:

- Python version: `PENDING_SPRINT_10:python_version`
- Package install command: `python3 -m pip install -e ".[dev]"`
- vLLM version: `PENDING_SPRINT_10:vllm_version`
- GPU/CUDA/driver: `PENDING_SPRINT_10:gpu_cuda_driver`
- Operating system/runtime notes: `PENDING_SPRINT_10:runtime_notes`

## Local Checks

These commands are expected to remain local and must not require network, API,
GPU, or real model access:

```bash
python3 -m unittest discover -s tests
python3 -m pytest
python3 persona_eval.py validate --persona-path data/personas.sample.jsonl
python3 dataset_readiness.py --persona-path data/personas.sample.jsonl --review-manifest reviews/personas.sample.review.jsonl
```

Diagnostic status: `PENDING_SPRINT_10:local_check_status`.

## Final Rerun Commands

Manual final rerun status: `PENDING_SPRINT_10`.

Run only after Sprint 10 gates are satisfied and the user has approved the full
benchmark:

```bash
python3 persona_eval.py run \
  --persona-path data/personas.full.jsonl \
  --out results/full_200 \
  --adapter vllm \
  --base-url <local-vllm-url> \
  --model-base <base-model-id> \
  --model-tuned <tuned-model-id> \
  --seeds 1,2 \
  --run-id full_200
```

Recompute aggregation:

```bash
python3 aggregate.py \
  --manifest results/full_200/manifest.json \
  --results results/full_200/results.jsonl \
  --out reports/full_200_recomputed
```

Recomputed aggregate hash: `PENDING_SPRINT_10:recomputed_aggregate_hash`.

## Metadata Verification Commands

Run after final artifacts exist:

```bash
git rev-parse HEAD
git status --short
shasum -a 256 data/personas.full.jsonl
shasum -a 256 results/full_200/manifest.json
shasum -a 256 results/full_200/results.jsonl
shasum -a 256 reports/full_200/aggregate_report.json
```

Verification status: `PENDING_SPRINT_10:hash_verification_status`.

## Metric Status Cross-Checks

`PENDING_SPRINT_10`.

Required final checks:

- PA status matches aggregate report: `PENDING_SPRINT_10:pa_status`
- Token-KL status matches aggregate report: `PENDING_SPRINT_10:token_kl_status`
- BC-F1 summaries match aggregate report: `PENDING_SPRINT_10:bc_f1_summaries`
- Result-dependent report sections are copied from aggregate JSON:
  `PENDING_SPRINT_10:report_consistency_status`

## Reproducibility Risks

`PENDING_SPRINT_10`. Fill only from final run metadata and observed rerun
behavior.
