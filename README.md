# Persona Drift Evaluation Harness

## Bottom Line

This repository is a fixture-first persona-drift evaluation harness. The current
implemented path validates the 10-row sample dataset, plans generation counts,
runs deterministic local mock outputs, and aggregates mock results at the
persona level. Sprint 5 adds deterministic readiness validators, review
manifests, and report schemas. It is still not ready to generate the
200-persona full dataset or run a 4,800-call full benchmark.

## Setup

Use Python 3.11 or newer.

```bash
python3 -m pip install -e ".[dev]"
```

The unit tests must not require network access, a GPU, vLLM, hosted APIs, or
external services.

## Validation Commands

```bash
python3 -m unittest discover -s tests
python3 -m pytest
python3 persona_eval.py validate --persona-path data/personas.sample.jsonl
python3 dataset_readiness.py --persona-path data/personas.sample.jsonl --review-manifest reviews/personas.sample.review.jsonl
```

Expected validation output:

```text
valid_persona_rows=10
dataset_readiness=blocked
```

## Planning Commands

Sample fixture arithmetic:

```bash
python3 persona_eval.py plan --persona-path data/personas.sample.jsonl --model-base base --model-tuned tuned --seeds 1
```

Expected output:

```text
planned_generation_calls=120
```

Smoke arithmetic:

```bash
python3 persona_eval.py plan --persona-count 20 --variants-per-persona 6 --model-count 2 --seed-count 1
```

Expected output:

```text
planned_generation_calls=240
```

Correct run counts:

- Sample: `10 personas * 6 variants * 2 models * 1 seed = 120 calls`.
- Smoke: `20 personas * 6 variants * 2 models * 1 seed = 240 calls`.
- Dev: `50 personas * 6 variants * 2 models * 2 seeds = 1,200 calls`.
- Full: `200 personas * 6 variants * 2 models * 2 seeds = 4,800 calls`.

## Mock Run

```bash
python3 persona_eval.py run --persona-path data/personas.sample.jsonl --out results/sprint5_mock --adapter mock --model-base base --model-tuned tuned --seeds 1 --run-id sprint5_mock
```

Expected output includes:

```text
planned_generation_calls=120
written_result_rows=60
manifest_path=results/sprint5_mock/manifest.json
results_path=results/sprint5_mock/results.jsonl
```

The result row count is `10 personas * 6 variants * 1 seed = 60` because each
row contains the matched base/tuned pair.

## Aggregation

```bash
python3 aggregate.py --manifest results/sprint5_mock/manifest.json --results results/sprint5_mock/results.jsonl --out reports/sprint5_mock
```

Expected output includes:

```text
aggregate_report_path=/Users/dyllon/Documents/persona-new/reports/sprint5_mock/aggregate_report.json
full_dataset_readiness=blocked
```

Aggregation writes:

- `reports/sprint5_mock/aggregate_report.json`
- `reports/sprint5_mock/chart_data/metric_availability.csv`
- `reports/sprint5_mock/chart_data/persona_shape.csv`
- `reports/sprint5_mock/chart_data/variant_type_breakdown.csv`

The report uses `persona_id` as the inference unit. Variants and seeds are
averaged inside each persona before cross-persona confidence intervals, deltas,
effect sizes, or p-values are computed.

## Token-KL Rules

Canonical Token-KL is aggregated only when `metrics.token_kl.status` is `ok`.
Unavailable rows are counted under `not_applicable`; endpoint-capped or
otherwise incomplete logprob paths are counted under `diagnostic_only`. Neither
state is coerced to zero.

Token-KL remains unavailable unless both models are scored on the same fixed
continuation with compatible tokenizer, vocabulary, chat template, and enough
logprob depth for the configured `k`.

## Persona Adherence Warning

Current Persona Adherence values are mock/plumbing only when present. The
aggregate report labels them as `mock_only_plumbing_not_real_persona_adherence`
and does not report them as real PA. Real PA remains blocked until an embedding
backend, contradiction judge, and calibration fixture are pinned.

## Full Dataset Gate

Do not create `data/personas.full.jsonl` until the readiness report is `ready`.
Run the readiness gate directly with:

```bash
python3 dataset_readiness.py --persona-path data/personas.sample.jsonl --review-manifest reviews/personas.sample.review.jsonl
```

Sprint 5 implements concrete local checks for:

- Sample schema tests pass.
- Variant validation tests pass.
- Source/license checks pass.
- PII and known-real-person indicators.
- Restricted-role indicators for real-person personas, medical decision roles,
  legal guarantee roles, self-harm roles, extremist roles, fraud/deception
  roles, and credential impersonation roles.
- Gold-label preview consistency between `expected_behavior` and
  `annotation.gold_labels`.
- Normalized exact duplicate detection.
- Deterministic near-duplicate detection using string similarity.
- Candidate boundary checks that block unexpected candidate/full files under
  `data/`.
- Review manifest schema validation.
- Low-confidence review coverage checks.
- Unvalidated candidates remain outside `data/`.

The review manifest is JSONL. Each row is validated by
`schemas/review_manifest.schema.json` and must include:

- `persona_id`
- `reviewer`
- `review_status`
- `reviewed_at`
- `review_reason`
- `low_confidence_flags`
- `semantic_equivalence_status`
- `nli_equivalence_status`
- `contradiction_status`
- `safety_review_status`
- `gold_label_review_status`

The current sample manifest is `reviews/personas.sample.review.jsonl`. It is a
contract fixture, not full-dataset approval evidence.

Full dataset generation is still blocked because semantic equivalence, NLI
equivalence, contradiction checks, embedding-cluster dedupe review, and
full-dataset review coverage remain `manual_required` or sample-scoped. These
states are intentional; mock or fixture checks must not be promoted to real
semantic validation.

Unvalidated candidate pools must stay outside `data/`. Use `candidates/` for
local candidate pools; it is ignored by git. Do not commit candidate pools or
`data/personas.full.jsonl` without explicit approval.

## Full Run Gate

Do not run the full `4,800`-call benchmark until a 20-persona smoke run has
completed and flagged outputs have been reviewed. Current non-dry staged runs
are capped at 20 personas.

## Autoresearch Verification

```bash
python3 auto_research.py run-once --tag sprint5 --description "Sprint 5 verification" -- python3 -m unittest discover -s tests
python3 auto_research.py summarize --tag sprint5
```

Autoresearch logs are local verification artifacts and are ignored by git.

## Known Limitations

- MockAdapter output is deterministic plumbing output, not a model benchmark.
- Full-dataset semantic/NLI validators and full-scope human-review evidence are
  incomplete.
- Token-KL has no canonical value for mock runs because aligned scoring is not
  available.
- PA is not real until pinned semantic and contradiction backends are added.

## Exact Next Command

After Sprint 5, the next implementation command is:

```bash
python3 dataset_readiness.py --persona-path data/personas.sample.jsonl --review-manifest reviews/personas.sample.review.jsonl --json
```

Use the JSON output to close the remaining `manual_required` gates before any
full dataset generation starts.
