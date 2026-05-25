# Persona Drift Evaluation Harness

## Bottom Line

This repository is a fixture-first persona-drift evaluation harness. The current
implemented path validates the 10-row sample dataset, plans generation counts,
runs deterministic local mock outputs, and aggregates mock results at the
persona level. It now includes deterministic candidate generation/selection,
promotion dry-run gates, Sprint 8 smoke-run scaffolding, and phase-aware
dev/full run guardrails. Sprint 7 promotion has produced the 200-row full
dataset. The next gate is Sprint 8: run a 20-persona real local-model smoke and
record smoke evidence before any 50-persona dev pass or 4,800-call full
benchmark.

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

## Candidate Generation

Sprint 6 candidate pools are deterministic local fixtures written outside
`data/`. The default fixture command writes 300 schema-valid candidates plus a
sidecar manifest with row count, candidate schema version, generator version,
source inventory, and output hash:

```bash
python3 candidate_generation.py create-fixtures --out candidates/sample_candidates.jsonl
python3 candidate_generation.py validate --candidate-path candidates/sample_candidates.jsonl
```

Expected validation output includes:

```text
valid_candidate_rows=300
manifest_valid=true
```

The candidate validator rejects duplicate candidate IDs, duplicate persona IDs,
missing provenance, missing variants, invalid variant counts, and any candidate
path under `data/`. Generated candidate pools and manifests remain ignored by
git; only `candidates/.gitignore` and `candidates/README.md` are tracked.

Batch R added a diversity selector for promotion prep:

```bash
python3 candidate_generation.py select \
  --candidate-path candidates/sprint7_candidates.jsonl \
  --out candidates/sprint7_selected_candidates.jsonl \
  --count 200
```

The selected candidate file is still not benchmark data. It must pass Sprint 7
promotion with full review evidence before anything is written under `data/`.

## Token-KL Rules

Canonical Token-KL is aggregated only when `metrics.token_kl.status` is `ok`.
Unavailable rows are counted under `not_applicable`; endpoint-capped or
otherwise incomplete logprob paths are counted under `diagnostic_only`. Neither
state is coerced to zero.

Token-KL remains unavailable unless both models are scored on the same fixed
continuation with compatible tokenizer, vocabulary, chat template, and enough
logprob depth for the configured `k`.

## Production/Open Model Matrix

The production/open model matrix is a local config template, not a run command:

```bash
python3 persona_eval.py validate-model-matrix --matrix-path configs/model_matrix.production_open.json
```

It separates same-family base/instruct drift pairs from standalone instruct
models and cross-family production comparisons. Same-family pairs may mark
Token-KL `canonical_possible` only as a conditional future state after aligned
fixed-continuation scoring proof. Standalone and cross-family comparisons keep
Token-KL `not_applicable` and use PA plus BC-F1 unless an explicit reviewed
paired-baseline exception is added.

## Persona Adherence Warning

Current Persona Adherence values are mock/plumbing only when present. The
aggregate report labels them as `mock_only_plumbing_not_real_persona_adherence`
and does not report them as real PA. Real PA remains blocked until an embedding
backend, contradiction judge, and calibration fixture are pinned.

## Full Dataset Gate

`data/personas.full.jsonl` exists only because the Sprint 7 promotion gate
passed. Future edits to that file must go through the same promotion path.
Run the readiness gate directly with:

```bash
python3 dataset_readiness.py --persona-path data/personas.full.jsonl --review-manifest reviews/personas.full.review.jsonl
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

The sample manifest is `reviews/personas.sample.review.jsonl`. It is a contract
fixture, not full-dataset approval evidence. The full review manifest is
`reviews/personas.full.review.jsonl` and covers the promoted 200-row dataset.
Semantic equivalence, NLI equivalence, contradiction, safety, and gold-label
evidence must cover every promoted row unless a real validator supplies
equivalent evidence. Mock or fixture checks must not be promoted to real
semantic validation.

Unvalidated candidate pools must stay outside `data/`. Use `candidates/` for
local candidate pools; it is ignored by git.

## Sprint 7 Promotion

Sprint 7 promotion scaffolding validates a proposed 200-row promotion set
without writing the full dataset when `--dry-run` is used:

```bash
python3 dataset_promotion.py validate-candidates --candidate-path candidates/approved_candidates.jsonl --review-path reviews/personas.full.review.jsonl
python3 dataset_promotion.py promote --candidate-path candidates/approved_candidates.jsonl --review-path reviews/personas.full.review.jsonl --out data/personas.full.jsonl --dry-run
```

Promotion remains blocked unless the candidate set has exactly 200 valid rows,
all required review evidence is present, low-confidence rows are reviewed,
restricted categories are absent, duplicate candidates are absent, and all
manual semantic/NLI/contradiction/safety/gold-label gates have passing evidence.
Non-dry writes are refused unless every gate passes.

Current promoted inputs are:

- Candidate pool: `candidates/sprint7_selected_candidates.jsonl`
- Full review manifest: `reviews/personas.full.review.jsonl`
- Promotion manifest: `reports/dataset_promotion_manifest.json`
- Full dataset: `data/personas.full.jsonl`

To recheck the promotion gate without writing:

```bash
python3 dataset_promotion.py promote \
  --candidate-path candidates/sprint7_selected_candidates.jsonl \
  --review-path reviews/personas.full.review.jsonl \
  --out data/personas.full.jsonl \
  --dry-run
```

## Dev And Full Run Gates

Do not run the 50-persona dev pass until:

- Sprint 8 smoke evidence exists and flagged outputs have been reviewed.
- A dev approval/override artifact exists.

Do not run the full `4,800`-call benchmark until the dev pass has completed and
explicit full-run approval exists. Non-dry run caps are phase-aware: smoke is
capped at 20 personas, dev at 50, and full at 200, with evidence gates for each
phase.

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

After Sprint 7 promotion, the next implementation task is Sprint 8 smoke
evidence. First validate the promoted dataset and promotion evidence:

```bash
python3 persona_eval.py validate --persona-path data/personas.full.jsonl
python3 dataset_readiness.py --persona-path data/personas.full.jsonl --review-manifest reviews/personas.full.review.jsonl
```

Then follow `runbooks/vllm_smoke.md` with an explicitly approved local vLLM
endpoint and pinned runtime metadata.
