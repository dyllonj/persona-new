# Remaining Sprints Plan

Status: draft for Sprints 6-11
Last updated: 2026-05-25

## Bottom Line

After Sprint 5, the project shifts from harness plumbing to controlled dataset
construction, staged real-model execution, and a final reproducibility package.
Do not create `data/personas.full.jsonl` or run the full `4,800`-call benchmark
until the readiness gates say those actions are allowed.

This file starts at Sprint 6 because Sprint 5 is already underway.

The intended sequence is:

- Sprint 6 creates auditable candidates, not benchmark data.
- Sprint 7 validates, reviews, and promotes exactly 200 benchmark personas.
- Sprint 8 proves a 20-persona real local-model smoke run.
- Sprint 9 proves a 50-persona dev run and operational stability.
- Sprint 10 runs the frozen 200-persona benchmark.
- Sprint 11 packages the final report, limitations, and reproducibility record.

## Global Hard Gates

These apply to every remaining sprint:

- Do not weaken `AGENTS.MD`.
- Do not revert unrelated user or agent changes.
- Use `python3`, not `python`.
- Commit after coherent implementation slices, after relevant checks pass.
- Do not run real models before Sprint 8.
- Do not run the full benchmark before Sprint 10.
- Keep unvalidated candidates outside `data/`.
- Do not place candidate pools under `data/`.
- Treat `data/` as validated benchmark input only.
- Do not write `data/personas.full.jsonl` before Sprint 7 promotion passes.
- Do not mark full dataset readiness as `ready` without real evidence.
- Do not require network, API, or GPU access in unit tests.
- Do not treat generation rows as independent observations.
- Aggregate by `persona_id`.
- Do not report canonical Token-KL unless aligned scoring is proven.
- Do not coerce unavailable or diagnostic metrics to zero.
- Keep vLLM as the default local real-run path.
- Treat TGI as secondary production-parity support only.
- Keep generated outputs ignored or untracked unless the user explicitly approves
  committing them.
- Pin or record code version, dataset hash, model identifiers, tokenizer details,
  and runtime configuration for every real run.

## Agent Ownership Model

Use separate agents where possible:

- Dataset agent: candidate generation, schemas, provenance, and promotion
  mechanics.
- Validation agent: safety filters, dedupe, semantic-equivalence checks, review
  manifests, and readiness evidence.
- Run-ops agent: vLLM configs, adapter contracts, runtime manifests, retries,
  and smoke/dev/full execution commands.
- Reporting agent: aggregation, statistics, charts/tables, limitations, and
  reproducibility package.

The agents should treat Sprint 5 outputs as the boundary contract. If Sprint 5
renames `dataset_readiness.py`, schemas, or readiness report fields, update the
command names in this file without changing the gates.

## Sprint 6: Candidate Generation Pipeline

### Objective

Create a deterministic candidate-generation pipeline outside `data/`. The goal is
to produce an auditable candidate pool that Sprint 7 can validate and review. Do
not promote any candidate into `data/personas.full.jsonl` in this sprint.

Target `300-500` candidates, with a hard minimum above `200`, because validation,
dedupe, semantic checks, and human review should be expected to reject rows.

### Target Files

- `candidate_generation.py`
- `schemas/candidate_persona.schema.json`
- `candidates/.gitignore`
- `candidates/README.md`
- `tests/test_candidate_generation.py`
- README updates if needed

### Implementation Tasks

- Implement candidate output under `candidates/`, not `data/`.
- Add `candidates/` or `candidate_pools/` to ignore rules unless the user
  explicitly approves committing a pool.
- Generate or import candidate persona rows with all source/provenance fields
  needed for eventual promotion.
- Write a candidate manifest for every pool, including row count, schema version,
  source inventory, generator version, and file hash.
- Preserve the six-variant convention:
  - `canonical`
  - `paraphrase`
  - `negation_preserving`
  - `distractor`
  - `instruction_prefix`
  - `temperature_robust`
- Add candidate metadata:
  - `candidate_id`
  - `generation_method`
  - `created_at`
  - `generator_version`
  - `source_trace`
  - `validation_status`
  - `promotion_status`
  - `low_confidence_flags`
- Support deterministic local fixture generation for tests.
- If model-assisted generation is added, it must be opt-in and disabled in
  tests.
- If model-assisted generation is used, log raw prompts, raw responses, model
  IDs, model versions, generation parameters, and post-processing decisions
  outside `data/`.
- Use public, permissively licensed, or explicitly approved sources only.
- Flag uncertain source claims, weak behavior labels, ambiguous variants, and
  incomplete provenance in `low_confidence_flags`.
- Add CLI commands if useful:
  - `python3 candidate_generation.py create-fixtures --out candidates/sample_candidates.jsonl`
  - `python3 candidate_generation.py validate --candidate-path ...`

### Tests

- Candidate files are written outside `data/`.
- Candidate rows validate against `schemas/candidate_persona.schema.json`.
- Candidate rows contain full source/provenance fields.
- Candidate rows preserve exactly six variants.
- Candidate IDs are unique.
- Candidate generation is deterministic in fixture mode.
- No network/API/GPU is required.
- Existing Sprint 0-5 tests still pass.

### Verification Commands

```bash
python3 -m unittest discover -s tests
python3 -m pytest
python3 candidate_generation.py create-fixtures --out candidates/sample_candidates.jsonl
python3 candidate_generation.py validate --candidate-path candidates/sample_candidates.jsonl
python3 aggregate.py --manifest results/sprint5_mock/manifest.json --results results/sprint5_mock/results.jsonl --out reports/sprint6_check
```

### Acceptance

- Candidate generation produces schema-valid candidate rows outside `data/`.
- Candidate pool contains more than 200 rows, preferably `300-500`.
- No candidate is promoted to the benchmark dataset.
- Candidate manifest exists and hashes the candidate output.
- Full dataset readiness remains blocked unless Sprint 7 evidence already
  exists.

### Stop Conditions

- Stop if candidates must be written under `data/`.
- Stop if generation requires network/API/GPU without explicit user approval.
- Stop if candidate provenance cannot be represented.
- Stop if the pipeline cannot generate enough candidates to tolerate validation
  attrition.

## Sprint 7: Dataset Validation And Promotion

### Objective

Validate candidate rows, collect review evidence, and promote exactly 200
approved rows into `data/personas.full.jsonl` only if the full dataset gate is
satisfied.

This is the only remaining sprint that should create or modify
`data/personas.full.jsonl`.

### Target Files

- `dataset_promotion.py`
- `schemas/review_manifest.schema.json`
- `reviews/personas.full.review.jsonl`
- `data/personas.full.jsonl`, only if promotion gate passes
- `tests/test_dataset_promotion.py`
- README updates

### Implementation Tasks

- Validate candidate rows using Sprint 5 readiness validators.
- Require review evidence for:
  - all low-confidence rows
  - at least 10 percent of all promoted rows
- Require review records to include reviewer identifier, review date, status,
  decision reason, and any unresolved concerns.
- Enforce safety filters:
  - real-person personas
  - medical decision roles
  - legal guarantee roles
  - self-harm roles
  - extremist roles
  - fraud/deception roles
  - credential impersonation roles
- Enforce dedupe:
  - normalized exact text
  - near-duplicate prompt/persona text
  - embedding-cluster review status or documented blocked status
- Require semantic equivalence evidence for controlled variants.
- If semantic/NLI evidence is unavailable for a row, require documented manual
  equivalence review before promotion.
- Require gold-label preview checks.
- Require source/license/provenance validation.
- Promote only validated rows.
- Promote exactly 200 rows. Do not fill shortfall with unreviewed candidates.
- Write a promotion manifest with:
  - candidate input hash
  - promoted output hash
  - validator versions
  - review manifest hash
  - rejection counts and reasons

### Tests

- Invalid candidates are rejected.
- Missing review evidence blocks promotion.
- Low-confidence rows without review are rejected.
- Duplicate candidates are rejected.
- Restricted-role candidates are rejected.
- Promotion cannot write fewer or more than 200 rows unless explicitly in dry-run
  mode.
- Promoted rows validate with `persona_eval.py validate`.
- `aggregate.full_dataset_readiness()` changes only when real evidence exists.

### Verification Commands

```bash
python3 -m unittest discover -s tests
python3 -m pytest
python3 dataset_promotion.py validate-candidates --candidate-path candidates/approved_candidates.jsonl --review-path reviews/personas.full.review.jsonl
python3 dataset_promotion.py promote --candidate-path candidates/approved_candidates.jsonl --review-path reviews/personas.full.review.jsonl --out data/personas.full.jsonl --dry-run
python3 persona_eval.py validate --persona-path data/personas.sample.jsonl
```

If promotion is actually allowed:

```bash
python3 dataset_promotion.py promote --candidate-path candidates/approved_candidates.jsonl --review-path reviews/personas.full.review.jsonl --out data/personas.full.jsonl
python3 persona_eval.py validate --persona-path data/personas.full.jsonl
```

### Acceptance

- Either promotion remains blocked with exact reasons, or
  `data/personas.full.jsonl` exists with exactly 200 validated rows and a
  promotion manifest.
- Dataset hash and review evidence are frozen for downstream runs.
- No unreviewed or unsafe candidates enter the full dataset.

### Stop Conditions

- Stop if review coverage is incomplete.
- Stop if safety or dedupe checks are missing.
- Stop if source/license provenance is ambiguous.
- Stop if exactly 200 valid rows cannot be promoted without lowering standards.

## Sprint 8: 20-Persona Real vLLM Smoke

### Objective

Run a controlled 20-persona real-model smoke test using local vLLM or an
explicit OpenAI-compatible endpoint. This proves real execution without running
the full benchmark.

Hosted APIs are out of scope by default. Use a local endpoint unless the user
explicitly approves a different runtime.

### Target Files

- `configs/smoke.vllm.json`
- `runbooks/vllm_smoke.md`
- optional adapter refinements in `persona_eval.py`
- `tests/test_vllm_adapter_contract.py`
- README updates

### Implementation Tasks

- Require explicit user-provided model IDs and endpoints.
- Use `--limit-personas 20`.
- Refuse to proceed unless `data/personas.full.jsonl` exists and validates.
- Record the dataset hash and promotion manifest hash in the run manifest.
- Record:
  - raw requests/responses
  - stop reason
  - truncation flag
  - usage
  - latency
  - model revision/hash
  - tokenizer hash
  - chat-template hash
  - serving stack/version
- Keep Token-KL `not_applicable` unless aligned scoring is implemented.
- Token-KL may move out of `not_applicable` only if the adapter proves fixed
  aligned continuation scoring, with identical prompt/context and scored
  continuation across both models.
- Add manual inspection workflow for flagged outputs.
- Generate aggregate report for the smoke run.

### Tests

- Adapter contract tests use mocked HTTP only.
- No real vLLM/network call in unit tests.
- Smoke command refuses to run more than 20 personas.
- Missing model IDs/endpoints fail clearly.
- Token-KL remains unavailable unless scoring capability is proven.
- Aggregate report uses persona-level inference.
- Raw request/response logging is covered by contract tests without real model
  calls.

### Verification Commands

Unit/local:

```bash
python3 -m unittest discover -s tests
python3 -m pytest
python3 persona_eval.py plan --persona-count 20 --variants-per-persona 6 --model-count 2 --seed-count 1
```

Manual real smoke, only with user-approved local endpoint:

```bash
python3 persona_eval.py run --persona-path data/personas.full.jsonl --limit-personas 20 --out results/vllm_smoke_20 --adapter vllm --base-url http://localhost:8000/v1 --model-base <base-model> --model-tuned <tuned-model> --seeds 1 --run-id vllm_smoke_20
python3 aggregate.py --manifest results/vllm_smoke_20/manifest.json --results results/vllm_smoke_20/results.jsonl --out reports/vllm_smoke_20
```

### Acceptance

- 20-persona smoke run completes or fails with a clear adapter/runtime reason.
- Aggregate report is produced if the run completes.
- Flagged outputs are reviewed before any dev run.
- Runtime, tokenizer, chat-template, dataset, and code metadata are present in
  the manifest.

### Stop Conditions

- Stop if local endpoint is unavailable.
- Stop if raw request/response logging is incomplete.
- Stop if output inspection reveals schema or prompt-rendering defects.
- Stop if Token-KL scoring cannot guarantee fixed aligned continuation but code
  attempts to report it as canonical.

## Sprint 9: 50-Persona Dev Run

### Objective

Run the dev benchmark: `50 personas * 6 variants * 2 models * 2 seeds = 1,200`
generation calls. Use this to stabilize runtime, output quality, and aggregation
before a full run.

### Target Files

- `configs/dev.vllm.json`
- `runbooks/dev_run.md`
- report artifacts under `reports/dev_*`, generated and ignored unless the user
  asks to preserve them
- README updates

### Implementation Tasks

- Require successful Sprint 8 smoke report and review evidence.
- Run exactly 50 personas and 2 seeds.
- Use deterministic run IDs.
- Track failure/retry policy.
- Aggregate the run.
- Allow prompt, runtime, or adapter tuning only if cache invalidation and rerun
  requirements are documented.
- Review:
  - truncation rates
  - adapter errors
  - behavior tag ambiguity
  - PA mock/real status
  - Token-KL availability
  - per-variant breakdowns
- Decide whether full run is allowed.
- Record the full-run go/no-go decision in a reviewed approval artifact.

### Tests

- Dev plan count equals `1,200`.
- Dev run command refuses to proceed without prior smoke evidence or explicit
  override.
- Aggregation remains persona-level.
- Retry/failure metadata remains auditable.

### Verification Commands

```bash
python3 -m unittest discover -s tests
python3 -m pytest
python3 persona_eval.py plan --persona-count 50 --variants-per-persona 6 --model-count 2 --seed-count 2
```

Manual dev run:

```bash
python3 persona_eval.py run --persona-path data/personas.full.jsonl --limit-personas 50 --out results/dev_50 --adapter vllm --base-url <local-vllm-url> --model-base <base-model> --model-tuned <tuned-model> --seeds 1,2 --run-id dev_50
python3 aggregate.py --manifest results/dev_50/manifest.json --results results/dev_50/results.jsonl --out reports/dev_50
```

### Acceptance

- Dev run completes and aggregates.
- Failure and ambiguity rates are low enough for full-run approval.
- Full-run approval is recorded explicitly.
- Any changed prompt/runtime/adapter setting has a documented rerun boundary.

### Stop Conditions

- Stop if truncation, adapter failures, or ambiguous extraction rates are too
  high.
- Stop if metrics are miscalibrated or unavailable beyond acceptable scope.
- Stop if the team cannot explain whether a dev-run result came from old or new
  prompts/settings.

## Sprint 10: Full 4,800-Call Run

### Objective

Run the full benchmark only after dataset, smoke, and dev gates pass.

The run is valid only against frozen inputs. If data, prompts, metrics, adapter
logic, or model configuration changes mid-run, invalidate the run and start a new
`run_id`.

### Target Files

- `configs/full.vllm.json`
- `runbooks/full_run.md`
- full result artifacts under `results/full_*`, generated and ignored unless the
  user asks to preserve them
- final aggregate report under `reports/full_*`

### Implementation Tasks

- Require:
  - `data/personas.full.jsonl` with exactly 200 validated rows
  - promotion manifest
  - review manifest
  - successful 20-persona smoke
  - successful 50-persona dev run
  - explicit full-run approval
- Pin or record:
  - code commit
  - dirty-worktree marker, if any
  - dataset hash
  - prompt/template hash
  - model IDs and revisions
  - tokenizer hash or tokenizer package/version
  - chat-template hash
  - vLLM/runtime version
- Run `200 * 6 * 2 * 2 = 4,800` generation calls.
- Aggregate by persona group.
- Record all runtime metadata.
- Preserve enough information for rerun and audit.
- Do not patch metrics, schemas, prompts, or datasets silently after seeing full
  results.

### Tests

- Full plan count equals `4,800`.
- Full run command is blocked unless all approval artifacts exist.
- Aggregation rejects row-level inference.
- Report contains metric availability and known limitations.

### Verification Commands

```bash
python3 -m unittest discover -s tests
python3 -m pytest
python3 persona_eval.py plan --persona-count 200 --variants-per-persona 6 --model-count 2 --seed-count 2
```

Manual full run:

```bash
python3 persona_eval.py run --persona-path data/personas.full.jsonl --out results/full_200 --adapter vllm --base-url <local-vllm-url> --model-base <base-model> --model-tuned <tuned-model> --seeds 1,2 --run-id full_200
python3 aggregate.py --manifest results/full_200/manifest.json --results results/full_200/results.jsonl --out reports/full_200
```

### Acceptance

- Full run completes with validated manifest/results.
- Aggregate report uses persona-level inference.
- Known limitations and metric availability are explicit.
- Artifact manifest includes hashes for dataset, manifest, results, aggregate
  report, code, and configs.

### Stop Conditions

- Stop if full-run approval artifacts are missing.
- Stop if the run cannot preserve raw request/response metadata.
- Stop if aggregation discovers schema or matching defects.
- Stop if a mid-run change makes results non-comparable.

## Sprint 11: Final Report And Reproducibility Package

### Objective

Package the dataset, methodology, run artifacts, aggregate report, and
limitations into a reproducible research deliverable.

The report should distinguish facts from inference. Claims must come from the
final frozen run or be explicitly labeled as pilot/dev-run evidence.

### Target Files

- `FINAL_REPORT.md`
- `METHODOLOGY.md`
- `REPRODUCIBILITY.md`
- `LIMITATIONS.md`
- `DATASET_CARD.md`
- `MODEL_RUN_CARD.md`
- final charts/tables, generated from aggregate report

### Implementation Tasks

- Document:
  - research question
  - dataset construction
  - validation and review workflow
  - model/runtime configuration
  - prompt template
  - metrics
  - statistical methods
  - results
  - limitations
  - reproducibility instructions
- Add release notes that explain what changed across Sprints 0-11 and which
  results are final versus diagnostic.
- Include exact hashes:
  - dataset hash
  - manifest hash
  - results hash
  - aggregate report hash
  - code commit
- Include metric availability:
  - Token-KL ok/diagnostic/not_applicable counts
  - PA real/mock/not_applicable status
  - BC-F1 field summaries
- Include failure analysis and flagged examples.
- Make claims only supported by the data.
- Keep PA caveats explicit if PA is still backed by mock embedding/NLI
  interfaces.
- Keep Token-KL caveats explicit unless fixed aligned continuation scoring was
  implemented and verified.
- Freeze final artifacts under versioned names so future reruns do not overwrite
  the publication package.

### Tests

- Report references existing artifact paths or hashes.
- Tables/charts are generated from aggregate JSON, not manually invented.
- Reproducibility commands are executable or clearly marked manual.
- No unsupported claims about full dataset readiness, PA, or Token-KL.
- Report text is consistent with the metric availability statuses in aggregate
  output.

### Verification Commands

```bash
python3 -m unittest discover -s tests
python3 -m pytest
python3 aggregate.py --manifest results/full_200/manifest.json --results results/full_200/results.jsonl --out reports/full_200_recomputed
```

### Acceptance

- Final report is internally consistent with artifacts.
- Reproducibility instructions are complete.
- Limitations are explicit.
- The user can hand the package to another reviewer without hidden context.
- Release notes and artifact hashes are sufficient to reconstruct what changed
  since the first guide/plan file.

### Stop Conditions

- Stop if final claims exceed the evidence.
- Stop if artifact hashes cannot be reproduced.
- Stop if model/runtime metadata is incomplete.
