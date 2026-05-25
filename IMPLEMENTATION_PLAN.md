# Persona Drift Evaluation Implementation Plan

Status: reviewed and corrected
Last updated: 2026-05-25

## Bottom Line

Build the evaluator in fixture-first slices. Do not generate the full 200-persona dataset or run real model comparisons until schema validation, prompt rendering, run counting, logging, result schemas, and metric contracts are enforced by tests.

Default sequence:

1. Sprint 0: dependency setup, schemas, fixtures, validation, and run-count math.
2. Sprint 1: deterministic prompt rendering, manifest/result schemas, cache keys, and mock adapters.
3. Sprint 2: metric plumbing for Persona Adherence, BC-F1, and Token-KL guardrails.
4. Sprint 3: mock smoke, tiny vLLM smoke, then 20-persona smoke; no full run.
5. Sprint 4: aggregation, statistics, charts, README, and readiness review for the 200-persona dataset.

The first useful product is a reproducible relative probe: base vs instruct, pre- vs post-finetune, or checkpoint A vs checkpoint B. This is not a leaderboard project.

## Ground Rules

`AGENTS.MD` is the controlling implementation contract. `guide.MD` is a corrected design reference.

Hard gates:

- Exactly six executable variants per persona: one `canonical` plus five perturbations.
- Correct generation-call counts: smoke `240`, dev `1,200`, full `4,800`.
- Dataset rows must use the nested `source.*` metadata contract from `AGENTS.MD`.
- vLLM is the default local serving path.
- TGI is only for existing deployment parity.
- Hosted endpoints are black-box comparison or tagger paths unless they provide aligned scoring.
- Token-KL is unavailable unless both models are scored on the same fixed continuation with compatible tokenizer, vocabulary, and chat template.
- BC-F1 reports field-level scores: stance exact match, primary_action exact match, secondary_modifiers F1.
- Statistical aggregation is by persona group, not generation row.
- No large generated dataset before fixture tests pass.
- Unvalidated candidate pools must stay outside `data/` and cannot be treated as benchmark rows.

## Target Repository Shape

Initial files:

- `pyproject.toml`: Python and dev dependency declaration.
- `program.md`: autoresearch operating instructions for agents.
- `auto_research.py`: tagged run setup, command capture, metric parsing, and TSV logging.
- `persona_eval.py`: single-file MVP harness until complexity justifies a package split.
- `schemas/persona_item.schema.json`: schema for persona JSONL rows.
- `schemas/behavior_tags.schema.json`: schema for extractor output.
- `schemas/run_manifest.schema.json`: schema for run-level metadata.
- `schemas/result_row.schema.json`: schema for per-row outputs.
- `data/personas.sample.jsonl`: 10 hand-checkable rows.
- `tests/test_counts.py`: sample/smoke/dev/full call-count assertions.
- `tests/test_schema.py`: valid and invalid persona-row fixtures.
- `tests/test_variants.py`: exactly-six-variant and type coverage tests.
- `tests/test_prompt_rendering.py`: deterministic prompt/system rendering and hash checks.
- `tests/test_behavior_f1.py`: field-level BC-F1 tests.
- `tests/test_token_kl.py`: aligned-scoring guards and toy KL tests.
- `tests/test_manifest.py`: run-manifest, result-row, and cache-key tests.
- `results/.gitkeep`: placeholder for local generated results.
- `README.md`: setup, commands, fixture expectations, limitations.

Later files:

- `data/personas.full.jsonl`: 200 validated rows, created only after validators and review workflow exist.
- `aggregate.py`: aggregation and chart generation.
- `charts/*.png`: generated after fixture, mock, or real runs.

## Autoresearch Control Plane

This repository can use an autoresearch loop, but only as a control plane around
the benchmark contract. It must not mutate metric definitions, source/provenance
requirements, variant counts, or statistical denominators in order to improve a
reported result.

Current control-plane files:

- `program.md`: the agent-facing operating program.
- `auto_research.py`: local controller for run initialization, command capture,
  log files, metric parsing, and `results.tsv` output.

Default controller commands:

```bash
python3 auto_research.py init --tag <tag>
python3 auto_research.py check
python3 auto_research.py run-once --tag <tag> --description "short experiment" -- python3 -m unittest discover -s tests
python3 auto_research.py summarize --tag <tag>
```

Autoresearch modes:

- Bootstrap mode: improve implementation coverage against the fixed contract.
- Benchmark mode: compare model/prompt/config experiments after fixture, mock,
  and smoke gates pass.

Bootstrap keep rule:

- Keep changes that make more hard gates executable, reduce failing tests, or
  simplify implementation without weakening the contract.

Benchmark keep rule:

- Keep changes only when the named metric improves and all hard gates still pass.

Discard immediately if an experiment weakens Token-KL validity, BC-F1 field
scoring, source metadata, call-count arithmetic, persona-level aggregation, or
the full-dataset gate.

## Canonical Schemas

### Persona Row

The persona row schema must use this behavior-label shape:

- `expected_behavior.stance`
- `expected_behavior.primary_action`
- `expected_behavior.secondary_modifiers`
- `annotation.gold_labels.stance`
- `annotation.gold_labels.primary_action`
- `annotation.gold_labels.secondary_modifiers`

Do not use `annotation.gold_labels.action_set` as the canonical implementation schema. It can appear only in migration notes if older guide examples are converted.

The `source` object must include:

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

`source.revision_or_hash` may be `null` only when the source artifact has no stable revision or hash. In that case, `source.modification_notes` must explain the fallback provenance evidence.

### Run Manifest

The run manifest must include:

- `run_id`
- `timestamp_utc`
- `code_commit`
- `dirty_worktree`
- `persona_jsonl_hash`
- `prompt_template_version`
- `prompt_template_hash`
- `chat_template_hash`
- `tokenizer_name`
- `tokenizer_hash`
- `model_base`
- `model_tuned`
- `model_base_revision_or_hash`
- `model_tuned_revision_or_hash`
- `adapter`
- `provider_or_endpoint`
- `serving_stack`
- `serving_stack_version`
- `scoring_capability`
- `gpu_cuda_driver`
- `decoding_params`
- `seeds`
- `stop_sequences`
- `extractor_version`
- `embedding_model_revision`
- `nli_or_judge_model_revision`

Mock runs may set unavailable local runtime fields to `not_available`, but the keys must still be present.

### Result Row

Each result row must include:

- `run_id`
- `persona_id`
- `variant_id`
- `variant_type`
- `seed`
- `prompt_text`
- `system_prompt`
- `prompt_hash`
- `system_prompt_hash`
- `model_pair`
- `base.response_text`
- `tuned.response_text`
- `base.stop_reason`
- `tuned.stop_reason`
- `base.truncation_flag`
- `tuned.truncation_flag`
- `base.usage`
- `tuned.usage`
- `base.latency_s`
- `tuned.latency_s`
- `base.raw_response`
- `tuned.raw_response`
- `behavior_tags`
- `metrics`
- `flags`

## Token-KL Status Contract

Every Token-KL result must include:

- `status`: one of `ok`, `not_applicable`, `diagnostic_only`, `error`.
- `reason_code`: machine-readable reason when not `ok`.
- `scoring_path`: `local_forward`, `vllm_prompt_logprobs`, `one_token_loop`, `hosted_top_logprobs`, or `none`.
- `fixed_continuation_id`
- `fixed_continuation_hash`
- `tokenizer_hash_match`
- `vocabulary_match`
- `chat_template_hash_match`
- `k`
- `endpoint_cap`
- `diagnostic_only`

Hosted or endpoint-capped top-logprobs are diagnostic unless the endpoint can score the same fixed continuation with enough depth for the configured `k`.

## Sprint 0: Dependencies, Schemas, Fixtures, And Counters

Goal: prove the data and run shape before model code exists.

Work:

- Create `pyproject.toml`.
- Require Python `>=3.11`.
- Add minimal dev dependencies: `pytest` and `jsonschema`.
- Create `schemas/persona_item.schema.json`.
- Create `schemas/behavior_tags.schema.json`.
- Create `schemas/run_manifest.schema.json`.
- Create `schemas/result_row.schema.json`.
- Create `data/personas.sample.jsonl` with 10 rows.
- Implement JSONL loader and schema validation in `persona_eval.py`.
- Implement `expected_call_count(personas, variants, model_count, seed_count)`.
- Add tests for:
  - sample: `10 * 6 * 2 * 1 = 120`
  - smoke: `20 * 6 * 2 * 1 = 240`
  - dev: `50 * 6 * 2 * 2 = 1,200`
  - full: `200 * 6 * 2 * 2 = 4,800`
- Add tests that reject rows without every required `source.*` field.
- Add tests that reject legacy behavior-label shape where canonical fields are missing.
- Add tests that reject rows with fewer or more than six executable variants.

Acceptance:

- `python3 -m pytest` passes locally.
- Every sample row has one `canonical` variant and five required perturbation types.
- Every sample row includes the exact nested `source.*` fields listed above.
- Every sample row uses canonical behavior labels: `stance`, `primary_action`, `secondary_modifiers`.
- No model endpoint, GPU, network access, or API key is required.

Stop conditions:

- If a source artifact cannot be licensed or attributed, do not include it in sample data.
- If the schema cannot express the known guide requirements cleanly, revise the schema before proceeding.

## Sprint 1: Prompt Rendering, Manifests, Results, Cache Keys, And Adapters

Goal: make execution reproducible without touching real models.

Work:

- Implement deterministic persona serialization.
- Implement deterministic system prompt rendering.
- Implement rendered prompt hash and system prompt hash.
- Define `BaseAdapter.generate()` and `BaseAdapter.score_continuation()` interfaces.
- `generate()` and `score_continuation()` must remain separate.
- Implement `MockAdapter` for tests.
- Implement run manifest creation and validation.
- Implement result-row creation and validation.
- Implement cache-key creation.
- Record dirty-worktree state when no commit hash is available.
- Include adapter identity, provider/endpoint, serving stack/version, and scoring capability in cache metadata.

Cache keys must include:

- rendered prompt hash
- system prompt hash
- model revision/hash
- tokenizer hash
- chat-template hash
- decoding params
- seed
- evaluator version
- adapter
- provider_or_endpoint
- serving_stack
- serving_stack_version
- scoring_capability

Acceptance:

- Prompt rendering is stable across repeated runs.
- Changing persona facts, prompt template, chat template, model ID, model revision, tokenizer hash, seed, adapter, endpoint, serving stack, or decoding params changes the cache key or cache metadata.
- Run manifest schema validates all required keys.
- Result-row schema validates mock outputs.
- Tests do not require GPU, network, or API keys.

Stop conditions:

- If adapter abstractions blur generation and scoring, split them before Sprint 2. Token-KL depends on that separation.

## Sprint 2: Metrics And Guardrails

Goal: implement metric plumbing with toy fixtures before real model outputs.

Work:

- Implement BC-F1 field scoring:
  - stance exact match
  - primary_action exact match
  - secondary_modifiers multilabel F1
  - optional weighted combined score
- Implement leave-one-out self-consistency and pairwise agreement.
- Implement tests for leave-one-out self-exclusion.
- Implement tests for pairwise agreement.
- Implement invalid stance/action pair detection.
- Implement strict schema-failure tests for behavior tags.
- Implement rule-first behavior extractor skeleton.
- Implement strict JSON validation for LLM tagger output, but keep the tagger disabled by default in tests.
- Implement Persona Adherence serialization and score interface.
- Add positive persona-response and persona-swapped negative calibration fixtures.
- Add a placeholder embedding backend interface and mock embeddings.
- Add a PA gate: no real PA score beyond mock/plumbing until embedding backend, NLI/judge backend, and calibration fixture are pinned.
- Implement Token-KL on toy top-k distributions.
- Add explicit Token-KL rejection for:
  - tokenizer mismatch
  - vocabulary mismatch
  - chat-template mismatch
  - missing fixed continuation
  - free-running output comparison
- Add endpoint-capped Token-KL diagnostic tests.
- Implement the Token-KL status contract.

Acceptance:

- BC-F1 tests show that modifier overlap cannot hide wrong stance or primary_action.
- BC-F1 tests prove leave-one-out majority does not include the scored variant.
- Token-KL tests pass for identical toy distributions, shifted distributions, invalid comparison rejections, and endpoint-capped diagnostic behavior.
- Persona Adherence tests prove deterministic serialization and calibrated-threshold plumbing, without claiming a universal threshold.
- All unavailable metrics return structured status metadata.

Stop conditions:

- If Token-KL cannot prove aligned scoring, return `not_applicable`; do not silently fall back to comparing free-running outputs.
- If PA calibration cannot be tested with positive and persona-swapped negative fixtures, do not report PA beyond mock/plumbing.

## Sprint 3: Mock Smoke, Tiny vLLM Smoke, And 20-Persona Smoke

Goal: prove the run path in stages without jumping straight to costly evaluation.

Work:

- Implement `VLLMOpenAIAdapter.generate()`.
- Implement capability-checked `score_continuation()` for aligned scoring, or return `not_applicable` with Token-KL status metadata.
- Add CLI subcommands:
  - `validate`
  - `plan`
  - `run`
- Add CLI flags:
  - `--persona-path`
  - `--out`
  - `--adapter`
  - `--base-url`
  - `--api-key-env`
  - `--serving-stack`
  - `--score-mode`
  - `--disable-token-kl`
  - `--model-base`
  - `--model-tuned`
  - `--model-base-alias`
  - `--model-tuned-alias`
  - `--seeds`
  - `--temperature`
  - `--max-tokens`
  - `--prompt-template-version`
  - `--run-id`
  - `--dry-run`
- Define `--seeds` parsing as comma-separated integers.
- Make `--model-base` and `--model-tuned` explicit pair-mode arguments; do not use ambiguous `--models base,tuned`.
- Add dry-run mode that prints planned call counts and manifest without calling a model.
- Run the 10-row sample against `MockAdapter`.
- Run a 1-2 persona tiny vLLM smoke only if local vLLM is available.
- Run the 20-persona smoke only after mock and tiny checks pass.

Acceptance:

- Sample dry-run reports `120` calls for `data/personas.sample.jsonl`.
- Phase arithmetic dry-run reports `240` calls for smoke.
- Mock run writes valid JSONL rows and a run manifest.
- Tiny vLLM smoke, if performed, records raw request/response, stop reason, truncation flag, usage, latency, model revision, tokenizer hash, adapter, provider/endpoint, serving stack, and scoring capability.
- Token-KL is emitted only when `score_continuation()` is implemented and capability-checked. Otherwise it returns structured `not_applicable`.

Stop conditions:

- Do not run the 20-persona smoke until dry-run math, mock run, manifest tests, and result-row tests pass.
- Do not report Token-KL for real runs unless `score_continuation()` is implemented and verified.

## Sprint 4: Aggregation, Reports, And Readiness For Full Dataset

Goal: make results interpretable and decide whether full data generation is justified.

Work:

- Implement `aggregate.py`.
- Construct persona-level deltas by matching base/tuned outputs within each `persona_id`, aggregating matched variants and seeds inside that persona, then bootstrapping over persona-level deltas.
- Report:
  - PA mean and 95% CI
  - BC-F1 field scores and 95% CI
  - Token-KL when valid
  - pass/fail deltas
  - effect size
  - p-value where hypothesis tests are used
  - multiple-comparison handling for secondary cuts
  - flagged persona counts and percentages
  - variant-type breakdowns
- Add paired bootstrap over persona-level deltas.
- Add McNemar pass/fail support, with exact mode for small discordant counts.
- Add paired permutation test or bootstrap for self-consistency deltas.
- Generate example charts from fixture or mock results.
- Write README with exact commands and known limitations.

Acceptance:

- Aggregation tests prove matched variants/seeds are aggregated inside each persona before persona-level bootstrap.
- Example report does not treat generation rows as independent observations.
- README explains when Token-KL is unavailable or diagnostic-only.
- Report schema includes effect size, p-value, multiple-comparison notes, and metric availability status.
- Full dataset work is explicitly gated on schema, validation, provenance, and human-review evidence.

Stop conditions:

- If aggregation cannot explain metric unavailability, fix result schema before generating more outputs.

## Full Dataset Gate

The 200-persona dataset can start only after:

- Sample schema tests pass.
- Variant validation tests pass.
- Source/license checks pass.
- PII and real-person filters are implemented and pass.
- Restricted-role filters are implemented and pass for:
  - real-person personas
  - medical decision roles
  - legal guarantee roles
  - self-harm roles
  - extremist roles
  - fraud or deception roles
  - credential impersonation roles
- Semantic equivalence validation is implemented and passes, or a documented manual equivalent is recorded with review evidence.
- NLI contradiction/equivalence checks are implemented where applicable, or documented manual review evidence is recorded.
- Gold-label preview checks are implemented and pass.
- Deduplication is implemented for:
  - normalized exact text
  - near-duplicate prompt/persona text
  - embedding similarity cluster review
- Human-review workflow is implemented with a companion review manifest or row metadata containing:
  - reviewer
  - review_status
  - reviewed_at
  - review_reason
  - low_confidence_flags
- Human review covers all low-confidence rows and at least 10% of all rows.
- Unvalidated candidate pools remain outside `data/`.

Full-run gate:

- Full `4,800` generation-call run can start only after a successful 20-persona smoke run and review of flagged outputs.

## Open Decisions

These should be decided before Sprint 2 ends:

- Whether `persona_eval.py` remains a single file through Sprint 3 or splits into a package earlier.
- Which embedding backend to pin for Persona Adherence.
- Which NLI or judge backend to use for fact contradiction checks.
- Whether OpenAI Responses is used only for optional tagger fallback or also for black-box model comparison.
- Exact chart set for the first README report.

Default choices unless evidence says otherwise:

- Keep `persona_eval.py` single-file through Sprint 2.
- Use local mock embeddings for tests and pin the real embedding backend later.
- Disable LLM tagger by default in tests.
- Use vLLM for full local evaluation.
- Treat hosted model outputs as comparison-only unless aligned scoring is available.

## First Development Command Set

Expected commands after Sprint 0 is implemented:

```bash
python3 -m pytest
python3 persona_eval.py validate --persona-path data/personas.sample.jsonl
python3 persona_eval.py plan --persona-path data/personas.sample.jsonl --model-base base --model-tuned tuned --seeds 1
python3 persona_eval.py plan --persona-count 20 --variants-per-persona 6 --model-count 2 --seed-count 1
```

Expected sample count:

```text
planned_generation_calls=120
```

Expected smoke phase arithmetic:

```text
planned_generation_calls=240
```

## Review Checklist

Before implementing, reviewers should answer:

- Does the plan preserve all hard gates from `AGENTS.MD`?
- Does any step require network, GPU, or API keys before fixtures pass?
- Does Token-KL stay unavailable unless aligned scoring is proven?
- Does BC-F1 report stance/action/modifier fields separately?
- Is the full dataset clearly gated behind provenance and validation?
- Are the first tests small enough to debug by inspection?
- Are sample-count and smoke-count commands arithmetically distinct?
