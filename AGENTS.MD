# Persona Drift Evaluation Implementation Guide

This repository is currently a planning workspace for a persona-drift evaluation harness. Treat `guide.MD` as a design draft, not as an implementation authority. Future implementation work must follow the verified corrections and gates in this file.

Last verification pass: 2026-05-25.

## Bottom Line

The design is viable enough to plan development: a three-metric persona-drift harness over persona-conditioned prompts with controlled variants.

Default implementation target:

- `Persona Adherence`: semantic persona alignment plus fact contradiction checks.
- `Token-KL`: teacher-forced next-token distribution shift between comparable model checkpoints.
- `Behavioral-Consistency F1`: behavior-label consistency across semantically equivalent variants.

Default serving target:

- Use `vLLM` for local full benchmark runs.
- Use hosted HTTP/OpenAI-compatible endpoints only for black-box comparison.
- Use TGI only when an existing deployment requires production-parity checks; do not make it the default new stack.

## Source Authority

Use this priority order when facts conflict:

1. Current repository files and generated artifacts.
2. Primary sources: ACL Anthology, arXiv, official project docs, official dataset cards, official GitHub repos.
3. `guide.MD`, only after reconciling it against this file.
4. Secondary summaries, only for discovery, never for final claims.

Every citation, dataset license, model ID, API parameter, and benchmark claim used in implementation docs must be verified from a primary source before it is committed.

## Commit Cadence

Make small, intentional commits after every few major file changes. A good
checkpoint is one coherent implementation slice, such as schemas plus their
tests, CLI behavior plus tests, metric plumbing plus tests, or docs plus
verification notes.

Commit only after the relevant local checks pass. At minimum, run the affected
unit tests; for sprint-level work, run the sprint's full verification command
set before committing. Do not mix unrelated changes in the same commit, and do
not stage generated autoresearch logs or cache files.

Use concise commit messages that name the completed slice, for example:

- `Add Sprint 2 behavior metric plumbing`
- `Tighten result-row metric schemas`
- `Add Sprint 3 mock run command`

## Verified Corrections To `guide.MD`

Do not copy these guide claims forward without correction.

### Run Counts

The early summary in `guide.MD` says `200 / 1,000 / 4,000` generation calls. That is wrong for the stated design.

Use these counts:

- Smoke: `20 personas * 6 variants * 2 models * 1 seed = 240 calls`.
- Dev: `50 personas * 6 variants * 2 models * 2 seeds = 1,200 calls`.
- Full: `200 personas * 6 variants * 2 models * 2 seeds = 4,800 calls`.

The token budget arithmetic is correct:

- `4,800 calls * 140 generated tokens = 672,000 generated tokens`.
- At `100 tokens/s`, pure decoding is about `1.87 hours`.
- At `250 tokens/s`, pure decoding is about `44.8 minutes`.

Implementation must include tests or assertions for these counts.

### Variant Representation

Use one representation consistently:

- `variants` must contain exactly six objects.
- One object must have `type: "canonical"`.
- Five objects must be controlled perturbations:
  - `paraphrase`
  - `negation_preserving`
  - `distractor`
  - `instruction_prefix`
  - `temperature_robust`

Do not store the canonical prompt separately for execution while also treating it as a variant. `seed_prompt` can hold metadata and the canonical text for human readability, but the executable set is `variants`.

### Dataset Licensing

Do not use `MIT-or-upstream-compatible` for Persona-Chat-derived records.

Verified source posture:

- ParlAI software license is MIT, but Persona-Chat task data documentation points to CC BY 4.0 terms for the dataset artifact.
- Synthetic-Persona-Chat public dataset card reports CC BY 4.0.

Every dataset row must carry source metadata:

- `source.dataset`
- `source.source_url`
- `source.license`
- `source.license_url`
- `source.split`
- `source.source_persona_id`
- `source.retrieved_at`
- `source.revision_or_hash`, when available
- `source.modification_notes`
- `source.redistribution_notes`

If the exact source artifact cannot be traced, exclude the row from any redistributable dataset.

### Paper Metadata

Fix paper titles, venues, and dates before adding a bibliography.

Known correction:

- The guide's `PAML: Dialogue Personalization by Fine-Tuning (2019)` entry is misnamed. The verified ACL 2019 paper is `Personalizing Dialogue Agents via Meta-Learning`.

Do not rely on memory for paper metadata. Recheck primary pages before writing README citations.

### API And Model IDs

Do not hard-code placeholder API model names in committed examples unless they are explicitly marked as placeholders.

Current verification found `gpt-5.5` in official OpenAI model docs, but model availability is time-sensitive. Future code must make API model IDs configurable and document that they must be checked against provider docs at run time.

The OpenAI Responses structured-output example must include:

- `import os` if it uses `os.environ`.
- `text.format.type = "json_schema"`.
- `strict: true` unless there is a documented reason to allow loose schema output.
- A configurable `model` value.

Use canonical Hugging Face model IDs unless local aliases are explicitly configured. For example:

- `meta-llama/Meta-Llama-3-8B`
- `meta-llama/Meta-Llama-3-8B-Instruct`

## Architecture Decisions

### Serving

Default: local `vLLM`.

Rationale:

- It provides OpenAI-compatible serving.
- It supports high-throughput local evaluation.
- It gives the best path for tokenizer-aware scoring and logprob work.

TGI:

- Valid for existing deployments and production-parity checks.
- Do not choose it as the default new implementation path because the upstream repo is in maintenance mode.

Hosted HTTP/OpenAI-compatible endpoints:

- Valid for black-box comparisons and LLM tag extraction.
- Not sufficient for canonical Token-KL unless they expose an aligned scoring path with enough logprob depth.

### Data Layout

Expected artifacts:

- `data/personas.sample.jsonl`: 10 hand-checkable rows.
- `data/personas.full.jsonl`: 200 validated rows.
- `schemas/persona_item.schema.json`: JSON Schema for dataset rows.
- `schemas/behavior_tags.schema.json`: JSON Schema for extracted behavior tags.
- `results/run_example.jsonl`: small sample output.
- `aggregate.py` or `notebooks/aggregate_metrics.ipynb`: aggregation and chart generation.
- `README.md`: exact setup, commands, versions, expected outputs, and known limitations.

Do not create large generated datasets without a deterministic generation and validation script.

## Metric Contracts

### Persona Adherence

Persona Adherence is a calibrated, relative metric. Do not present it as an absolute universal persona score.

Required behavior:

- Serialize persona facts, traits, values, and forbidden behaviors deterministically.
- Compute semantic similarity with a pinned embedding model.
- Check fact contradictions with a pinned NLI/judge method.
- Calibrate thresholds on positive persona-response pairs and persona-swapped negatives.
- Report `PA-mean`, `PA-pass@threshold`, and `fact_contradiction_rate`.

Do not choose a universal cosine threshold by intuition.

### Token-KL

Token-KL is valid only when all of these are true:

- Base and tuned models share tokenizer and vocabulary.
- Chat template and prompt rendering are fixed and hashed.
- Both models are scored on the same fixed continuation.
- The scoring path returns comparable next-token probabilities or logprobs.

Do not compute canonical Token-KL from two models' free-running generations. Independent generations produce different prefixes, so stepwise KL is not interpretable.

Accepted scoring paths:

- Local model forward pass over a fixed target response.
- vLLM prompt/scoring path over a fixed continuation.
- A one-token scoring loop over the same teacher-forced prefix.

Hosted endpoint top-logprobs are allowed only as capped diagnostics unless they can score the same fixed continuation with sufficient depth.

Recommended default:

- `k = 50` for local scoring if available.
- If an endpoint caps top-logprobs below `k`, record the cap and do not compare it to local full-depth KL as equivalent.

Fallbacks when Token-KL is invalid:

- MAUVE or sequence-level distribution comparison.
- Token-frequency JS divergence.
- Behavior-level metrics only, with Token-KL marked `not_applicable`.

### Behavioral-Consistency F1

BC-F1 must score behavior, not wording.

Use field-level scoring:

- `stance`: exact match.
- `primary_action`: exact match.
- `secondary_modifiers`: multilabel F1.
- Optional combined score can be a weighted aggregate, but the field scores must be reported separately.

Do not let a large modifier set hide a wrong stance or primary action.

For self-consistency without gold labels:

- Use leave-one-out majority voting, or
- Use pairwise agreement across variants.

Do not score a variant against a majority vote that includes itself.

Extractor contract:

- Run deterministic rules first.
- Use an LLM tagger only for ambiguous cases.
- LLM tagger output must be strict JSON Schema.
- Flag logically invalid pairs, such as `support` plus `refuse`.
- Store whether parser fallback, JSON repair, or human review was used.

## Statistical Reporting

The unit of inference is the persona group, not the individual generation row.

Required reporting:

- Mean and 95% confidence interval.
- Absolute delta versus baseline.
- Effect size.
- P-value when using hypothesis tests.
- Number and percentage of flagged personas.
- Multiple-comparison handling for secondary cuts across metrics, variants, and domains.

Recommended tests:

- Paired bootstrap confidence intervals over persona-level deltas for PA, BC-F1, and Token-KL.
- McNemar's test for paired pass/fail outcomes.
- Exact McNemar when discordant counts are small.
- Paired permutation test or bootstrap for self-consistency deltas.

Do not treat `4,800` full-run generations as independent observations.

## Reproducibility Requirements

Every run row or run manifest must include:

- `run_id`
- `timestamp_utc`
- `code_commit` or explicit dirty-worktree marker
- `persona_jsonl_hash`
- `prompt_template_version`
- `prompt_template_hash`
- `chat_template_hash`
- `tokenizer_name`
- `tokenizer_hash`
- `model_name`
- `model_revision_or_hash`
- `serving_stack`
- `serving_stack_version`
- `GPU/CUDA/driver` when local
- `decoding_params`
- `seed`
- `stop_sequences`
- `stop_reason`
- `truncation_flag`
- `raw_request`
- `raw_response`
- `usage`
- `latency_s`
- `extractor_version`
- `embedding_model_revision`
- `nli_or_judge_model_revision`

Cache keys must include at least:

- rendered prompt hash
- system prompt hash
- model revision/hash
- tokenizer hash
- chat template hash
- decoding params
- seed
- evaluator version

Never reuse cached outputs across prompt-template, tokenizer, or chat-template changes.

## Validation Pipeline

A persona row is valid only after these checks pass:

1. Schema validation.
2. Source/provenance validation.
3. License metadata validation.
4. PII and real-person impersonation filter.
5. Harmful or restricted role filter.
6. Variant count and variant-type validation.
7. Semantic equivalence validation for controlled variants.
8. NLI contradiction/equivalence check where applicable.
9. Gold-label preview check.
10. Deduplication by exact text and embedding similarity.
11. Human review for all low-confidence rows and at least 10% of all rows.

For public benchmark data, exclude:

- real-person personas
- medical decision roles
- legal guarantee roles
- self-harm roles
- extremist roles
- fraud or deception roles
- roles that require impersonating credentials

Restricted internal packs can include specialized domains only if clearly labeled and not redistributed.

## Test Plan For First Implementation

Start with small, hand-checkable fixtures:

- Arithmetic fixture: verifies `240`, `1,200`, and `4,800` call counts.
- Schema fixture: one valid persona row and one invalid row per required field group.
- Variant fixture: verifies exactly six executable variants and required variant types.
- PA fixture: tests persona serialization and threshold calibration plumbing.
- Token-KL fixture: tests identical distributions, shifted distributions, tokenizer mismatch rejection, and free-running-output rejection.
- BC-F1 fixture: tests stance exact match, primary action exact match, modifier F1, and leave-one-out self-consistency.
- Logging fixture: verifies run manifest completeness and cache-key fields.

Do not implement the full 200-row dataset before these fixtures pass.

## Development Readiness Gate

Planning can start when the plan preserves these constraints:

- It uses the corrected call counts.
- It uses exactly six executable variants per persona.
- It treats dataset licensing and provenance as first-class row metadata.
- It makes `vLLM` the default local serving path.
- It makes model IDs configurable.
- It implements Token-KL only through aligned scoring.
- It reports BC-F1 field-level scores.
- It aggregates statistics by persona group.
- It logs enough metadata to reproduce or invalidate every result.
- It includes a small fixture-first test plan before any large generation run.

If a proposed implementation violates one of these constraints, fix the plan before writing code.

## Verified Source Pointers

Use these as starting points, but recheck before committing implementation docs:

- Persona-Chat paper: https://aclanthology.org/P18-1205/
- Persona-Chat task/license docs: https://parl.ai/projects/personachat/ and https://raw.githubusercontent.com/facebookresearch/ParlAI/main/parlai/tasks/personachat/LICENSE_DOCUMENTATION
- Dialogue NLI: https://aclanthology.org/P19-1363/
- CheckList: https://aclanthology.org/2020.acl-main.442/
- Contrast Sets: https://aclanthology.org/2020.findings-emnlp.117/
- InCharacter: https://aclanthology.org/2024.acl-long.102/
- RoleLLM: https://aclanthology.org/2024.findings-acl.878/
- SocialBench: https://aclanthology.org/2024.findings-acl.125/
- TRAIT: https://aclanthology.org/2025.findings-naacl.469/
- PersonaGym: https://aclanthology.org/2025.findings-emnlp.368/
- Can LLM Agents Maintain a Persona in Discourse?: https://aclanthology.org/2025.emnlp-main.1487/
- Identity Drift: https://arxiv.org/abs/2412.00804
- Persona Vectors: https://www.anthropic.com/research/persona-vectors
- Assistant Axis: https://www.anthropic.com/research/assistant-axis
- IFEval: https://arxiv.org/abs/2311.07911
- HELM: https://arxiv.org/abs/2211.09110
- Lessons from the Trenches / lm-eval: https://arxiv.org/abs/2405.14782
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- vLLM PagedAttention paper: https://arxiv.org/abs/2309.06180
- TGI API docs: https://huggingface.co/docs/text-generation-inference/main/en/reference/api_reference
- TGI architecture docs: https://huggingface.co/docs/text-generation-inference/main/en/architecture
- TGI GitHub maintenance status: https://github.com/huggingface/text-generation-inference
- OpenAI Responses API: https://developers.openai.com/api/reference/resources/responses/methods/create
- OpenAI Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- OpenAI models: https://developers.openai.com/api/docs/models
- Synthetic-Persona-Chat dataset: https://huggingface.co/datasets/google/Synthetic-Persona-Chat
