# Sprint 1 Contracts

Status: restored from `AGENTS.MD`, `IMPLEMENTATION_PLAN.md`, and Sprint 1 review findings.
Last updated: 2026-05-25.

## Bottom Line

Sprint 1 is a local reproducibility slice. It may render prompts, build manifests,
construct mock result rows, and define adapter boundaries. It must not call real
models, use network access, require API keys, require a GPU, generate the full
200-persona dataset, aggregate metrics, draw charts, or compute real PA, BC-F1,
or canonical Token-KL.

## Prompt Rendering

- `prompt_template_version` is `v1`.
- Render these sections in stable order:
  - persona facts
  - persona traits
  - persona values
  - forbidden behaviors
  - user task/question
- Empty optional sections must render deterministically as explicit empty
  sections, not disappear.
- `prompt_hash` is SHA-256 over the exact rendered prompt string.
- `system_prompt_hash` is SHA-256 over the exact system prompt string.
- `prompt_template_hash` is SHA-256 over the exact template source for the
  declared version.

## Adapter Boundary

Adapters must keep generation and aligned continuation scoring separate:

- `generate(request) -> GenerationResult`
- `score_continuation(request) -> ScoreContinuationResult`

`GenerationRequest` and `ScoreContinuationRequest` must carry enough audit
coordinates to serialize and replay the attempted call:

- `run_id`
- `persona_id`
- `variant_id`
- `variant_type`
- `model_alias`
- `model_id`
- `model_revision_or_hash`
- `tokenizer_name`
- `tokenizer_hash`
- `chat_template_hash`
- `prompt_template_version`
- `prompt_template_hash`
- `prompt_hash`
- `system_prompt_hash`
- `seed`
- `decoding_params`
- adapter/provider/serving metadata where applicable

`GenerationResult` must include:

- `status`
- `reason_code`
- `response_text`
- `stop_reason`
- `truncation_flag`
- `usage`
- `latency_s`
- `raw_response`

`MockAdapter.generate` must be deterministic and local.

`MockAdapter.score_continuation` must return:

- `status = "not_applicable"`
- `reason_code = "aligned_scoring_unavailable"`
- `scoring_path = "none"`

Do not fake continuation scoring from generated text.

## Manifest Contract

Run manifests must contain every key required by
`schemas/run_manifest.schema.json`. Mock or unavailable runtime fields may use
`not_available`, but the key must exist and have the correct type.

Specific requirements:

- `persona_jsonl_hash` hashes exact file bytes.
- `dirty_worktree` is a JSON boolean.
- `seeds` is an array of integers.
- `decoding_params` is an object.
- `stop_sequences` is an array of strings.

## Result Row Contract

Result rows must contain every key required by
`schemas/result_row.schema.json`.

Specific requirements:

- Preserve `raw_request` and `raw_response` for base and tuned outputs.
- `raw_request` must include run/persona/variant/model/tokenizer/template audit
  coordinates.
- `base` and `tuned` outputs must include `status` and `reason_code`.
- `metrics.token_kl` is a structured status object.
- `metrics.persona_adherence` is `{"status": "not_run"}` in Sprint 1.
- `metrics.behavioral_consistency_f1` is `{"status": "not_run"}` in Sprint 1.
- `behavior_tags` is `{"status": "not_run"}` in Sprint 1 unless a stricter
  validated extractor output exists.

## Token-KL Status Contract

Every `metrics.token_kl` object must include:

- `status`: one of `ok`, `not_applicable`, `diagnostic_only`, `error`
- `value`: numeric only when `status = "ok"`, otherwise `null`
- `reason_code`
- `scoring_path`: one of `local_forward`, `vllm_prompt_logprobs`,
  `one_token_loop`, `hosted_top_logprobs`, `none`
- `fixed_continuation_id`
- `fixed_continuation_hash`
- `tokenizer_hash_match`
- `vocabulary_match`
- `chat_template_hash_match`
- `k`
- `endpoint_cap`
- `diagnostic_only`

Canonical `ok` Token-KL requires aligned scoring metadata, a numeric `value`, a
non-`none` scoring path, matching tokenizer/vocabulary/chat template, a fixed
continuation id/hash, and `diagnostic_only = false`.

Endpoint-capped hosted logprob output is `diagnostic_only` unless it can score
the same fixed continuation with sufficient depth for the configured `k`.

Free-running generation output must never be used to produce canonical Token-KL.

## Cache Key Contract

The canonical cache-key payload must contain exactly these fields:

- `prompt_hash`
- `system_prompt_hash`
- `model_id`
- `model_revision_or_hash`
- `tokenizer_hash`
- `chat_template_hash`
- `decoding_params`
- `seed`
- `evaluator_version`
- `adapter`
- `provider_or_endpoint`
- `serving_stack`
- `serving_stack_version`
- `scoring_capability`

Serialize with:

```python
json.dumps(payload, sort_keys=True, separators=(",", ":"))
```

Hash with SHA-256 and return `sha256:<hex>`.

Unrelated runtime noise such as latency, process id, or wall-clock timing must
not change the cache key.
