# Production/Open Model Matrix

## Bottom Line

Use `configs/model_matrix.production_open.json` to describe model-comparison
intent before any GPU or hosted endpoint is touched. The matrix is configuration
only. It does not make any model a runtime dependency, and it does not authorize
Token-KL unless the normal aligned fixed-continuation scoring gates are proven.

Validate the matrix locally:

```bash
python3 persona_eval.py validate-model-matrix --matrix-path configs/model_matrix.production_open.json
```

Expected template output includes:

```text
model_matrix_status=template_valid
real_run_readiness=blocked
```

The readiness block is intentional while endpoint and revision/hash placeholders
remain. To require a filled, execution-ready matrix:

```bash
python3 persona_eval.py validate-model-matrix --matrix-path configs/model_matrix.production_open.json --require-real-run-ready
```

That command must fail until every `provider_or_endpoint` and
`required_revision_or_hash` placeholder has been replaced with reproducible
runtime values.

## Comparison Types

### Same-Family Drift Pairs

Same-family base/instruct pairs, such as `Qwen/Qwen2.5-7B` versus
`Qwen/Qwen2.5-7B-Instruct`, may declare `token_kl_applicability:
canonical_possible`. That means possible after proof, not available now.

Canonical Token-KL still requires all aligned-scoring metadata:

- same tokenizer family
- same vocabulary
- fixed prompt rendering hash
- shared chat-template policy
- fixed continuation scoring
- comparable next-token probabilities or logprobs

Before that proof exists, result rows must keep Token-KL `not_applicable` or
`diagnostic_only`.

### Standalone Instruct Models

Standalone production/open instruct entries are production comparators, not drift
pairs. Use Persona Adherence and Behavioral-Consistency F1. Token-KL must be
`not_applicable` unless an explicit paired baseline is configured and reviewed.

### Cross-Family Comparisons

Cross-family production comparisons are PA/BC-F1 only unless a future reviewed
exception proves the aligned Token-KL constraints. Llama-vs-Qwen,
Mistral-vs-Gemma, and similar comparisons must not declare canonical Token-KL
possible in this matrix.

## Result-Row Guard

If a result row carries `model_pair.token_kl_applicability` or
`model_pair.metric_applicability.token_kl`, validation rejects
`metrics.token_kl.status: ok` when the applicability is `not_applicable` or
`diagnostic_only`. Aggregation uses the same result-row validator, so invalid
canonical Token-KL cannot enter aggregate summaries through that path.

## Stop Conditions

Stop before execution if:

- `validate-model-matrix` fails.
- `--require-real-run-ready` fails for a run intended to use the matrix.
- Any model revision/hash is missing or placeholder-shaped.
- Any endpoint is missing, hosted unexpectedly, or not approved for the run.
- A cross-family or standalone comparison attempts canonical Token-KL.
- A same-family pair lacks aligned fixed-continuation scoring proof.
