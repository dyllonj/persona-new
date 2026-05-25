# 50-Persona Dev Run Runbook

## Bottom Line

Do not execute the 50-persona dev run until the dev preflight passes or an
explicit smoke-evidence override is recorded. The expected plan is exactly:

`50 personas * 6 variants * 2 models * 2 seeds = 1,200 generation calls`.

Current non-dry staged execution remains capped at 20 personas. Do not bypass
that cap in this runbook.

## Required Inputs

- `data/personas.full.jsonl`, created only after the dataset promotion gate.
- A successful Sprint 8 smoke evidence JSON file.
- Filled runtime values from `configs/dev.vllm.json`.
- Explicit model IDs, model revisions/hashes, tokenizer hash, chat-template
  hash, vLLM/runtime version, and GPU/CUDA/driver summary.

Smoke evidence must be a JSON object with:

```json
{
  "evidence_type": "smoke",
  "status": "passed",
  "persona_count": 20,
  "variants_per_persona": 6,
  "model_count": 2,
  "seed_count": 1,
  "planned_generation_calls": 240,
  "manifest_path": "results/vllm_smoke_20/manifest.json",
  "results_path": "results/vllm_smoke_20/results.jsonl",
  "aggregate_report_path": "reports/vllm_smoke_20/aggregate_report.json"
}
```

## Preflight

Run count arithmetic:

```bash
python3 persona_eval.py plan --persona-count 50 --variants-per-persona 6 --model-count 2 --seed-count 2
```

Run the dev gate:

```bash
python3 persona_eval.py preflight --stage dev --persona-count 50 --variants-per-persona 6 --model-count 2 --seed-count 2 --smoke-evidence <smoke-evidence.json>
```

If smoke evidence is intentionally overridden, the override must be explicit:

```bash
python3 persona_eval.py preflight --stage dev --persona-count 50 --variants-per-persona 6 --model-count 2 --seed-count 2 --allow-dev-without-smoke-evidence --approval-override-reason "<operator/date/reason>"
```

## Execution Command

Only after the cap is intentionally lifted in an approved implementation slice
and preflight has passed:

```bash
python3 persona_eval.py run --run-stage dev --persona-path data/personas.full.jsonl --limit-personas 50 --out results/dev_50_<YYYYMMDD> --adapter vllm --base-url <local-vllm-url> --serving-stack-version <vllm-version> --model-base <base-model-id> --model-tuned <tuned-model-id> --model-base-revision-or-hash <base-revision> --model-tuned-revision-or-hash <tuned-revision> --tokenizer-name <tokenizer-name> --tokenizer-hash <tokenizer-hash> --chat-template-hash <chat-template-hash> --gpu-cuda-driver <gpu-cuda-driver-summary> --promotion-manifest-path <promotion-manifest.json> --review-manifest-path <full-review.jsonl> --smoke-evidence <smoke-evidence.json> --seeds 1,2 --run-id dev_50_<YYYYMMDD>
python3 aggregate.py --manifest results/dev_50_<YYYYMMDD>/manifest.json --results results/dev_50_<YYYYMMDD>/results.jsonl --out reports/dev_50_<YYYYMMDD>
```

## Review Checklist

- Planned call count is `1,200`.
- Manifest records code commit, dirty marker, dataset hash, prompt hash, model
  IDs/revisions, tokenizer/chat-template hashes, and runtime version.
- Token-KL remains `not_applicable` unless aligned scoring is proven.
- Review truncation rate, adapter errors, behavior-tag ambiguity, PA status,
  Token-KL availability, and per-variant breakdowns.
- Record a full-run go/no-go decision only after dev evidence is reviewed.

## Stop Conditions

- Smoke evidence is missing and no explicit override is recorded.
- `data/personas.full.jsonl` is absent, changed unexpectedly, or not validated.
- Any runtime or prompt setting changes without cache invalidation and rerun
  notes.
- The team cannot explain whether a dev result came from old or new settings.
