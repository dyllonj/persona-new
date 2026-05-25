# 50-Persona Dev Run Runbook

## Bottom Line

Do not execute the 50-persona dev run until the dev preflight passes with real
promotion, review, smoke, and dev-approval evidence. The expected plan is
exactly:

`50 personas * 6 variants * 2 models * 2 seeds = 1,200 generation calls`.

Non-dry execution is phase-capped: dev may run at up to 50 personas only when
`--run-stage dev` is explicit and every dev evidence gate passes.

## Required Inputs

- `data/personas.full.jsonl`, created only after the dataset promotion gate.
- Sprint 7 dataset promotion manifest for the exact promoted dataset hash.
- Full review manifest for the promoted 200-persona dataset.
- A successful Sprint 8 smoke evidence JSON file.
- Explicit dev-run approval JSON with the promoted dataset hash.
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

Dev-run approval must be a JSON object with:

```json
{
  "approval_type": "dev_run_approval",
  "status": "approved",
  "approved_by": "<name-or-team>",
  "approved_at": "2026-05-25T00:00:00Z",
  "persona_count": 50,
  "planned_generation_calls": 1200,
  "dataset_hash": "sha256:<data-personas-full-jsonl-hash>"
}
```

## Preflight

Run count arithmetic:

```bash
python3 persona_eval.py plan --persona-count 50 --variants-per-persona 6 --model-count 2 --seed-count 2
```

Run the dev gate:

```bash
python3 persona_eval.py preflight --stage dev --persona-path data/personas.full.jsonl --limit-personas 50 --model-count 2 --seed-count 2 --promotion-manifest <promotion-manifest.json> --review-manifest <full-review.jsonl> --smoke-evidence <smoke-evidence.json> --dev-run-approval <dev-run-approval.json>
```

## Execution Command

Only after the cap is intentionally lifted in an approved implementation slice
and preflight has passed:

```bash
python3 persona_eval.py run --run-stage dev --persona-path data/personas.full.jsonl --limit-personas 50 --out results/dev_50_<YYYYMMDD> --adapter vllm --base-url <local-vllm-url> --serving-stack-version <vllm-version> --model-base <base-model-id> --model-tuned <tuned-model-id> --model-base-revision-or-hash <base-revision> --model-tuned-revision-or-hash <tuned-revision> --tokenizer-name <tokenizer-name> --tokenizer-hash <tokenizer-hash> --chat-template-hash <chat-template-hash> --gpu-cuda-driver <gpu-cuda-driver-summary> --promotion-manifest-path <promotion-manifest.json> --review-manifest-path <full-review.jsonl> --smoke-evidence <smoke-evidence.json> --dev-run-approval <dev-run-approval.json> --seeds 1,2 --run-id dev_50_<YYYYMMDD>
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

- Promotion, review, smoke, or dev approval evidence is missing.
- `data/personas.full.jsonl` is absent, changed unexpectedly, or not validated.
- Any runtime or prompt setting changes without cache invalidation and rerun
  notes.
- The team cannot explain whether a dev result came from old or new settings.
