# Full 200-Persona Run Runbook

## Bottom Line

Do not execute the full benchmark until every full preflight artifact exists and
passes validation. The expected plan is exactly:

`200 personas * 6 variants * 2 models * 2 seeds = 4,800 generation calls`.

Non-dry execution is phase-capped: full may run at up to 200 personas only when
`--run-stage full` is explicit and every full-run evidence gate passes.

## Required Inputs

- `data/personas.full.jsonl` with exactly 200 validated rows.
- Dataset promotion manifest with `manifest_type: "dataset_promotion"`,
  `status: "promoted"`, `persona_count: 200`, and matching `dataset_hash`.
- Full review manifest JSONL with exactly 200 approved rows and evidence for
  semantic equivalence, NLI equivalence, contradiction, safety, and gold labels.
- Successful 20-persona smoke evidence.
- Successful 50-persona dev evidence.
- Explicit full-run approval JSON.
- Filled runtime values from `configs/full.vllm.json`.

Full-run approval must be a JSON object with:

```json
{
  "approval_type": "full_run_approval",
  "status": "approved",
  "approved_by": "<name-or-team>",
  "approved_at": "2026-05-25T00:00:00Z",
  "persona_count": 200,
  "planned_generation_calls": 4800,
  "dataset_hash": "sha256:<data-personas-full-jsonl-hash>"
}
```

## Preflight

Run count arithmetic:

```bash
python3 persona_eval.py plan --persona-count 200 --variants-per-persona 6 --model-count 2 --seed-count 2
```

Run the full gate:

```bash
python3 persona_eval.py preflight --stage full --persona-path data/personas.full.jsonl --model-count 2 --seed-count 2 --promotion-manifest <promotion-manifest.json> --review-manifest <full-review.jsonl> --smoke-evidence <smoke-evidence.json> --dev-evidence <dev-evidence.json> --full-run-approval <full-run-approval.json> --model-base <base-model-id> --model-tuned <tuned-model-id> --model-base-revision-or-hash <base-revision> --model-tuned-revision-or-hash <tuned-revision> --tokenizer-name <tokenizer-name> --tokenizer-hash <tokenizer-hash> --chat-template-hash <chat-template-hash> --serving-stack-version <vllm-version>
```

The full preflight blocks if any approval artifact is missing, if the dataset
hashes do not match, or if runtime metadata still uses placeholders.

## Execution Command

Only after the cap is intentionally lifted in an approved implementation slice
and preflight has passed:

```bash
python3 persona_eval.py run --run-stage full --persona-path data/personas.full.jsonl --out results/full_200_<YYYYMMDD> --adapter vllm --base-url <local-vllm-url> --serving-stack-version <vllm-version> --model-base <base-model-id> --model-tuned <tuned-model-id> --model-base-revision-or-hash <base-revision> --model-tuned-revision-or-hash <tuned-revision> --tokenizer-name <tokenizer-name> --tokenizer-hash <tokenizer-hash> --chat-template-hash <chat-template-hash> --gpu-cuda-driver <gpu-cuda-driver-summary> --promotion-manifest-path <promotion-manifest.json> --review-manifest-path <full-review.jsonl> --smoke-evidence <smoke-evidence.json> --dev-evidence <dev-evidence.json> --full-run-approval <full-run-approval.json> --seeds 1,2 --run-id full_200_<YYYYMMDD>
python3 aggregate.py --manifest results/full_200_<YYYYMMDD>/manifest.json --results results/full_200_<YYYYMMDD>/results.jsonl --out reports/full_200_<YYYYMMDD>
```

## Required Manifest Metadata

The run manifest must record:

- `code_commit`
- `dirty_worktree`
- `persona_jsonl_hash`
- `prompt_template_hash`
- `model_base` and `model_tuned`
- `model_base_revision_or_hash` and `model_tuned_revision_or_hash`
- `tokenizer_name` and `tokenizer_hash`
- `chat_template_hash`
- `serving_stack_version`

## Stop Conditions

- Promotion, review, smoke, dev, or full approval artifacts are missing.
- `data/personas.full.jsonl` is not exactly 200 valid rows.
- Any artifact hash mismatches the dataset or approval record.
- Metrics, schemas, prompts, runtime settings, or model configuration change
  after the run starts.
- Raw request/response metadata cannot be preserved.
