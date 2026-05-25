# Pre-GPU Checklist

## Bottom Line

Do not rent or use GPU capacity until the local repo passes the checks below
without changing `data/personas.full.jsonl`, weakening staged caps, or replacing
approval gates with placeholders.

## Local Checks Before Renting GPU

Run from the repository root:

```bash
python3 persona_eval.py validate --persona-path data/personas.full.jsonl
python3 dataset_readiness.py --persona-path data/personas.full.jsonl --review-manifest reviews/personas.full.review.jsonl --promotion-manifest reports/dataset_promotion_manifest.json
python3 persona_eval.py plan --persona-count 20 --variants-per-persona 6 --model-count 2 --seed-count 1
python3 persona_eval.py plan --persona-count 50 --variants-per-persona 6 --model-count 2 --seed-count 2
python3 persona_eval.py plan --persona-count 200 --variants-per-persona 6 --model-count 2 --seed-count 2
python3 -m json.tool configs/model_pair.qwen2_5_7b.json >/dev/null
python3 -m json.tool approvals/dev_run_approval.template.json >/dev/null
python3 -m json.tool approvals/full_run_approval.template.json >/dev/null
python3 -m unittest discover -s tests
python3 -m pytest
```

Expected plan outputs:

```text
planned_generation_calls=240
planned_generation_calls=1200
planned_generation_calls=4800
```

Also confirm the worktree does not contain accidental changes to the promoted
dataset:

```bash
git diff -- data/personas.full.jsonl
```

Expected output is empty.

## Values To Collect On The GPU Host

Collect these values before any real smoke command:

- `code_commit`: exact `git rev-parse HEAD` used on the GPU host.
- `dirty_worktree`: exact `git status --porcelain` state.
- `dataset_hash`: `sha256` of `data/personas.full.jsonl`.
- `promotion_manifest_hash`: `sha256` of `reports/dataset_promotion_manifest.json`.
- `review_manifest_hash`: `sha256` of `reviews/personas.full.review.jsonl`.
- `model_base`: `Qwen/Qwen2.5-7B`.
- `model_tuned`: `Qwen/Qwen2.5-7B-Instruct`.
- `model_base_revision_or_hash`: immutable Hugging Face revision or local model artifact hash.
- `model_tuned_revision_or_hash`: immutable Hugging Face revision or local model artifact hash.
- `tokenizer_name`: tokenizer path/name actually loaded by the serving stack.
- `tokenizer_hash`: deterministic hash of the tokenizer artifact actually loaded.
- `chat_template_hash`: deterministic hash of the chat template used to render prompts.
- `serving_stack`: `vllm`.
- `serving_stack_version`: exact vLLM version.
- `gpu_cuda_driver`: GPU model, CUDA version, driver version, and visible device count.
- `base_url`: shared local OpenAI-compatible endpoint, or both `base_url_base` and `base_url_tuned`.
- `decoding_params`: temperature `0.0`, max tokens `140`.
- `seeds`: smoke `1`; dev/full `1,2`.
- `raw_request_response_logging_status`: must be `enabled`.

If base and tuned models use separate servers, record both endpoints and pass
them with:

```bash
--base-url-base <local-base-endpoint> --base-url-tuned <local-tuned-endpoint>
```

If they share one server, use:

```bash
--base-url <local-shared-endpoint>
```

## GPU Smoke Sequence

After values are collected and the endpoint is already running on the GPU host:

```bash
python3 persona_eval.py run \
  --run-stage smoke \
  --persona-path data/personas.full.jsonl \
  --limit-personas 20 \
  --out results/vllm_smoke_20 \
  --adapter vllm \
  --base-url <local-vllm-openai-compatible-endpoint> \
  --model-base Qwen/Qwen2.5-7B \
  --model-tuned Qwen/Qwen2.5-7B-Instruct \
  --model-base-revision-or-hash <base-revision-or-hash> \
  --model-tuned-revision-or-hash <tuned-revision-or-hash> \
  --tokenizer-name <tokenizer-name-or-path> \
  --tokenizer-hash <tokenizer-hash> \
  --chat-template-hash <chat-template-hash> \
  --serving-stack-version <vllm-version> \
  --gpu-cuda-driver <gpu-cuda-driver-summary> \
  --promotion-manifest-path reports/dataset_promotion_manifest.json \
  --seeds 1 \
  --temperature 0.0 \
  --max-tokens 140 \
  --disable-token-kl \
  --run-id vllm_smoke_20
```

Then aggregate and build evidence:

```bash
python3 aggregate.py --manifest results/vllm_smoke_20/manifest.json --results results/vllm_smoke_20/results.jsonl --out reports/vllm_smoke_20
python3 smoke_evidence.py --manifest results/vllm_smoke_20/manifest.json --results results/vllm_smoke_20/results.jsonl --aggregate-report reports/vllm_smoke_20/aggregate_report.json --out reports/vllm_smoke_20/smoke_evidence.json
```

## Stop Conditions

Stop before renting GPU if any local check fails.

Stop before starting vLLM if any required runtime value is unknown,
placeholder-shaped, or cannot be reproduced on the GPU host.

Stop before the smoke run if:

- `data/personas.full.jsonl` differs from the promoted dataset hash.
- `reports/dataset_promotion_manifest.json` is missing or mismatched.
- `reviews/personas.full.review.jsonl` is missing or mismatched.
- The endpoint is hosted, remote, or not explicitly approved for this run.
- The endpoint is not OpenAI-compatible.
- The run command exceeds 20 personas, 6 variants, 2 models, or 1 seed.
- Raw request/response logging would be disabled.
- The command omits model revision/hash, tokenizer hash, chat-template hash, vLLM version, or GPU/CUDA/driver metadata.
- The harness attempts canonical Token-KL during Sprint 8 smoke.

Stop after the smoke run if:

- `results/vllm_smoke_20/manifest.json` or `results/vllm_smoke_20/results.jsonl` is missing.
- The aggregate command fails or `reports/vllm_smoke_20/aggregate_report.json` does not validate.
- `python3 smoke_evidence.py ...` fails.
- Smoke evidence does not report `persona_count=20`, `variants_per_persona=6`, `model_count=2`, `seed_count=1`, and `planned_generation_calls=240`.
- Any flagged, truncated, errored, or ambiguous outputs are not reviewed before dev approval.

Do not start the 50-persona dev run until smoke evidence is valid and
`approvals/dev_run_approval.template.json` has been copied to a real approval
artifact with every placeholder replaced.
