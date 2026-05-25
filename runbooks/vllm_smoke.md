# vLLM Smoke Runbook

## Bottom Line

This is Sprint 8 prep only. Do not run the real smoke until Sprint 7 has
promoted `data/personas.full.jsonl`, produced a promotion manifest, and a local
vLLM OpenAI-compatible endpoint is explicitly approved for the run.

The smoke run is capped at 20 personas and must be explicit:

```bash
python3 persona_eval.py plan --persona-count 20 --variants-per-persona 6 --model-count 2 --seed-count 1
```

Expected output:

```text
planned_generation_calls=240
```

## Required Inputs

- `data/personas.full.jsonl` exists and validates.
- Sprint 7 promotion manifest path is known and hashable.
- Local vLLM OpenAI-compatible endpoint is already running.
- Base and tuned model IDs are explicit provider/runtime model IDs.
- Base and tuned model revisions or hashes are recorded.
- Tokenizer name/path and tokenizer hash are recorded.
- Chat-template hash is recorded.
- vLLM version and GPU/CUDA/driver summary are recorded.

Hosted APIs are out of scope unless the user explicitly approves them.

## Command Shape

Replace every angle-bracket value before running:

```bash
python3 persona_eval.py run \
  --run-stage smoke \
  --persona-path data/personas.full.jsonl \
  --limit-personas 20 \
  --out results/vllm_smoke_20 \
  --adapter vllm \
  --base-url http://localhost:8000/v1 \
  --model-base <base-model-id> \
  --model-tuned <tuned-model-id> \
  --model-base-revision-or-hash <base-model-revision-or-hash> \
  --model-tuned-revision-or-hash <tuned-model-revision-or-hash> \
  --tokenizer-name <tokenizer-name-or-path> \
  --tokenizer-hash <tokenizer-hash> \
  --chat-template-hash <chat-template-hash> \
  --serving-stack-version <vllm-version> \
  --gpu-cuda-driver <gpu-cuda-driver-summary> \
  --promotion-manifest-path <promotion-manifest-path> \
  --seeds 1 \
  --temperature 0.0 \
  --max-tokens 140 \
  --disable-token-kl \
  --run-id vllm_smoke_20
```

Then aggregate only if the smoke run wrote a valid manifest and results file:

```bash
python3 aggregate.py --manifest results/vllm_smoke_20/manifest.json --results results/vllm_smoke_20/results.jsonl --out reports/vllm_smoke_20
```

## Manifest Checks

Before accepting the smoke output, inspect `results/vllm_smoke_20/manifest.json`
for:

- `persona_jsonl_hash`
- `promotion_manifest_hash`
- base/tuned model IDs and revisions
- tokenizer name and hash
- chat-template hash
- serving stack and version
- GPU/CUDA/driver summary
- decoding params and seeds
- `raw_request_response_logging_status: enabled`

Each result row must include base and tuned model outputs with raw request,
raw response, usage, latency, stop reason, and truncation flag.

## Token-KL

Keep `--disable-token-kl` for the Sprint 8 smoke. Canonical Token-KL remains
`not_applicable` until fixed aligned continuation scoring is proven with the
same rendered prompt/context, same fixed continuation, matching tokenizer and
chat-template hashes, and sufficient logprob depth.

## Stop Conditions

Stop before real execution if:

- `data/personas.full.jsonl` is missing or does not validate.
- Promotion manifest evidence is missing.
- Any explicit runtime metadata is unavailable.
- The endpoint is not local or not explicitly approved.
- The requested persona limit is above 20.
- Raw request/response logging would be disabled.
- The code attempts to report canonical Token-KL without proven aligned scoring.
