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
- If base and tuned models use separate servers, both local endpoints are known.
- Base and tuned model IDs are explicit provider/runtime model IDs.
- Base and tuned model revisions or hashes are recorded.
- Tokenizer name/path and tokenizer hash are recorded.
- Chat-template hash is recorded.
- The base-model endpoint has an explicit chat-template policy for `/chat/completions`.
- vLLM version and GPU/CUDA/driver summary are recorded.
- If a production/open model matrix entry is used, the matrix is real-run ready,
  license-reviewed, and passed with `--model-matrix` plus `--model-matrix-entry`.

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
  --model-matrix <real-run-ready-model-matrix-json> \
  --model-matrix-entry <matrix-entry-id> \
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

If the base and tuned models are served from separate local endpoints, replace
`--base-url` with:

```bash
  --base-url-base http://localhost:8001/v1 \
  --base-url-tuned http://localhost:8002/v1 \
```

Then aggregate only if the smoke run wrote a valid manifest and results file:

```bash
python3 aggregate.py --manifest results/vllm_smoke_20/manifest.json --results results/vllm_smoke_20/results.jsonl --out reports/vllm_smoke_20
```

Then build the smoke evidence artifact required by the dev/full gates:

```bash
python3 smoke_evidence.py --manifest results/vllm_smoke_20/manifest.json --results results/vllm_smoke_20/results.jsonl --aggregate-report reports/vllm_smoke_20/aggregate_report.json --out reports/vllm_smoke_20/smoke_evidence.json
```

## Manifest Checks

Before accepting the smoke output, inspect `results/vllm_smoke_20/manifest.json`
for:

- `persona_jsonl_hash`
- `promotion_manifest_hash`
- base/tuned model IDs and revisions
- `model_endpoints.base.provider_or_endpoint`
- `model_endpoints.tuned.provider_or_endpoint`
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
- The base model endpoint cannot safely serve `/chat/completions` with a recorded chat-template policy.
- A selected model matrix entry is not real-run ready, license-reviewed, or compatible with the requested Token-KL score mode.
- The requested persona limit is above 20.
- Raw request/response logging would be disabled.
- The code attempts to report canonical Token-KL without proven aligned scoring.
- `smoke_evidence.py` fails to produce `reports/vllm_smoke_20/smoke_evidence.json`.
