import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from persona_eval import (
    VLLMOpenAIAdapter,
    _adapters_from_args,
    build_generation_request,
    build_score_continuation_request,
    load_jsonl,
    render_prompt,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class FakeHTTPResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


class VLLMAdapterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        row = load_jsonl(SAMPLE_PATH)[0]
        cls.row = row
        cls.variant = row["variants"][0]
        cls.rendered = render_prompt(row, cls.variant)

    def build_request(self):
        adapter = VLLMOpenAIAdapter(
            base_url="http://localhost:8000/v1",
            serving_stack_version="vllm-fixture",
            timeout_s=3.0,
        )
        request = build_generation_request(
            rendered=self.rendered,
            run_id="vllm-contract",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="base",
            model_id="meta-llama/Meta-Llama-3-8B",
            model_revision_or_hash="fixture-model-revision",
            tokenizer_name="meta-llama/Meta-Llama-3-8B",
            tokenizer_hash="sha256:fixture-tokenizer",
            chat_template_hash="sha256:fixture-chat-template",
            seed=1,
            decoding_params={"temperature": 0.0, "max_tokens": 32},
            stop_sequences=["<END>"],
            adapter=adapter,
        )
        return adapter, request

    def test_generate_posts_openai_payload_and_preserves_raw_logging(self):
        adapter, request = self.build_request()
        response_payload = {
            "id": "chatcmpl-fixture",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "fixture response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 2,
                "total_tokens": 13,
            },
        }

        with mock.patch(
            "urllib.request.urlopen",
            return_value=FakeHTTPResponse(response_payload),
        ) as urlopen:
            result = adapter.generate(request)

        http_request = urlopen.call_args.args[0]
        self.assertEqual(http_request.full_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 3.0)
        payload = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "meta-llama/Meta-Llama-3-8B")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 32)
        self.assertEqual(payload["seed"], 1)
        self.assertEqual(payload["stop"], ["<END>"])

        self.assertEqual(result.response_text, "fixture response")
        self.assertEqual(result.stop_reason, "stop")
        self.assertFalse(result.truncation_flag)
        self.assertEqual(result.usage, {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13})
        self.assertEqual(result.raw_response, response_payload)

        model_output = result.to_model_output(request)
        self.assertEqual(model_output["raw_request"]["adapter"], "vllm")
        self.assertEqual(model_output["raw_request"]["provider_or_endpoint"], "http://localhost:8000/v1")
        self.assertEqual(model_output["raw_request"]["serving_stack"], "vllm")
        self.assertEqual(model_output["raw_request"]["serving_stack_version"], "vllm-fixture")
        self.assertEqual(model_output["raw_request"]["tokenizer_hash"], "sha256:fixture-tokenizer")
        self.assertEqual(model_output["raw_request"]["chat_template_hash"], "sha256:fixture-chat-template")
        self.assertEqual(model_output["raw_response"], response_payload)

    def test_length_finish_reason_sets_truncation_flag(self):
        adapter, request = self.build_request()
        response_payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "truncated"},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
        }

        with mock.patch("urllib.request.urlopen", return_value=FakeHTTPResponse(response_payload)):
            result = adapter.generate(request)

        self.assertEqual(result.stop_reason, "length")
        self.assertTrue(result.truncation_flag)

    def test_score_continuation_stays_not_applicable_without_proven_alignment(self):
        adapter, _ = self.build_request()
        request = build_score_continuation_request(
            rendered=self.rendered,
            run_id="vllm-contract",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="tuned",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            fixed_continuation="fixed continuation",
            fixed_continuation_id="fixture-continuation",
            k=50,
            adapter=adapter,
        )

        status = adapter.score_continuation(request).to_token_kl_status()

        self.assertEqual(status["status"], "not_applicable")
        self.assertEqual(status["reason_code"], "aligned_scoring_unavailable")
        self.assertEqual(status["scoring_path"], "none")
        self.assertEqual(status["fixed_continuation_id"], "fixture-continuation")
        self.assertEqual(status["fixed_continuation_hash"], sha256_text("fixed continuation"))
        self.assertEqual(status["k"], 50)
        self.assertFalse(status["diagnostic_only"])

    def test_distinct_endpoint_args_are_recorded_in_raw_requests_without_network(self):
        args = argparse.Namespace(
            adapter="vllm",
            base_url=None,
            base_url_base="http://localhost:8001/v1",
            base_url_tuned="http://localhost:8002/v1",
            api_key_env=None,
            serving_stack=None,
            serving_stack_version="vllm-fixture",
        )

        adapters = _adapters_from_args(args)
        base_request = build_generation_request(
            rendered=self.rendered,
            run_id="vllm-contract",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="base",
            model_id="Qwen/Qwen2.5-7B",
            seed=1,
            adapter=adapters.base,
        )
        tuned_request = build_generation_request(
            rendered=self.rendered,
            run_id="vllm-contract",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="tuned",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            seed=1,
            adapter=adapters.tuned,
        )

        self.assertEqual(base_request.to_raw_request()["provider_or_endpoint"], "http://localhost:8001/v1")
        self.assertEqual(tuned_request.to_raw_request()["provider_or_endpoint"], "http://localhost:8002/v1")

    def test_shared_base_url_remains_backward_compatible(self):
        args = argparse.Namespace(
            adapter="vllm",
            base_url="http://localhost:8000/v1",
            base_url_base=None,
            base_url_tuned=None,
            api_key_env=None,
            serving_stack=None,
            serving_stack_version="vllm-fixture",
        )

        adapters = _adapters_from_args(args)

        self.assertEqual(adapters.base.provider_or_endpoint, "http://localhost:8000/v1")
        self.assertEqual(adapters.tuned.provider_or_endpoint, "http://localhost:8000/v1")

    def test_vllm_dry_run_refuses_more_than_twenty_personas(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "persona_eval.py"),
                "run",
                "--persona-count",
                "21",
                "--variants-per-persona",
                "6",
                "--adapter",
                "vllm",
                "--base-url",
                "http://localhost:8000/v1",
                "--model-base",
                "base-model",
                "--model-tuned",
                "tuned-model",
                "--seeds",
                "1",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("capped at 20 personas", completed.stderr)

    def test_vllm_non_dry_requires_promoted_full_dataset_path_before_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "persona_eval.py"),
                    "run",
                    "--persona-path",
                    str(SAMPLE_PATH),
                    "--limit-personas",
                    "1",
                    "--out",
                    str(Path(tmp) / "vllm-smoke"),
                    "--adapter",
                    "vllm",
                    "--base-url",
                    "http://localhost:8000/v1",
                    "--model-base",
                    "base-model",
                    "--model-tuned",
                    "tuned-model",
                    "--model-base-revision-or-hash",
                    "base-revision",
                    "--model-tuned-revision-or-hash",
                    "tuned-revision",
                    "--tokenizer-name",
                    "fixture-tokenizer",
                    "--tokenizer-hash",
                    "sha256:fixture-tokenizer",
                    "--chat-template-hash",
                    "sha256:fixture-chat-template",
                    "--serving-stack-version",
                    "vllm-fixture",
                    "--promotion-manifest-path",
                    str(SAMPLE_PATH),
                    "--seeds",
                    "1",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--persona-path data/personas.full.jsonl", completed.stderr)


if __name__ == "__main__":
    unittest.main()
