import unittest
from pathlib import Path

from persona_eval import (
    MockAdapter,
    build_generation_request,
    build_score_continuation_request,
    load_jsonl,
    render_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        row = load_jsonl(SAMPLE_PATH)[0]
        cls.rendered = render_prompt(row, row["variants"][0])

    def test_mock_generate_returns_valid_deterministic_result(self):
        adapter = MockAdapter()
        request = build_generation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id="fixture_001",
            variant_id="seed_001_v0",
            variant_type="canonical",
            model_alias="base",
            model_id="base",
            seed=7,
            decoding_params={"temperature": 0.0, "max_tokens": 32},
            adapter=adapter,
        )

        first = adapter.generate(request)
        second = adapter.generate(request)

        self.assertEqual(first, second)
        self.assertEqual(first.stop_reason, "mock_complete")
        self.assertFalse(first.truncation_flag)
        self.assertEqual(first.latency_s, 0.0)
        self.assertEqual(first.usage["total_tokens"], first.usage["prompt_tokens"] + first.usage["completion_tokens"])
        self.assertEqual(first.raw_response["adapter"], "mock")
        self.assertEqual(first.raw_response["run_id"], "test-run")
        self.assertEqual(first.to_model_output(request)["status"], "ok")
        self.assertIn("tokenizer_hash", first.to_model_output(request)["raw_request"])
        self.assertEqual(first.to_model_output(request)["raw_request"]["scoring_capability"], "none")

    def test_score_continuation_request_carries_audit_coordinates(self):
        adapter = MockAdapter()
        request = build_score_continuation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id="fixture_001",
            variant_id="seed_001_v0",
            variant_type="canonical",
            model_alias="base",
            model_id="base",
            model_revision_or_hash="base-rev",
            tokenizer_name="fixture-tokenizer",
            tokenizer_hash="sha256:tokenizer",
            chat_template_hash="sha256:chat",
            seed=7,
            decoding_params={"temperature": 0.0, "max_tokens": 32},
            stop_sequences=["\n\n"],
            fixed_continuation="fixed response",
            fixed_continuation_id="fixture-continuation",
            adapter=adapter,
        )
        raw_request = request.to_raw_request()

        self.assertEqual(raw_request["run_id"], "test-run")
        self.assertEqual(raw_request["persona_id"], "fixture_001")
        self.assertEqual(raw_request["variant_id"], "seed_001_v0")
        self.assertEqual(raw_request["variant_type"], "canonical")
        self.assertEqual(raw_request["model_alias"], "base")
        self.assertEqual(raw_request["model_id"], "base")
        self.assertEqual(raw_request["model_revision_or_hash"], "base-rev")
        self.assertEqual(raw_request["tokenizer_name"], "fixture-tokenizer")
        self.assertEqual(raw_request["tokenizer_hash"], "sha256:tokenizer")
        self.assertEqual(raw_request["chat_template_hash"], "sha256:chat")
        self.assertEqual(raw_request["prompt_template_version"], self.rendered.prompt_template_version)
        self.assertEqual(raw_request["prompt_template_hash"], self.rendered.prompt_template_hash)
        self.assertEqual(raw_request["prompt_hash"], self.rendered.prompt_hash)
        self.assertEqual(raw_request["system_prompt_hash"], self.rendered.system_prompt_hash)
        self.assertEqual(raw_request["seed"], 7)
        self.assertEqual(raw_request["decoding_params"], {"temperature": 0.0, "max_tokens": 32})
        self.assertEqual(raw_request["stop_sequences"], ["\n\n"])
        self.assertEqual(raw_request["adapter"], "mock")
        self.assertEqual(raw_request["provider_or_endpoint"], "local_mock")
        self.assertEqual(raw_request["serving_stack"], "mock")
        self.assertEqual(raw_request["serving_stack_version"], "sprint1")
        self.assertEqual(raw_request["scoring_capability"], "none")

    def test_mock_score_continuation_returns_not_applicable(self):
        adapter = MockAdapter()
        request = build_score_continuation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id="fixture_001",
            variant_id="seed_001_v0",
            variant_type="canonical",
            model_alias="base",
            model_id="base",
            fixed_continuation="fixed response",
            fixed_continuation_id="fixture-continuation",
            adapter=adapter,
        )

        result = adapter.score_continuation(request)
        status = result.to_token_kl_status()

        self.assertEqual(status["status"], "not_applicable")
        self.assertEqual(status["reason_code"], "aligned_scoring_unavailable")
        self.assertEqual(status["scoring_path"], "none")
        self.assertFalse(status["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
