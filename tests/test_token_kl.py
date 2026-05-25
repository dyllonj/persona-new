import unittest
from pathlib import Path

from persona_eval import (
    MockAdapter,
    PersonaValidationError,
    TokenKLUnavailableError,
    build_generation_request,
    canonical_token_kl_from_generation_outputs,
    load_jsonl,
    render_prompt,
    token_kl_from_aligned_topk,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class TokenKLGuardrailTests(unittest.TestCase):
    def token_kl(self, base_steps, tuned_steps, **overrides):
        kwargs = {
            "fixed_continuation_id": "fixture-continuation",
            "fixed_continuation": "fixed response",
            "tokenizer_hash_match": True,
            "vocabulary_match": True,
            "chat_template_hash_match": True,
            "k": 50,
        }
        kwargs.update(overrides)
        return token_kl_from_aligned_topk(base_steps, tuned_steps, **kwargs)

    def test_identical_aligned_distributions_are_near_zero(self):
        status = self.token_kl(
            [{"a": 0.7, "b": 0.3}, {"x": 0.5, "y": 0.5}],
            [{"a": 0.7, "b": 0.3}, {"x": 0.5, "y": 0.5}],
        )

        self.assertEqual(status["status"], "ok")
        self.assertAlmostEqual(status["value"], 0.0, places=10)
        self.assertFalse(status["diagnostic_only"])

    def test_shifted_aligned_distributions_are_positive(self):
        status = self.token_kl(
            [{"a": 0.9, "b": 0.1}],
            [{"a": 0.1, "b": 0.9}],
        )

        self.assertEqual(status["status"], "ok")
        self.assertGreater(status["value"], 0.0)

    def test_logprob_inputs_are_supported(self):
        status = self.token_kl(
            [{"a": -0.3566749439, "b": -1.2039728043}],
            [{"a": -0.3566749439, "b": -1.2039728043}],
            input_type="logprob",
        )

        self.assertEqual(status["status"], "ok")
        self.assertAlmostEqual(status["value"], 0.0, places=9)

    def test_mismatches_are_not_applicable(self):
        cases = [
            ("tokenizer_hash_match", False, "tokenizer_hash_mismatch"),
            ("vocabulary_match", False, "vocabulary_mismatch"),
            ("chat_template_hash_match", False, "chat_template_hash_mismatch"),
        ]

        for field, value, reason_code in cases:
            with self.subTest(field=field):
                status = self.token_kl([{"a": 1.0}], [{"a": 1.0}], **{field: value})
                self.assertEqual(status["status"], "not_applicable")
                self.assertEqual(status["reason_code"], reason_code)
                self.assertIsNone(status["value"])

    def test_missing_fixed_continuation_is_not_applicable(self):
        status = self.token_kl(
            [{"a": 1.0}],
            [{"a": 1.0}],
            fixed_continuation_id=None,
            fixed_continuation=None,
        )

        self.assertEqual(status["status"], "not_applicable")
        self.assertEqual(status["reason_code"], "missing_fixed_continuation")

    def test_endpoint_cap_below_k_is_diagnostic_only(self):
        status = self.token_kl(
            [{"a": 1.0}],
            [{"a": 1.0}],
            scoring_path="hosted_top_logprobs",
            endpoint_cap=5,
            k=50,
        )

        self.assertEqual(status["status"], "diagnostic_only")
        self.assertEqual(status["reason_code"], "endpoint_top_logprobs_cap_below_k")
        self.assertEqual(status["scoring_path"], "hosted_top_logprobs")
        self.assertTrue(status["diagnostic_only"])

    def test_invalid_distributions_are_rejected(self):
        with self.assertRaises(PersonaValidationError):
            self.token_kl([{"a": -1.0}], [{"a": 1.0}])
        with self.assertRaises(PersonaValidationError):
            self.token_kl([], [])

    def test_free_running_outputs_cannot_create_canonical_token_kl(self):
        row = load_jsonl(SAMPLE_PATH)[0]
        variant = row["variants"][0]
        rendered = render_prompt(row, variant)
        adapter = MockAdapter()
        base_request = build_generation_request(
            rendered=rendered,
            run_id="test-run",
            persona_id=row["persona_id"],
            variant_id=variant["variant_id"],
            variant_type=variant["type"],
            model_alias="base",
            model_id="base",
            seed=1,
            adapter=adapter,
        )
        tuned_request = build_generation_request(
            rendered=rendered,
            run_id="test-run",
            persona_id=row["persona_id"],
            variant_id=variant["variant_id"],
            variant_type=variant["type"],
            model_alias="tuned",
            model_id="tuned",
            seed=1,
            adapter=adapter,
        )

        with self.assertRaises(TokenKLUnavailableError):
            canonical_token_kl_from_generation_outputs(
                adapter.generate(base_request),
                adapter.generate(tuned_request),
            )


if __name__ == "__main__":
    unittest.main()
