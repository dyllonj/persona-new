import copy
import hashlib
import unittest
from pathlib import Path

from persona_eval import (
    CACHE_KEY_FIELDS,
    EVALUATOR_VERSION,
    MockAdapter,
    PersonaValidationError,
    RESULT_ROW_SCHEMA_PATH,
    ScoreContinuationResult,
    TokenKLUnavailableError,
    build_cache_key_payload,
    build_generation_request,
    build_result_row,
    build_score_continuation_request,
    cache_key_from_payload,
    canonical_token_kl_from_generation_outputs,
    behavior_consistency_f1,
    behavior_tags_ok,
    create_run_manifest,
    endpoint_capped_token_kl_status,
    load_jsonl,
    render_prompt,
    score_persona_adherence_mock,
    validate_result_row,
    validate_run_manifest,
    validate_schema_file,
    validate_token_kl_status,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class ManifestAndResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = load_jsonl(SAMPLE_PATH)[0]
        cls.variant = cls.row["variants"][0]
        cls.rendered = render_prompt(cls.row, cls.variant)

    def build_sprint2_result_row(self):
        adapter = MockAdapter()
        base_request = build_generation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="base",
            model_id="base",
            seed=1,
            adapter=adapter,
        )
        tuned_request = build_generation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="tuned",
            model_id="tuned",
            seed=1,
            adapter=adapter,
        )
        score_request = build_score_continuation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="tuned",
            model_id="tuned",
            fixed_continuation="fixed response",
            fixed_continuation_id="fixture-continuation",
        )
        return build_result_row(
            run_id="test-run",
            persona_row=self.row,
            variant=self.variant,
            seed=1,
            rendered=self.rendered,
            model_pair={"base": "base", "tuned": "tuned"},
            base_request=base_request,
            base_result=adapter.generate(base_request),
            tuned_request=tuned_request,
            tuned_result=adapter.generate(tuned_request),
            score_result=adapter.score_continuation(score_request),
            behavior_tags=behavior_tags_ok(self.row["expected_behavior"]),
            persona_adherence_metric=score_persona_adherence_mock(
                self.row,
                "I am cautious with vendor risk and ask for evidence before making recommendations.",
            ),
            behavioral_consistency_metric=behavior_consistency_f1(
                self.row["expected_behavior"],
                self.row["annotation"]["gold_labels"],
            ),
        )

    def test_manifest_includes_required_keys_and_exact_file_hash(self):
        manifest = create_run_manifest(
            persona_path=SAMPLE_PATH,
            model_base="base",
            model_tuned="tuned",
            seeds=[1],
            run_id="test-run",
            timestamp_utc="2026-05-25T00:00:00Z",
            code_commit="test-commit",
            dirty_worktree=True,
            decoding_params={"temperature": 0.0},
            stop_sequences=["\n\n"],
        )
        required = {
            "run_id",
            "timestamp_utc",
            "code_commit",
            "dirty_worktree",
            "persona_jsonl_hash",
            "prompt_template_version",
            "prompt_template_hash",
            "chat_template_hash",
            "tokenizer_name",
            "tokenizer_hash",
            "model_base",
            "model_tuned",
            "model_base_revision_or_hash",
            "model_tuned_revision_or_hash",
            "adapter",
            "provider_or_endpoint",
            "serving_stack",
            "serving_stack_version",
            "scoring_capability",
            "gpu_cuda_driver",
            "decoding_params",
            "seeds",
            "stop_sequences",
            "extractor_version",
            "embedding_model_revision",
            "nli_or_judge_model_revision",
        }

        self.assertTrue(required.issubset(manifest))
        self.assertIs(manifest["dirty_worktree"], True)
        expected_hash = "sha256:" + hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest()
        self.assertEqual(manifest["persona_jsonl_hash"], expected_hash)
        validate_run_manifest(manifest)

    def test_result_row_includes_required_keys_and_token_kl_not_applicable(self):
        adapter = MockAdapter()
        base_request = build_generation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="base",
            model_id="base",
            model_revision_or_hash="base-rev",
            seed=1,
            adapter=adapter,
        )
        tuned_request = build_generation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="tuned",
            model_id="tuned",
            model_revision_or_hash="tuned-rev",
            seed=1,
            adapter=adapter,
        )
        score_request = build_score_continuation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="tuned",
            model_id="tuned",
            fixed_continuation="fixed response",
            fixed_continuation_id="fixture-continuation",
        )

        base_result = adapter.generate(base_request)
        tuned_result = adapter.generate(tuned_request)
        score_result = adapter.score_continuation(score_request)
        result_row = build_result_row(
            run_id="test-run",
            persona_row=self.row,
            variant=self.variant,
            seed=1,
            rendered=self.rendered,
            model_pair={"base": "base", "tuned": "tuned"},
            base_request=base_request,
            base_result=base_result,
            tuned_request=tuned_request,
            tuned_result=tuned_result,
            score_result=score_result,
        )

        validate_result_row(result_row)
        self.assertEqual(result_row["behavior_tags"], {"status": "not_run"})
        self.assertEqual(result_row["base"]["raw_response"], base_result.raw_response)
        self.assertEqual(result_row["tuned"]["raw_response"], tuned_result.raw_response)
        self.assertEqual(result_row["base"]["raw_request"]["run_id"], "test-run")
        self.assertEqual(result_row["base"]["raw_request"]["persona_id"], self.row["persona_id"])
        self.assertEqual(result_row["base"]["raw_request"]["variant_id"], self.variant["variant_id"])
        self.assertEqual(result_row["base"]["raw_request"]["variant_type"], self.variant["type"])
        self.assertEqual(result_row["base"]["raw_request"]["model_alias"], "base")
        self.assertEqual(result_row["tuned"]["raw_request"]["model_alias"], "tuned")
        self.assertIn("chat_template_hash", result_row["base"]["raw_request"])
        self.assertIn("tokenizer_hash", result_row["base"]["raw_request"])
        self.assertEqual(result_row["metrics"]["token_kl"]["status"], "not_applicable")
        self.assertEqual(
            result_row["metrics"]["token_kl"]["reason_code"],
            "aligned_scoring_unavailable",
        )

    def test_manifest_rejects_non_boolean_dirty_worktree(self):
        manifest = create_run_manifest(
            persona_path=SAMPLE_PATH,
            model_base="base",
            model_tuned="tuned",
            seeds=[1],
            run_id="test-run",
            timestamp_utc="2026-05-25T00:00:00Z",
            code_commit="test-commit",
            dirty_worktree=False,
        )
        manifest["dirty_worktree"] = "false"
        with self.assertRaises(PersonaValidationError):
            validate_run_manifest(manifest)

    def test_manifest_rejects_bad_runtime_field_types(self):
        bad_values = {
            "seeds": "1",
            "decoding_params": "not an object",
            "stop_sequences": "none",
        }

        for field, value in bad_values.items():
            manifest = create_run_manifest(
                persona_path=SAMPLE_PATH,
                model_base="base",
                model_tuned="tuned",
                seeds=[1],
                run_id="test-run",
                timestamp_utc="2026-05-25T00:00:00Z",
                code_commit="test-commit",
                dirty_worktree=False,
            )
            manifest[field] = value
            with self.subTest(field=field):
                with self.assertRaises(PersonaValidationError):
                    validate_run_manifest(manifest)

    def test_result_row_rejects_missing_structured_metrics(self):
        adapter = MockAdapter()
        base_request = build_generation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="base",
            model_id="base",
            seed=1,
            adapter=adapter,
        )
        tuned_request = build_generation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="tuned",
            model_id="tuned",
            seed=1,
            adapter=adapter,
        )
        score_request = build_score_continuation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="tuned",
            model_id="tuned",
            fixed_continuation="fixed response",
            fixed_continuation_id="fixture-continuation",
        )
        result_row = build_result_row(
            run_id="test-run",
            persona_row=self.row,
            variant=self.variant,
            seed=1,
            rendered=self.rendered,
            model_pair={"base": "base", "tuned": "tuned"},
            base_request=base_request,
            base_result=adapter.generate(base_request),
            tuned_request=tuned_request,
            tuned_result=adapter.generate(tuned_request),
            score_result=adapter.score_continuation(score_request),
        )
        result_row["metrics"] = {}

        with self.assertRaises(PersonaValidationError):
            validate_result_row(result_row)

    def test_result_row_accepts_sprint2_metric_status_objects(self):
        result_row = self.build_sprint2_result_row()

        validate_result_row(result_row)
        self.assertEqual(result_row["behavior_tags"]["status"], "ok")
        self.assertEqual(result_row["metrics"]["persona_adherence"]["status"], "mock_only")
        self.assertEqual(result_row["metrics"]["behavioral_consistency_f1"]["status"], "ok")

    def test_result_row_schema_rejects_sparse_mock_only_pa_metric(self):
        result_row = self.build_sprint2_result_row()
        result_row["metrics"]["persona_adherence"] = {
            "status": "mock_only",
            "reason_code": "mock_backends_not_real_pa",
        }

        with self.assertRaises(PersonaValidationError):
            validate_schema_file(result_row, RESULT_ROW_SCHEMA_PATH)

    def test_result_row_schema_rejects_incomplete_ok_behavior_metric(self):
        result_row = self.build_sprint2_result_row()
        result_row["metrics"]["behavioral_consistency_f1"] = {
            "status": "ok",
            "reason_code": None,
        }

        with self.assertRaises(PersonaValidationError):
            validate_schema_file(result_row, RESULT_ROW_SCHEMA_PATH)

    def test_token_kl_rejects_invalid_ok_and_diagnostic_states(self):
        with self.assertRaises(PersonaValidationError):
            validate_token_kl_status(
                {
                    "status": "ok",
                    "value": None,
                    "reason_code": None,
                    "scoring_path": "none",
                    "fixed_continuation_id": None,
                    "fixed_continuation_hash": None,
                    "tokenizer_hash_match": None,
                    "vocabulary_match": None,
                    "chat_template_hash_match": None,
                    "k": None,
                    "endpoint_cap": None,
                    "diagnostic_only": False,
                }
            )

        with self.assertRaises(PersonaValidationError):
            validate_token_kl_status(
                {
                    "status": "diagnostic_only",
                    "value": None,
                    "reason_code": "endpoint_top_logprobs_cap_below_k",
                    "scoring_path": "hosted_top_logprobs",
                    "fixed_continuation_id": "fixture-continuation",
                    "fixed_continuation_hash": "sha256:" + "0" * 64,
                    "tokenizer_hash_match": True,
                    "vocabulary_match": True,
                    "chat_template_hash_match": True,
                    "k": 50,
                    "endpoint_cap": 5,
                    "diagnostic_only": False,
                }
            )

    def test_valid_ok_token_kl_requires_aligned_scoring_metadata(self):
        status = ScoreContinuationResult(
            status="ok",
            value=0.0,
            reason_code=None,
            scoring_path="local_forward",
            fixed_continuation_id="fixture-continuation",
            fixed_continuation_hash="sha256:" + "0" * 64,
            tokenizer_hash_match=True,
            vocabulary_match=True,
            chat_template_hash_match=True,
            k=50,
            endpoint_cap=None,
            diagnostic_only=False,
        ).to_token_kl_status()

        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["value"], 0.0)

    def test_endpoint_capped_scoring_can_be_represented_as_diagnostic_only(self):
        status = endpoint_capped_token_kl_status(
            fixed_continuation_id="fixture-continuation",
            fixed_continuation="fixed response",
            k=50,
            endpoint_cap=5,
        )

        self.assertEqual(status["status"], "diagnostic_only")
        self.assertEqual(status["scoring_path"], "hosted_top_logprobs")
        self.assertEqual(status["k"], 50)
        self.assertEqual(status["endpoint_cap"], 5)
        self.assertTrue(status["diagnostic_only"])

    def test_free_running_generation_outputs_cannot_create_canonical_token_kl(self):
        adapter = MockAdapter()
        base_request = build_generation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
            model_alias="base",
            model_id="base",
            seed=1,
            adapter=adapter,
        )
        tuned_request = build_generation_request(
            rendered=self.rendered,
            run_id="test-run",
            persona_id=self.row["persona_id"],
            variant_id=self.variant["variant_id"],
            variant_type=self.variant["type"],
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


class CacheKeyTests(unittest.TestCase):
    def base_payload(self):
        return build_cache_key_payload(
            prompt_hash="sha256:prompt",
            system_prompt_hash="sha256:system",
            model_id="base",
            model_revision_or_hash="base-rev",
            tokenizer_hash="sha256:tokenizer",
            chat_template_hash="sha256:chat",
            decoding_params={"temperature": 0.0, "max_tokens": 32},
            seed=1,
            evaluator_version=EVALUATOR_VERSION,
            adapter="mock",
            provider_or_endpoint="local_mock",
            serving_stack="mock",
            serving_stack_version="sprint1",
            scoring_capability="none",
        )

    def changed_payload_value(self, field, value):
        if isinstance(value, dict):
            changed = dict(value)
            changed["cache_test_field"] = "changed"
            return changed
        if isinstance(value, int):
            return value + 1
        return f"{value}-changed"

    def test_cache_key_changes_when_each_required_field_changes(self):
        payload = self.base_payload()
        original_key = cache_key_from_payload(payload)

        for field in CACHE_KEY_FIELDS:
            changed = dict(payload)
            changed[field] = self.changed_payload_value(field, payload[field])
            with self.subTest(field=field):
                self.assertNotEqual(cache_key_from_payload(changed), original_key)

    def test_cache_key_ignores_unrelated_runtime_noise(self):
        payload = self.base_payload()
        noisy_payload = dict(payload)
        noisy_payload["latency_s"] = 99.9
        noisy_payload["pid"] = 12345

        self.assertEqual(cache_key_from_payload(payload), cache_key_from_payload(noisy_payload))

    def test_cache_key_requires_contract_fields(self):
        payload = self.base_payload()
        del payload["prompt_hash"]
        with self.assertRaises(PersonaValidationError):
            cache_key_from_payload(payload)

    def test_evaluator_version_tracks_sprint2_metric_semantics(self):
        self.assertEqual(EVALUATOR_VERSION, "sprint2")


if __name__ == "__main__":
    unittest.main()
