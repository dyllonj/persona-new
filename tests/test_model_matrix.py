import copy
import subprocess
import sys
import unittest
from pathlib import Path

from model_matrix import (
    ModelMatrixValidationError,
    load_model_matrix,
    resolve_model_matrix_entry,
    validate_model_matrix,
)
from persona_eval import expected_call_count


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "model_matrix.production_open.json"


class ModelMatrixTests(unittest.TestCase):
    def load_matrix(self):
        return load_model_matrix(MATRIX_PATH)

    def make_runtime_ready_values(self, matrix):
        matrix["template_status"] = "real_run_ready"
        matrix["real_run_ready"] = True
        for collection in ("drift_pairs",):
            for pair in matrix[collection]:
                for model_key in ("base_model", "instruct_model"):
                    model = pair[model_key]
                    model["provider_or_endpoint"] = "http://localhost:8000/v1"
                    model["required_revision_or_hash"] = f"{model['model_id']}-revision"
                    model["license_review_status"] = "approved"
                    model["license_review_evidence"] = ["fixture license review approval"]
        for model in matrix["standalone_instruct_models"]:
            model["provider_or_endpoint"] = "http://localhost:8000/v1"
            model["required_revision_or_hash"] = f"{model['model_id']}-revision"
            model["license_review_status"] = "approved"
            model["license_review_evidence"] = ["fixture license review approval"]
        for comparison in matrix["cross_family_comparisons"]:
            for model_key in ("left_model", "right_model"):
                model = comparison[model_key]
                model["provider_or_endpoint"] = "http://localhost:8000/v1"
                model["required_revision_or_hash"] = f"{model['model_id']}-revision"
                model["license_review_status"] = "approved"
                model["license_review_evidence"] = ["fixture license review approval"]

    def test_valid_qwen_same_family_drift_pair_template_passes(self):
        matrix = self.load_matrix()
        matrix["drift_pairs"] = [
            pair for pair in matrix["drift_pairs"] if pair["pair_id"] == "qwen2_5_7b_base_vs_instruct"
        ]
        matrix["cross_family_comparisons"] = []
        matrix["standalone_instruct_models"] = []

        report = validate_model_matrix(matrix)

        self.assertEqual(report["status"], "template_valid")
        self.assertFalse(report["real_run_ready"])
        self.assertEqual(report["drift_pair_count"], 1)

    def test_invalid_llama_vs_qwen_canonical_token_kl_config_fails(self):
        matrix = self.load_matrix()
        comparison = matrix["cross_family_comparisons"][0]
        self.assertEqual(
            comparison["comparison_id"],
            "llama3_1_8b_instruct_vs_qwen2_5_7b_instruct",
        )
        comparison["token_kl_applicability"] = "canonical_possible"

        with self.assertRaisesRegex(ModelMatrixValidationError, "cross-family comparison cannot declare canonical"):
            validate_model_matrix(matrix)

    def test_standalone_instruct_with_not_applicable_token_kl_passes(self):
        matrix = self.load_matrix()
        for model in matrix["standalone_instruct_models"]:
            self.assertEqual(model["role"], "standalone_instruct")
            self.assertEqual(model["token_kl_applicability"], "not_applicable")

        report = validate_model_matrix(matrix)

        self.assertEqual(report["standalone_instruct_model_count"], 5)

    def test_placeholder_revisions_block_real_run_readiness_but_not_template_validation(self):
        matrix = self.load_matrix()

        template_report = validate_model_matrix(matrix)

        self.assertEqual(template_report["status"], "template_valid")
        self.assertGreater(template_report["placeholder_count"], 0)
        with self.assertRaisesRegex(ModelMatrixValidationError, "not real-run ready"):
            validate_model_matrix(matrix, require_real_run_ready=True)

    def test_real_run_readiness_requires_top_level_ready_declarations(self):
        matrix = self.load_matrix()
        self.make_runtime_ready_values(matrix)
        matrix["template_status"] = "template_with_placeholders"
        matrix["real_run_ready"] = False

        with self.assertRaisesRegex(ModelMatrixValidationError, "template_status must be real_run_ready"):
            validate_model_matrix(matrix, require_real_run_ready=True)

    def test_license_review_required_blocks_real_run_readiness_without_evidence(self):
        matrix = self.load_matrix()
        self.make_runtime_ready_values(matrix)
        del matrix["drift_pairs"][0]["base_model"]["license_review_status"]
        del matrix["drift_pairs"][0]["base_model"]["license_review_evidence"]

        with self.assertRaisesRegex(ModelMatrixValidationError, "license review evidence"):
            validate_model_matrix(matrix, require_real_run_ready=True)

    def test_real_run_ready_requires_license_evidence_and_no_placeholders(self):
        matrix = self.load_matrix()
        self.make_runtime_ready_values(matrix)

        report = validate_model_matrix(matrix, require_real_run_ready=True)

        self.assertEqual(report["status"], "real_run_ready")
        self.assertTrue(report["real_run_ready"])
        self.assertEqual(report["placeholder_count"], 0)
        self.assertEqual(report["license_review_blocker_count"], 0)

    def test_real_run_ready_status_rejects_remaining_placeholders(self):
        matrix = self.load_matrix()
        matrix["template_status"] = "real_run_ready"
        matrix["real_run_ready"] = True

        with self.assertRaisesRegex(ModelMatrixValidationError, "placeholders remain"):
            validate_model_matrix(matrix)

    def test_resolves_cross_family_entry_policy(self):
        matrix = self.load_matrix()

        policy = resolve_model_matrix_entry(matrix, "llama3_1_8b_instruct_vs_qwen2_5_7b_instruct")

        self.assertEqual(policy["entry_type"], "cross_family_comparison")
        self.assertEqual(policy["token_kl_applicability"], "not_applicable")

    def test_production_matrix_does_not_alter_plan_counts(self):
        validate_model_matrix(self.load_matrix())

        self.assertEqual(expected_call_count(20, 6, 2, 1), 240)
        self.assertEqual(expected_call_count(50, 6, 2, 2), 1200)
        self.assertEqual(expected_call_count(200, 6, 2, 2), 4800)

    def test_cli_validate_model_matrix_reports_template_blocked(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "persona_eval.py"),
                "validate-model-matrix",
                "--matrix-path",
                str(MATRIX_PATH),
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("model_matrix_status=template_valid\n", completed.stdout)
        self.assertIn("real_run_readiness=blocked\n", completed.stdout)
        self.assertIn("drift_pairs=3\n", completed.stdout)

    def test_canonical_possible_requires_alignment_contract_fields(self):
        matrix = copy.deepcopy(self.load_matrix())
        del matrix["drift_pairs"][0]["alignment_contract"]["fixed_continuation_scoring_required"]

        with self.assertRaisesRegex(ModelMatrixValidationError, "fixed_continuation_scoring_required"):
            validate_model_matrix(matrix)


if __name__ == "__main__":
    unittest.main()
