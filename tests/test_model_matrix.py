import copy
import subprocess
import sys
import unittest
from pathlib import Path

from model_matrix import (
    ModelMatrixValidationError,
    load_model_matrix,
    validate_model_matrix,
)
from persona_eval import expected_call_count


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "model_matrix.production_open.json"


class ModelMatrixTests(unittest.TestCase):
    def load_matrix(self):
        return load_model_matrix(MATRIX_PATH)

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

    def test_real_run_ready_status_rejects_remaining_placeholders(self):
        matrix = self.load_matrix()
        matrix["template_status"] = "real_run_ready"
        matrix["real_run_ready"] = True

        with self.assertRaisesRegex(ModelMatrixValidationError, "placeholders remain"):
            validate_model_matrix(matrix)

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
