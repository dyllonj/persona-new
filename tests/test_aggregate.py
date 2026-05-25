import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import aggregate
from persona_eval import (
    PersonaValidationError,
    ScoreContinuationResult,
    endpoint_capped_token_kl_status,
    load_jsonl,
    score_persona_adherence_mock,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class AggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.tmp_root = Path(cls.tmpdir.name)
        cls.run_dir = cls.tmp_root / "mock-run"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "persona_eval.py"),
                "run",
                "--persona-path",
                str(SAMPLE_PATH),
                "--out",
                str(cls.run_dir),
                "--adapter",
                "mock",
                "--model-base",
                "base",
                "--model-tuned",
                "tuned",
                "--seeds",
                "1",
                "--run-id",
                "aggregate-unit-run",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr)
        cls.manifest_path = cls.run_dir / "manifest.json"
        cls.results_path = cls.run_dir / "results.jsonl"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.rows = aggregate.load_result_rows(cls.results_path)
        cls.sample_rows_by_id = {
            row["persona_id"]: row
            for row in load_jsonl(SAMPLE_PATH)
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def write_temp_run(self, rows):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        manifest_path = root / "manifest.json"
        results_path = root / "results.jsonl"
        manifest_path.write_text(json.dumps(self.manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with results_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self.addCleanup(tmp.cleanup)
        return manifest_path, results_path

    def copy_rows(self):
        return json.loads(json.dumps(self.rows))

    def set_bc(self, row, value):
        row["metrics"]["behavioral_consistency_f1"] = {
            "status": "ok",
            "reason_code": None,
            "stance_exact": value,
            "primary_action_exact": value,
            "secondary_modifiers_precision": value,
            "secondary_modifiers_recall": value,
            "secondary_modifiers_f1": value,
            "combined_score": value,
        }

    def test_aggregate_reads_validates_manifest_and_results(self):
        report = aggregate.build_report(
            manifest_path=self.manifest_path,
            results_path=self.results_path,
        )

        self.assertEqual(report["run_id"], "aggregate-unit-run")
        self.assertEqual(report["counts"]["persona_count"], 10)
        self.assertEqual(report["counts"]["row_count"], 60)
        self.assertEqual(report["counts"]["inference_unit"], "persona_id")
        self.assertEqual(report["variant_type_breakdowns"]["canonical"]["row_count"], 10)
        self.assertTrue(report["source_manifest_hash"].startswith("sha256:"))
        self.assertTrue(report["source_results_hash"].startswith("sha256:"))

        bad_rows = self.copy_rows()[:1]
        bad_rows[0]["metrics"] = {}
        manifest_path, results_path = self.write_temp_run(bad_rows)
        with self.assertRaises(PersonaValidationError):
            aggregate.build_report(manifest_path=manifest_path, results_path=results_path)

    def test_persona_level_grouping_not_row_level_inference(self):
        rows = self.copy_rows()
        first_persona = rows[0]["persona_id"]
        second_persona = next(row["persona_id"] for row in rows if row["persona_id"] != first_persona)
        selected = [
            row
            for row in rows
            if row["persona_id"] == first_persona
        ][:6]
        selected += [
            row
            for row in rows
            if row["persona_id"] == second_persona
        ][:1]
        for row in selected:
            self.set_bc(row, 1.0 if row["persona_id"] == first_persona else 0.0)

        manifest_path, results_path = self.write_temp_run(selected)
        report = aggregate.build_report(manifest_path=manifest_path, results_path=results_path)
        combined = report["metric_summaries"]["behavioral_consistency_f1"]["fields"]["combined_score"]

        self.assertEqual(report["counts"]["row_count"], 7)
        self.assertEqual(combined["available_row_count"], 7)
        self.assertEqual(combined["inference_unit_count"], 2)
        self.assertEqual(combined["inference_unit"], "persona_id")
        self.assertEqual(combined["mean"], 0.5)

    def test_unavailable_and_diagnostic_token_kl_are_counts_not_zero(self):
        rows = []
        seen = set()
        for row in self.copy_rows():
            if row["persona_id"] in seen:
                continue
            rows.append(row)
            seen.add(row["persona_id"])
            if len(rows) == 3:
                break
        rows[0]["metrics"]["token_kl"] = ScoreContinuationResult(
            status="ok",
            value=0.4,
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
        rows[2]["metrics"]["token_kl"] = endpoint_capped_token_kl_status(
            fixed_continuation_id="fixture-continuation",
            fixed_continuation="fixed response",
            k=50,
            endpoint_cap=5,
        )

        manifest_path, results_path = self.write_temp_run(rows)
        report = aggregate.build_report(manifest_path=manifest_path, results_path=results_path)
        token_kl = report["metric_summaries"]["token_kl"]

        self.assertEqual(token_kl["ok_row_count"], 1)
        self.assertEqual(token_kl["canonical_ok"]["mean"], 0.4)
        self.assertEqual(token_kl["canonical_ok"]["inference_unit_count"], 1)
        self.assertEqual(token_kl["not_applicable_count"], 1)
        self.assertEqual(token_kl["diagnostic_only_count"], 1)
        self.assertEqual(token_kl["status_counts"]["diagnostic_only"], 1)

    def test_aggregate_rejects_canonical_token_kl_conflicting_with_matrix_applicability(self):
        rows = self.copy_rows()[:1]
        rows[0]["model_pair"]["token_kl_applicability"] = "not_applicable"
        rows[0]["metrics"]["token_kl"] = ScoreContinuationResult(
            status="ok",
            value=0.4,
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
        manifest_path, results_path = self.write_temp_run(rows)

        with self.assertRaisesRegex(PersonaValidationError, "token_kl.status=ok conflicts"):
            aggregate.build_report(manifest_path=manifest_path, results_path=results_path)

    def test_bc_f1_field_scores_aggregate_correctly(self):
        rows = self.copy_rows()[:2]
        self.set_bc(rows[0], 0.25)
        self.set_bc(rows[1], 0.75)
        manifest_path, results_path = self.write_temp_run(rows)
        report = aggregate.build_report(manifest_path=manifest_path, results_path=results_path)
        fields = report["metric_summaries"]["behavioral_consistency_f1"]["fields"]

        self.assertEqual(fields["stance_exact"]["mean"], 0.5)
        self.assertEqual(fields["primary_action_exact"]["mean"], 0.5)
        self.assertEqual(fields["secondary_modifiers_f1"]["mean"], 0.5)
        self.assertEqual(fields["combined_score"]["mean"], 0.5)

    def test_pa_mock_only_is_labeled_as_plumbing_not_real_pa(self):
        rows = self.copy_rows()[:1]
        persona_row = self.sample_rows_by_id[rows[0]["persona_id"]]
        rows[0]["metrics"]["persona_adherence"] = score_persona_adherence_mock(
            persona_row,
            "I ask for evidence and avoid overclaiming certainty.",
        )
        manifest_path, results_path = self.write_temp_run(rows)
        report = aggregate.build_report(manifest_path=manifest_path, results_path=results_path)
        pa = report["metric_summaries"]["persona_adherence"]

        self.assertEqual(pa["real_persona_adherence"]["status"], "not_applicable")
        self.assertEqual(pa["mock_plumbing"]["status"], "mock_only")
        self.assertEqual(pa["mock_plumbing"]["label"], "mock_only_plumbing_not_real_persona_adherence")
        self.assertIn("not reported as real Persona Adherence", pa["warning"])

    def test_statistics_helpers_operate_over_persona_level_values(self):
        stats = aggregate.paired_delta_statistics(
            {
                "persona_a": [(1.0, 2.0), (1.0, 2.0)],
                "persona_b": [(2.0, 1.0)],
            }
        )

        self.assertEqual(stats["status"], "ok")
        self.assertEqual(stats["inference_unit_count"], 2)
        self.assertEqual(stats["persona_deltas"], {"persona_a": 1.0, "persona_b": -1.0})
        self.assertEqual(stats["ci95"]["inference_unit_count"], 2)
        self.assertIn(stats["effect_size"]["status"], {"ok", "not_applicable"})
        self.assertIn(stats["p_value"]["status"], {"ok", "not_applicable"})

    def test_readiness_gate_remains_blocked_when_full_dataset_validators_are_missing(self):
        readiness = aggregate.full_dataset_readiness(
            persona_path=SAMPLE_PATH,
            review_manifest_path=ROOT / "reviews" / "personas.sample.review.jsonl",
        )

        self.assertEqual(readiness["status"], "blocked")
        self.assertEqual(readiness["checks"]["pii_and_real_person_filters_exist"]["status"], "pass")
        self.assertEqual(readiness["checks"]["restricted_role_filters_exist"]["status"], "pass")
        self.assertIn("semantic_equivalence_validation_exists", readiness["blocking_checks"])
        self.assertIn("nli_contradiction_equivalence_checks_exist", readiness["blocking_checks"])
        self.assertIn("human_review_coverage_sufficient", readiness["blocking_checks"])
        self.assertEqual(readiness["checks"]["source_license_checks_pass"]["status"], "pass")

    def test_report_readiness_is_bound_to_manifest_persona_hash(self):
        bad_manifest = copy.deepcopy(self.manifest)
        bad_manifest["persona_path"] = str(ROOT / "README.md")
        manifest_path, results_path = self.write_temp_run(self.copy_rows()[:1])
        manifest_path.write_text(json.dumps(bad_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")

        with self.assertRaises(PersonaValidationError):
            aggregate.build_report(manifest_path=manifest_path, results_path=results_path)

    def test_report_contains_statistical_metadata_and_availability(self):
        report = aggregate.build_report(
            manifest_path=self.manifest_path,
            results_path=self.results_path,
        )
        output_delta = report["paired_output_deltas"]["numeric_deltas"]["total_tokens"]

        aggregate.validate_aggregate_report(report)
        self.assertIn("effect_size", output_delta)
        self.assertIn("p_value", output_delta)
        self.assertIn("reason_code", output_delta["effect_size"])
        self.assertIn("multiple_comparison_note", report["statistical_method_notes"])
        self.assertIn("metric_availability", report["variant_type_breakdowns"]["canonical"])
        self.assertIn("availability_summaries", report)

    def test_cli_writes_report_and_chart_data(self):
        out_dir = self.tmp_root / "aggregate-cli"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "aggregate.py"),
                "--manifest",
                str(self.manifest_path),
                "--results",
                str(self.results_path),
                "--out",
                str(out_dir),
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("full_dataset_readiness=blocked", completed.stdout)
        self.assertTrue((out_dir / "aggregate_report.json").exists())
        self.assertTrue((out_dir / "chart_data" / "metric_availability.csv").exists())
        self.assertTrue((out_dir / "chart_data" / "variant_type_breakdown.csv").exists())

    def test_readme_examples_match_actual_cli_commands(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn(
            "python3 persona_eval.py run --persona-path data/personas.sample.jsonl --out results/sprint5_mock --adapter mock --model-base base --model-tuned tuned --seeds 1 --run-id sprint5_mock",
            readme,
        )
        self.assertIn(
            "python3 aggregate.py --manifest results/sprint5_mock/manifest.json --results results/sprint5_mock/results.jsonl --out reports/sprint5_mock",
            readme,
        )
        self.assertIn(
            "python3 dataset_readiness.py --persona-path data/personas.sample.jsonl --review-manifest reviews/personas.sample.review.jsonl",
            readme,
        )
        self.assertIn(
            "python3 persona_eval.py plan --persona-count 20 --variants-per-persona 6 --model-count 2 --seed-count 1",
            readme,
        )


if __name__ == "__main__":
    unittest.main()
