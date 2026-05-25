import csv
import tempfile
import unittest
from pathlib import Path

import auto_research


class AutoResearchTests(unittest.TestCase):
    def test_parse_planned_generation_calls(self):
        metrics = auto_research.parse_metrics("planned_generation_calls=240\n")
        self.assertEqual(metrics["planned_generation_calls"], "240")
        self.assertEqual(metrics["primary_metric"], "not_available")

    def test_parse_named_primary_metric(self):
        metrics = auto_research.parse_metrics("primary_metric=0.8125\n")
        self.assertEqual(metrics["primary_metric_name"], "primary_metric")
        self.assertEqual(metrics["primary_metric"], "0.8125")

    def test_parse_domain_metric(self):
        metrics = auto_research.parse_metrics("BC-F1: 0.74\n")
        self.assertEqual(metrics["primary_metric_name"], "bc_f1")
        self.assertEqual(metrics["primary_metric"], "0.74")

    def test_does_not_parse_bare_token_kl_as_primary_metric(self):
        metrics = auto_research.parse_metrics("Token-KL: 0.29\n")
        self.assertEqual(metrics["primary_metric_name"], "not_available")
        self.assertEqual(metrics["primary_metric"], "not_available")

    def test_parse_unittest_status(self):
        metrics = auto_research.parse_metrics("Ran 3 tests in 0.001s\n\nOK\n")
        self.assertEqual(metrics["primary_metric_name"], "unittest_status")
        self.assertEqual(metrics["primary_metric"], "1")

    def test_reject_invalid_tag(self):
        with self.assertRaises(ValueError):
            auto_research.validate_tag("../bad")

    def test_ensure_run_dir_writes_results_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = auto_research.ensure_run_dir(root, "smoke-a")
            results_path = directory / "results.tsv"
            self.assertTrue(results_path.exists())
            with results_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)
            self.assertEqual(header, list(auto_research.RESULT_FIELDS))

    def test_shell_join_quotes_spaces(self):
        rendered = auto_research.shell_join(["python", "-c", "print('hi there')"])
        self.assertIn('"print(\'hi there\')"', rendered)


if __name__ == "__main__":
    unittest.main()
