import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from persona_eval import validate_result_row, validate_run_manifest


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class RunCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "persona_eval.py"), *args],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_run_dry_run_prints_sample_count_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dry-run"
            completed = self.run_cli(
                "run",
                "--persona-path",
                str(SAMPLE_PATH),
                "--out",
                str(out),
                "--adapter",
                "mock",
                "--model-base",
                "base",
                "--model-tuned",
                "tuned",
                "--seeds",
                "1",
                "--dry-run",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "planned_generation_calls=120\n")
            self.assertFalse(out.exists())

    def test_run_dry_run_supports_explicit_20_persona_arithmetic(self):
        completed = self.run_cli(
            "run",
            "--persona-count",
            "20",
            "--variants-per-persona",
            "6",
            "--adapter",
            "mock",
            "--model-base",
            "base",
            "--model-tuned",
            "tuned",
            "--seeds",
            "1",
            "--dry-run",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "planned_generation_calls=240\n")

    def test_run_dry_run_parses_comma_separated_seeds(self):
        completed = self.run_cli(
            "run",
            "--persona-path",
            str(SAMPLE_PATH),
            "--adapter",
            "mock",
            "--model-base",
            "base",
            "--model-tuned",
            "tuned",
            "--seeds",
            "1,2",
            "--limit-personas",
            "1",
            "--dry-run",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "planned_generation_calls=24\n")

    def test_mock_run_writes_valid_manifest_and_result_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mock-run"
            completed = self.run_cli(
                "run",
                "--persona-path",
                str(SAMPLE_PATH),
                "--out",
                str(out),
                "--adapter",
                "mock",
                "--model-base",
                "base",
                "--model-tuned",
                "tuned",
                "--seeds",
                "1",
                "--run-id",
                "test-mock-run",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("planned_generation_calls=120\n", completed.stdout)
            self.assertIn("written_result_rows=60\n", completed.stdout)

            manifest_path = out / "manifest.json"
            results_path = out / "results.jsonl"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(results_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_run_manifest(manifest)
            self.assertEqual(manifest["run_id"], "test-mock-run")
            self.assertEqual(manifest["adapter"], "mock")
            self.assertEqual(manifest["provider_or_endpoint"], "local_mock")
            self.assertEqual(manifest["scoring_capability"], "none")
            self.assertEqual(manifest["seeds"], [1])

            result_rows = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(result_rows), 10 * 6 * 1)
            for row in result_rows:
                validate_result_row(row)
                self.assertEqual(row["metrics"]["token_kl"]["status"], "not_applicable")
                self.assertEqual(
                    row["metrics"]["token_kl"]["reason_code"],
                    "aligned_scoring_unavailable",
                )
                self.assertIn("raw_request", row["base"])
                self.assertIn("raw_response", row["base"])
                self.assertIn("raw_request", row["tuned"])
                self.assertIn("raw_response", row["tuned"])
                self.assertEqual(row["base"]["raw_request"]["adapter"], "mock")
                self.assertEqual(row["tuned"]["raw_request"]["provider_or_endpoint"], "local_mock")
                self.assertIn("tokenizer_hash", row["base"]["raw_request"])
                self.assertIn("chat_template_hash", row["base"]["raw_request"])

    def test_disable_token_kl_uses_disabled_by_user_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "disabled-token-kl"
            completed = self.run_cli(
                "run",
                "--persona-path",
                str(SAMPLE_PATH),
                "--out",
                str(out),
                "--adapter",
                "mock",
                "--model-base",
                "base",
                "--model-tuned",
                "tuned",
                "--seeds",
                "1",
                "--limit-personas",
                "1",
                "--disable-token-kl",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = [
                json.loads(line)
                for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(row["metrics"]["token_kl"]["status"], "not_applicable")
                self.assertEqual(row["metrics"]["token_kl"]["reason_code"], "disabled_by_user")
                self.assertEqual(row["metrics"]["token_kl"]["scoring_path"], "none")

    def test_vllm_adapter_requires_explicit_base_url(self):
        completed = self.run_cli(
            "run",
            "--persona-path",
            str(SAMPLE_PATH),
            "--adapter",
            "vllm",
            "--model-base",
            "base",
            "--model-tuned",
            "tuned",
            "--seeds",
            "1",
            "--dry-run",
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--base-url is required", completed.stderr)


if __name__ == "__main__":
    unittest.main()
