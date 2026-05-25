import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "persona_eval.py"), *args],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_validate_succeeds_on_sample(self):
        completed = self.run_cli("validate", "--persona-path", str(SAMPLE_PATH))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("valid_persona_rows=10", completed.stdout)

    def test_plan_with_sample_path_prints_correct_count(self):
        completed = self.run_cli(
            "plan",
            "--persona-path",
            str(SAMPLE_PATH),
            "--model-base",
            "base",
            "--model-tuned",
            "tuned",
            "--seeds",
            "1",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "planned_generation_calls=120\n")

    def test_plan_rejects_partial_model_pair(self):
        completed = self.run_cli(
            "plan",
            "--persona-path",
            str(SAMPLE_PATH),
            "--model-base",
            "base",
            "--seeds",
            "1",
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--model-base and --model-tuned", completed.stderr)

    def test_plan_with_explicit_counts_prints_correct_count(self):
        completed = self.run_cli(
            "plan",
            "--persona-count",
            "20",
            "--variants-per-persona",
            "6",
            "--model-count",
            "2",
            "--seed-count",
            "1",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "planned_generation_calls=240\n")


if __name__ == "__main__":
    unittest.main()
