import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from persona_eval import hash_file_bytes, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


def expanded_persona_rows(count):
    sample_rows = load_jsonl(SAMPLE_PATH)
    rows = []
    for index in range(count):
        row = copy.deepcopy(sample_rows[index % len(sample_rows)])
        persona_id = f"expanded_{index:03d}"
        row["persona_id"] = persona_id
        row["source"]["source_persona_id"] = f"{persona_id}_source"
        row["seed_prompt"]["prompt_id"] = f"{persona_id}_seed"
        for variant_index, variant in enumerate(row["variants"]):
            variant["variant_id"] = f"{persona_id}_v{variant_index}"
        rows.append(row)
    return rows


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def touch_artifacts(root, stage):
    artifact_dir = root / stage
    artifact_dir.mkdir()
    manifest_path = artifact_dir / "manifest.json"
    results_path = artifact_dir / "results.jsonl"
    report_path = artifact_dir / "aggregate_report.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    results_path.write_text("", encoding="utf-8")
    report_path.write_text("{}\n", encoding="utf-8")
    return manifest_path, results_path, report_path


def write_run_evidence(root, stage, persona_count, seed_count, planned_calls):
    manifest_path, results_path, report_path = touch_artifacts(root, stage)
    evidence_path = root / f"{stage}_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence_type": stage,
                "status": "passed",
                "persona_count": persona_count,
                "variants_per_persona": 6,
                "model_count": 2,
                "seed_count": seed_count,
                "planned_generation_calls": planned_calls,
                "manifest_path": str(manifest_path),
                "results_path": str(results_path),
                "aggregate_report_path": str(report_path),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return evidence_path


def write_full_review_manifest(path, persona_rows):
    gate = {
        "status": "manual_pass",
        "reviewer_override": False,
        "evidence": ["fixture full-run review evidence"],
    }
    review_rows = [
        {
            "persona_id": row["persona_id"],
            "reviewer": "guardrail_test_reviewer",
            "review_status": "approved",
            "reviewed_at": "2026-05-25T00:00:00Z",
            "review_reason": "Fixture approval for full preflight guardrail test.",
            "low_confidence_flags": [],
            "semantic_equivalence_status": dict(gate),
            "nli_equivalence_status": dict(gate),
            "contradiction_status": dict(gate),
            "safety_review_status": dict(gate),
            "gold_label_review_status": dict(gate),
        }
        for row in persona_rows
    ]
    write_jsonl(path, review_rows)


class RunGuardrailTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "persona_eval.py"), *args],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_dev_and_full_plan_counts_cli(self):
        dev = self.run_cli(
            "plan",
            "--persona-count",
            "50",
            "--variants-per-persona",
            "6",
            "--model-count",
            "2",
            "--seed-count",
            "2",
        )
        full = self.run_cli(
            "plan",
            "--persona-count",
            "200",
            "--variants-per-persona",
            "6",
            "--model-count",
            "2",
            "--seed-count",
            "2",
        )

        self.assertEqual(dev.returncode, 0, dev.stderr)
        self.assertEqual(dev.stdout, "planned_generation_calls=1200\n")
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertEqual(full.stdout, "planned_generation_calls=4800\n")

    def test_dev_preflight_blocks_without_smoke_evidence(self):
        completed = self.run_cli(
            "preflight",
            "--stage",
            "dev",
            "--persona-count",
            "50",
            "--variants-per-persona",
            "6",
            "--model-count",
            "2",
            "--seed-count",
            "2",
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--smoke-evidence is required for dev preflight", completed.stderr)

    def test_dev_run_attempt_blocks_without_smoke_evidence(self):
        rows = expanded_persona_rows(50)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persona_path = root / "dev_personas.jsonl"
            out = root / "dev_run"
            write_jsonl(persona_path, rows)

            completed = self.run_cli(
                "run",
                "--run-stage",
                "dev",
                "--persona-path",
                str(persona_path),
                "--out",
                str(out),
                "--adapter",
                "mock",
                "--model-base",
                "base",
                "--model-tuned",
                "tuned",
                "--seeds",
                "1,2",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("--smoke-evidence is required for dev preflight", completed.stderr)
            self.assertFalse(out.exists())

    def test_dev_preflight_allows_explicit_smoke_override(self):
        completed = self.run_cli(
            "preflight",
            "--stage",
            "dev",
            "--persona-count",
            "50",
            "--variants-per-persona",
            "6",
            "--model-count",
            "2",
            "--seed-count",
            "2",
            "--allow-dev-without-smoke-evidence",
            "--approval-override-reason",
            "operator accepted missing smoke evidence for fixture test",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("planned_generation_calls=1200\n", completed.stdout)
        self.assertIn("smoke_evidence_gate=explicitly_overridden\n", completed.stdout)

    def test_full_preflight_blocks_without_promotion_manifest(self):
        rows = expanded_persona_rows(200)
        with tempfile.TemporaryDirectory() as tmp:
            persona_path = Path(tmp) / "personas.full.jsonl"
            write_jsonl(persona_path, rows)

            completed = self.run_cli(
                "preflight",
                "--stage",
                "full",
                "--persona-path",
                str(persona_path),
                "--model-count",
                "2",
                "--seed-count",
                "2",
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--promotion-manifest is required for full preflight", completed.stderr)

    def test_full_preflight_passes_with_required_artifacts_and_metadata(self):
        rows = expanded_persona_rows(200)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persona_path = root / "personas.full.jsonl"
            review_path = root / "personas.full.review.jsonl"
            promotion_path = root / "dataset_promotion.json"
            approval_path = root / "full_run_approval.json"
            write_jsonl(persona_path, rows)
            dataset_hash = hash_file_bytes(persona_path)
            write_full_review_manifest(review_path, rows)
            promotion_path.write_text(
                json.dumps(
                    {
                        "manifest_type": "dataset_promotion",
                        "status": "promoted",
                        "persona_count": 200,
                        "dataset_hash": dataset_hash,
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            approval_path.write_text(
                json.dumps(
                    {
                        "approval_type": "full_run_approval",
                        "status": "approved",
                        "approved_by": "guardrail_test_approver",
                        "approved_at": "2026-05-25T00:00:00Z",
                        "persona_count": 200,
                        "planned_generation_calls": 4800,
                        "dataset_hash": dataset_hash,
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            smoke_evidence = write_run_evidence(root, "smoke", 20, 1, 240)
            dev_evidence = write_run_evidence(root, "dev", 50, 2, 1200)

            completed = self.run_cli(
                "preflight",
                "--stage",
                "full",
                "--persona-path",
                str(persona_path),
                "--model-count",
                "2",
                "--seed-count",
                "2",
                "--promotion-manifest",
                str(promotion_path),
                "--review-manifest",
                str(review_path),
                "--smoke-evidence",
                str(smoke_evidence),
                "--dev-evidence",
                str(dev_evidence),
                "--full-run-approval",
                str(approval_path),
                "--model-base",
                "meta-llama/Meta-Llama-3-8B",
                "--model-tuned",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "--model-base-revision-or-hash",
                "base-revision-fixture",
                "--model-tuned-revision-or-hash",
                "tuned-revision-fixture",
                "--tokenizer-name",
                "meta-llama/Meta-Llama-3-8B",
                "--tokenizer-hash",
                "sha256:" + "1" * 64,
                "--chat-template-hash",
                "sha256:" + "2" * 64,
                "--serving-stack-version",
                "vllm-fixture-version",
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("preflight_stage=full\n", completed.stdout)
        self.assertIn("planned_generation_calls=4800\n", completed.stdout)
        self.assertIn("preflight_status=pass\n", completed.stdout)


if __name__ == "__main__":
    unittest.main()
