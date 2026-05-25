import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import candidate_generation
from persona_eval import PersonaValidationError


ROOT = Path(__file__).resolve().parents[1]


class CandidateGenerationTests(unittest.TestCase):
    def test_fixture_generation_is_deterministic(self):
        first = candidate_generation.generate_fixture_candidates(count=6)
        second = candidate_generation.generate_fixture_candidates(count=6)

        self.assertEqual(first, second)

    def test_fixture_rows_validate_and_include_required_metadata(self):
        rows = candidate_generation.generate_fixture_candidates(count=3)
        required = {
            "candidate_id",
            "generation_method",
            "created_at",
            "generator_version",
            "source_trace",
            "validation_status",
            "promotion_status",
            "low_confidence_flags",
        }

        for row in rows:
            candidate_generation.validate_candidate_row(row)
            self.assertTrue(required.issubset(row))
            self.assertEqual(len(row["variants"]), 6)
            self.assertEqual(
                {variant["type"] for variant in row["variants"]},
                candidate_generation.REQUIRED_VARIANT_TYPES,
            )
            self.assertEqual(row["generation_method"], "deterministic_fixture")
            self.assertEqual(row["promotion_status"]["status"], "not_promoted")

    def test_writes_candidates_and_manifest_outside_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidates" / "sample_candidates.jsonl"

            out_path, manifest_path, manifest = candidate_generation.create_fixture_candidate_file(
                out,
                count=12,
            )
            rows = candidate_generation.load_candidate_rows(out_path)
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 12)
        self.assertEqual(manifest, loaded_manifest)
        self.assertEqual(manifest["row_count"], 12)
        self.assertEqual(
            manifest["candidate_schema_version"],
            candidate_generation.CANDIDATE_SCHEMA_VERSION,
        )
        self.assertEqual(manifest["generator_version"], candidate_generation.GENERATOR_VERSION)
        self.assertTrue(manifest["output_hash"].startswith("sha256:"))
        self.assertEqual(
            manifest["source_inventory"][0]["dataset"],
            "local_synthetic_sprint6_candidates",
        )

    def test_default_fixture_count_exceeds_sprint_6_minimum(self):
        self.assertGreater(candidate_generation.DEFAULT_FIXTURE_COUNT, 200)

    def test_rejects_duplicate_candidate_ids(self):
        rows = candidate_generation.generate_fixture_candidates(count=2)
        rows[1]["candidate_id"] = rows[0]["candidate_id"]

        with self.assertRaisesRegex(PersonaValidationError, "duplicate candidate_id"):
            candidate_generation.validate_candidate_rows(rows)

    def test_rejects_missing_variants_and_invalid_variant_count(self):
        missing = candidate_generation.generate_fixture_candidates(count=1)[0]
        del missing["variants"]
        with self.assertRaisesRegex(PersonaValidationError, "variants"):
            candidate_generation.validate_candidate_row(missing)

        invalid_count = candidate_generation.generate_fixture_candidates(count=1)[0]
        invalid_count["variants"] = invalid_count["variants"][:5]
        with self.assertRaisesRegex(PersonaValidationError, "six|too short"):
            candidate_generation.validate_candidate_row(invalid_count)

    def test_rejects_missing_provenance(self):
        row = candidate_generation.generate_fixture_candidates(count=1)[0]
        del row["source_trace"]
        with self.assertRaisesRegex(PersonaValidationError, "source_trace"):
            candidate_generation.validate_candidate_row(row)

        row = candidate_generation.generate_fixture_candidates(count=1)[0]
        del row["source"]["license"]
        with self.assertRaisesRegex(PersonaValidationError, "source"):
            candidate_generation.validate_candidate_row(row)

    def test_rejects_candidate_paths_under_data(self):
        with self.assertRaisesRegex(PersonaValidationError, "outside data"):
            candidate_generation.assert_candidate_path_allowed(ROOT / "data" / "bad_candidates.jsonl")

        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "candidate_generation.py"),
                "create-fixtures",
                "--out",
                str(ROOT / "data" / "bad_candidates.jsonl"),
                "--count",
                "1",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("outside data", completed.stderr)

    def test_cli_create_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sample_candidates.jsonl"
            create = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "candidate_generation.py"),
                    "create-fixtures",
                    "--out",
                    str(out),
                    "--count",
                    "4",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(create.returncode, 0, create.stderr)
            self.assertIn("candidate_rows=4", create.stdout)

            validate = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "candidate_generation.py"),
                    "validate",
                    "--candidate-path",
                    str(out),
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(validate.returncode, 0, validate.stderr)
        self.assertIn("valid_candidate_rows=4", validate.stdout)
        self.assertIn("manifest_valid=true", validate.stdout)

    def test_manifest_validation_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sample_candidates.jsonl"
            out_path, manifest_path, _manifest = candidate_generation.create_fixture_candidate_file(
                out,
                count=2,
            )
            rows = candidate_generation.load_candidate_rows(out_path)
            bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bad_manifest["output_hash"] = "sha256:" + ("0" * 64)

            with self.assertRaisesRegex(PersonaValidationError, "output_hash"):
                candidate_generation.validate_candidate_manifest(
                    bad_manifest,
                    candidate_path=out_path,
                    rows=rows,
                )

    def test_validation_does_not_mutate_rows(self):
        rows = candidate_generation.generate_fixture_candidates(count=2)
        before = copy.deepcopy(rows)

        candidate_generation.validate_candidate_rows(rows)

        self.assertEqual(rows, before)


if __name__ == "__main__":
    unittest.main()
