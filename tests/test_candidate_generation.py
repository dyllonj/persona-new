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
        self.assertTrue(manifest["content_hashes"]["output_hash"].startswith("sha256:"))
        self.assertEqual(len(manifest["candidate_hashes"]), 12)
        self.assertIn("dedupe_diversity_report", manifest)
        self.assertIn("blocker_reasons", manifest)
        self.assertIn("rejection_reasons", manifest)
        self.assertEqual(
            manifest["source_inventory"][0]["dataset"],
            "local_synthetic_sprint6_candidates",
        )

    def test_default_fixture_count_exceeds_sprint_6_minimum(self):
        self.assertGreater(candidate_generation.DEFAULT_FIXTURE_COUNT, 200)

    def test_default_fixture_pool_is_diverse_enough_to_select_200(self):
        rows = candidate_generation.generate_fixture_candidates()

        report = candidate_generation.candidate_dedupe_diversity_report(rows)
        selected, selection_report = candidate_generation.select_diverse_candidates(rows)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["row_count"], 300)
        self.assertEqual(report["exact_duplicate_group_count"], 0)
        self.assertEqual(report["near_duplicate_pair_count"], 0)
        self.assertEqual(report["blocker_reasons"], [])
        self.assertEqual(report["rejection_reasons"], [])
        self.assertEqual(len(selected), 200)
        self.assertEqual(selection_report["status"], "pass")
        self.assertEqual(selection_report["selected_candidate_count"], 200)
        self.assertEqual(selection_report["rejection_reasons"], [])

    def test_diverse_selector_rejects_duplicate_rows_before_capacity(self):
        rows = candidate_generation.generate_fixture_candidates(count=205)
        duplicate = copy.deepcopy(rows[0])
        duplicate["candidate_id"] = "candidate_duplicate"
        duplicate["persona_id"] = "candidate_duplicate"
        duplicate["source_trace"]["trace_id"] = "trace_candidate_duplicate"
        duplicate["source_trace"]["source_id"] = "duplicate_source"
        duplicate["source"]["source_persona_id"] = "duplicate_source"
        duplicate["seed_prompt"]["prompt_id"] = "candidate_duplicate_seed"
        for index, variant in enumerate(duplicate["variants"]):
            variant["variant_id"] = f"candidate_duplicate_v{index}"

        selected, selection_report = candidate_generation.select_diverse_candidates(
            [rows[0], duplicate, *rows[1:]],
            target_count=200,
        )

        self.assertEqual(len(selected), 200)
        self.assertEqual(selection_report["status"], "pass")
        self.assertEqual(selection_report["rejection_counts"], {"exact_duplicate_candidate_text": 1})
        self.assertEqual(
            selection_report["rejection_reasons"][0]["matched_persona_id"],
            rows[0]["persona_id"],
        )

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
            selected = Path(tmp) / "selected_candidates.jsonl"
            create = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "candidate_generation.py"),
                    "create-fixtures",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(create.returncode, 0, create.stderr)
            self.assertIn("candidate_rows=300", create.stdout)

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
            select = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "candidate_generation.py"),
                    "select",
                    "--candidate-path",
                    str(out),
                    "--out",
                    str(selected),
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(validate.returncode, 0, validate.stderr)
        self.assertIn("valid_candidate_rows=300", validate.stdout)
        self.assertIn("manifest_valid=true", validate.stdout)
        self.assertEqual(select.returncode, 0, select.stderr)
        self.assertIn("selection_status=pass", select.stdout)
        self.assertIn("selected_candidate_rows=200", select.stdout)

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
