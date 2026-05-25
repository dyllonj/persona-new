import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import dataset_promotion
from persona_eval import (
    PersonaValidationError,
    hash_file_bytes,
    load_jsonl,
    validate_dataset_promotion_manifest,
    validate_persona_row,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"
CURRENT_BAD_POOL_PATH = ROOT / "candidates" / "sample_candidates.jsonl"


def _unique_words(index, count):
    words = []
    counter = 0
    while len(words) < count:
        digest = hashlib.sha256(f"promotion-{index}-{counter}".encode("utf-8")).hexdigest()
        words.extend(f"w{digest[offset:offset + 8]}" for offset in range(0, 56, 8))
        counter += 1
    return words[:count]


def _promotion_row(sample_rows, index):
    row = copy.deepcopy(sample_rows[index % len(sample_rows)])
    row["persona_id"] = f"promo_{index:03d}"
    row["source"]["source_persona_id"] = f"source_{index:03d}"
    row["seed_prompt"]["prompt_id"] = f"prompt_{index:03d}"

    words = _unique_words(index, 30)
    row["persona_spec"]["domain"] = " ".join(words[0:3])
    row["persona_spec"]["facts"] = [
        " ".join(words[3:7]),
        " ".join(words[7:11]),
        " ".join(words[11:15]),
    ]
    row["persona_spec"]["traits"]["tone"] = [" ".join(words[15:17])]
    row["persona_spec"]["traits"]["style"] = [" ".join(words[17:19])]
    row["persona_spec"]["traits"]["values"] = [" ".join(words[19:21])]
    row["persona_spec"]["forbidden_behaviors"] = [" ".join(words[21:23])]
    row["seed_prompt"]["text"] = " ".join(words[23:27])
    row["seed_prompt"]["intent"] = words[27]
    row["seed_prompt"]["topic"] = words[28]

    for variant_index, variant in enumerate(row["variants"]):
        variant["variant_id"] = f"{row['persona_id']}_{variant['type']}"
        start = min(variant_index * 4, len(words) - 4)
        variant["text"] = " ".join(words[start:start + 4])
        if "validation" in variant:
            variant["validation"] = {"equivalence_status": "manual_fixture_check"}

    validate_persona_row(row)
    return row


def _gate(status="manual_pass"):
    return {
        "status": status,
        "reviewer_override": False,
        "evidence": ["Sprint 7 promotion fixture evidence."],
    }


def _review_for(persona_id, *, status="approved", concerns=None):
    return {
        "persona_id": persona_id,
        "reviewer": "sprint7_fixture_reviewer",
        "review_status": status,
        "reviewed_at": "2026-05-25T00:00:00Z",
        "review_reason": "Sprint 7 promotion fixture review.",
        "unresolved_concerns": [] if concerns is None else concerns,
        "low_confidence_flags": [],
        "semantic_equivalence_status": _gate(),
        "nli_equivalence_status": _gate(),
        "contradiction_status": _gate(),
        "safety_review_status": _gate("passed"),
        "gold_label_review_status": _gate(),
    }


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


class DatasetPromotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_rows = load_jsonl(SAMPLE_PATH)
        cls.valid_rows_200 = [_promotion_row(cls.sample_rows, index) for index in range(200)]
        cls.valid_reviews_200 = [_review_for(row["persona_id"]) for row in cls.valid_rows_200]

    def evaluate(self, candidate_rows, review_rows, *, dry_run=True, out_name="personas.full.jsonl"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        candidate_path = root / "candidates.jsonl"
        review_path = root / "reviews.jsonl"
        out_path = root / out_name
        _write_jsonl(candidate_path, candidate_rows)
        _write_jsonl(review_path, review_rows)
        return dataset_promotion.evaluate_promotion(
            candidate_path=candidate_path,
            review_path=review_path,
            out_path=out_path,
            dry_run=dry_run,
        ), out_path

    def test_dry_run_requires_exactly_200_valid_rows(self):
        (report, _), out_path = self.evaluate([self.valid_rows_200[0]], [_review_for("promo_000")])

        self.assertEqual(report["status"], "blocked")
        self.assertIn("promotion_requires_exactly_200_valid_rows", report["rejection_counts"])
        self.assertFalse(out_path.exists())

    def test_missing_review_evidence_blocks_promotion(self):
        (report, _), _ = self.evaluate([self.valid_rows_200[0]], [])

        self.assertEqual(report["status"], "blocked")
        self.assertIn("review_evidence_missing", report["rejection_counts"])
        self.assertIn("minimum_review_coverage_not_met", report["rejection_counts"])

    def test_missing_review_manifest_reports_structured_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.jsonl"
            review_path = root / "missing.review.jsonl"
            _write_jsonl(candidate_path, [self.valid_rows_200[0]])

            report, _ = dataset_promotion.evaluate_promotion(
                candidate_path=candidate_path,
                review_path=review_path,
                out_path=root / "personas.full.jsonl",
                dry_run=True,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("review_manifest_missing", report["rejection_counts"])
        self.assertIn("review_evidence_missing", report["rejection_counts"])

    def test_low_confidence_rows_require_review(self):
        candidate = copy.deepcopy(self.valid_rows_200[0])
        candidate["low_confidence_flags"] = ["semantic_equivalence_manual_required"]

        (report, _), _ = self.evaluate([candidate], [])

        self.assertEqual(report["status"], "blocked")
        self.assertIn("low_confidence_review_required", report["rejection_counts"])

    def test_restricted_categories_block_promotion(self):
        candidate = copy.deepcopy(self.valid_rows_200[0])
        candidate["seed_prompt"]["text"] = "Act as a doctor and diagnose the patient, then prescribe treatment."

        (report, _), _ = self.evaluate([candidate], [_review_for(candidate["persona_id"])])

        self.assertEqual(report["status"], "blocked")
        self.assertIn("restricted_category_blocked", report["rejection_counts"])

    def test_duplicate_candidates_block_promotion(self):
        first = copy.deepcopy(self.valid_rows_200[0])
        second = copy.deepcopy(first)
        second["persona_id"] = "promo_dup"
        second["source"]["source_persona_id"] = "source_dup"
        for variant in second["variants"]:
            variant["variant_id"] = f"promo_dup_{variant['type']}"

        reviews = [_review_for(first["persona_id"]), _review_for(second["persona_id"])]
        (report, _), _ = self.evaluate([first, second], reviews)

        self.assertEqual(report["status"], "blocked")
        self.assertIn("duplicate_candidate_blocked", report["rejection_counts"])

    def test_unresolved_review_concerns_block_promotion(self):
        review = _review_for("promo_000", concerns=["semantic equivalence evidence is incomplete"])

        (report, _), _ = self.evaluate([self.valid_rows_200[0]], [review])

        self.assertEqual(report["status"], "blocked")
        self.assertIn("unresolved_review_concerns", report["rejection_counts"])

    def test_dry_run_ready_never_writes_full_dataset(self):
        (report, promoted_rows), out_path = self.evaluate(self.valid_rows_200, self.valid_reviews_200)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["rejection_counts"], {})
        self.assertEqual(len(promoted_rows), 200)
        self.assertFalse(out_path.exists())
        self.assertFalse(report["write_permitted"])

    def test_current_bad_pool_dry_run_blocks_with_exact_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_review_path = root / "personas.full.review.jsonl"
            out_path = root / "personas.full.jsonl"

            report, _ = dataset_promotion.evaluate_promotion(
                candidate_path=CURRENT_BAD_POOL_PATH,
                review_path=missing_review_path,
                out_path=out_path,
                dry_run=True,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(
            report["rejection_counts"],
            {
                "duplicate_candidate_blocked": 4350,
                "minimum_review_coverage_not_met": 1,
                "promotion_requires_exactly_200_valid_rows": 1,
                "review_evidence_missing": 300,
                "review_manifest_missing": 1,
            },
        )
        self.assertEqual(report["filter_summary"]["near_duplicate_pairs"], 4350)
        self.assertEqual(report["candidate_row_count"], 300)
        self.assertEqual(report["valid_candidate_count"], 300)
        reasons = {row["reason_code"]: row for row in report["rejection_reasons"]}
        self.assertEqual(
            reasons["duplicate_candidate_blocked"]["detail"],
            (
                "Deterministic near-duplicate scan found 4,350 candidate pairs at or above threshold 0.92; "
                "first observed pair candidate_0001/candidate_0011 similarity=0.990."
            ),
        )
        self.assertFalse(out_path.exists())

    def test_partial_semantic_and_nli_review_gates_block_promotion(self):
        reviews = copy.deepcopy(self.valid_reviews_200)
        incomplete_gate = {
            "status": "manual_required",
            "reviewer_override": False,
            "evidence": [],
        }
        for review in reviews[20:]:
            review["semantic_equivalence_status"] = copy.deepcopy(incomplete_gate)
            review["nli_equivalence_status"] = copy.deepcopy(incomplete_gate)
            review["contradiction_status"] = copy.deepcopy(incomplete_gate)

        (report, _), _ = self.evaluate(self.valid_rows_200, reviews)

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["approved_candidate_count"], 20)
        self.assertEqual(report["rejection_counts"]["semantic_equivalence_evidence_missing"], 180)
        self.assertEqual(report["rejection_counts"]["nli_equivalence_evidence_missing"], 180)
        self.assertEqual(report["rejection_counts"]["contradiction_evidence_missing"], 180)

    def test_non_dry_happy_path_writes_downstream_compatible_manifest_in_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.jsonl"
            review_path = root / "reviews.jsonl"
            out_path = root / "personas.full.jsonl"
            manifest_path = root / "dataset_promotion_manifest.json"
            _write_jsonl(candidate_path, self.valid_rows_200)
            _write_jsonl(review_path, self.valid_reviews_200)

            report, promoted_rows = dataset_promotion.evaluate_promotion(
                candidate_path=candidate_path,
                review_path=review_path,
                out_path=out_path,
                dry_run=False,
            )
            dataset_promotion.write_promotion_outputs(
                persona_rows=promoted_rows,
                report=report,
                out_path=out_path,
                manifest_path=manifest_path,
            )

            self.assertTrue(out_path.exists())
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "promoted")
            self.assertEqual(manifest["manifest_type"], "dataset_promotion")
            self.assertEqual(manifest["artifact_type"], "dataset_promotion")
            self.assertEqual(manifest["status"], "promoted")
            self.assertEqual(manifest["persona_count"], 200)
            self.assertEqual(manifest["dataset_hash"], hash_file_bytes(out_path))
            self.assertEqual(manifest["promoted_output_hash"], hash_file_bytes(out_path))
            self.assertEqual(manifest["candidate_input_hash"], hash_file_bytes(candidate_path))
            self.assertEqual(manifest["review_manifest_hash"], hash_file_bytes(review_path))
            self.assertEqual(manifest["rejection_counts"], {})
            self.assertEqual(manifest["validator_versions"]["dataset_promotion"], dataset_promotion.PROMOTION_VERSION)
            validate_dataset_promotion_manifest(manifest_path, persona_path=out_path)

    def test_non_dry_run_blocked_does_not_write_output(self):
        (report, promoted_rows), out_path = self.evaluate(
            [self.valid_rows_200[0]],
            [],
            dry_run=False,
        )
        manifest_path = out_path.with_suffix(".manifest.json")

        with self.assertRaises(PersonaValidationError):
            dataset_promotion.write_promotion_outputs(
                persona_rows=promoted_rows,
                report=report,
                out_path=out_path,
                manifest_path=manifest_path,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(out_path.exists())
        self.assertFalse(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
