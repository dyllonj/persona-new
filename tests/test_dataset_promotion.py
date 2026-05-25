import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import dataset_promotion
from persona_eval import PersonaValidationError, load_jsonl, validate_persona_row


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


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
