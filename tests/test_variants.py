import copy
import unittest
from pathlib import Path

from persona_eval import REQUIRED_VARIANT_TYPES, PersonaValidationError, load_jsonl, validate_persona_row


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class VariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(SAMPLE_PATH)

    def test_sample_rows_have_exactly_six_executable_variants(self):
        for row in self.rows:
            self.assertEqual(len(row["variants"]), 6)

    def test_sample_rows_have_required_variant_types(self):
        for row in self.rows:
            self.assertEqual({variant["type"] for variant in row["variants"]}, REQUIRED_VARIANT_TYPES)

    def test_rejects_fewer_than_six_variants(self):
        row = copy.deepcopy(self.rows[0])
        row["variants"] = row["variants"][:5]
        with self.assertRaises(PersonaValidationError):
            validate_persona_row(row)

    def test_rejects_more_than_six_variants(self):
        row = copy.deepcopy(self.rows[0])
        row["variants"].append(copy.deepcopy(row["variants"][0]))
        row["variants"][-1]["variant_id"] = "seed_001_extra"
        with self.assertRaises(PersonaValidationError):
            validate_persona_row(row)

    def test_rejects_missing_required_variant_type(self):
        row = copy.deepcopy(self.rows[0])
        row["variants"][0]["type"] = "paraphrase"
        with self.assertRaises(PersonaValidationError):
            validate_persona_row(row)

    def test_rejects_duplicate_variant_ids(self):
        row = copy.deepcopy(self.rows[0])
        row["variants"][1]["variant_id"] = row["variants"][0]["variant_id"]
        with self.assertRaises(PersonaValidationError):
            validate_persona_row(row)


if __name__ == "__main__":
    unittest.main()
