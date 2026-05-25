import copy
import unittest
from pathlib import Path

from persona_eval import (
    REQUIRED_SOURCE_FIELDS,
    PersonaValidationError,
    load_jsonl,
    validate_persona_row,
    validate_personas,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(SAMPLE_PATH)

    def test_sample_has_ten_rows(self):
        self.assertEqual(len(self.rows), 10)

    def test_schema_accepts_valid_sample_rows(self):
        validated = validate_personas(SAMPLE_PATH)
        self.assertEqual(len(validated), 10)

    def test_schema_rejects_missing_source_fields(self):
        for field in sorted(REQUIRED_SOURCE_FIELDS):
            row = copy.deepcopy(self.rows[0])
            del row["source"][field]
            with self.subTest(field=field):
                with self.assertRaises(PersonaValidationError):
                    validate_persona_row(row)

    def test_schema_rejects_invalid_source_date(self):
        row = copy.deepcopy(self.rows[0])
        row["source"]["retrieved_at"] = "2026-99-99"
        with self.assertRaises(PersonaValidationError):
            validate_persona_row(row)

    def test_schema_rejects_legacy_behavior_shape_without_canonical_fields(self):
        row = copy.deepcopy(self.rows[0])
        row["annotation"]["gold_labels"] = {
            "stance": "neutral_to_skeptical",
            "action_set": ["request_evidence"],
        }
        with self.assertRaises(PersonaValidationError):
            validate_persona_row(row)

    def test_behavior_labels_use_only_canonical_fields(self):
        for row in self.rows:
            self.assertEqual(
                set(row["expected_behavior"]),
                {"stance", "primary_action", "secondary_modifiers"},
            )
            self.assertEqual(
                set(row["annotation"]["gold_labels"]),
                {"stance", "primary_action", "secondary_modifiers"},
            )


if __name__ == "__main__":
    unittest.main()
