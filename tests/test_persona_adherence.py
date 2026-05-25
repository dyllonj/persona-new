import copy
import unittest
from pathlib import Path

from persona_eval import (
    load_jsonl,
    calibrate_persona_adherence_mock,
    persona_adherence_calibration_fixtures,
    persona_adherence_status_not_applicable,
    score_persona_adherence_mock,
    serialize_persona_for_adherence,
    validate_persona_adherence_status,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class PersonaAdherenceScaffoldingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(SAMPLE_PATH)
        cls.row = cls.rows[0]

    def test_persona_serialization_is_deterministic(self):
        first = serialize_persona_for_adherence(self.row)
        second = serialize_persona_for_adherence(self.row)

        self.assertEqual(first, second)
        self.assertIn("Persona Facts:", first)
        self.assertIn("Persona Traits:", first)
        self.assertIn("Persona Values:", first)
        self.assertIn("Forbidden Behaviors:", first)

    def test_persona_serialization_changes_when_persona_changes(self):
        changed = copy.deepcopy(self.row)
        changed["persona_spec"]["facts"][0] = "I require implementation logs."

        self.assertNotEqual(
            serialize_persona_for_adherence(self.row),
            serialize_persona_for_adherence(changed),
        )

    def test_mock_calibration_separates_positive_from_swapped_negative(self):
        fixtures = persona_adherence_calibration_fixtures(self.rows[:3])
        calibration = calibrate_persona_adherence_mock(fixtures)

        self.assertEqual(calibration["status"], "mock_only")
        self.assertTrue(calibration["separates_positive_from_negative"])
        self.assertGreater(calibration["positive_mean"], calibration["persona_swapped_negative_mean"])
        self.assertIsNone(calibration["threshold"])

    def test_pa_mock_status_does_not_report_real_score(self):
        response = "I prefer concise summaries and ask for evidence before making recommendations."

        metric = score_persona_adherence_mock(self.row, response)

        self.assertEqual(metric["status"], "mock_only")
        self.assertEqual(metric["reason_code"], "mock_backends_not_real_pa")
        self.assertIsNone(metric["value"])
        self.assertIsNone(metric["threshold"])
        self.assertIsNone(metric["pass_at_threshold"])
        self.assertTrue(metric["mock_only"])
        self.assertGreater(metric["mock_score"], 0.0)
        validate_persona_adherence_status(metric)

    def test_pa_without_pinned_backends_is_not_applicable(self):
        metric = persona_adherence_status_not_applicable(
            persona_serialization_hash="sha256:" + "0" * 64,
        )

        self.assertEqual(metric["status"], "not_applicable")
        self.assertEqual(metric["reason_code"], "real_pa_backends_not_pinned")
        self.assertIsNone(metric["value"])
        self.assertFalse(metric["mock_only"])
        validate_persona_adherence_status(metric)


if __name__ == "__main__":
    unittest.main()
