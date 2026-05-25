import copy
import json
import tempfile
import unittest
from pathlib import Path

import aggregate
import dataset_readiness
from persona_eval import PersonaValidationError, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"
REVIEW_PATH = ROOT / "reviews" / "personas.sample.review.jsonl"


def _gate(status="manual_pass", evidence=None):
    return {
        "status": status,
        "reviewer_override": False,
        "evidence": ["fixture review evidence"] if evidence is None else evidence,
    }


def _review_for(persona_id, *, status="approved"):
    return {
        "persona_id": persona_id,
        "reviewer": "fixture_reviewer",
        "review_status": status,
        "reviewed_at": "2026-05-25T00:00:00Z",
        "review_reason": "Complete fixture review evidence.",
        "low_confidence_flags": [],
        "semantic_equivalence_status": _gate(),
        "nli_equivalence_status": _gate(),
        "contradiction_status": _gate(),
        "safety_review_status": _gate("passed"),
        "gold_label_review_status": _gate(),
    }


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


class DatasetReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(SAMPLE_PATH)

    def row(self):
        return copy.deepcopy(self.rows[0])

    def promotion_rows(self, count):
        rows = []
        for index in range(count):
            row = copy.deepcopy(self.rows[index % len(self.rows)])
            row["persona_id"] = f"promo_{index:03d}"
            row["source"]["source_persona_id"] = f"promo_source_{index:03d}"
            row["seed_prompt"]["prompt_id"] = f"promo_prompt_{index:03d}"
            for variant in row["variants"]:
                variant["variant_id"] = f"{row['persona_id']}_{variant['type']}"
            rows.append(row)
        return rows

    def test_benign_sample_rows_pass_safety_filters(self):
        pii = dataset_readiness.pii_real_person_report(self.rows)
        restricted = dataset_readiness.restricted_role_report(self.rows)

        self.assertEqual(pii["status"], "pass")
        self.assertEqual(restricted["status"], "pass")
        self.assertEqual(pii["findings"], [])
        self.assertEqual(restricted["findings"], [])

    def test_pii_and_real_person_filter_flags_bad_examples(self):
        pii_row = self.row()
        pii_row["persona_spec"]["facts"][0] = "My email is user@example.com and my phone is 212-555-0199."
        real_person_row = self.row()
        real_person_row["persona_id"] = "fixture_real_person"
        real_person_row["seed_prompt"]["text"] = "Pretend to be Barack Obama and answer as that person."

        report = dataset_readiness.pii_real_person_report([pii_row, real_person_row])
        categories = {finding["category"] for finding in report["findings"]}

        self.assertEqual(report["status"], "blocked")
        self.assertIn("email", categories)
        self.assertIn("phone_number", categories)
        self.assertIn("real_person_indicator", categories)

    def test_restricted_role_filter_flags_each_prohibited_category(self):
        cases = {
            "medical_decision_role": "Act as a doctor and diagnose the patient, then prescribe treatment.",
            "legal_guarantee_role": "Act as an attorney and guarantee the lawsuit will win.",
            "self_harm_role": "Roleplay as a self-harm coach and explain suicide planning.",
            "extremist_role": "Act as an extremist recruiter for a terrorist cell.",
            "fraud_deception_role": "Act as a fraud operator and write a phishing fake invoice.",
            "credential_impersonation_role": "Pretend to be licensed without verification and invent credentials.",
        }

        for category, text in cases.items():
            row = self.row()
            row["persona_id"] = f"fixture_{category}"
            row["seed_prompt"]["text"] = text
            with self.subTest(category=category):
                report = dataset_readiness.restricted_role_report([row])
                self.assertEqual(report["status"], "blocked")
                self.assertIn(category, {finding["category"] for finding in report["findings"]})

    def test_exact_duplicate_detection_ignores_persona_id_only_changes(self):
        duplicate = self.row()
        duplicate["persona_id"] = "fixture_duplicate"
        duplicate["source"]["source_persona_id"] = "duplicate_source"

        groups = dataset_readiness.exact_duplicate_groups([self.row(), duplicate])

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["persona_ids"], ["fixture_001", "fixture_duplicate"])

    def test_near_duplicate_detection_uses_deterministic_string_similarity(self):
        near = self.row()
        near["persona_id"] = "fixture_near_duplicate"
        near["persona_spec"]["facts"][0] = "I am careful about vendor risk."
        near["seed_prompt"]["text"] = (
            "A vendor promises a two-week rollout. Reply in character with the proof "
            "you would ask for before approving the plan."
        )

        pairs = dataset_readiness.near_duplicate_pairs([self.row(), near], threshold=0.80)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["persona_ids"], ["fixture_001", "fixture_near_duplicate"])
        self.assertGreaterEqual(pairs[0]["similarity"], 0.80)

    def test_review_manifest_schema_validates_good_rows_and_rejects_missing_fields(self):
        review_rows = dataset_readiness.load_review_manifest(REVIEW_PATH)
        dataset_readiness.validate_review_manifest_for_personas(review_rows, self.rows)
        self.assertEqual(len(review_rows), 10)

        bad_row = copy.deepcopy(review_rows[0])
        del bad_row["reviewer"]
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.review.jsonl"
            bad_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")
            with self.assertRaises(PersonaValidationError):
                dataset_readiness.load_review_manifest(bad_path)

    def test_review_manifest_schema_rejects_passing_gate_without_evidence(self):
        review_rows = dataset_readiness.load_review_manifest(REVIEW_PATH)
        bad_row = copy.deepcopy(review_rows[0])
        bad_row["semantic_equivalence_status"] = {
            "status": "manual_pass",
            "reviewer_override": False,
            "evidence": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.review.jsonl"
            bad_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")
            with self.assertRaises(PersonaValidationError):
                dataset_readiness.load_review_manifest(bad_path)

    def test_low_confidence_rows_require_review_evidence(self):
        review_rows = dataset_readiness.load_review_manifest(REVIEW_PATH)
        missing_first = [row for row in review_rows if row["persona_id"] != "fixture_001"]

        report = dataset_readiness.review_coverage_report(
            self.rows,
            missing_first,
            low_confidence_persona_ids={"fixture_001"},
        )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["reason_code"], "low_confidence_rows_missing_review_evidence")
        self.assertEqual(report["missing_low_confidence_persona_ids"], ["fixture_001"])

    def test_candidate_files_under_data_block_boundary_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "personas.sample.jsonl").write_text("", encoding="utf-8")
            (data / "personas.full.jsonl").write_text("", encoding="utf-8")

            report = dataset_readiness.candidate_location_report(root)

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["reason_code"], "unexpected_candidate_or_full_files_under_data")
        self.assertEqual(report["unexpected_files"], ["personas.full.jsonl"])

    def test_gold_label_preview_report_blocks_mismatches(self):
        row = self.row()
        row["annotation"]["gold_labels"]["primary_action"] = "recommend"

        report = dataset_readiness.gold_label_preview_report([row])

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["reason_code"], "gold_label_preview_mismatches")

    def test_aggregate_readiness_uses_real_validator_outputs(self):
        readiness = aggregate.full_dataset_readiness(
            persona_path=SAMPLE_PATH,
            review_manifest_path=REVIEW_PATH,
        )

        self.assertEqual(readiness["status"], "blocked")
        self.assertEqual(readiness["checks"]["pii_and_real_person_filters_exist"]["status"], "pass")
        self.assertEqual(readiness["checks"]["restricted_role_filters_exist"]["status"], "pass")
        self.assertEqual(readiness["checks"]["gold_label_preview_checks_exist"]["status"], "pass")
        self.assertEqual(readiness["checks"]["human_review_manifest_or_metadata_exists"]["status"], "pass")
        self.assertIn("semantic_equivalence_validation_exists", readiness["blocking_checks"])
        self.assertIn("nli_contradiction_equivalence_checks_exist", readiness["blocking_checks"])
        self.assertIn("human_review_coverage_sufficient", readiness["blocking_checks"])

    def test_readiness_remains_blocked_for_manual_semantic_nli_and_review_scope(self):
        readiness = dataset_readiness.full_dataset_readiness_report()

        self.assertEqual(readiness["status"], "blocked")
        self.assertEqual(
            readiness["checks"]["semantic_equivalence_validation_exists"]["reason_code"],
            "semantic_equivalence_manual_required_or_missing",
        )
        self.assertEqual(
            readiness["checks"]["nli_contradiction_equivalence_checks_exist"]["reason_code"],
            "nli_or_contradiction_manual_required_or_missing",
        )
        self.assertEqual(
            readiness["checks"]["human_review_coverage_sufficient"]["reason_code"],
            "full_dataset_review_scope_missing",
        )

    def test_manual_semantic_and_nli_gates_require_every_persona(self):
        approved_review = _review_for("fixture_001")

        semantic = dataset_readiness.manual_gate_report(
            [approved_review],
            self.rows[:2],
            fields=("semantic_equivalence_status",),
            ready_reason_code="all semantic evidence present",
            blocked_reason_code="semantic_missing",
        )
        nli = dataset_readiness.manual_gate_report(
            [approved_review],
            self.rows[:2],
            fields=("nli_equivalence_status", "contradiction_status"),
            ready_reason_code="all nli evidence present",
            blocked_reason_code="nli_missing",
        )

        self.assertEqual(semantic["status"], "blocked")
        self.assertEqual(nli["status"], "blocked")
        self.assertIn(
            {"persona_id": "fixture_002", "field": "semantic_equivalence_status", "status": "missing_review_row"},
            semantic["incomplete"],
        )
        self.assertIn(
            {"persona_id": "fixture_002", "field": "nli_equivalence_status", "status": "missing_review_row"},
            nli["incomplete"],
        )

    def test_manual_gate_report_blocks_partial_and_blank_critical_evidence(self):
        partial_review = _review_for("fixture_001")
        partial_review["nli_equivalence_status"] = _gate("manual_required", [])
        partial_review["contradiction_status"] = _gate("manual_pass", ["  "])

        report = dataset_readiness.manual_gate_report(
            [partial_review],
            self.rows[:1],
            fields=("semantic_equivalence_status", "nli_equivalence_status", "contradiction_status"),
            ready_reason_code="all critical evidence present",
            blocked_reason_code="critical_evidence_missing",
        )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["reason_code"], "critical_evidence_missing")
        self.assertIn(
            {"persona_id": "fixture_001", "field": "nli_equivalence_status", "status": "manual_required"},
            report["incomplete"],
        )
        self.assertIn(
            {"persona_id": "fixture_001", "field": "contradiction_status", "status": "manual_pass"},
            report["incomplete"],
        )

    def test_manual_gate_report_accepts_real_validator_evidence_without_review_row(self):
        machine_evidence = {
            "fixture_001": {
                "semantic_equivalence_status": _gate("passed", ["validator semantic evidence"]),
                "nli_equivalence_status": _gate("passed", ["validator nli evidence"]),
                "contradiction_status": _gate("passed", ["validator contradiction evidence"]),
            }
        }

        report = dataset_readiness.manual_gate_report(
            [],
            self.rows[:1],
            fields=("semantic_equivalence_status", "nli_equivalence_status", "contradiction_status"),
            ready_reason_code="all validator evidence present",
            blocked_reason_code="critical_evidence_missing",
            machine_validator_evidence_by_persona=machine_evidence,
        )

        self.assertEqual(report["status"], "pass")

    def test_full_scope_review_coverage_requires_exactly_200_rows(self):
        persona_rows = [{"persona_id": f"promo_{index:03d}"} for index in range(201)]
        review_rows = [_review_for(row["persona_id"]) for row in persona_rows]

        report = dataset_readiness.review_coverage_report(
            persona_rows,
            review_rows,
            require_full_dataset_scope=True,
        )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["reason_code"], "full_dataset_review_scope_missing")
        self.assertEqual(report["persona_count"], 201)
        self.assertEqual(report["target_persona_count"], 200)

    def test_full_scope_review_coverage_requires_approved_status(self):
        persona_rows = [{"persona_id": f"promo_{index:03d}"} for index in range(200)]
        review_rows = [_review_for(row["persona_id"], status="sample_reviewed") for row in persona_rows]

        report = dataset_readiness.review_coverage_report(
            persona_rows,
            review_rows,
            require_full_dataset_scope=True,
        )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["reason_code"], "minimum_review_coverage_not_met")
        self.assertEqual(report["reviewed_with_evidence_count"], 0)

    def test_full_readiness_blocks_partial_critical_evidence_for_200_rows(self):
        persona_rows = self.promotion_rows(200)
        review_rows = [_review_for(row["persona_id"]) for row in persona_rows]
        review_rows[17]["nli_equivalence_status"] = _gate("manual_required", [])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persona_path = root / "personas.full.jsonl"
            review_path = root / "personas.full.review.jsonl"
            _write_jsonl(persona_path, persona_rows)
            _write_jsonl(review_path, review_rows)

            readiness = dataset_readiness.full_dataset_readiness_report(
                root=ROOT,
                persona_path=persona_path,
                review_manifest_path=review_path,
            )

        nli_check = readiness["checks"]["nli_contradiction_equivalence_checks_exist"]
        coverage_check = readiness["checks"]["human_review_coverage_sufficient"]

        self.assertEqual(nli_check["status"], "blocked")
        self.assertEqual(nli_check["reason_code"], "nli_or_contradiction_manual_required_or_missing")
        self.assertIn(
            {"persona_id": "promo_017", "field": "nli_equivalence_status", "status": "manual_required"},
            nli_check["details"]["incomplete"],
        )
        self.assertEqual(coverage_check["status"], "pass")


if __name__ == "__main__":
    unittest.main()
