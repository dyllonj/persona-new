import unittest

from persona_eval import (
    PersonaValidationError,
    behavior_consistency_f1,
    behavior_tags_ok,
    extract_behavior_tags_rule_first,
    leave_one_out_majority_behavior_labels,
    pairwise_behavior_agreement,
    validate_behavior_tags,
)


class BehaviorMetricTests(unittest.TestCase):
    def labels(self, **overrides):
        value = {
            "stance": "neutral_to_skeptical",
            "primary_action": "request_evidence",
            "secondary_modifiers": ["avoid_overclaiming", "summarize_risk"],
        }
        value.update(overrides)
        return value

    def test_exact_match_scores_all_fields(self):
        metric = behavior_consistency_f1(self.labels(), self.labels())

        self.assertEqual(metric["status"], "ok")
        self.assertEqual(metric["stance_exact"], 1.0)
        self.assertEqual(metric["primary_action_exact"], 1.0)
        self.assertEqual(metric["secondary_modifiers_precision"], 1.0)
        self.assertEqual(metric["secondary_modifiers_recall"], 1.0)
        self.assertEqual(metric["secondary_modifiers_f1"], 1.0)
        self.assertEqual(metric["combined_score"], 1.0)

    def test_modifier_precision_recall_and_f1(self):
        metric = behavior_consistency_f1(
            self.labels(secondary_modifiers=["avoid_overclaiming", "extra"]),
            self.labels(secondary_modifiers=["avoid_overclaiming", "summarize_risk"]),
        )

        self.assertEqual(metric["secondary_modifiers_precision"], 0.5)
        self.assertEqual(metric["secondary_modifiers_recall"], 0.5)
        self.assertEqual(metric["secondary_modifiers_f1"], 0.5)

    def test_wrong_stance_cannot_be_hidden_by_modifier_overlap(self):
        metric = behavior_consistency_f1(
            self.labels(stance="support"),
            self.labels(),
        )

        self.assertEqual(metric["stance_exact"], 0.0)
        self.assertEqual(metric["primary_action_exact"], 1.0)
        self.assertEqual(metric["secondary_modifiers_f1"], 1.0)
        self.assertEqual(metric["combined_score"], 0.0)

    def test_wrong_primary_action_cannot_be_hidden_by_modifier_overlap(self):
        metric = behavior_consistency_f1(
            self.labels(primary_action="recommend"),
            self.labels(),
        )

        self.assertEqual(metric["stance_exact"], 1.0)
        self.assertEqual(metric["primary_action_exact"], 0.0)
        self.assertEqual(metric["secondary_modifiers_f1"], 1.0)
        self.assertEqual(metric["combined_score"], 0.0)

    def test_empty_modifier_sets_are_deterministic(self):
        metric = behavior_consistency_f1(
            self.labels(secondary_modifiers=[]),
            self.labels(secondary_modifiers=[]),
        )

        self.assertEqual(metric["secondary_modifiers_precision"], 1.0)
        self.assertEqual(metric["secondary_modifiers_recall"], 1.0)
        self.assertEqual(metric["secondary_modifiers_f1"], 1.0)
        self.assertEqual(metric["combined_score"], 1.0)

    def test_behavior_tag_schema_rejects_invalid_objects(self):
        with self.assertRaises(PersonaValidationError):
            validate_behavior_tags({"status": "ok"})
        with self.assertRaises(PersonaValidationError):
            behavior_tags_ok(self.labels(secondary_modifiers=["dup", "dup"]))
        with self.assertRaises(PersonaValidationError):
            behavior_tags_ok({"stance": "support", "primary_action": "refuse", "secondary_modifiers": []})

    def test_extractor_skeleton_returns_structured_status(self):
        parsed = extract_behavior_tags_rule_first("Please ask for evidence and avoid overclaiming.")
        ambiguous = extract_behavior_tags_rule_first("This sentence has no controlled behavior cue.")

        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["parser"], "rule_first")
        self.assertEqual(parsed["llm_tagger_status"], "disabled")
        self.assertEqual(parsed["labels"]["primary_action"], "request_evidence")
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertEqual(ambiguous["reason_code"], "rule_parser_no_match")
        self.assertTrue(ambiguous["human_review_required"])

    def test_leave_one_out_excludes_scored_variant(self):
        behaviors = [
            self.labels(stance="support", primary_action="recommend", secondary_modifiers=["only_self"]),
            self.labels(),
            self.labels(),
        ]

        majority = leave_one_out_majority_behavior_labels(behaviors, 0)

        self.assertEqual(majority["status"], "ok")
        self.assertEqual(majority["excluded_index"], 0)
        self.assertEqual(majority["labels"], self.labels())

    def test_pairwise_agreement_uses_unordered_non_self_pairs(self):
        behaviors = [self.labels(), self.labels(), self.labels(primary_action="recommend")]

        agreement = pairwise_behavior_agreement(behaviors)

        self.assertEqual(agreement["status"], "ok")
        self.assertEqual(agreement["pair_count"], 3)
        self.assertEqual(agreement["stance_exact_mean"], 1.0)
        self.assertLess(agreement["primary_action_exact_mean"], 1.0)
        self.assertLess(agreement["combined_score_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
