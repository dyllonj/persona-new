import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DOCS = [
    "FINAL_REPORT.md",
    "METHODOLOGY.md",
    "REPRODUCIBILITY.md",
    "LIMITATIONS.md",
    "DATASET_CARD.md",
    "MODEL_RUN_CARD.md",
]


class ReportingSkeletonTests(unittest.TestCase):
    def read_doc(self, name):
        path = ROOT / name
        self.assertTrue(path.exists(), f"{name} is missing")
        return path.read_text(encoding="utf-8")

    def test_reporting_docs_exist_and_are_marked_as_skeletons(self):
        for name in REPORT_DOCS:
            with self.subTest(name=name):
                text = self.read_doc(name)
                self.assertIn("PENDING_SPRINT_10", text)
                self.assertIn("Final-run evidence", text)
                self.assertIn("Diagnostic/pilot evidence", text)

    def test_required_result_dependent_sections_are_pending(self):
        combined = "\n".join(self.read_doc(name) for name in REPORT_DOCS)
        required_pending_patterns = {
            "dataset hash": r"Dataset hash[^\n]*PENDING_SPRINT_10",
            "manifest hash": r"Manifest hash[^\n]*PENDING_SPRINT_10",
            "results hash": r"Results hash[^\n]*PENDING_SPRINT_10",
            "aggregate hash": r"Aggregate hash[^\n]*PENDING_SPRINT_10",
            "code commit": r"Code commit[^\n]*PENDING_SPRINT_10",
            "model IDs": r"Model IDs[^\n]*PENDING_SPRINT_10",
            "tokenizer/chat-template metadata": (
                r"Tokenizer/chat-template metadata[^\n]*PENDING_SPRINT_10"
            ),
            "PA status": r"PA status[^\n]*PENDING_SPRINT_10",
            "Token-KL status": r"Token-KL status[^\n]*PENDING_SPRINT_10",
            "BC-F1 summaries": r"BC-F1 summaries[^\n]*PENDING_SPRINT_10",
        }
        for label, pattern in required_pending_patterns.items():
            with self.subTest(label=label):
                self.assertRegex(combined, pattern)

    def test_final_report_separates_final_from_diagnostic_evidence(self):
        text = self.read_doc("FINAL_REPORT.md")
        final_index = text.index("### Final-run evidence")
        diagnostic_index = text.index("### Diagnostic/pilot evidence")
        self.assertLess(final_index, diagnostic_index)
        self.assertRegex(text, r"must not be described as a full\s+benchmark result")
        self.assertIn("PENDING_SPRINT_10:final_flagged_examples", text)
        self.assertIn("PENDING_SPRINT_10:diagnostic_examples_if_used", text)

    def test_reproducibility_final_commands_are_manual_and_pending(self):
        text = self.read_doc("REPRODUCIBILITY.md")
        self.assertIn("Manual final rerun status: `PENDING_SPRINT_10`", text)
        self.assertIn("python3 persona_eval.py run", text)
        self.assertIn("python3 aggregate.py", text)
        self.assertIn("<local-vllm-url>", text)

    def test_docs_do_not_make_obvious_final_claims(self):
        combined = "\n".join(self.read_doc(name) for name in REPORT_DOCS)
        banned_patterns = [
            r"\bthe full benchmark shows\b",
            r"\bwe found that\b",
            r"\bfinal results show\b",
            r"\bproves persona adherence\b",
            r"\bcanonical Token-KL is [0-9]",
        ]
        for pattern in banned_patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, combined, flags=re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
