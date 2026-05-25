import copy
import unittest
from pathlib import Path

from persona_eval import load_jsonl, render_prompt


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


class PromptRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = load_jsonl(SAMPLE_PATH)[0]
        cls.variant = cls.row["variants"][0]

    def test_prompt_rendering_is_deterministic(self):
        first = render_prompt(self.row, self.variant)
        second = render_prompt(self.row, self.variant)
        self.assertEqual(first, second)
        self.assertIn("Persona Facts:", first.prompt_text)
        self.assertIn("Persona Traits:", first.prompt_text)
        self.assertIn("Persona Values:", first.prompt_text)
        self.assertIn("Forbidden Behaviors:", first.prompt_text)
        self.assertIn("User Task / Question:", first.prompt_text)

    def test_empty_optional_sections_render_consistently(self):
        row = copy.deepcopy(self.row)
        row["persona_spec"]["facts"] = []
        row["persona_spec"]["traits"]["tone"] = []
        row["persona_spec"]["traits"]["style"] = []
        row["persona_spec"]["traits"]["values"] = []
        row["persona_spec"]["forbidden_behaviors"] = []

        rendered = render_prompt(row, self.variant)

        self.assertIn("Persona Facts:\n- [none]", rendered.prompt_text)
        self.assertIn("Persona Values:\n- [none]", rendered.prompt_text)
        self.assertIn("Forbidden Behaviors:\n- [none]", rendered.prompt_text)
        self.assertIn("- tone: [none]", rendered.prompt_text)
        self.assertIn("- style: [none]", rendered.prompt_text)

    def test_prompt_hash_changes_when_rendered_prompt_text_changes(self):
        original = render_prompt(self.row, self.variant)
        row = copy.deepcopy(self.row)
        row["persona_spec"]["facts"][0] = "I require audited implementation evidence."

        changed = render_prompt(row, self.variant)

        self.assertNotEqual(original.prompt_text, changed.prompt_text)
        self.assertNotEqual(original.prompt_hash, changed.prompt_hash)

    def test_system_prompt_hash_changes_when_system_prompt_changes(self):
        original = render_prompt(self.row, self.variant)
        changed = render_prompt(self.row, self.variant, system_prompt="Different system prompt.")

        self.assertEqual(original.prompt_text, changed.prompt_text)
        self.assertNotEqual(original.system_prompt, changed.system_prompt)
        self.assertNotEqual(original.system_prompt_hash, changed.system_prompt_hash)


if __name__ == "__main__":
    unittest.main()
