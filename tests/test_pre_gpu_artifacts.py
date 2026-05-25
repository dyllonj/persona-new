import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PreGPUArtifactTests(unittest.TestCase):
    def test_qwen_model_pair_config_has_required_runtime_placeholders(self):
        path = ROOT / "configs" / "model_pair.qwen2_5_7b.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["base"]["model_id"], "Qwen/Qwen2.5-7B")
        self.assertEqual(payload["tuned"]["model_id"], "Qwen/Qwen2.5-7B-Instruct")
        for side in ("base", "tuned"):
            self.assertIn("revision_or_hash", payload[side])
            self.assertIn("tokenizer_name", payload[side])
            self.assertIn("tokenizer_hash", payload[side])
            self.assertIn("chat_template_hash", payload[side])
            self.assertIn("fill-from-gpu-host", payload[side]["revision_or_hash"])
            self.assertIn("fill-from-gpu-host", payload[side]["tokenizer_hash"])
            self.assertIn("fill-from-gpu-host", payload[side]["chat_template_hash"])
        self.assertEqual(payload["serving_policy"]["default_stack"], "vllm")
        self.assertFalse(payload["serving_policy"]["hosted_apis_allowed_by_default"])
        self.assertFalse(payload["serving_policy"]["execute_with_placeholder_hashes"])

    def test_approval_templates_are_json_placeholders_not_approved_artifacts(self):
        expected = {
            "dev_run_approval.template.json": ("dev_run_approval", 50, 1200),
            "full_run_approval.template.json": ("full_run_approval", 200, 4800),
        }
        for filename, (approval_type, persona_count, planned_calls) in expected.items():
            with self.subTest(filename=filename):
                path = ROOT / "approvals" / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["approval_type"], approval_type)
                self.assertEqual(payload["persona_count"], persona_count)
                self.assertEqual(payload["planned_generation_calls"], planned_calls)
                self.assertNotEqual(payload["status"], "approved")
                self.assertIn("not approval evidence", payload["placeholder_warning"])
                self.assertIn("<", json.dumps(payload))

    def test_pre_gpu_runbook_names_required_checks_and_stop_conditions(self):
        text = (ROOT / "runbooks" / "pre_gpu_checklist.md").read_text(encoding="utf-8")

        self.assertIn("python3 persona_eval.py validate --persona-path data/personas.full.jsonl", text)
        self.assertIn("python3 dataset_readiness.py --persona-path data/personas.full.jsonl", text)
        self.assertIn("python3 -m unittest discover -s tests", text)
        self.assertIn("python3 -m pytest", text)
        self.assertIn("python3 smoke_evidence.py", text)
        self.assertIn("Stop Conditions", text)
        self.assertIn("planned_generation_calls=240", text)
        self.assertIn("planned_generation_calls=1200", text)
        self.assertIn("planned_generation_calls=4800", text)


if __name__ == "__main__":
    unittest.main()
