import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import aggregate
import smoke_evidence
from persona_eval import (
    EVALUATOR_VERSION,
    EXTRACTOR_VERSION,
    METRIC_VERSION,
    PROMPT_TEMPLATE_VERSION,
    MockAdapter,
    PersonaValidationError,
    _build_run_rows,
    _write_run_outputs,
    create_run_manifest,
    load_jsonl,
    validate_run_evidence_file,
    validate_run_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"


def expanded_persona_rows(count):
    sample_rows = load_jsonl(SAMPLE_PATH)
    rows = []
    for index in range(count):
        row = copy.deepcopy(sample_rows[index % len(sample_rows)])
        persona_id = f"smoke_fixture_{index:03d}"
        row["persona_id"] = persona_id
        row["source"]["source_persona_id"] = f"{persona_id}_source"
        row["seed_prompt"]["prompt_id"] = f"{persona_id}_seed"
        for variant_index, variant in enumerate(row["variants"]):
            variant["variant_id"] = f"{persona_id}_v{variant_index}"
        rows.append(row)
    return rows


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class SmokeEvidenceTests(unittest.TestCase):
    def build_smoke_artifacts(self, root: Path, persona_count: int = 20):
        persona_path = root / "personas.jsonl"
        output_dir = root / "results" / "vllm_smoke_20"
        report_dir = root / "reports" / "vllm_smoke_20"
        persona_rows = expanded_persona_rows(persona_count)
        write_jsonl(persona_path, persona_rows)

        run_id = "vllm_smoke_20_fixture"
        seeds = [1]
        decoding_params = {"temperature": 0.0, "max_tokens": 140}
        manifest = create_run_manifest(
            persona_path=persona_path,
            model_base="base",
            model_tuned="tuned",
            seeds=seeds,
            run_id=run_id,
            timestamp_utc="2026-05-25T00:00:00Z",
            code_commit="test-commit",
            dirty_worktree=True,
            decoding_params=decoding_params,
            extractor_version=EXTRACTOR_VERSION,
        )
        manifest["harness_version"] = EVALUATOR_VERSION
        manifest["metric_version"] = METRIC_VERSION
        manifest["model_base_alias"] = "base"
        manifest["model_tuned_alias"] = "tuned"
        manifest["score_mode"] = "disabled"
        validate_run_manifest(manifest)

        args = argparse.Namespace(
            model_base="base",
            model_tuned="tuned",
            model_base_alias="base",
            model_tuned_alias="tuned",
            model_base_revision_or_hash="not_available",
            model_tuned_revision_or_hash="not_available",
            tokenizer_name="not_available",
            tokenizer_hash="not_available",
            chat_template_hash="not_available",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            disable_token_kl=True,
            score_mode="disabled",
        )
        adapter = MockAdapter()
        result_rows = _build_run_rows(
            rows=persona_rows,
            args=args,
            base_adapter=adapter,
            tuned_adapter=adapter,
            run_id=run_id,
            seeds=seeds,
            decoding_params=decoding_params,
            stop_sequences=[],
        )
        manifest_path, results_path = _write_run_outputs(
            output_dir=output_dir,
            manifest=manifest,
            rows=result_rows,
        )
        report = aggregate.build_report(manifest_path=manifest_path, results_path=results_path)
        written = aggregate.write_report(report, report_dir)
        return manifest_path, results_path, Path(written["aggregate_report_json"])

    def test_builds_valid_smoke_evidence_from_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, results_path, report_path = self.build_smoke_artifacts(root)
            out_path = root / "reports" / "vllm_smoke_20" / "smoke_evidence.json"

            evidence = smoke_evidence.build_smoke_evidence(
                manifest_path=manifest_path,
                results_path=results_path,
                aggregate_report_path=report_path,
                out_path=out_path,
            )

            self.assertTrue(out_path.exists())
            self.assertEqual(evidence["persona_count"], 20)
            self.assertEqual(evidence["variants_per_persona"], 6)
            self.assertEqual(evidence["model_count"], 2)
            self.assertEqual(evidence["seed_count"], 1)
            self.assertEqual(evidence["planned_generation_calls"], 240)
            self.assertEqual(evidence["matched_result_rows"], 120)
            validate_run_evidence_file(
                out_path,
                expected_stage="smoke",
                expected_persona_count=20,
                expected_seed_count=1,
                expected_call_count_value=240,
            )

    def test_rejects_non_smoke_persona_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, results_path, report_path = self.build_smoke_artifacts(root, persona_count=19)
            out_path = root / "reports" / "vllm_smoke_20" / "smoke_evidence.json"

            with self.assertRaisesRegex(PersonaValidationError, "persona_count=20"):
                smoke_evidence.build_smoke_evidence(
                    manifest_path=manifest_path,
                    results_path=results_path,
                    aggregate_report_path=report_path,
                    out_path=out_path,
                )


if __name__ == "__main__":
    unittest.main()
