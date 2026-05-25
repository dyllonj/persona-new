import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

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
    hash_file_bytes,
    load_jsonl,
    validate_run_evidence_file,
    validate_run_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "personas.sample.jsonl"
FULL_PATH = ROOT / "data" / "personas.full.jsonl"
PROMOTION_MANIFEST_PATH = ROOT / "reports" / "dataset_promotion_manifest.json"


class FakeVLLMAdapter(MockAdapter):
    adapter_name = "vllm"
    serving_stack = "vllm"
    serving_stack_version = "vllm-fixture"
    scoring_capability = "none"

    def __init__(self, endpoint):
        self.provider_or_endpoint = endpoint


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
    def build_smoke_artifacts(self, root: Path, persona_count: int = 20, *, real_runtime: bool = True):
        persona_path = root / "personas.jsonl"
        output_dir = root / "results" / "vllm_smoke_20"
        report_dir = root / "reports" / "vllm_smoke_20"
        if real_runtime:
            manifest_persona_path = FULL_PATH
            persona_rows = load_jsonl(FULL_PATH)[:persona_count]
        else:
            manifest_persona_path = persona_path
            persona_rows = expanded_persona_rows(persona_count)
            write_jsonl(persona_path, persona_rows)

        run_id = "vllm_smoke_20_fixture"
        seeds = [1]
        decoding_params = {"temperature": 0.0, "max_tokens": 140}
        adapter_name = "vllm" if real_runtime else "mock"
        serving_stack = "vllm" if real_runtime else "mock"
        serving_stack_version = "vllm-fixture" if real_runtime else "sprint1"
        provider = "model_specific_endpoints" if real_runtime else "local_mock"
        manifest = create_run_manifest(
            persona_path=manifest_persona_path,
            model_base="Qwen/Qwen2.5-7B" if real_runtime else "base",
            model_tuned="Qwen/Qwen2.5-7B-Instruct" if real_runtime else "tuned",
            seeds=seeds,
            run_id=run_id,
            timestamp_utc="2026-05-25T00:00:00Z",
            code_commit="test-commit",
            dirty_worktree=True,
            chat_template_hash="sha256:" + "2" * 64 if real_runtime else "not_available",
            tokenizer_name="Qwen/Qwen2.5-7B" if real_runtime else "not_available",
            tokenizer_hash="sha256:" + "1" * 64 if real_runtime else "not_available",
            model_base_revision_or_hash="qwen-base-fixture-revision" if real_runtime else "not_available",
            model_tuned_revision_or_hash="qwen-tuned-fixture-revision" if real_runtime else "not_available",
            adapter=adapter_name,
            provider_or_endpoint=provider,
            provider_or_endpoint_base="http://localhost:8001/v1" if real_runtime else None,
            provider_or_endpoint_tuned="http://localhost:8002/v1" if real_runtime else None,
            serving_stack=serving_stack,
            serving_stack_version=serving_stack_version,
            gpu_cuda_driver="fixture-gpu-cuda-driver" if real_runtime else "not_available",
            decoding_params=decoding_params,
            extractor_version=EXTRACTOR_VERSION,
            promotion_manifest_path=PROMOTION_MANIFEST_PATH if real_runtime else None,
        )
        manifest["harness_version"] = EVALUATOR_VERSION
        manifest["metric_version"] = METRIC_VERSION
        manifest["model_base_alias"] = "base"
        manifest["model_tuned_alias"] = "tuned"
        manifest["score_mode"] = "disabled"
        validate_run_manifest(manifest)

        args = argparse.Namespace(
            model_base=manifest["model_base"],
            model_tuned=manifest["model_tuned"],
            model_base_alias="base",
            model_tuned_alias="tuned",
            model_base_revision_or_hash=manifest["model_base_revision_or_hash"],
            model_tuned_revision_or_hash=manifest["model_tuned_revision_or_hash"],
            tokenizer_name=manifest["tokenizer_name"],
            tokenizer_hash=manifest["tokenizer_hash"],
            chat_template_hash=manifest["chat_template_hash"],
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            disable_token_kl=True,
            score_mode="disabled",
        )
        base_adapter = FakeVLLMAdapter("http://localhost:8001/v1") if real_runtime else MockAdapter()
        tuned_adapter = FakeVLLMAdapter("http://localhost:8002/v1") if real_runtime else MockAdapter()
        result_rows = _build_run_rows(
            rows=persona_rows,
            args=args,
            base_adapter=base_adapter,
            tuned_adapter=tuned_adapter,
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
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "aggregate_report.json"
        report = {
            "report_schema_version": "aggregate_report_v2",
            "aggregation_version": "test",
            "run_id": run_id,
            "source_manifest_hash": hash_file_bytes(manifest_path),
            "source_results_hash": hash_file_bytes(results_path),
            "availability_summaries": {
                "behavioral_consistency_f1": {},
                "persona_adherence": {},
                "token_kl": {},
            },
            "metric_summaries": {
                "behavioral_consistency_f1": {},
                "persona_adherence": {},
                "token_kl": {},
            },
            "paired_output_deltas": {
                "numeric_deltas": {
                    "total_tokens": {
                        "status": "not_applicable",
                        "effect_size": {"status": "not_applicable", "value": None},
                        "p_value": {"status": "not_applicable", "value": None},
                    }
                },
                "truncation_pass_fail_mcnemar": {"status": "not_applicable", "value": None},
            },
            "statistical_method_notes": {
                "inference_unit": "persona_id",
                "multiple_comparison_note": "fixture report for smoke evidence tests",
            },
            "full_dataset_readiness": {
                "status": "ready",
                "checks": {"fixture": {"status": "ready"}},
                "blocking_checks": [],
            },
            "counts": {
                "persona_count": persona_count,
                "row_count": len(result_rows),
            },
        }
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return manifest_path, results_path, report_path

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
            self.assertEqual(evidence["adapter"], "vllm")
            self.assertEqual(evidence["serving_stack"], "vllm")
            self.assertEqual(evidence["runtime_metadata_policy"]["tokenizer_hash"], "shared_cli_value")
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

    def test_rejects_mock_artifacts_as_smoke_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, results_path, report_path = self.build_smoke_artifacts(root, real_runtime=False)
            out_path = root / "reports" / "vllm_smoke_20" / "smoke_evidence.json"

            with self.assertRaisesRegex(PersonaValidationError, "real vllm/openai-compatible adapter"):
                smoke_evidence.build_smoke_evidence(
                    manifest_path=manifest_path,
                    results_path=results_path,
                    aggregate_report_path=report_path,
                    out_path=out_path,
                )

    def test_rejects_stale_evidence_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, results_path, report_path = self.build_smoke_artifacts(root)
            out_path = root / "reports" / "vllm_smoke_20" / "smoke_evidence.json"
            smoke_evidence.build_smoke_evidence(
                manifest_path=manifest_path,
                results_path=results_path,
                aggregate_report_path=report_path,
                out_path=out_path,
            )

            with results_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")

            with self.assertRaisesRegex(PersonaValidationError, "results_hash does not match"):
                validate_run_evidence_file(
                    out_path,
                    expected_stage="smoke",
                    expected_persona_count=20,
                    expected_seed_count=1,
                    expected_call_count_value=240,
                )


if __name__ == "__main__":
    unittest.main()
