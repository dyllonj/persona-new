# Remaining Work Coordinator Goal

You are the Remaining Work Coordinator for `/Users/dyllon/Documents/persona-new`.

Goal: spawn, manage, and integrate specialist agents to unblock Sprint 9 safely. Current state: Batch R is complete; Sprint 9 dev still cannot run because Sprint 7 promotion is blocked on the missing full review manifest and Sprint 8 smoke evidence has not been produced. Do not run Sprint 10.

## Read First

- `AGENTS.MD`
- `REMAINING_SPRINTS_PLAN.md`
- `README.md`
- `dataset_readiness.py`
- `dataset_promotion.py`
- `candidate_generation.py`
- `persona_eval.py`
- `aggregate.py`
- `schemas/`
- `tests/`
- `reports/sprint7_promotion_blocked_manifest.json` if present
- `candidates/sprint7_selected_candidates.jsonl` if present
- Any Sprint 8 smoke evidence files if present

## Hard Constraints

- Do not start Sprint 10.
- Do not run `dev_50` until Sprint 7 promotion and Sprint 8 smoke evidence are valid.
- Do not bypass staged caps. Implement intentional phase-aware caps only.
- Do not create or modify `data/personas.full.jsonl` except through the approved Sprint 7 promotion path.
- Do not fabricate review, semantic, NLI, contradiction, or smoke evidence.
- Do not require network/API/GPU in unit tests.
- Preserve unrelated WIP from other agents.
- Use `python3`.
- Commit after coherent slices once checks pass.

## Current Blockers

- Batch R candidate diversity, review-gate hardening, promotion-gate hardening,
  and phase-aware run gates are complete.
- `candidates/sprint7_selected_candidates.jsonl` is the expected selected
  200-row candidate input, if present.
- No promoted `data/personas.full.jsonl`.
- No full review manifest.
- Sprint 8 smoke evidence was not supplied to Sprint 9.
- Sprint 9 remains blocked until promotion and smoke evidence are real.

## Batch R: Completed Reference

Do not rerun Batch R unless a regression appears. Its completed scope was:

- Candidate diversity selection and candidate manifest hardening.
- Full-review evidence gate hardening.
- Promotion gate hardening for exact 200-row promotion.
- Phase-aware smoke/dev/full run caps and evidence gates.

Current verification reported by the Batch R coordinator:

- `python3 -m unittest discover -s tests` passed with 158 tests.
- `python3 -m pytest` passed with 158 tests.
- Candidate selection produced 200 selected candidates.
- Promotion dry-run remained blocked only because the full review manifest was
  missing.
- Dev/full planning still reported 1,200 and 4,800 calls.

## Batch S: Run Next

### Agent S1: Sprint 7 Promotion Execution

Task:

- Run candidate validation, safety filters, dedupe, review checks, semantic/NLI/contradiction checks, and promotion dry-run.
- If and only if all gates pass, promote exactly 200 rows to `data/personas.full.jsonl`.
- Write promotion manifest with hashes, validator versions, review manifest hash, rejection counts, and reasons.

Verify:

```bash
python3 -m unittest discover -s tests
python3 -m pytest
python3 dataset_promotion.py validate-candidates ...
python3 dataset_promotion.py promote ... --dry-run
python3 persona_eval.py validate --persona-path data/personas.full.jsonl
```

Stop if fewer or more than 200 rows can be promoted without weakening standards.

## Batch T: Sequential Evidence Gates

### Agent T1: Sprint 8 Smoke Evidence

Task:

- Run only after Sprint 7 promotion succeeds.
- Execute 20-persona local vLLM smoke if endpoint/model metadata are available.
- Aggregate smoke results.
- Review flagged outputs.
- Write smoke evidence JSON for Sprint 9.

Stop if local endpoint/model metadata are unavailable or raw request/response logging is incomplete.

### Agent T2: Sprint 9 Dev Run

Task:

- Run only after valid promotion, smoke evidence, and dev cap-lift gates exist.
- Execute 50-persona dev run: `50 * 6 * 2 * 2 = 1,200` calls.
- Aggregate results.
- Report truncation/error/ambiguity rates and go/no-go for Sprint 10.

Stop if smoke evidence is missing, cap-lift is unapproved, or aggregate validation fails.

## Coordinator Responsibilities

- Keep a live status table: agent, files owned, status, blockers, tests run.
- Avoid file ownership collisions. If two agents need the same file, serialize their edits.
- Merge only coherent, tested slices.
- After each merged slice, run at least:
  - `python3 -m unittest discover -s tests`
  - `python3 -m pytest`
  - `git status --short`
  - `git diff --stat`
- Commit coherent slices after tests pass.
- Do not mark Sprint 9 executable until Sprint 7 promotion and Sprint 8 smoke evidence are both real and validated.

## Final Report

- Which agents ran.
- Files changed by each agent.
- Tests/checks run.
- Whether `data/personas.full.jsonl` was created, and why.
- Promotion status and dataset hash if promoted.
- Smoke evidence path if produced.
- Whether Sprint 9 dev is now executable.
- Whether Sprint 10 remains blocked.
- Confidence level.
