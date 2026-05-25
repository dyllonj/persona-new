# Persona Autoresearch Program

This file is the agent-facing research program for this repository. It adapts
Karpathy's `autoresearch` pattern to the persona-drift evaluation spec in
`AGENTS.MD`, `guide.MD`, and `IMPLEMENTATION_PLAN.md`.

## Bottom Line

Run autonomous research only after the fixed benchmark contract is explicit and
testable. The agent may improve implementation quality, adapter coverage, metric
plumbing, and report ergonomics, but it must not move the benchmark goalposts to
make an experiment look better.

The core rule:

- The evaluator contract is the anchor.
- The experiment surface is mutable.
- Results are logged every time.
- A change is kept only if it improves the chosen objective without violating
  any hard gate.

## Setup

To start a new autoresearch run, work with the user to:

1. Agree on a short run tag, such as `may25-sprint0` or `smoke-vllm-a`.
2. Confirm the current branch is intentional.
3. Read the controlling files:
   - `AGENTS.MD`
   - `IMPLEMENTATION_PLAN.md`
   - `guide.MD`
   - `program.md`
   - `auto_research.py`
4. Initialize logging:

```bash
python3 auto_research.py init --tag <tag>
```

5. Check repository readiness:

```bash
python3 auto_research.py check
```

Do not begin an autonomous loop if the readiness check fails.

## Fixed Contract

These are not experiment variables:

- Exactly six executable variants per persona.
- Correct call counts:
  - sample: `120`
  - smoke: `240`
  - dev: `1,200`
  - full: `4,800`
- Nested `source.*` provenance metadata.
- vLLM as the default local serving path.
- TGI as secondary production-parity support only.
- Token-KL only through aligned scoring on the same fixed continuation.
- BC-F1 as field-level behavior scoring.
- Persona-level statistical aggregation.
- No full dataset before schema, provenance, validation, and review gates pass.

If an experiment requires weakening one of these rules, discard the experiment.

## Modes

### Bootstrap Mode

Use this while the evaluator harness is not fully implemented.

Allowed work:

- Implement or improve `pyproject.toml`.
- Implement or improve `persona_eval.py`.
- Implement or improve JSON Schemas under `schemas/`.
- Implement or improve sample fixtures under `data/`.
- Implement or improve tests under `tests/`.
- Improve result logging and reproducibility checks.

Primary objective:

- More hard gates enforced by tests.
- Fewer failing tests.
- No new network, GPU, or API-key requirement for fixture tests.

Keep a change only if:

- `python3 -m unittest discover -s tests` or the repository's active test command passes, and
- the implementation enforces more of the fixed contract, or simplifies code
  while preserving the same enforced contract.

### Benchmark Mode

Use this only after the fixture harness, mock run, and smoke path exist.

Allowed work:

- Add new experiment configs.
- Add adapter support without changing metric definitions.
- Improve report generation without changing raw metric semantics.
- Add analysis cuts that are secondary and clearly labeled.

Not allowed:

- Changing schema definitions after the benchmark baseline unless the change is a
  bug fix and old results are invalidated.
- Changing metric formulas to improve a result.
- Dropping failing personas, variants, or seeds from the denominator.
- Reporting Token-KL when the status contract says it is unavailable or
  diagnostic-only.

Primary objective:

- The exact objective must be named in the run description before each run.
- Examples:
  - `maximize BC-F1 field consistency on mock fixture`
  - `minimize flagged invalid behavior pairs`
  - `reduce Token-KL not_applicable causes by implementing aligned local scoring`

## Run Command

Use the controller to capture every run:

```bash
python3 auto_research.py run-once \
  --tag <tag> \
  --description "short description of the experiment" \
  -- python3 -m unittest discover -s tests
```

When the benchmark CLI exists, dry-run and smoke commands should also be logged:

```bash
python3 auto_research.py run-once \
  --tag <tag> \
  --description "sample count dry-run" \
  -- python3 persona_eval.py plan --persona-path data/personas.sample.jsonl --model-base base --model-tuned tuned --seeds 1
```

The controller writes:

- `results/autoresearch/<tag>/results.tsv`
- `results/autoresearch/<tag>/logs/*.log`

The TSV is intentionally uncommitted experiment state unless the user asks to
preserve a run log in version control.

## Keep Or Discard

After each run:

1. Read the command status and parsed metrics.
2. Inspect the diff.
3. Decide whether the change is a keep candidate or discard candidate.
4. Record the decision in the next run description or summary.

Default keep rules:

- Keep if tests pass and the change enforces more of the fixed contract.
- Keep if metrics improve and all hard gates still pass.
- Keep if equal metrics come with meaningfully simpler code.

Default discard rules:

- Discard crashes unless they expose a useful bug that is immediately fixed.
- Discard any change that weakens schemas, provenance, run counting, Token-KL
  validity, BC-F1 field scoring, or persona-level aggregation.
- Discard any change that makes fixture tests require network, GPU, or API keys.

The controller does not reset git by default. Do not use destructive git
commands unless the user explicitly authorizes that for the run.

## Output Discipline

Do not flood the context with full logs. Use:

```bash
python3 auto_research.py summarize --tag <tag>
```

For failing runs, inspect the smallest useful excerpt:

```bash
tail -n 80 results/autoresearch/<tag>/logs/<log-file>.log
```

## Stop Conditions

Stop the autonomous loop and report status when:

- The fixed contract is ambiguous.
- A required source/license/provenance fact cannot be verified.
- The benchmark would need a full dataset before fixture tests pass.
- The run requires credentials, paid API calls, or local GPU access not already
  approved for this workspace.
- Three consecutive attempts fail for the same root cause.

Otherwise, continue with small experiments that preserve the contract.
