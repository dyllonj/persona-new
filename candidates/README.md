# Candidate Pools

Sprint 6 candidate pools live here, outside `data/`. Generated JSONL files and
their sidecar manifests are ignored by git by default.

Create the deterministic local fixture pool:

```bash
python3 candidate_generation.py create-fixtures --out candidates/sample_candidates.jsonl
```

Validate a candidate pool and its sidecar manifest when present:

```bash
python3 candidate_generation.py validate --candidate-path candidates/sample_candidates.jsonl
```

Do not move candidate pools into `data/`. `data/` is reserved for validated
benchmark inputs after the Sprint 7 promotion gate.
