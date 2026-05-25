# Candidate Pools

Sprint 6 candidate pools live here, outside `data/`. Generated JSONL files and
their sidecar manifests are ignored by git by default.

Create the deterministic local fixture pool:

```bash
python3 candidate_generation.py create-fixtures --out candidates/sample_candidates.jsonl
```

The default pool contains 300 local synthetic candidates. The generator uses
deterministic diversity profiles so a clean 200-row subset can be selected
without network, API, or GPU access:

```bash
python3 candidate_generation.py select \
  --candidate-path candidates/sample_candidates.jsonl \
  --out candidates/selected_candidates.jsonl
```

Validate a candidate pool and its sidecar manifest when present:

```bash
python3 candidate_generation.py validate --candidate-path candidates/sample_candidates.jsonl
```

Sidecar manifests include row counts, output/content hashes, per-candidate row
hashes, generator and validator versions, source inventory, variant types, a
dedupe/diversity report, and explicit rejection or blocker reasons. Clean
default fixture pools should have empty rejection and blocker reason lists.

Do not move candidate pools into `data/`. `data/` is reserved for validated
benchmark inputs after the Sprint 7 promotion gate.
