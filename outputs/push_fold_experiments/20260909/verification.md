# Verification

Publication and honor-split update on 2026-09-14:

- Restored the pinned Python environment and rebuilt the production tile-mask backend.
- Production native self-test passed; full Python discovery passed **130 tests**.
- A 200-table benchmark reproduced the saved per-table metrics exactly.
- Regenerated all 20,000 mixed-mean tables: raw observations and per-table
  metrics are byte-identical; all aggregate measurements match.
- Honor bins use own hand plus public copies before discard, excluding
  genbutsu and retaining exactly one declared opponent. All 16 cells reconcile
  to the published unsplit honor rows.
- Independent pooling of saved coordinate counts reproduces split and unsplit
  table-clustered errors. The [validation record](honor_split_validation_20260914.json)
  contains **51 passing checks**.
- Intermediate report tables and former Section 6.17 were removed; current
  result subsections are 6.1–6.6. The danger and honor tables match both papers.

Completed on 2026-09-09 using the pinned environment and the production CMake
`necessary_tiles_masks` backend.

- Full Python discovery: **129 tests passed** in 30.141 seconds.
- Production native `--self-test`: passed, including the three-versus-four
  live-tile boundary for two-riichi value decisions.
- Fixed Python/native outer-kanchan feature agreement: passed.
- Native continuation integration: all-push, full-fold, mixed mean and mixed
  minimax; physical state, after-draw shanten, safety ranks, compulsory riichi
  discards, ron winners, aggregation and seed-shard reproducibility passed.
- Seeded 64-trial native and Python runs with kanchan preference 1 completed
  without scoring errors. Each produced 45 tenpai hands; the engines have
  different random generators, so this is a behavior check rather than a
  claim of identical sampled hands.
- Legacy single-riichi observations and summary remained byte-identical in
  a 32-table comparison against the preserved earlier production binary.
- The late-riichi eligibility correction affects the mixed policy. All-push
  and full-fold controls remained byte-identical in 128-table before/current
  fallback comparisons with real scoring; evidence is embedded in the manifest.
- A 200-table saved stream reaggregated to exactly the same measurements.
- All four continuation results passed **201 artifact checks** for paired
  trial ranges, policies, counts, payments, point conservation, safety,
  candidate/physical denominators and source provenance in [validation.json](validation.json).
- The danger factorial checks matching trial seeds, source hashes and all
  four raw/aggregate count reconciliations, with paired-cluster uncertainty.
- All four first-tenpai value companions completed with zero scoring errors.
- `git diff --check`: passed.

Paper-table extraction on the same date validated all **56 selected cells**
(14 safety/tile categories at turns 6/9/12/15) against the mixed-mean source:
unique cells, counts, hit probabilities and 20,000 independent-table denominators.
The generated table matches the main paper §15.2, companion report §6.16 and
standalone readout exactly. Source SHA-256, result links and whitespace were
checked; the extraction script also passed Python compilation. These publication
checks use the existing experiment results and do not constitute new rollouts.

Re-run the main checks with:

```sh
/private/tmp/hand-sim-venv/bin/python -m unittest discover -s tests -p 'test_*.py'
/private/tmp/hand-efficiency-build/hand_efficiency_sim_cpp --self-test
/private/tmp/hand-sim-venv/bin/python outputs/validate_push_fold_results.py
git diff --check
```
