# Push/fold and tile-danger experiments

The requested safety preset, after-draw state machine, and both multiple-riichi
selectors are implemented. Each continuation policy was run on 20,000 tables
with seed `20260809`, trial IDs `0–19999`, and the `extremeA` weights: future
ukeire `0.5/2/1`, improvement `0.1`, all value weights `1`, and reversal penalty
`25`, at most once per hand through turn 9. Kanchan preference is zero in the
continuation comparisons.

| Policy used by all four seats | Ron ending | Tsumo ending | Exhaustion | Chosen danger: one riichi | Chosen danger: two riichi |
|---|---:|---:|---:|---:|---:|
| All push | 64.53% | 24.50% | 10.98% | 4.57% | 8.72% |
| Full fold | 9.74% | 35.68% | 54.59% | 0.44% | No exposure |
| Mixed, mean safety | 24.64% | 36.86% | 38.51% | 1.29% | 1.43% |
| Mixed, minimax safety | 24.81% | 36.73% | 38.47% | 1.29% | 1.51% |

Ron endings include multiple ron once per table. Danger conditions on the
number of declared opponents at that decision, excludes the observer from
that count, and includes observers already locked into riichi. These
populations depend on earlier policy choices and survival. Each table uses
one policy at every seat, so this comparison does not establish the best
policy for one player facing fixed opponents.

The two mixed presets give similar overall outcomes. Their per-seat paired
point differences are all below nine points in magnitude. Both are retained
as configurable heuristics; average safety ranks are not probabilities.

## Results and denominators

- [Honor danger by visible copies](honor_danger_by_visible_copies.md): non-genbutsu
  honors split by own hand plus public copies immediately before discard, at
  turns 6/9/12/15. The [JSON](honor_danger_by_visible_copies.json) and
  [CSV](honor_danger_by_visible_copies.csv) include counts and clustered errors;
  [per-table coordinate totals](honor_danger_by_visible_copies.trials.jsonl)
  preserve the counts needed for future regrouping. The split reconciles
  exactly to the original mixed-mean honor rows.
- [Paper danger table](tile_danger_mixed_mean_by_turn.md): mixed mean safety,
  exactly one riichi opponent, all tile categories at observer turns 6/9/12/15,
  with candidate counts and table-clustered uncertainty. The companion
  [CSV](tile_danger_mixed_mean_by_turn.csv) and [JSON](tile_danger_mixed_mean_by_turn.json)
  preserve all 56 selected cells and the source checksum. Regenerate with
  `python outputs/build_push_fold_paper_tables.py` from the repository root.
- [Comparison](comparison.md) and [machine-readable comparison](comparison.json):
  all four policies, actual ron/tsumo values, and paired-table differences.
- [All-push JSON](all_push_20k.json), [full-fold JSON](full_fold_20k.json),
  [mixed mean JSON](mixed_mean_20k.json), and [mixed minimax JSON](mixed_minimax_20k.json).
- [All-push CSV](all_push_20k.csv), [full-fold CSV](full_fold_20k.csv),
  [mixed mean CSV](mixed_mean_20k.csv), and [mixed minimax CSV](mixed_minimax_20k.csv):
  all 34 tile types and rank categories, observer turn, 1/2/3 riichi opponents,
  visible suji, and separate candidate/chosen denominators and clustered errors.
- Each aggregate also includes physical red/ordinary safety cells, own and
  public copy counts, per-seat outcomes, declaration wait distributions and
  unseen/live winning-copy counts. Matching `*.trials.jsonl` files retain
  the per-table payments and outcomes used for pairing.
- [Validation manifest](validation.json): seed/range checks, count
  reconciliation, conservation, source hashes, and control-equivalence evidence.

Primary danger counts each held tile type once per decision. The physical
diagnostic can count both an ordinary representative and a red representative,
so its candidate total is separate. Both candidate and chosen-discard standard
errors are clustered by independent table, including zero-exposure tables.

The stopping rule is any legal ron/tsumo or exhaustive draw, including dama
ron and multiple ron. The production artifacts retain a legacy native summary
field `config.stop_at="tenpai"`; that field is unused by this mode. The current
engine reports `table_terminal`. This metadata correction does not alter any
observations or measurements.

## Tanyao and outer-kanchan diagnosis

[Four-way danger comparison](ablations/danger_factorial_summary_20k.md)
crosses tanyao weight `0/1` with pure 2/8 kanchan preference `0/1`. Each cell
uses 20,000 paired tables in the original single-riichi mode, which stops at
the second declaration. The original baseline remains unchanged.

| Tanyao | Kanchan preference | Turn 12: double-suji 4/6 minus suji 2/8 | Turn 12: half-suji 4/6 minus no-suji 2/8 |
|---:|---:|---:|---:|
| 1 | 0 | +1.578 pp | +0.815 pp |
| 0 | 0 | +1.170 pp | +0.419 pp |
| 1 | 1 | +1.361 pp | +0.565 pp |
| 0 | 1 | +0.961 pp | +0.265 pp |

Neither adjustment reverses the ordering in the point estimates. With tanyao
on, the kanchan preference narrows the two gaps by `0.217 ± 0.098` and
`0.251 ± 0.059` percentage points, respectively (paired-table standard errors).
The 2/8 share among kanchan waits rises from about 22.39% to 24.08%. The
tanyao-removal contrast is less precise. These are policy diagnostics, not
confirmation that the requested safety ordering matches human play.

The danger summary also retains nine-category wait distributions, kanchan
winning-rank shares, own/public-copy cells, raw/aggregate reconciliation,
matching trial seeds, and all six paired contrasts. Original and new danger
CSVs include every exact tile identity, not only 4/6.

## First-tenpai hand-value companion

The old danger stream lacks the target's physical tile identities, so exact
red/ura scoring uses separate paired first-tenpai rollouts with the same four
policy configurations. Each has 20,000 tables. This sample stops at the first
policy-accepted tenpai and does not impose the danger experiment's four-live-
tile declaration condition.

Scores include projected riichi and the separately sampled ura, with no
ippatsu. Every seat keeps its actual seat-wind yaku, while point values are
normalized to non-dealer ron. Each hand first averages its unseen physical
winning copies; the headline then averages those hand means. Live-wall
weighting remains a separate diagnostic.

| Tanyao | Kanchan preference | Scored hands | Mean ron | Hand SE |
|---:|---:|---:|---:|---:|
| 1 | 0 | 19,851 | 5,180.5 | 22.7 |
| 0 | 0 | 19,836 | 5,178.4 | 22.9 |
| 1 | 1 | 19,851 | 5,179.0 | 22.7 |
| 0 | 1 | 19,836 | 5,177.3 | 22.9 |

All four scoring runs have zero scoring errors. Files named
`ablations/first_tenpai_tanyao*_kanchan*_20k.json` contain the rollout wait and
availability statistics; their shared-table wrapper intentionally does not
score value. The separate `ablations/first_tenpai_value_tanyao*_kanchan*_20k.json`
files contain the scored means, per-table values, and source checksums.
The [value factorial summary](ablations/value_factorial_summary_20k.md) also
reconciles the rollout and scoring counts and reports paired differences on
the intersection of tables with a score in both variants.

The [verification record](verification.md) lists the completed tests and
reproducibility checks.

## Reproduction

The honor split was added on 2026-09-14 by regenerating the same mixed-mean
20,000 tables. The raw stream, per-table metrics and every aggregate measurement
exactly reproduce the original result. The [split validation record](honor_split_validation_20260914.json)
contains 51 reproduction, clustering and publication checks. To reproduce the
split, retain the mixed-mean raw stream from the command below, then run:

```sh
python outputs/extract_push_fold_honor_danger.py /private/tmp/mixed.jsonl \
  --visible-copies own-plus-public \
  --output-stem outputs/push_fold_experiments/20260909/honor_danger_by_visible_copies
```

The extractor checks the original unsplit honor totals and saves per-table
own/public coordinate counts. These counts support new pooled probabilities
and cluster errors; exposed-decision counts cannot be summed across coordinates
because one decision can expose more than one coordinate. Zero observed hits
are not a general safety guarantee, including the all-four-visible bin.

Use the pinned Python environment and the CMake production binary documented
in the repository README. For each continuation policy:

```sh
/private/tmp/hand-sim-venv/bin/python outputs/run_push_fold_experiments.py \
  --native-binary /private/tmp/hand-efficiency-build/hand_efficiency_sim_cpp \
  --trials 20000 --seed 20260809 --policy value \
  --future-ukeire-weight 0.5 --future-ukeire-weight-1-shanten 2 \
  --future-ukeire-weight-2-shanten 1 --improvement-weight 0.1 \
  --dora-ukeire-weight 1 --allow-shanten-reversal \
  --shanten-reversal-penalty 25 --mode mixed --safety-strategy mean_safety \
  --observations /private/tmp/mixed.jsonl --output /private/tmp/mixed.json \
  --csv /private/tmp/mixed.csv --trial-metrics /private/tmp/mixed.trials.jsonl
```

Change mode to `all_push` or `full_fold`; the fourth run uses mode `mixed`
and strategy `minimax_safety`. Defaults are dora penalty 1, half-push maximum
rank 5, cheap maximum total han 1, valuable-dama line 5200 normalized points,
two-riichi minimum 7700 actual points, and haneman lock enabled. All native
commands, scoring configuration and timings are embedded in each result.
The 200-table benchmark took 2.43 seconds for native rollout/decision scoring
and 0.85 seconds for aggregation. Production continuation runs took roughly
7–8.4 minutes including aggregation while other experiments ran concurrently.

For the original single-riichi factorial, use
`outputs/analyze_tile_danger_by_turn.py` with the same explicit weights and
`--tanyao-weight 0|1 --kanchan-outer-wait-weight 0|1`. Preserve the initial
baseline and write new outputs under separate names.

For the value companion, use `outputs/hand_monte_carlo_report.py` with those
same weights, `--engine native --shared-four-player-table`, and retain
`--observations`. Then run:

```sh
/private/tmp/hand-sim-venv/bin/python outputs/score_first_tenpai_values.py \
  --input /private/tmp/first-tenpai.jsonl --expected-seed 20260809 \
  --output /private/tmp/first-tenpai-value.json
```

The scorer checks enabled reversal, homogeneous policy metadata, unique trial
IDs and matching trial seeds. It took about five seconds per 20,000-table
scoring pass. Calls, open-hand yaku, opponent policy heterogeneity, honba,
ippatsu, and whole-game placement incentives remain outside these experiments.
