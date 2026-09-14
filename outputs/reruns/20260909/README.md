# Simulator fixes and experiment reruns, 2026-09-09

All six full experiments completed. The reference ron mean is unchanged at **5368.934469677947 points** (standard error 12.894486260951517), from 70,210 accepted tenpai among 100,000 hands. All five 20,000-table experiments reproduce their saved outcomes and wait statistics exactly. Their post-rejection win counters now distinguish ron from tsumo.

The original result files remain in `outputs/`. Use this directory for the refreshed aggregates. [comparison.json](comparison.json) records every change to existing statistics; [manifest.json](manifest.json) records commands, elapsed times, and hashes of the implementation and original result files. [benchmark.json](benchmark.json) contains the separate 1,000-trial benchmarks.

## Corrected continuation counts

Each row counts table endings after at least one cheap-hand rejection anywhere at the table. The eventual winner need not be the player whose tenpai was rejected. All rows use seed `20260809`, trials `0–19999`, the value policy, future weights `2.0/1.0`, and reversal penalty `25`.

| Experiment | Previously labeled tsumo | Correct tsumo | Correct ron |
|---|---:|---:|---:|
| [Plain continuation](shared_four_player_extremeA_dora1_continue_rejected_20k.json) | 84 | 84 | 0 |
| [5200-point dama line](shared_four_player_extremeA_valuable_dama_3han40fu_20k.json) | 104 | 92 | 12 |
| [8000-point dama line](shared_four_player_extremeA_valuable_dama_4han40fu_20k.json) | 91 | 87 | 4 |
| [5200-point line plus haneman](shared_four_player_extremeA_valuable_dama_3han40fu_haneman_20k.json) | 115 | 98 | 17 |
| [8000-point line plus haneman](shared_four_player_extremeA_valuable_dama_4han40fu_haneman_20k.json) | 102 | 93 | 9 |

For every experiment, accepted tenpai, dama tsumo, dama ron, and exhaustion after rejection sum to the number of tables with a rejection. The old mislabeled count equals the corrected tsumo-plus-ron count. Overall outcome totals were already correct.

## Reference hand value and wait categories

The [100,000-hand reference](hand_monte_carlo_value_reversal25_wait_matrix_100k.json) uses seed `20260813`, trials `0–99999`, the value policy, future weights **0.5/0.5**, and reversal penalty `25`. Scoring is non-dealer riichi ron without ippatsu, includes separately sampled ura, and weights winning tiles by visible-information unseen copies. Outcomes, scores, physical-copy totals, and turn statistics reproduce the original result.

The original reference file used eight wait categories. The current classifier already separates three-sided waits and gives standard shapes priority over seven pairs. The refreshed distribution therefore has:

| Category | Original count | Refreshed count | Refreshed share |
|---|---:|---:|---:|
| Three-sided wait (`sanmenchan`) | Not separate | 3312 | 4.72% |
| Other | 6029 | 2718 | 3.87% |
| Seven-pairs single wait | 6775 | 6774 | 9.65% |

All other category counts are unchanged. Combining these three categories conserves the raw hand counts, winning-type opportunities, unseen/live copies, and next-draw probability sums, including the tile-category breakdowns. This regrouping comes from the classifier already present before these four fixes. The Chinese report's reference tables now use the refreshed nine-category result; older comparisons retain explicitly labeled eight-category grouping.

## Fixes and checks

- Python `analyze` forwards both shanten-specific future weights. The weighted fixture now reports `40.54` instead of silently using defaults and reporting `36.86`.
- Python policy comparisons preserve those weights in the baseline. The [20-trial same-policy diagnostic](python_same_policy_comparison_20.json), seed `17`, has identical baseline and candidate outcomes.
- Continuation reports count `dama_tsumo_after_rejection` and `dama_ron_after_rejection` separately.
- Python records every permitted reversal and updates every pending recovery. The [multi-reversal diagnostic](python_multiple_reversals_trial1.json), trial `1`, seed `17`, records turns `1, 2, 3, 4` and recovery delays `5, 4, 3, 2`. The full experiments retain `max_per_hand=1`, including the documented budget reset after rejected tenpai in continuation mode.

Validation passed: **80 unit tests**, native self-test with the production `necessary_tiles_masks` backend, Python/native weighted-fixture agreement, seeded same-policy comparisons in both engines, six aggregate comparisons, original-result hash checks, and `git diff --check`. No scoring errors or invalid winning-copy weights occurred in the reruns.

## Reproduce

Install the pinned simulator requirements and build the default CMake backend as described in the repository README. Then run:

```sh
/private/tmp/hand-sim-venv/bin/python outputs/reruns/20260909/rerun_experiments.py \
  --binary /private/tmp/hand-efficiency-build/hand_efficiency_sim_cpp --jobs 4
/private/tmp/hand-sim-venv/bin/python outputs/reruns/20260909/compare_results.py
```

Add `--benchmark 1000` to run a separate short benchmark under `/private/tmp`. The full runner reads settings from the original saved metadata and writes only into this dated directory. Native observations and intermediate scoring decisions are temporary; the aggregates, commands, hashes, and comparisons are durable.
