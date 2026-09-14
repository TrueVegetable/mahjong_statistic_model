# Discard danger after exactly one riichi

20,000 shared-table trials produced **148,466 defender decisions** and
**1,584,686 candidate tile-type observations**. Sampling starts after the
first riichi discard and ends at the second declaration or an earlier terminal
event. Turn means the defender's own draw number.

All tile groups are shown below. The underlying JSON/CSV also retain all 34
exact identities separately. Each cell is the candidate ron-hit percentage;
rows other than genbutsu exclude known-safe tiles. This is a descriptive table,
not an imposed safety ranking.

| Tile group / safety | Turn 6 | Turn 9 | Turn 12 | Turn 15 |
|---|---:|---:|---:|---:|
| Genbutsu, all tiles | 0.00% | 0.00% | 0.00% | 0.00% |
| Suji 1/9 | 0.60% | 0.75% | 1.09% | 1.28% |
| Honors | 0.74% | 0.64% | 0.85% | 1.12% |
| Double-suji 4/6 | 3.55% | 3.69% | 4.54% | 5.68% |
| Double-suji 5 | 2.18% | 3.37% | 4.66% | 7.58% |
| Suji 2/8 | 2.08% | 2.60% | 2.96% | 5.00% |
| Suji 3/7 | 4.31% | 4.63% | 6.09% | 9.24% |
| No-suji 1/9 | 5.06% | 6.35% | 7.50% | 9.12% |
| Half-suji 4/6 | 6.87% | 8.07% | 10.15% | 11.88% |
| Half-suji 5 | 6.02% | 8.32% | 10.04% | 12.29% |
| No-suji 2/8 | 6.32% | 7.75% | 9.34% | 10.91% |
| No-suji 3/7 | 8.83% | 10.09% | 11.86% | 14.47% |
| No-suji 4/6 | 10.37% | 12.34% | 13.81% | 17.33% |
| No-suji 5 | 10.36% | 12.97% | 14.62% | 15.83% |

At turn 12, the no-suji, half-suji, and double-suji 4/6 cells contain
10,622, 15,595, and 5,306 candidate observations
across 4,634, 5,477, and 2,850 tables. Table-clustered standard errors are
0.403, 0.296, and 0.357 percentage points respectively. The turn-6 double-suji
cell has only 761 observations and a larger 0.801-point standard error.

All **415,731 known-safe candidates had zero ron hits**. Across all ranks,
safety classes, and turns, candidate danger was 6.085% (SE 0.033 percentage
points). The actual attacking policy's chosen discards hit 4.274% of the
time; that statistic has a separate denominator of 148,466 decisions.

## Definitions and scope

- Each distinct tile type held in the defender's 14-tile hand counts once per
  decision. Duplicate physical copies do not multiply the denominator.
- A hit means the tile completes the riichi opponent's actual locked hand and
  that opponent is not furiten on any winning tile in its river. Furiten
  riichi hands remain in the denominator.
- Suji is computed before the candidate discard, using the target's entire
  river and discards safely passed after riichi. Other players' pre-riichi
  rivers do not automatically make tiles safe. Hidden waits supply hit labels
  only; they never supply suji or known-safe labels.
- Known-safe tiles take priority. For other suited tiles, 4/5/6 require both
  same-suit rank-three anchors for double suji; one gives half-suji. Outer
  ranks have one anchor. Honors are separate.
- The second declaration discard is tested for ron against the first player
  before stopping. A declaration requires at least four live tiles remaining.
- This is the unfiltered `extremeA` attacking baseline, with future weights
  2.0/1.0, dora weight 1.0, and reversal penalty 25. Folding, calls, valuable
  dama, cheap-bad-wait rejection, and non-target dama ron are absent.
- Later turns condition on survival without a win or second riichi. These
  rates are model estimates, and need calibration before use for human play.

19,763 tables entered the single-riichi observation period; 237 did not.
Terminations were second riichi (10,531), target ron (6,345), target tsumo
(2,174), and exhaustion after riichi (713).

## Data and reproduction

[JSON](tile_danger_after_riichi_by_turn_20k.json) and
[CSV](tile_danger_after_riichi_by_turn_20k.csv) include every observed turn,
34 exact tile identities, rank categories, safety classes, numerators,
denominators, exposed tables, decisions, and table-clustered errors. They also
include the target's current own turn and its turns elapsed since declaration.
Unobserved cells are omitted; a missing cell is not a zero-risk estimate.
The [validation manifest](tile_danger_after_riichi_by_turn_20k.validation.json)
records checks, source hashes, and the native command.

```sh
/private/tmp/hand-sim-venv/bin/python outputs/analyze_tile_danger_by_turn.py \
  --native-binary /private/tmp/hand-efficiency-build/hand_efficiency_sim_cpp \
  --trials 20000 --seed 20260809 \
  --output outputs/tile_danger_after_riichi_by_turn_20k.json \
  --csv outputs/tile_danger_after_riichi_by_turn_20k.csv
```

The native benchmark used 200 tables and took 2.07 seconds. The confirmatory
20,000-table run used trial IDs 0–19999, seed 20260809, east round, and the
`necessary_tiles_masks` backend; it took 197.65 seconds. Raw observations were
temporary files under `/private/tmp`. Regenerate them to add conditioning
dimensions absent from the aggregates.
