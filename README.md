# Statistical Mahjong: Why?

This repository contains a Chinese riichi-mahjong paper and its Python/C++ simulators. The local attack/fold model studies expected value with explicit tile counts, rivers, furiten and opponent behavior. A separate closed-hand, shared-table generator estimates average hand value, wait distributions, discard danger and live-wall wait quality under one configurable policy.

## Read the paper and final results

- [Main paper](outputs/statistical-mahjong-the-whys-cn.md): the expected-value model and worked examples.
- [Hand and table model report](outputs/hand_efficiency_simulator_report_zh.md): model definitions, retained results and behavioral limitations.
- [Final push/fold comparison](outputs/push_fold_experiments/20260909/README.md): paired 20,000-table policies, danger data and reproducibility records.
- [Danger by turn](outputs/push_fold_experiments/20260909/tile_danger_mixed_mean_by_turn.md): held candidate tiles against exactly one riichi opponent.
- [Honor danger by visible copies](outputs/push_fold_experiments/20260909/honor_danger_by_visible_copies.md): own hand plus public copies, with counts and table-clustered uncertainty.

Source code, tests, papers and durable experiment evidence are versioned together. Intermediate experiment files remain available under `outputs/` for provenance, even when their tables are omitted from the paper. Local environments, binaries, logs, disposable raw streams and superseded root-level exploratory copies are excluded. Paired `*.trials.jsonl` result summaries are retained for statistical checks.

## Main files

- `outputs/iishanten_wall_sim.py` — readable Python reference model, calibration code, and tests.
- `outputs/iishanten_wall_sim.cpp` — faster C++ simulation engine.
- `outputs/statistical-mahjong-the-whys-cn.md` — Chinese research article.
- `outputs/iishanten_calibrated_weights.json` — calibrated wait-template weights.
- `outputs/hand_efficiency_sim.py` — seeded closed-hand generator using shanten, then a tunable shallow-lookahead ukeire score, with wait/danger aggregates and replaceable policy/scoring hooks.
- `outputs/hand_efficiency_sim.cpp` — native batch engine with the same policy features and physical-wall observations.
- `outputs/hand_monte_carlo_report.py` — Python/native rollout wrapper for non-dealer ron value, han/fu, wait, remaining-copy, and wait-hit statistics.
- `outputs/analyze_tile_danger_by_turn.py` — single-riichi discard danger by defender turn and observer-visible suji, with candidate and chosen-discard denominators.
- `tests/` — deterministic and physical-state checks.

## Run

Run the Python reference simulator:

```sh
python3 outputs/iishanten_wall_sim.py --trials 10000 --json
```

Build and run the C++ engine:

```sh
c++ -std=c++17 -O3 outputs/iishanten_wall_sim.cpp -o outputs/iishanten_wall_sim_cpp
./outputs/iishanten_wall_sim_cpp --trials 10000 --json
```

Run the checks after installing the pinned dependencies below:

```sh
python3 -m unittest discover -s tests -v
./outputs/iishanten_wall_sim_cpp --self-test
```

## Closed-hand Monte Carlo generator

Install the pinned hand analyzer/scorer in a local environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r outputs/requirements-hand-simulator.txt
```

Inspect one hand or run a deterministic shard:

```sh
python3 outputs/hand_efficiency_sim.py analyze 123456789m9p11p23s --pretty
python3 outputs/hand_efficiency_sim.py simulate --trials 10000 --stop-at tenpai --pretty
python3 outputs/hand_efficiency_sim.py simulate --trials 10000 --scorer mahjong --observations outputs/hand_trials.jsonl
```

Enable the value-aware preset, customize individual terms, or compare it with
pure efficiency on the same seeded walls:

```sh
python3 outputs/hand_efficiency_sim.py analyze 234567m22p345s11z \
  --policy value --dora-indicator 4m --seat-wind east --pretty
python3 outputs/hand_efficiency_sim.py simulate --trials 10000 --policy custom \
  --dora-ukeire-weight 0.5 --yakuhai-weight 1 --tanyao-weight 0.25 --pretty
python3 outputs/hand_efficiency_sim.py compare --trials 10000 --policy value \
  --scorer mahjong --pretty
python3 outputs/hand_efficiency_sim.py compare --trials 10000 --policy pure \
  --allow-shanten-reversal --stop-at tenpai --pretty
```

For larger batches, build the native engine. By default, CMake fetches the
pinned GPL-3.0
[Necessary and Unnecessary Tiles](https://github.com/tomohxx/necessary-and-unnecessary-tiles)
calculator. Its lookup tables return shanten-improving draw and
shanten-preserving discard masks directly, avoiding repeated 34-tile scans:

```sh
cmake -S outputs -B build/hand-efficiency -DCMAKE_BUILD_TYPE=Release
cmake --build build/hand-efficiency --parallel
./build/hand-efficiency/hand_efficiency_sim_cpp --self-test
./build/hand-efficiency/hand_efficiency_sim_cpp simulate --trials 100000 \
  --observations outputs/hand_trials_cpp.jsonl
./build/hand-efficiency/hand_efficiency_sim_cpp compare --trials 100000 \
  --policy value
```

The C++ source also has a dependency-free fallback build:

```sh
c++ -std=c++17 -O3 outputs/hand_efficiency_sim.cpp -o outputs/hand_efficiency_sim_cpp
```

To build with only MIT-licensed Nyanten instead of the GPL mask backend:

```sh
cmake -S outputs -B build/hand-efficiency-nyanten -DCMAKE_BUILD_TYPE=Release \
  -DHAND_EFFICIENCY_USE_TILE_MASKS=OFF
cmake --build build/hand-efficiency-nyanten --parallel
```

The default native binary links GPL-3.0 code and must follow GPL-3.0 terms if
distributed. See `outputs/THIRD_PARTY_HAND_SIMULATOR.md` for pinned revisions.

Keep scoring off the rollout hot path, then score only realized winning hands
with the pinned Python adapter:

```sh
python3 outputs/hand_efficiency_sim.py score-observations \
  outputs/hand_trials_cpp.jsonl --pretty
```

Generate the combined research report with the native rollout engine and
Python scorer:

```sh
python3 outputs/hand_monte_carlo_report.py \
  --trials 100000 \
  --engine native \
  --native-binary build/hand-efficiency/hand_efficiency_sim_cpp \
  --policy value \
  --seat-wind south \
  --output outputs/value_policy_report.json \
  --pretty
```

To measure first-tenpai race selection rather than one isolated hand, run four
same-policy players on one shared physical table and stop at the first accepted
tenpai. This mode reports wait and seat distributions but deliberately skips
hand scoring:

```sh
python3 outputs/hand_monte_carlo_report.py \
  --trials 20000 \
  --engine native \
  --native-binary build/hand-efficiency/hand_efficiency_sim_cpp \
  --policy value \
  --allow-shanten-reversal \
  --shared-four-player-table \
  --output outputs/shared_four_player_first_tenpai.json \
  --pretty
```

Add `--reject-cheap-bad-wait-riichi` to that shared-table command to exclude a
nondealer first-tenpai sample when its visible-information riichi ron value
without ura is exactly 1,300 and its primary wait is kanchan, penchan, or a
shanpon/tanki that has no unseen honor winning copy. A shanpon containing an
honor is exempt only while at least one copy of that honor wait remains unseen. This is a post-race
sample-selection experiment; rejected players do not continue the hand.

Add `--continue-rejected-tenpai` together with that rejection flag to make the
selection rule part of the table rollout. A rejected player stays dama and
continues choosing efficiency-policy discards; its post-tenpai reversal budget
is refreshed and the turn cutoff is lifted, so it may temporarily leave tenpai
when the configured reversal score justifies the improvement. The table then
stops at the first eligible tenpai, a later dama tsumo, or wall exhaustion.
`dama_tsumo` is reported as its own outcome and is not included in the accepted
tenpai wait distribution.

`--valuable-bad-wait-dama-line 3han40fu` or `4han40fu` adds a value-based dama
branch to that continued rollout. The line is evaluated before riichi, without
ura, using normalized nondealer ron value (5,200 or 8,000 points); thus the
lower line also includes 4-han 25-fu chiitoitsu. At least one unseen winning
tile must have a valid non-riichi yaku and reach the line. The player may take
tenpai-preserving shape improvements but never break tenpai; a resulting good
wait declares riichi and stops the table. Actual dama ron and tsumo end the
table and are reported separately.

Add `--haneman-or-higher-dama` to lock the hand when every unseen winning tile
is worth at least 12,000 points. A tile with a pre-riichi ron yaku uses its
normalized nondealer ron value; a tile without a ron yaku must reach the same
line by closed-hand tsumo. This rule takes priority over declaring riichi on a
good wait. After locking, the player tsumogiris every non-winning draw and can
end the table by non-furiten ron or tsumo.

`--engine auto` (the default) uses a built native binary when it finds one and
otherwise runs the Python reference engine. All policy-weight flags pass
through unchanged. Every rollout stops at first tenpai. In single-hand mode, the wrapper scores
every remaining winning-tile copy as non-dealer riichi ron, averages unequal
waits within each hand by unseen-copy count, and then averages those hand
values. A separate ura indicator is sampled from the dead wall and included in
scoring; it is never exposed to the discard policy. The primary score and
han/fu distributions use `4 - concealed - publicly visible` copies. An exact
live-wall-weighted result remains available only as a diagnostic.

Each first-tenpai hand receives one defense-oriented primary wait class using
the priority `sanmenchan`, `ryanmen`, `kanchan`, `penchan`, `shanpon`, `tanki`,
`chiitoitsu_tanki`, `kokushi`, then `other`. Classification uses the whole
13-tile structure and complete winning-tile set, so an ambiguous winning tile
is not double counted. `sanmenchan` requires three same-suit suji wins and the
removable five-tile run between the outer wins, such as `23456` waiting on
`1/4/7`. Unresolved complex shapes are `other`. The report gives mutually
exclusive hand shares, a separate `other_by_distinct_wait_tile_types` breakdown keyed by the exact
number of distinct winning tile types; it reports both share within `other`
and share of all accepted tenpai hands, and a tile-hit matrix split by
winning-tile groups `19`, `28`, `37`, `456`, and
`honor`. Remaining tiles are reported both as visible-information copies and
exact live-wall copies. A hit rate is the probability that the next tile
sampled from the remaining live wall completes the relevant cell.

The policy compares discards lexicographically: lowest post-discard shanten first, then largest ukeire score. For 1- and 2-shanten hands,

```text
ukeire score = immediate ukeire
              + 0.5 × expected next-step ukeire
              + 0.1 × improvement-tile copies
```

Next-step ukeire looks ahead exactly one acceptance draw and one best pure-efficiency discard. Its expectation is weighted by the remaining copies of each immediate acceptance tile. An improvement tile keeps the same shanten but, after the best discard, strictly increases immediate ukeire. Use `--future-ukeire-weight` as the fallback, or override it with `--future-ukeire-weight-1-shanten` and `--future-ukeire-weight-2-shanten`. Three-shanten and earlier hands deliberately skip this lookahead. `--improvement-weight` tunes same-shanten improvement copies; setting all applicable weights to zero recovers the original shanten-then-immediate-ukeire policy.

An optional one-step shanten-reversal policy admits only `0 -> 1` and `1 -> 2`
discards through own turn 9. It can be selected at most once per hand and deep
evaluates only the two candidates with the best recovery-copy/value prescore:

```text
reversal score = recovery copies
                 + 0.5 x (copy-weighted ukeire after recovery
                          - best current immediate ukeire)
                 + dora ukeire bonus
                 - 25
```

The recovered average ukeire must first strictly exceed the best current
immediate ukeire. The reversal is then taken only when its score strictly
exceeds the best normal ukeire score; equality keeps the lower-shanten hand. Hypothetical recovery
uses one draw and one strict immediate-ukeire discard, with no recursive
reversal or improvement-tile scan. Enable it with
`--allow-shanten-reversal`; tune it with `--shanten-reversal-penalty`,
`--shanten-reversal-future-weight`, `--shanten-reversal-max-turn`, and
`--shanten-reversal-candidate-limit`. Trial and aggregate output report
opportunities, selections, transition type, recovery rate, and recovery time.
The signed recovered-ukeire gain prevents a reversal from receiving a shape
bonus when its recovered hand is no better than the best current shape.
With this policy, `first_tenpai` means the first tenpai accepted by the policy,
not an earlier tenpai opportunity that it deliberately broke.

`--policy pure` remains the default. `--policy value` starts all hand-value
weights at `1.0`; `--policy custom` starts them at zero. The complete ordering
is lexicographic:

```text
1. lowest shanten
2. highest base ukeire score + dora_ukeire_weight × retained dora
3. highest weighted tie score
```

The tie score combines retained dora, suited tiles immediately adjacent to the
actual dora, yakuhai value, and tanyao-compatible suited tiles (ranks 2–8).
Yakuhai contributes two units for a double seat/round wind, one for a single
value wind or dragon, and zero otherwise. Aka dora contributes one dora unit;
if a five tile type is discarded, a dora-aware policy discards an ordinary
copy before its red copy. Configure the terms with
`--dora-ukeire-weight`, `--dora-tiebreak-weight`,
`--dora-neighbor-weight`, `--yakuhai-weight`, and `--tanyao-weight`.
These are intentionally shallow heuristics rather than a hand-value estimate,
so they remain cheap enough for Monte Carlo policy comparisons.

Shared-table runs can take a synchronized tile-availability snapshot with
`--tile-availability-turn N`. The engine continues through earlier wins and
tenpai; a hand that has accepted tenpai is locked and tsumogiris until every
player has completed turn `N`. For each observer and tile type, the report uses

```text
U = 4 - own known copies - public copies
E = U * live wall positions / all unseen positions
A = actual live wall copies / E
```

Rows with `E == 0` are skipped. Output includes fixed table-level categories
(`round_wind`, `seat_wind_for_one_player`, `dragons`, `19`, `28`, `37`, `46`,
`5`) and observer-relative categories (`double_wind`, `seat_wind`,
`round_wind`, `guest_wind`, plus the suited and dragon groups). It also reports
each observer seat separately, raw 34-type diagnostics, table-clustered standard
errors, and companion ratios for the hidden dead wall and opponents' hands.
Even trial IDs fit the category factors; odd trial IDs compare both category
systems on the same validation tiles using paired table-level errors.

To estimate discard danger against exactly one riichi opponent:

```sh
/private/tmp/hand-sim-venv/bin/python outputs/analyze_tile_danger_by_turn.py \
  --native-binary /private/tmp/hand-efficiency-build/hand_efficiency_sim_cpp \
  --trials 20000 --seed 20260809 \
  --output outputs/tile_danger_after_riichi_by_turn_20k.json \
  --csv outputs/tile_danger_after_riichi_by_turn_20k.csv
```

This runner defaults to the `extremeA` value policy: future weights 2.0 at
one shanten and 1.0 at two shanten, dora ukeire weight 1.0, and one-step
reversal with penalty 25. The native mode is `simulate-table --tile-danger`.
It starts observing after the first riichi discard and stops at a second
riichi declaration, target ron/tsumo, defender tsumo, or exhaustion. A second
declaration discard is still tested against the first riichi player before
stopping. This experiment permits a declaration only with at least four live
tiles remaining, following the convention in the
[USPML rules reference](https://uspml.com/documents/uspml_reference.pdf).

The primary denominator counts each distinct tile type in the defender's
14-tile hand once per decision; hits are actual target winning tiles with
full-wait permanent furiten applied. All target hands, including furiten
hands, remain in this denominator. The chosen policy discard has a separate
diagnostic denominator. Errors are clustered by independent table, so
repeated decisions from one hand are not treated as independent trials.

The primary turn axis is the defender's own draw number. The output also
groups by the riichi player's current own turn and by the number of that
player's turns elapsed since declaration. Suji uses only the target's river
(including the declaration tile) and discards that safely passed after
riichi. Other players' earlier discards are not automatically safe. Candidate
classification happens before its discard. `genbutsu` takes priority;
otherwise a suited outer tile uses its one rank-3 anchor, and a middle
4/5/6 is `double_suji`, `half_suji`, or `no_suji` according to whether two,
one, or zero same-suit anchors are safe. Honors are separate. Suji is a
visible feature, not a guarantee against every wait shape.

This baseline keeps the other players attacking; it does not enable cheap
bad-wait rejection, valuable dama, folding, calls, or non-target dama ron.
The target always claims an available win, so there is no voluntary-pass
furiten. Later-turn cells describe hands that survived the stopping rules.
The JSON/CSV contain per-turn category and exact-tile counts; sparse cells
need their denominators and standard errors checked before use.
The [20,000-table danger results](outputs/tile_danger_after_riichi_by_turn_20k.md)
include same-rank suji comparisons, denominators, and reproduction details.

To continue through multiple riichi declarations with defensive play:

```sh
/private/tmp/hand-sim-venv/bin/python outputs/run_push_fold_experiments.py \
  --native-binary /private/tmp/hand-efficiency-build/hand_efficiency_sim_cpp \
  --trials 20000 --seed 20260809 --mode mixed \
  --safety-strategy mean_safety \
  --output /private/tmp/mixed.json --csv /private/tmp/mixed.csv \
  --trial-metrics /private/tmp/mixed.trials.jsonl
```

This separate `simulate-table --push-fold --score-service` mode plays the
closed table to ron, tsumo, or exhaustion. It checks all opponents for legal
ron, including dama, and permits multiple ron. The Python callback supplies
exact tenpai values; ordinary native discard evaluation does not call the
scorer. The runner applies actual dealer-aware payments, riichi deposits,
and exhaustive-draw transfers. Unclaimed deposits remain in a conserved pot.
Realized riichi wins include ura; ippatsu and honba are excluded. Decision
thresholds exclude ura and ippatsu. All players accept legal wins, so no
voluntary-pass furiten event occurs under these policies.

`--mode all_push` keeps attacking, `--mode full_fold` folds unlocked hands
after an opponent declares, and `--mode mixed` uses the
[specified state machine](outputs/push_fold_policy_plan.md). It evaluates
the best post-discard shanten **after drawing**. Against one riichi,
two-shanten hands fold and one-shanten hands half-push; good or medium-value
bad tenpai pushes, while cheap bad tenpai half-pushes. Against two riichi,
only good or sufficiently valuable tenpai pushes. Riichi and haneman-dama
locks retain their required tsumogiri behavior. Once folding starts it is
sticky; half-push failure folds immediately.

Safety ranks follow the requested preset. Known dora and red fives worsen a
non-genbutsu rank by `--dora-danger-steps` (default 1), including beyond the
worst base rank. Equal ranks prefer more own-plus-public copies. For multiple
opponents, `mean_safety` minimizes the mean rank, then the worst rank;
`minimax_safety` reverses those priorities. Half-push requires the best
achievable shanten and a rank no worse than `--half-push-max-tier` (default 5,
suji 3/7) against every declared opponent. The cheap threshold is
`--cheap-max-total-han 1`; the two-riichi bad-wait threshold is
`--two-riichi-bad-wait-min-ron 7700`, using the minimum actual dealer-aware
ron value across unseen winning copies. The prior dama guards use
`--valuable-dama-points 5200|8000` in non-dealer-normalized points.

The result separates distinct-held-type danger from chosen-discard danger
for all 34 tiles, by own turn and 1/2/3 declared opponents. It also retains
physical red/ordinary safety cells with own/public counts, per-seat outcomes,
declaration waits, unseen/live winning copies, and table-clustered errors.
The raw JSONL contains individual decisions. Use `--observations` to retain
it; otherwise the runner removes its temporary stream after aggregation.
`--input` reaggregates a saved stream using its recorded policy metadata.

The independent `--kanchan-outer-wait-weight` (default 0) adds a value
tie-break bonus only when a complete tenpai hand has exactly one structural
winning tile type, that tile is a suited 2 or 8, and it completes a kanchan.
It preserves shanten and ukeire priority and does not relabel that wait as
good in the defensive policy. Cross this option with `--tanyao-weight 0|1`
in the existing single-riichi runner to test the two hypotheses separately.

Use `outputs/summarize_push_fold_comparison.py` with repeated
`--input NAME=aggregate.json` and `--metrics NAME=trials.jsonl` arguments,
`--output comparison.md`, and optional `--json comparison.json` for paired
comparisons. Every seat uses the named policy; table outcome differences
therefore do not estimate one player's advantage against fixed opponents.

The [completed push/fold experiments](outputs/push_fold_experiments/20260909/README.md)
contain four paired 20,000-table policy runs, the four-way tanyao/kanchan danger
comparison, exact first-tenpai value companions, and validation evidence.

All counts subtract the concealed hand, river, and dora indicator. Exact tile-type ties use a seeded random choice, and closed hands declare riichi after their first legal tenpai discard. Trial seeds depend only on the base seed and trial ID, so `--start-trial` can split a run into reproducible, mergeable shards. The ordinary first-tenpai summary exposes wait signatures, acceptance-copy shares, per-tile wait probabilities (with permanent-furiten hands excluded from that legacy ron-wait denominator), realized tsumo and counterfactual ron score means, and their raw numerators. Those first-tenpai frequencies differ from the observer-conditioned danger experiment above. Trial JSONL retains the complete tenpai and win observations; the Python engine's `--include-decisions` option also records all three ukeire-score components.

For a fixed seed, all native backends emit identical trial observations. On the
development machine, the full advanced policy ran about 308 trials/second with
direct masks, 94 with Nyanten, and 15 with fallback C++; throughput depends on
compiler, CPU, and stopping mode. Setting both look-ahead weights to zero is
much faster because it skips all second-ply branches.

In a 100,000-hand native `value`-policy run (seed `20260813`, penalty 25),
reversal increased average unseen-copy-weighted non-dealer riichi ron from
5,316.2 to 5,368.9. Tenpai rate fell from 71.40% to 70.21%, while average
unseen winning copies rose from 5.740 to 5.802. Exact live-wall winning copies fell from 1.292 to 1.180 and
furiten became more frequent, exposing the delay cost that visible ukeire alone
does not capture. The reversal run selected 53,152 reversals and took 453.41
seconds including Python scoring. Treat the default as an experimental policy
rather than a recommendation.

The [2026-09-09 rerun summary](outputs/reruns/20260909/README.md) contains the
refreshed 100,000-hand reference and five 20,000-table continuation experiments,
with exact commands and comparisons. The reference ron mean remains 5,368.9;
the reference wait tables use nine categories, and post-rejection dama ron and
tsumo are counted separately. Original aggregates remain available for comparison.

`DiscardEvaluator.rank(...)` is the Python extension point for value-, safety-, or search-based policies. `HandValueHeuristicEvaluator` exposes the built-in value features as ordinary constructor weights, making programmatic policy grids straightforward. `ScoreEvaluator.score(...)` is independent; the included `MahjongRepositoryScorer` adapter uses `mahjong==2.0.0`. The Python hot shanten path uses RiichiEnv's Rust lookup tables, pinned at `riichienv==0.4.8`; the default native build uses direct necessary/discard masks, with Nyanten and built-in fallbacks available. Detailed analysis cross-checks shanten against MahjongRepository. The one-player mode treats opponents as hidden tile consumers; shared-table modes simulate their decisions explicitly. Both remain closed-hand models.

The local EV model uses explicit wait templates; the hand generator evaluates complete closed hands and can continue a shared table to its terminal outcome. Neither models calls or full-game placement incentives. Results remain conditional on the specified player behavior and sampling rules.
