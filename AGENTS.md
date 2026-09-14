# AGENTS.md

## Purpose

This repository supports the future-work section of a riichi-mahjong paper. Its central goal is to replace empirical constants with outputs generated from one coherent hand and table model:

1. average hand value;
2. wait-type and winning-tile distributions;
3. discard danger;
4. wait quality for tsumo probability and riichi-race calculations.

Keep these four outputs as the main story. Hand-selection mechanisms such as future ukeire, one-step shanten reversal, cheap-bad-wait riichi rejection, and valuable dama are explanatory policy components, not the purpose of the project.

The simulator is closed-hand and Monte Carlo oriented. There is no UI requirement. Prefer deterministic, shardable batch interfaces and keep expensive scoring away from the rollout hot path.

## Repository map

- `README.md`: runnable overview and public command reference.
- `outputs/hand_efficiency_sim.cpp`: fast native closed-hand and shared-table simulator.
- `outputs/hand_efficiency_sim.py`: readable Python reference implementation and single-hand analysis interface.
- `outputs/hand_monte_carlo_report.py`: batch wrapper, scoring adapter, shared-table experiments, and aggregate reporting.
- `outputs/analyze_tile_availability_by_turn.py`: turn-6/9/12 availability and discard/own-count analysis.
- `outputs/analyze_tile_danger_by_turn.py`: danger after exactly one riichi, grouped by observer turn and observer-visible safety/suji.
- `outputs/hand_push_fold_native.inc`: closed-table continuation, per-opponent safety, and push/fold state machine.
- `outputs/hand_push_fold_scoring.py`: cached exact tenpai scoring callback and value directives.
- `outputs/run_push_fold_experiments.py`: continuation runner, payment settlement, and joint-danger aggregation.
- `outputs/summarize_push_fold_comparison.py`: paired-table policy comparison.
- `outputs/score_first_tenpai_values.py`: exact physical-copy scoring for the all-seat, non-dealer-normalized first-tenpai companion.
- `outputs/validate_push_fold_results.py`: persisted continuation-result reconciliation and provenance checks.
- `outputs/requirements-hand-simulator.txt`: pinned Python scoring and hand-analysis dependencies.
- `outputs/CMakeLists.txt`: native build, including the optional GPL tile-mask backend.
- `outputs/THIRD_PARTY_HAND_SIMULATOR.md`: dependency revisions and licensing notes.
- `tests/`: deterministic policy, scoring, physical-state, and Python/C++ agreement checks.
- `outputs/hand_efficiency_simulator_report_zh.md`: current Chinese technical report.
- `outputs/statistical-mahjong-the-whys-cn.md`: reference paper and required Chinese prose style.
- `outputs/tile_availability_by_discard_count_20k.csv`: flat turn/category/exact-tile availability table.
- `outputs/tile_availability_turn6_9_12_comparison_20k.json`: detailed availability comparison and diagnostics.

Large JSON files in `outputs/` are experiment results. Treat them as evidence, not fixtures to rewrite casually.

## Working-tree warning

The working tree contains many intentional untracked and modified files. They are project work, not disposable build debris.

- Do not reset, clean, checkout, or delete unrelated files.
- Inspect `git status --short` before editing.
- Use `apply_patch` for source and document edits.
- Put disposable builds and raw trial streams under `/private/tmp` unless durable retention is requested.
- Do not claim that a result is committed merely because it exists in the workspace.

## Core policy model

Discard selection is lexicographic:

1. minimize post-discard shanten;
2. maximize ukeire score;
3. apply value and shape tie-breakers.

For one- and two-shanten hands:

```text
ukeire score = immediate ukeire
              + future_ukeire_weight(shanten)
                * copy-weighted ukeire after one accepted draw
              + improvement_weight * improvement-tile copies
              + dora_ukeire_weight * retained dora
```

The current strong policy, called `extremeA` in result filenames, uses:

- one-shanten future-ukeire weight: `2.0`;
- two-shanten future-ukeire weight: `1.0`;
- three-shanten-or-earlier future lookahead: disabled;
- fallback future-ukeire weight: `0.5`;
- improvement-tile weight: `0.1`;
- value-policy dora/yakuhai/tanyao weights: `1.0` unless an experiment overrides them.

Shanten always takes priority over ukeire unless the optional one-step reversal rule is enabled.

### One-step shanten reversal

The optional reversal policy permits only `0 -> 1` and `1 -> 2`, through own turn 9, normally once per hand. It deep-evaluates at most two candidates and never recursively reverses during hypothetical recovery.

```text
reversal score = recovery copies
                 + 0.5 * (copy-weighted ukeire after recovery
                          - best current immediate ukeire)
                 + dora ukeire bonus
                 - 25
```

The recovered ukeire must strictly improve on current immediate ukeire, and the adjusted reversal score must strictly exceed the best strict-shanten ukeire score. The default penalty is `25`, not `20`.

Do not make reversal the centerpiece of the paper. Explain it only where policy realism and selection effects are discussed.

## Wait classification invariants

Wait categories are mutually exclusive and assigned from the full 13-tile structure and complete winning-tile set in this priority:

1. `sanmenchan`;
2. `ryanman`;
3. `kanchan`;
4. `penchan`;
5. `shanpon`;
6. `tanki`;
7. `chiitoitsu_tanki`;
8. `kokushi`;
9. `other`.

Do not classify each winning tile independently and then double count a hand. Ambiguous shapes such as `1233` and `3345` must be resolved using whole-hand structure and the priority rules. `other` is additionally separated by the number of distinct winning tile types.

`sanmenchan` requires three same-suit suji winning tiles and the removable five-tile run between the outer wins, such as `23456` waiting on `1/4/7`.

For the cheap-bad-wait rule, an honor-containing shanpon counts as good only if at least one copy of the honor winning tile remains unseen.

## Scoring conventions

- Primary reported hand value is non-dealer ron unless a result explicitly says otherwise.
- Riichi and a separately sampled ura-dora indicator are included in riichi scoring.
- Ura dora is hidden from discard decisions.
- Unequal waits are averaged by visible-information unseen winning-copy count, not exact live-wall count.
- Keep ron and tsumo statistics separate. Do not derive tsumo payment mechanically from the ron mean.
- The pinned Python `mahjong==2.0.0` scorer may be used after native rollout observations are generated.

The current 100,000-hand reference result is `outputs/reruns/20260909/hand_monte_carlo_value_reversal25_wait_matrix_100k.json`. It reports a non-dealer, no-ippatsu riichi-ron mean of about `5368.9`, close to the paper's empirical `5300` value. Its future weights are `0.5/0.5`, not the `extremeA` weights. Preserve the exact experimental definition when comparing these numbers. The original file in `outputs/` is retained with its older eight-category wait grouping; use the dated rerun for the current nine-category distribution.

## Shared four-player selection experiments

Shared-table mode gives four same-policy players one physical table and normally stops at the first policy-accepted tenpai. It is used to measure selection effects that a one-player rollout cannot reproduce.

Important variants:

- `--reject-cheap-bad-wait-riichi`: rejects exactly 1300-point non-dealer bad-wait riichi samples.
- `--continue-rejected-tenpai`: rejected hands stay dama and continue until eligible tenpai, dama win, or exhaustion.
- `--valuable-bad-wait-dama-line 3han40fu|4han40fu`: sufficiently valuable bad waits remain dama and may accept only tenpai-preserving changes that improve the wait or preserve yaku and value.
- `--haneman-or-higher-dama`: if every unseen winning tile satisfies the haneman minimum under the documented ron/tsumo-yaku rules, lock the hand and tsumogiri until ron or tsumo.

Do not weaken these value guards. In particular, never accept a tenpai-preserving change that leaves a bad wait but removes the required yaku or lowers the hand below its qualifying value line.

The latest corresponding result files are the non-`pre_transition_guard` variants named:

- `outputs/reruns/20260909/shared_four_player_extremeA_valuable_dama_3han40fu_haneman_20k.json`;
- `outputs/reruns/20260909/shared_four_player_extremeA_valuable_dama_4han40fu_haneman_20k.json`.

The dated reruns preserve outcomes and wait statistics while correcting the post-rejection ron/tsumo split. `dama_tsumo_after_rejection` and `dama_ron_after_rejection` count table endings after any cheap-hand rejection; they need not involve the player whose hand was rejected. Original aggregates remain in `outputs/`; reproducibility commands and comparisons are in `outputs/reruns/20260909/README.md`.

## Tile-availability model

This metric estimates where unseen copies are located. It is intended for wait quality, tsumo probability, and riichi-race win rate. It is not by itself discard danger.

For observer `p` and tile type `t`:

```text
U = 4 - own known copies - all public copies
E_live = U * remaining live-wall positions / all unseen positions
A_live = actual live-wall copies / E_live
```

Skip cells with `E_live == 0`. Companion opponent-hand and hidden-dead-wall ratios are useful diagnostics.

### Current checkpoints

The synchronized shared-table experiment sampled turns 6, 9, and 12. Each checkpoint has 20,000 paired tables and 80,000 player views, using seed `20260809` and the `extremeA` value policy with dora weighting and reversal penalty 25.

The mean absolute deviation of the eight fixed category factors from 1 grows from approximately:

```text
turn 6:  0.162
turn 9:  0.211
turn 12: 0.267
```

Thus turn 6 is closer to random allocation, while turn 12 is much more conditioned by prior discards and retained hand structure.

### Use discarded copies and own copies as separate dimensions

Do not infer the availability cell from the wait label. The primary lookup for each tile is:

```text
A[turn][tile category or exact tile][discarded copies][own copies]
```

- `discarded copies` means copies in all four rivers.
- Subtract the visible dora indicator from public counts when computing the river-discard dimension.
- The dora indicator still reduces `U`, because it is visibly unavailable.
- `own copies` means the observer's actual total holding of that tile type.
- Wait type supplies only the set of winning tiles. Each winning tile uses its own `(discarded, own)` lookup cell.
- If all four copies are already accounted for, the winning-copy contribution is zero.

This two-dimensional representation is useful even when own copies are zero. Such cells commonly describe sequence-completing or improvement tiles. They are not tanki cells.

Critical correction: a standard honor tanki and a seven-pairs honor tanki both hold one copy of the winning tile. An honor shanpon holds two. A tanki `2 -> 3 discarded copies` transition cannot remain a wait because the third discard exhausts the last winning copy. Earlier conversational labels that called own-count zero “ordinary tanki” were wrong; the saved report has been corrected. The underlying counts were not wrong.

At turn 12, for honors with one own copy, average live-wall copies are:

```text
0 discarded: 0.814
1 discarded: 0.895
2 discarded: 0.583
3 discarded: 0.000
```

The `0 -> 1` increase is a conditional selection effect: seeing one discard weakens the signal that the other copies are grouped in an opponent's hand enough to overcome losing one candidate copy. Treat this as descriptive under the simulated policy, not as a causal law of human play.

Use `outputs/tile_availability_by_discard_count_20k.csv` for the flat data. It contains all 34 exact tile identities, category aggregates, all three turns, exact river-discard count, own-copy count, sample size, mean unseen copies, mean live-wall copies, live-wall factor, opponent copies, and opponent-hand factor.

Suggested production lookup:

1. exact tile, turn, discarded count, and own count when the cell is sufficiently large;
2. tile-category cell at the same turn and `(discarded, own)` coordinates;
3. interpolation between turns 6, 9, and 12;
4. unadjusted random allocation when no supported cell exists.

Do not extrapolate freely beyond the sampled turns. Clamp to the nearest measured turn until later checkpoints are simulated.

The raw JSONL checkpoint streams used to build the comparison were temporary files under `/private/tmp` and may not survive a new environment. The aggregate JSON and CSV are durable. Regenerate raw streams before changing the grouping or computing a new conditional statistic that is absent from those aggregates.

## Single-riichi tile danger

Native `simulate-table --tile-danger` observes defenders only after the first
riichi discard and stops at the second declaration, target ron/tsumo, defender
tsumo, or exhaustion. A declaration requires at least four live tiles remaining
in this experiment. The second declaration discard is still tested for ron
against the first riichi player before stopping.

- Primary grain: one distinct held tile type per defender's 14-tile decision.
  The policy-chosen discard has a separate diagnostic denominator.
- Primary turn: observer's own draw number. Secondary axes are target's current
  own turn and target's current turn minus its declaration turn.
- Safe types: target's entire river plus all safely passed post-declaration
  discards. Other players' pre-riichi discards are not automatically safe.
  Classify before the candidate discard; never infer safety from hidden waits.
- `genbutsu` takes priority, honors are separate, outer suited tiles use one
  same-suit rank-three anchor, and middle 4/5/6 require both anchors for
  `double_suji`; one anchor is `half_suji`.
- Truth label: candidate completes the locked target hand and the target is
  not furiten on any winning tile in its river. Include furiten targets in
  the denominator. The target never voluntarily passes a win.
- Cluster uncertainty by table, accumulating all decisions in a table before
  computing cluster moments. Include zero-exposure tables in the independent
  table count; also report the number of exposed tables and decisions.
- Baseline policy is unfiltered `extremeA` with reversal penalty 25. Other
  players keep attacking. No folding, calls, cheap-bad-wait rejection,
  valuable-dama filtering, or non-target dama ron is modeled in this mode.

Do not substitute these probabilities for danger against multiple riichi or
open opponents. Later-turn cells are conditional on surviving the stopping
rules; category mixtures and actual-discard rates have different denominators.

The first 20,000-table result is `outputs/tile_danger_after_riichi_by_turn_20k.json`
with companion CSV, Markdown readout, and validation manifest. It contains
148,466 decisions and 1,584,686 candidate observations from 19,763 exposed
tables. At observer turn 12, held non-genbutsu 4/6 tiles have candidate hit
rates 13.81% (no suji), 10.15% (half suji), and 4.54% (double suji).

## Closed-table push/fold continuation

`simulate-table --push-fold --score-service` is separate from the single-riichi
stopping mode. Run it through `outputs/run_push_fold_experiments.py`; the
interactive native callback needs exact values from the pinned Python scorer.
The accepted specification is `outputs/push_fold_policy_plan.md`.

- Modes are `all_push`, `full_fold`, and `mixed`; every seat uses the selected
  policy. Multiple-threat selectors are `mean_safety` and `minimax_safety`.
- Evaluate best achievable post-discard shanten after the draw. Half-push
  first finds that minimum, then filters for adjusted rank at most 5 against
  every declared opponent. Failure immediately enters sticky full fold.
- Preserve the requested eleven safety tiers. Known dora/aka lose a
  configurable rank except genbutsu; do not cap the adjusted rank at 10.
  More own-plus-public copies break rank ties, then seeded RNG.
- Against one riichi: two-shanten-or-worse folds, one-shanten half-pushes;
  good, valuable dama, and medium bad tenpai push; cheap bad tenpai half-pushes.
  Against two: only good tenpai or minimum intended-action ron value at least
  7700 actual dealer-aware points pushes. Unlocked hands fold against three.
- Already-riichi hands and haneman-dama locks tsumogiri. Preserve existing
  valuable-dama transition guards. Full folding remains sticky and accepts
  legal wins. No-threat hands retain the configured shanten reversal policy;
  half-push and threatened tenpai do not voluntarily reverse.
- New cheap classification means all unseen projected-riichi winning copies
  have total han at most 1 by default. It is separate from the prior exact
  1300-point non-dealer rejection. Value guards use 5200/8000 normalized
  non-dealer points; the two-riichi threshold uses actual dealer-aware points.
- Ron checks all opponents, full-wait permanent furiten, and the temporary
  furiten state. Every legal win is accepted, so no voluntary pass sets
  temporary or riichi-pass furiten under the available policies. Houtei can
  supply yaku on the last discard. Multiple ron sums payments and awards the
  riichi pot to the winner closest after the discarder.
- Terminal scoring includes actual ura for riichi, haitei/houtei, dealer
  payments, riichi deposits, and exhaustive-draw transfers. Ippatsu and honba
  are explicitly excluded. Exhaustion carries unclaimed sticks separately;
  player point deltas plus this pot must sum to zero.
- Primary danger counts each distinct held type once per decision; red and
  ordinary copies have a separate physical-conditioned diagnostic. Report
  declared-opponent union hits separately from any-opponent hits. Both held
  candidates and chosen discards have table-clustered uncertainty.
- `--kanchan-outer-wait-weight` defaults to 0. A nonzero value adds only a
  value tie-break for a complete single-type kanchan wait on suited 2/8;
  compound waits do not qualify. This does not make those waits good for
  push/fold. Preserve byte-identical legacy output when its weight is zero.

The completed four-policy 20,000-table comparison and factorial diagnostics
are under `outputs/push_fold_experiments/20260909/`; start with its `README.md`.
The controls are `all_push` and `full_fold`, with `mixed_mean` and
`mixed_minimax` as the accepted defensive presets. These results use paired
trial IDs with all four seats adopting the same policy, not a focal-player
best-response experiment. The original single-riichi baseline is retained.
The two policy ablations narrow the disputed turn-12 danger gaps but do not
reverse their point-estimate ordering.

Shared-table `hand_monte_carlo_report.py` intentionally leaves value scoring
disabled because its sample includes East-seat hands. For the factorial
companion, `score_first_tenpai_values.py` preserves actual seat-wind yaku and
reprices every seat to non-dealer ron; it includes sampled ura and excludes
ippatsu. Use the `first_tenpai_value_*` files for those scores, not the
unscored fields of the `first_tenpai_*` rollout reports. This first-tenpai
sample differs from the first-legal-riichi danger sample's four-live-tile
condition.

## Report-writing rules

The Chinese report is part of the paper's future-work material, not a development diary.

- Match the tone and terminology of `outputs/statistical-mahjong-the-whys-cn.md`.
- Use established Chinese mahjong terms rather than Japanese kanji or unexplained Japanese romanization.
- Remove language that narrates implementation history, coding steps, or the development process.
- Lead with how the simulator can replace empirical average-value, wait-distribution, and danger inputs.
- Explain the composition of ukeire score clearly in the main model section.
- Keep future-ukeire weighting, shanten reversal, one-player versus four-player selection, cheap bad-wait rejection, and valuable dama together in the policy-realism/experimental-effects section.
- State limitations precisely: current core experiments are closed-hand and do not yet replace a complete four-player game or a fully estimated discard-danger table.
- When changing numerical claims, update both the nearby table and every summary/conclusion that repeats the claim.
- If available, use the `humanizer` skill as a final prose pass, but do not trade away technical precision.

## Setup, build, and verification

Create an isolated Python environment and install the pinned requirements:

```sh
python3 -m venv /private/tmp/hand-sim-venv
/private/tmp/hand-sim-venv/bin/pip install -r outputs/requirements-hand-simulator.txt
```

Build the fast native engine:

```sh
cmake -S outputs -B /private/tmp/hand-efficiency-build -DCMAKE_BUILD_TYPE=Release
cmake --build /private/tmp/hand-efficiency-build --parallel
```

The default CMake build fetches and links GPL-3.0 tile-mask code. Network permission may be required. For a dependency-free fallback:

```sh
c++ -std=c++17 -O3 outputs/hand_efficiency_sim.cpp -o /private/tmp/hand_efficiency_sim_cpp
```

Minimum verification after code changes:

```sh
/private/tmp/hand-sim-venv/bin/python -m unittest discover -s tests -p 'test_*.py'
/private/tmp/hand-efficiency-build/hand_efficiency_sim_cpp --self-test
git diff --check
```

The last complete Python run passed 129 tests, and the production native self-test passed. Do not report success after running tests without the pinned Python dependencies; missing `mahjong` produces many false setup errors.

For changes affecting native/Python policy behavior, also require fixed-fixture agreement and a seeded short simulation comparison. For changes affecting availability, verify physical conservation for every observer and tile:

```text
live wall + hidden dead wall + opponents' hands = unseen copies
```

For each checkpoint, also verify total unseen positions:

```text
turn 6:  46 live + 13 dead + 39 opponents = 98
turn 9:  34 live + 13 dead + 39 opponents = 86
turn 12: 22 live + 13 dead + 39 opponents = 74
```

## Reproducibility and experiment discipline

- Trial seeds depend only on base seed and trial ID. Preserve this property so shards remain mergeable.
- Use paired seeds when comparing policies.
- Record the policy name, every non-default weight, seed, trial range, engine/backend, stopping condition, and scoring convention in result metadata.
- Keep hypothesis-generating runs separate from confirmatory full runs.
- Do not compare a one-player first-tenpai sample directly with a four-player race sample without naming the selection difference.
- Do not mix exact live-wall weighting with visible-information unseen-copy weighting in the same headline statistic.
- Report denominators and skipped cells, especially for rare `(discarded, own)` combinations.
- Treat high-discard-count exact-tile cells as sparse until their sample sizes are checked.

## Known boundaries and likely next work

- The single-riichi danger mode conditions held candidate discards on observer-visible state and counts actual target ron hits with furiten. A complete empirical replacement still requires additional opponent policies, folding, calls, and joint danger against multiple opponents.
- Calls, open-hand yaku, opponent strategy heterogeneity, and full-game score incentives remain outside the current closed-hand core. Closed-table folding is available in the separate push/fold mode.
- Tile-availability estimates should eventually be cross-validated across policy families, not only the `extremeA` value policy.
- If extending availability, useful next checks include more turn checkpoints, smoothing sparse two-dimensional cells, and testing whether exact tile identity materially improves held-out error over the current category system.
- Before another large experiment, benchmark a seeded short run and estimate total runtime. Keep scoring post-processing separate unless the experimental stopping rule requires value during rollout.

When uncertain, preserve the existing semantics and add a paired diagnostic before changing the headline model.
