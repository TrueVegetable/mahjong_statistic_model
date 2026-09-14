# Proposed closed-hand push/fold extension

Status: specification implemented on 2026-09-09; paired production experiments
are recorded under `push_fold_experiments/20260909`. The shanten timing,
medium bad-wait action, and two-threat strategy variants below incorporate
the user's corrections. This document preserves the policy assumptions and
experimental rationale.

## Objective and experiment boundary

Extend the shared physical table so defensive choices can affect the four
research outputs: hand value, waits, discard danger, and wait quality. Preserve
the existing single-riichi stopping mode and its saved baseline. Add a separate
mode that continues after further riichi declarations and records observations
by the number of declared opponents, excluding the observer itself.

The existing baseline covers all 34 exact tile identities. The expanded
[readout](tile_danger_after_riichi_by_turn_20k.md) displays all rank groups;
the [CSV](tile_danger_after_riichi_by_turn_20k.csv) retains exact tiles and turns.
New defensive policies will change both candidate holdings and surviving
tables, so their results must be regenerated and labeled separately.

## Discard rules

Use the requested configurable ordering, numbered from safest to riskiest:

0. Genbutsu.
1. Suji 1/9.
2. Honors.
3. Double-suji 4/5/6.
4. Suji 2/8.
5. Suji 3/7.
6. No-suji 1/9.
7. Half-suji 4/5/6.
8. No-suji 2/8.
9. No-suji 3/7.
10. No-suji 4/5/6.

For each opponent separately, construct genbutsu and suji only from visible
information, with a separate post-declaration timeline for each opponent.
Earlier discards by other players do not automatically become safe against a
new riichi declarer.

Suggested parameter `dora_danger_steps=1` worsens a non-genbutsu candidate's
tier. Keep the arithmetic score above 10 if necessary, so penalized tiles in
the riskiest tier remain distinguishable. Known indicator dora and red fives
qualify; hidden ura does not. Initially apply one configured penalty when a
tile has any known dora bonus, without stacking multiple bonuses. An indicator
tile is not itself penalized merely for being the indicator.

Within equal adjusted tiers, prefer more known copies: own holdings plus all
public copies, including the visible indicator, with no double counting.
Record own and public counts separately for later analysis. More copies is a
tie-break heuristic, not proof of safety. Resolve remaining ties by seeded
policy RNG. Evaluate red and ordinary physical copies separately when their
penalties differ, while keeping the 34-type danger denominator explicit.

- **Full fold:** choose the safest adjusted candidate, regardless of shanten.
- **Half push:** require adjusted safety no worse than suji 3/7, then use the
  existing ukeire/value ordering within the eligible set. After the draw,
  compute the minimum shanten achievable by a legal discard before applying
  safety filtering. Half-push candidates must attain that minimum; preserving
  only the pre-draw shanten is insufficient. A draw that enables tenpai enters
  the tenpai branch, even if the hand was one shanten before drawing.
  If no eligible tile meets the condition, fold on that decision immediately.
  Disable voluntary shanten reversal in half-push mode.
- **Full push:** preserve tenpai without a safety filter. Already-riichi hands
  tsumogiri except when winning. Dama hands retain the existing permitted
  tenpai-preserving improvements and yaku/value guards. Literal tsumogiri for
  every dama hand would remove those existing improvements; keep it only for
  states that already require it, including the haneman lock.

The riichi lock takes precedence over all later push/fold decisions. A player
who has declared riichi cannot decide to fold when a second opponent declares.
This is a rule constraint, not a policy preference; see the
[USPML rules reference](https://uspml.com/documents/uspml_reference.pdf).

## State and value decisions

Check a legal win first. For a player not already in riichi, choose a defensive
mode before deciding whether to declare. Half-pushing and folding players do
not newly declare riichi, since that would remove their ability to fold.
Full-pushing players use the previous riichi/dama selection rules.

Hand state in the table below is evaluated after drawing: check for a win,
then use the best achievable post-discard shanten. For tenpai candidates,
evaluate wait quality and value on the resulting thirteen-tile hand. Keep
those assessments attached to the candidate discard; a good wait available
through one discard does not license selecting a different, nonqualifying
wait against two opponents. Safety filtering must not move the decision back
into the pre-draw one-shanten branch. Already-riichi and persistent-fold states
retain their previously specified precedence.

| Opponent riichi count | Hand state | Proposed action |
|---|---|---|
| 0 | Any | Existing policy |
| 1 | Two shanten or worse | Full fold |
| 1 | One shanten | Half push |
| 1 | Tenpai, good wait | Full push |
| 1 | Qualifying expensive dama | Full push with existing dama guards |
| 1 | Cheap bad-wait tenpai | Half push; fold immediately if it fails |
| 1 | Medium-value bad-wait tenpai | Full push; retain as a tunable policy choice |
| 2 | Not tenpai | Full fold |
| 2 | Tenpai, good wait or qualifying expensive bad wait | Full push |
| 2 | Other tenpai | Full fold |
| 3 | Not already in riichi | Full fold as a conservative initial default |

Reevaluate an unlocked, non-folding hand after every draw. Half push can become
full push on good, medium-value bad, or qualifying expensive tenpai, subject
to the opponent count and existing declaration rules. Full folding is sticky while
opponent riichi remains active, but always takes a legal win. Test reentry as
a separate option later; otherwise the policy can oscillate between modes.

“Good wait” needs an explicit table over all nine whole-hand wait categories.
Reuse the existing favorable sequence shapes and live-honor exceptions, but
do not equate “not rejected by the 1300-point rule” with “good wait”: that rule
has exemptions unrelated to wait quality. Explicitly classify seven-pairs,
kokushi, and other/compound waits. A structurally good wait with no unseen
winning copies must not qualify automatically. Furiten and very low remaining
wait counts should be recorded as sensitivity dimensions rather than silently
treated as ordinary healthy tenpai.

Suggested cheap threshold is `cheap_max_total_han=1`, including known dora and
the riichi han in the projected riichi score, excluding ura and ippatsu. Keep
this separate from the existing exact-1300, non-dealer rejection rule. For
unequal waits, initially call a hand cheap only when all unseen winning copies
are at or below the threshold. Compare thresholds 1 and 2 after a pilot.

For two opponents, use `two_riichi_bad_wait_min_ron=7700` in actual scored ron
points, including the player's dealer status and known dora. Score the intended
riichi or dama action consistently; do not add riichi to a hand staying dama.
Exclude speculative ura/ippatsu and initially exclude honba/deposits from this
intrinsic hand-value threshold. Use the minimum over unseen winning copies
as a conservative new push criterion; record the unseen-copy-weighted mean
alongside it. This is a new criterion and does not change the existing dama
qualification semantics. The existing value helper deliberately normalizes to
non-dealer value, so it cannot be reused unchanged for the 7700 rule.

## Sanity checks on the assumptions

The proposed ordering is an interpretable baseline, not an empirical law.
At observer turn 12 in the current unfiltered sample, suji 2/8 had 2.96%
candidate danger, compared with 4.54% for double-suji 4/6. Half-suji 4/6 had
10.15%, compared with 9.34% for no-suji 2/8. Honors also depend heavily on own
and public copy counts. Preserve the requested order as a named preset and
compare alternatives rather than claiming the ordering has been validated.

The user reports the opposite ordering in empirical data for double-suji 4/6
versus suji 2/8, and half-suji 4/6 versus no-suji 2/8. Treat this as a model
diagnostic, rather than changing the safety preset to fit the current
simulation. Two policy hypotheses should be separated:

- The current tanyao term rewards each retained suited 2–8 tile regardless of
  whether a tanyao hand is realistically available. It acts in the value tie
  score after shanten and ukeire, not as a shanten override. Removing it with
  `--tanyao-weight 0` is therefore a clean first ablation, with all other
  weights and rules preserved.
- There is no explicit preference between pure kanchan waits on 2/8 and those
  on 4/5/6. Add a separately configurable tenpai tie-break preference for the
  former, using whole-hand wait classification and retaining shanten/ukeire
  priority. This should not automatically relabel those kanchan as good waits
  in the defensive state machine. Avoid double-counting a compound wait or
  applying a kanchan preference to every winning tile independently.

First compare the unchanged baseline against tanyao weight zero using paired
seeds in the existing single-riichi mode. Then use a 2-by-2 comparison: tanyao
on/off crossed with the new kanchan preference on/off. Report the two danger
ordering contrasts, kanchan winning-rank shares, overall wait distribution,
hand value, and copy-conditioned danger where supported. This separates the
effects instead of attributing a combined change to one rule. Check empirical
definitions of turn, suji anchors, declaration-tile suji, and candidate versus
actual-discard sampling before interpreting a remaining discrepancy.

For two threats, an arithmetic mean of tier numbers is only an ordinal
heuristic. Safe against one and very dangerous against the other can look
misleadingly moderate. Implement both accepted strategies as explicit presets:
`mean_safety` minimizes mean adjusted tier, breaking ties by the worst tier;
`minimax_safety` minimizes the worst adjusted tier, breaking ties by the mean.
Both then use the same known-copy tie-break and seeded RNG. Initially these
are heuristic tier scores, not calibrated danger probabilities.
A later probability-based selector can minimize summed expected deal-in loss;
it needs calibrated probabilities and opponent values rather than tier IDs.
Do not interpret averaged danger as the probability of hitting either player.
Measure the latter directly using the joint hidden hands and furiten states.

The two-shanten fold and one-shanten half-push rules are reasonable starting
assumptions, but they ignore turn, live-wall length, hand value, and how much
ukeire is lost. Log these quantities. Likewise, automatically pushing every
good wait against two riichi is intentionally coarse, especially for furiten,
nearly exhausted waits, or very low-value hands.

## Implementation and experiment sequence

0. **Policy diagnosis before defensive-policy changes.** Run the paired
   tanyao-off ablation in the current single-riichi mode. Add the kanchan-rank
   preference independently and complete the factorial comparison above.
   Preserve the current results and the user's safety-order preset.
1. **Visible safety and deterministic selectors.** Add configurable tiers,
   dora penalties, seen-copy tie breaks, half-push constraints, and per-threat
   safety timelines. Test each rule using fixed physical states.
2. **Complete table progression.** Introduce a separate continuation mode
   with all-player ron/tsumo, temporary and riichi-pass furiten, riichi locks,
   and a stated multiple-ron convention. Proposed default is multiple ron with
   summed payments; record individual hit masks and the union. Calls and full
   game/placement incentives remain outside scope. The current target-only
   ron path is insufficient for evaluating folding policies.
3. **Value-aware state machine.** Extend the cached Python scoring/directive
   interface at tenpai decisions, keeping scoring outside routine native
   discard evaluation. Discover fresh directives for each policy/trajectory;
   old directive files become invalid when defensive discards change the hand.
   Preserve existing valuable-dama transition guards and verify convergence
   of any replay-based directive discovery before a full run.
4. **Verification.** Cover riichi lock precedence, differing safety against
   two opponents, dora at the worst tier, red-copy selection, immediate
   half-push failure, draw-induced one-shanten-to-tenpai transitions, medium
   bad-wait pushing against one opponent, threshold boundaries, all wait categories, unequal waits,
   dealer scoring, full/temporary furiten, and multiple ron. Check physical
   conservation, seeded Python/native fixtures, shard reproducibility, and
   preservation of the existing single-riichi mode.
5. **Paired experiments.** Benchmark a short run first. Compare continued
   all-push, full-fold, and the mixed policy on identical trial seeds, including
   both mean-safety and minimax-safety variants for multiple opponents, then
   select a small set of sensitivity variants for 20,000-table runs. Keep the
   old stop-at-second-riichi result as its own baseline; it is not directly
   comparable to a continued-hand outcome rate.

Report danger for all 34 tiles and rank categories by own turn and 1/2/3
opponent riichi, separately for all candidates and chosen discards. Retain
safe tier, dora status, own/public counts, declared/undeclared observer state,
mode transitions, and the independent table count. Report ron, tsumo, deal-in,
exhaustion, retained tenpai, wait quality, and realized payments. Cluster
uncertainty by table and use paired-table differences for policy comparisons.
Do not call score comparisons complete EV until all modeled payments,
riichi deposits, and exhaustive-draw transfers are included.
