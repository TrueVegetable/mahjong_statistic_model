#!/usr/bin/env python3
"""Run closed-hand Monte Carlo trials and summarize value and wait statistics.

The native engine is selected automatically when a built binary is available;
otherwise the Python reference engine is used.  Every rollout stops at first
tenpai.  All remaining winning-tile copies are then scored as non-dealer ron;
unequal waits are averaged within each hand by remaining-copy count.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO


OUTPUTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUTS_DIR.parent
sys.path.insert(0, str(OUTPUTS_DIR))

import hand_efficiency_sim as sim  # noqa: E402


WAIT_CATEGORIES = (
    "sanmenchan",
    "ryanmen",
    "kanchan",
    "penchan",
    "shanpon",
    "tanki",
    "chiitoitsu_tanki",
    "kokushi",
    "other",
)
TILE_TYPE_CATEGORIES = ("19", "28", "37", "456", "honor")
# Categories used by the optional shared-table tile-availability experiment.
# Keep these separate from TILE_TYPE_CATEGORIES: the existing report groups
# 4/5/6 together, whereas availability is intended to distinguish 46 from 5.
AVAILABILITY_TABLE_CATEGORIES = (
    "round_wind",
    "seat_wind_for_one_player",
    "dragons",
    "19",
    "28",
    "37",
    "46",
    "5",
)
AVAILABILITY_PLAYER_CATEGORIES = (
    "double_wind",
    "seat_wind",
    "round_wind",
    "guest_wind",
    "dragons",
    "19",
    "28",
    "37",
    "46",
    "5",
)
AVAILABILITY_ALL_CATEGORIES = tuple(
    dict.fromkeys((*AVAILABILITY_TABLE_CATEGORIES, *AVAILABILITY_PLAYER_CATEGORIES))
)
DAMA_LOCK_FLAG = 1 << 63
DAMA_TSUMOGIRI_FLAG = 1 << 62
DAMA_REJECT_TRANSITION_FLAG = 1 << 61
DAMA_WAIT_MASK = (1 << sim.TILE_TYPE_COUNT) - 1
WIND_NAMES = ("east", "south", "west", "north")
DIRECTION_TO_TILE_TYPE = {seat: 27 + index for index, seat in enumerate(WIND_NAMES)}


def _can_remove(counts: Sequence[int], removals: Mapping[int, int]) -> tuple[int, ...] | None:
    remaining = list(counts)
    for tile_type, count in removals.items():
        if not 0 <= tile_type < sim.TILE_TYPE_COUNT or remaining[tile_type] < count:
            return None
        remaining[tile_type] -= count
    return tuple(remaining)


def _is_melds(counts: tuple[int, ...], expected_melds: int) -> bool:
    return any(len(melds) == expected_melds for melds in sim._meld_decompositions(counts))


def _is_melds_and_pair(counts: tuple[int, ...], expected_melds: int) -> bool:
    for pair, count in enumerate(counts):
        if count < 2:
            continue
        remainder = list(counts)
        remainder[pair] -= 2
        if _is_melds(tuple(remainder), expected_melds):
            return True
    return False


def primary_wait_category(
    hand_counts: Sequence[int], wait_types: Iterable[int]
) -> str:
    """Classify a whole tenpai hand once, using defense-oriented priority."""
    counts = tuple(int(count) for count in hand_counts)
    if len(counts) != sim.TILE_TYPE_COUNT or sum(counts) != 13:
        raise ValueError("primary wait classification requires 13 tiles in 34 format")
    waits = tuple(sorted(set(int(tile_type) for tile_type in wait_types)))
    if not waits:
        return "other"

    # Sanmenchan: three same-suit suji wins, supported by all five tiles
    # between the outer wins. Removing that five-tile run leaves two melds
    # and one pair (e.g. 23456 waiting on 1/4/7).
    if len(waits) == 3:
        low, middle, high = waits
        if (
            low < 27
            and middle == low + 3
            and high == low + 6
            and low // 9 == high // 9
        ):
            remainder = _can_remove(counts, {tile: 1 for tile in range(low + 1, high)})
            if remainder is not None and _is_melds_and_pair(remainder, 2):
                return "sanmenchan"

    # Ryanmen: two suji-separated wins supported by the two tiles between them.
    if len(waits) == 2:
        low, high = waits
        if low < 27 and high == low + 3 and low // 9 == high // 9:
            remainder = _can_remove(counts, {low + 1: 1, low + 2: 1})
            if remainder is not None and _is_melds_and_pair(remainder, 3):
                return "ryanmen"

    if len(waits) == 1:
        wait = waits[0]
        if wait < 27:
            rank = wait % 9 + 1
            # Kanchan: remove the suited tiles immediately on both sides.
            if 2 <= rank <= 8:
                remainder = _can_remove(counts, {wait - 1: 1, wait + 1: 1})
                if remainder is not None and _is_melds_and_pair(remainder, 3):
                    return "kanchan"
            # Penchan: 12 waiting on 3, or 89 waiting on 7.
            removals = None
            if rank == 3:
                removals = {wait - 2: 1, wait - 1: 1}
            elif rank == 7:
                removals = {wait + 1: 1, wait + 2: 1}
            if removals is not None:
                remainder = _can_remove(counts, removals)
                if remainder is not None and _is_melds_and_pair(remainder, 3):
                    return "penchan"

    # Shanpon: two pairs plus three complete melds.
    if len(waits) == 2:
        remainder = _can_remove(counts, {waits[0]: 2, waits[1]: 2})
        if remainder is not None and _is_melds(remainder, 3):
            return "shanpon"

    # Tanki: one singleton plus four complete melds.
    if len(waits) == 1:
        remainder = _can_remove(counts, {waits[0]: 1})
        if remainder is not None and _is_melds(remainder, 4):
            return "tanki"

    # Special hands are separate only when no standard wait above has priority.
    for wait in waits:
        complete = list(counts)
        if complete[wait] == 4:
            continue
        complete[wait] += 1
        if sim._is_chiitoitsu_agari(complete):
            return "chiitoitsu_tanki"
    for wait in waits:
        complete = list(counts)
        if complete[wait] == 4:
            continue
        complete[wait] += 1
        if sim._is_kokushi_agari(complete):
            return "kokushi"
    return "other"


def tile_type_category(tile_type: int) -> str:
    """Group a 34-format tile as 19, 28, 37, 456, or honor."""
    if not 0 <= tile_type < sim.TILE_TYPE_COUNT:
        raise ValueError(f"tile type must be 0..33, got {tile_type}")
    if tile_type >= 27:
        return "honor"
    rank = tile_type % 9 + 1
    if rank in (1, 9):
        return "19"
    if rank in (2, 8):
        return "28"
    if rank in (3, 7):
        return "37"
    return "456"


def _distribution(counter: Mapping[Any, float], total: float) -> dict[str, Any]:
    def sort_key(item: tuple[Any, float]) -> tuple[int, float | str]:
        key = item[0]
        if isinstance(key, (int, float)):
            return (0, float(key))
        return (1, str(key))

    return {
        str(key): {"count": count, "share": count / total if total else None}
        for key, count in sorted(counter.items(), key=sort_key)
    }


@dataclass
class MonteCarloReportCollector:
    scorer: sim.ScoreEvaluator
    score_tenpai: bool = True
    reject_cheap_bad_wait_riichi: bool = False
    trials: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    tenpai_hands: int = 0
    tenpai_turn_sum: int = 0
    chiitoitsu_tenpai_hands: int = 0
    furiten_hands: int = 0
    dead_wait_hands: int = 0
    visible_scored_hands: int = 0
    visible_hand_point_sum: float = 0.0
    visible_hand_point_square_sum: float = 0.0
    live_scored_hands: int = 0
    live_hand_point_sum: float = 0.0
    live_hand_point_square_sum: float = 0.0
    visible_han_counts: Counter[int] = field(default_factory=Counter)
    visible_fu_counts: Counter[int] = field(default_factory=Counter)
    visible_han_fu_counts: Counter[str] = field(default_factory=Counter)
    live_han_counts: Counter[int] = field(default_factory=Counter)
    live_fu_counts: Counter[int] = field(default_factory=Counter)
    live_han_fu_counts: Counter[str] = field(default_factory=Counter)
    scoring_errors: Counter[str] = field(default_factory=Counter)
    score_evaluations: int = 0
    invalid_visible_copy_weight: int = 0
    invalid_live_copy_weight: int = 0
    distinct_wait_type_sum: int = 0
    visible_wait_copy_sum: int = 0
    live_wait_copy_sum: int = 0
    live_wall_tile_sum: int = 0
    wait_tile_opportunities: Counter[str] = field(default_factory=Counter)
    wait_visible_copies: Counter[str] = field(default_factory=Counter)
    wait_live_copies: Counter[str] = field(default_factory=Counter)
    eligible_hands_by_wait: Counter[str] = field(default_factory=Counter)
    other_hands_by_wait_type_count: Counter[int] = field(default_factory=Counter)
    next_draw_hit_probability_sum: Counter[str] = field(default_factory=Counter)
    eligible_hands_by_tile_type: Counter[str] = field(default_factory=Counter)
    next_draw_hit_probability_sum_by_tile_type: Counter[str] = field(default_factory=Counter)
    eligible_hands_by_tile_and_wait: Counter[str] = field(default_factory=Counter)
    next_draw_hit_probability_sum_by_tile_and_wait: Counter[str] = field(
        default_factory=Counter
    )
    overall_next_draw_hit_probability_sum: float = 0.0
    first_tenpai_seat_counts: Counter[str] = field(default_factory=Counter)
    cheap_bad_wait_candidates: int = 0
    cheap_bad_wait_rejections: int = 0
    cheap_bad_wait_rejections_by_wait: Counter[str] = field(default_factory=Counter)
    cheap_bad_wait_rejections_by_seat: Counter[str] = field(default_factory=Counter)
    reversal_opportunities: int = 0
    reversals_selected: int = 0
    reversals_recovered: int = 0
    reversal_turn_sum: int = 0
    reversal_recovery_turn_sum: int = 0
    reversal_transitions: Counter[str] = field(default_factory=Counter)
    reversal_recovery_count_sum: float = 0.0
    reversal_recovered_ukeire_sum: float = 0.0
    reversal_current_ukeire_sum: float = 0.0
    reversal_ukeire_gain_sum: float = 0.0
    reversal_strict_score_sum: float = 0.0
    reversal_adjusted_score_sum: float = 0.0
    continuation_tables_with_rejection: int = 0
    continuation_eligible_after_rejection: int = 0
    continuation_tsumo_after_rejection: int = 0
    continuation_ron_after_rejection: int = 0
    continuation_exhausted_after_rejection: int = 0
    continuation_shanten_reversals: int = 0
    valuable_dama_directives: dict[str, int] = field(default_factory=dict)
    valuable_dama_directive_reasons: dict[str, str] = field(default_factory=dict)
    valuable_dama_tables: int = 0
    valuable_dama_ron: int = 0
    valuable_dama_tsumo: int = 0
    haneman_dama_tables: int = 0
    haneman_dama_ron: int = 0
    haneman_dama_tsumo: int = 0

    def add_row(self, row: Mapping[str, Any]) -> None:
        self.trials += 1
        self.outcomes[str(row.get("outcome", "unknown"))] += 1
        self.reversal_opportunities += int(row.get("shanten_reversal_opportunities", 0))
        for reversal in row.get("shanten_reversals", ()):
            self.reversals_selected += 1
            self.reversal_turn_sum += int(reversal["turn"])
            self.reversal_recovery_count_sum += float(reversal.get("recovery_count", 0.0))
            self.reversal_recovered_ukeire_sum += float(
                reversal.get("recovered_ukeire_expected", 0.0)
            )
            self.reversal_current_ukeire_sum += float(reversal.get("current_ukeire", 0.0))
            self.reversal_ukeire_gain_sum += float(reversal.get("ukeire_gain", 0.0))
            self.reversal_strict_score_sum += float(reversal.get("strict_score", 0.0))
            self.reversal_adjusted_score_sum += float(
                reversal.get("adjusted_reversal_score", 0.0)
            )
            transition = f'{int(reversal["from_shanten"])}->{int(reversal["to_shanten"])}'
            self.reversal_transitions[transition] += 1
            if bool(reversal.get("recovered", False)):
                self.reversals_recovered += 1
                self.reversal_recovery_turn_sum += int(reversal["recovery_turns"])
        snapshot = row.get("first_tenpai")
        if snapshot is None:
            return
        if self.reject_cheap_bad_wait_riichi and self._reject_cheap_bad_wait(row, snapshot):
            return
        self._add_tenpai(row, snapshot)

    def add_continuation_row(self, row: Mapping[str, Any]) -> None:
        """Select the first eligible event from a native silent-tenpai trajectory."""
        rejected_events = 0
        reversals = 0
        valuable_seats: set[int] = set()
        active_dama_reasons: dict[int, str] = {}
        haneman_entered = False
        bad_wait_value_entered = False
        for event in row.get("table_events", ()):
            kind = str(event.get("kind", ""))
            seat = int(event.get("seat", -1))
            if kind == "shanten_reversal":
                reversals += 1
                continue
            if kind == "tenpai":
                snapshot = event["tenpai"]
                event_key = (
                    f'{int(row["trial_id"])}:{seat}:{int(event.get("turn", 0))}:'
                    f'{snapshot.get("hand", sim.format_counts(snapshot["hand_counts"]))}'
                )
                if event_key in self.valuable_dama_directives:
                    mask = self.valuable_dama_directives[event_key]
                    if mask & DAMA_LOCK_FLAG:
                        valuable_seats.add(seat)
                        reason = self.valuable_dama_directive_reasons.get(
                            event_key, "valuable_bad_wait"
                        )
                        active_dama_reasons[seat] = reason
                        haneman_entered |= reason == "haneman_or_higher"
                        bad_wait_value_entered |= reason == "valuable_bad_wait"
                        continue
                    valuable_seats.discard(seat)
                    active_dama_reasons.pop(seat, None)
                elif seat in valuable_seats:
                    continue
                candidate = dict(row)
                candidate.update({
                    "outcome": "tenpai",
                    "player_wind": 27 + seat,
                    "first_tenpai": snapshot,
                })
                if self._reject_cheap_bad_wait(candidate, snapshot):
                    rejected_events += 1
                    continue
                self.continuation_tables_with_rejection += int(rejected_events > 0)
                self.continuation_eligible_after_rejection += int(rejected_events > 0)
                self.continuation_shanten_reversals += reversals
                self.valuable_dama_tables += int(bad_wait_value_entered)
                self.haneman_dama_tables += int(haneman_entered)
                reject_enabled = self.reject_cheap_bad_wait_riichi
                self.reject_cheap_bad_wait_riichi = False
                try:
                    self.add_row(candidate)
                finally:
                    self.reject_cheap_bad_wait_riichi = reject_enabled
                return
            if kind in ("dama_tsumo", "dama_ron"):
                candidate = dict(row)
                candidate.update({
                    "outcome": kind,
                    "player_wind": 27 + seat,
                    "first_tenpai": None,
                    "win": event.get("win"),
                })
                self.continuation_tables_with_rejection += int(rejected_events > 0)
                self.continuation_tsumo_after_rejection += int(
                    rejected_events > 0 and kind == "dama_tsumo"
                )
                self.continuation_ron_after_rejection += int(
                    rejected_events > 0 and kind == "dama_ron"
                )
                self.continuation_shanten_reversals += reversals
                self.valuable_dama_tables += int(bad_wait_value_entered)
                self.haneman_dama_tables += int(haneman_entered)
                valuable_win = active_dama_reasons.get(seat) == "valuable_bad_wait"
                self.valuable_dama_tsumo += int(valuable_win and kind == "dama_tsumo")
                self.valuable_dama_ron += int(valuable_win and kind == "dama_ron")
                haneman_win = active_dama_reasons.get(seat) == "haneman_or_higher"
                self.haneman_dama_tsumo += int(haneman_win and kind == "dama_tsumo")
                self.haneman_dama_ron += int(haneman_win and kind == "dama_ron")
                self.add_row(candidate)
                return
        exhausted = dict(row)
        exhausted.update({"outcome": "exhausted", "first_tenpai": None, "win": None})
        self.continuation_tables_with_rejection += int(rejected_events > 0)
        self.continuation_exhausted_after_rejection += int(rejected_events > 0)
        self.continuation_shanten_reversals += reversals
        self.valuable_dama_tables += int(bad_wait_value_entered)
        self.haneman_dama_tables += int(haneman_entered)
        self.add_row(exhausted)

    @staticmethod
    def _snapshot_primary_category(snapshot: Mapping[str, Any]) -> str:
        waits = snapshot.get("waits", ())
        hand_counts = snapshot.get("hand_counts")
        if hand_counts is None:
            hand_counts = sim.counts_from_physical(snapshot["hand_physical"])
        return primary_wait_category(
            hand_counts, (int(wait["tile_type"]) for wait in waits)
        )

    def _reject_cheap_bad_wait(
        self, row: Mapping[str, Any], snapshot: Mapping[str, Any]
    ) -> bool:
        player_wind = int(row["player_wind"])
        if player_wind == 27:
            return False
        waits = snapshot.get("waits", ())
        primary_category = self._snapshot_primary_category(snapshot)
        bad_wait = primary_category in ("kanchan", "penchan", "shanpon")
        if primary_category in ("shanpon", "tanki"):
            honor_with_remaining_copy = any(
                int(wait["tile_type"]) >= 27
                and int(wait.get("visible_remaining", 0)) > 0
                for wait in waits
            )
            bad_wait = not honor_with_remaining_copy
        if not bad_wait:
            return False

        self.cheap_bad_wait_candidates += 1
        hand = tuple(int(tile) for tile in snapshot["hand_physical"])
        context = sim.ScoreContext(
            is_tsumo=False,
            is_riichi=True,
            is_ippatsu=False,
            is_dealer=False,
            is_haitei=False,
            is_tenhou=False,
            is_chiihou=False,
            player_wind=player_wind,
            round_wind=int(row["round_wind"]),
            dora_indicators=(int(row["dora_indicator"]),),
            ura_dora_indicators=(),
        )
        scores: list[int] = []
        for wait in waits:
            for tile_id in wait.get("visible_remaining_tile_ids", ()):
                score = self.scorer.score(
                    tuple(sorted((*hand, int(tile_id)))), int(tile_id), context
                )
                if not self._valid_score(score):
                    return False
                assert score.points is not None
                scores.append(int(score.points))
        # A dead honor tanki is no longer exempt. Score its structural hand
        # with one non-concealed physical copy even though its decision-visible
        # remaining count is zero.
        if not scores and primary_category == "tanki":
            wait_type = int(waits[0]["tile_type"])
            representative = next(
                tile_id
                for tile_id in sim.PHYSICAL_COPIES_BY_TYPE[wait_type]
                if tile_id not in hand
            )
            score = self.scorer.score(
                tuple(sorted((*hand, representative))), representative, context
            )
            if not self._valid_score(score):
                return False
            assert score.points is not None
            scores.append(int(score.points))
        if not scores or not math.isclose(sum(scores) / len(scores), 1300.0, abs_tol=1e-9):
            return False
        self.cheap_bad_wait_rejections += 1
        self.cheap_bad_wait_rejections_by_wait[primary_category] += 1
        self.cheap_bad_wait_rejections_by_seat[WIND_NAMES[player_wind - 27]] += 1
        return True

    def _add_tenpai(self, row: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
        self.tenpai_hands += 1
        player_wind = int(row.get("player_wind", 28))
        self.first_tenpai_seat_counts[WIND_NAMES[player_wind - 27]] += 1
        self.tenpai_turn_sum += int(snapshot.get("turn", 0))
        self.furiten_hands += int(bool(snapshot.get("furiten", False)))
        waits = snapshot.get("waits", ())
        self.distinct_wait_type_sum += len(waits)
        self.visible_wait_copy_sum += int(snapshot.get("visible_ukeire", 0))
        self.live_wait_copy_sum += int(snapshot.get("live_wall_ukeire", 0))
        live_wall_tiles = int(snapshot.get("live_wall_tiles_remaining", 0))
        self.live_wall_tile_sum += live_wall_tiles
        self.dead_wait_hands += int(int(snapshot.get("live_wall_ukeire", 0)) == 0)
        primary_category = self._snapshot_primary_category(snapshot)
        if primary_category == "other":
            distinct_wait_types = len({int(wait["tile_type"]) for wait in waits})
            self.other_hands_by_wait_type_count[distinct_wait_types] += 1
        self.chiitoitsu_tenpai_hands += int(primary_category == "chiitoitsu_tanki")
        present: set[str] = {primary_category}
        present_tile_types: set[str] = set()
        present_cells: set[str] = set()
        hand_live_copies: Counter[str] = Counter()
        hand_live_copies_by_tile_type: Counter[str] = Counter()
        hand_live_copies_by_tile_and_wait: Counter[str] = Counter()
        for wait in waits:
            categories = (primary_category,)
            tile_category = tile_type_category(int(wait["tile_type"]))
            present.update(categories)
            present_tile_types.add(tile_category)
            split = 1.0 / len(categories)
            visible = int(wait.get("visible_remaining", 0))
            live = int(wait.get("live_wall_remaining", 0))
            hand_live_copies_by_tile_type[tile_category] += live
            for category in categories:
                cell = f"{tile_category}|{category}"
                present_cells.add(cell)
                self.wait_tile_opportunities[category] += split
                self.wait_visible_copies[category] += visible * split
                self.wait_live_copies[category] += live * split
                hand_live_copies[category] += live * split
                hand_live_copies_by_tile_and_wait[cell] += live * split
        for category in present:
            self.eligible_hands_by_wait[category] += 1
        for tile_category in present_tile_types:
            self.eligible_hands_by_tile_type[tile_category] += 1
        for cell in present_cells:
            self.eligible_hands_by_tile_and_wait[cell] += 1
        if live_wall_tiles:
            for category in WAIT_CATEGORIES:
                self.next_draw_hit_probability_sum[category] += hand_live_copies[category] / live_wall_tiles
            for tile_category in TILE_TYPE_CATEGORIES:
                self.next_draw_hit_probability_sum_by_tile_type[tile_category] += (
                    hand_live_copies_by_tile_type[tile_category] / live_wall_tiles
                )
                for category in WAIT_CATEGORIES:
                    cell = f"{tile_category}|{category}"
                    self.next_draw_hit_probability_sum_by_tile_and_wait[cell] += (
                        hand_live_copies_by_tile_and_wait[cell] / live_wall_tiles
                    )
            self.overall_next_draw_hit_probability_sum += int(snapshot.get("live_wall_ukeire", 0)) / live_wall_tiles
        if self.score_tenpai:
            self._score_tenpai_waits(row, snapshot, waits)

    def _score_tenpai_waits(
        self,
        row: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        waits: Sequence[Mapping[str, Any]],
    ) -> None:
        player_wind = int(row["player_wind"])
        if player_wind == 27:
            raise ValueError("non-dealer ron report cannot contain East-seat trials")
        hand = tuple(int(tile) for tile in snapshot["hand_physical"])
        # The report evaluates the first legal tenpai discard as a riichi hand.
        # This also covers the rare turn-zero structural snapshot emitted when
        # the initial 13 tiles are already tenpai.
        riichi = True
        context = sim.ScoreContext(
            is_tsumo=False,
            is_riichi=riichi,
            is_ippatsu=False,
            is_dealer=False,
            is_haitei=False,
            is_tenhou=False,
            is_chiihou=False,
            player_wind=player_wind,
            round_wind=int(row["round_wind"]),
            dora_indicators=(int(row["dora_indicator"]),),
            ura_dora_indicators=(int(row["ura_dora_indicator"]),),
        )
        cache: dict[int, sim.HandScore] = {}
        visible_scores: list[sim.HandScore] = []
        live_scores: list[sim.HandScore] = []
        for wait in waits:
            visible_ids = tuple(int(tile) for tile in wait.get("visible_remaining_tile_ids", ()))
            live_ids = tuple(int(tile) for tile in wait.get("live_wall_remaining_tile_ids", ()))
            if len(visible_ids) != int(wait.get("visible_remaining", 0)):
                raise ValueError("visible wait count lacks matching physical tile IDs")
            if len(live_ids) != int(wait.get("live_wall_remaining", 0)):
                raise ValueError("live wait count lacks matching physical tile IDs")
            for tile_id in set((*visible_ids, *live_ids)):
                if tile_id not in cache:
                    complete = tuple(sorted((*hand, tile_id)))
                    cache[tile_id] = self.scorer.score(complete, tile_id, context)
                    self.score_evaluations += 1
                    if not self._valid_score(cache[tile_id]):
                        self.scoring_errors[
                            cache[tile_id].error or "incomplete score result"
                        ] += 1
            for tile_id in visible_ids:
                score = cache[tile_id]
                if self._valid_score(score):
                    visible_scores.append(score)
                else:
                    self.invalid_visible_copy_weight += 1
            for tile_id in live_ids:
                score = cache[tile_id]
                if self._valid_score(score):
                    live_scores.append(score)
                else:
                    self.invalid_live_copy_weight += 1
        self._add_weighted_hand(visible_scores, live=False)
        self._add_weighted_hand(live_scores, live=True)

    def _valid_score(self, score: sim.HandScore) -> bool:
        return score.error is None and score.points is not None and score.han is not None and score.fu is not None

    def _add_weighted_hand(self, scores: Sequence[sim.HandScore], *, live: bool) -> None:
        if not scores:
            return
        weight = 1.0 / len(scores)
        expected_points = sum(int(score.points) for score in scores if score.points is not None) * weight
        if live:
            self.live_scored_hands += 1
            self.live_hand_point_sum += expected_points
            self.live_hand_point_square_sum += expected_points * expected_points
            han_counts, fu_counts, joint = self.live_han_counts, self.live_fu_counts, self.live_han_fu_counts
        else:
            self.visible_scored_hands += 1
            self.visible_hand_point_sum += expected_points
            self.visible_hand_point_square_sum += expected_points * expected_points
            han_counts, fu_counts, joint = (
                self.visible_han_counts,
                self.visible_fu_counts,
                self.visible_han_fu_counts,
            )
        for score in scores:
            assert score.han is not None and score.fu is not None
            han_counts[score.han] += weight
            fu_counts[score.fu] += weight
            joint[f"{score.han}han_{score.fu}fu"] += weight

    def to_dict(self, *, run: Mapping[str, Any]) -> dict[str, Any]:
        def average(total: float, count: int) -> float | None:
            return total / count if count else None

        def mean_standard_error(total: float, square_total: float, count: int) -> float | None:
            if count < 2:
                return None
            variance = (square_total - total * total / count) / (count - 1)
            return math.sqrt(max(0.0, variance) / count)

        total_wait_opportunities = sum(self.wait_tile_opportunities.values())
        total_visible_copies = sum(self.wait_visible_copies.values())
        total_live_copies = sum(self.wait_live_copies.values())
        wait_distribution: dict[str, Any] = {}
        hit_rates: dict[str, Any] = {}
        for category in WAIT_CATEGORIES:
            opportunities = self.wait_tile_opportunities[category]
            visible = self.wait_visible_copies[category]
            live = self.wait_live_copies[category]
            eligible = self.eligible_hands_by_wait[category]
            probability_sum = self.next_draw_hit_probability_sum[category]
            wait_distribution[category] = {
                "tenpai_hands_with_wait": eligible,
                "hand_share": eligible / self.tenpai_hands if self.tenpai_hands else None,
                "wait_tile_opportunities": opportunities,
                "wait_tile_share": opportunities / total_wait_opportunities if total_wait_opportunities else None,
                "visible_remaining_copies": visible,
                "visible_copy_share": visible / total_visible_copies if total_visible_copies else None,
                "live_wall_remaining_copies": live,
                "live_wall_copy_share": live / total_live_copies if total_live_copies else None,
            }
            hit_rates[category] = {
                "eligible_tenpai_hands": eligible,
                "hand_share": eligible / self.tenpai_hands if self.tenpai_hands else None,
                "average_next_draw_hit_probability": (
                    probability_sum / self.tenpai_hands if self.tenpai_hands else None
                ),
                "average_next_draw_hit_probability_when_exposed": (
                    probability_sum / eligible if eligible else None
                ),
            }

        other_total = self.eligible_hands_by_wait["other"]
        other_by_wait_type_count = {
            str(wait_type_count): {
                "tenpai_hands": count,
                "share_of_other": count / other_total if other_total else None,
                "share_of_all_tenpai": count / self.tenpai_hands if self.tenpai_hands else None,
            }
            for wait_type_count, count in sorted(self.other_hands_by_wait_type_count.items())
        }

        hit_rates_by_tile_type: dict[str, Any] = {}
        hit_rate_matrix: dict[str, Any] = {}
        for tile_category in TILE_TYPE_CATEGORIES:
            eligible = self.eligible_hands_by_tile_type[tile_category]
            probability_sum = self.next_draw_hit_probability_sum_by_tile_type[tile_category]
            hit_rates_by_tile_type[tile_category] = {
                "eligible_tenpai_hands": eligible,
                "hand_share": eligible / self.tenpai_hands if self.tenpai_hands else None,
                "average_next_draw_hit_probability": (
                    probability_sum / self.tenpai_hands if self.tenpai_hands else None
                ),
                "average_next_draw_hit_probability_when_exposed": (
                    probability_sum / eligible if eligible else None
                ),
            }
            hit_rate_matrix[tile_category] = {}
            for category in WAIT_CATEGORIES:
                cell = f"{tile_category}|{category}"
                cell_eligible = self.eligible_hands_by_tile_and_wait[cell]
                cell_probability_sum = self.next_draw_hit_probability_sum_by_tile_and_wait[cell]
                hit_rate_matrix[tile_category][category] = {
                    "eligible_tenpai_hands": cell_eligible,
                    "hand_share": (
                        cell_eligible / self.tenpai_hands if self.tenpai_hands else None
                    ),
                    "average_next_draw_hit_probability": (
                        cell_probability_sum / self.tenpai_hands
                        if self.tenpai_hands else None
                    ),
                    "average_next_draw_hit_probability_when_exposed": (
                        cell_probability_sum / cell_eligible if cell_eligible else None
                    ),
                }

        return {
            "run": dict(run),
            "trials": self.trials,
            "outcomes": dict(sorted(self.outcomes.items())),
            "first_tenpai_seat_distribution": _distribution(
                self.first_tenpai_seat_counts, self.tenpai_hands
            ),
            "riichi_selection_filter": {
                "enabled": self.reject_cheap_bad_wait_riichi,
                "definition": (
                    "exclude nondealer first-tenpai hands whose visible-information riichi ron "
                    "value without ura is exactly 1300 and whose primary wait is kanchan, "
                    "penchan, or shanpon/tanki without a visible remaining honor wait copy"
                ),
                "bad_wait_nondealer_candidates": self.cheap_bad_wait_candidates,
                "rejected_hands": self.cheap_bad_wait_rejections,
                "accepted_tenpai_hands": self.tenpai_hands,
                "rejections_by_wait": dict(
                    sorted(self.cheap_bad_wait_rejections_by_wait.items())
                ),
                "rejections_by_seat": dict(
                    sorted(self.cheap_bad_wait_rejections_by_seat.items())
                ),
            },
            "continued_rejected_tenpai": {
                "tables_with_at_least_one_rejection": self.continuation_tables_with_rejection,
                "rejected_tenpai_events": self.cheap_bad_wait_rejections,
                "eligible_tenpai_after_rejection": self.continuation_eligible_after_rejection,
                "dama_tsumo_after_rejection": self.continuation_tsumo_after_rejection,
                "dama_ron_after_rejection": self.continuation_ron_after_rejection,
                "exhausted_after_rejection": self.continuation_exhausted_after_rejection,
                "shanten_reversals_after_rejection": self.continuation_shanten_reversals,
            },
            "valuable_bad_wait_dama": {
                "tables_entering_valuable_dama": self.valuable_dama_tables,
                "dama_ron": self.valuable_dama_ron,
                "dama_tsumo": self.valuable_dama_tsumo,
            },
            "haneman_or_higher_dama": {
                "tables_entering_haneman_dama": self.haneman_dama_tables,
                "dama_ron": self.haneman_dama_ron,
                "dama_tsumo": self.haneman_dama_tsumo,
            },
            "nondealer_ron_score": {
                "definition": (
                    "each first-tenpai wait is scored as non-dealer ron; each hand first averages "
                    "its waits by remaining physical-copy count"
                ),
                "tenpai_hands": self.tenpai_hands,
                "unseen_copy_weighted": {
                    "scored_hands": self.visible_scored_hands,
                    "average_points": average(self.visible_hand_point_sum, self.visible_scored_hands),
                    "average_points_standard_error": mean_standard_error(
                        self.visible_hand_point_sum,
                        self.visible_hand_point_square_sum,
                        self.visible_scored_hands,
                    ),
                },
                "live_wall_weighted_diagnostic": {
                    "scored_hands": self.live_scored_hands,
                    "average_points": average(self.live_hand_point_sum, self.live_scored_hands),
                    "average_points_standard_error": mean_standard_error(
                        self.live_hand_point_sum,
                        self.live_hand_point_square_sum,
                        self.live_scored_hands,
                    ),
                },
                "score_evaluations": self.score_evaluations,
                "invalid_visible_copy_weight": self.invalid_visible_copy_weight,
                "invalid_live_copy_weight": self.invalid_live_copy_weight,
                "scoring_errors": dict(sorted(self.scoring_errors.items())),
            },
            "score_distribution_definition": (
                "each scored tenpai hand contributes total weight 1, divided among its winning "
                "physical copies by unseen count (four minus concealed and publicly visible copies)"
            ),
            "han_distribution": _distribution(self.visible_han_counts, self.visible_scored_hands),
            "fu_distribution": _distribution(self.visible_fu_counts, self.visible_scored_hands),
            "han_fu_joint_distribution": _distribution(
                self.visible_han_fu_counts, self.visible_scored_hands
            ),
            "live_wall_weighted_han_distribution": _distribution(
                self.live_han_counts, self.live_scored_hands
            ),
            "live_wall_weighted_fu_distribution": _distribution(
                self.live_fu_counts, self.live_scored_hands
            ),
            "wait_distribution": {
                "definition": (
                    "primary percentage is the share of first-tenpai hands containing each wait; "
                    "each hand receives exactly one whole-hand structural category, so hand shares "
                    "sum to one"
                ),
                "percentage_basis": "tenpai hands assigned to the wait category",
                "categories": wait_distribution,
                "other_by_distinct_wait_tile_types": {
                    "definition": (
                        "hands in the primary other category, separated by the exact number "
                        "of distinct winning tile types"
                    ),
                    "other_tenpai_hands": other_total,
                    "categories": other_by_wait_type_count,
                },
            },
            "average_wait_tiles_remaining_at_first_tenpai": {
                "tenpai_hands": self.tenpai_hands,
                "mean_first_tenpai_turn": average(self.tenpai_turn_sum, self.tenpai_hands),
                "chiitoitsu_tenpai_hands": self.chiitoitsu_tenpai_hands,
                "chiitoitsu_tenpai_rate": average(
                    self.chiitoitsu_tenpai_hands, self.tenpai_hands
                ),
                "furiten_hands": self.furiten_hands,
                "dead_wait_hands": self.dead_wait_hands,
                "distinct_wait_tile_types": average(self.distinct_wait_type_sum, self.tenpai_hands),
                "visible_remaining_copies": average(self.visible_wait_copy_sum, self.tenpai_hands),
                "live_wall_remaining_copies": average(self.live_wait_copy_sum, self.tenpai_hands),
            },
            "tile_hit_rate_by_wait": {
                "definition": (
                    "probability that the next tile sampled uniformly from the remaining live wall "
                    "completes a hand in each primary whole-hand wait category"
                ),
                "overall_average_next_draw_hit_probability": (
                    self.overall_next_draw_hit_probability_sum / self.tenpai_hands
                    if self.tenpai_hands else None
                ),
                "categories": hit_rates,
            },
            "tile_hit_rate_by_tile_type_and_wait": {
                "definition": (
                    "next-live-wall-draw hit probability split by winning tile rank group and wait "
                    "type; hand_share is the percentage of first-tenpai hands containing at least "
                    "one winning tile in the cell. One hand may occur in multiple tile-rank cells, "
                    "but has only one primary wait type"
                ),
                "tile_type_categories": {
                    "19": "suited terminals 1 and 9",
                    "28": "suited ranks 2 and 8",
                    "37": "suited ranks 3 and 7",
                    "456": "suited middle ranks 4, 5, and 6",
                    "honor": "winds and dragons",
                },
                "by_tile_type": hit_rates_by_tile_type,
                "by_tile_type_and_wait": hit_rate_matrix,
            },
            "shanten_reversal": {
                "opportunities": self.reversal_opportunities,
                "selected": self.reversals_selected,
                "selection_rate_per_opportunity": average(
                    self.reversals_selected, self.reversal_opportunities
                ),
                "mean_selection_turn": average(self.reversal_turn_sum, self.reversals_selected),
                "recovered": self.reversals_recovered,
                "recovery_rate": average(self.reversals_recovered, self.reversals_selected),
                "mean_recovery_turns": average(
                    self.reversal_recovery_turn_sum, self.reversals_recovered
                ),
                "transitions": dict(sorted(self.reversal_transitions.items())),
                "mean_recovery_copies": average(
                    self.reversal_recovery_count_sum, self.reversals_selected
                ),
                "mean_recovered_ukeire": average(
                    self.reversal_recovered_ukeire_sum, self.reversals_selected
                ),
                "mean_current_ukeire": average(
                    self.reversal_current_ukeire_sum, self.reversals_selected
                ),
                "mean_ukeire_gain": average(
                    self.reversal_ukeire_gain_sum, self.reversals_selected
                ),
                "mean_strict_score": average(
                    self.reversal_strict_score_sum, self.reversals_selected
                ),
                "mean_adjusted_score": average(
                    self.reversal_adjusted_score_sum, self.reversals_selected
                ),
            },
            "raw": {
                "tenpai_turn_sum": self.tenpai_turn_sum,
                "first_tenpai_seat_counts": dict(sorted(self.first_tenpai_seat_counts.items())),
                "cheap_bad_wait_candidates": self.cheap_bad_wait_candidates,
                "cheap_bad_wait_rejections": self.cheap_bad_wait_rejections,
                "chiitoitsu_tenpai_hands": self.chiitoitsu_tenpai_hands,
                "visible_scored_hands": self.visible_scored_hands,
                "visible_hand_point_sum": self.visible_hand_point_sum,
                "visible_hand_point_square_sum": self.visible_hand_point_square_sum,
                "live_scored_hands": self.live_scored_hands,
                "live_hand_point_sum": self.live_hand_point_sum,
                "live_hand_point_square_sum": self.live_hand_point_square_sum,
                "distinct_wait_type_sum": self.distinct_wait_type_sum,
                "visible_wait_copy_sum": self.visible_wait_copy_sum,
                "live_wait_copy_sum": self.live_wait_copy_sum,
                "visible_han_counts": dict(sorted(self.visible_han_counts.items())),
                "visible_fu_counts": dict(sorted(self.visible_fu_counts.items())),
                "visible_han_fu_counts": dict(sorted(self.visible_han_fu_counts.items())),
                "live_han_counts": dict(sorted(self.live_han_counts.items())),
                "live_fu_counts": dict(sorted(self.live_fu_counts.items())),
                "live_han_fu_counts": dict(sorted(self.live_han_fu_counts.items())),
                "wait_tile_opportunities": {
                    category: self.wait_tile_opportunities[category] for category in WAIT_CATEGORIES
                },
                "wait_visible_copies": {
                    category: self.wait_visible_copies[category] for category in WAIT_CATEGORIES
                },
                "wait_live_copies": {
                    category: self.wait_live_copies[category] for category in WAIT_CATEGORIES
                },
                "eligible_hands_by_wait": {
                    category: self.eligible_hands_by_wait[category] for category in WAIT_CATEGORIES
                },
                "other_hands_by_wait_type_count": dict(
                    sorted(self.other_hands_by_wait_type_count.items())
                ),
                "eligible_hands_by_tile_type": {
                    category: self.eligible_hands_by_tile_type[category]
                    for category in TILE_TYPE_CATEGORIES
                },
                "eligible_hands_by_tile_and_wait": {
                    f"{tile_category}|{category}": self.eligible_hands_by_tile_and_wait[
                        f"{tile_category}|{category}"
                    ]
                    for tile_category in TILE_TYPE_CATEGORIES
                    for category in WAIT_CATEGORIES
                },
                "next_draw_hit_probability_sum": {
                    category: self.next_draw_hit_probability_sum[category] for category in WAIT_CATEGORIES
                },
                "next_draw_hit_probability_sum_by_tile_type": {
                    category: self.next_draw_hit_probability_sum_by_tile_type[category]
                    for category in TILE_TYPE_CATEGORIES
                },
                "next_draw_hit_probability_sum_by_tile_and_wait": {
                    f"{tile_category}|{category}": (
                        self.next_draw_hit_probability_sum_by_tile_and_wait[
                            f"{tile_category}|{category}"
                        ]
                    )
                    for tile_category in TILE_TYPE_CATEGORIES
                    for category in WAIT_CATEGORIES
                },
                "overall_next_draw_hit_probability_sum": self.overall_next_draw_hit_probability_sum,
                "reversal_opportunities": self.reversal_opportunities,
                "reversals_selected": self.reversals_selected,
                "reversals_recovered": self.reversals_recovered,
                "reversal_turn_sum": self.reversal_turn_sum,
                "reversal_recovery_turn_sum": self.reversal_recovery_turn_sum,
                "reversal_transitions": dict(sorted(self.reversal_transitions.items())),
                "reversal_recovery_count_sum": self.reversal_recovery_count_sum,
                "reversal_recovered_ukeire_sum": self.reversal_recovered_ukeire_sum,
                "reversal_current_ukeire_sum": self.reversal_current_ukeire_sum,
                "reversal_ukeire_gain_sum": self.reversal_ukeire_gain_sum,
                "reversal_strict_score_sum": self.reversal_strict_score_sum,
                "reversal_adjusted_score_sum": self.reversal_adjusted_score_sum,
            },
        }


def _availability_wind_type(value: Any, *, field_name: str = "wind") -> int:
    """Normalize a wind name/index/type to the 34-format tile type."""
    if isinstance(value, str):
        name = value.lower()
        if name in WIND_NAMES:
            return 27 + WIND_NAMES.index(name)
        # Native output normally uses integer tile types, but accepting a
        # compact ``1z``-style name makes the reader useful for hand fixtures.
        if name.endswith("z") and name[:-1].isdigit():
            number = int(name[:-1])
            if 1 <= number <= 4:
                return 26 + number
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a wind, not a boolean")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field_name}: {value!r}") from error
    if 0 <= number < 4:
        number += 27
    if 27 <= number <= 30:
        return number
    raise ValueError(f"invalid {field_name}: {value!r}")


def availability_rank_category(tile_type: int) -> str:
    """Return the requested rank/honor availability category."""
    if not 0 <= tile_type < sim.TILE_TYPE_COUNT:
        raise ValueError(f"tile type must be 0..33, got {tile_type}")
    if tile_type >= 27:
        return "dragons" if tile_type >= 31 else "guest_wind"
    rank = tile_type % 9 + 1
    if rank in (1, 9):
        return "19"
    if rank in (2, 8):
        return "28"
    if rank in (3, 7):
        return "37"
    if rank in (4, 6):
        return "46"
    return "5"


def availability_table_category(tile_type: int, round_wind: Any) -> str:
    """Map a tile to a table-level category.

    The three non-round winds are deliberately one category: each is the seat
    wind for one player, while the category itself is independent of observer.
    """
    if not 0 <= tile_type < sim.TILE_TYPE_COUNT:
        raise ValueError(f"tile type must be 0..33, got {tile_type}")
    round_type = _availability_wind_type(round_wind, field_name="round_wind")
    if tile_type == round_type:
        return "round_wind"
    if 27 <= tile_type <= 30:
        return "seat_wind_for_one_player"
    return availability_rank_category(tile_type)


def availability_player_category(
    tile_type: int, observer_seat: Any, round_wind: Any
) -> str:
    """Map a tile to a player-relative category.

    Wind priority is double > seat > round > guest.  The round/seat wind
    overlap is therefore represented exactly once as ``double_wind``.
    """
    if not 0 <= tile_type < sim.TILE_TYPE_COUNT:
        raise ValueError(f"tile type must be 0..33, got {tile_type}")
    seat = int(observer_seat)
    if not 0 <= seat < 4:
        # A caller may pass a 34-format seat-wind tile type.
        seat_type = _availability_wind_type(observer_seat, field_name="observer_seat")
    else:
        seat_type = 27 + seat
    round_type = _availability_wind_type(round_wind, field_name="round_wind")
    if 27 <= tile_type <= 30:
        if seat_type == round_type == tile_type:
            return "double_wind"
        if tile_type == seat_type:
            return "seat_wind"
        if tile_type == round_type:
            return "round_wind"
        return "guest_wind"
    return availability_rank_category(tile_type)


def availability_score(
    actual_live_copies: float,
    unseen_copies: float,
    live_wall_tiles_remaining: float,
    all_unseen_positions: float,
) -> float | None:
    """Calculate A = L / (U * W / N), skipping a zero expectation."""
    expected = (
        float(unseen_copies) * float(live_wall_tiles_remaining)
        / float(all_unseen_positions)
        if all_unseen_positions else 0.0
    )
    if expected == 0.0:
        return None
    return float(actual_live_copies) / expected


def aggregate_availability_scores(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate score samples with table-clustered standard error.

    This small public helper mirrors the collector's reporting convention and
    is useful for tests or downstream analyses that already have U/E/A rows.
    """
    valid: list[Mapping[str, Any]] = []
    skipped = 0
    for sample in samples:
        score = sample.get("score")
        expected = float(sample.get("expected", 0.0))
        if score is None or expected == 0.0:
            skipped += 1
            continue
        valid.append(sample)
    scores = [float(sample["score"]) for sample in valid]
    actual = sum(float(sample.get("actual", 0.0)) for sample in valid)
    expected = sum(float(sample.get("expected", 0.0)) for sample in valid)
    by_table: dict[Any, list[float]] = defaultdict(list)
    for sample in valid:
        by_table[sample.get("table_id")].append(float(sample["score"]))
    table_means = [sum(values) / len(values) for values in by_table.values() if values]
    return {
        "mean_score": sum(scores) / len(scores) if scores else None,
        "pooled_score": actual / expected if expected else None,
        "valid_observations": len(valid),
        "skipped_observations": skipped,
        "skipped_zero_expected": skipped,
        "table_clusters": len(table_means),
        "table_clustered_standard_error": _availability_sample_standard_error(table_means),
        "actual_live_copies": actual,
        "expected_live_copies": expected,
    }


def _availability_counts(value: Any, *, field_name: str) -> tuple[int, ...]:
    """Read a native 34-count array and validate its physical-copy bounds."""
    if isinstance(value, Mapping):
        # This is a convenience for hand-authored fixtures.  Native output is
        # an array, so missing mapping entries naturally mean zero copies.
        counts = [0] * sim.TILE_TYPE_COUNT
        for key, count in value.items():
            if isinstance(key, str) and key.isdigit():
                tile_type = int(key)
            elif isinstance(key, str):
                try:
                    tile_type = next(
                        index for index in range(sim.TILE_TYPE_COUNT)
                        if sim.tile_name(index) == key
                    )
                except StopIteration as error:
                    raise ValueError(f"invalid tile key in {field_name}: {key!r}") from error
            else:
                tile_type = int(key)
            if not 0 <= tile_type < sim.TILE_TYPE_COUNT:
                raise ValueError(f"invalid tile type in {field_name}: {tile_type}")
            counts[tile_type] = int(count)
    else:
        try:
            counts = [int(count) for count in value]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name} must be a 34-count array") from error
        if len(counts) != sim.TILE_TYPE_COUNT:
            raise ValueError(f"{field_name} must contain 34 tile counts")
    if any(count < 0 or count > 4 for count in counts):
        raise ValueError(f"{field_name} contains a count outside 0..4")
    return tuple(counts)


def _availability_sample_standard_error(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance) / len(values))


@dataclass
class _AvailabilityAggregate:
    """Sufficient statistics for mean/pooled ratios and table-clustered SE."""

    ratio_sum: float = 0.0
    valid_count: int = 0
    skipped_count: int = 0
    actual_sum: float = 0.0
    expected_sum: float = 0.0
    table_count: int = 0
    table_mean: float = 0.0
    table_mean_m2: float = 0.0
    table_pooled_count: int = 0
    table_pooled_mean: float = 0.0
    table_pooled_m2: float = 0.0
    # Kept only for the small direct-use/testing API.  The high-volume
    # collector passes cluster=False and uses online finish_table statistics.
    direct_by_table: dict[Any, list[float]] = field(default_factory=dict)

    def add(
        self, table_id: Any, actual: float, expected: float, *, cluster: bool = True
    ) -> None:
        ratio = actual / expected if expected != 0.0 else None
        if cluster:
            group = self.direct_by_table.setdefault(table_id, [0.0, 0.0, 0.0, 0.0, 0.0])
            if ratio is None:
                group[4] += 1.0
            else:
                group[0] += ratio
                group[1] += 1.0
                group[2] += actual
                group[3] += expected
        if ratio is None:
            self.skipped_count += 1
            return
        self.ratio_sum += ratio
        self.valid_count += 1
        self.actual_sum += actual
        self.expected_sum += expected

    def finish_table(
        self, ratio_sum: float, valid_count: int, actual_sum: float, expected_sum: float
    ) -> None:
        """Add one table's grouped means to the online cluster statistics."""
        if valid_count:
            table_ratio = ratio_sum / valid_count
            self.table_count += 1
            delta = table_ratio - self.table_mean
            self.table_mean += delta / self.table_count
            self.table_mean_m2 += delta * (table_ratio - self.table_mean)
        if expected_sum:
            table_pooled = actual_sum / expected_sum
            # The pooled score uses the same table clusters but has a separate
            # distribution from the arithmetic mean of per-tile ratios.
            self.table_pooled_count += 1
            current_count = self.table_pooled_count
            delta = table_pooled - self.table_pooled_mean
            self.table_pooled_mean += delta / current_count
            self.table_pooled_m2 += delta * (table_pooled - self.table_pooled_mean)

    def summary(self) -> dict[str, Any]:
        mean_ratio = self.ratio_sum / self.valid_count if self.valid_count else None
        pooled_ratio = self.actual_sum / self.expected_sum if self.expected_sum else None
        if self.direct_by_table:
            table_means = [
                values[0] / values[1]
                for values in self.direct_by_table.values() if values[1]
            ]
            table_pooled = [
                values[2] / values[3]
                for values in self.direct_by_table.values() if values[3]
            ]
            table_count = len(table_means)
            table_se = _availability_sample_standard_error(table_means)
            pooled_se = _availability_sample_standard_error(table_pooled)
        else:
            table_count = self.table_count
            table_se = (
                math.sqrt(
                    max(0.0, self.table_mean_m2 / (self.table_count - 1))
                    / self.table_count
                )
                if self.table_count > 1 else None
            )
            pooled_se = (
                math.sqrt(
                    max(0.0, self.table_pooled_m2 / (self.table_pooled_count - 1))
                    / self.table_pooled_count
                )
                if self.table_pooled_count > 1 else None
            )
        return {
            "mean_ratio": mean_ratio,
            "pooled_ratio": pooled_ratio,
            # Explicit score aliases make the JSON self-describing for callers
            # that treat availability as a score rather than a ratio.
            "mean_score": mean_ratio,
            "pooled_score": pooled_ratio,
            "valid_observations": self.valid_count,
            "skipped_zero_expected": self.skipped_count,
            "skipped_observations": self.skipped_count,
            "actual_live_copies": self.actual_sum,
            "expected_live_copies": self.expected_sum,
            "tables": table_count,
            "table_clusters": table_count,
            "table_clustered_standard_error": table_se,
            "pooled_ratio_table_clustered_standard_error": pooled_se,
        }


class TileAvailabilityCollector:
    """Aggregate native shared-table availability checkpoints.

    The native engine emits one ``availability_checkpoint`` per trial.  This
    collector intentionally consumes the raw 34-type compartments and derives
    U/E/A itself, so the formula remains auditable and Python/native output
    cannot silently disagree.
    """

    def __init__(self, target_turn: int, *, default_round_wind: Any = "east") -> None:
        if target_turn < 0:
            raise ValueError("tile availability turn must be nonnegative")
        self.target_turn = int(target_turn)
        self.default_round_wind = _availability_wind_type(default_round_wind)
        self.rows_seen = 0
        self.snapshot_count = 0
        self.table_ids: set[Any] = set()
        self._fallback_table_id = 0
        self._all: dict[str, dict[str, _AvailabilityAggregate]] = {
            "table_level": {
                category: _AvailabilityAggregate() for category in AVAILABILITY_TABLE_CATEGORIES
            },
            "player_relative": {
                category: _AvailabilityAggregate() for category in AVAILABILITY_PLAYER_CATEGORIES
            },
        }
        self._by_seat: dict[str, dict[str, dict[str, _AvailabilityAggregate]]] = {
            view: {
                seat_name: {
                    category: _AvailabilityAggregate()
                    for category in (
                        AVAILABILITY_TABLE_CATEGORIES
                        if view == "table_level" else AVAILABILITY_PLAYER_CATEGORIES
                    )
                }
                for seat_name in WIND_NAMES
            }
            for view in ("table_level", "player_relative")
        }
        self._raw: dict[str, dict[int, _AvailabilityAggregate]] = {
            "overall": {tile: _AvailabilityAggregate() for tile in range(sim.TILE_TYPE_COUNT)},
            **{
                seat_name: {tile: _AvailabilityAggregate() for tile in range(sim.TILE_TYPE_COUNT)}
                for seat_name in WIND_NAMES
            },
        }
        self._estimation: dict[str, dict[str, _AvailabilityAggregate]] = {
            "table_level": {
                category: _AvailabilityAggregate() for category in AVAILABILITY_TABLE_CATEGORIES
            },
            "player_relative": {
                category: _AvailabilityAggregate() for category in AVAILABILITY_PLAYER_CATEGORIES
            },
        }
        # Companion location diagnostics use the same category and observer
        # dimensions as the live-wall report.  Live-wall aggregates remain the
        # established top-level schema; only the hidden compartments need a
        # second set of accumulators here.
        self._location_all: dict[str, dict[str, dict[str, _AvailabilityAggregate]]] = {
            location: {
                "table_level": {
                    category: _AvailabilityAggregate()
                    for category in AVAILABILITY_TABLE_CATEGORIES
                },
                "player_relative": {
                    category: _AvailabilityAggregate()
                    for category in AVAILABILITY_PLAYER_CATEGORIES
                },
            }
            for location in ("hidden_dead_wall", "opponent_hands")
        }
        self._location_by_seat: dict[
            str, dict[str, dict[str, dict[str, _AvailabilityAggregate]]]
        ] = {
            location: {
                view: {
                    seat_name: {
                        category: _AvailabilityAggregate()
                        for category in (
                            AVAILABILITY_TABLE_CATEGORIES
                            if view == "table_level" else AVAILABILITY_PLAYER_CATEGORIES
                        )
                    }
                    for seat_name in WIND_NAMES
                }
                for view in ("table_level", "player_relative")
            }
            for location in ("hidden_dead_wall", "opponent_hands")
        }
        # Validation is retained as compact per-player bytes arrays.  Each
        # validation tile is reconstructed only after estimation factors are
        # fitted, keeping 100k-table reports bounded in memory.
        self._validation_by_tile: list[tuple[Any, int, bytes, bytes, int, int, int]] = []

    @staticmethod
    def _checkpoint(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
        checkpoint = row.get("availability_checkpoint")
        if checkpoint is None:
            # Accept the shorter aliases used by early native prototypes and
            # hand-authored fixtures; current C++ emits the first key.
            for key in ("tile_availability_checkpoint", "availability_snapshot"):
                checkpoint = row.get(key)
                if checkpoint is not None:
                    break
        if checkpoint is None:
            return None
        if not isinstance(checkpoint, Mapping):
            raise ValueError("availability checkpoint must be an object")
        return checkpoint

    @staticmethod
    def _seat_name(value: Any) -> str:
        seat = int(value)
        if not 0 <= seat < 4:
            raise ValueError(f"availability observer seat must be 0..3, got {value!r}")
        return WIND_NAMES[seat]

    def add_row(self, row: Mapping[str, Any]) -> None:
        self.rows_seen += 1
        checkpoint = self._checkpoint(row)
        if checkpoint is None:
            return
        target = checkpoint.get(
            "completed_own_turn", checkpoint.get("target_own_turn", self.target_turn)
        )
        if int(target) != self.target_turn:
            raise ValueError(
                f"availability checkpoint turn {target} does not match requested {self.target_turn}"
            )
        players = checkpoint.get("players")
        if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
            raise ValueError("availability checkpoint players must be an array")
        if len(players) != 4:
            raise ValueError("availability checkpoint must contain four players")
        table_id: Any = row.get("trial_id", self._fallback_table_id)
        self._fallback_table_id += 1
        try:
            table_id = int(table_id)
        except (TypeError, ValueError):
            table_id = str(table_id)
        self.table_ids.add(table_id)
        self.snapshot_count += 1
        round_wind = _availability_wind_type(
            row.get("round_wind", self.default_round_wind), field_name="round_wind"
        )
        checkpoint_live = checkpoint.get("live_wall_tiles_remaining")
        checkpoint_live_int = int(checkpoint_live) if checkpoint_live is not None else None
        checkpoint_hidden = checkpoint.get("hidden_dead_wall_tiles_remaining")
        checkpoint_hidden_int = int(checkpoint_hidden) if checkpoint_hidden is not None else None
        checkpoint_opponents = checkpoint.get("opponent_hand_tiles")
        checkpoint_opponents_int = (
            int(checkpoint_opponents) if checkpoint_opponents is not None else None
        )
        table_groups: dict[tuple[Any, ...], list[Any]] = {}

        def record_table_group(
            key: tuple[Any, ...], aggregate: _AvailabilityAggregate,
            actual: float, expected: float,
        ) -> None:
            aggregate.add(table_id, actual, expected, cluster=False)
            if expected == 0.0:
                return
            group = table_groups.setdefault(key, [aggregate, 0.0, 0, 0.0, 0.0])
            group[1] += actual / expected
            group[2] += 1
            group[3] += actual
            group[4] += expected

        for player_index, player_value in enumerate(players):
            if not isinstance(player_value, Mapping):
                raise ValueError("availability player must be an object")
            seat = int(player_value.get("seat", player_index))
            seat_name = self._seat_name(seat)
            own = player_value.get("own_known_counts", player_value.get("hand_counts"))
            public = player_value.get("public_counts")
            unseen_value = player_value.get("unseen_counts")
            live_value = player_value.get("live_wall_counts")
            hidden_value = player_value.get("hidden_dead_wall_counts")
            opponent_value = player_value.get("opponent_hand_counts")
            visible_value = player_value.get("visible_counts")
            derived_unseen: tuple[int, ...] | None = None
            if own is None and unseen_value is None:
                if visible_value is None:
                    raise ValueError("availability player lacks own_known_counts/unseen_counts")
                visible_counts = _availability_counts(
                    visible_value, field_name="visible_counts"
                )
                derived_unseen = tuple(4 - count for count in visible_counts)
            if public is None:
                public_counts = (
                    _availability_counts(visible_value, field_name="visible_counts")
                    if own is None and visible_value is not None
                    else (0,) * sim.TILE_TYPE_COUNT
                )
            else:
                public_counts = _availability_counts(public, field_name="public_counts")
            if own is None:
                unseen_counts = (
                    _availability_counts(unseen_value, field_name="unseen_counts")
                    if unseen_value is not None else derived_unseen
                )
                assert unseen_counts is not None
                visible_counts = tuple(4 - count for count in unseen_counts)
                own_counts = tuple(max(0, visible_counts[index] - public_counts[index]) for index in range(34))
            else:
                own_counts = _availability_counts(own, field_name="own_known_counts")
                visible_counts = tuple(own_counts[index] + public_counts[index] for index in range(34))
                if any(count > 4 for count in visible_counts):
                    raise ValueError("availability visible counts exceed four copies")
                unseen_counts = tuple(4 - count for count in visible_counts)
            if unseen_value is not None:
                declared_unseen = _availability_counts(unseen_value, field_name="unseen_counts")
                if declared_unseen != unseen_counts:
                    raise ValueError("availability unseen_counts do not equal 4-visible")
            if live_value is None:
                raise ValueError("availability player lacks live_wall_counts")
            if hidden_value is None:
                raise ValueError("availability player lacks hidden_dead_wall_counts")
            if opponent_value is None:
                raise ValueError("availability player lacks opponent_hand_counts")
            live_counts = _availability_counts(live_value, field_name="live_wall_counts")
            hidden_counts = _availability_counts(
                hidden_value, field_name="hidden_dead_wall_counts"
            )
            opponent_counts = _availability_counts(
                opponent_value, field_name="opponent_hand_counts"
            )
            live_tiles = int(
                player_value.get(
                    "live_wall_tiles",
                    checkpoint_live_int if checkpoint_live_int is not None else sum(live_counts),
                )
            )
            if sum(live_counts) != live_tiles:
                raise ValueError("availability live_wall_counts total mismatch")
            if checkpoint_live_int is not None and live_tiles != checkpoint_live_int:
                raise ValueError("availability player/live checkpoint total mismatch")
            hidden_tiles = int(
                player_value.get(
                    "hidden_dead_wall_tiles",
                    checkpoint_hidden_int
                    if checkpoint_hidden_int is not None else sum(hidden_counts),
                )
            )
            opponent_tiles = int(
                player_value.get(
                    "opponent_hand_tiles",
                    checkpoint_opponents_int
                    if checkpoint_opponents_int is not None else sum(opponent_counts),
                )
            )
            if sum(hidden_counts) != hidden_tiles:
                raise ValueError("availability hidden_dead_wall_counts total mismatch")
            if sum(opponent_counts) != opponent_tiles:
                raise ValueError("availability opponent_hand_counts total mismatch")
            if checkpoint_hidden_int is not None and hidden_tiles != checkpoint_hidden_int:
                raise ValueError("availability player/hidden checkpoint total mismatch")
            if checkpoint_opponents_int is not None and opponent_tiles != checkpoint_opponents_int:
                raise ValueError("availability player/opponent checkpoint total mismatch")
            unseen_tiles = sum(unseen_counts)
            if player_value.get("unseen_tiles") is not None and int(player_value["unseen_tiles"]) != unseen_tiles:
                raise ValueError("availability unseen_counts total mismatch")
            if unseen_tiles <= 0:
                # There cannot normally be no unseen position before the wall
                # is empty, but retaining the row makes the zero-denominator
                # behavior explicit and auditable.
                unseen_tiles = 0
            for tile_type in range(sim.TILE_TYPE_COUNT):
                if (
                    live_counts[tile_type]
                    + hidden_counts[tile_type]
                    + opponent_counts[tile_type]
                    != unseen_counts[tile_type]
                ):
                    raise ValueError(
                        "availability live/hidden/opponent compartments do not conserve unseen"
                    )
            if live_tiles + hidden_tiles + opponent_tiles != unseen_tiles:
                raise ValueError("availability compartment totals do not equal unseen positions")
            expected_by_tile = [
                count * live_tiles / unseen_tiles if unseen_tiles else 0.0
                for count in unseen_counts
            ]
            estimate = (table_id % 2 == 0) if isinstance(table_id, int) else (self.snapshot_count % 2 == 0)
            if not estimate:
                self._validation_by_tile.append(
                    (
                        table_id,
                        seat,
                        bytes(live_counts),
                        bytes(unseen_counts),
                        live_tiles,
                        unseen_tiles,
                        round_wind,
                    )
                )
            for tile_type in range(sim.TILE_TYPE_COUNT):
                actual = float(live_counts[tile_type])
                expected = float(expected_by_tile[tile_type])
                table_category = availability_table_category(tile_type, round_wind)
                player_category = availability_player_category(tile_type, seat, round_wind)
                for view, category in (
                    ("table_level", table_category), ("player_relative", player_category)
                ):
                    record_table_group(
                        ("all", view, category), self._all[view][category], actual, expected
                    )
                    record_table_group(
                        ("seat", view, seat_name, category),
                        self._by_seat[view][seat_name][category], actual, expected,
                    )
                    if estimate:
                        self._estimation[view][category].add(
                            table_id, actual, expected, cluster=False
                        )
                for location, location_counts, location_tiles in (
                    ("hidden_dead_wall", hidden_counts, hidden_tiles),
                    ("opponent_hands", opponent_counts, opponent_tiles),
                ):
                    location_actual = float(location_counts[tile_type])
                    location_expected = (
                        float(unseen_counts[tile_type]) * location_tiles / unseen_tiles
                        if unseen_tiles else 0.0
                    )
                    for view, category in (
                        ("table_level", table_category),
                        ("player_relative", player_category),
                    ):
                        record_table_group(
                            ("location", location, "all", view, category),
                            self._location_all[location][view][category],
                            location_actual,
                            location_expected,
                        )
                        record_table_group(
                            ("location", location, "seat", view, seat_name, category),
                            self._location_by_seat[location][view][seat_name][category],
                            location_actual,
                            location_expected,
                        )
                record_table_group(
                    ("raw", "overall", tile_type),
                    self._raw["overall"][tile_type], actual, expected,
                )
                record_table_group(
                    ("raw", seat_name, tile_type),
                    self._raw[seat_name][tile_type], actual, expected,
                )
        for aggregate, ratio_sum, valid_count, actual_sum, expected_sum in table_groups.values():
            aggregate.finish_table(ratio_sum, valid_count, actual_sum, expected_sum)
        # The native checkpoint is one complete table.  If no rows had a
        # player-level record, the malformed input would have failed above.

    @staticmethod
    def _metric_summary(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
        if not pairs:
            return {
                "observations": 0, "mae": None, "mse": None, "bias": None,
                "actual_live_copies": 0.0, "predicted_live_copies": 0.0,
            }
        errors = [predicted - actual for actual, predicted in pairs]
        return {
            "observations": len(pairs),
            "mae": sum(abs(error) for error in errors) / len(errors),
            "mse": sum(error * error for error in errors) / len(errors),
            "bias": sum(errors) / len(errors),
            "actual_live_copies": sum(actual for actual, _ in pairs),
            "predicted_live_copies": sum(predicted for _, predicted in pairs),
        }

    @staticmethod
    def _confidence_interval(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"mean": None, "standard_error": None, "ci95": None, "tables": 0}
        mean = sum(values) / len(values)
        standard_error = _availability_sample_standard_error(values)
        return {
            "mean": mean,
            "standard_error": standard_error,
            "ci95": (
                [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
                if standard_error is not None else None
            ),
            "tables": len(values),
        }

    def _validation_report(self) -> dict[str, Any]:
        factors: dict[str, dict[str, float]] = {"table_level": {}, "player_relative": {}}
        for view in factors:
            for category, aggregate in self._estimation[view].items():
                pooled = aggregate.summary()["pooled_ratio"]
                if pooled is not None:
                    factors[view][category] = float(pooled)

        metrics: dict[str, Any] = {}
        table_errors: dict[str, dict[Any, list[float]]] = {
            # Each table stores absolute-error sum, squared-error sum,
            # signed-error sum, and exact validation tile count.
            "table_level": defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]),
            "player_relative": defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]),
        }
        # Apply both fitted models to exactly the same validation tile records.
        # This keeps MAE/MSE comparable even though the two views have different
        # category counts and wind granularity.
        by_view: dict[str, dict[str, list[tuple[float, float]]]] = {
            "table_level": defaultdict(list), "player_relative": defaultdict(list)
        }
        by_seat: dict[str, dict[str, list[tuple[float, float]]]] = {
            "table_level": defaultdict(list), "player_relative": defaultdict(list)
        }
        for (
            table_id,
            seat,
            actual_counts,
            unseen_counts,
            live_tiles,
            unseen_tiles,
            round_wind,
        ) in self._validation_by_tile:
            seat_name = WIND_NAMES[seat]
            for tile_type in range(sim.TILE_TYPE_COUNT):
                actual = float(actual_counts[tile_type])
                expected = (
                    float(unseen_counts[tile_type]) * live_tiles / unseen_tiles
                    if unseen_tiles else 0.0
                )
                table_category = availability_table_category(tile_type, round_wind)
                player_category = availability_player_category(tile_type, seat, round_wind)
                for view, category in (
                    ("table_level", table_category), ("player_relative", player_category)
                ):
                    factor = factors[view].get(category)
                    if factor is None:
                        continue
                    predicted = factor * expected
                    pair = (actual, predicted)
                    by_view[view][category].append(pair)
                    by_seat[view][seat_name].append(pair)
                    table_entry = table_errors[view][table_id]
                    error = predicted - actual
                    table_entry[0] += abs(error)
                    table_entry[1] += error * error
                    table_entry[2] += error
                    table_entry[3] += 1.0
        for view in ("table_level", "player_relative"):
            category_names = (
                AVAILABILITY_TABLE_CATEGORIES
                if view == "table_level" else AVAILABILITY_PLAYER_CATEGORIES
            )
            metrics[view] = {
                "estimation_factors": factors[view],
                "by_category": {
                    category: self._metric_summary(by_view[view][category])
                    for category in category_names
                },
                "by_observer_seat": {
                    seat: self._metric_summary(by_seat[view][seat]) for seat in WIND_NAMES
                },
            }
            # Overall metrics retain the exact per-tile validation unit.
            all_pairs = [
                pair
                for category_pairs in by_view[view].values()
                for pair in category_pairs
            ]
            metrics[view]["overall"] = {
                **self._metric_summary(all_pairs),
                "tables": len(table_errors[view]),
            }

        # Paired table-level comparison uses the same validation tables.  A
        # positive difference means player-relative prediction has larger error.
        common_tables = sorted(
            set(table_errors["table_level"]) & set(table_errors["player_relative"]),
            key=str,
        )
        paired = {
            "difference_direction": "player_relative_minus_table_level",
            "tables": len(common_tables),
        }
        for metric_index, name in ((0, "mae"), (1, "mse"), (2, "bias")):
            differences = [
                (
                    table_errors["player_relative"][table_id][metric_index]
                    / table_errors["player_relative"][table_id][3]
                    if table_errors["player_relative"][table_id][3] else 0.0
                )
                - (
                    table_errors["table_level"][table_id][metric_index]
                    / table_errors["table_level"][table_id][3]
                    if table_errors["table_level"][table_id][3] else 0.0
                )
                for table_id in common_tables
            ]
            paired[name] = self._confidence_interval(differences)
        return {
            "split_rule": "trial_id % 2 == 0 is estimation; odd trial IDs are validation",
            "estimation_tables": sum(
                1 for table_id in self.table_ids if isinstance(table_id, int) and table_id % 2 == 0
            ),
            "validation_tables": sum(
                1 for table_id in self.table_ids if isinstance(table_id, int) and table_id % 2 != 0
            ),
            "observation_unit": "validation tile (table, observer, tile type)",
            "metrics": metrics,
            "paired_table_level_differences": paired,
        }

    def _category_report(
        self,
        aggregates: Mapping[str, _AvailabilityAggregate],
        categories: Sequence[str],
        *,
        location: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for category in categories:
            summary = aggregates[category].summary()
            if location is not None:
                # Keep the historical actual_live_copies/expected_live_copies
                # fields intact while giving companion locations unambiguous
                # names in the new report section.
                summary["actual_copies"] = summary["actual_live_copies"]
                summary["expected_copies"] = summary["expected_live_copies"]
                summary[f"actual_{location}_copies"] = summary["actual_live_copies"]
                summary[f"expected_{location}_copies"] = summary["expected_live_copies"]
            result[category] = summary
        return result

    def _location_report(self, location: str) -> dict[str, Any]:
        if location == "live_wall":
            all_aggregates = self._all
            by_seat = self._by_seat
        else:
            all_aggregates = self._location_all[location]
            by_seat = self._location_by_seat[location]
        return {
            "table_level": {
                "categories": self._category_report(
                    all_aggregates["table_level"],
                    AVAILABILITY_TABLE_CATEGORIES,
                    location=location,
                ),
                "by_observer_seat": {
                    seat: self._category_report(
                        by_seat["table_level"][seat],
                        AVAILABILITY_TABLE_CATEGORIES,
                        location=location,
                    )
                    for seat in WIND_NAMES
                },
            },
            "player_relative": {
                "categories": self._category_report(
                    all_aggregates["player_relative"],
                    AVAILABILITY_PLAYER_CATEGORIES,
                    location=location,
                ),
                "by_observer_seat": {
                    seat: self._category_report(
                        by_seat["player_relative"][seat],
                        AVAILABILITY_PLAYER_CATEGORIES,
                        location=location,
                    )
                    for seat in WIND_NAMES
                },
            },
        }

    def _raw_report(self) -> dict[str, Any]:
        def tile_report(aggregates: Mapping[int, _AvailabilityAggregate]) -> dict[str, Any]:
            return {
                str(tile_type): {
                    **aggregates[tile_type].summary(),
                    "tile_type": tile_type,
                    "tile_name": sim.tile_name(tile_type),
                }
                for tile_type in range(sim.TILE_TYPE_COUNT)
            }
        return {
            "by_tile_type": tile_report(self._raw["overall"]),
            "by_observer_seat": {
                seat: tile_report(self._raw[seat]) for seat in WIND_NAMES
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "turn": self.target_turn,
            "formula": {
                "visible_copies": "own_known_counts + public_counts",
                "unseen_copies_U": "4 - visible_copies",
                "expected_live_copies_E": "U * live_wall_tiles_remaining / sum(U)",
                "availability_score_A": "actual live_wall_counts / E",
                "zero_expected_behavior": "skip data point when E == 0",
            },
            "snapshots": self.snapshot_count,
            "player_observations": self.snapshot_count * 4,
            "tables": len(self.table_ids),
            "table_level": {
                "categories": self._category_report(
                    self._all["table_level"], AVAILABILITY_TABLE_CATEGORIES
                ),
                "by_observer_seat": {
                    seat: self._category_report(
                        self._by_seat["table_level"][seat], AVAILABILITY_TABLE_CATEGORIES
                    ) for seat in WIND_NAMES
                },
            },
            "player_relative": {
                "categories": self._category_report(
                    self._all["player_relative"], AVAILABILITY_PLAYER_CATEGORIES
                ),
                "by_observer_seat": {
                    seat: self._category_report(
                        self._by_seat["player_relative"][seat], AVAILABILITY_PLAYER_CATEGORIES
                    ) for seat in WIND_NAMES
                },
            },
            "raw_34_tile_diagnostics": self._raw_report(),
            "companion_location_scores": {
                "units": {
                    "counts": "physical tile copies",
                    "scores": "dimensionless actual/conditional-expected ratio",
                    "expected": "conditional expected copies in the named hidden compartment",
                },
                "definition": {
                    "unseen_copies_U": "4 - visible copies for this observer and tile type",
                    "unseen_positions_N": "sum(U), equal to live wall + hidden dead wall + opponents' hands",
                    "conditional_expected_E": "U * compartment_size / N",
                    "availability_score": "actual compartment copies / E",
                    "zero_expected_behavior": "skip data point when E == 0",
                    "conservation_check": "live + hidden dead wall + opponents' hands == unseen for every player/tile",
                },
                "live_wall": self._location_report("live_wall"),
                "hidden_dead_wall": self._location_report("hidden_dead_wall"),
                "opponent_hands": self._location_report("opponent_hands"),
            },
            "estimation_validation": self._validation_report(),
        }


# Short alias for downstream callers that do not need to distinguish this
# collector from the legacy MonteCarloReportCollector.
AvailabilityCollector = TileAvailabilityCollector


class _AvailabilityOnlyScorer:
    """Placeholder scorer for native checkpoints, which contain no tenpai rows."""

    name = "not_used_for_tile_availability"

    def score(self, hand: Any, win_tile: Any, context: Any) -> Any:
        del hand, win_tile, context
        raise RuntimeError("the availability-only report does not score hands")


def _policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "future_ukeire_weight": args.future_ukeire_weight,
        "future_ukeire_weight_1_shanten": args.future_ukeire_weight_1_shanten,
        "future_ukeire_weight_2_shanten": args.future_ukeire_weight_2_shanten,
        "improvement_weight": args.improvement_weight,
        "dora_ukeire_weight": args.dora_ukeire_weight,
        "dora_tiebreak_weight": args.dora_tiebreak_weight,
        "dora_neighbor_weight": args.dora_neighbor_weight,
        "yakuhai_weight": args.yakuhai_weight,
        "tanyao_weight": args.tanyao_weight,
        "kanchan_outer_wait_weight": getattr(args, "kanchan_outer_wait_weight", 0.0),
        "shanten_reversal": sim.ShantenReversalConfig(
            enabled=args.allow_shanten_reversal,
            penalty=args.shanten_reversal_penalty,
            recovered_ukeire_weight=args.shanten_reversal_future_weight,
            max_turn=args.shanten_reversal_max_turn,
            candidate_limit=args.shanten_reversal_candidate_limit,
        ),
    }


def _find_native_binary(explicit: Path | None) -> Path | None:
    candidates = [explicit] if explicit is not None else [
        PROJECT_ROOT / "build" / "hand-efficiency" / "hand_efficiency_sim_cpp",
        OUTPUTS_DIR / "hand_efficiency_sim_cpp",
    ]
    return next((path.resolve() for path in candidates if path is not None and path.is_file()), None)


def _native_command(args: argparse.Namespace, binary: Path, observations: Path) -> list[str]:
    command = [
        str(binary),
        "simulate-table" if args.shared_four_player_table else "simulate",
        "--trials", str(args.trials),
        "--seed", str(args.seed),
        "--start-trial", str(args.start_trial),
        "--stop-at", "tenpai",
        "--seat-wind", args.seat_wind,
        "--round-wind", args.round_wind,
        "--policy", args.policy,
        "--future-ukeire-weight", str(args.future_ukeire_weight),
        "--improvement-weight", str(args.improvement_weight),
        "--observations", str(observations),
    ]
    for option, value in (
        ("--future-ukeire-weight-1-shanten", args.future_ukeire_weight_1_shanten),
        ("--future-ukeire-weight-2-shanten", args.future_ukeire_weight_2_shanten),
        ("--dora-ukeire-weight", args.dora_ukeire_weight),
        ("--dora-tiebreak-weight", args.dora_tiebreak_weight),
        ("--dora-neighbor-weight", args.dora_neighbor_weight),
        ("--yakuhai-weight", args.yakuhai_weight),
        ("--tanyao-weight", args.tanyao_weight),
        ("--shanten-reversal-penalty", args.shanten_reversal_penalty),
        ("--shanten-reversal-future-weight", args.shanten_reversal_future_weight),
        ("--shanten-reversal-max-turn", args.shanten_reversal_max_turn),
        ("--shanten-reversal-candidate-limit", args.shanten_reversal_candidate_limit),
    ):
        if value is not None:
            command.extend((option, str(value)))
    if args.allow_shanten_reversal:
        command.append("--allow-shanten-reversal")
    if getattr(args, "kanchan_outer_wait_weight", 0.0):
        command.extend(("--kanchan-outer-wait-weight", str(args.kanchan_outer_wait_weight)))
    if args.continue_rejected_tenpai:
        command.append("--continue-rejected-tenpai")
    tile_availability_turn = getattr(args, "tile_availability_turn", None)
    if tile_availability_turn is not None:
        command.extend(("--tile-availability-turn", str(tile_availability_turn)))
    directives = getattr(args, "_valuable_dama_directives", None)
    if directives is not None:
        command.extend(("--valuable-dama-directives", str(directives)))
    return command


def _pre_riichi_value_dama(
    row: Mapping[str, Any], event: Mapping[str, Any], scorer: sim.ScoreEvaluator,
    threshold_points: int, *, bad_wait_only: bool,
) -> tuple[bool, int]:
    snapshot = event["tenpai"]
    category = MonteCarloReportCollector._snapshot_primary_category(snapshot)
    waits = snapshot.get("waits", ())
    if bad_wait_only:
        bad_wait = category in ("kanchan", "penchan", "shanpon")
        if category in ("shanpon", "tanki"):
            bad_wait = not any(
                int(wait["tile_type"]) >= 27
                and int(wait.get("visible_remaining", 0)) > 0
                for wait in waits
            )
        if not bad_wait:
            return False, 0
    seat = int(event["seat"])
    hand = tuple(int(tile) for tile in snapshot["hand_physical"])
    context = sim.ScoreContext(
        is_tsumo=False,
        is_riichi=False,
        is_ippatsu=False,
        is_dealer=False,
        is_haitei=False,
        is_tenhou=False,
        is_chiihou=False,
        player_wind=27 + seat,
        round_wind=int(row["round_wind"]),
        dora_indicators=(int(row["dora_indicator"]),),
        ura_dora_indicators=(),
    )
    qualifies = False
    ron_mask = 0
    for wait in waits:
        tile_type = int(wait["tile_type"])
        for tile_id in wait.get("visible_remaining_tile_ids", ()):
            score = scorer.score(tuple(sorted((*hand, int(tile_id)))), int(tile_id), context)
            if score.error is not None or score.points is None:
                continue
            ron_mask |= 1 << tile_type
            qualifies |= int(score.points) >= threshold_points
    return qualifies, ron_mask


def _valuable_bad_wait_dama(
    row: Mapping[str, Any], event: Mapping[str, Any], scorer: sim.ScoreEvaluator,
    threshold_points: int,
) -> tuple[bool, int]:
    return _pre_riichi_value_dama(
        row, event, scorer, threshold_points, bad_wait_only=True
    )


def _haneman_minimum_value_dama(
    row: Mapping[str, Any], event: Mapping[str, Any], scorer: sim.ScoreEvaluator,
) -> tuple[bool, int]:
    snapshot = event["tenpai"]
    hand = tuple(int(tile) for tile in snapshot["hand_physical"])
    seat = int(event["seat"])
    base_context = dict(
        is_riichi=False,
        is_ippatsu=False,
        is_dealer=False,
        is_haitei=False,
        is_tenhou=False,
        is_chiihou=False,
        player_wind=27 + seat,
        round_wind=int(row["round_wind"]),
        dora_indicators=(int(row["dora_indicator"]),),
        ura_dora_indicators=(),
    )
    ron_context = sim.ScoreContext(is_tsumo=False, **base_context)
    tsumo_context = sim.ScoreContext(is_tsumo=True, **base_context)
    ron_mask = 0
    winning_copies = 0
    for wait in snapshot.get("waits", ()):
        tile_type = int(wait["tile_type"])
        tile_has_ron_yaku = False
        for tile_id in wait.get("visible_remaining_tile_ids", ()):
            winning_copies += 1
            complete = tuple(sorted((*hand, int(tile_id))))
            ron = scorer.score(complete, int(tile_id), ron_context)
            if ron.error is None and ron.points is not None:
                if int(ron.points) < 12000:
                    return False, 0
                tile_has_ron_yaku = True
                continue
            tsumo = scorer.score(complete, int(tile_id), tsumo_context)
            if tsumo.error is not None or tsumo.points is None or int(tsumo.points) < 12000:
                return False, 0
        if tile_has_ron_yaku:
            ron_mask |= 1 << tile_type
    return winning_copies > 0, ron_mask


def _discover_valuable_dama_directives(
    path: Path, scorer: sim.ScoreEvaluator, threshold_points: int | None,
    existing: dict[str, int], reasons: dict[str, str], *, haneman_or_higher: bool,
) -> int:
    added = 0
    probe = MonteCarloReportCollector(
        scorer, score_tenpai=False, reject_cheap_bad_wait_riichi=True
    )
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            active_valuable_masks: dict[int, int] = {}
            active_reasons: dict[int, str] = {}
            for event in row.get("table_events", ()):
                kind = str(event.get("kind", ""))
                if kind in ("dama_ron", "dama_tsumo"):
                    break
                if kind != "tenpai":
                    continue
                snapshot = event["tenpai"]
                seat = int(event["seat"])
                key = (
                    f'{int(row["trial_id"])}:{seat}:'
                    f'{int(event.get("turn", 0))}:'
                    f'{snapshot.get("hand", sim.format_counts(snapshot["hand_counts"]))}'
                )
                if key in existing:
                    if existing[key] & DAMA_LOCK_FLAG:
                        active_valuable_masks[seat] = existing[key] & DAMA_WAIT_MASK
                        active_reasons[seat] = reasons.get(key, "valuable_bad_wait")
                        continue
                    else:
                        active_valuable_masks.pop(seat, None)
                        active_reasons.pop(seat, None)
                qualifies = False
                ron_mask = 0
                reason = ""
                if haneman_or_higher:
                    haneman_qualifies, haneman_ron_mask = _haneman_minimum_value_dama(
                        row, event, scorer
                    )
                    if haneman_qualifies:
                        qualifies = True
                        ron_mask = haneman_ron_mask
                        reason = "haneman_or_higher"
                if not qualifies and threshold_points is not None:
                    qualifies, ron_mask = _valuable_bad_wait_dama(
                        row, event, scorer, threshold_points
                    )
                    if qualifies:
                        reason = "valuable_bad_wait"
                if qualifies:
                    if (
                        active_valuable_masks.get(seat) == ron_mask
                        and active_reasons.get(seat) == reason
                    ):
                        continue
                    existing[key] = DAMA_LOCK_FLAG | ron_mask
                    if reason == "haneman_or_higher":
                        existing[key] |= DAMA_TSUMOGIRI_FLAG
                    reasons[key] = reason
                    added += 1
                    break
                if seat in active_valuable_masks:
                    category = MonteCarloReportCollector._snapshot_primary_category(snapshot)
                    remains_bad_wait = category in ("kanchan", "penchan", "shanpon")
                    if category in ("shanpon", "tanki"):
                        remains_bad_wait = not any(
                            int(wait["tile_type"]) >= 27
                            and int(wait.get("visible_remaining", 0)) > 0
                            for wait in snapshot.get("waits", ())
                        )
                    if (
                        active_reasons.get(seat) == "valuable_bad_wait"
                        and remains_bad_wait
                    ):
                        # Reject a bad-to-bad shape change that no longer has a
                        # valid pre-riichi yaku at the original value line.  On
                        # the next native pass this directive makes the player
                        # discard the drawn tile and retain the previous hand.
                        existing[key] = DAMA_REJECT_TRANSITION_FLAG
                    else:
                        # A good (or otherwise non-bad) wait is eligible to
                        # leave dama and end the race normally.
                        existing[key] = 0
                    reasons.pop(key, None)
                    added += 1
                    break
                candidate = dict(row)
                candidate.update({
                    "player_wind": 27 + int(event["seat"]),
                    "first_tenpai": snapshot,
                })
                if probe._reject_cheap_bad_wait(candidate, snapshot):
                    continue
                break
    return added


def _consume_jsonl(
    path: Path,
    collector: MonteCarloReportCollector,
    *,
    continue_rejected_tenpai: bool = False,
    availability_collector: TileAvailabilityCollector | None = None,
) -> None:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if availability_collector is not None:
                    availability_collector.add_row(row)
                if continue_rejected_tenpai:
                    collector.add_continuation_row(row)
                else:
                    collector.add_row(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid observation at {path}:{line_number}: {error}") from error


def _run_native(
    args: argparse.Namespace,
    binary: Path,
    observations: Path,
    collector: MonteCarloReportCollector,
    availability_collector: TileAvailabilityCollector | None = None,
) -> dict[str, Any]:
    observations.parent.mkdir(parents=True, exist_ok=True)
    threshold_name = args.valuable_bad_wait_dama_line
    directives: dict[str, int] = {}
    directive_reasons: dict[str, str] = {}
    completed: subprocess.CompletedProcess[str]
    if threshold_name is None and not args.haneman_or_higher_dama:
        completed = subprocess.run(
            _native_command(args, binary, observations),
            check=True, capture_output=True, text=True,
        )
    else:
        threshold_points = (
            None if threshold_name is None
            else 5200 if threshold_name == "3han40fu" else 8000
        )
        directives_path = observations.with_suffix(".dama-directives.txt")
        for _pass in range(8):
            directives_path.write_text(
                "".join(f"{key} {mask}\n" for key, mask in sorted(directives.items())),
                encoding="utf-8",
            )
            args._valuable_dama_directives = directives_path
            completed = subprocess.run(
                _native_command(args, binary, observations),
                check=True, capture_output=True, text=True,
            )
            added = _discover_valuable_dama_directives(
                observations, collector.scorer, threshold_points, directives,
                directive_reasons, haneman_or_higher=args.haneman_or_higher_dama,
            )
            if not added:
                break
        else:
            raise RuntimeError("valuable dama directive discovery did not converge")
        collector.valuable_dama_directives = dict(directives)
        collector.valuable_dama_directive_reasons = dict(directive_reasons)
    summary = json.loads(completed.stdout)
    _consume_jsonl(
        observations, collector,
        continue_rejected_tenpai=args.continue_rejected_tenpai,
        availability_collector=availability_collector,
    )
    return {
        "engine": "native",
        "binary": str(binary),
        "engine_version": summary.get("engine_version"),
        "evaluator_backend": summary.get("evaluator_backend"),
        "valuable_dama_directives": len(directives),
    }


def _run_python(
    args: argparse.Namespace,
    observations: Path | None,
    collector: MonteCarloReportCollector,
    availability_collector: TileAvailabilityCollector | None = None,
) -> dict[str, Any]:
    if availability_collector is not None:
        raise ValueError("tile availability requires the native shared-table engine")
    evaluator = sim.make_policy_evaluator(args.policy, **_policy_kwargs(args))
    config = sim.SimulationConfig(
        stop_at="tenpai",
        riichi_on_tenpai=True,
        player_wind=27 + WIND_NAMES.index(args.seat_wind),
        round_wind=27 + WIND_NAMES.index(args.round_wind),
    )
    stream: TextIO | None = None
    try:
        if observations is not None:
            observations.parent.mkdir(parents=True, exist_ok=True)
            stream = observations.open("w", encoding="utf-8")
        for trial_id in range(args.start_trial, args.start_trial + args.trials):
            result = sim.simulate_trial(
                trial_id,
                args.seed,
                evaluator=evaluator,
                scorer=None,
                config=config,
            )
            row = result.to_dict()
            # Native rows carry these explicitly; add them for one collector contract.
            row["player_wind"] = config.player_wind
            row["round_wind"] = config.round_wind
            collector.add_row(row)
            if stream is not None:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        if stream is not None:
            stream.close()
    return {
        "engine": "python",
        "engine_version": sim.ENGINE_VERSION,
        "evaluator_backend": "riichienv" if sim._fast_shanten_function() is not None else "mahjong",
    }


def run_report(args: argparse.Namespace, scorer: sim.ScoreEvaluator | None = None) -> dict[str, Any]:
    if args.trials < 0 or args.start_trial < 0:
        raise ValueError("trials and start-trial must be nonnegative")
    if args.seat_wind == "east":
        raise ValueError("average non-dealer ron score requires south, west, or north seat")
    tile_availability_turn = getattr(args, "tile_availability_turn", None)
    if tile_availability_turn is not None:
        if tile_availability_turn < 0 or tile_availability_turn > 17:
            raise ValueError("tile availability turn must be between 0 and 17")
        if not args.shared_four_player_table:
            raise ValueError("tile availability requires --shared-four-player-table")
    if scorer is None:
        scorer = (
            _AvailabilityOnlyScorer()
            if tile_availability_turn is not None else sim.MahjongRepositoryScorer()
        )
    availability_collector = (
        TileAvailabilityCollector(tile_availability_turn, default_round_wind=args.round_wind)
        if tile_availability_turn is not None else None
    )
    collector = MonteCarloReportCollector(
        scorer,
        score_tenpai=not args.shared_four_player_table,
        reject_cheap_bad_wait_riichi=args.reject_cheap_bad_wait_riichi,
    )
    native_binary = _find_native_binary(args.native_binary)
    engine = args.engine
    if engine == "auto":
        engine = "native" if native_binary is not None else "python"
    if engine == "native" and native_binary is None:
        raise ValueError("native engine requested but no binary was found; pass --native-binary")
    if args.shared_four_player_table and engine != "native":
        raise ValueError("shared four-player table mode requires the native engine")
    if args.reject_cheap_bad_wait_riichi and not args.shared_four_player_table:
        raise ValueError("cheap bad-wait riichi rejection requires shared four-player table mode")
    if args.continue_rejected_tenpai and not args.reject_cheap_bad_wait_riichi:
        raise ValueError("continued rejected tenpai requires --reject-cheap-bad-wait-riichi")
    if args.valuable_bad_wait_dama_line and not args.continue_rejected_tenpai:
        raise ValueError("valuable bad-wait dama requires --continue-rejected-tenpai")
    if args.haneman_or_higher_dama and not args.continue_rejected_tenpai:
        raise ValueError("haneman-or-higher dama requires --continue-rejected-tenpai")

    run_metadata: dict[str, Any]
    if engine == "native":
        assert native_binary is not None
        if args.observations is not None:
            run_metadata = _run_native(
                args, native_binary, args.observations, collector, availability_collector
            )
        else:
            with tempfile.TemporaryDirectory(prefix="hand-monte-carlo-") as directory:
                run_metadata = _run_native(
                    args,
                    native_binary,
                    Path(directory) / "trials.jsonl",
                    collector,
                    availability_collector,
                )
    else:
        run_metadata = _run_python(args, args.observations, collector, availability_collector)

    evaluator = sim.make_policy_evaluator(args.policy, **_policy_kwargs(args))
    availability_mode = tile_availability_turn is not None
    run_metadata.update({
        "seed": args.seed,
        "start_trial": args.start_trial,
        "trials": args.trials,
        "seat_wind": args.seat_wind,
        "round_wind": args.round_wind,
        "riichi_on_tenpai": True,
        "score_assumes_riichi": not availability_mode,
        "ura_dora_sampled": not availability_mode,
        "ura_dora_hidden_from_discard_policy": not availability_mode,
        "policy": evaluator.name,
        "policy_parameters": evaluator.parameters(),
        "scorer": scorer.name,
        "observations": str(args.observations) if args.observations is not None else None,
        "simulation_scope": (
            "shared_four_player_availability_checkpoint"
            if tile_availability_turn is not None
            else "shared_four_player_continue_rejected_tenpai"
            if args.continue_rejected_tenpai
            else "shared_four_player_first_tenpai"
            if args.shared_four_player_table else "single_hand_first_tenpai"
        ),
        "tenpai_scoring_enabled": not args.shared_four_player_table,
        "reject_cheap_bad_wait_riichi": args.reject_cheap_bad_wait_riichi,
        "continue_rejected_tenpai": args.continue_rejected_tenpai,
        "valuable_bad_wait_dama_line": args.valuable_bad_wait_dama_line,
        "haneman_or_higher_dama": args.haneman_or_higher_dama,
    })
    if tile_availability_turn is not None:
        run_metadata["tile_availability_turn"] = tile_availability_turn
    payload = collector.to_dict(run=run_metadata)
    if availability_collector is not None:
        payload["tile_availability"] = availability_collector.to_dict()
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=sim.DEFAULT_SEED)
    parser.add_argument("--start-trial", type=int, default=0)
    parser.add_argument("--engine", choices=("auto", "python", "native"), default="auto")
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--seat-wind", choices=("south", "west", "north"), default="south")
    parser.add_argument("--round-wind", choices=WIND_NAMES, default="east")
    parser.add_argument(
        "--tile-availability-turn",
        type=int,
        help=(
            "collect a synchronized four-player native availability checkpoint after this "
            "many own turns (for example, 9)"
        ),
    )
    parser.add_argument("--policy", choices=("pure", "value", "custom"), default="pure")
    parser.add_argument(
        "--shared-four-player-table",
        action="store_true",
        help=(
            "run four same-policy players on one table; normally stop at the first accepted "
            "tenpai, or continue to the synchronized checkpoint in tile-availability mode"
        ),
    )
    parser.add_argument(
        "--reject-cheap-bad-wait-riichi",
        action="store_true",
        help=(
            "exclude nondealer 1300-point riichi hands with kanchan, penchan, or "
            "shanpon/tanki without a visible remaining honor wait copy from shared-table statistics"
        ),
    )
    parser.add_argument(
        "--continue-rejected-tenpai",
        action="store_true",
        help=(
            "continue after rejected shared-table tenpai; stay dama, improve the hand, "
            "and stop separately on a later tsumo"
        ),
    )
    parser.add_argument(
        "--valuable-bad-wait-dama-line",
        choices=("3han40fu", "4han40fu"),
        help=(
            "keep qualifying bad waits dama at a pre-riichi nondealer-ron value line; "
            "3han40fu also admits 4han25fu chiitoitsu by point value"
        ),
    )
    parser.add_argument(
        "--haneman-or-higher-dama",
        action="store_true",
        help=(
            "lock the hand and tsumogiri when every unseen winning tile is worth at "
            "least 12000 points by pre-riichi ron with yaku, or by closed-hand tsumo "
            "when ron has no yaku"
        ),
    )
    parser.add_argument("--future-ukeire-weight", type=float, default=0.5)
    parser.add_argument("--future-ukeire-weight-1-shanten", type=float)
    parser.add_argument("--future-ukeire-weight-2-shanten", type=float)
    parser.add_argument("--improvement-weight", type=float, default=0.1)
    parser.add_argument("--dora-ukeire-weight", type=float)
    parser.add_argument("--dora-tiebreak-weight", type=float)
    parser.add_argument("--dora-neighbor-weight", type=float)
    parser.add_argument("--yakuhai-weight", type=float)
    parser.add_argument("--tanyao-weight", type=float)
    parser.add_argument("--kanchan-outer-wait-weight", type=float, default=0.0)
    parser.add_argument("--allow-shanten-reversal", action="store_true")
    parser.add_argument("--shanten-reversal-penalty", type=float, default=25.0)
    parser.add_argument("--shanten-reversal-future-weight", type=float, default=0.5)
    parser.add_argument("--shanten-reversal-max-turn", type=int, default=9)
    parser.add_argument("--shanten-reversal-candidate-limit", type=int, default=2)
    parser.add_argument("--observations", type=Path, help="optionally retain trial JSONL")
    parser.add_argument("--output", type=Path, help="write the aggregate JSON report")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_report(args)
    text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
