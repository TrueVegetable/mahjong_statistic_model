#!/usr/bin/env python3
"""Seeded closed-hand simulator for tile-efficiency research.

After every draw, the policy keeps the lowest-shanten hand.  Among equally
low-shanten hands it maximizes a visible-information ukeire score.  The score
adds shallow future and improvement terms at 1- and 2-shanten; both are
tunable, and neither recurses.  Optional dora, dora-neighbor, yakuhai, and
tanyao retention features provide a cheap value-aware policy layer.  Ties
between tile *types* are sampled.

The simulator is intended to generate inputs for wait-type, tile-danger, and
hand-value models.  It therefore keeps policy evaluation, hand scoring, trial
generation, and aggregation behind separate interfaces.  Calls, kans, ron
events, opponents' decisions, and placement utility are intentionally outside
this first structural layer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, TextIO


ENGINE_VERSION = "closed-hand-efficiency-4"
TILE_TYPE_COUNT = 34
PHYSICAL_TILE_COUNT = 136
INITIAL_HAND_TILES = 13
OPPONENT_HAND_TILES = 39
DEAD_WALL_TILES = 14
LIVE_WALL_TILES = 70
DEFAULT_SEED = 20260813
MASK_64 = (1 << 64) - 1
RED_FIVE_IDS = frozenset((16, 52, 88))
PHYSICAL_COPIES_BY_TYPE = tuple(tuple(tile_type * 4 + copy for copy in range(4)) for tile_type in range(34))
TERMINAL_HONOR_TYPES = frozenset((0, 8, 9, 17, 18, 26, *range(27, 34)))
WAIT_SHAPE_ORDER = {
    "ryanmen": 0,
    "kanchan": 1,
    "penchan": 2,
    "shanpon": 3,
    "tanki": 4,
    "chiitoitsu_tanki": 5,
    "kokushi_13way": 6,
    "kokushi_single": 7,
}


def tile_name(tile_type: int) -> str:
    """Return a 34-format tile type in compact mpsz notation."""
    if not 0 <= tile_type < TILE_TYPE_COUNT:
        raise ValueError(f"tile type must be 0..33, got {tile_type}")
    if tile_type < 27:
        return f"{tile_type % 9 + 1}{'mps'[tile_type // 9]}"
    return f"{tile_type - 26}z"


def physical_tile_name(tile_id: int) -> str:
    """Return mpsz notation for one of the 136 physical tile IDs."""
    if not 0 <= tile_id < PHYSICAL_TILE_COUNT:
        raise ValueError(f"physical tile must be 0..135, got {tile_id}")
    if tile_id in RED_FIVE_IDS:
        return f"0{'mps'[tile_id // 36]}"
    return tile_name(tile_id // 4)


def counts_from_types(tile_types: Iterable[int]) -> tuple[int, ...]:
    counts = [0] * TILE_TYPE_COUNT
    for tile_type in tile_types:
        if not 0 <= tile_type < TILE_TYPE_COUNT:
            raise ValueError(f"tile type must be 0..33, got {tile_type}")
        counts[tile_type] += 1
        if counts[tile_type] > 4:
            raise ValueError(f"more than four copies of {tile_name(tile_type)}")
    return tuple(counts)


def counts_from_physical(tile_ids: Iterable[int]) -> tuple[int, ...]:
    return counts_from_types(tile_id // 4 for tile_id in tile_ids)


def format_counts(counts: Sequence[int]) -> str:
    """Format a 34-count hand as canonical compact mpsz notation."""
    _validate_counts(counts)
    groups: list[str] = []
    for suit, suffix in enumerate("mps"):
        digits = "".join(str(rank + 1) * counts[suit * 9 + rank] for rank in range(9))
        if digits:
            groups.append(digits + suffix)
    honors = "".join(str(rank + 1) * counts[27 + rank] for rank in range(7))
    if honors:
        groups.append(honors + "z")
    return "".join(groups)


def parse_hand(text: str) -> tuple[int, ...]:
    """Parse compact mpsz notation; red zeroes are treated as five types."""
    tiles: list[int] = []
    digits: list[str] = []
    for char in text.lower().replace(" ", ""):
        if char.isdigit():
            digits.append(char)
            continue
        if char not in "mpsz" or not digits:
            raise ValueError(f"invalid hand notation near {char!r}: {text!r}")
        suit = "mpsz".index(char)
        for digit in digits:
            rank = int(digit)
            if suit == 3:
                if not 1 <= rank <= 7:
                    raise ValueError("honor ranks must be 1..7")
                tiles.append(26 + rank)
            else:
                if rank == 0:
                    rank = 5
                if not 1 <= rank <= 9:
                    raise ValueError("suited ranks must be 0..9")
                tiles.append(suit * 9 + rank - 1)
        digits.clear()
    if digits:
        raise ValueError("hand notation ends with digits but no suit")
    return counts_from_types(tiles)


def parse_physical_hand(text: str) -> tuple[int, ...]:
    """Parse a hand while preserving explicit red-five zero notation."""
    counts = parse_hand(text)
    red_types: Counter[int] = Counter()
    digits: list[str] = []
    for char in text.lower().replace(" ", ""):
        if char.isdigit():
            digits.append(char)
            continue
        suit = "mpsz".index(char)
        if suit < 3:
            red_types[suit * 9 + 4] += digits.count("0")
        digits.clear()
    if any(count > 1 for count in red_types.values()):
        raise ValueError("at most one red five exists in each suit")

    physical: list[int] = []
    for tile_type, count in enumerate(counts):
        copies = list(PHYSICAL_COPIES_BY_TYPE[tile_type])
        red_id = tile_type * 4 if tile_type in (4, 13, 22) else None
        if red_id is not None:
            ordinary = [tile_id for tile_id in copies if tile_id != red_id]
            if red_types[tile_type]:
                physical.append(red_id)
                physical.extend(ordinary[: count - 1])
            else:
                physical.extend(ordinary[:count])
                if count == 4:
                    physical.append(red_id)
        else:
            physical.extend(copies[:count])
    return tuple(physical)


def parse_single_tile_type(text: str) -> int:
    counts = parse_hand(text)
    if sum(counts) != 1:
        raise ValueError(f"expected exactly one tile, got {text!r}")
    return next(tile_type for tile_type, count in enumerate(counts) if count)


def _validate_counts(counts: Sequence[int], expected_total: int | None = None) -> None:
    if len(counts) != TILE_TYPE_COUNT:
        raise ValueError(f"expected 34 tile counts, got {len(counts)}")
    if any(not isinstance(count, int) or not 0 <= count <= 4 for count in counts):
        raise ValueError("every tile count must be an integer from 0 through 4")
    if expected_total is not None and sum(counts) != expected_total:
        raise ValueError(f"expected {expected_total} tiles, got {sum(counts)}")


def _require_mahjong() -> Any:
    try:
        import mahjong  # noqa: F401
        from mahjong.shanten import Shanten
    except ImportError as exc:  # pragma: no cover - exercised by CLI environments
        raise RuntimeError(
            "The hand simulator requires mahjong==2.0.0. "
            "Install outputs/requirements-hand-simulator.txt."
        ) from exc
    return Shanten


@lru_cache(maxsize=1)
def _fast_shanten_function() -> Any | None:
    """Load RiichiEnv's Rust lookup-table calculator when it is installed."""
    try:
        from riichienv import calculate_shanten
    except ImportError:
        return None
    return calculate_shanten


def _canonical_physical_hand(counts: Sequence[int]) -> list[int]:
    physical: list[int] = []
    for tile_type, count in enumerate(counts):
        if count:
            physical.extend(PHYSICAL_COPIES_BY_TYPE[tile_type][:count])
    return physical


@dataclass(frozen=True)
class ShantenBreakdown:
    standard: int
    chiitoitsu: int
    kokushi: int
    minimum: int
    best_forms: tuple[str, ...]


@lru_cache(maxsize=500_000)
def shanten_breakdown(counts: tuple[int, ...]) -> ShantenBreakdown:
    """Calculate standard, seven-pairs, and thirteen-orphans shanten."""
    _validate_counts(counts)
    if sum(counts) not in (13, 14):
        raise ValueError("version 1 evaluates closed 13- or 14-tile hands only")
    analyzer = _require_mahjong()
    standard = analyzer.calculate_shanten_for_regular_hand(counts)
    chiitoitsu = analyzer.calculate_shanten_for_chiitoitsu_hand(counts)
    kokushi = analyzer.calculate_shanten_for_kokushi_hand(counts)
    minimum = min(standard, chiitoitsu, kokushi)
    # Cross-check the fast production path whenever a detailed result is
    # requested.  This makes backend/version drift fail loudly in analysis and
    # tests without putting the Python decomposition on every Monte Carlo call.
    if shanten(counts) != minimum:
        raise RuntimeError("RiichiEnv and MahjongRepository shanten results disagree")
    values = {"standard": standard, "chiitoitsu": chiitoitsu, "kokushi": kokushi}
    return ShantenBreakdown(
        standard=standard,
        chiitoitsu=chiitoitsu,
        kokushi=kokushi,
        minimum=minimum,
        best_forms=tuple(name for name, value in values.items() if value == minimum),
    )


@lru_cache(maxsize=500_000)
def _shanten_cached(counts: tuple[int, ...]) -> int:
    fast = _fast_shanten_function()
    if fast is not None:
        return int(fast(_canonical_physical_hand(counts)))
    # Correct but substantially slower fallback, useful for analysis on a
    # platform for which RiichiEnv does not publish a wheel.
    return int(_require_mahjong().calculate_shanten(counts))


def shanten(counts: Sequence[int]) -> int:
    counts_tuple = tuple(counts)
    _validate_counts(counts_tuple)
    if sum(counts_tuple) not in (13, 14):
        raise ValueError("version 1 evaluates closed 13- or 14-tile hands only")
    return _shanten_cached(counts_tuple)


@lru_cache(maxsize=500_000)
def structural_ukeire_types(counts: tuple[int, ...]) -> tuple[int, ...]:
    """Return tile types which reduce shanten, without applying visibility."""
    _validate_counts(counts, expected_total=13)
    current = _shanten_cached(counts)
    result: list[int] = []
    fast = _fast_shanten_function()
    physical = _canonical_physical_hand(counts) if fast is not None else None
    for tile_type, count in enumerate(counts):
        if count == 4:
            continue
        if fast is not None:
            assert physical is not None
            physical.append(tile_type * 4 + count)
            after_shanten = int(fast(physical))
            physical.pop()
        else:
            after = list(counts)
            after[tile_type] += 1
            after_shanten = _shanten_cached(tuple(after))
        if after_shanten < current:
            result.append(tile_type)
    return tuple(result)


def _meld_decompositions(counts: tuple[int, ...]) -> tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]:
    """Enumerate complete standard-hand meld decompositions after removing a pair."""
    return _meld_decompositions_cached(counts)


@lru_cache(maxsize=100_000)
def _meld_decompositions_cached(
    counts: tuple[int, ...],
) -> tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]:
    try:
        first = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        return ((),)

    results: list[tuple[tuple[str, tuple[int, ...]], ...]] = []
    mutable = list(counts)
    if mutable[first] >= 3:
        mutable[first] -= 3
        for rest in _meld_decompositions_cached(tuple(mutable)):
            results.append((("triplet", (first, first, first)),) + rest)
        mutable[first] += 3

    rank = first % 9
    if first < 27 and rank <= 6 and mutable[first + 1] and mutable[first + 2]:
        mutable[first] -= 1
        mutable[first + 1] -= 1
        mutable[first + 2] -= 1
        for rest in _meld_decompositions_cached(tuple(mutable)):
            results.append((("sequence", (first, first + 1, first + 2)),) + rest)
    return tuple(results)


@lru_cache(maxsize=100_000)
def standard_decompositions(
    counts: tuple[int, ...],
) -> tuple[tuple[int, tuple[tuple[str, tuple[int, ...]], ...]], ...]:
    """Return every pair-plus-four-meld decomposition of a closed complete hand."""
    _validate_counts(counts, expected_total=14)
    results: list[tuple[int, tuple[tuple[str, tuple[int, ...]], ...]]] = []
    for pair, count in enumerate(counts):
        if count < 2:
            continue
        remainder = list(counts)
        remainder[pair] -= 2
        for melds in _meld_decompositions(tuple(remainder)):
            if len(melds) == 4:
                results.append((pair, melds))
    return tuple(results)


def _is_chiitoitsu_agari(counts: Sequence[int]) -> bool:
    return sum(counts) == 14 and sum(count == 2 for count in counts) == 7


def _is_kokushi_agari(counts: Sequence[int]) -> bool:
    return (
        sum(counts) == 14
        and all(counts[index] >= 1 for index in TERMINAL_HONOR_TYPES)
        and sum(counts[index] for index in TERMINAL_HONOR_TYPES) == 14
        and any(counts[index] >= 2 for index in TERMINAL_HONOR_TYPES)
    )


@lru_cache(maxsize=100_000)
def wait_shapes(hand_13: tuple[int, ...], winning_type: int) -> tuple[str, ...]:
    """Classify every valid interpretation of one winning tile.

    Complex hands can give the same tile more than one valid wait shape.  The
    function retains all such labels instead of silently imposing an arbitrary
    decomposition.
    """
    _validate_counts(hand_13, expected_total=13)
    if not 0 <= winning_type < TILE_TYPE_COUNT or hand_13[winning_type] == 4:
        return ()
    complete = list(hand_13)
    complete[winning_type] += 1
    complete_tuple = tuple(complete)
    if shanten(complete_tuple) != -1:
        return ()

    shapes: set[str] = set()
    if _is_chiitoitsu_agari(complete):
        shapes.add("chiitoitsu_tanki")
    if _is_kokushi_agari(complete):
        distinct_before = sum(hand_13[index] > 0 for index in TERMINAL_HONOR_TYPES)
        shapes.add("kokushi_13way" if distinct_before == 13 else "kokushi_single")

    for pair, melds in standard_decompositions(complete_tuple):
        if pair == winning_type:
            shapes.add("tanki")
        for kind, tiles in melds:
            if winning_type not in tiles:
                continue
            if kind == "triplet":
                shapes.add("shanpon")
                continue
            start = tiles[0] % 9 + 1
            win_rank = winning_type % 9 + 1
            if win_rank == start + 1:
                shapes.add("kanchan")
            elif (start == 1 and win_rank == 3) or (start == 7 and win_rank == 7):
                shapes.add("penchan")
            else:
                shapes.add("ryanmen")
    return tuple(sorted(shapes, key=WAIT_SHAPE_ORDER.__getitem__))


@dataclass(frozen=True)
class DecisionContext:
    turn: int
    hand: tuple[int, ...]
    river: tuple[int, ...]
    visible_tile_ids: tuple[int, ...]
    drawn_tile: int
    dora_indicator: int
    player_wind: int = 28
    round_wind: int = 27
    shanten_reversals_used: int = 0


@dataclass(frozen=True)
class ShantenReversalConfig:
    enabled: bool = False
    penalty: float = 25.0
    recovered_ukeire_weight: float = 0.5
    max_turn: int = 9
    candidate_limit: int = 2
    max_per_hand: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.penalty) or self.penalty < 0.0:
            raise ValueError("shanten reversal penalty must be finite and nonnegative")
        if not math.isfinite(self.recovered_ukeire_weight) or self.recovered_ukeire_weight < 0.0:
            raise ValueError("shanten reversal future weight must be finite and nonnegative")
        if self.max_turn < 0 or self.candidate_limit < 1 or self.max_per_hand < 1:
            raise ValueError("shanten reversal limits must be positive (max turn may be zero)")


@dataclass(frozen=True)
class CandidateFeatures:
    discard_type: int
    shanten: int
    shanten_forms: tuple[str, ...]
    ukeire_types: tuple[int, ...]
    ukeire_remaining: tuple[int, ...]
    ukeire_count: int
    future_ukeire_by_type: tuple[float, ...]
    future_ukeire_expected: float
    improvement_types: tuple[int, ...]
    improvement_remaining: tuple[int, ...]
    improvement_count: int
    ukeire_score: float
    ukeire_computed: bool
    base_ukeire_score: float = 0.0
    dora_ukeire_bonus: float = 0.0
    retained_dora: int = 0
    retained_dora_neighbors: int = 0
    retained_yakuhai_value: int = 0
    retained_tanyao_tiles: int = 0
    value_tiebreak_score: float = 0.0
    strict_best_shanten: int = 99
    reversal_eligible: bool = False
    reversal_deep_evaluated: bool = False
    is_shanten_reversal: bool = False
    reversal_recovery_count: int = 0
    reversal_future_ukeire_expected: float = 0.0
    reversal_current_ukeire: int = 0
    reversal_ukeire_gain: float = 0.0
    reversal_adjusted_score: float = -math.inf


class DiscardEvaluator(Protocol):
    """Policy hook.  Larger lexicographic ranks are preferred."""

    name: str

    def rank(self, features: CandidateFeatures, context: DecisionContext) -> tuple[float, ...]: ...


class PureTileEfficiencyEvaluator:
    """Minimize shanten, then maximize a shallow-lookahead ukeire score.

    For 1- and 2-shanten hands the default score is::

        immediate ukeire
        + 0.5 * E[next-step immediate ukeire | an immediate acceptance]
        + 0.1 * improvement-tile copies

    The expectation is weighted by the visible-information remaining count of
    each immediate acceptance tile.  An improvement tile does not lower
    shanten, but after the best pure-efficiency discard it strictly increases
    immediate ukeire.  Neither component recurses beyond that one draw.
    """

    name = "shanten_then_ukeire_score"
    # Lets the engine skip ukeire for a discard once another discard is known
    # to have lower shanten.  General evaluators receive complete features.
    shanten_dominates = True

    def __init__(
        self,
        *,
        future_ukeire_weight: float = 0.5,
        future_ukeire_weight_1_shanten: float | None = None,
        future_ukeire_weight_2_shanten: float | None = None,
        improvement_weight: float = 0.1,
        shanten_reversal: ShantenReversalConfig | None = None,
    ) -> None:
        supplied_weights = (
            future_ukeire_weight,
            future_ukeire_weight_1_shanten,
            future_ukeire_weight_2_shanten,
        )
        if any(weight is not None and weight < 0.0 for weight in supplied_weights) or improvement_weight < 0.0:
            raise ValueError("ukeire-score weights must be nonnegative")
        self.future_ukeire_weight = float(future_ukeire_weight)
        self.future_ukeire_weights_by_shanten = {
            "1": float(future_ukeire_weight if future_ukeire_weight_1_shanten is None else future_ukeire_weight_1_shanten),
            "2": float(future_ukeire_weight if future_ukeire_weight_2_shanten is None else future_ukeire_weight_2_shanten),
            "3_plus": 0.0,
        }
        self.improvement_weight = float(improvement_weight)
        self.needs_future_ukeire = any(self.future_ukeire_weights_by_shanten.values())
        self.needs_improvement_ukeire = self.improvement_weight != 0.0
        self.shanten_reversal = shanten_reversal or ShantenReversalConfig()

    def parameters(self) -> dict[str, Any]:
        return {
            "future_ukeire_weight": self.future_ukeire_weight,
            "future_ukeire_weights_by_shanten": self.future_ukeire_weights_by_shanten,
            "improvement_weight": self.improvement_weight,
            "advanced_shanten": [1, 2],
            "shanten_reversal": asdict(self.shanten_reversal),
        }

    def future_weight_for_shanten(self, shanten_value: int) -> float:
        if shanten_value == 1:
            return self.future_ukeire_weights_by_shanten["1"]
        if shanten_value == 2:
            return self.future_ukeire_weights_by_shanten["2"]
        return 0.0

    def compose_ukeire_score(self, features: CandidateFeatures) -> float:
        return (
            features.ukeire_count
            + self.future_weight_for_shanten(features.shanten) * features.future_ukeire_expected
            + self.improvement_weight * features.improvement_count
        )

    def rank(self, features: CandidateFeatures, context: DecisionContext) -> tuple[float, ...]:
        del context
        return (-float(features.shanten), features.ukeire_score)


def dora_type_from_indicator(indicator_type: int) -> int:
    """Map one 34-format indicator to its actual dora tile type."""
    if not 0 <= indicator_type < TILE_TYPE_COUNT:
        raise ValueError(f"indicator type must be 0..33, got {indicator_type}")
    if indicator_type < 27:
        suit_start = indicator_type // 9 * 9
        return suit_start + (indicator_type % 9 + 1) % 9
    if indicator_type < 31:  # winds cycle East -> South -> West -> North
        return 27 + (indicator_type - 27 + 1) % 4
    return 31 + (indicator_type - 31 + 1) % 3  # dragons White -> Green -> Red


def yakuhai_value(tile_type: int, player_wind: int, round_wind: int) -> int:
    """Return 2 for double wind, 1 for one value source, otherwise 0."""
    if tile_type in range(31, 34):
        return 1
    if tile_type in range(27, 31):
        return int(tile_type == player_wind) + int(tile_type == round_wind)
    return 0


def _is_tanyao_tile(tile_type: int) -> bool:
    return tile_type < 27 and 1 <= tile_type % 9 <= 7


def _dora_neighbor_types(dora_type: int | None) -> frozenset[int]:
    if dora_type is None or dora_type >= 27:
        return frozenset()
    suit_start = dora_type // 9 * 9
    rank = dora_type % 9
    return frozenset(
        suit_start + neighbor
        for neighbor in (rank - 1, rank + 1)
        if 0 <= neighbor <= 8
    )


def _preferred_discard_copies(
    hand: Sequence[int], discard_type: int, *, preserve_red_five: bool
) -> list[int]:
    copies = [tile_id for tile_id in hand if tile_id // 4 == discard_type]
    if preserve_red_five:
        ordinary = [tile_id for tile_id in copies if tile_id not in RED_FIVE_IDS]
        if ordinary:
            return ordinary
    return copies


class HandValueHeuristicEvaluator(PureTileEfficiencyEvaluator):
    """Pure efficiency plus configurable, deliberately shallow value hints.

    Shanten remains the first lexicographic component. ``dora_ukeire_weight``
    can trade immediate/future ukeire against retained dora only within the
    same shanten. The remaining features are combined into a tie score that is
    considered only when adjusted ukeire scores are equal.
    """

    name = "shanten_then_value_ukeire"
    preserve_red_five = False

    def __init__(
        self,
        *,
        future_ukeire_weight: float = 0.5,
        future_ukeire_weight_1_shanten: float | None = None,
        future_ukeire_weight_2_shanten: float | None = None,
        improvement_weight: float = 0.1,
        shanten_reversal: ShantenReversalConfig | None = None,
        dora_ukeire_weight: float = 1.0,
        dora_tiebreak_weight: float = 1.0,
        dora_neighbor_weight: float = 1.0,
        yakuhai_weight: float = 1.0,
        tanyao_weight: float = 1.0,
        kanchan_outer_wait_weight: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            future_ukeire_weight=future_ukeire_weight,
            future_ukeire_weight_1_shanten=future_ukeire_weight_1_shanten,
            future_ukeire_weight_2_shanten=future_ukeire_weight_2_shanten,
            improvement_weight=improvement_weight,
            shanten_reversal=shanten_reversal,
        )
        weights = (
            dora_ukeire_weight,
            dora_tiebreak_weight,
            dora_neighbor_weight,
            yakuhai_weight,
            tanyao_weight,
            kanchan_outer_wait_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("hand-value heuristic weights must be nonnegative")
        self.dora_ukeire_weight = float(dora_ukeire_weight)
        self.dora_tiebreak_weight = float(dora_tiebreak_weight)
        self.dora_neighbor_weight = float(dora_neighbor_weight)
        self.yakuhai_weight = float(yakuhai_weight)
        self.tanyao_weight = float(tanyao_weight)
        self.kanchan_outer_wait_weight = float(kanchan_outer_wait_weight)
        # Physical-copy choice matters for red fives even when the tile-type
        # decision is unchanged.  Only dora-aware policies keep the red copy.
        self.preserve_red_five = bool(self.dora_ukeire_weight or self.dora_tiebreak_weight)
        if name is not None:
            self.name = name

    def parameters(self) -> dict[str, Any]:
        return {
            **super().parameters(),
            **({"kanchan_outer_wait_weight": self.kanchan_outer_wait_weight}
               if self.kanchan_outer_wait_weight else {}),
            "dora_ukeire_weight": self.dora_ukeire_weight,
            "dora_tiebreak_weight": self.dora_tiebreak_weight,
            "dora_neighbor_weight": self.dora_neighbor_weight,
            "yakuhai_weight": self.yakuhai_weight,
            "tanyao_weight": self.tanyao_weight,
        }

    def finalize_features(self, features: CandidateFeatures, context: DecisionContext) -> CandidateFeatures:
        discard_id: int | None = None
        if features.discard_type >= 0:
            copies = _preferred_discard_copies(
                context.hand,
                features.discard_type,
                preserve_red_five=self.preserve_red_five,
            )
            if not copies:
                raise ValueError("discard feature does not exist in physical hand")
            discard_id = copies[0]

        retained = list(context.hand)
        if discard_id is not None:
            retained.remove(discard_id)
        dora_type = (
            dora_type_from_indicator(context.dora_indicator // 4)
            if 0 <= context.dora_indicator < PHYSICAL_TILE_COUNT
            else None
        )
        neighbors = _dora_neighbor_types(dora_type)
        retained_dora = sum(
            int(tile_id in RED_FIVE_IDS) + int(dora_type is not None and tile_id // 4 == dora_type)
            for tile_id in retained
        )
        retained_neighbors = sum(tile_id // 4 in neighbors for tile_id in retained)
        retained_yakuhai = sum(
            yakuhai_value(tile_id // 4, context.player_wind, context.round_wind)
            for tile_id in retained
        )
        retained_tanyao = sum(_is_tanyao_tile(tile_id // 4) for tile_id in retained)
        dora_bonus = self.dora_ukeire_weight * retained_dora
        tie_score = (
            self.dora_tiebreak_weight * retained_dora
            + self.dora_neighbor_weight * retained_neighbors
            + self.yakuhai_weight * retained_yakuhai
            + self.tanyao_weight * retained_tanyao
        )
        if self.kanchan_outer_wait_weight and features.shanten == 0:
            retained_counts = counts_from_physical(retained)
            wins = structural_ukeire_types(retained_counts)
            if (len(wins) == 1 and wins[0] < 27 and wins[0] % 9 in (1, 7)
                    and "kanchan" in wait_shapes(retained_counts, wins[0])):
                tie_score += self.kanchan_outer_wait_weight
        return replace(
            features,
            base_ukeire_score=features.ukeire_score,
            dora_ukeire_bonus=dora_bonus,
            ukeire_score=features.ukeire_score + dora_bonus,
            retained_dora=retained_dora,
            retained_dora_neighbors=retained_neighbors,
            retained_yakuhai_value=retained_yakuhai,
            retained_tanyao_tiles=retained_tanyao,
            value_tiebreak_score=tie_score,
        )

    def rank(self, features: CandidateFeatures, context: DecisionContext) -> tuple[float, ...]:
        del context
        return (-float(features.shanten), features.ukeire_score, features.value_tiebreak_score)


def make_policy_evaluator(
    policy: str = "pure",
    *,
    future_ukeire_weight: float = 0.5,
    future_ukeire_weight_1_shanten: float | None = None,
    future_ukeire_weight_2_shanten: float | None = None,
    improvement_weight: float = 0.1,
    shanten_reversal: ShantenReversalConfig | None = None,
    dora_ukeire_weight: float | None = None,
    dora_tiebreak_weight: float | None = None,
    dora_neighbor_weight: float | None = None,
    yakuhai_weight: float | None = None,
    tanyao_weight: float | None = None,
    kanchan_outer_wait_weight: float = 0.0,
) -> PureTileEfficiencyEvaluator:
    """Build a named preset while keeping every weight directly tunable.

    ``pure`` preserves the original policy. ``value`` starts every hand-value
    weight at 1.0. ``custom`` starts them at zero, so experiments can enable
    one feature at a time.
    """
    if policy not in ("pure", "value", "custom"):
        raise ValueError("policy must be 'pure', 'value', or 'custom'")
    supplied = (
        dora_ukeire_weight,
        dora_tiebreak_weight,
        dora_neighbor_weight,
        yakuhai_weight,
        tanyao_weight,
    )
    if policy == "pure":
        if kanchan_outer_wait_weight or any(weight not in (None, 0, 0.0) for weight in supplied):
            raise ValueError("hand-value weights require --policy value or --policy custom")
        return PureTileEfficiencyEvaluator(
            future_ukeire_weight=future_ukeire_weight,
            future_ukeire_weight_1_shanten=future_ukeire_weight_1_shanten,
            future_ukeire_weight_2_shanten=future_ukeire_weight_2_shanten,
            improvement_weight=improvement_weight,
            shanten_reversal=shanten_reversal,
        )

    default = 1.0 if policy == "value" else 0.0
    resolved = tuple(default if weight is None else weight for weight in supplied)
    return HandValueHeuristicEvaluator(
        future_ukeire_weight=future_ukeire_weight,
        future_ukeire_weight_1_shanten=future_ukeire_weight_1_shanten,
        future_ukeire_weight_2_shanten=future_ukeire_weight_2_shanten,
        improvement_weight=improvement_weight,
        shanten_reversal=shanten_reversal,
        dora_ukeire_weight=resolved[0],
        dora_tiebreak_weight=resolved[1],
        dora_neighbor_weight=resolved[2],
        yakuhai_weight=resolved[3],
        tanyao_weight=resolved[4],
        kanchan_outer_wait_weight=kanchan_outer_wait_weight,
        name="shanten_then_value_ukeire" if policy == "value" else "custom_hand_value",
    )


@dataclass(frozen=True)
class RankedCandidate:
    features: CandidateFeatures
    rank: tuple[float, ...]


def _visible_ukeire(
    hand_13: tuple[int, ...],
    external_visible: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return live acceptance types/counts for a 13-tile hand.

    ``external_visible`` excludes the concealed hand and includes every known
    discard or indicator.  It may be a list, Counter, or 34-count tuple.
    """
    types: list[int] = []
    remaining: list[int] = []
    for tile_type in structural_ukeire_types(hand_13):
        count = max(0, 4 - hand_13[tile_type] - external_visible[tile_type])
        if count:
            types.append(tile_type)
            remaining.append(count)
    return tuple(types), tuple(remaining)


@dataclass(frozen=True)
class _PostDrawChoice:
    shanten: int
    ukeire_count: int


def _best_post_draw_choice(
    hand_13: tuple[int, ...],
    draw_type: int,
    external_visible: tuple[int, ...],
) -> _PostDrawChoice:
    """Look ahead one draw and one pure-efficiency discard, without recursion."""
    if hand_13[draw_type] + external_visible[draw_type] >= 4:
        raise ValueError(f"cannot draw a fifth visible copy of {tile_name(draw_type)}")
    hand_14 = list(hand_13)
    hand_14[draw_type] += 1
    post_hands: list[tuple[int, int, tuple[int, ...]]] = []
    for discard_type, count in enumerate(hand_14):
        if not count:
            continue
        after = hand_14.copy()
        after[discard_type] -= 1
        after_tuple = tuple(after)
        post_hands.append((_shanten_cached(after_tuple), discard_type, after_tuple))
    best_shanten = min(row[0] for row in post_hands)
    best_ukeire = 0
    for post_shanten, discard_type, after_tuple in post_hands:
        if post_shanten != best_shanten:
            continue
        next_external = list(external_visible)
        next_external[discard_type] += 1
        _, remaining = _visible_ukeire(after_tuple, next_external)
        best_ukeire = max(best_ukeire, sum(remaining))
    return _PostDrawChoice(shanten=best_shanten, ukeire_count=best_ukeire)


def _advanced_ukeire_features(
    hand_13: tuple[int, ...],
    post_shanten: int,
    external_visible: tuple[int, ...],
    ukeire_types: tuple[int, ...],
    ukeire_remaining: tuple[int, ...],
    *,
    include_future: bool = True,
    include_improvements: bool = True,
) -> tuple[tuple[float, ...], float, tuple[int, ...], tuple[int, ...]]:
    """Compute one-step future ukeire and same-shanten improvement copies."""
    if post_shanten not in (1, 2) or not (include_future or include_improvements):
        return (), 0.0, (), ()

    current_ukeire = sum(ukeire_remaining)
    future_by_type: list[float] = []
    if include_future:
        for draw_type in ukeire_types:
            choice = _best_post_draw_choice(hand_13, draw_type, external_visible)
            future_by_type.append(float(choice.ukeire_count))
    future_expected = (
        sum(value * count for value, count in zip(future_by_type, ukeire_remaining, strict=True)) / current_ukeire
        if current_ukeire and include_future
        else 0.0
    )

    if not include_improvements:
        return tuple(future_by_type), future_expected, (), ()
    direct_types = set(ukeire_types)
    improvement_types: list[int] = []
    improvement_remaining: list[int] = []
    for draw_type in range(TILE_TYPE_COUNT):
        remaining = max(0, 4 - hand_13[draw_type] - external_visible[draw_type])
        if not remaining or draw_type in direct_types:
            continue
        choice = _best_post_draw_choice(hand_13, draw_type, external_visible)
        if choice.shanten == post_shanten and choice.ukeire_count > current_ukeire:
            improvement_types.append(draw_type)
            improvement_remaining.append(remaining)
    return tuple(future_by_type), future_expected, tuple(improvement_types), tuple(improvement_remaining)


def rank_discards(
    context: DecisionContext,
    evaluator: DiscardEvaluator,
    *,
    prune_dominated: bool = True,
) -> tuple[RankedCandidate, ...]:
    """Evaluate one action per discard tile type, avoiding duplicate-copy bias."""
    if len(context.hand) != 14:
        raise ValueError("discard ranking requires a 14-tile hand")
    hand_counts = counts_from_physical(context.hand)
    visible_counts = Counter(tile_id // 4 for tile_id in context.visible_tile_ids)
    post_hands: list[tuple[int, tuple[int, ...], int]] = []
    for discard_type in sorted(set(tile_id // 4 for tile_id in context.hand)):
        after = list(hand_counts)
        after[discard_type] -= 1
        after_tuple = tuple(after)
        post_shanten = shanten(after_tuple)
        post_hands.append((discard_type, after_tuple, post_shanten))

    best_shanten = min(row[2] for row in post_hands)
    may_prune = prune_dominated and bool(getattr(evaluator, "shanten_dominates", False))
    include_improvements = bool(getattr(evaluator, "needs_improvement_ukeire", False))
    reversal = getattr(evaluator, "shanten_reversal", ShantenReversalConfig())
    reversal_active = (
        reversal.enabled
        and best_shanten in (0, 1)
        and context.turn <= reversal.max_turn
        and context.shanten_reversals_used < reversal.max_per_hand
    )
    candidates: list[RankedCandidate] = []
    for discard_type, after_tuple, post_shanten in post_hands:
        cheap_reversal = reversal_active and post_shanten == best_shanten + 1
        compute_ukeire = not may_prune or post_shanten == best_shanten or cheap_reversal
        compute_advanced = compute_ukeire and (not may_prune or post_shanten == best_shanten)
        external_visible = tuple(visible_counts[tile_type] + (tile_type == discard_type) for tile_type in range(34))
        if compute_ukeire:
            live_types, remaining = _visible_ukeire(after_tuple, external_visible)
            if compute_advanced:
                future_weight_for_shanten = getattr(evaluator, "future_weight_for_shanten", None)
                include_future = bool(
                    future_weight_for_shanten(post_shanten)
                    if future_weight_for_shanten is not None
                    else getattr(evaluator, "needs_future_ukeire", False)
                )
                future_by_type, future_expected, improvement_types, improvement_remaining = _advanced_ukeire_features(
                    after_tuple,
                    post_shanten,
                    external_visible,
                    live_types,
                    remaining,
                    include_future=include_future,
                    include_improvements=include_improvements,
                )
            else:
                future_by_type = improvement_types = improvement_remaining = ()
                future_expected = 0.0
        else:
            live_types = remaining = future_by_type = improvement_types = improvement_remaining = ()
            future_expected = 0.0
        features = CandidateFeatures(
            discard_type=discard_type,
            shanten=post_shanten,
            # Detailed form ties require the slower Python decomposition.
            # A custom evaluator that needs them can call shanten_breakdown;
            # the default Monte Carlo policy deliberately keeps this hot-path
            # field empty.
            shanten_forms=(),
            ukeire_types=live_types,
            ukeire_remaining=remaining,
            ukeire_count=sum(remaining),
            future_ukeire_by_type=future_by_type,
            future_ukeire_expected=future_expected,
            improvement_types=improvement_types,
            improvement_remaining=improvement_remaining,
            improvement_count=sum(improvement_remaining),
            ukeire_score=0.0,
            ukeire_computed=compute_ukeire,
            strict_best_shanten=best_shanten,
            reversal_eligible=cheap_reversal and bool(sum(remaining)),
        )
        composer = getattr(evaluator, "compose_ukeire_score", None)
        score = float(composer(features)) if composer is not None else float(features.ukeire_count)
        features = replace(features, base_ukeire_score=score, ukeire_score=score)
        finalizer = getattr(evaluator, "finalize_features", None)
        if finalizer is not None:
            features = finalizer(features, context)
        candidates.append(RankedCandidate(features=features, rank=tuple(evaluator.rank(features, context))))

    if not reversal_active:
        return tuple(candidates)

    # Prune using only recovery breadth and already-cheap value features.
    reversal_indices = [
        index for index, candidate in enumerate(candidates)
        if candidate.features.reversal_eligible
    ]
    reversal_indices.sort(
        key=lambda index: (
            candidates[index].features.ukeire_count + candidates[index].features.dora_ukeire_bonus,
            candidates[index].features.value_tiebreak_score,
            -candidates[index].features.discard_type,
        ),
        reverse=True,
    )
    reversal_indices = reversal_indices[: reversal.candidate_limit]
    strict_current_ukeire = max(
        candidate.features.ukeire_count
        for candidate in candidates
        if candidate.features.shanten == best_shanten
    )
    for index in reversal_indices:
        candidate = candidates[index]
        features = candidate.features
        post_hand = next(row[1] for row in post_hands if row[0] == features.discard_type)
        external_visible = tuple(
            visible_counts[tile_type] + (tile_type == features.discard_type)
            for tile_type in range(TILE_TYPE_COUNT)
        )
        future_by_type, future_expected, _, _ = _advanced_ukeire_features(
            post_hand,
            features.shanten,
            external_visible,
            features.ukeire_types,
            features.ukeire_remaining,
            include_future=True,
            include_improvements=False,
        )
        ukeire_gain = future_expected - strict_current_ukeire
        base_score = features.ukeire_count + reversal.recovered_ukeire_weight * ukeire_gain
        features = replace(
            features,
            future_ukeire_by_type=future_by_type,
            future_ukeire_expected=future_expected,
            base_ukeire_score=base_score,
            ukeire_score=base_score,
            reversal_deep_evaluated=True,
            is_shanten_reversal=True,
            reversal_recovery_count=features.ukeire_count,
            reversal_future_ukeire_expected=future_expected,
            reversal_current_ukeire=strict_current_ukeire,
            reversal_ukeire_gain=ukeire_gain,
        )
        if finalizer is not None:
            features = finalizer(features, context)
        features = replace(features, reversal_adjusted_score=features.ukeire_score - reversal.penalty)
        candidates[index] = RankedCandidate(features=features, rank=candidate.rank)

    strict_score = max(
        candidate.features.ukeire_score
        for candidate in candidates
        if candidate.features.shanten == best_shanten
    )
    best_reversal_score = max(
        (
            candidates[index].features.reversal_adjusted_score
            for index in reversal_indices
            if candidates[index].features.reversal_ukeire_gain > 1e-12
        ),
        default=-math.inf,
    )
    reversal_wins = best_reversal_score > strict_score + 1e-12
    reranked: list[RankedCandidate] = []
    for candidate in candidates:
        features = candidate.features
        qualifies_as_reversal = (
            features.reversal_deep_evaluated and features.reversal_ukeire_gain > 1e-12
        )
        if reversal_wins and qualifies_as_reversal:
            rank = (1.0, features.reversal_adjusted_score, features.value_tiebreak_score)
        elif not reversal_wins and features.shanten == best_shanten:
            rank = (1.0, *tuple(evaluator.rank(features, context)))
        else:
            rank = (0.0, *tuple(evaluator.rank(features, context)))
        reranked.append(RankedCandidate(features=features, rank=rank))
    return tuple(reranked)


def choose_discard(
    context: DecisionContext,
    evaluator: DiscardEvaluator,
    rng: random.Random,
) -> tuple[int, RankedCandidate, tuple[RankedCandidate, ...]]:
    candidates = rank_discards(context, evaluator)
    best_rank = max(candidate.rank for candidate in candidates)
    best = [candidate for candidate in candidates if candidate.rank == best_rank]
    chosen = rng.choice(best)
    physical_copies = _preferred_discard_copies(
        context.hand,
        chosen.features.discard_type,
        preserve_red_five=bool(getattr(evaluator, "preserve_red_five", False)),
    )
    return rng.choice(physical_copies), chosen, candidates


@dataclass(frozen=True)
class ScoreContext:
    is_tsumo: bool
    is_riichi: bool
    is_ippatsu: bool
    is_dealer: bool
    is_haitei: bool
    is_tenhou: bool
    is_chiihou: bool
    player_wind: int
    round_wind: int
    dora_indicators: tuple[int, ...]
    ura_dora_indicators: tuple[int, ...]


@dataclass(frozen=True)
class HandScore:
    points: int | None
    han: int | None
    fu: int | None
    yaku: tuple[str, ...]
    cost: Mapping[str, int | str]
    error: str | None = None


class ScoreEvaluator(Protocol):
    name: str

    def score(self, hand: tuple[int, ...], win_tile: int, context: ScoreContext) -> HandScore: ...


class MahjongRepositoryScorer:
    """Adapter for the pinned MIT-licensed ``mahjong`` package."""

    name = "mahjong_repository_2"

    def __init__(self, *, has_aka_dora: bool = True) -> None:
        _require_mahjong()
        from mahjong.hand_calculating.hand_config import OptionalRules

        self._options = OptionalRules(has_aka_dora=has_aka_dora)

    def score(self, hand: tuple[int, ...], win_tile: int, context: ScoreContext) -> HandScore:
        from mahjong.hand_calculating.hand import HandCalculator
        from mahjong.hand_calculating.hand_config import HandConfig

        config = HandConfig(
            is_tsumo=context.is_tsumo,
            is_riichi=context.is_riichi,
            is_ippatsu=context.is_ippatsu,
            is_haitei=context.is_haitei,
            is_tenhou=context.is_tenhou,
            is_chiihou=context.is_chiihou,
            player_wind=context.player_wind,
            round_wind=context.round_wind,
            options=self._options,
        )
        result = HandCalculator.estimate_hand_value(
            hand,
            win_tile,
            dora_indicators=context.dora_indicators,
            ura_dora_indicators=context.ura_dora_indicators if context.is_riichi else None,
            config=config,
        )
        if result.error:
            return HandScore(points=None, han=None, fu=None, yaku=(), cost={}, error=str(result.error))
        cost = dict(result.cost)
        return HandScore(
            points=int(cost["total"]),
            han=int(result.han),
            fu=int(result.fu),
            yaku=tuple(str(yaku) for yaku in result.yaku),
            cost=cost,
        )


@dataclass(frozen=True)
class WaitTile:
    tile_type: int
    name: str
    visible_remaining: int
    live_wall_remaining: int
    visible_remaining_tile_ids: tuple[int, ...]
    live_wall_remaining_tile_ids: tuple[int, ...]
    shapes: tuple[str, ...]


@dataclass(frozen=True)
class TenpaiSnapshot:
    turn: int
    hand: str
    hand_counts: tuple[int, ...]
    hand_physical: tuple[int, ...]
    signature: str
    waits: tuple[WaitTile, ...]
    visible_ukeire: int
    live_wall_ukeire: int
    live_wall_tiles_remaining: int
    furiten: bool
    riichi_declared: bool


@dataclass(frozen=True)
class DecisionRecord:
    turn: int
    drawn_tile: int
    drawn_name: str
    discarded_tile: int
    discarded_name: str
    forced_by_riichi: bool
    post_discard_shanten: int
    post_discard_ukeire: int
    post_discard_future_ukeire: float
    post_discard_improvement_ukeire: int
    post_discard_ukeire_score: float
    evaluator_rank: tuple[float, ...]
    post_discard_base_ukeire_score: float = 0.0
    post_discard_dora_ukeire_bonus: float = 0.0
    retained_dora: int = 0
    retained_dora_neighbors: int = 0
    retained_yakuhai_value: int = 0
    retained_tanyao_tiles: int = 0
    value_tiebreak_score: float = 0.0
    strict_best_shanten: int = 99
    reversal_opportunity: bool = False
    is_shanten_reversal: bool = False
    reversal_recovery_count: int = 0
    reversal_future_ukeire_expected: float = 0.0
    reversal_current_ukeire: int = 0
    reversal_ukeire_gain: float = 0.0
    reversal_adjusted_score: float | None = None


@dataclass(frozen=True)
class ShantenReversalRecord:
    turn: int
    from_shanten: int
    to_shanten: int
    discard_type: int
    recovery_count: int
    recovered_ukeire_expected: float
    current_ukeire: int
    ukeire_gain: float
    strict_score: float
    adjusted_reversal_score: float
    recovered: bool
    recovery_turns: int | None


@dataclass(frozen=True)
class WinRecord:
    turn: int
    tile: int
    tile_name: str
    hand: str
    hand_physical: tuple[int, ...]
    wait_shapes: tuple[str, ...]
    ippatsu: bool
    tsumo_score: HandScore | None
    ron_score: HandScore | None


@dataclass(frozen=True)
class TrialResult:
    engine_version: str
    trial_id: int
    trial_seed: int
    policy: str
    policy_parameters: Mapping[str, Any]
    outcome: str
    initial_hand: str
    initial_hand_physical: tuple[int, ...]
    dora_indicator: int
    dora_indicator_name: str
    ura_dora_indicator: int
    ura_dora_indicator_name: str
    first_tenpai: TenpaiSnapshot | None
    win: WinRecord | None
    decisions: tuple[DecisionRecord, ...]
    shanten_reversals: tuple[ShantenReversalRecord, ...] = ()
    shanten_reversal_opportunities: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationConfig:
    stop_at: str = "win"
    riichi_on_tenpai: bool = True
    opponent_draws_per_turn: int = 3
    player_wind: int = 28  # South in 34 format
    round_wind: int = 27  # East in 34 format
    collect_decisions: bool = False

    def __post_init__(self) -> None:
        if self.stop_at not in ("tenpai", "win"):
            raise ValueError("stop_at must be 'tenpai' or 'win'")
        if not 0 <= self.opponent_draws_per_turn <= 3:
            raise ValueError("opponent_draws_per_turn must be 0..3")
        if self.player_wind not in range(27, 31) or self.round_wind not in range(27, 31):
            raise ValueError("winds must be 34-format East/South/West/North indices 27..30")


def splitmix64(value: int) -> int:
    """Stateless 64-bit mixing used to derive stable per-trial streams."""
    value = (value + 0x9E3779B97F4A7C15) & MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK_64
    return (value ^ (value >> 31)) & MASK_64


def trial_seed(base_seed: int, trial_id: int) -> int:
    if trial_id < 0:
        raise ValueError("trial_id must be nonnegative")
    return splitmix64((base_seed & MASK_64) ^ splitmix64(trial_id))


def _make_tenpai_snapshot(
    hand: Sequence[int],
    river: Sequence[int],
    visible_tile_ids: Sequence[int],
    live_wall: Sequence[int],
    turn: int,
    riichi_declared: bool,
) -> TenpaiSnapshot:
    counts = counts_from_physical(hand)
    if shanten(counts) != 0:
        raise ValueError("tenpai snapshot requires a 0-shanten hand")
    visible_counts = Counter(tile_id // 4 for tile_id in visible_tile_ids)
    live_counts = Counter(tile_id // 4 for tile_id in live_wall)
    known_tile_ids = set(hand) | set(visible_tile_ids)
    waits: list[WaitTile] = []
    for tile_type in structural_ukeire_types(counts):
        shapes = wait_shapes(counts, tile_type)
        if not shapes:
            continue
        visible_remaining = max(0, 4 - counts[tile_type] - visible_counts[tile_type])
        visible_remaining_tile_ids = tuple(
            tile_id for tile_id in PHYSICAL_COPIES_BY_TYPE[tile_type] if tile_id not in known_tile_ids
        )
        live_wall_remaining_tile_ids = tuple(
            tile_id for tile_id in live_wall if tile_id // 4 == tile_type
        )
        if len(visible_remaining_tile_ids) != visible_remaining:
            raise RuntimeError("visible wait copy count disagrees with physical tile identities")
        if len(live_wall_remaining_tile_ids) != live_counts[tile_type]:
            raise RuntimeError("live wait copy count disagrees with physical tile identities")
        waits.append(
            WaitTile(
                tile_type=tile_type,
                name=tile_name(tile_type),
                visible_remaining=visible_remaining,
                live_wall_remaining=live_counts[tile_type],
                visible_remaining_tile_ids=visible_remaining_tile_ids,
                live_wall_remaining_tile_ids=live_wall_remaining_tile_ids,
                shapes=shapes,
            )
        )
    shape_names = sorted({shape for wait in waits for shape in wait.shapes}, key=WAIT_SHAPE_ORDER.__getitem__)
    wait_types = {wait.tile_type for wait in waits}
    furiten = any(tile_id // 4 in wait_types for tile_id in river)
    return TenpaiSnapshot(
        turn=turn,
        hand=format_counts(counts),
        hand_counts=counts,
        hand_physical=tuple(sorted(hand)),
        signature="+".join(shape_names),
        waits=tuple(waits),
        visible_ukeire=sum(wait.visible_remaining for wait in waits),
        live_wall_ukeire=sum(wait.live_wall_remaining for wait in waits),
        live_wall_tiles_remaining=len(live_wall),
        furiten=furiten,
        riichi_declared=riichi_declared,
    )


def _score_win(
    scorer: ScoreEvaluator | None,
    hand: tuple[int, ...],
    win_tile: int,
    *,
    is_tsumo: bool,
    riichi: bool,
    ippatsu: bool,
    is_dealer: bool,
    is_haitei: bool,
    is_tenhou: bool,
    is_chiihou: bool,
    player_wind: int,
    round_wind: int,
    dora_indicator: int,
    ura_dora_indicator: int,
) -> HandScore | None:
    if scorer is None:
        return None
    return scorer.score(
        hand,
        win_tile,
        ScoreContext(
            is_tsumo=is_tsumo,
            is_riichi=riichi,
            is_ippatsu=ippatsu,
            is_dealer=is_dealer,
            is_haitei=is_haitei,
            is_tenhou=is_tenhou,
            is_chiihou=is_chiihou,
            player_wind=player_wind,
            round_wind=round_wind,
            dora_indicators=(dora_indicator,),
            ura_dora_indicators=(ura_dora_indicator,),
        ),
    )


def simulate_trial(
    trial_id: int,
    base_seed: int = DEFAULT_SEED,
    *,
    evaluator: DiscardEvaluator | None = None,
    scorer: ScoreEvaluator | None = None,
    config: SimulationConfig | None = None,
) -> TrialResult:
    """Simulate one closed hand with independent wall and policy RNG streams."""
    evaluator = evaluator or PureTileEfficiencyEvaluator()
    config = config or SimulationConfig()
    parameter_method = getattr(evaluator, "parameters", None)
    policy_parameters = dict(parameter_method()) if parameter_method is not None else {}
    seed = trial_seed(base_seed, trial_id)
    wall_rng = random.Random(splitmix64(seed ^ 0x57414C4C))
    policy_rng = random.Random(splitmix64(seed ^ 0x504F4C494359))

    wall = list(range(PHYSICAL_TILE_COUNT))
    wall_rng.shuffle(wall)
    hand = wall[:INITIAL_HAND_TILES]
    # The concealed opponent hands remain physically allocated even though this
    # single-hand model does not inspect or act on them.
    opponent_hands = wall[INITIAL_HAND_TILES : INITIAL_HAND_TILES + OPPONENT_HAND_TILES]
    dead_start = INITIAL_HAND_TILES + OPPONENT_HAND_TILES
    dead_wall = wall[dead_start : dead_start + DEAD_WALL_TILES]
    live_wall = wall[dead_start + DEAD_WALL_TILES :]
    assert len(set(hand + opponent_hands + dead_wall + live_wall)) == PHYSICAL_TILE_COUNT
    assert len(live_wall) == LIVE_WALL_TILES

    dora_indicator, ura_dora_indicator = dead_wall[:2]
    river: list[int] = []
    decisions: list[DecisionRecord] = []
    initial_hand_physical = tuple(sorted(hand))
    initial_hand = format_counts(counts_from_physical(hand))
    first_tenpai: TenpaiSnapshot | None = None
    riichi_turn: int | None = None
    reversals_used = 0
    reversal_opportunities = 0
    reversal_states: list[dict[str, Any]] = []

    def reversal_records() -> tuple[ShantenReversalRecord, ...]:
        return tuple(ShantenReversalRecord(**state) for state in reversal_states)
    # East, South, West, and North draw at live-wall offsets 0, 1, 2, and 3.
    # The preceding seats' draws are hidden consumption in this single-player
    # layer, just like later opponent draws.
    cursor = config.player_wind - 27

    if shanten(counts_from_physical(hand)) == 0:
        first_tenpai = _make_tenpai_snapshot(
            hand,
            river,
            (dora_indicator,),
            live_wall[cursor:],
            turn=0,
            riichi_declared=False,
        )
        if config.stop_at == "tenpai":
            return TrialResult(
                ENGINE_VERSION,
                trial_id,
                seed,
                evaluator.name,
                policy_parameters,
                "tenpai",
                initial_hand,
                initial_hand_physical,
                dora_indicator,
                physical_tile_name(dora_indicator),
                ura_dora_indicator,
                physical_tile_name(ura_dora_indicator),
                first_tenpai,
                None,
                (),
            )

    turn = 0
    while cursor < len(live_wall):
        turn += 1
        drawn_tile = live_wall[cursor]
        cursor += 1
        hand.append(drawn_tile)
        complete_counts = counts_from_physical(hand)
        if shanten(complete_counts) == -1:
            before_win = list(hand)
            before_win.remove(drawn_tile)
            shapes = wait_shapes(counts_from_physical(before_win), drawn_tile // 4)
            ippatsu = riichi_turn is not None and turn == riichi_turn + 1
            first_draw_yakuman = riichi_turn is None and turn == 1 and first_tenpai is not None
            scored_hand = tuple(sorted(hand))
            is_dealer = config.player_wind == 27
            tsumo_score = _score_win(
                scorer,
                scored_hand,
                drawn_tile,
                is_tsumo=True,
                riichi=riichi_turn is not None,
                ippatsu=ippatsu,
                is_dealer=is_dealer,
                is_haitei=cursor == len(live_wall),
                is_tenhou=first_draw_yakuman and is_dealer,
                is_chiihou=first_draw_yakuman and not is_dealer,
                player_wind=config.player_wind,
                round_wind=config.round_wind,
                dora_indicator=dora_indicator,
                ura_dora_indicator=ura_dora_indicator,
            )
            ron_score = _score_win(
                scorer,
                scored_hand,
                drawn_tile,
                is_tsumo=False,
                riichi=riichi_turn is not None,
                ippatsu=ippatsu,
                is_dealer=is_dealer,
                is_haitei=False,
                is_tenhou=False,
                is_chiihou=False,
                player_wind=config.player_wind,
                round_wind=config.round_wind,
                dora_indicator=dora_indicator,
                ura_dora_indicator=ura_dora_indicator,
            )
            return TrialResult(
                ENGINE_VERSION,
                trial_id,
                seed,
                evaluator.name,
                policy_parameters,
                "win",
                initial_hand,
                initial_hand_physical,
                dora_indicator,
                physical_tile_name(dora_indicator),
                ura_dora_indicator,
                physical_tile_name(ura_dora_indicator),
                first_tenpai,
                WinRecord(
                    turn=turn,
                    tile=drawn_tile,
                    tile_name=physical_tile_name(drawn_tile),
                    hand=format_counts(complete_counts),
                    hand_physical=scored_hand,
                    wait_shapes=shapes,
                    ippatsu=ippatsu,
                    tsumo_score=tsumo_score,
                    ron_score=ron_score,
                ),
                tuple(decisions),
                reversal_records(),
                reversal_opportunities,
            )

        if riichi_turn is not None:
            discarded = drawn_tile
            hand.remove(discarded)
            river.append(discarded)
            if config.collect_decisions:
                post_counts = counts_from_physical(hand)
                visible_counts = Counter(tile_id // 4 for tile_id in (*river, dora_indicator))
                ukeire = sum(
                    max(0, 4 - post_counts[t] - visible_counts[t]) for t in structural_ukeire_types(post_counts)
                )
                decisions.append(
                    DecisionRecord(
                        turn,
                        drawn_tile,
                        physical_tile_name(drawn_tile),
                        discarded,
                        physical_tile_name(discarded),
                        True,
                        shanten(post_counts),
                        ukeire,
                        0.0,
                        0,
                        float(ukeire),
                        (),
                        float(ukeire),
                    )
                )
        else:
            context = DecisionContext(
                turn=turn,
                hand=tuple(hand),
                river=tuple(river),
                visible_tile_ids=tuple((*river, dora_indicator)),
                drawn_tile=drawn_tile,
                dora_indicator=dora_indicator,
                player_wind=config.player_wind,
                round_wind=config.round_wind,
                shanten_reversals_used=reversals_used,
            )
            discarded, chosen, candidates = choose_discard(context, evaluator, policy_rng)
            reversal_opportunity = any(
                candidate.features.reversal_eligible for candidate in candidates
            )
            reversal_opportunities += int(reversal_opportunity)
            if chosen.features.is_shanten_reversal:
                strict_score = max(
                    candidate.features.ukeire_score
                    for candidate in candidates
                    if candidate.features.shanten == chosen.features.strict_best_shanten
                )
                reversals_used += 1
                reversal_states.append({
                    "turn": turn,
                    "from_shanten": chosen.features.strict_best_shanten,
                    "to_shanten": chosen.features.shanten,
                    "discard_type": chosen.features.discard_type,
                    "recovery_count": chosen.features.reversal_recovery_count,
                    "recovered_ukeire_expected": chosen.features.reversal_future_ukeire_expected,
                    "current_ukeire": chosen.features.reversal_current_ukeire,
                    "ukeire_gain": chosen.features.reversal_ukeire_gain,
                    "strict_score": strict_score,
                    "adjusted_reversal_score": chosen.features.reversal_adjusted_score,
                    "recovered": False,
                    "recovery_turns": None,
                })
            hand.remove(discarded)
            river.append(discarded)
            if config.collect_decisions:
                decisions.append(
                    DecisionRecord(
                        turn,
                        drawn_tile,
                        physical_tile_name(drawn_tile),
                        discarded,
                        physical_tile_name(discarded),
                        False,
                        chosen.features.shanten,
                        chosen.features.ukeire_count,
                        chosen.features.future_ukeire_expected,
                        chosen.features.improvement_count,
                        chosen.features.ukeire_score,
                        chosen.rank,
                        chosen.features.base_ukeire_score,
                        chosen.features.dora_ukeire_bonus,
                        chosen.features.retained_dora,
                        chosen.features.retained_dora_neighbors,
                        chosen.features.retained_yakuhai_value,
                        chosen.features.retained_tanyao_tiles,
                        chosen.features.value_tiebreak_score,
                        chosen.features.strict_best_shanten,
                        reversal_opportunity,
                        chosen.features.is_shanten_reversal,
                        chosen.features.reversal_recovery_count,
                        chosen.features.reversal_future_ukeire_expected,
                        chosen.features.reversal_current_ukeire,
                        chosen.features.reversal_ukeire_gain,
                        chosen.features.reversal_adjusted_score
                        if chosen.features.reversal_deep_evaluated
                        else None,
                    )
                )

        post_discard_shanten = shanten(counts_from_physical(hand))
        for reversal_state in reversal_states:
            if (
                not reversal_state["recovered"]
                and turn > reversal_state["turn"]
                and post_discard_shanten <= reversal_state["from_shanten"]
            ):
                reversal_state["recovered"] = True
                reversal_state["recovery_turns"] = turn - reversal_state["turn"]
        post_discard_tenpai = post_discard_shanten == 0
        if post_discard_tenpai and riichi_turn is None:
            declare = config.riichi_on_tenpai
            # An initially tenpai 13-tile hand cannot legally declare riichi
            # before its first draw/discard.  In a full-to-win run, replace the
            # turn-zero structural snapshot with the first legal declaration
            # snapshot.  A stop-at-tenpai run has already returned it.
            if first_tenpai is None or first_tenpai.turn == 0:
                first_tenpai = _make_tenpai_snapshot(
                    hand,
                    river,
                    (*river, dora_indicator),
                    live_wall[cursor:],
                    turn=turn,
                    riichi_declared=declare,
                )
            if declare:
                riichi_turn = turn
            if config.stop_at == "tenpai":
                return TrialResult(
                    ENGINE_VERSION,
                    trial_id,
                    seed,
                    evaluator.name,
                    policy_parameters,
                    "tenpai",
                    initial_hand,
                    initial_hand_physical,
                    dora_indicator,
                    physical_tile_name(dora_indicator),
                    ura_dora_indicator,
                    physical_tile_name(ura_dora_indicator),
                    first_tenpai,
                    None,
                    tuple(decisions),
                    reversal_records(),
                    reversal_opportunities,
                )

        cursor = min(len(live_wall), cursor + config.opponent_draws_per_turn)

    return TrialResult(
        ENGINE_VERSION,
        trial_id,
        seed,
        evaluator.name,
        policy_parameters,
        "exhausted",
        initial_hand,
        initial_hand_physical,
        dora_indicator,
        physical_tile_name(dora_indicator),
        ura_dora_indicator,
        physical_tile_name(ura_dora_indicator),
        first_tenpai,
        None,
        tuple(decisions),
        reversal_records(),
        reversal_opportunities,
    )


@dataclass
class SimulationAggregate:
    trials: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    tenpai_hands: int = 0
    ron_eligible_tenpai_hands: int = 0
    tenpai_turn_sum: int = 0
    wins: int = 0
    win_turn_sum: int = 0
    wait_signatures: Counter[str] = field(default_factory=Counter)
    tile_wait_hands: Counter[int] = field(default_factory=Counter)
    tile_shape_wait_hands: Counter[tuple[int, str]] = field(default_factory=Counter)
    visible_wait_copies_by_shape: Counter[str] = field(default_factory=Counter)
    visible_wait_copies: float = 0.0
    live_wait_copies_by_shape: Counter[str] = field(default_factory=Counter)
    live_wait_copies: float = 0.0
    score_counts: Counter[str] = field(default_factory=Counter)
    score_point_sums: Counter[str] = field(default_factory=Counter)
    score_point_square_sums: Counter[str] = field(default_factory=Counter)
    score_han_sums: Counter[str] = field(default_factory=Counter)
    score_fu_sums: Counter[str] = field(default_factory=Counter)
    yaku_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    score_errors: Counter[tuple[str, str]] = field(default_factory=Counter)
    reversal_opportunities: int = 0
    reversals_selected: int = 0
    reversals_recovered: int = 0
    reversal_turn_sum: int = 0
    reversal_recovery_turn_sum: int = 0
    reversal_transitions: Counter[str] = field(default_factory=Counter)

    def add(self, result: TrialResult) -> None:
        self.trials += 1
        self.outcomes[result.outcome] += 1
        self.reversal_opportunities += result.shanten_reversal_opportunities
        for reversal in result.shanten_reversals:
            self.reversals_selected += 1
            self.reversal_turn_sum += reversal.turn
            self.reversal_transitions[f"{reversal.from_shanten}->{reversal.to_shanten}"] += 1
            if reversal.recovered:
                self.reversals_recovered += 1
                self.reversal_recovery_turn_sum += reversal.recovery_turns or 0
        snapshot = result.first_tenpai
        if snapshot is not None:
            self.tenpai_hands += 1
            self.tenpai_turn_sum += snapshot.turn
            self.wait_signatures[snapshot.signature] += 1
            for wait in snapshot.waits:
                split = 1.0 / len(wait.shapes)
                for shape in wait.shapes:
                    self.visible_wait_copies_by_shape[shape] += wait.visible_remaining * split
                    self.live_wait_copies_by_shape[shape] += wait.live_wall_remaining * split
                self.visible_wait_copies += wait.visible_remaining
                self.live_wait_copies += wait.live_wall_remaining
            if not snapshot.furiten:
                self.ron_eligible_tenpai_hands += 1
                for wait in snapshot.waits:
                    if wait.visible_remaining == 0:
                        continue
                    self.tile_wait_hands[wait.tile_type] += 1
                    split = 1.0 / len(wait.shapes)
                    for shape in wait.shapes:
                        self.tile_shape_wait_hands[(wait.tile_type, shape)] += split

        if result.win is not None:
            self.wins += 1
            self.win_turn_sum += result.win.turn
            self._add_score("tsumo", result.win.tsumo_score)
            self._add_score("ron", result.win.ron_score)

    def _add_score(self, kind: str, score: HandScore | None) -> None:
        if score is None:
            return
        if score.error is not None or score.points is None:
            self.score_errors[(kind, score.error or "unknown")] += 1
            return
        self.score_counts[kind] += 1
        self.score_point_sums[kind] += score.points
        self.score_point_square_sums[kind] += score.points * score.points
        self.score_han_sums[kind] += score.han or 0
        self.score_fu_sums[kind] += score.fu or 0
        for yaku in score.yaku:
            self.yaku_counts[(kind, yaku)] += 1

    def merge(self, other: "SimulationAggregate") -> None:
        """Merge a separately generated shard using its raw numerators."""
        for field_name in (
            "trials",
            "tenpai_hands",
            "ron_eligible_tenpai_hands",
            "tenpai_turn_sum",
            "wins",
            "win_turn_sum",
            "reversal_opportunities",
            "reversals_selected",
            "reversals_recovered",
            "reversal_turn_sum",
            "reversal_recovery_turn_sum",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        for field_name in (
            "outcomes",
            "wait_signatures",
            "tile_wait_hands",
            "tile_shape_wait_hands",
            "visible_wait_copies_by_shape",
            "live_wait_copies_by_shape",
            "score_counts",
            "score_point_sums",
            "score_point_square_sums",
            "score_han_sums",
            "score_fu_sums",
            "yaku_counts",
            "score_errors",
            "reversal_transitions",
        ):
            getattr(self, field_name).update(getattr(other, field_name))
        self.visible_wait_copies += other.visible_wait_copies
        self.live_wait_copies += other.live_wait_copies

    def to_dict(
        self,
        *,
        seed: int,
        start_trial: int,
        policy: str,
        scorer: str | None,
        policy_parameters: Mapping[str, Any] | None = None,
        config: SimulationConfig | None = None,
    ) -> dict[str, Any]:
        def ratio(numerator: float, denominator: float) -> float | None:
            return numerator / denominator if denominator else None

        def binomial_se(successes: float, observations: int) -> float | None:
            if not observations:
                return None
            probability = successes / observations
            return math.sqrt(probability * (1.0 - probability) / observations)

        def mean_se(total: float, square_total: float, observations: int) -> float | None:
            if observations < 2:
                return None
            variance = max(0.0, (square_total - total * total / observations) / (observations - 1))
            return math.sqrt(variance / observations)

        tile_danger: dict[str, Any] = {}
        for tile_type in range(TILE_TYPE_COUNT):
            wait_hands = self.tile_wait_hands[tile_type]
            by_shape = {
                shape: self.tile_shape_wait_hands[(tile_type, shape)] / self.ron_eligible_tenpai_hands
                for shape in WAIT_SHAPE_ORDER
                if self.tile_shape_wait_hands[(tile_type, shape)] and self.ron_eligible_tenpai_hands
            }
            tile_danger[tile_name(tile_type)] = {
                "wait_hands": wait_hands,
                "ron_eligible_tenpai_hands": self.ron_eligible_tenpai_hands,
                "probability": ratio(wait_hands, self.ron_eligible_tenpai_hands),
                "standard_error": binomial_se(wait_hands, self.ron_eligible_tenpai_hands),
                "probability_by_shape_fractional": by_shape,
            }

        scores: dict[str, Any] = {}
        for kind in ("tsumo", "ron"):
            count = self.score_counts[kind]
            scores[kind] = {
                "count": count,
                "mean_points": ratio(self.score_point_sums[kind], count),
                "mean_points_standard_error": mean_se(
                    self.score_point_sums[kind], self.score_point_square_sums[kind], count
                ),
                "mean_han": ratio(self.score_han_sums[kind], count),
                "mean_fu": ratio(self.score_fu_sums[kind], count),
                "yaku_counts": {
                    yaku: value for (score_kind, yaku), value in sorted(self.yaku_counts.items()) if score_kind == kind
                },
                "errors": {
                    error: value
                    for (score_kind, error), value in sorted(self.score_errors.items())
                    if score_kind == kind
                },
            }

        return {
            "engine_version": ENGINE_VERSION,
            "seed": seed,
            "start_trial": start_trial,
            "trials": self.trials,
            "policy": policy,
            "policy_parameters": dict(policy_parameters) if policy_parameters is not None else None,
            "scorer": scorer,
            "config": asdict(config) if config is not None else None,
            "outcomes": dict(sorted(self.outcomes.items())),
            "tenpai": {
                "hands": self.tenpai_hands,
                "rate": ratio(self.tenpai_hands, self.trials),
                "rate_standard_error": binomial_se(self.tenpai_hands, self.trials),
                "mean_first_turn": ratio(self.tenpai_turn_sum, self.tenpai_hands),
                "ron_eligible_hands": self.ron_eligible_tenpai_hands,
                "furiten_hands": self.tenpai_hands - self.ron_eligible_tenpai_hands,
            },
            "wins": {
                "hands": self.wins,
                "rate": ratio(self.wins, self.trials),
                "rate_standard_error": binomial_se(self.wins, self.trials),
                "mean_turn": ratio(self.win_turn_sum, self.wins),
            },
            "shanten_reversal": {
                "opportunities": self.reversal_opportunities,
                "selected": self.reversals_selected,
                "selection_rate_per_opportunity": ratio(
                    self.reversals_selected, self.reversal_opportunities
                ),
                "mean_selection_turn": ratio(self.reversal_turn_sum, self.reversals_selected),
                "recovered": self.reversals_recovered,
                "recovery_rate": ratio(self.reversals_recovered, self.reversals_selected),
                "mean_recovery_turns": ratio(
                    self.reversal_recovery_turn_sum, self.reversals_recovered
                ),
                "transitions": dict(sorted(self.reversal_transitions.items())),
            },
            "wait_signatures": {
                signature: {
                    "hands": count,
                    "share": ratio(count, self.tenpai_hands),
                    "share_standard_error": binomial_se(count, self.tenpai_hands),
                }
                for signature, count in sorted(self.wait_signatures.items())
            },
            "acceptance_copy_share_by_shape": {
                shape: {
                    "visible_copies": self.visible_wait_copies_by_shape[shape],
                    "visible_share": ratio(self.visible_wait_copies_by_shape[shape], self.visible_wait_copies),
                    "live_wall_copies": self.live_wait_copies_by_shape[shape],
                    "live_wall_share": ratio(self.live_wait_copies_by_shape[shape], self.live_wait_copies),
                }
                for shape in WAIT_SHAPE_ORDER
                if self.visible_wait_copies_by_shape[shape] or self.live_wait_copies_by_shape[shape]
            },
            "tile_danger_at_first_tenpai": tile_danger,
            "score_values_on_realized_winning_hands": scores,
            "raw": {
                "tenpai_turn_sum": self.tenpai_turn_sum,
                "win_turn_sum": self.win_turn_sum,
                "visible_wait_copies": self.visible_wait_copies,
                "live_wait_copies": self.live_wait_copies,
                "reversal_opportunities": self.reversal_opportunities,
                "reversals_selected": self.reversals_selected,
                "reversals_recovered": self.reversals_recovered,
                "reversal_turn_sum": self.reversal_turn_sum,
                "reversal_recovery_turn_sum": self.reversal_recovery_turn_sum,
                "reversal_transitions": dict(self.reversal_transitions),
                "score_point_sums": dict(self.score_point_sums),
                "score_point_square_sums": dict(self.score_point_square_sums),
                "score_han_sums": dict(self.score_han_sums),
                "score_fu_sums": dict(self.score_fu_sums),
            },
        }


def run_batch(
    trials: int,
    base_seed: int = DEFAULT_SEED,
    *,
    start_trial: int = 0,
    evaluator: DiscardEvaluator | None = None,
    scorer: ScoreEvaluator | None = None,
    config: SimulationConfig | None = None,
    observation_stream: TextIO | None = None,
) -> SimulationAggregate:
    """Run a deterministic shard and optionally stream one JSON object per trial."""
    if trials < 0 or start_trial < 0:
        raise ValueError("trials and start_trial must be nonnegative")
    evaluator = evaluator or PureTileEfficiencyEvaluator()
    config = config or SimulationConfig()
    aggregate = SimulationAggregate()
    for trial_id in range(start_trial, start_trial + trials):
        result = simulate_trial(trial_id, base_seed, evaluator=evaluator, scorer=scorer, config=config)
        aggregate.add(result)
        if observation_stream is not None:
            observation_stream.write(json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    return aggregate


def compare_policy_batches(
    trials: int,
    base_seed: int = DEFAULT_SEED,
    *,
    start_trial: int = 0,
    candidate: PureTileEfficiencyEvaluator,
    scorer: ScoreEvaluator | None = None,
    config: SimulationConfig | None = None,
) -> dict[str, Any]:
    """Compare pure efficiency with another policy on identical trial seeds."""
    config = config or SimulationConfig()
    baseline = PureTileEfficiencyEvaluator(
        future_ukeire_weight=candidate.future_ukeire_weight,
        future_ukeire_weight_1_shanten=candidate.future_ukeire_weights_by_shanten["1"],
        future_ukeire_weight_2_shanten=candidate.future_ukeire_weights_by_shanten["2"],
        improvement_weight=candidate.improvement_weight,
    )
    baseline_aggregate = run_batch(
        trials,
        base_seed,
        start_trial=start_trial,
        evaluator=baseline,
        scorer=scorer,
        config=config,
    )
    candidate_aggregate = run_batch(
        trials,
        base_seed,
        start_trial=start_trial,
        evaluator=candidate,
        scorer=scorer,
        config=config,
    )

    def summary(evaluator: PureTileEfficiencyEvaluator, aggregate: SimulationAggregate) -> dict[str, Any]:
        return aggregate.to_dict(
            seed=base_seed,
            start_trial=start_trial,
            policy=evaluator.name,
            policy_parameters=evaluator.parameters(),
            scorer=scorer.name if scorer else None,
            config=config,
        )

    baseline_summary = summary(baseline, baseline_aggregate)
    candidate_summary = summary(candidate, candidate_aggregate)

    def difference(candidate_value: float | None, baseline_value: float | None) -> float | None:
        if candidate_value is None or baseline_value is None:
            return None
        return candidate_value - baseline_value

    delta = {
        "tenpai_rate": difference(candidate_summary["tenpai"]["rate"], baseline_summary["tenpai"]["rate"]),
        "mean_first_tenpai_turn": difference(
            candidate_summary["tenpai"]["mean_first_turn"], baseline_summary["tenpai"]["mean_first_turn"]
        ),
        "win_rate": difference(candidate_summary["wins"]["rate"], baseline_summary["wins"]["rate"]),
        "mean_win_turn": difference(
            candidate_summary["wins"]["mean_turn"], baseline_summary["wins"]["mean_turn"]
        ),
        "reversals_selected": candidate_summary["shanten_reversal"]["selected"],
    }
    for kind in ("tsumo", "ron"):
        delta[f"mean_{kind}_points"] = difference(
            candidate_summary["score_values_on_realized_winning_hands"][kind]["mean_points"],
            baseline_summary["score_values_on_realized_winning_hands"][kind]["mean_points"],
        )
    return {
        "paired_by_trial_seed": True,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta_candidate_minus_baseline": delta,
    }


def score_observation_file(path: Path, scorer: ScoreEvaluator) -> dict[str, Any]:
    """Score winning rows emitted by the native JSONL engine.

    Tsumo is the realized result. Ron is a counterfactual value for the same
    completed hand, which keeps riichi/ippatsu but excludes tsumo-only flags.
    """
    aggregate = SimulationAggregate()
    rows = 0
    wins = 0
    engines: Counter[str] = Counter()
    evaluator_backends: Counter[str] = Counter()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {error.msg}") from error
            rows += 1
            engines[str(row.get("engine_version", "unknown"))] += 1
            evaluator_backends[str(row.get("evaluator_backend", "unknown"))] += 1
            win = row.get("win")
            if win is None:
                continue
            wins += 1
            try:
                hand = tuple(int(tile) for tile in win["hand_physical"])
                win_tile = int(win["tile"])
                player_wind = int(row["player_wind"])
                round_wind = int(row["round_wind"])
                dora_indicator = int(row["dora_indicator"])
                ura_dora_indicator = int(row["ura_dora_indicator"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid win observation on {path}:{line_number}") from error
            snapshot = row.get("first_tenpai") or {}
            riichi = bool(win.get("riichi", snapshot.get("riichi_declared", False)))
            ippatsu = bool(win.get("ippatsu", False))
            common = {
                "is_riichi": riichi,
                "is_ippatsu": ippatsu,
                "is_dealer": player_wind == 27,
                "player_wind": player_wind,
                "round_wind": round_wind,
                "dora_indicators": (dora_indicator,),
                "ura_dora_indicators": (ura_dora_indicator,),
            }
            tsumo = scorer.score(
                hand,
                win_tile,
                ScoreContext(
                    is_tsumo=True,
                    is_haitei=bool(win.get("haitei", False)),
                    is_tenhou=bool(win.get("tenhou", False)),
                    is_chiihou=bool(win.get("chiihou", False)),
                    **common,
                ),
            )
            ron = scorer.score(
                hand,
                win_tile,
                ScoreContext(
                    is_tsumo=False,
                    is_haitei=False,
                    is_tenhou=False,
                    is_chiihou=False,
                    **common,
                ),
            )
            aggregate._add_score("tsumo", tsumo)
            aggregate._add_score("ron", ron)

    scored = aggregate.to_dict(seed=0, start_trial=0, policy="native_observations", scorer=scorer.name)
    raw = scored["raw"]
    return {
        "source": str(path),
        "rows": rows,
        "wins": wins,
        "engines": dict(sorted(engines.items())),
        "evaluator_backends": dict(sorted(evaluator_backends.items())),
        "scorer": scorer.name,
        "score_values_on_realized_winning_hands": scored["score_values_on_realized_winning_hands"],
        "raw": {
            "score_point_sums": raw["score_point_sums"],
            "score_point_square_sums": raw["score_point_square_sums"],
            "score_han_sums": raw["score_han_sums"],
            "score_fu_sums": raw["score_fu_sums"],
        },
    }


def analyze_hand(
    hand_text: str,
    visible_text: str = "",
    *,
    policy: str = "pure",
    dora_indicator_text: str | None = None,
    player_wind: int = 28,
    round_wind: int = 27,
    future_ukeire_weight: float = 0.5,
    future_ukeire_weight_1_shanten: float | None = None,
    future_ukeire_weight_2_shanten: float | None = None,
    improvement_weight: float = 0.1,
    shanten_reversal: ShantenReversalConfig | None = None,
    turn: int = 0,
    shanten_reversals_used: int = 0,
    dora_ukeire_weight: float | None = None,
    dora_tiebreak_weight: float | None = None,
    dora_neighbor_weight: float | None = None,
    yakuhai_weight: float | None = None,
    tanyao_weight: float | None = None,
    kanchan_outer_wait_weight: float = 0.0,
) -> dict[str, Any]:
    counts = parse_hand(hand_text)
    visible = parse_hand(visible_text) if visible_text else (0,) * TILE_TYPE_COUNT
    evaluator = make_policy_evaluator(
        policy,
        future_ukeire_weight=future_ukeire_weight,
        future_ukeire_weight_1_shanten=future_ukeire_weight_1_shanten,
        future_ukeire_weight_2_shanten=future_ukeire_weight_2_shanten,
        improvement_weight=improvement_weight,
        shanten_reversal=shanten_reversal,
        dora_ukeire_weight=dora_ukeire_weight,
        dora_tiebreak_weight=dora_tiebreak_weight,
        dora_neighbor_weight=dora_neighbor_weight,
        yakuhai_weight=yakuhai_weight,
        tanyao_weight=tanyao_weight,
        kanchan_outer_wait_weight=kanchan_outer_wait_weight,
    )
    indicator_type = parse_single_tile_type(dora_indicator_text) if dora_indicator_text else None
    indicator_id = indicator_type * 4 if indicator_type is not None else -1
    external_visible = list(visible)
    if indicator_type is not None:
        external_visible[indicator_type] += 1
    total = sum(counts)
    if total not in (13, 14):
        raise ValueError("analyze expects 13 or 14 concealed tiles")
    breakdown = shanten_breakdown(counts)
    payload: dict[str, Any] = {
        "hand": format_counts(counts),
        "shanten": asdict(breakdown),
        "policy": {"name": evaluator.name, **evaluator.parameters()},
        "dora_indicator": tile_name(indicator_type) if indicator_type is not None else None,
        "dora": tile_name(dora_type_from_indicator(indicator_type)) if indicator_type is not None else None,
        "player_wind": tile_name(player_wind),
        "round_wind": tile_name(round_wind),
    }
    if total == 13:
        ukeire_types, ukeire_remaining = _visible_ukeire(counts, external_visible)
        future_by_type, future_expected, improvement_types, improvement_remaining = _advanced_ukeire_features(
            counts,
            breakdown.minimum,
            tuple(external_visible),
            ukeire_types,
            ukeire_remaining,
            include_future=bool(evaluator.future_weight_for_shanten(breakdown.minimum)),
            include_improvements=evaluator.needs_improvement_ukeire,
        )
        features = CandidateFeatures(
            discard_type=-1,
            shanten=breakdown.minimum,
            shanten_forms=breakdown.best_forms,
            ukeire_types=ukeire_types,
            ukeire_remaining=ukeire_remaining,
            ukeire_count=sum(ukeire_remaining),
            future_ukeire_by_type=future_by_type,
            future_ukeire_expected=future_expected,
            improvement_types=improvement_types,
            improvement_remaining=improvement_remaining,
            improvement_count=sum(improvement_remaining),
            ukeire_score=0.0,
            ukeire_computed=True,
        )
        score = evaluator.compose_ukeire_score(features)
        features = replace(features, base_ukeire_score=score, ukeire_score=score)
        finalizer = getattr(evaluator, "finalize_features", None)
        if finalizer is not None:
            features = finalizer(
                features,
                DecisionContext(
                    turn=turn,
                    hand=parse_physical_hand(hand_text),
                    river=(),
                    visible_tile_ids=(),
                    drawn_tile=-1,
                    dora_indicator=indicator_id,
                    player_wind=player_wind,
                    round_wind=round_wind,
                    shanten_reversals_used=shanten_reversals_used,
                ),
            )
        ukeire = [
            {
                "tile": tile_name(tile_type),
                "remaining": remaining,
                "future_ukeire": future,
                "wait_shapes": wait_shapes(counts, tile_type) if breakdown.minimum == 0 else (),
            }
            for tile_type, remaining, future in zip(
                features.ukeire_types,
                features.ukeire_remaining,
                features.future_ukeire_by_type or (0.0,) * len(features.ukeire_types),
                strict=True,
            )
        ]
        payload["ukeire"] = ukeire
        payload["ukeire_count"] = features.ukeire_count
        payload["future_ukeire_expected"] = features.future_ukeire_expected
        payload["improvement_ukeire"] = [
            {"tile": tile_name(tile_type), "remaining": remaining}
            for tile_type, remaining in zip(
                features.improvement_types,
                features.improvement_remaining,
                strict=True,
            )
        ]
        payload["improvement_ukeire_count"] = features.improvement_count
        payload["base_ukeire_score"] = features.base_ukeire_score
        payload["dora_ukeire_bonus"] = features.dora_ukeire_bonus
        payload["ukeire_score"] = features.ukeire_score
        payload["value_features"] = {
            "retained_dora": features.retained_dora,
            "retained_dora_neighbors": features.retained_dora_neighbors,
            "retained_yakuhai_value": features.retained_yakuhai_value,
            "retained_tanyao_tiles": features.retained_tanyao_tiles,
            "tiebreak_score": features.value_tiebreak_score,
        }
        return payload

    # The analyzer has no physical identity information; assign deterministic
    # unique IDs solely to reuse the policy feature pipeline.
    physical = list(parse_physical_hand(hand_text))
    visible_physical: list[int] = []
    for tile_type, count in enumerate(visible):
        visible_physical.extend(tile_type * 4 + copy for copy in range(count))
    context = DecisionContext(
        turn=turn,
        hand=tuple(physical),
        river=(),
        visible_tile_ids=tuple((*visible_physical, *(() if indicator_id < 0 else (indicator_id,)))),
        drawn_tile=physical[-1],
        dora_indicator=indicator_id,
        player_wind=player_wind,
        round_wind=round_wind,
        shanten_reversals_used=shanten_reversals_used,
    )
    ranked = rank_discards(context, evaluator, prune_dominated=False)
    best_rank = max(candidate.rank for candidate in ranked)
    payload["discards"] = [
        {
            "tile": tile_name(candidate.features.discard_type),
            "shanten": candidate.features.shanten,
            "forms": shanten_breakdown(
                tuple(
                    count - (tile_type == candidate.features.discard_type)
                    for tile_type, count in enumerate(counts)
                )
            ).best_forms,
            "ukeire_count": candidate.features.ukeire_count,
            "future_ukeire_expected": candidate.features.future_ukeire_expected,
            "improvement_ukeire_count": candidate.features.improvement_count,
            "base_ukeire_score": candidate.features.base_ukeire_score,
            "dora_ukeire_bonus": candidate.features.dora_ukeire_bonus,
            "ukeire_score": candidate.features.ukeire_score,
            "strict_best_shanten": candidate.features.strict_best_shanten,
            "reversal_eligible": candidate.features.reversal_eligible,
            "reversal_deep_evaluated": candidate.features.reversal_deep_evaluated,
            "is_shanten_reversal": candidate.features.is_shanten_reversal,
            "reversal_recovery_count": candidate.features.reversal_recovery_count,
            "reversal_future_ukeire_expected": candidate.features.reversal_future_ukeire_expected,
            "reversal_current_ukeire": candidate.features.reversal_current_ukeire,
            "reversal_ukeire_gain": candidate.features.reversal_ukeire_gain,
            "reversal_adjusted_score": (
                candidate.features.reversal_adjusted_score
                if candidate.features.reversal_deep_evaluated
                else None
            ),
            "value_features": {
                "retained_dora": candidate.features.retained_dora,
                "retained_dora_neighbors": candidate.features.retained_dora_neighbors,
                "retained_yakuhai_value": candidate.features.retained_yakuhai_value,
                "retained_tanyao_tiles": candidate.features.retained_tanyao_tiles,
                "tiebreak_score": candidate.features.value_tiebreak_score,
            },
            "ukeire_computed": candidate.features.ukeire_computed,
            "ukeire": [
                {"tile": tile_name(tile_type), "remaining": remaining, "future_ukeire": future}
                for tile_type, remaining, future in zip(
                    candidate.features.ukeire_types,
                    candidate.features.ukeire_remaining,
                    candidate.features.future_ukeire_by_type
                    or (0.0,) * len(candidate.features.ukeire_types),
                    strict=True,
                )
            ],
            "improvement_ukeire": [
                {"tile": tile_name(tile_type), "remaining": remaining}
                for tile_type, remaining in zip(
                    candidate.features.improvement_types,
                    candidate.features.improvement_remaining,
                    strict=True,
                )
            ],
            "rank": candidate.rank,
            "best": candidate.rank == best_rank,
        }
        for candidate in ranked
    ]
    return payload


def _add_policy_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_policy: str,
    choices: tuple[str, ...] = ("pure", "value", "custom"),
) -> None:
    parser.add_argument("--policy", choices=choices, default=default_policy)
    parser.add_argument("--future-ukeire-weight", type=float, default=0.5)
    parser.add_argument("--future-ukeire-weight-1-shanten", type=float)
    parser.add_argument("--future-ukeire-weight-2-shanten", type=float)
    parser.add_argument("--improvement-weight", type=float, default=0.1)
    parser.add_argument("--dora-ukeire-weight", type=float, default=None)
    parser.add_argument("--dora-tiebreak-weight", type=float, default=None)
    parser.add_argument("--dora-neighbor-weight", type=float, default=None)
    parser.add_argument("--yakuhai-weight", type=float, default=None)
    parser.add_argument("--tanyao-weight", type=float, default=None)
    parser.add_argument("--kanchan-outer-wait-weight", type=float, default=0.0)
    parser.add_argument("--allow-shanten-reversal", action="store_true")
    parser.add_argument("--shanten-reversal-penalty", type=float, default=25.0)
    parser.add_argument("--shanten-reversal-future-weight", type=float, default=0.5)
    parser.add_argument("--shanten-reversal-max-turn", type=int, default=9)
    parser.add_argument("--shanten-reversal-candidate-limit", type=int, default=2)


def _reversal_from_args(args: argparse.Namespace) -> ShantenReversalConfig:
    return ShantenReversalConfig(
        enabled=args.allow_shanten_reversal,
        penalty=args.shanten_reversal_penalty,
        recovered_ukeire_weight=args.shanten_reversal_future_weight,
        max_turn=args.shanten_reversal_max_turn,
        candidate_limit=args.shanten_reversal_candidate_limit,
    )


def _evaluator_from_args(args: argparse.Namespace) -> PureTileEfficiencyEvaluator:
    return make_policy_evaluator(
        args.policy,
        future_ukeire_weight=args.future_ukeire_weight,
        future_ukeire_weight_1_shanten=args.future_ukeire_weight_1_shanten,
        future_ukeire_weight_2_shanten=args.future_ukeire_weight_2_shanten,
        improvement_weight=args.improvement_weight,
        shanten_reversal=_reversal_from_args(args),
        dora_ukeire_weight=args.dora_ukeire_weight,
        dora_tiebreak_weight=args.dora_tiebreak_weight,
        dora_neighbor_weight=args.dora_neighbor_weight,
        yakuhai_weight=args.yakuhai_weight,
        tanyao_weight=args.tanyao_weight,
        kanchan_outer_wait_weight=args.kanchan_outer_wait_weight,
    )


def _wind_type(name: str) -> int:
    return 27 + ("east", "south", "west", "north").index(name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="inspect one 13- or 14-tile hand")
    analyze.add_argument("hand", help="compact mpsz notation, e.g. 123m456p789s1123z")
    analyze.add_argument("--visible", default="", help="other visible tiles to subtract from ukeire")
    analyze.add_argument("--dora-indicator", help="one indicator tile, e.g. 4m")
    analyze.add_argument("--seat-wind", choices=("east", "south", "west", "north"), default="south")
    analyze.add_argument("--round-wind", choices=("east", "south", "west", "north"), default="east")
    analyze.add_argument("--turn", type=int, default=0)
    analyze.add_argument("--shanten-reversals-used", type=int, default=0)
    _add_policy_arguments(analyze, default_policy="pure")
    analyze.add_argument("--pretty", action="store_true")

    simulate = subparsers.add_parser("simulate", help="run a deterministic Monte Carlo shard")
    simulate.add_argument("--trials", type=int, default=1_000)
    simulate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    simulate.add_argument("--start-trial", type=int, default=0)
    simulate.add_argument("--stop-at", choices=("tenpai", "win"), default="win")
    simulate.add_argument("--scorer", choices=("none", "mahjong"), default="none")
    simulate.add_argument("--seat-wind", choices=("east", "south", "west", "north"), default="south")
    simulate.add_argument("--round-wind", choices=("east", "south", "west", "north"), default="east")
    _add_policy_arguments(simulate, default_policy="pure")
    simulate.add_argument("--no-riichi", action="store_true")
    simulate.add_argument("--observations", type=Path, help="stream trial-level JSONL to this path")
    simulate.add_argument("--include-decisions", action="store_true")
    simulate.add_argument("--pretty", action="store_true")

    compare = subparsers.add_parser("compare", help="compare a candidate policy with strict pure efficiency")
    compare.add_argument("--trials", type=int, default=1_000)
    compare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    compare.add_argument("--start-trial", type=int, default=0)
    compare.add_argument("--stop-at", choices=("tenpai", "win"), default="win")
    compare.add_argument("--scorer", choices=("none", "mahjong"), default="none")
    compare.add_argument("--seat-wind", choices=("east", "south", "west", "north"), default="south")
    compare.add_argument("--round-wind", choices=("east", "south", "west", "north"), default="east")
    _add_policy_arguments(compare, default_policy="value", choices=("pure", "value", "custom"))
    compare.add_argument("--no-riichi", action="store_true")
    compare.add_argument("--pretty", action="store_true")

    score = subparsers.add_parser("score-observations", help="score native winning-hand JSONL")
    score.add_argument("observations", type=Path)
    score.add_argument("--scorer", choices=("mahjong",), default="mahjong")
    score.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "analyze":
        payload = analyze_hand(
            args.hand,
            args.visible,
            policy=args.policy,
            dora_indicator_text=args.dora_indicator,
            player_wind=_wind_type(args.seat_wind),
            round_wind=_wind_type(args.round_wind),
            future_ukeire_weight=args.future_ukeire_weight,
            future_ukeire_weight_1_shanten=args.future_ukeire_weight_1_shanten,
            future_ukeire_weight_2_shanten=args.future_ukeire_weight_2_shanten,
            improvement_weight=args.improvement_weight,
            shanten_reversal=_reversal_from_args(args),
            turn=args.turn,
            shanten_reversals_used=args.shanten_reversals_used,
            dora_ukeire_weight=args.dora_ukeire_weight,
            dora_tiebreak_weight=args.dora_tiebreak_weight,
            dora_neighbor_weight=args.dora_neighbor_weight,
            yakuhai_weight=args.yakuhai_weight,
            tanyao_weight=args.tanyao_weight,
            kanchan_outer_wait_weight=args.kanchan_outer_wait_weight,
        )
    elif args.command == "simulate":
        evaluator = _evaluator_from_args(args)
        scorer = MahjongRepositoryScorer() if args.scorer == "mahjong" else None
        config = SimulationConfig(
            stop_at=args.stop_at,
            riichi_on_tenpai=not args.no_riichi,
            player_wind=_wind_type(args.seat_wind),
            round_wind=_wind_type(args.round_wind),
            collect_decisions=args.include_decisions,
        )
        stream = None
        try:
            if args.observations:
                args.observations.parent.mkdir(parents=True, exist_ok=True)
                stream = args.observations.open("w", encoding="utf-8")
            aggregate = run_batch(
                args.trials,
                args.seed,
                start_trial=args.start_trial,
                evaluator=evaluator,
                scorer=scorer,
                config=config,
                observation_stream=stream,
            )
        finally:
            if stream is not None:
                stream.close()
        payload = aggregate.to_dict(
            seed=args.seed,
            start_trial=args.start_trial,
            policy=evaluator.name,
            policy_parameters=evaluator.parameters(),
            scorer=scorer.name if scorer else None,
            config=config,
        )
    elif args.command == "compare":
        evaluator = _evaluator_from_args(args)
        scorer = MahjongRepositoryScorer() if args.scorer == "mahjong" else None
        config = SimulationConfig(
            stop_at=args.stop_at,
            riichi_on_tenpai=not args.no_riichi,
            player_wind=_wind_type(args.seat_wind),
            round_wind=_wind_type(args.round_wind),
        )
        payload = compare_policy_batches(
            args.trials,
            args.seed,
            start_trial=args.start_trial,
            candidate=evaluator,
            scorer=scorer,
            config=config,
        )
    else:
        scorer = MahjongRepositoryScorer()
        payload = score_observation_file(args.observations, scorer)
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
