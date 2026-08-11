#!/usr/bin/env python3
"""Reference finite-wall simulator for the iishanten examples.

The state is deliberately local: the named blocks are legal, physical mahjong
blocks, while filler tiles are not parsed as a complete hand.  The decision is
South's ninth discard after the dealer's riichi declaration: dealer/South have
9 river tiles, West/North have 8, the live wall has 36, and South has 88 unseen
positions (three 13-tile hands, 13 ura-dora tiles, and the live wall).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping


ENGINE_VERSION = "python-legal-wall-1"
INITIAL_TURN = 9
LIVE_WALL_TILES = 36
UNKNOWN_POSITIONS = 88
DEAD_WALL_TILES = 14

# Illustrative point and behaviour assumptions.  They are model inputs, not
# population estimates.
DEALER_RON_LOSS = 7_500.0
DEALER_TSUMO_PAYMENT = 3_400.0
DEALER_IPPATSU_RON_LOSS = 10_900.0
DEALER_IPPATSU_TSUMO_PAYMENT = 4_600.0
SIDE_RON_LOSS = 4_345.0
SIDE_TSUMO_PAYMENT = 4_767.0 / 4.0
RIICHI_SIDE_RON_LOSS = 5_300.0
RIICHI_SIDE_TSUMO_PAYMENT = 1_800.0
HERO_RIICHI_COST = 1_000.0
EXISTING_RIICHI_STICKS = 1_000.0
FOLD_IMPERFECT_TSUMOGIRI = 0.02
DEFAULT_SIDE_RIICHI_ON_TENPAI = 0.50

SIDE_STATE_PROBABILITIES = (("T", 0.12), ("H", 0.23), ("F", 0.65))
CATEGORY_STRATEGIES = ("suji37", "nosuji2378", "allout")
FIXED_DANGER_STRATEGIES = ("fixed05", "fixed10", "fixed15")
HERO_STRATEGIES = CATEGORY_STRATEGIES + FIXED_DANGER_STRATEGIES

# Per-discard deal-in rates (turns 1--19), retained from the previous reference
# model.  The simulator begins after turn 9, so future choices use 10 onward.
TURN_DANGER = {
    "no_28": (4.7, 5.2, 5.8, 6.2, 6.7, 7.1, 7.5, 8.0, 8.6, 9.2, 9.9, 10.6, 11.4, 12.3, 13.3, 14.3, 15.4, 16.7, 19.1),
    "no_37": (5.8, 6.3, 6.7, 7.1, 7.5, 7.9, 8.4, 8.9, 9.5, 10.1, 10.8, 11.6, 12.4, 13.3, 14.3, 15.4, 16.6, 17.9, 20.4),
    "no_46": (5.7, 6.9, 8.0, 8.9, 9.7, 10.5, 11.3, 12.2, 13.1, 14.1, 15.1, 16.3, 17.5, 18.8, 20.1, 21.7, 23.2, 24.9, 27.8),
    "no_5": (5.7, 6.6, 7.7, 8.5, 9.4, 10.2, 11.0, 11.9, 12.8, 13.8, 14.9, 16.0, 17.2, 18.5, 19.9, 21.3, 22.9, 24.7, 27.5),
    "suji_37": (5.6, 5.3, 5.2, 5.2, 5.3, 5.2, 5.3, 5.3, 5.5, 5.6, 5.7, 6.0, 6.2, 6.6, 7.0, 7.4, 8.0, 8.5, 9.8),
}
TURN_DANGER = {key: tuple(x / 100.0 for x in values) for key, values in TURN_DANGER.items()}

# Koba, Tenhou Phoenix East-South 2019: 171,650 hanchan and 1,300,316
# riichi, after the declaration tile is discarded; furiten and kokushi riichi
# are excluded.  These are *candidate-tile winning probabilities decomposed by
# wait form*, not the proportion of hands with each wait form.
KOBA_SOURCE = {
    "hanchan": 171_650,
    "riichi": 1_300_316,
    "condition": "after riichi declaration discard; excludes furiten and kokushi riichi",
}
KOBA_CONTRIBUTIONS = {
    "honor": {"tanki": 0.0056, "shanpon": 0.0120},
    "one_nine": {"tanki": 0.0047, "shanpon": 0.0101, "ryanmen": 0.0461},
    "two_eight": {"tanki": 0.0048, "shanpon": 0.0089, "kanchan": 0.0118, "ryanmen": 0.0466},
    "three_seven": {"tanki": 0.0038, "shanpon": 0.0066, "kanchan": 0.0118, "penchan": 0.0097, "ryanmen": 0.0474},
    "four_five_six": {"tanki": 0.0044, "shanpon": 0.0067, "kanchan": 0.0083, "ryanmen": 0.0762},
}
# Preserve the totals printed in the source table.  Because each displayed
# form contribution is rounded separately, two totals differ by 0.01 point
# from the sum of those displayed cells.
KOBA_TOTALS = {
    "honor": 0.0176,
    "one_nine": 0.0609,
    "two_eight": 0.0720,
    "three_seven": 0.0794,
    "four_five_six": 0.0956,
}

PLAYER_HERO = "hero"
PLAYER_DEALER = "dealer"
SIDE_PLAYERS = ("side1", "side2")
# Stable engine identifiers: side1 is West and side2 is North.
PLAYER_WEST, PLAYER_NORTH = SIDE_PLAYERS
ALL_PLAYERS = (PLAYER_HERO, PLAYER_DEALER, *SIDE_PLAYERS)
RIVER_LENGTHS = {PLAYER_DEALER: 9, PLAYER_HERO: 8, "side1": 8, "side2": 8}


def tile(suit: int, rank: int) -> int:
    """Return a 0--33 tile type.  Suits 0--2 are m/p/s; suit 3 is honors."""
    if suit == 3:
        if not 1 <= rank <= 7:
            raise ValueError("honor rank must be 1..7")
        return 27 + rank - 1
    if not (0 <= suit < 3 and 1 <= rank <= 9):
        raise ValueError("number tile must have suit 0..2 and rank 1..9")
    return suit * 9 + rank - 1


def suit_of(t: int) -> int | None:
    return t // 9 if 0 <= t < 27 else None


def rank_of(t: int) -> int | None:
    return t % 9 + 1 if 0 <= t < 27 else None


def tile_name(t: int) -> str:
    if t < 27:
        return f"{rank_of(t)}{'mps'[t // 9]}"
    return f"{t - 26}z"


def counter_from(tiles: Iterable[int]) -> Counter[int]:
    return Counter(tiles)


@dataclass(frozen=True)
class WaitTemplate:
    """A legal, local tenpai shape whose reserved tiles are in owner's hand."""

    kind: str  # ryanmen, kanchan, penchan, shanpon, tanki
    winners: frozenset[int]
    reserved_tiles: tuple[int, ...]
    owner: str
    suit: int | None = None
    source: str = ""

    def valid(self) -> bool:
        counts = counter_from(self.reserved_tiles)
        if self.kind == "ryanmen":
            if self.suit is None or len(self.winners) != 2 or len(self.reserved_tiles) != 2:
                return False
            a, b = sorted(self.winners)
            return suit_of(a) == self.suit == suit_of(b) and b - a == 3 and counts == Counter((a + 1, a + 2))
        if self.kind == "kanchan":
            if self.suit is None or len(self.winners) != 1 or len(self.reserved_tiles) != 2:
                return False
            w = next(iter(self.winners))
            return suit_of(w) == self.suit and 2 <= rank_of(w) <= 8 and counts == Counter((w - 1, w + 1))
        if self.kind == "penchan":
            if self.suit is None or len(self.winners) != 1 or len(self.reserved_tiles) != 2:
                return False
            w = next(iter(self.winners))
            return (rank_of(w) == 3 and counts == Counter((w - 2, w - 1))) or (rank_of(w) == 7 and counts == Counter((w + 1, w + 2)))
        if self.kind == "shanpon":
            return len(self.winners) == 2 and len(self.reserved_tiles) == 4 and all(counts[w] == 2 for w in self.winners)
        if self.kind == "tanki":
            return len(self.winners) == 1 and len(self.reserved_tiles) == 1 and counts[next(iter(self.winners))] == 1
        return False


@dataclass(frozen=True)
class TenpaiPlan:
    improvement: int
    wait: WaitTemplate
    entry_discard: int


@dataclass(frozen=True)
class IishantenTemplate:
    """Local blocks plus an exact improvement-to-tenpai mapping.

    ``reserved_tiles`` are tiles that must be in the current concealed hand.
    Filler tiles do not create additional recorded waits.
    """

    key: str
    label: str
    owner: str
    reserved_tiles: tuple[int, ...]
    plans: tuple[TenpaiPlan, ...]
    component_waits: tuple[WaitTemplate, ...] = ()

    @property
    def improvements(self) -> frozenset[int]:
        return frozenset(plan.improvement for plan in self.plans)

    def plan_for(self, drawn: int) -> TenpaiPlan | None:
        for plan in self.plans:
            if plan.improvement == drawn:
                return plan
        return None


@dataclass(frozen=True)
class WeightConfig:
    """Identifiable maximum-entropy weights.

    ``kind_log_weights`` has one total weight per wait form.  Tanki is the
    zero-valued reference level.  ``location_log_weights`` contains only
    mirror-symmetric rank/tile-class effects.  The legacy mappings remain in
    the shared JSON shape but are empty; they are not fitted.
    """

    kind_log_weights: Mapping[str, float]
    position_log_weights: Mapping[str, float]
    regularization: float = 1.0
    source: str = "uniform legal-template prior; not fitted"
    internal_log_weights: Mapping[str, float] = field(default_factory=dict)
    location_log_weights: Mapping[str, float] = field(default_factory=dict)


DEFAULT_WEIGHTS = WeightConfig(
    kind_log_weights={k: 0.0 for k in ("ryanmen", "kanchan", "penchan", "shanpon", "tanki")},
    position_log_weights={"terminal": 0.0, "outer": 0.0, "middle": 0.0, "honor": 0.0},
)

# Three proposal-refinement rounds of 20,000 complete legal states, followed by
# an independent 20,000-state diagnostic sample.  The deliberately compact
# family has five form weights and mirror-symmetric rank groups only.  It does
# not use suit-specific or individual-tile parameters.
FITTED_WEIGHTS = WeightConfig(
    kind_log_weights={
        "ryanmen": 2.103018709966594,
        "kanchan": 0.7931893428052499,
        "penchan": 0.5155417096676611,
        "shanpon": -2.3198482669846596,
        "tanki": 0.0,
    },
    position_log_weights={},
    regularization=0.0005,
    source="three-stage constrained fit: 3x20000 legal states; independent n=20000 validation",
    location_log_weights={
        "ryanmen:2": 0.0015405314187024607,
        "ryanmen:3": 0.0329485265810187,
        "kanchan:3": -0.00028869099824344594,
        "kanchan:4": -0.13696828240052153,
        "kanchan:5": -1.4625905725590236,
        "tanki:one_nine": -0.14524019480503936,
        "tanki:two_eight": -0.13309023220683117,
        "tanki:three_seven": -0.29201299611410114,
        "tanki:four_five_six": -0.21129329045558723,
        "shanpon:one_nine": -0.18418856738816006,
        "shanpon:two_eight": -0.28592479100670826,
        "shanpon:three_seven": -0.5103770898962516,
        "shanpon:four_five_six": -0.5505124453465684,
    },
)

# Audited independent-sample diagnostics for the constants above.  The simple
# family intentionally preserves its residual instead of adding per-tile knobs.
FITTED_CALIBRATION_RESIDUALS = {
    "honor.tanki": (0.0056, 0.005310855123274208, 0.00022059428302150612),
    "honor.shanpon": (0.0120, 0.012265996352330822, 0.0003706026779560319),
    "honor.total": (0.0176, 0.017576851475605032, 0.00043128665943888714),
    "one_nine.tanki": (0.0047, 0.004664423665122782, 0.00022316528105675523),
    "one_nine.shanpon": (0.0101, 0.010410294503189723, 0.00035862901646536294),
    "one_nine.ryanmen": (0.0461, 0.04131346674823036, 0.0006030025269382469),
    "one_nine.total": (0.0609, 0.05638818491654286, 0.0007362265694838548),
    "two_eight.tanki": (0.0048, 0.004485281309876328, 0.00021859319377225752),
    "two_eight.shanpon": (0.0089, 0.009122975091447483, 0.0003286168171230516),
    "two_eight.kanchan": (0.0118, 0.01155068803344365, 0.0003446684120773785),
    "two_eight.ryanmen": (0.0466, 0.04165215119317192, 0.0006040978106467116),
    "two_eight.total": (0.0720, 0.06681109562793938, 0.0007996889870266644),
    "three_seven.tanki": (0.0038, 0.003839979172994316, 0.00020225074453237944),
    "three_seven.shanpon": (0.0066, 0.007061656614743784, 0.0002917011408677534),
    "three_seven.kanchan": (0.0118, 0.010587061222718791, 0.0003303089702252966),
    "three_seven.penchan": (0.0097, 0.008743003427778019, 0.0003015419519323806),
    "three_seven.ryanmen": (0.0474, 0.0431834945979954, 0.0006104817056291894),
    "three_seven.total": (0.0794, 0.07341519503623031, 0.0008358913785545709),
    "four_five_six.tanki": (0.0044, 0.003933695300420697, 0.00016591389033503279),
    "four_five_six.shanpon": (0.0067, 0.00642911078350293, 0.00022737746658039093),
    "four_five_six.kanchan": (0.0083, 0.006824637733271008, 0.00021617505294646534),
    "four_five_six.ryanmen": (0.0762, 0.08334112401567725, 0.0005045557209977696),
    "four_five_six.total": (0.0956, 0.1005285678328719, 0.0006168760494804513),
}


@dataclass(frozen=True)
class Shape:
    key: str
    label: str


def wait_position(template: WaitTemplate) -> str:
    w = min(template.winners)
    if w >= 27:
        return "honor"
    r = rank_of(w)
    if r in (1, 9):
        return "terminal"
    if r in (2, 8):
        return "outer"
    return "middle"


def template_location(template: WaitTemplate) -> str:
    """Primary mirror-symmetric location label, used for debug output."""
    if template.kind == "ryanmen":
        first = int(template.source.split("/")[0])
        return f"ryanmen_{min(first, 7 - first)}"
    if template.kind == "kanchan":
        rank = int(template.source)
        return f"kanchan_{min(rank, 10 - rank)}"
    return template.kind


CALIBRATION_PARAMETER_NAMES = (
    "kind:ryanmen", "kind:kanchan", "kind:penchan", "kind:shanpon",
    "ryanmen:2", "ryanmen:3",
    "kanchan:3", "kanchan:4", "kanchan:5",
    "tanki:one_nine", "tanki:two_eight", "tanki:three_seven", "tanki:four_five_six",
    "shanpon:one_nine", "shanpon:two_eight", "shanpon:three_seven", "shanpon:four_five_six",
)


@lru_cache(maxsize=None)
def calibration_feature_counts(template: WaitTemplate) -> tuple[int, ...]:
    """Return the 17 constrained features; honors/reference locations are zero."""
    values = Counter()
    if template.kind != "tanki":
        values[f"kind:{template.kind}"] = 1
    if template.kind == "ryanmen":
        group = min(int(template.source.split("/")[0]), 7 - int(template.source.split("/")[0]))
        if group != 1:
            values[f"ryanmen:{group}"] = 1
    elif template.kind == "kanchan":
        rank = int(template.source)
        group = min(rank, 10 - rank)
        if group != 2:
            values[f"kanchan:{group}"] = 1
    elif template.kind == "tanki":
        klass = tile_class(next(iter(template.winners)))
        if klass != "honor":
            values[f"tanki:{klass}"] = 1
    elif template.kind == "shanpon":
        for winner in template.winners:
            klass = tile_class(winner)
            if klass != "honor":
                values[f"shanpon:{klass}"] += 1
    return tuple(values[name] for name in CALIBRATION_PARAMETER_NAMES)


def weight_parameters(weights: WeightConfig) -> tuple[float, ...]:
    values = []
    for name in CALIBRATION_PARAMETER_NAMES:
        if name.startswith("kind:"):
            values.append(weights.kind_log_weights.get(name.split(":", 1)[1], 0.0))
        else:
            values.append(weights.location_log_weights.get(name, 0.0))
    return tuple(values)


def weights_from_parameters(parameters: Iterable[float], *, regularization: float, source: str) -> WeightConfig:
    mapped = dict(zip(CALIBRATION_PARAMETER_NAMES, parameters))
    kind_weights = {kind: mapped.get(f"kind:{kind}", 0.0) for kind in DEFAULT_WEIGHTS.kind_log_weights}
    locations = {name: value for name, value in mapped.items() if not name.startswith("kind:")}
    return WeightConfig(kind_weights, {}, regularization, source, {}, locations)


@lru_cache(maxsize=None)
def enumerate_wait_templates(owner: str) -> tuple[WaitTemplate, ...]:
    templates: list[WaitTemplate] = []
    for suit in range(3):
        # 12/45/78 style ryanmen are intentionally excluded: the legal starts
        # are the six standard 14,25,36,47,58,69 winner pairs.
        for first in range(1, 7):
            low, high = tile(suit, first), tile(suit, first + 3)
            templates.append(WaitTemplate("ryanmen", frozenset((low, high)), (low + 1, low + 2), owner, suit, f"{first}/{first + 3}"))
        for r in range(2, 9):
            w = tile(suit, r)
            templates.append(WaitTemplate("kanchan", frozenset((w,)), (w - 1, w + 1), owner, suit, str(r)))
        for w_rank, reserve_ranks in ((3, (1, 2)), (7, (8, 9))):
            w = tile(suit, w_rank)
            templates.append(WaitTemplate("penchan", frozenset((w,)), tuple(tile(suit, r) for r in reserve_ranks), owner, suit, str(w_rank)))
    for first in range(34):
        for second in range(first + 1, 34):
            templates.append(WaitTemplate("shanpon", frozenset((first, second)), (first, first, second, second), owner, None, f"{tile_name(first)}/{tile_name(second)}"))
    for w in range(34):
        templates.append(WaitTemplate("tanki", frozenset((w,)), (w,), owner, suit_of(w), tile_name(w)))
    assert all(template.valid() for template in templates)
    return tuple(templates)


def template_weight(template: WaitTemplate, weights: WeightConfig) -> float:
    parameters = weight_parameters(weights)
    return math.exp(sum(value * count for value, count in zip(parameters, calibration_feature_counts(template))))


_WAIT_SAMPLING_MASSES: dict[tuple[str, tuple[float, ...]], tuple[float, ...]] = {}
_WAIT_SUBSET_MASSES: dict[tuple[tuple[WaitTemplate, ...], tuple[float, ...]], tuple[float, ...]] = {}


def sample_wait_template(rng: random.Random, owner: str, weights: WeightConfig = DEFAULT_WEIGHTS, *, allowed: Iterable[WaitTemplate] | None = None) -> WaitTemplate:
    candidates = tuple(allowed) if allowed is not None else enumerate_wait_templates(owner)
    if not candidates:
        raise ValueError("no legal wait templates available")
    parameters = weight_parameters(weights)
    if not any(parameters):
        return rng.choice(candidates)
    if allowed is None:
        key = (owner, parameters)
        masses = _WAIT_SAMPLING_MASSES.get(key)
        if masses is None:
            masses = tuple(template_weight(t, weights) for t in candidates)
            _WAIT_SAMPLING_MASSES[key] = masses
    else:
        key = (candidates, parameters)
        masses = _WAIT_SUBSET_MASSES.get(key)
        if masses is None:
            masses = tuple(template_weight(t, weights) for t in candidates)
            _WAIT_SUBSET_MASSES[key] = masses
    return rng.choices(candidates, weights=masses, k=1)[0]


def _compatible_template(rng: random.Random, owner: str, forbidden: set[int], weights: WeightConfig) -> WaitTemplate:
    candidates = [t for t in enumerate_wait_templates(owner) if not (t.winners & forbidden)]
    return sample_wait_template(rng, owner, weights, allowed=candidates)


def make_iishanten_template(rng: random.Random, key: str, owner: str, *, forbidden: set[int] | None = None, weights: WeightConfig = DEFAULT_WEIGHTS) -> IishantenTemplate:
    """Create concrete local blocks.  Each completion leaves another legal wait."""
    forbidden = set() if forbidden is None else set(forbidden)
    if key not in {"ryanmen_ryanmen", "ryanmen_ryankan", "ryanmen_kanchan"}:
        raise ValueError(f"unknown shape {key}")
    kinds = {
        "ryanmen_ryanmen": ("ryanmen", "ryanmen"),
        "ryanmen_ryankan": ("ryanmen", "kanchan", "kanchan"),
        "ryanmen_kanchan": ("ryanmen", "kanchan"),
    }[key]
    blocks: list[WaitTemplate] = []
    used = Counter()
    for kind in kinds:
        choices = [t for t in enumerate_wait_templates(owner) if t.kind == kind and not (t.winners & forbidden) and all(used[x] + t.reserved_tiles.count(x) <= 4 for x in t.reserved_tiles)]
        choice = sample_wait_template(rng, owner, weights, allowed=choices)
        blocks.append(choice)
        used.update(choice.reserved_tiles)
    plans: list[TenpaiPlan] = []
    # Completing any block produces a tenpai wait selected from a different
    # currently represented block.  The discarded tile is filler, selected when
    # the hand is materialised, so it cannot accidentally be a winner.
    for index, block in enumerate(blocks):
        target = blocks[(index + 1) % len(blocks)]
        for improvement in block.winners:
            plans.append(TenpaiPlan(improvement, target, -1))
    labels = {"ryanmen_ryanmen": "两面＋两面", "ryanmen_ryankan": "两面＋两嵌", "ryanmen_kanchan": "两面＋嵌张"}
    return IishantenTemplate(key, labels[key], owner, tuple(used.elements()), tuple(plans), tuple(blocks))


def bind_entry_discards(template: IishantenTemplate, hand: Iterable[int], *, unavailable: Iterable[int] = ()) -> IishantenTemplate:
    """Bind each route to a deterministic filler discard present in this hand."""
    available = tuple(sorted(hand))
    excluded = set(unavailable)
    bound: list[TenpaiPlan] = []
    for plan in template.plans:
        choices = [t for t in available if t not in excluded and t not in plan.wait.winners and t not in template.reserved_tiles]
        if not choices:
            # The local structure can occupy most of a synthetic hand.  A tile
            # outside the resulting winners still gives a legal deterministic
            # entry discard and is checked again when the route is taken.
            choices = [t for t in available if t not in excluded and t not in plan.wait.winners]
        if not choices:
            raise ValueError("cannot bind an entry discard without furiten")
        bound.append(replace(plan, entry_discard=choices[0]))
    return replace(template, plans=tuple(bound))


@dataclass
class PlayerState:
    name: str
    hand: list[int]
    river: list[int]
    wait: WaitTemplate | None = None
    iishanten: IishantenTemplate | None = None
    state: str = "F"
    riichi: bool = False
    riichi_paid: bool = False
    genbutsu: set[int] = field(default_factory=set)
    folded: bool = False

    def is_furiten(self) -> bool:
        return self.wait is not None and bool(self.wait.winners & set(self.river))


@dataclass
class WallState:
    players: dict[str, PlayerState]
    dora_indicator: int
    ura_dora: list[int]
    live_wall: list[int]
    current_candidate: int
    current_category: str | None
    hero_template: IishantenTemplate
    initial_side_tenpai: tuple[str, ...]
    riichi_sticks: int = 1

    def zones(self) -> Iterable[Iterable[int]]:
        for player in self.players.values():
            yield player.hand
            yield player.river
        yield (self.dora_indicator,)
        yield self.ura_dora
        yield self.live_wall

    def physical_counts(self) -> Counter[int]:
        total: Counter[int] = Counter()
        for zone in self.zones():
            total.update(zone)
        return total

    def assert_invariants(self) -> None:
        assert len(self.players[PLAYER_HERO].hand) == 14
        assert len(self.players[PLAYER_DEALER].hand) == 13
        assert all(len(self.players[name].hand) == 13 for name in SIDE_PLAYERS)
        assert all(len(self.players[name].river) == RIVER_LENGTHS[name] for name in ALL_PLAYERS)
        assert len(self.ura_dora) == 13 and len(self.live_wall) == LIVE_WALL_TILES
        assert self.physical_counts() == Counter({t: 4 for t in range(34)})
        assert sum(map(len, self.zones())) == 136
        assert all(not p.is_furiten() for p in self.players.values())
        assert self.current_candidate in self.players[PLAYER_HERO].hand

    def assert_after_current_discard(self) -> None:
        assert len(self.players[PLAYER_HERO].hand) == 13
        expected = {PLAYER_DEALER: 9, PLAYER_HERO: 9, "side1": 8, "side2": 8}
        assert all(len(self.players[name].river) == expected[name] for name in ALL_PLAYERS)
        assert len(self.live_wall) == 36
        assert 3 * 13 + 13 + len(self.live_wall) == 88
        assert self.physical_counts() == Counter({t: 4 for t in range(34)})


def _take(counts: Counter[int], t: int) -> int:
    if counts[t] <= 0:
        raise ValueError(f"tile exhausted: {tile_name(t)}")
    counts[t] -= 1
    return t


def _take_random(counts: Counter[int], rng: random.Random, *, forbidden: set[int] | None = None) -> int:
    forbidden = forbidden or set()
    candidates = [t for t in range(34) if counts[t] > 0 and t not in forbidden]
    if not candidates:
        raise ValueError("no tile can be drawn")
    return _take(counts, rng.choice(candidates))


def _take_template(counts: Counter[int], template: WaitTemplate) -> list[int]:
    taken: list[int] = []
    for t in template.reserved_tiles:
        taken.append(_take(counts, t))
    return taken


def categorical_state(rng: random.Random) -> str:
    roll = rng.random()
    total = 0.0
    for state, probability in SIDE_STATE_PROBABILITIES:
        total += probability
        if roll < total:
            return state
    return "F"


def _candidate_for(category: str | None, override: int | None = None) -> int:
    if override is not None:
        return override
    return tile(1, 3) if category != "nosuji456" else tile(1, 4)


def suji_anchors(candidate: int) -> frozenset[int]:
    if candidate >= 27:
        return frozenset()
    s, r = suit_of(candidate), rank_of(candidate)
    return frozenset(tile(s, x) for x in (r - 3, r + 3) if 1 <= x <= 9)


def _river_with_constraint(counts: Counter[int], rng: random.Random, player: PlayerState, *, length: int, required: tuple[int, ...] = (), forbidden: set[int] = frozenset()) -> list[int]:
    river: list[int] = []
    for t in required:
        river.append(_take(counts, t))
    while len(river) < length:
        block = set(forbidden)
        if player.wait is not None:
            block |= set(player.wait.winners)
        river.append(_take_random(counts, rng, forbidden=block))
    return river


def _category_holds(dealer_river: list[int], candidate: int, category: str) -> bool:
    seen = set(dealer_river)
    anchors = suji_anchors(candidate)
    if category == "genbutsu":
        return candidate in seen
    if category == "suji37":
        return candidate not in seen and bool(anchors & seen)
    if category == "nosuji2378":
        return candidate not in seen and not (anchors & seen)
    if category == "nosuji456":
        # This implementation's "无筋 4/5/6" means neither associated suji
        # anchor is visible.  For 4p, both 1p and 7p must be absent.
        return candidate not in seen and not (anchors & seen)
    raise ValueError(f"unknown current category {category}")


def _future_wait_winners(template: IishantenTemplate) -> set[int]:
    return set().union(*(set(plan.wait.winners) for plan in template.plans))


def generate_initial_state(rng: random.Random, shape_key: str, *, current_category: str | None = "nosuji2378", candidate_override: int | None = None, weights: WeightConfig = DEFAULT_WEIGHTS, dealer_weights: WeightConfig | None = None, max_attempts: int = 10_000) -> WallState:
    """Draw a whole state without replacement.  A failed state is discarded."""
    candidate = _candidate_for(current_category, candidate_override)
    for _ in range(max_attempts):
        try:
            counts: Counter[int] = Counter({t: 4 for t in range(34)})
            hero_template = make_iishanten_template(rng, shape_key, PLAYER_HERO, forbidden={candidate}, weights=weights)
            hero_hand = list(hero_template.reserved_tiles)
            for t in hero_hand:
                _take(counts, t)
            # South holds the physical fourteenth candidate tile before its
            # ninth discard.  It is not a virtual card or a river alias.
            hero_hand.append(_take(counts, candidate))
            while len(hero_hand) < 14:
                hero_hand.append(_take_random(counts, rng))

            dealer_wait = _compatible_template(rng, PLAYER_DEALER, set(), dealer_weights or weights)
            dealer_hand = _take_template(counts, dealer_wait)
            dealer = PlayerState(PLAYER_DEALER, dealer_hand, [], dealer_wait, state="T", riichi=True, riichi_paid=True)

            sides: list[PlayerState] = []
            for name in SIDE_PLAYERS:
                state = categorical_state(rng)
                if state == "T":
                    wait = _compatible_template(rng, name, set(), weights)
                    hand = _take_template(counts, wait)
                    sides.append(PlayerState(name, hand, [], wait, state="T"))
                elif state == "H":
                    template = make_iishanten_template(rng, "ryanmen_kanchan", name, weights=weights)
                    hand = []
                    for t in template.reserved_tiles:
                        hand.append(_take(counts, t))
                    sides.append(PlayerState(name, hand, [], None, template, state="H"))
                else:
                    sides.append(PlayerState(name, [], [], None, None, state="F"))

            hero = PlayerState(PLAYER_HERO, hero_hand, [], None, hero_template, state="H")
            players = {PLAYER_HERO: hero, PLAYER_DEALER: dealer, **{p.name: p for p in sides}}

            # Initial state: dealer's riichi discard is in a nine-tile river;
            # South is holding its current fourteenth tile; West/North each
            # have eight river tiles and act before the dealer's next draw.
            dealer_required: tuple[int, ...]
            if current_category == "genbutsu":
                dealer_required = (candidate,)
            elif current_category == "suji37":
                anchors = tuple(suji_anchors(candidate))
                dealer_required = (anchors[0],)
            else:
                dealer_required = ()
            dealer.river = _river_with_constraint(counts, rng, dealer, length=RIVER_LENGTHS[PLAYER_DEALER], required=dealer_required)
            hero.river = _river_with_constraint(counts, rng, hero, length=RIVER_LENGTHS[PLAYER_HERO], forbidden=set(hero_template.improvements) | _future_wait_winners(hero_template))
            for side in sides:
                side.river = _river_with_constraint(counts, rng, side, length=RIVER_LENGTHS[side.name], forbidden=_future_wait_winners(side.iishanten) if side.iishanten else set())
            if current_category is not None and not _category_holds(dealer.river, candidate, current_category):
                continue
            if any(p.is_furiten() for p in players.values()):
                continue

            dora = _take_random(counts, rng)
            for name, p in players.items():
                target_hand = 14 if name == PLAYER_HERO else 13
                while len(p.hand) < target_hand:
                    p.hand.append(_take_random(counts, rng))
            hero_template = bind_entry_discards(hero_template, hero.hand, unavailable=(candidate,))
            hero.iishanten = hero_template
            for side in sides:
                if side.iishanten is not None:
                    side.iishanten = bind_entry_discards(side.iishanten, side.hand)
            ura = [_take_random(counts, rng) for _ in range(13)]
            live = [_take_random(counts, rng) for _ in range(LIVE_WALL_TILES)]
            wall = WallState(players, dora, ura, live, candidate, current_category, hero_template, tuple(p.name for p in sides if p.state == "T"))
            wall.assert_invariants()
            return wall
        except ValueError:
            continue
    raise RuntimeError("could not construct a legal physical initial state")


def _remove_one(items: list[int], t: int) -> None:
    items.remove(t)


def _draw(wall: WallState) -> int | None:
    return wall.live_wall.pop(0) if wall.live_wall else None


def _discard(player: PlayerState, t: int) -> None:
    _remove_one(player.hand, t)
    player.river.append(t)


def _safe_through(player: PlayerState, opponent: PlayerState, t: int) -> None:
    # Genbutsu belongs to the riichi player that actually saw the tile pass.
    if opponent.riichi and t not in opponent.wait.winners:
        opponent.genbutsu.add(t)


def _ronners(players: Iterable[PlayerState], discarder: str, t: int) -> list[PlayerState]:
    result = []
    for p in players:
        # Ron is governed by the actual wait, not by the behavioural riichi
        # flag.  Open T-state side hands can ron as well.
        if p.name != discarder and not p.folded and p.wait is not None and t in p.wait.winners:
            assert not p.is_furiten()
            result.append(p)
    return result


def _number_safety(opponent: PlayerState, t: int) -> str:
    if t in opponent.genbutsu or t in opponent.river:
        return "genbutsu"
    if t >= 27:
        return "honor"
    anchors = suji_anchors(t)
    if anchors and anchors.issubset(set(opponent.river) | opponent.genbutsu):
        return "double_suji"
    if anchors and anchors & (set(opponent.river) | opponent.genbutsu):
        return "suji"
    return "nosuji"


def _danger_rate_for_tile(opponent: PlayerState, t: int, turn: int) -> float:
    safety = _number_safety(opponent, t)
    if safety == "genbutsu":
        return 0.0
    if t >= 27:
        return 0.045
    rank = rank_of(t)
    index = max(1, min(19, turn)) - 1
    if safety in ("suji", "double_suji") and rank in (3, 7):
        return TURN_DANGER["suji_37"][index]
    if rank in (2, 8):
        return TURN_DANGER["no_28"][index]
    if rank in (3, 7):
        return TURN_DANGER["no_37"][index]
    return TURN_DANGER["no_5" if rank == 5 else "no_46"][index]


def _joint_danger(hero: PlayerState, opponents: Iterable[PlayerState], t: int, turn: int) -> float:
    # Strategy thresholds use behavioural danger only.  Actual ron always uses
    # the wait templates; this value never decides whether a tile wins.
    risks = []
    for p in opponents:
        if not p.riichi:
            continue
        risks.append(_danger_rate_for_tile(p, t, turn))
    return 1.0 - math.prod(1.0 - r for r in risks)


def _threshold(strategy: str, turn: int) -> float:
    index = max(1, min(19, turn)) - 1
    if strategy == "suji37":
        return TURN_DANGER["suji_37"][index]
    if strategy == "nosuji2378":
        return max(TURN_DANGER["no_28"][index], TURN_DANGER["no_37"][index])
    return {"allout": 1.0, "fixed05": 0.05, "fixed10": 0.10, "fixed15": 0.15, "fold": -1.0}[strategy]


def _pick_discard(player: PlayerState, drawn: int, opponents: Iterable[PlayerState], strategy: str, rng: random.Random, turn: int) -> tuple[int, bool]:
    legal = [t for t in player.hand if player.wait is None or t not in player.wait.winners]
    if player.iishanten is not None:
        protected = {plan.entry_discard for plan in player.iishanten.plans}
        preserved = [t for t in legal if t not in protected]
        if preserved:
            legal = preserved
    if not legal:
        legal = list(player.hand)
    if strategy == "fold" or player.folded:
        player.folded = True
        ranked = sorted(legal, key=lambda x: _joint_danger(player, opponents, x, turn))
        if rng.random() < FOLD_IMPERFECT_TSUMOGIRI and drawn in legal:
            return drawn, True
        return ranked[0], True
    danger = _joint_danger(player, opponents, drawn, turn)
    if danger <= _threshold(strategy, turn) and drawn in legal:
        return drawn, False
    player.folded = True
    return min(legal, key=lambda x: _joint_danger(player, opponents, x, turn)), True


@dataclass(frozen=True)
class Result:
    payoff: float
    event: str
    route: str
    reached_tenpai: bool
    current_dealer_hit: bool
    current_side_hit: bool
    current_ippatsu_loss: float
    continuation_payoff: float
    wait_kinds: tuple[str, ...]


def _ura_count(wall: WallState, rng: random.Random) -> int:
    # Retains the paper's 12 single + 1 pair local ura model.
    roll = rng.randrange(34)
    return 2 if roll == 0 else 1 if roll <= 12 else 0


def _hero_score(*, tsumo: bool, ippatsu: bool, ura: int) -> float:
    """Child 40-fu score table, starting from the stated 4-han hand.

    Ura and ippatsu change the hand's han, so they cross the mangan/haneman
    boundary instead of being incorrectly priced as a fixed 1,000 points.
    ``tsumo`` is retained for a common Python/C++ interface: the returned value
    is the total received from the table, before the already-modelled sticks.
    """
    han = 4 + int(tsumo) + int(ippatsu) + ura
    total = 8_000.0 if han <= 5 else 12_000.0 if han <= 7 else 16_000.0
    return total


def _settle_draw(hero: PlayerState, players: Iterable[PlayerState]) -> float:
    tenpai = sum(p.wait is not None and not p.folded for p in players)
    hero_tenpai = hero.wait is not None and not hero.folded
    if tenpai in (0, 4):
        return 0.0
    return (3000.0 / tenpai if hero_tenpai else -3000.0 / (4 - tenpai))


def set_folded(player: PlayerState) -> None:
    """A player who gives up no longer has an attacking wait or improvements."""
    player.folded = True
    player.wait = None
    player.iishanten = None
    player.riichi = False


def _enter_tenpai(player: PlayerState, drawn: int, rng: random.Random, *, riichi_probability: float) -> bool:
    if player.iishanten is None:
        return False
    plan = player.iishanten.plan_for(drawn)
    if plan is None:
        return False
    wait = plan.wait
    if plan.entry_discard < 0 or plan.entry_discard not in player.hand or plan.entry_discard in wait.winners:
        raise AssertionError("unbound or unavailable tenpai entry discard")
    _discard(player, plan.entry_discard)
    player.wait = wait
    player.iishanten = None
    player.riichi = player.name == PLAYER_HERO or rng.random() < riichi_probability
    player.riichi_paid = player.riichi
    assert not player.is_furiten()
    return True


def _resolve_discard(wall: WallState, discarder: PlayerState, t: int, rng: random.Random, *, ippatsu_dealer: bool) -> tuple[float | None, str | None]:
    ronnners = _ronners(wall.players.values(), discarder.name, t)
    if ronnners:
        hero_delta = 0.0
        for winner in ronnners:
            if winner.name == PLAYER_HERO:
                hero_delta += _hero_score(tsumo=False, ippatsu=False, ura=_ura_count(wall, rng)) + 1_000.0 * wall.riichi_sticks
            elif discarder.name == PLAYER_HERO:
                hero_delta -= DEALER_IPPATSU_RON_LOSS if winner.name == PLAYER_DEALER and ippatsu_dealer else DEALER_RON_LOSS if winner.name == PLAYER_DEALER else RIICHI_SIDE_RON_LOSS if winner.riichi else SIDE_RON_LOSS
        if discarder.name == PLAYER_HERO:
            return hero_delta, "hero_deal_later"
        if any(winner.name == PLAYER_HERO for winner in ronnners):
            return hero_delta, "hero_ron"
        return 0.0, "side_ron"
    for p in wall.players.values():
        if p.name != discarder.name:
            _safe_through(discarder, p, t)
    return None, None


def simulate_branch(shape_key: str, rng: random.Random, *, hero_strategy: str, side_mode: str, side_riichi_probability: float, current_category: str = "nosuji2378", current_passed_tile: int | None = None, weights: WeightConfig = DEFAULT_WEIGHTS) -> Result:
    wall = generate_initial_state(rng, shape_key, current_category=current_category, candidate_override=current_passed_tile, weights=weights)
    hero, dealer = wall.players[PLAYER_HERO], wall.players[PLAYER_DEALER]
    if side_mode == "heads_up":
        # West and North remain physical full-fold players: they keep their
        # hands, take their turns, consume wall tiles, and can deal into an
        # active player.  Folding removes only their own waits/improvements.
        for name in SIDE_PLAYERS:
            set_folded(wall.players[name])
    candidate = wall.current_candidate
    current_discard = candidate
    if hero_strategy == "fold":
        # Immediate fold is a different current action, not "push the named
        # candidate once, then fold next turn".  The 2% imperfect branch keeps
        # the just-drawn candidate; otherwise choose the lowest joint danger
        # from the actual fourteen-tile hand without protecting local blocks.
        decision_rng = random.Random()
        decision_rng.setstate(rng.getstate())
        if decision_rng.random() >= FOLD_IMPERFECT_TSUMOGIRI:
            active = [p for p in wall.players.values() if p.name != PLAYER_HERO and p.riichi]
            current_discard = min(hero.hand, key=lambda tile: _joint_danger(hero, active, tile, INITIAL_TURN))
    # South's actual fourteenth tile is checked before becoming its ninth river
    # tile.  This preserves all 136 physical cards at both sides of the action.
    current_dealer_hit = dealer.wait is not None and current_discard in dealer.wait.winners
    current_side_hit = any(wall.players[n].state == "T" and wall.players[n].wait is not None and current_discard in wall.players[n].wait.winners for n in SIDE_PLAYERS) if side_mode == "full" else False
    current_loss, _ = _resolve_discard(wall, hero, current_discard, rng, ippatsu_dealer=True)
    _discard(hero, current_discard)
    if hero_strategy == "fold" and current_loss is None:
        set_folded(hero)
    wall.assert_after_current_discard()
    if current_loss is not None:
        return Result(current_loss, "hero_deal_current", "current", False, current_dealer_hit, current_side_hit, DEALER_IPPATSU_RON_LOSS if current_dealer_hit else 0.0, 0.0, tuple(p.wait.kind for p in wall.players.values() if p.wait))

    payoff = 0.0
    reached_tenpai = False
    dealer_ippatsu = True
    route = "continued"
    # West and North act before the dealer's next draw.  Their discard can be
    # an ippatsu ron for the dealer, but that terminal event costs hero zero.
    order_names = (*SIDE_PLAYERS, PLAYER_DEALER, PLAYER_HERO)
    order = [wall.players[n] for n in order_names]
    turn = 10
    while wall.live_wall:
        for player in order:
            drawn = _draw(wall)
            if drawn is None:
                break
            player.hand.append(drawn)
            if player.wait is not None and drawn in player.wait.winners:
                if player.name == PLAYER_HERO:
                    payoff += _hero_score(tsumo=True, ippatsu=False, ura=_ura_count(wall, rng)) + 1_000.0 * wall.riichi_sticks
                    return Result(payoff, "hero_tsumo", route, reached_tenpai, False, False, 0.0, payoff, tuple(p.wait.kind for p in wall.players.values() if p.wait))
                if player.name == PLAYER_DEALER:
                    payment = DEALER_IPPATSU_TSUMO_PAYMENT if dealer_ippatsu else DEALER_TSUMO_PAYMENT
                    payoff -= payment
                    return Result(payoff, "dealer_ippatsu_tsumo" if dealer_ippatsu else "dealer_tsumo", route, reached_tenpai, False, False, 0.0, payoff, tuple(p.wait.kind for p in wall.players.values() if p.wait))
                payoff -= RIICHI_SIDE_TSUMO_PAYMENT if player.riichi else SIDE_TSUMO_PAYMENT
                return Result(payoff, "side_tsumo", route, reached_tenpai, False, False, 0.0, payoff, tuple(p.wait.kind for p in wall.players.values() if p.wait))
            # Riichi hands are forced to tsumogiri after the self-tsumo check.
            if player.riichi:
                _discard(player, drawn)
                terminal, event = _resolve_discard(wall, player, drawn, rng, ippatsu_dealer=dealer_ippatsu)
                if terminal is not None:
                    return Result(payoff + terminal, event or "riichi_ron", route, reached_tenpai, False, False, 0.0, payoff + terminal, tuple(p.wait.kind for p in wall.players.values() if p.wait))
                if player.name == PLAYER_DEALER:
                    dealer_ippatsu = False
                    turn += 1
                continue
            if player.state == "H" and _enter_tenpai(player, drawn, rng, riichi_probability=side_riichi_probability):
                if player.riichi:
                    wall.riichi_sticks += 1
                    if player.name == PLAYER_HERO:
                        payoff -= HERO_RIICHI_COST
                if player.name == PLAYER_HERO:
                    reached_tenpai, route = True, player.wait.kind
                terminal, event = _resolve_discard(wall, player, player.river[-1], rng, ippatsu_dealer=dealer_ippatsu)
                if terminal is not None:
                    return Result(payoff + terminal, event or "ron", route, reached_tenpai, False, False, 0.0, payoff + terminal, tuple(p.wait.kind for p in wall.players.values() if p.wait))
                continue
            if player.name == PLAYER_HERO:
                opponents = [dealer] + ([wall.players[n] for n in SIDE_PLAYERS] if side_mode == "full" else [])
                discard, folded = _pick_discard(player, drawn, opponents, hero_strategy, rng, turn)
                if folded:
                    route = "fold"
            else:
                opponents = [p for p in wall.players.values() if p.name != player.name and p.riichi]
                side_strategy = "nosuji2378" if player.state == "T" else "suji37" if player.state == "H" else "fold"
                discard, folded = _pick_discard(player, drawn, opponents, side_strategy, rng, turn)
            _discard(player, discard)
            if folded:
                set_folded(player)
            terminal, event = _resolve_discard(wall, player, discard, rng, ippatsu_dealer=dealer_ippatsu)
            if terminal is not None:
                if player.name == PLAYER_HERO:
                    payoff += terminal
                    return Result(payoff, event or "hero_deal_after_riichi", route, reached_tenpai, False, False, 0.0, payoff, tuple(p.wait.kind for p in wall.players.values() if p.wait))
                payoff += terminal
                return Result(payoff, event or "side_ron", route, reached_tenpai, False, False, 0.0, payoff, tuple(p.wait.kind for p in wall.players.values() if p.wait))
            if player.name == PLAYER_DEALER:
                dealer_ippatsu = False
                turn += 1

    payoff += _settle_draw(hero, wall.players.values())
    return Result(payoff, "draw_tenpai" if hero.wait else "draw_noten", route, reached_tenpai, False, False, 0.0, payoff, tuple(p.wait.kind for p in wall.players.values() if p.wait))


@dataclass
class Summary:
    ev: float
    se: float
    events: Counter[str]
    routes: Counter[str]
    reached_tenpai: int
    current_dealer_hits: int
    current_side_hits: int
    current_ippatsu_loss: float
    continuation_total: float
    current_safe_count: int
    wait_kinds: Counter[str]
    payoffs: list[float] = field(repr=False)


def run(shape: object, trials: int, seed: int, *, hero_strategy: str, side_mode: str, side_riichi_probability: float, current_passed_tile: int | None = None, current_category: str = "nosuji2378", weights: WeightConfig = DEFAULT_WEIGHTS) -> Summary:
    shape_key = shape.key if hasattr(shape, "key") else str(shape)
    outcomes: list[Result] = []
    # Independent per-route seeds make common-random-number comparisons stable.
    for index in range(trials):
        outcomes.append(simulate_branch(shape_key, random.Random(seed + 1_000_003 * index), hero_strategy=hero_strategy, side_mode=side_mode, side_riichi_probability=side_riichi_probability, current_category=current_category, current_passed_tile=current_passed_tile, weights=weights))
    payoffs = [x.payoff for x in outcomes]
    mean = sum(payoffs) / trials
    variance = sum((x - mean) ** 2 for x in payoffs) / trials
    safe = [x for x in outcomes if not x.current_dealer_hit and not x.current_side_hit]
    kinds = Counter(kind for x in outcomes for kind in x.wait_kinds)
    for kind in ("ryanmen", "kanchan", "penchan", "shanpon", "tanki"):
        kinds.setdefault(kind, 0)
    return Summary(mean, math.sqrt(variance / trials), Counter(x.event for x in outcomes), Counter(x.route for x in outcomes), sum(x.reached_tenpai for x in outcomes), sum(x.current_dealer_hit for x in outcomes), sum(x.current_side_hit for x in outcomes), sum(x.current_ippatsu_loss for x in outcomes), sum(x.continuation_payoff for x in safe), len(safe), kinds, payoffs)


def paired_difference(attack: Summary, fold: Summary) -> tuple[float, float]:
    differences = [a - b for a, b in zip(attack.payoffs, fold.payoffs)]
    mean = sum(differences) / len(differences)
    variance = sum((x - mean) ** 2 for x in differences) / len(differences)
    return mean, math.sqrt(variance / len(differences))


def current_push_ev(continuation_ev: float, danger: float) -> float:
    return -danger * DEALER_IPPATSU_RON_LOSS + (1.0 - danger) * continuation_ev


def break_even_danger(continuation_ev: float, fold_ev: float) -> float:
    return (continuation_ev - fold_ev) / (DEALER_IPPATSU_RON_LOSS + continuation_ev)


def tile_class(t: int) -> str:
    if t >= 27:
        return "honor"
    r = rank_of(t)
    return "one_nine" if r in (1, 9) else "two_eight" if r in (2, 8) else "three_seven" if r in (3, 7) else "four_five_six"


def flattened_koba_targets(*, include_totals: bool = False) -> dict[str, float]:
    result = {f"{klass}.{kind}": value for klass, row in KOBA_CONTRIBUTIONS.items() for kind, value in row.items()}
    if include_totals:
        result.update({f"{klass}.total": value for klass, value in KOBA_TOTALS.items()})
    return result


CALIBRATION_KEYS = tuple(flattened_koba_targets())
CALIBRATION_KEY_INDEX = {key: index for index, key in enumerate(CALIBRATION_KEYS)}
CALIBRATION_TRAINING_SEED = 20260809
CALIBRATION_VALIDATION_SEED = 20260811
CALIBRATION_TRAINING_TRIALS = 20_000
CALIBRATION_VALIDATION_TRIALS = 10_000


@dataclass
class CalibrationBucket:
    feature_counts: tuple[int, ...]
    samples: int
    opportunities: list[int]
    hits: list[int]


@dataclass
class CalibrationStatistics:
    trials: int
    seed: int
    buckets: dict[tuple[int, ...], CalibrationBucket]
    proposal_parameters: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationObservation:
    wait: WaitTemplate
    dealer_river: frozenset[int]
    candidate_types: frozenset[int]
    proposal_feature_counts: tuple[int, ...]


def _state_template_features(state: WallState) -> tuple[int, ...]:
    templates: list[WaitTemplate] = []
    for player in state.players.values():
        if player.wait is not None:
            templates.append(player.wait)
        if player.iishanten is not None:
            templates.extend(player.iishanten.component_waits)
    return tuple(sum(calibration_feature_counts(template)[index] for template in templates) for index in range(len(CALIBRATION_PARAMETER_NAMES)))


def sample_calibration_observation(rng: random.Random, weights: WeightConfig = DEFAULT_WEIGHTS) -> CalibrationObservation:
    """Build the exact South-seat physical state needed by wait calibration.

    It allocates all 136 tiles: dealer 13, South 14, West/North 13 each;
    rivers 9/8/8/8; one visible and 13 hidden dead-wall tiles; and a 36-tile
    live wall.  Only the dealer wait and river are returned because the rest are
    nuisance allocations after their physical invariants have been checked.
    """
    shape_key = rng.choice(tuple(shape.key for shape in SHAPES))
    candidate = rng.randrange(34)
    # Only the target (dealer) generator carries the fitted family here.  The
    # other local structures remain nuisance draws from the fixed uniform legal
    # prior while still reserving real tiles in the accepted 136-tile state.
    state = generate_initial_state(
        rng,
        shape_key,
        current_category=None,
        candidate_override=candidate,
        weights=DEFAULT_WEIGHTS,
        dealer_weights=weights,
    )
    dealer = state.players[PLAYER_DEALER]
    wait = dealer.wait
    assert wait is not None and not dealer.is_furiten()
    # koba's denominator enumerates candidate tile *types* immediately after
    # the riichi declaration discard; it is not conditional on one particular
    # opponent currently holding that type.  Genbutsu are excluded, while wall
    # and visible-tile effects enter through whether a legal wait state survives.
    candidates = frozenset(tile for tile in range(34) if tile not in dealer.river)
    assert candidates
    # The calibration family is the dealer's wait generator.  Hero and side
    # templates are nuisance allocations drawn from the fixed legal-template
    # proposal, but they are still physically present before the state is
    # accepted.  Reweighting only the target template keeps the estimator for
    # that generator identifiable while retaining rejection from four-copy,
    # river, furiten, dead-wall, and live-wall constraints.
    return CalibrationObservation(wait, frozenset(dealer.river), candidates, calibration_feature_counts(wait))


def calibration_samples(*, trials: int = CALIBRATION_TRAINING_TRIALS, seed: int = CALIBRATION_TRAINING_SEED, weights: WeightConfig = DEFAULT_WEIGHTS) -> tuple[CalibrationObservation, ...]:
    """Materialise calibration observations; aggregation is preferred for fits."""
    return tuple(sample_calibration_observation(random.Random(seed + 104_729 * index), weights) for index in range(trials))


def aggregate_calibration_samples(samples: Iterable[CalibrationObservation], *, seed: int = 0, proposal_weights: WeightConfig = DEFAULT_WEIGHTS) -> CalibrationStatistics:
    buckets: dict[tuple[int, ...], CalibrationBucket] = {}
    trials = 0
    for observation in samples:
        trials += 1
        wait = observation.wait
        features = observation.proposal_feature_counts
        bucket = buckets.get(features)
        if bucket is None:
            bucket = CalibrationBucket(features, 0, [0] * len(CALIBRATION_KEYS), [0] * len(CALIBRATION_KEYS))
            buckets[features] = bucket
        bucket.samples += 1
        for candidate in observation.candidate_types:
            klass = tile_class(candidate)
            for kind in KOBA_CONTRIBUTIONS[klass]:
                index = CALIBRATION_KEY_INDEX[f"{klass}.{kind}"]
                bucket.opportunities[index] += 1
                bucket.hits[index] += int(wait.kind == kind and candidate in wait.winners)
    return CalibrationStatistics(trials, seed, buckets, weight_parameters(proposal_weights))


def calibration_sufficient_statistics(*, trials: int = CALIBRATION_TRAINING_TRIALS, seed: int = CALIBRATION_TRAINING_SEED, weights: WeightConfig = DEFAULT_WEIGHTS) -> CalibrationStatistics:
    """Stream full states into compact sufficient statistics."""
    buckets: dict[tuple[int, ...], CalibrationBucket] = {}
    for index in range(trials):
        observation = sample_calibration_observation(random.Random(seed + 104_729 * index), weights)
        wait = observation.wait
        features = observation.proposal_feature_counts
        bucket = buckets.get(features)
        if bucket is None:
            bucket = CalibrationBucket(features, 0, [0] * len(CALIBRATION_KEYS), [0] * len(CALIBRATION_KEYS))
            buckets[features] = bucket
        bucket.samples += 1
        for candidate in observation.candidate_types:
            klass = tile_class(candidate)
            for kind in KOBA_CONTRIBUTIONS[klass]:
                cell = CALIBRATION_KEY_INDEX[f"{klass}.{kind}"]
                bucket.opportunities[cell] += 1
                bucket.hits[cell] += int(wait.kind == kind and candidate in wait.winners)
    return CalibrationStatistics(trials, seed, buckets, weight_parameters(weights))


def _aggregated_prediction(statistics: CalibrationStatistics, parameters: tuple[float, ...], *, gradient: bool = False) -> tuple[dict[str, float], list[list[float]] | None]:
    numerators = [0.0] * len(CALIBRATION_KEYS)
    denominators = [0.0] * len(CALIBRATION_KEYS)
    numerator_derivatives = [[0.0] * len(parameters) for _ in CALIBRATION_KEYS] if gradient else None
    denominator_derivatives = [[0.0] * len(parameters) for _ in CALIBRATION_KEYS] if gradient else None
    for bucket in statistics.buckets.values():
        mass = math.exp(sum((value - proposal) * count for value, proposal, count in zip(parameters, statistics.proposal_parameters, bucket.feature_counts)))
        for cell in range(len(CALIBRATION_KEYS)):
            numerator = mass * bucket.hits[cell]
            denominator = mass * bucket.opportunities[cell]
            numerators[cell] += numerator
            denominators[cell] += denominator
            if gradient:
                assert numerator_derivatives is not None and denominator_derivatives is not None
                for parameter, count in enumerate(bucket.feature_counts):
                    if count:
                        numerator_derivatives[cell][parameter] += numerator * count
                        denominator_derivatives[cell][parameter] += denominator * count
    predictions = {key: numerators[index] / denominators[index] for index, key in enumerate(CALIBRATION_KEYS)}
    if not gradient:
        return predictions, None
    derivatives = [[0.0] * len(parameters) for _ in CALIBRATION_KEYS]
    assert numerator_derivatives is not None and denominator_derivatives is not None
    for cell in range(len(CALIBRATION_KEYS)):
        fitted = numerators[cell] / denominators[cell]
        for parameter in range(len(parameters)):
            derivatives[cell][parameter] = (numerator_derivatives[cell][parameter] - fitted * denominator_derivatives[cell][parameter]) / denominators[cell]
    return predictions, derivatives


def full_state_contributions(samples: Iterable[CalibrationObservation] | CalibrationStatistics, weights: WeightConfig) -> dict[str, float]:
    statistics = samples if isinstance(samples, CalibrationStatistics) else aggregate_calibration_samples(samples)
    return _aggregated_prediction(statistics, weight_parameters(weights))[0]


def _add_total_residuals(residual: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    for klass, row in KOBA_CONTRIBUTIONS.items():
        keys = [f"{klass}.{kind}" for kind in row]
        target = KOBA_TOTALS[klass]
        fitted = sum(residual[key]["fitted"] for key in keys)
        se = math.sqrt(sum(residual[key]["standard_error"] ** 2 for key in keys))
        residual[f"{klass}.total"] = {"target": target, "fitted": fitted, "difference": fitted - target, "standard_error": se}
    return residual


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system with pivoted Gaussian elimination."""
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("singular calibration normal matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column])]
    return [augmented[row][-1] for row in range(size)]


def fit_max_entropy(targets: Mapping[str, float] | None = None, *, iterations: int = 250, learning_rate: float = 0.025, regularization: float = 0.0005, samples: tuple[CalibrationObservation, ...] | CalibrationStatistics | None = None) -> tuple[WeightConfig, dict[str, dict[str, float]]]:
    """Fit 17 constrained kind/location parameters by damped Gauss--Newton.

    ``learning_rate`` is retained for command/API compatibility but the solver
    now chooses each step by backtracking.  This removes the material residual
    left by the old fixed-rate Adam loop and gives a meaningful infeasibility
    check before any temptation to add per-tile parameters.
    """
    target_map = dict(flattened_koba_targets() if targets is None else targets)
    statistics = calibration_sufficient_statistics() if samples is None else samples if isinstance(samples, CalibrationStatistics) else aggregate_calibration_samples(samples)
    parameters = list(statistics.proposal_parameters)
    scale = 0.01  # objective residuals are in percentage-point units
    damping = 1e-3

    def objective(values: tuple[float, ...]) -> float:
        predicted, _ = _aggregated_prediction(statistics, values)
        residual_sum = sum(((predicted[key] - target_map[key]) / scale) ** 2 for key in CALIBRATION_KEYS)
        return residual_sum + regularization * sum(value * value for value in values)

    current_objective = objective(tuple(parameters))
    for _ in range(iterations):
        predicted, derivatives = _aggregated_prediction(statistics, tuple(parameters), gradient=True)
        assert derivatives is not None
        size = len(parameters)
        normal = [[0.0] * size for _ in range(size)]
        right = [-regularization * value for value in parameters]
        for cell, key in enumerate(CALIBRATION_KEYS):
            residual_value = (predicted[key] - target_map[key]) / scale
            jacobian = [value / scale for value in derivatives[cell]]
            for row in range(size):
                right[row] -= jacobian[row] * residual_value
                for column in range(size):
                    normal[row][column] += jacobian[row] * jacobian[column]
        for index in range(size):
            normal[index][index] += regularization + damping
        try:
            step = _solve_linear_system(normal, right)
        except ValueError:
            damping *= 10.0
            continue
        accepted = False
        factor = 1.0
        while factor >= 1.0 / 1024.0:
            proposal = tuple(max(-8.0, min(8.0, value + factor * delta)) for value, delta in zip(parameters, step))
            proposal_objective = objective(proposal)
            if proposal_objective < current_objective:
                improvement = current_objective - proposal_objective
                parameters = list(proposal)
                current_objective = proposal_objective
                damping = max(1e-9, damping / 2.0)
                accepted = True
                if improvement < 1e-10:
                    break
                break
            factor /= 2.0
        if not accepted:
            damping *= 10.0
            if damping > 1e9:
                break
    fitted = weights_from_parameters(parameters, regularization=regularization, source=f"aggregated full-state max-entropy fit; n={statistics.trials}; seed={statistics.seed}")
    return fitted, calibration_residuals(fitted, targets=target_map, samples=statistics)


def calibration_residuals(
    weights: WeightConfig,
    *,
    targets: Mapping[str, float] | None = None,
    samples: tuple[CalibrationObservation, ...] | CalibrationStatistics | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate fixed weights on legal full states without re-fitting them."""
    target_map = dict(flattened_koba_targets() if targets is None else targets)
    statistics = calibration_sufficient_statistics() if samples is None else samples if isinstance(samples, CalibrationStatistics) else aggregate_calibration_samples(samples)
    predicted = full_state_contributions(statistics, weights)
    opportunities = Counter()
    for bucket in statistics.buckets.values():
        for index, key in enumerate(CALIBRATION_KEYS):
            opportunities[key] += bucket.opportunities[index]
    residual = {
        key: {
            "target": target,
            "fitted": predicted[key],
            "difference": predicted[key] - target,
            "standard_error": math.sqrt(predicted[key] * (1.0 - predicted[key]) / opportunities[key]) if opportunities[key] else 0.0,
        }
        for key, target in target_map.items()
    }
    return _add_total_residuals(residual)


def calibration_diagnostics(weights: WeightConfig, *, trials: int = CALIBRATION_VALIDATION_TRIALS, seed: int = CALIBRATION_VALIDATION_SEED) -> dict[str, dict[str, float]]:
    """Sample full legal states for joint contributions and Monte-Carlo SEs.

    Every non-genbutsu candidate of a target tile class is an opportunity for
    every target wait form.  Thus the numerator estimates ``P(kind and winner)``
    rather than the conditional ``P(winner | kind)``.
    """
    per_sample: list[tuple[list[int], list[int]]] = []
    total_hits = [0] * len(CALIBRATION_KEYS)
    total_opportunities = [0] * len(CALIBRATION_KEYS)
    for index in range(trials):
        observation = sample_calibration_observation(random.Random(seed + 104_729 * index), weights)
        wait = observation.wait
        hits = [0] * len(CALIBRATION_KEYS)
        opportunities = [0] * len(CALIBRATION_KEYS)
        for candidate in observation.candidate_types:
            klass = tile_class(candidate)
            for kind in KOBA_CONTRIBUTIONS[klass]:
                cell = CALIBRATION_KEY_INDEX[f"{klass}.{kind}"]
                opportunities[cell] += 1
                hits[cell] += int(wait.kind == kind and candidate in wait.winners)
        per_sample.append((hits, opportunities))
        for cell in range(len(CALIBRATION_KEYS)):
            total_hits[cell] += hits[cell]
            total_opportunities[cell] += opportunities[cell]
    expected = flattened_koba_targets()
    result = {}
    for cell, (key, target) in enumerate(expected.items()):
        value = total_hits[cell] / total_opportunities[cell]
        mean_opportunities = total_opportunities[cell] / trials
        influences = [(hits[cell] - value * opportunities[cell]) / mean_opportunities for hits, opportunities in per_sample]
        influence_mean = sum(influences) / trials
        variance = sum((item - influence_mean) ** 2 for item in influences) / max(1, trials - 1)
        result[key] = {"target": target, "fitted": value, "difference": value - target, "standard_error": math.sqrt(variance / trials)}
    return _add_total_residuals(result)


def joint_contribution_counts(observations: Iterable[tuple[WaitTemplate, set[int]]]) -> tuple[Counter[str], Counter[str]]:
    """Count joint target cells from ``(wait, own_river)`` observations.

    This small standalone unit is intentionally used by tests: its denominator
    covers all valid non-genbutsu candidates, including samples whose wait has a
    different form from the target cell.
    """
    hits: Counter[str] = Counter()
    opportunities: Counter[str] = Counter()
    for wait, own_river in observations:
        for candidate in range(34):
            if candidate in own_river:
                continue
            for kind in KOBA_CONTRIBUTIONS[tile_class(candidate)]:
                key = f"{tile_class(candidate)}.{kind}"
                opportunities[key] += 1
                hits[key] += int(wait.kind == kind and candidate in wait.winners)
    return hits, opportunities


def calibration_payload(config: WeightConfig = FITTED_WEIGHTS, targets: Mapping[str, float] | None = None, *, diagnostics_trials: int = 0) -> dict[str, object]:
    fitted = config
    if diagnostics_trials:
        residual = calibration_diagnostics(fitted, trials=diagnostics_trials)
    elif fitted == FITTED_WEIGHTS and targets is None:
        residual = {
            key: {"target": target, "fitted": value, "difference": value - target, "standard_error": standard_error}
            for key, (target, value, standard_error) in FITTED_CALIBRATION_RESIDUALS.items()
        }
    else:
        residual = calibration_residuals(fitted, targets=targets)
    base = [cell for key, cell in residual.items() if not key.endswith(".total")]
    weighted_rmse = math.sqrt(sum(cell["difference"] ** 2 for cell in base) / len(base)) if base else 0.0
    max_total_error = max(abs(cell["difference"]) for key, cell in residual.items() if key.endswith(".total"))
    return {
        "engine_version": ENGINE_VERSION,
        "source": KOBA_SOURCE,
        "targets": flattened_koba_targets(include_totals=True) if targets is None else dict(targets),
        "weights": asdict(fitted),
        "residuals": residual,
        "weighted_rmse": weighted_rmse,
        "validation_trials": diagnostics_trials or (20_000 if fitted == FITTED_WEIGHTS and targets is None else 0),
        "acceptance": {
            "basic_rmse_target": 0.0025,
            "main_total_error_target": 0.005,
            "observed_basic_rmse": weighted_rmse,
            "observed_max_total_error": max_total_error,
            "met": weighted_rmse <= 0.0025 and max_total_error <= 0.005,
        },
        "note": "Targets are candidate-tile winning contributions, not per-hand wait-form shares. The compact mirror-symmetric model keeps its independent-sample residual instead of adding suit- or tile-specific parameters.",
    }


def load_weights(path: Path | None) -> WeightConfig:
    if path is None:
        return FITTED_WEIGHTS
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = raw.get("weights", raw)
    return WeightConfig(dict(raw["kind_log_weights"]), dict(raw["position_log_weights"]), float(raw.get("regularization", 1.0)), str(raw.get("source", f"loaded from {path}")), dict(raw.get("internal_log_weights", {})), dict(raw.get("location_log_weights", {})))


SHAPES = (Shape("ryanmen_ryanmen", "两面＋两面"), Shape("ryanmen_ryankan", "两面＋两嵌"), Shape("ryanmen_kanchan", "两面＋嵌张"))


def _json_summary(summary: Summary, fold: Summary, trials: int, pair: tuple[float, float], calibration: dict[str, object]) -> dict[str, object]:
    event_rates = {key: value / trials for key, value in summary.events.items()}
    any_hit = (summary.current_dealer_hits + summary.current_side_hits) / trials
    continuation = summary.continuation_total / summary.current_safe_count if summary.current_safe_count else 0.0
    return {"ev": summary.ev, "se": summary.se, "fold_ev": fold.ev, "fold_se": fold.se, "ev_difference": summary.ev - fold.ev, "paired_ev_difference_se": pair[1], "break_even_danger": break_even_danger(continuation, fold.ev), "event_counts": dict(summary.events), "event_rates": event_rates, "route_counts": dict(summary.routes), "wait_kind_counts": dict(summary.wait_kinds), "current_hit_rates": {"dealer": summary.current_dealer_hits / trials, "initial_side": summary.current_side_hits / trials, "any": any_hit}, "current_dealer_ippatsu_contribution": -summary.current_ippatsu_loss / trials, "continuation_ev_after_safe": continuation, "calibration_residuals": calibration["residuals"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--side-mode", choices=("full", "heads_up"), default="full")
    parser.add_argument("--shape", choices=("all", "ryanmen_ryanmen", "ryanmen_ryankan", "ryanmen_kanchan"), default="all")
    parser.add_argument("--hero-strategy", choices=("all", "category", "fixed", "fold", *HERO_STRATEGIES), default="all")
    parser.add_argument("--side-riichi-probability", type=float, default=DEFAULT_SIDE_RIICHI_ON_TENPAI)
    parser.add_argument("--current-passed-tile", type=int, default=None, metavar="TILE_TYPE")
    parser.add_argument("--current-category", choices=("all", "genbutsu", "suji37", "nosuji2378", "nosuji456"), default="nosuji2378")
    parser.add_argument("--fold-ev", type=float, default=None)
    parser.add_argument("--fold-se", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="emit one structured object")
    parser.add_argument("--weights", type=Path, default=None, help="calibrated JSON weights shared with the C++ engine")
    parser.add_argument("--self-test", action="store_true", help="run deterministic physical-state checks and exit")
    parser.add_argument("--write-calibration", type=Path, default=None)
    args = parser.parse_args()
    if not 0 <= args.side_riichi_probability <= 1:
        parser.error("side-riichi-probability must be between 0 and 1")
    if args.current_passed_tile is not None and not 0 <= args.current_passed_tile < 34:
        parser.error("current-passed-tile must be between 0 and 33")
    if (args.fold_ev is None) != (args.fold_se is None):
        parser.error("fold-ev and fold-se must be supplied together")
    weights = load_weights(args.weights)
    if args.self_test:
        for key in (shape.key for shape in SHAPES):
            first = generate_initial_state(random.Random(args.seed), key, weights=weights)
            second = generate_initial_state(random.Random(args.seed), key, weights=weights)
            first.assert_invariants()
            assert first.physical_counts() == second.physical_counts()
        print("self-test: ok")
        return
    if args.write_calibration:
        args.write_calibration.write_text(json.dumps(calibration_payload(weights), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    strategies = HERO_STRATEGIES if args.hero_strategy == "all" else CATEGORY_STRATEGIES if args.hero_strategy == "category" else FIXED_DANGER_STRATEGIES if args.hero_strategy == "fixed" else (args.hero_strategy,)
    shapes = [s for s in SHAPES if args.shape == "all" or s.key == args.shape]
    calibration = calibration_payload(weights)
    records = []
    categories = ("genbutsu", "suji37", "nosuji2378", "nosuji456") if args.current_category == "all" else (args.current_category,)
    for shape in shapes:
        for category in categories:
            fold = run(shape.key, args.trials, args.seed, hero_strategy="fold", side_mode=args.side_mode, side_riichi_probability=args.side_riichi_probability, current_category=category, current_passed_tile=args.current_passed_tile, weights=weights)
            for strategy in strategies:
                attack = run(shape.key, args.trials, args.seed, hero_strategy=strategy, side_mode=args.side_mode, side_riichi_probability=args.side_riichi_probability, current_category=category, current_passed_tile=args.current_passed_tile, weights=weights)
                pair = paired_difference(attack, fold)
                record = {"shape": shape.key, "strategy": strategy, "current_category": category, **_json_summary(attack, fold, args.trials, pair, calibration)}
                records.append(record)
    if args.json:
        action_order = ["hero_current_discard_check", "west_draw_discard", "north_draw_discard", "dealer_draw"]
        print(json.dumps({"engine_version": ENGINE_VERSION, "engine": "python", "seed": args.seed, "trials": args.trials, "config": {"side_mode": args.side_mode, "side_riichi_probability": args.side_riichi_probability, "seat_map": {"dealer": "east", "hero": "south", "side1": "west", "side2": "north"}, "river_lengths": {"dealer": 9, "hero": 9, "side1": 8, "side2": 8}, "action_order": action_order, "live_wall_tiles": LIVE_WALL_TILES, "hero_unseen_positions": UNKNOWN_POSITIONS, "weights": asdict(weights)}, "results": records, "calibration": calibration}, ensure_ascii=False, indent=2))
        return
    for r in records:
        print(f"{r['shape']}\t{r['strategy']}\t{r['current_category']}\tEV={r['ev']:.2f}±{r['se']:.2f}\tΔfold={r['ev_difference']:.2f}±{r['paired_ev_difference_se']:.2f}\tdealer-hit={100*r['current_hit_rates']['dealer']:.3f}%\tside-hit={100*r['current_hit_rates']['initial_side']:.3f}%")


if __name__ == "__main__":
    main()
