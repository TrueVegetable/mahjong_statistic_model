"""Small, deterministic push/fold policy helpers shared by experiments."""
from __future__ import annotations

from typing import Iterable, Mapping

from analyze_tile_danger_by_turn import safety_class

_TIER = {
    "genbutsu": 0, "suji": 1, "honor": 2, "double_suji": 3,
    "suji28": 4, "suji37": 5, "no_suji19": 6, "half_suji": 7,
    "no_suji28": 8, "no_suji37": 9, "no_suji46": 10,
}


def safety_tier(tile_type: int, safe_types: Iterable[int], *, is_dora: bool = False,
                dora_steps: int = 1) -> int:
    cls = safety_class(tile_type, safe_types)
    if cls == "genbutsu":
        tier = 0
    elif tile_type >= 27:
        tier = 2
    else:
        rank = tile_type % 9 + 1
        base = {"suji": (1 if rank in (1, 9) else 4 if rank in (2, 8) else 5),
                "double_suji": 3, "half_suji": 7, "no_suji":
                (6 if rank in (1, 9) else 8 if rank in (2, 8) else
                 9 if rank in (3, 7) else 10)}[cls]
        tier = base
    return tier if not is_dora or tier == 0 else tier + dora_steps


def candidate_key(tiers: Iterable[int], known_copies: int, strategy: str) -> tuple:
    values = tuple(tiers)
    if strategy == "mean_safety":
        return (sum(values) / len(values), max(values), -known_copies)
    if strategy == "minimax_safety":
        return (max(values), sum(values) / len(values), -known_copies)
    raise ValueError(strategy)


def eligible_half_push(candidates: Iterable[Mapping], *, max_tier: int = 5) -> list[Mapping]:
    items = list(candidates)
    if not items:
        return []
    best = min(int(x["post_shanten"]) for x in items)
    return [x for x in items if int(x["post_shanten"]) == best
            and max(map(int, x["tiers"])) <= max_tier]
