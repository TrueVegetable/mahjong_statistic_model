#!/usr/bin/env python3
"""Cached scoring callback for native closed-hand push/fold decisions."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import hand_efficiency_sim as sim
from hand_monte_carlo_report import primary_wait_category

TILE_TYPES = 34
NO_POINTS = -1
SCORE_RESPONSE_FIELDS = (
    "good_wait", "cheap_all_unseen", "exact_1300_bad_wait", "valuable_dama",
    "valuable_ron_mask", "haneman_lock", "haneman_ron_mask", "legal_ron_mask",
    "min_dama_ron_points", "min_riichi_ron_points", "mean_riichi_ron_points",
)


@dataclass(frozen=True)
class ScoringConfig:
    cheap_max_total_han: int = 1
    valuable_bad_wait_points: int | None = 5200
    two_riichi_bad_wait_min_ron: int = 7700
    enable_haneman_lock: bool = True


def _valid(score: sim.HandScore) -> bool:
    return score.error is None and score.points is not None and score.han is not None


def _player_wind(request: Mapping[str, Any]) -> int:
    """Accept native's seat and the older explicit-wind spelling."""
    if "seat" in request:
        seat = int(request["seat"])
        if not 0 <= seat < 4:
            raise ValueError("seat must be in 0..3")
        return 27 + seat
    wind = int(request["player_wind"])
    if not 27 <= wind <= 30:
        raise ValueError("player_wind must be in 27..30")
    return wind


def _known_physical(request: Mapping[str, Any]) -> tuple[int, ...]:
    raw = request.get("known_physical", request.get("known_physical_tiles"))
    if raw is None:
        raise ValueError("request must contain known_physical")
    known = tuple(sorted(map(int, raw)))
    if len(set(known)) != len(known) or any(tile < 0 or tile >= 136 for tile in known):
        raise ValueError("known_physical must contain unique physical tile ids")
    return known


def _dora_indicators(request: Mapping[str, Any]) -> tuple[int, ...]:
    raw = request.get("dora_indicators")
    if raw is None:
        raw = (request["dora_indicator"],)
    return tuple(map(int, raw))


class PushFoldScorer:
    """Memoizes completed hands and exposes the fixed native numeric response."""

    def __init__(self, scorer: sim.ScoreEvaluator | None = None) -> None:
        self.scorer = scorer or sim.MahjongRepositoryScorer()

    @lru_cache(maxsize=250_000)
    def _score(self, complete_hand: tuple[int, ...], win: int, player_wind: int,
               round_wind: int, dora: tuple[int, ...], riichi: bool, tsumo: bool,
               ura: tuple[int, ...], ippatsu: bool, haitei: bool,
               is_dealer: bool) -> sim.HandScore:
        context = sim.ScoreContext(
            is_tsumo=tsumo, is_riichi=riichi, is_ippatsu=ippatsu,
            # Guards use non-dealer-normalized values but retain player_wind,
            # so seat-wind yaku remain in the hand's scoring context.
            is_dealer=is_dealer, is_haitei=haitei, is_tenhou=False,
            is_chiihou=False, player_wind=player_wind, round_wind=round_wind,
            dora_indicators=dora, ura_dora_indicators=ura if riichi else (),
        )
        score = self.scorer.score(complete_hand, win, context)
        if not is_dealer and _valid(score):
            return self._nondealer_normalized(score, tsumo)
        return score

    @staticmethod
    def _nondealer_normalized(score: sim.HandScore, tsumo: bool) -> sim.HandScore:
        """Reprice an already evaluated hand with non-dealer payments.

        ``ScoreContext.is_dealer`` is deliberately kept for policy adapters,
        but the pinned third-party calculator derives dealer status from the
        player wind.  Repricing from its returned han/fu keeps actual wind
        yaku while making guard thresholds comparable to non-dealer values.
        """
        from mahjong.constants import SOUTH
        from mahjong.hand_calculating.hand_config import HandConfig
        from mahjong.hand_calculating.scores import ScoresCalculator

        assert score.han is not None and score.fu is not None
        yaku_level = str(score.cost.get("yaku_level", "")).lower()
        is_yakuman = "yakuman" in yaku_level and "kazoe" not in yaku_level
        cost = dict(ScoresCalculator.calculate_scores(
            int(score.han), int(score.fu),
            HandConfig(is_tsumo=tsumo, player_wind=SOUTH), is_yakuman=is_yakuman,
        ))
        return sim.HandScore(
            points=int(cost["total"]), han=score.han, fu=score.fu,
            yaku=score.yaku, cost=cost,
        )

    def _score_request(self, request: Mapping[str, Any], win: int, *, riichi: bool,
                       tsumo: bool, ura: Sequence[int] = (), ippatsu: bool = False,
                       haitei: bool = False, is_dealer: bool,
                       complete_hand: Sequence[int] | None = None) -> sim.HandScore:
        hand = (tuple(sorted(map(int, request["hand_physical"])))
                if complete_hand is None else tuple(sorted(map(int, complete_hand))))
        if len(set(hand)) != len(hand) or any(tile < 0 or tile >= 136 for tile in hand):
            raise ValueError("hand_physical must contain distinct physical tile ids")
        if len(hand) == 13:
            if win in hand:
                raise ValueError("13-tile hand already contains winning physical tile")
            hand = tuple(sorted((*hand, int(win))))
        if len(hand) != 14 or int(win) not in hand:
            raise ValueError("scoring requires a completed 14-tile hand containing win_tile")
        return self._score(
            hand, int(win), _player_wind(request), int(request["round_wind"]),
            _dora_indicators(request), riichi, tsumo, tuple(map(int, ura)),
            ippatsu, haitei, is_dealer,
        )

    @staticmethod
    def _structural_wins(counts: Sequence[int]) -> tuple[int, ...]:
        wins: list[int] = []
        for tile in range(TILE_TYPES):
            complete = list(counts)
            if complete[tile] < 4:
                complete[tile] += 1
                if sim.shanten(complete) == -1:
                    wins.append(tile)
        return tuple(wins)

    @staticmethod
    def _unseen_wins(known: set[int], structural_types: Sequence[int]) -> list[int]:
        return [physical for tile_type in structural_types
                for physical in sim.PHYSICAL_COPIES_BY_TYPE[tile_type]
                if physical not in known]

    @staticmethod
    def _representative_win(hand: Sequence[int], tile_type: int) -> int:
        held = set(map(int, hand))
        return next(tile for tile in sim.PHYSICAL_COPIES_BY_TYPE[tile_type]
                    if tile not in held)

    @staticmethod
    def _legacy_bad_wait(category: str, unseen_types: set[int]) -> bool:
        if category in ("kanchan", "penchan"):
            return True
        if category not in ("shanpon", "tanki"):
            return False
        return not any(tile >= 27 for tile in unseen_types)

    @staticmethod
    def _score_points(score: sim.HandScore) -> int:
        return int(score.points) if _valid(score) else 0

    @staticmethod
    def _assert_expected_error(score: sim.HandScore, *, allow_no_yaku: bool) -> None:
        if _valid(score):
            return
        error = (score.error or "incomplete score result").lower().replace("_", " ")
        if allow_no_yaku and "no yaku" in error:
            return
        raise ValueError(f"structural winning hand failed scoring: {score.error}")

    def score_request(self, request: Mapping[str, Any], config: ScoringConfig) -> list[int | float]:
        hand = tuple(sorted(map(int, request["hand_physical"])))
        counts = tuple(sim.counts_from_physical(hand))
        if len(hand) != 13 or sum(counts) != 13 or len(set(hand)) != 13:
            raise ValueError("request must contain 13 distinct physical hand tiles")
        known = set(_known_physical(request))
        if not set(hand).issubset(known):
            raise ValueError("known_physical must include the hand")

        structural_types = self._structural_wins(counts)
        unseen = self._unseen_wins(known, structural_types)
        unseen_types = {tile // 4 for tile in unseen}
        category = primary_wait_category(counts, structural_types)
        honor_live = any(tile >= 27 for tile in unseen_types)
        good_wait = int(bool(unseen) and (category in ("sanmenchan", "ryanmen") or (
            category in ("shanpon", "tanki", "chiitoitsu_tanki") and honor_live
        )))

        # All decision guards intentionally use non-dealer-normalized values.
        riichi_scores = [self._score_request(request, tile, riichi=True, tsumo=False,
                                             is_dealer=False) for tile in unseen]
        dama_scores = [self._score_request(request, tile, riichi=False, tsumo=False,
                                           is_dealer=False) for tile in unseen]
        for score in riichi_scores:
            self._assert_expected_error(score, allow_no_yaku=False)
        for score in dama_scores:
            self._assert_expected_error(score, allow_no_yaku=True)
        cheap = int(bool(riichi_scores) and all(
            _valid(score) and int(score.han) <= config.cheap_max_total_han
            for score in riichi_scores
        ))
        exact_1300 = int(
            _player_wind(request) != 27
            and self._legacy_bad_wait(category, unseen_types)
            and bool(riichi_scores)
            and all(_valid(score) for score in riichi_scores)
            and sum(int(score.points) for score in riichi_scores) / len(riichi_scores) == 1300
        )

        valuable_mask = 0
        valuable = False
        if (config.valuable_bad_wait_points is not None
                and self._legacy_bad_wait(category, unseen_types)):
            for tile, score in zip(unseen, dama_scores):
                if _valid(score):
                    valuable_mask |= 1 << (tile // 4)
                    valuable |= int(score.points) >= config.valuable_bad_wait_points

        legal_mask = 0
        for tile_type in structural_types:
            representative = self._representative_win(hand, tile_type)
            score = self._score_request(request, representative, riichi=False,
                                        tsumo=False, is_dealer=False)
            self._assert_expected_error(score, allow_no_yaku=True)
            if _valid(score):
                legal_mask |= 1 << tile_type

        haneman_mask = 0
        haneman = bool(unseen) and config.enable_haneman_lock
        for tile in unseen:
            ron = self._score_request(request, tile, riichi=False, tsumo=False,
                                      is_dealer=False)
            self._assert_expected_error(ron, allow_no_yaku=True)
            if _valid(ron):
                if int(ron.points) < 12000:
                    haneman = False
                else:
                    haneman_mask |= 1 << (tile // 4)
                continue
            # Tsumo is an alternative only if dama ron has no yaku.
            tsumo = self._score_request(request, tile, riichi=False, tsumo=True,
                                        is_dealer=False)
            self._assert_expected_error(tsumo, allow_no_yaku=False)
            if not (_valid(tsumo) and int(tsumo.points) >= 12000):
                haneman = False

        actual_dama_points = [self._score_points(self._score_request(
            request, tile, riichi=False, tsumo=False,
            is_dealer=_player_wind(request) == 27)) for tile in unseen]
        actual_riichi_points = [self._score_points(self._score_request(
            request, tile, riichi=True, tsumo=False,
            is_dealer=_player_wind(request) == 27)) for tile in unseen]
        values = (
            good_wait, cheap, exact_1300, int(valuable), valuable_mask,
            int(haneman), haneman_mask, legal_mask,
            min(actual_dama_points, default=NO_POINTS),
            min(actual_riichi_points, default=NO_POINTS),
            sum(actual_riichi_points) / len(actual_riichi_points)
            if actual_riichi_points else float(NO_POINTS),
        )
        return list(values)

    def score_realized_win(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Score one terminal win in the requested ron or tsumo context only."""
        win = int(request["win_tile"])
        riichi = bool(request.get("is_riichi", False))
        ura = tuple(map(int, request.get("ura_dora_indicators", ()))) if riichi else ()
        score = self._score_request(
            request, win, riichi=riichi, tsumo=bool(request["is_tsumo"]), ura=ura,
            ippatsu=bool(request.get("ippatsu", False)),
            haitei=bool(request.get("haitei", False)),
            is_dealer=_player_wind(request) == 27,
            complete_hand=request["hand_physical"],
        )
        return {
            "points": int(score.points) if _valid(score) else NO_POINTS,
            "han": int(score.han) if score.han is not None else None,
            "fu": int(score.fu) if score.fu is not None else None,
            "yaku": list(score.yaku), "cost": dict(score.cost), "error": score.error,
        }


def score_request(request: Mapping[str, Any], config: ScoringConfig) -> list[int | float]:
    return PushFoldScorer().score_request(request, config)
