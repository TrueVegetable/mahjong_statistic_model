#!/usr/bin/env python3
"""Order-free hypergeometric winning-tile model.

Each active hand has K winning tiles distributed over its own position universe
U. Every active player receives the same number m of future draws. Inactive
players are folded: their draw positions remain inside U but cannot terminate
the hand.

The main output uses the paper's simple approximation:

1. hypergeometric zero-hit probabilities determine exhaustive-draw probability;
2. expected active-draw winner counts weight whose winner appears first;
3. equal future draw counts split each owner's first winner symmetrically among
   the active drawers.

An exact finite sum over the hypergeometric counts is printed as a diagnostic.
For two active players an optional ordered calculation measures the error from
letting the opponent act before the hero in every cycle.
"""

from __future__ import annotations

import argparse
import itertools
import math
from functools import cache


def hypergeom_pmf(universe: int, winners: int, draws: int, hits: int) -> float:
    if hits < 0 or hits > winners or hits > draws:
        return 0.0
    misses = draws - hits
    if misses > universe - winners:
        return 0.0
    return (
        math.comb(winners, hits)
        * math.comb(universe - winners, misses)
        / math.comb(universe, draws)
    )


def zero_hit_probability(universe: int, winners: int, active_draws: int) -> float:
    if winners > universe - active_draws:
        return 0.0
    return math.comb(universe - active_draws, winners) / math.comb(universe, winners)


def exact_count_owner_probabilities(
    universes: list[int],
    winners: list[int],
    active_draws: int,
) -> tuple[float, list[float]]:
    distributions = [
        [hypergeom_pmf(universe, count, active_draws, hits) for hits in range(count + 1)]
        for universe, count in zip(universes, winners)
    ]
    draw_probability = 0.0
    owner_probabilities = [0.0] * len(winners)

    for hit_counts in itertools.product(*(range(count + 1) for count in winners)):
        probability = math.prod(
            distributions[index][hits] for index, hits in enumerate(hit_counts)
        )
        total_hits = sum(hit_counts)
        if total_hits == 0:
            draw_probability += probability
            continue
        for index, hits in enumerate(hit_counts):
            owner_probabilities[index] += probability * hits / total_hits

    return draw_probability, owner_probabilities


def expected_count_model(
    universes: list[int],
    winners: list[int],
    active_players: int,
    draws_per_active: int,
) -> tuple[float, list[float], list[float]]:
    total_active_draws = active_players * draws_per_active
    means = [
        total_active_draws * count / universe
        for universe, count in zip(universes, winners)
    ]
    draw_probability = math.prod(
        zero_hit_probability(universe, count, total_active_draws)
        for universe, count in zip(universes, winners)
    )
    mean_total = sum(means)
    if mean_total == 0:
        weights = [0.0] * active_players
    else:
        weights = [mean / mean_total for mean in means]
    owner_probabilities = [(1.0 - draw_probability) * weight for weight in weights]
    return draw_probability, means, owner_probabilities


def ordered_two_player_distribution(
    hero_universe: int,
    opponent_universe: int,
    hero_winners: int,
    opponent_winners: int,
    draws_per_active: int,
) -> dict[str, float]:
    """Independent-position diagnostic with opponent acting before hero.

    Independent placement can assign both abstract winner sets to one draw slot.
    In that case the drawer's self-draw is resolved before any discard.
    """

    schedule = ("O", "H") * draws_per_active

    @cache
    def solve(draw_index: int, hero_left: int, opponent_left: int) -> tuple[float, ...]:
        if draw_index == len(schedule):
            return (1.0, 0.0, 0.0, 0.0, 0.0)

        hero_hit = hero_left / (hero_universe - draw_index)
        opponent_hit = opponent_left / (opponent_universe - draw_index)
        actor = schedule[draw_index]
        result = [0.0] * 5  # draw, hero tsumo, hero ron, opponent tsumo, hero deal-in

        continuation = solve(draw_index + 1, hero_left, opponent_left)
        neither = (1.0 - hero_hit) * (1.0 - opponent_hit)
        for index, probability in enumerate(continuation):
            result[index] += neither * probability

        hero_only = hero_hit * (1.0 - opponent_hit)
        opponent_only = (1.0 - hero_hit) * opponent_hit
        both = hero_hit * opponent_hit

        if actor == "H":
            result[1] += hero_only + both
            result[4] += opponent_only
        else:
            result[2] += hero_only
            result[3] += opponent_only + both

        return tuple(result)

    values = solve(0, hero_winners, opponent_winners)
    return dict(zip(("draw", "hero_tsumo", "hero_ron", "opponent_tsumo", "hero_deal"), values))


def two_player_ev(
    distribution: dict[str, float],
    *,
    hero_tsumo: float,
    hero_ron: float,
    opponent_tsumo_payment: float,
    opponent_ron_loss: float,
    draw_payoff: float,
) -> float:
    return (
        distribution["hero_tsumo"] * hero_tsumo
        + distribution["hero_ron"] * hero_ron
        - distribution["opponent_tsumo"] * opponent_tsumo_payment
        - distribution["hero_deal"] * opponent_ron_loss
        + distribution["draw"] * draw_payoff
    )


def parse_opponent_winners(text: str, expected_count: int) -> list[int]:
    values = [int(value) for value in text.split(",") if value]
    if len(values) != expected_count:
        raise ValueError(
            f"expected {expected_count} opponent winner counts, received {len(values)}"
        )
    if any(value < 0 for value in values):
        raise ValueError("winner counts must be non-negative")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-players", type=int, choices=(2, 3), default=2)
    parser.add_argument("--draws-per-active", type=int, default=9)
    parser.add_argument("--hero-winners", type=int, default=4)
    parser.add_argument("--hero-unseen", type=int, default=88)
    parser.add_argument(
        "--opponent-winners",
        default="4",
        help="comma-separated winner counts; supply one value for two active players and two for three",
    )
    parser.add_argument("--opponent-universe", type=int, default=136)
    parser.add_argument("--verify-order", action="store_true")
    parser.add_argument("--hero-ron", type=float, default=8_000.0)
    parser.add_argument("--hero-tsumo", type=float, default=8_000.0)
    parser.add_argument("--opponent-ron-loss", type=float, default=7_500.0)
    parser.add_argument("--opponent-tsumo-payment", type=float, default=3_400.0)
    parser.add_argument("--draw-payoff", type=float, default=0.0)
    args = parser.parse_args()

    if args.draws_per_active < 0:
        parser.error("draws-per-active must be non-negative")
    if args.hero_winners < 0:
        parser.error("hero-winners must be non-negative")

    try:
        opponent_winners = parse_opponent_winners(
            args.opponent_winners, args.active_players - 1
        )
    except ValueError as error:
        parser.error(str(error))

    universes = [args.hero_unseen] + [args.opponent_universe] * len(opponent_winners)
    winners = [args.hero_winners] + opponent_winners
    active_draws = args.active_players * args.draws_per_active
    if any(active_draws > universe for universe in universes):
        parser.error("total active draws cannot exceed a position universe")
    if any(count > universe for universe, count in zip(universes, winners)):
        parser.error("winner count cannot exceed its position universe")

    draw_probability, means, approximate_owners = expected_count_model(
        universes, winners, args.active_players, args.draws_per_active
    )
    exact_draw, exact_owners = exact_count_owner_probabilities(
        universes, winners, active_draws
    )

    print(f"active_players={args.active_players}")
    print(f"draws_per_active={args.draws_per_active}")
    print(f"universes={universes}")
    print(f"winner_counts={winners}")
    print(f"draw_probability={draw_probability:.9%}")
    print("owner\texpected_active_winners\tapprox_first_owner\texact_count_first_owner")
    for index, (mean, approximate, exact) in enumerate(
        zip(means, approximate_owners, exact_owners)
    ):
        label = "hero" if index == 0 else f"opponent_{index}"
        print(f"{label}\t{mean:.6f}\t{approximate:.9%}\t{exact:.9%}")

    print("owner\tdrawer\tapprox_terminal_probability")
    for owner_index, owner_probability in enumerate(approximate_owners):
        owner_label = "hero" if owner_index == 0 else f"opponent_{owner_index}"
        for drawer_index in range(args.active_players):
            drawer_label = "hero" if drawer_index == 0 else f"opponent_{drawer_index}"
            print(
                f"{owner_label}\t{drawer_label}\t"
                f"{owner_probability / args.active_players:.9%}"
            )

    print(f"probability_sum={draw_probability + sum(approximate_owners):.12f}")
    print(f"exact_count_probability_sum={exact_draw + sum(exact_owners):.12f}")

    if args.active_players == 2:
        approximate_distribution = {
            "draw": draw_probability,
            "hero_tsumo": approximate_owners[0] / 2.0,
            "hero_ron": approximate_owners[0] / 2.0,
            "opponent_tsumo": approximate_owners[1] / 2.0,
            "hero_deal": approximate_owners[1] / 2.0,
        }
        payoff_args = {
            "hero_tsumo": args.hero_tsumo,
            "hero_ron": args.hero_ron,
            "opponent_tsumo_payment": args.opponent_tsumo_payment,
            "opponent_ron_loss": args.opponent_ron_loss,
            "draw_payoff": args.draw_payoff,
        }
        approximate_ev = two_player_ev(approximate_distribution, **payoff_args)
        print(f"approx_EV={approximate_ev:.2f}")

        if args.verify_order:
            exact_count_distribution = {
                "draw": exact_draw,
                "hero_tsumo": exact_owners[0] / 2.0,
                "hero_ron": exact_owners[0] / 2.0,
                "opponent_tsumo": exact_owners[1] / 2.0,
                "hero_deal": exact_owners[1] / 2.0,
            }
            ordered = ordered_two_player_distribution(
                args.hero_unseen,
                args.opponent_universe,
                args.hero_winners,
                opponent_winners[0],
                args.draws_per_active,
            )
            order_free_ev = two_player_ev(exact_count_distribution, **payoff_args)
            ordered_ev = two_player_ev(ordered, **payoff_args)
            print("ordered_event\tprobability")
            for event, probability in ordered.items():
                print(f"{event}\t{probability:.9%}")
            print(f"order_free_exact_count_EV={order_free_ev:.2f}")
            print(f"ordered_EV={ordered_ev:.2f}")
            print(f"ordered_minus_order_free_EV={ordered_ev - order_free_ev:.2f}")


if __name__ == "__main__":
    main()
