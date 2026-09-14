import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs"))
import hand_efficiency_sim as sim  # noqa: E402
import hand_monte_carlo_report as report  # noqa: E402


class ConstantRonScorer:
    name = "constant_ron"

    def score(self, hand, win_tile, context):
        del hand, win_tile
        if context.is_tsumo or context.is_dealer:
            raise AssertionError("report must request non-dealer ron scoring")
        return sim.HandScore(3900, 3, 40, ("test",), {"total": 3900})


class AvailabilityCategoryTests(unittest.TestCase):
    """Contract tests for the player-independent and player-relative views."""

    def test_table_level_categories_are_complete_and_non_overlapping(self):
        category = report.availability_table_category
        # The round wind has priority over the generic other-winds bucket.
        self.assertEqual(category(27, 27), "round_wind")
        self.assertEqual(category(28, 27), "seat_wind_for_one_player")
        self.assertEqual(category(31, 27), "dragons")
        self.assertEqual(category(0, 27), "19")
        self.assertEqual(category(1, 27), "28")
        self.assertEqual(category(2, 27), "37")
        self.assertEqual(category(3, 27), "46")
        self.assertEqual(category(4, 27), "5")
        self.assertEqual(category(5, 27), "46")
        self.assertEqual(category(8, 27), "19")

    def test_player_relative_honor_priority_handles_double_and_guest_winds(self):
        category = report.availability_player_category
        # East in an East round is double wind; it must not be reported as two
        # separate observations or fall through to a guest wind.
        self.assertEqual(category(27, 27, 27), "double_wind")
        self.assertEqual(category(28, 27, 27), "guest_wind")
        # South in an East round has a seat wind, East is the round wind, and
        # West/North are guest winds.
        self.assertEqual(category(28, 28, 27), "seat_wind")
        self.assertEqual(category(27, 28, 27), "round_wind")
        self.assertEqual(category(29, 28, 27), "guest_wind")
        self.assertEqual(category(31, 28, 27), "dragons")
        # Number categories are independent of the observing player and red
        # five is represented by the ordinary 5 tile type.
        self.assertEqual(category(4, 28, 27), "5")
        self.assertEqual(category(sim.parse_single_tile_type("0m"), 28, 27), "5")

    def test_availability_score_skips_zero_expected_denominator(self):
        self.assertAlmostEqual(
            report.availability_score(2, 4, 34, 86),
            1.2647058823529411,
        )
        aggregate = report._AvailabilityAggregate()
        aggregate.add(0, actual=1.0, expected=0.0)
        aggregate.add(0, actual=2.0, expected=1.0)
        result = aggregate.summary()
        self.assertEqual(result["valid_observations"], 1)
        self.assertEqual(result["skipped_zero_expected"], 1)
        self.assertAlmostEqual(result["mean_score"], 2.0)

    def test_availability_aggregation_reports_mean_and_pooled_ratio(self):
        # The samples deliberately have different expected counts: arithmetic
        # mean of ratios is 1.5, while pooled actual/expected is 14/10.
        samples = [
            {"table_id": 0, "score": 1.0, "actual": 6.0, "expected": 6.0},
            {"table_id": 1, "score": 2.0, "actual": 8.0, "expected": 4.0},
            {"table_id": 2, "score": None, "actual": 0.0, "expected": 0.0},
        ]
        aggregate = report._AvailabilityAggregate()
        for sample in samples:
            aggregate.add(sample["table_id"], sample["actual"], sample["expected"])
        result = aggregate.summary()
        self.assertAlmostEqual(result["mean_score"], 1.5)
        self.assertAlmostEqual(result["pooled_score"], 14.0 / 10.0)
        self.assertEqual(result["valid_observations"], 2)
        self.assertEqual(result["skipped_zero_expected"], 1)

    def test_availability_aggregation_clusters_player_observations_by_table(self):
        # Four perspectives from one wall are one table-level cluster.  A
        # second table contributes the only between-table variation here.
        samples = [
            {"table_id": 0, "score": 0.0, "actual": 0.0, "expected": 1.0},
            {"table_id": 0, "score": 2.0, "actual": 2.0, "expected": 1.0},
            {"table_id": 0, "score": 0.0, "actual": 0.0, "expected": 1.0},
            {"table_id": 0, "score": 2.0, "actual": 2.0, "expected": 1.0},
            {"table_id": 1, "score": 1.0, "actual": 1.0, "expected": 1.0},
            {"table_id": 1, "score": 1.0, "actual": 1.0, "expected": 1.0},
            {"table_id": 1, "score": 1.0, "actual": 1.0, "expected": 1.0},
            {"table_id": 1, "score": 1.0, "actual": 1.0, "expected": 1.0},
        ]
        aggregate = report._AvailabilityAggregate()
        for sample in samples:
            aggregate.add(sample["table_id"], sample["actual"], sample["expected"])
        result = aggregate.summary()
        self.assertAlmostEqual(result["mean_score"], 1.0)
        self.assertAlmostEqual(result["table_clustered_standard_error"], 0.0)
        self.assertEqual(result["tables"], 2)

    def test_tile_availability_collector_consumes_checkpoint_and_skips_exhausted_tile(self):
        # Build a conservation-valid synthetic turn-9 table.  Every player
        # owns the same 13 tile types; the public river/dora expose four
        # copies of several other types, giving the collector both valid and
        # zero-expected observations without invoking the native binary.
        own = [1 if tile < 13 else 0 for tile in range(34)]
        public = [0] * 34
        remaining_public = 37
        for tile in range(13, 34):
            public[tile] = min(4, remaining_public)
            remaining_public -= public[tile]
            if not remaining_public:
                break
        self.assertEqual(sum(public), 37)
        unseen = [4 - own[tile] - public[tile] for tile in range(34)]
        self.assertTrue(all(0 <= count <= 4 for count in unseen))

        live = [0] * 34
        hidden = [0] * 34
        live_left = 34
        for tile in range(34):
            residual = unseen[tile] - (3 * own[tile])
            self.assertGreaterEqual(residual, 0)
            live[tile] = min(residual, live_left)
            live_left -= live[tile]
            hidden[tile] = residual - live[tile]
        self.assertEqual(sum(live), 34)
        self.assertEqual(sum(hidden), 13)

        players = []
        for seat in range(4):
            opponents = [3 * count for count in own]
            players.append({
                "seat": seat,
                "own_known_counts": own,
                "public_counts": public,
                "unseen_counts": unseen,
                "live_wall_counts": live,
                "hidden_dead_wall_counts": hidden,
                "opponent_hand_counts": opponents,
                "unseen_tiles": sum(unseen),
                "live_wall_tiles": sum(live),
                "hidden_dead_wall_tiles": sum(hidden),
                "opponent_hand_tiles": sum(opponents),
            })
        row = {
            "trial_id": 0,
            "round_wind": 27,
            "outcome": "availability_checkpoint",
            "availability_checkpoint": {
                "target_own_turn": 9,
                "completed_own_turn": 9,
                "live_wall_tiles_remaining": 34,
                "hidden_dead_wall_tiles_remaining": 13,
                "opponent_hand_tiles": 39,
                "public_counts": public,
                "players": players,
            },
        }
        collector = report.TileAvailabilityCollector(9)
        collector.add_row(row)
        payload = collector.to_dict()

        self.assertEqual(payload["snapshots"], 1)
        self.assertEqual(payload["tables"], 1)
        self.assertEqual(
            tuple(payload["table_level"]["categories"]),
            report.AVAILABILITY_TABLE_CATEGORIES,
        )
        self.assertEqual(
            tuple(payload["player_relative"]["categories"]),
            report.AVAILABILITY_PLAYER_CATEGORIES,
        )
        five = payload["table_level"]["categories"]["5"]
        self.assertGreater(five["valid_observations"], 0)
        self.assertGreaterEqual(five["skipped_zero_expected"], 4)
        east_double = payload["player_relative"]["categories"]["double_wind"]
        self.assertGreater(east_double["valid_observations"], 0)
        self.assertEqual(
            len(payload["raw_34_tile_diagnostics"]["by_tile_type"]), 34
        )

        companions = payload["companion_location_scores"]
        self.assertEqual(
            set(("live_wall", "hidden_dead_wall", "opponent_hands")),
            set(companions) - {"units", "definition"},
        )
        self.assertEqual(companions["units"]["counts"], "physical tile copies")
        self.assertIn("conditional_expected_E", companions["definition"])
        for location in ("live_wall", "hidden_dead_wall", "opponent_hands"):
            location_summary = companions[location]["table_level"]["categories"]["19"]
            self.assertIn("actual_copies", location_summary)
            self.assertIn("expected_copies", location_summary)
            if location_summary["expected_copies"]:
                self.assertAlmostEqual(
                    location_summary["pooled_ratio"],
                    location_summary["actual_copies"]
                    / location_summary["expected_copies"],
                )

        # The compartment conservation check is per observer and tile, not
        # merely a table-total check.
        broken = copy.deepcopy(row)
        broken["availability_checkpoint"]["players"][0]["opponent_hand_counts"][0] += 1
        with self.assertRaises(ValueError):
            report.TileAvailabilityCollector(9).add_row(broken)


class WaitSensitiveRonScorer:
    name = "wait_sensitive_ron"

    def score(self, hand, win_tile, context):
        del hand, context
        if win_tile // 4 == 5:
            return sim.HandScore(7000, 4, 40, ("high",), {"total": 7000})
        return sim.HandScore(1000, 1, 30, ("low",), {"total": 1000})


class CheapRonScorer:
    name = "cheap_ron"

    def score(self, hand, win_tile, context):
        del hand, win_tile, context
        return sim.HandScore(1300, 1, 40, ("riichi",), {"total": 1300})


class LowValueYakuScorer:
    name = "low_value_yaku"

    def score(self, hand, win_tile, context):
        del hand, win_tile, context
        return sim.HandScore(2600, 2, 40, ("test_yaku",), {"total": 2600})


class ChiitoitsuValueScorer:
    name = "chiitoitsu_value"

    def score(self, hand, win_tile, context):
        del hand, win_tile, context
        return sim.HandScore(6400, 4, 25, ("chiitoitsu",), {"total": 6400})


class HanemanRonScorer:
    name = "haneman_ron"

    def score(self, hand, win_tile, context):
        del hand, win_tile, context
        return sim.HandScore(12000, 6, 30, ("test_yaku",), {"total": 12000})


class MixedRonTsumoHanemanScorer:
    name = "mixed_ron_tsumo_haneman"

    def score(self, hand, win_tile, context):
        del hand
        if win_tile // 4 == 2 and not context.is_tsumo:
            return sim.HandScore(None, None, None, (), {}, error="no yaku")
        return sim.HandScore(12000, 6, 30, ("test_yaku",), {"total": 12000})


def observation(*, outcome="tenpai", waits=None, live_wall_tiles=20, hand="3345m123p123s111z"):
    waits = waits if waits is not None else [
        {
            "tile_type": 2,
            "name": "3m",
            "visible_remaining": 2,
            "live_wall_remaining": 1,
            "visible_remaining_tile_ids": [10, 11],
            "live_wall_remaining_tile_ids": [10],
            "shapes": ["ryanmen", "tanki"],
        },
        {
            "tile_type": 5,
            "name": "6m",
            "visible_remaining": 4,
            "live_wall_remaining": 2,
            "visible_remaining_tile_ids": [20, 21, 22, 23],
            "live_wall_remaining_tile_ids": [20, 21],
            "shapes": ["ryanmen"],
        },
    ]
    hand_physical = list(sim.parse_physical_hand(hand))
    snapshot = {
        "turn": 5,
        "waits": waits,
        "visible_ukeire": sum(wait["visible_remaining"] for wait in waits),
        "live_wall_ukeire": sum(wait["live_wall_remaining"] for wait in waits),
        "live_wall_tiles_remaining": live_wall_tiles,
        "hand_counts": list(sim.counts_from_physical(hand_physical)),
        "hand_physical": hand_physical,
        "furiten": False,
        "riichi_declared": True,
    }
    return {
        "outcome": outcome,
        "first_tenpai": snapshot,
        "win": None,
        "player_wind": 28,
        "round_wind": 27,
        "dora_indicator": 12,
        "ura_dora_indicator": 13,
    }


class CollectorTests(unittest.TestCase):
    def test_wait_score_and_hit_statistics(self):
        collector = report.MonteCarloReportCollector(ConstantRonScorer())
        collector.add_row(observation())
        collector.add_row(
            observation(
                outcome="exhausted",
                live_wall_tiles=10,
                hand="123456789m123p1z",
                waits=[{
                    "tile_type": 27,
                    "name": "1z",
                    "visible_remaining": 3,
                    "live_wall_remaining": 1,
                    "visible_remaining_tile_ids": [109, 110, 111],
                    "live_wall_remaining_tile_ids": [109],
                    "shapes": ["tanki"],
                }],
            )
        )
        payload = collector.to_dict(run={"engine": "test"})

        scores = payload["nondealer_ron_score"]
        self.assertEqual(scores["unseen_copy_weighted"]["average_points"], 3900)
        self.assertEqual(scores["live_wall_weighted_diagnostic"]["average_points"], 3900)
        self.assertAlmostEqual(payload["han_distribution"]["3"]["share"], 1.0)
        self.assertAlmostEqual(payload["fu_distribution"]["40"]["share"], 1.0)
        self.assertAlmostEqual(payload["han_fu_joint_distribution"]["3han_40fu"]["count"], 2)
        averages = payload["average_wait_tiles_remaining_at_first_tenpai"]
        self.assertEqual(averages["distinct_wait_tile_types"], 1.5)
        self.assertEqual(averages["visible_remaining_copies"], 4.5)
        self.assertEqual(averages["live_wall_remaining_copies"], 2.0)
        rates = payload["tile_hit_rate_by_wait"]["categories"]
        self.assertAlmostEqual(rates["ryanmen"]["average_next_draw_hit_probability"], 0.075)
        self.assertAlmostEqual(rates["ryanmen"]["average_next_draw_hit_probability_when_exposed"], 0.15)
        self.assertAlmostEqual(rates["tanki"]["average_next_draw_hit_probability"], 0.05)
        self.assertEqual(rates["other"]["average_next_draw_hit_probability"], 0.0)
        waits = payload["wait_distribution"]["categories"]
        self.assertEqual(waits["ryanmen"]["hand_share"], 0.5)
        self.assertEqual(waits["tanki"]["hand_share"], 0.5)
        matrix = payload["tile_hit_rate_by_tile_type_and_wait"]
        self.assertAlmostEqual(
            matrix["by_tile_type_and_wait"]["37"]["ryanmen"][
                "average_next_draw_hit_probability"
            ],
            0.025,
        )
        self.assertAlmostEqual(
            matrix["by_tile_type_and_wait"]["456"]["ryanmen"][
                "average_next_draw_hit_probability"
            ],
            0.05,
        )
        self.assertAlmostEqual(
            matrix["by_tile_type_and_wait"]["honor"]["tanki"][
                "average_next_draw_hit_probability"
            ],
            0.05,
        )

    def test_unequal_wait_values_are_weighted_by_remaining_copies(self):
        collector = report.MonteCarloReportCollector(WaitSensitiveRonScorer())
        collector.add_row(observation())
        payload = collector.to_dict(run={})
        scores = payload["nondealer_ron_score"]
        # Two 3m copies at 1000 and four 6m copies at 7000.
        self.assertEqual(scores["unseen_copy_weighted"]["average_points"], 5000)
        # The sampled live wall has one 3m and two 6m copies.
        self.assertEqual(scores["live_wall_weighted_diagnostic"]["average_points"], 5000)
        self.assertAlmostEqual(payload["han_distribution"]["1"]["share"], 1 / 3)
        self.assertAlmostEqual(payload["han_distribution"]["4"]["share"], 2 / 3)

    def test_whole_hand_priority_avoids_ambiguous_tile_double_count(self):
        collector = report.MonteCarloReportCollector(ConstantRonScorer())
        collector.add_row(
            observation(
                hand="1233m123p123s111z",
                waits=[{
                    "tile_type": 2,
                    "name": "3m",
                    "visible_remaining": 2,
                    "live_wall_remaining": 1,
                    "visible_remaining_tile_ids": [10, 11],
                    "live_wall_remaining_tile_ids": [10],
                    "shapes": ["penchan", "tanki"],
                }],
            )
        )
        payload = collector.to_dict(run={})
        waits = payload["wait_distribution"]["categories"]
        self.assertEqual(waits["penchan"]["hand_share"], 1.0)
        self.assertEqual(waits["tanki"]["hand_share"], 0.0)
        hits = payload["tile_hit_rate_by_wait"]["categories"]
        self.assertEqual(hits["penchan"]["average_next_draw_hit_probability"], 0.05)
        self.assertEqual(hits["tanki"]["average_next_draw_hit_probability"], 0.0)

        # 3345 waiting on 3/6 is one ryanmen hand even though 3 can also be a pair.
        self.assertEqual(
            report.primary_wait_category(
                sim.parse_hand("3345m123p123s111z"), (2, 5)
            ),
            "ryanmen",
        )

        # 23456 with two complete melds and a pair waits on 1/4/7.
        self.assertEqual(
            report.primary_wait_category(
                sim.parse_hand("23456m123p123s11z"), (0, 3, 6)
            ),
            "sanmenchan",
        )

    def test_chiitoitsu_kokushi_and_sanmenchan_are_separate(self):
        self.assertEqual(
            report.primary_wait_category(
                sim.parse_hand("1122m1122p1122s1z"), (27,)
            ),
            "chiitoitsu_tanki",
        )
        self.assertEqual(
            report.primary_wait_category(
                sim.parse_hand("19m19p19s1234567z"), sim.TERMINAL_HONOR_TYPES
            ),
            "kokushi",
        )
        self.assertEqual(
            report.primary_wait_category(
                sim.parse_hand("23456m123p123s11z"), (0, 3, 6)
            ),
            "sanmenchan",
        )

    def test_other_waits_are_split_by_distinct_winning_tile_types(self):
        waits = [
            {
                "tile_type": tile_type,
                "name": sim.tile_name(tile_type),
                "visible_remaining": 1,
                "live_wall_remaining": 0,
                "visible_remaining_tile_ids": [tile_type * 4],
                "live_wall_remaining_tile_ids": [],
                "shapes": [],
            }
            for tile_type in (0, 1, 2, 3)
        ]
        collector = report.MonteCarloReportCollector(ConstantRonScorer())
        collector.add_row(observation(hand="123456789m123p1z", waits=waits))
        payload = collector.to_dict(run={})
        breakdown = payload["wait_distribution"]["other_by_distinct_wait_tile_types"]
        self.assertEqual(breakdown["other_tenpai_hands"], 1)
        self.assertEqual(
            breakdown["categories"]["4"],
            {"tenpai_hands": 1, "share_of_other": 1.0, "share_of_all_tenpai": 1.0},
        )

    def test_scoring_always_includes_riichi_and_sampled_ura(self):
        contexts = []

        class CapturingScorer(ConstantRonScorer):
            def score(self, hand, win_tile, context):
                contexts.append(context)
                return super().score(hand, win_tile, context)

        row = observation()
        row["first_tenpai"]["riichi_declared"] = False
        row["ura_dora_indicator"] = 77
        collector = report.MonteCarloReportCollector(CapturingScorer())
        collector.add_row(row)
        self.assertTrue(contexts)
        self.assertTrue(all(context.is_riichi for context in contexts))
        self.assertTrue(all(context.ura_dora_indicators == (77,) for context in contexts))

    def test_cheap_bad_wait_riichi_filter(self):
        bad = observation(
            hand="1233m123p123s111z",
            waits=[{
                "tile_type": 2,
                "name": "3m",
                "visible_remaining": 2,
                "live_wall_remaining": 1,
                "visible_remaining_tile_ids": [10, 11],
                "live_wall_remaining_tile_ids": [10],
                "shapes": ["penchan", "tanki"],
            }],
        )
        collector = report.MonteCarloReportCollector(
            CheapRonScorer(), score_tenpai=False, reject_cheap_bad_wait_riichi=True
        )
        collector.add_row(bad)
        payload = collector.to_dict(run={})
        self.assertEqual(payload["riichi_selection_filter"]["rejected_hands"], 1)
        self.assertEqual(payload["riichi_selection_filter"]["accepted_tenpai_hands"], 0)
        self.assertEqual(payload["riichi_selection_filter"]["rejections_by_wait"], {"penchan": 1})

        honor_tanki = observation(
            hand="123456789m123p1z",
            waits=[{
                "tile_type": 27,
                "name": "1z",
                "visible_remaining": 3,
                "live_wall_remaining": 1,
                "visible_remaining_tile_ids": [109, 110, 111],
                "live_wall_remaining_tile_ids": [109],
                "shapes": ["tanki"],
            }],
        )
        collector.add_row(honor_tanki)
        self.assertEqual(collector.tenpai_hands, 1)

        dead_honor_tanki = observation(
            hand="123456789m123p1z",
            waits=[{
                "tile_type": 27,
                "name": "1z",
                "visible_remaining": 0,
                "live_wall_remaining": 0,
                "visible_remaining_tile_ids": [],
                "live_wall_remaining_tile_ids": [],
                "shapes": ["tanki"],
            }],
        )
        collector.add_row(dead_honor_tanki)
        self.assertEqual(collector.tenpai_hands, 1)
        self.assertEqual(collector.cheap_bad_wait_rejections, 2)

        live_honor_shanpon = observation(
            hand="12355m123p123s11z",
            waits=[
                {
                    "tile_type": 4,
                    "name": "5m",
                    "visible_remaining": 2,
                    "live_wall_remaining": 1,
                    "visible_remaining_tile_ids": [18, 19],
                    "live_wall_remaining_tile_ids": [18],
                    "shapes": ["shanpon"],
                },
                {
                    "tile_type": 27,
                    "name": "1z",
                    "visible_remaining": 1,
                    "live_wall_remaining": 1,
                    "visible_remaining_tile_ids": [110],
                    "live_wall_remaining_tile_ids": [110],
                    "shapes": ["shanpon"],
                },
            ],
        )
        collector.add_row(live_honor_shanpon)
        self.assertEqual(collector.tenpai_hands, 2)

        dead_honor_shanpon = copy.deepcopy(live_honor_shanpon)
        dead_honor_shanpon["first_tenpai"]["waits"][1].update({
            "visible_remaining": 0,
            "live_wall_remaining": 0,
            "visible_remaining_tile_ids": [],
            "live_wall_remaining_tile_ids": [],
        })
        collector.add_row(dead_honor_shanpon)
        self.assertEqual(collector.tenpai_hands, 2)
        self.assertEqual(collector.cheap_bad_wait_rejections, 3)
        self.assertEqual(collector.cheap_bad_wait_rejections_by_wait["shanpon"], 1)

        dealer_bad = bad.copy()
        dealer_bad["player_wind"] = 27
        collector.add_row(dealer_bad)
        self.assertEqual(collector.tenpai_hands, 3)

    def test_continued_rejected_tenpai_stops_on_eligible_or_dama_tsumo(self):
        bad = observation(
            hand="1233m123p123s111z",
            waits=[{
                "tile_type": 2,
                "name": "3m",
                "visible_remaining": 2,
                "live_wall_remaining": 1,
                "visible_remaining_tile_ids": [10, 11],
                "live_wall_remaining_tile_ids": [10],
                "shapes": ["penchan", "tanki"],
            }],
        )["first_tenpai"]
        good = observation()["first_tenpai"]
        base = observation(outcome="trajectory")
        base["trial_id"] = 1
        base["first_tenpai"] = None
        base["table_events"] = [
            {"kind": "tenpai", "seat": 1, "turn": 5, "tenpai": bad},
            {"kind": "shanten_reversal", "seat": 1, "turn": 6},
            {"kind": "tenpai", "seat": 2, "turn": 6, "tenpai": good},
        ]
        collector = report.MonteCarloReportCollector(
            CheapRonScorer(), score_tenpai=False, reject_cheap_bad_wait_riichi=True
        )
        collector.add_continuation_row(base)
        self.assertEqual(collector.trials, 1)
        self.assertEqual(collector.outcomes, {"tenpai": 1})
        self.assertEqual(collector.cheap_bad_wait_rejections, 1)
        self.assertEqual(collector.tenpai_hands, 1)
        self.assertEqual(collector.first_tenpai_seat_counts["west"], 1)
        self.assertEqual(collector.continuation_tables_with_rejection, 1)
        self.assertEqual(collector.continuation_eligible_after_rejection, 1)
        self.assertEqual(collector.continuation_shanten_reversals, 1)

        tsumo_row = copy.deepcopy(base)
        tsumo_row["table_events"] = [
            {"kind": "tenpai", "seat": 1, "turn": 5, "tenpai": bad},
            {"kind": "dama_tsumo", "seat": 1, "turn": 6, "win": {"tile": 10}},
        ]
        tsumo_collector = report.MonteCarloReportCollector(
            CheapRonScorer(), score_tenpai=False, reject_cheap_bad_wait_riichi=True
        )
        tsumo_collector.add_continuation_row(tsumo_row)
        self.assertEqual(tsumo_collector.trials, 1)
        self.assertEqual(tsumo_collector.outcomes, {"dama_tsumo": 1})
        self.assertEqual(tsumo_collector.tenpai_hands, 0)
        self.assertEqual(tsumo_collector.continuation_tsumo_after_rejection, 1)
        self.assertEqual(tsumo_collector.continuation_ron_after_rejection, 0)

        ron_row = copy.deepcopy(tsumo_row)
        ron_row["table_events"][1]["kind"] = "dama_ron"
        ron_collector = report.MonteCarloReportCollector(
            CheapRonScorer(), score_tenpai=False, reject_cheap_bad_wait_riichi=True
        )
        ron_collector.add_continuation_row(ron_row)
        self.assertEqual(ron_collector.trials, 1)
        self.assertEqual(ron_collector.outcomes, {"dama_ron": 1})
        self.assertEqual(ron_collector.tenpai_hands, 0)
        self.assertEqual(ron_collector.continuation_tsumo_after_rejection, 0)
        self.assertEqual(ron_collector.continuation_ron_after_rejection, 1)
        continuation = ron_collector.to_dict(run={})["continued_rejected_tenpai"]
        self.assertEqual(continuation["dama_tsumo_after_rejection"], 0)
        self.assertEqual(continuation["dama_ron_after_rejection"], 1)

    def test_valuable_dama_lower_line_includes_four_han_chiitoitsu_value(self):
        row = observation(
            hand="1233m123p123s111z",
            waits=[{
                "tile_type": 2,
                "name": "3m",
                "visible_remaining": 1,
                "live_wall_remaining": 1,
                "visible_remaining_tile_ids": [10],
                "live_wall_remaining_tile_ids": [10],
                "shapes": ["penchan"],
            }],
        )
        row["trial_id"] = 1
        event = {"kind": "tenpai", "seat": 1, "turn": 5, "tenpai": row["first_tenpai"]}
        lower, mask = report._valuable_bad_wait_dama(
            row, event, ChiitoitsuValueScorer(), 5200
        )
        upper, _ = report._valuable_bad_wait_dama(
            row, event, ChiitoitsuValueScorer(), 8000
        )
        self.assertTrue(lower)
        self.assertEqual(mask, 1 << 2)
        self.assertFalse(upper)

    def test_valuable_dama_rejects_bad_to_bad_transition_below_value_line(self):
        bad = observation(
            hand="1233m123p123s111z",
            waits=[{
                "tile_type": 2,
                "name": "3m",
                "visible_remaining": 2,
                "live_wall_remaining": 1,
                "visible_remaining_tile_ids": [10, 11],
                "live_wall_remaining_tile_ids": [10],
                "shapes": ["penchan", "tanki"],
            }],
        )["first_tenpai"]
        row = observation(outcome="trajectory")
        row.update({"trial_id": 7, "first_tenpai": None})
        row["table_events"] = [
            {"kind": "tenpai", "seat": 1, "turn": 5, "tenpai": bad},
            {"kind": "tenpai", "seat": 1, "turn": 6, "tenpai": bad},
        ]
        old_key = f'7:1:5:{sim.format_counts(bad["hand_counts"])}'
        new_key = f'7:1:6:{sim.format_counts(bad["hand_counts"])}'
        directives = {old_key: report.DAMA_LOCK_FLAG | (1 << 2)}
        reasons = {old_key: "valuable_bad_wait"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            added = report._discover_valuable_dama_directives(
                path, LowValueYakuScorer(), 5200, directives, reasons,
                haneman_or_higher=False,
            )
        self.assertEqual(added, 1)
        self.assertEqual(directives[new_key], report.DAMA_REJECT_TRANSITION_FLAG)

    def test_haneman_dama_accepts_good_wait_before_riichi(self):
        row = observation()
        row["trial_id"] = 1
        event = {"kind": "tenpai", "seat": 1, "turn": 5, "tenpai": row["first_tenpai"]}
        qualifies, mask = report._haneman_minimum_value_dama(
            row, event, HanemanRonScorer()
        )
        bad_wait_only, _ = report._pre_riichi_value_dama(
            row, event, HanemanRonScorer(), 12000, bad_wait_only=True
        )
        self.assertTrue(qualifies)
        self.assertEqual(mask, (1 << 2) | (1 << 5))
        self.assertFalse(bad_wait_only)

    def test_haneman_minimum_uses_tsumo_for_ron_no_yaku_waits(self):
        row = observation()
        row["trial_id"] = 1
        event = {"kind": "tenpai", "seat": 1, "turn": 5, "tenpai": row["first_tenpai"]}
        qualifies, mask = report._haneman_minimum_value_dama(
            row, event, MixedRonTsumoHanemanScorer()
        )
        below_minimum, _ = report._haneman_minimum_value_dama(
            row, event, WaitSensitiveRonScorer()
        )
        self.assertTrue(qualifies)
        self.assertEqual(mask, 1 << 5)
        self.assertFalse(below_minimum)

    def test_shanten_reversal_diagnostics_are_aggregated(self):
        row = observation()
        row["shanten_reversal_opportunities"] = 3
        row["shanten_reversals"] = [{
            "turn": 7,
            "from_shanten": 1,
            "to_shanten": 2,
            "recovered": True,
            "recovery_turns": 2,
        }]
        collector = report.MonteCarloReportCollector(ConstantRonScorer())
        collector.add_row(row)
        payload = collector.to_dict(run={})["shanten_reversal"]
        self.assertEqual(payload["opportunities"], 3)
        self.assertEqual(payload["selected"], 1)
        self.assertEqual(payload["recovery_rate"], 1.0)
        self.assertEqual(payload["mean_recovery_turns"], 2.0)
        self.assertEqual(payload["transitions"], {"1->2": 1})

    def test_python_engine_end_to_end(self):
        args = report._parser().parse_args([
            "--trials", "2",
            "--engine", "python",
            "--policy", "custom",
            "--tanyao-weight", "1",
            "--allow-shanten-reversal",
        ])
        payload = report.run_report(args, scorer=ConstantRonScorer())
        self.assertEqual(payload["trials"], 2)
        self.assertEqual(payload["run"]["engine"], "python")
        self.assertEqual(payload["run"]["policy"], "custom_hand_value")
        self.assertTrue(payload["run"]["policy_parameters"]["shanten_reversal"]["enabled"])


@unittest.skipUnless(shutil.which("c++"), "C++ compiler is not installed")
class NativeWrapperTests(unittest.TestCase):
    def test_native_engine_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "hand_efficiency_sim_cpp"
            subprocess.run(
                [
                    "c++", "-std=c++17", "-O2",
                    str(ROOT / "outputs" / "hand_efficiency_sim.cpp"),
                    "-o", str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            args = report._parser().parse_args([
                "--trials", "2",
                "--engine", "native",
                "--native-binary", str(binary),
                "--policy", "value",
            ])
            payload = report.run_report(args, scorer=ConstantRonScorer())
            table_args = report._parser().parse_args([
                "--trials", "2",
                "--engine", "native",
                "--native-binary", str(binary),
                "--policy", "value",
                "--shared-four-player-table",
            ])
            table_payload = report.run_report(table_args, scorer=ConstantRonScorer())
        self.assertEqual(payload["trials"], 2)
        self.assertEqual(payload["run"]["engine"], "native")
        self.assertEqual(payload["run"]["policy"], "shanten_then_value_ukeire")
        self.assertEqual(
            table_payload["run"]["simulation_scope"],
            "shared_four_player_first_tenpai",
        )
        self.assertFalse(table_payload["run"]["tenpai_scoring_enabled"])
        self.assertEqual(
            sum(
                row["count"]
                for row in table_payload["first_tenpai_seat_distribution"].values()
            ),
            table_payload["outcomes"].get("tenpai", 0),
        )

    def test_native_tile_availability_report_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "hand_efficiency_sim_cpp"
            subprocess.run(
                [
                    "c++", "-std=c++17", "-O2",
                    str(ROOT / "outputs" / "hand_efficiency_sim.cpp"),
                    "-o", str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            args = report._parser().parse_args([
                "--trials", "1",
                "--engine", "native",
                "--native-binary", str(binary),
                "--shared-four-player-table",
                "--tile-availability-turn", "9",
                "--round-wind", "east",
            ])
            payload = report.run_report(args)

            invalid_args = report._parser().parse_args([
                "--trials", "1",
                "--engine", "native",
                "--native-binary", str(binary),
                "--tile-availability-turn", "9",
            ])
            with self.assertRaises(ValueError):
                report.run_report(invalid_args)

        availability = payload["tile_availability"]
        self.assertEqual(payload["run"]["engine"], "native")
        self.assertEqual(availability["turn"], 9)
        self.assertEqual(availability["snapshots"], 1)
        self.assertEqual(availability["tables"], 1)
        self.assertEqual(
            tuple(availability["table_level"]["categories"]),
            report.AVAILABILITY_TABLE_CATEGORIES,
        )
        self.assertGreater(
            availability["player_relative"]["categories"]["double_wind"][
                "valid_observations"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
