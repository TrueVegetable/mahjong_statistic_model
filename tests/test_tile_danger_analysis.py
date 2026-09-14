import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
import analyze_tile_danger_by_turn as danger
import hand_efficiency_sim as sim


def fixture(trial_id, *, hit=True, safe_types=()):
    hand = list(sim.parse_hand("123456789m11p23s"))
    own = [0] * 34
    own[0], own[1], own[2], own[18], own[25] = 3, 3, 3, 3, 2
    public = [0] * 34
    for tile in range(3, 17):
        public[tile] = 1
    return {
        "trial_id": trial_id,
        "danger": {
            "riichi_seat": 0, "riichi_turn": 4, "riichi_draw_index": 13,
            "riichi_hand_counts": hand, "winning_types": [18, 21],
            "furiten_at_riichi": False, "terminal_reason": "exhaustion",
            "decisions": [{
                "observer_seat": 1, "observer_turn": 5, "riichi_player_turn": 5,
                "turns_since_riichi": 1, "draw_index": 14, "live_wall_remaining": 40,
                "own_counts": own, "public_counts": public,
                "safe_types": list(safe_types), "riichi_river_counts": [0] * 34,
                "candidates": [
                    {"tile_type": 0, "ron_hit": False, "chosen": True},
                    {"tile_type": 1, "ron_hit": False, "chosen": False},
                    {"tile_type": 2, "ron_hit": False, "chosen": False},
                    {"tile_type": 18, "ron_hit": hit, "chosen": False},
                    {"tile_type": 25, "ron_hit": False, "chosen": False},
                ],
            }],
        },
    }


class SafetyClassTests(unittest.TestCase):
    def test_middle_requires_both_suji_anchors(self):
        # 4m has 1m and 7m anchors; one is half-suji, both double suji.
        self.assertEqual(danger.safety_class(3, [0]), "half_suji")
        self.assertEqual(danger.safety_class(3, [0, 6]), "double_suji")
        self.assertEqual(danger.safety_class(3, [9]), "no_suji")

    def test_outer_suji_is_same_suit_and_honors_are_separate(self):
        self.assertEqual(danger.safety_class(0, [3]), "suji")
        self.assertEqual(danger.safety_class(0, [12]), "no_suji")
        self.assertEqual(danger.safety_class(27, []), "honor")

    def test_only_observed_safe_types_produce_genbutsu(self):
        self.assertEqual(danger.safety_class(5, [5]), "genbutsu")
        self.assertEqual(danger.safety_class(5, []), "no_suji")


class AggregateTests(unittest.TestCase):
    def test_primary_counts_distinct_tile_types_not_physical_copies(self):
        result = danger.aggregate_rows([fixture(1)])
        row = next(r for r in result["rows"] if r["axis"] == "all_turns" and r["tile_level"] == "all" and r["safety"] == "all")
        self.assertEqual(row["candidate_opportunities"], 5)
        self.assertEqual(row["ron_hits"], 1)
        self.assertEqual(row["chosen_opportunities"], 1)
        self.assertEqual(row["chosen_ron_hits"], 0)

    def test_rejects_pre_riichi_decision_and_genbutsu_hit(self):
        bad = fixture(1)
        bad["danger"]["decisions"][0]["draw_index"] = 13
        with self.assertRaises(ValueError): danger.aggregate_rows([bad])
        bad = fixture(1, safe_types=[18])
        with self.assertRaises(ValueError): danger.aggregate_rows([bad])

    def test_table_clustered_standard_error(self):
        left = fixture(1)
        left["danger"]["furiten_at_riichi"] = True
        left["danger"]["decisions"][0]["safe_types"] = [18]
        left["danger"]["decisions"][0]["riichi_river_counts"][18] = 1
        left["danger"]["decisions"][0]["public_counts"][18] = 1
        left["danger"]["decisions"][0]["public_counts"][16] = 0
        left["danger"]["decisions"][0]["candidates"][3]["ron_hit"] = False
        right = fixture(2, hit=True)
        result = danger.aggregate_rows([left, right])
        row = next(r for r in result["rows"] if r["axis"] == "all_turns" and r["tile_level"] == "tile" and r["tile"] == sim.tile_name(18) and r["safety"] == "all")
        self.assertEqual(row["candidate_opportunities"], 2)
        self.assertEqual(row["ron_hits"], 1)
        self.assertAlmostEqual(row["table_clustered_standard_error"], 0.5)

    def test_clustered_se_uses_tables_with_two_decisions_and_zero_exposure(self):
        first = fixture(1)
        second = copy.deepcopy(first["danger"]["decisions"][0])
        second["draw_index"] = 15
        second["riichi_player_turn"] = 6
        second["turns_since_riichi"] = 2
        second["public_counts"][3] += 1
        second["safe_types"] = [3]
        first["danger"]["decisions"].append(second)
        furiten = fixture(2)
        furiten["danger"]["furiten_at_riichi"] = True
        furiten["danger"]["decisions"][0]["safe_types"] = [18]
        furiten["danger"]["decisions"][0]["riichi_river_counts"][18] = 1
        furiten["danger"]["decisions"][0]["public_counts"][18] = 1
        furiten["danger"]["decisions"][0]["public_counts"][16] = 0
        furiten["danger"]["decisions"][0]["candidates"][3]["ron_hit"] = False
        empty = {"trial_id": 3, "danger": {"riichi_seat": -1, "decisions": []}}
        result = danger.aggregate_rows([first, furiten, empty])
        row = next(r for r in result["rows"] if r["axis"] == "all_turns" and r["tile_level"] == "all" and r["safety"] == "all")
        self.assertEqual((row["candidate_opportunities"], row["ron_hits"], row["table_clusters"]), (15, 2, 3))
        self.assertAlmostEqual(row["table_clustered_standard_error"], math.sqrt(4 / 3) / 15)

    def test_candidate_set_must_match_distinct_held_types(self):
        bad = fixture(1)
        bad["danger"]["decisions"][0]["candidates"].append({"tile_type": 0, "ron_hit": False, "chosen": False})
        with self.assertRaises(ValueError): danger.aggregate_rows([bad])

    def test_native_command_forwards_nondefault_policy_options(self):
        args = danger.parser().parse_args([
            "--future-ukeire-weight", "0.75",
            "--dora-tiebreak-weight", "2.0",
            "--shanten-reversal-max-turn", "12",
            "--no-shanten-reversal",
        ])
        command = danger.native_command(args, Path("/private/tmp/danger.jsonl"))
        self.assertIn("--dora-tiebreak-weight", command)
        self.assertIn("2.0", command)
        self.assertIn("--shanten-reversal-max-turn", command)
        self.assertNotIn("--allow-shanten-reversal", command)

    def test_rejects_mixed_or_mismatched_source_metadata(self):
        first, second = fixture(1), fixture(2)
        first["round_wind"], second["round_wind"] = 27, 28
        with self.assertRaisesRegex(ValueError, "inconsistent input"):
            danger.aggregate_rows([first, second])
        with self.assertRaisesRegex(ValueError, "native summary metadata"):
            danger.aggregate_rows([first], expected_metadata={
                "evaluator_backend": "wrong_backend", "config": {"round_wind": 27},
            })


if __name__ == "__main__":
    unittest.main()
