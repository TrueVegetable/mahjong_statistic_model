import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs"))

import hand_efficiency_sim as sim  # noqa: E402
import score_first_tenpai_values as values  # noqa: E402


class FakeScorer:
    name = "fake"

    def __init__(self):
        self.contexts = []

    def score(self, hand, win_tile, context):
        self.contexts.append(context)
        self.assert_hand(hand, win_tile)
        # 1 han 30 fu and 1 han 50 fu normalize to 1000 and 1600.
        return sim.HandScore(9999, 1, 30 if win_tile % 4 == 1 else 50, (), {})

    @staticmethod
    def assert_hand(hand, win_tile):
        if len(hand) != 14 or win_tile not in hand:
            raise AssertionError("post-scorer did not complete the 13-tile hand")


def row(trial=0, *, player_wind=27):
    hand = list(sim.parse_physical_hand("123456789m11p23s"))
    return {
        "trial_id": trial,
        "trial_seed": sim.trial_seed(20260809, trial),
        "engine_version": "test", "evaluator_backend": "test", "policy": "value",
        "policy_parameters": {"shanten_reversal": {"enabled": True}},
        "round_wind": 27, "player_wind": player_wind,
        "dora_indicator": 0, "ura_dora_indicator": 4,
        "first_tenpai": {
            "hand_physical": hand,
            "waits": [
                {"tile_type": 18, "visible_remaining": 2, "visible_remaining_tile_ids": [73, 74],
                 "live_wall_remaining": 1, "live_wall_remaining_tile_ids": [73]},
                {"tile_type": 21, "visible_remaining": 0, "visible_remaining_tile_ids": [],
                 "live_wall_remaining": 0, "live_wall_remaining_tile_ids": []},
            ],
        },
    }


class FirstTenpaiValueTests(unittest.TestCase):
    def test_visible_copy_weighting_and_dealer_wind_normalization(self):
        fake = FakeScorer()
        result = values.score_rows([row()], 20260809, fake)
        trial = result["trials"][0]
        self.assertEqual(trial["mean_ron_points"], 1300)
        self.assertEqual(trial["mean_live_ron_points"], 1000)
        self.assertEqual(trial["waitcategory"], "ryanmen")
        self.assertEqual(result["visible_copy_weighted"]["mean_ron_points"], 1300)
        self.assertTrue(fake.contexts)
        self.assertTrue(all(context.player_wind == 27 for context in fake.contexts))
        self.assertTrue(all(not context.is_dealer for context in fake.contexts))
        self.assertTrue(all(context.is_riichi and not context.is_tsumo for context in fake.contexts))
        self.assertTrue(all(context.ura_dora_indicators == (4,) for context in fake.contexts))

    def test_real_scorer_accepts_east_seat_and_returns_nondealer_value(self):
        result = values.score_rows([row()], 20260809)
        self.assertGreater(result["trials"][0]["mean_ron_points"], 0)
        self.assertEqual(result["diagnostics"]["first_tenpai_hands"], 1)

    def test_metadata_and_seed_validation_fail_fast(self):
        bad = row()
        bad["policy_parameters"] = {"shanten_reversal": {"enabled": False}}
        with self.assertRaises(ValueError):
            values.score_rows([bad], 20260809, FakeScorer())
        bad = row()
        bad["trial_seed"] = 0
        with self.assertRaises(ValueError):
            values.score_rows([bad], 20260809, FakeScorer())

    def test_cli_writes_durable_result_and_trial_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, metrics = root / "raw.jsonl", root / "values.json", root / "trials.jsonl"
            source.write_text(json.dumps(row()) + "\n")
            self.assertEqual(values.main([
                "--input", str(source), "--output", str(output), "--trial-metrics", str(metrics),
            ]), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["inputs"]["tables"], 1)
            self.assertTrue(result["inputs"]["source_sha256"])
            self.assertEqual(len(metrics.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
