import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
from hand_push_fold_policy import candidate_key, eligible_half_push, safety_tier


class PushFoldPolicyTests(unittest.TestCase):
    def test_safety_categories(self):
        self.assertEqual(safety_tier(0, []), 6)
        self.assertEqual(safety_tier(3, [0]), 7)
        self.assertEqual(safety_tier(27, []), 2)
        self.assertEqual(safety_tier(3, [0, 6]), 3)
        self.assertEqual(safety_tier(1, [4]), 4)
        self.assertEqual(safety_tier(2, []), 9)
        self.assertEqual(safety_tier(0, []), 6)
        self.assertEqual(safety_tier(3, []), 10)
        self.assertEqual(safety_tier(1, []), 8)
        self.assertEqual(safety_tier(2, []), 9)
        self.assertEqual(safety_tier(3, []), 10)

    def test_dora_and_candidate_keys(self):
        self.assertEqual(safety_tier(0, [0], is_dora=True), 0)
        self.assertEqual(safety_tier(1, [], is_dora=True, dora_steps=2), 10)
        self.assertLess(candidate_key([0, 10], 0, "mean_safety"), candidate_key([6, 6], 0, "mean_safety"))
        self.assertGreater(candidate_key([0, 10], 0, "minimax_safety"), candidate_key([6, 6], 0, "minimax_safety"))
        self.assertLess(candidate_key([6, 6], 2, "mean_safety"), candidate_key([6, 6], 1, "mean_safety"))

    def test_half_push_shanten_then_safety(self):
        self.assertEqual(eligible_half_push([
            {"post_shanten": 1, "tiers": [3]}, {"post_shanten": 2, "tiers": [0]},
        ]), [{"post_shanten": 1, "tiers": [3]}])
        self.assertEqual(eligible_half_push([
            {"post_shanten": 1, "tiers": [1]}, {"post_shanten": 1, "tiers": [6]},
        ]), [{"post_shanten": 1, "tiers": [1]}])


if __name__ == "__main__":
    unittest.main()
