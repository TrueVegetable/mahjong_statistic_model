import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
import hand_efficiency_sim as sim  # noqa: E402


class KanchanOuterWaitTests(unittest.TestCase):
    def test_outer_kanchan_bonus_changes_only_value_tiebreak(self):
        kwargs = dict(policy="custom", future_ukeire_weight=0, improvement_weight=0,
                      tanyao_weight=0)
        base = sim.analyze_hand("134m123789p789s11z", kanchan_outer_wait_weight=0, **kwargs)
        tuned = sim.analyze_hand("134m123789p789s11z", kanchan_outer_wait_weight=1, **kwargs)
        b = {r["tile"]: r for r in base["discards"]}
        t = {r["tile"]: r for r in tuned["discards"]}
        self.assertEqual(b["4m"]["shanten"], t["4m"]["shanten"])
        self.assertEqual(b["4m"]["ukeire_count"], t["4m"]["ukeire_count"])
        self.assertEqual(t["4m"]["value_features"]["tiebreak_score"], b["4m"]["value_features"]["tiebreak_score"] + 1)

    def test_middle_and_compound_waits_receive_no_bonus(self):
        kwargs = dict(policy="custom", future_ukeire_weight=0, improvement_weight=0,
                      tanyao_weight=0)
        for hand in ("1345m123789p789s1z",):
            a = sim.analyze_hand(hand, kanchan_outer_wait_weight=0, **kwargs)
            b = sim.analyze_hand(hand, kanchan_outer_wait_weight=1, **kwargs)
            for left, right in zip(a["discards"], b["discards"]):
                self.assertEqual(left["value_features"]["tiebreak_score"], right["value_features"]["tiebreak_score"])

    def test_default_policy_parameters_omit_zero_field(self):
        p = sim.make_policy_evaluator("value").parameters()
        self.assertNotIn("kanchan_outer_wait_weight", p)


if __name__ == "__main__":
    unittest.main()
