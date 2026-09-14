import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
import hand_efficiency_sim as sim
import run_push_fold_experiments as experiment


def terminal_fixture(tsumo=False, declared=(1,)):
    hand = list(sim.parse_physical_hand("123456789m11p234s"))
    return {
        "trial_id": 0, "round_wind": 27, "dora_indicator": 132, "ura_dora_indicator": 133,
        "push_fold": {"riichi_seats": list(declared), "decisions": [],
                      "terminal": {"reason": "tsumo" if tsumo else "ron",
                                   "loser": -1 if tsumo else 0,
                                   "riichi_sticks": len(declared), "tenpai_seats": [],
                                   "winners": [{"seat": 1, "hand_physical": hand,
                                                "winning_tile": hand[-1], "is_riichi": 1 in declared,
                                                "is_ippatsu": False, "is_haitei": False}]}},
    }


class SettlementTests(unittest.TestCase):
    def test_candidate_and_chosen_uncertainty_cluster_by_table(self):
        # Two exposed tables, each with two candidates; one has a hit and one
        # has none. A third table has no exposure but still belongs in G.
        values = [4, 1, 2, 1, 2, 2, 1, 2, 8, 1, 1, 2]
        cell = experiment.clustered_rate(values, 3)
        self.assertEqual(cell["ron_hit_probability"], .25)
        self.assertAlmostEqual(cell["table_clustered_se"], (0.75 ** .5) / 4)
        self.assertEqual(cell["chosen_ron_hit_probability"], .5)
        self.assertAlmostEqual(cell["chosen_table_clustered_se"], (0.75 ** .5) / 2)
        self.assertEqual(cell["independent_tables"], 3)
        self.assertEqual(cell["exposed_tables"], 2)

    def test_ron_and_tsumo_use_actual_payment_breakdowns(self):
        ron = experiment.settle_table(terminal_fixture())
        tsumo = experiment.settle_table(terminal_fixture(True))
        self.assertEqual(sum(ron["net_points"]), 0)
        self.assertEqual(sum(tsumo["net_points"]), 0)
        self.assertEqual(ron["net_points"][2:], [0, 0])
        self.assertLess(tsumo["net_points"][0], tsumo["net_points"][2])
        self.assertEqual(tsumo["net_points"][2], tsumo["net_points"][3])

    def test_exhaustion_transfers_and_carried_pot(self):
        row = terminal_fixture(declared=(1, 2))
        row["push_fold"]["terminal"].update(reason="exhausted", winners=[], loser=-1, tenpai_seats=[1, 2])
        result = experiment.settle_table(row)
        self.assertEqual(result["net_points"], [-1500, 500, 500, -1500])
        self.assertEqual(result["carried_riichi_pot"], 2000)

    def test_multi_ron_sum_and_pot_awarded_once(self):
        row = terminal_fixture(declared=(1, 2))
        second = copy.deepcopy(row["push_fold"]["terminal"]["winners"][0])
        second["seat"] = 2
        row["push_fold"]["terminal"]["winners"].append(second)
        result = experiment.settle_table(row)
        self.assertEqual(sum(result["net_points"]), 0)
        payments = {x["seat"]: x["points"] for x in result["payments"]}
        self.assertEqual(result["net_points"][0], -sum(payments.values()))
        self.assertEqual(result["net_points"][1], payments[1] + 1000)
        self.assertEqual(result["net_points"][2], payments[2] - 1000)


if __name__ == "__main__":
    unittest.main()
