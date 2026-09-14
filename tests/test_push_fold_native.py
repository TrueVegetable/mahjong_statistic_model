import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs"))

import hand_efficiency_sim as sim  # noqa: E402
from hand_push_fold_policy import candidate_key, safety_tier  # noqa: E402
import run_push_fold_experiments as experiments  # noqa: E402


class NativePushFoldTests(unittest.TestCase):
    """Short seeded end-to-end checks for the native score-service mode."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="push-fold-native-test-")
        cls.binary = Path(cls.tmp.name) / "hand_efficiency_sim_cpp"
        subprocess.run(
            ["c++", "-std=c++17", "-O3", str(ROOT / "outputs/hand_efficiency_sim.cpp"),
             "-o", str(cls.binary)],
            check=True,
        )
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_trials(cls, *, mode="mixed", safety="mean_safety", start=0, trials=64):
        key = (mode, safety, start, trials)
        if key in cls.runs:
            return cls.runs[key]
        observations = Path(cls.tmp.name) / f"{mode}-{safety}-{start}-{trials}.jsonl"
        args = experiments.parser().parse_args([
            "--native-binary", str(cls.binary), "--trials", str(trials),
            "--start-trial", str(start), "--seed", "17", "--round-wind", "east",
            "--mode", mode, "--safety-strategy", safety,
            "--future-ukeire-weight", "0", "--future-ukeire-weight-1-shanten", "0",
            "--future-ukeire-weight-2-shanten", "0", "--improvement-weight", "0",
        ])
        run = experiments.run_native(args, observations)
        rows = [json.loads(line) for line in observations.read_text().splitlines() if line]
        cls.runs[key] = (run, rows)
        return cls.runs[key]

    @staticmethod
    def _best_post_shanten(own):
        best = 99
        for tile, count in enumerate(own):
            if count:
                post = list(own)
                post[tile] -= 1
                best = min(best, sim.shanten(tuple(post)))
        return best

    @staticmethod
    def _candidate_key(candidate, strategy):
        return candidate_key(
            candidate["safety_tiers"], int(candidate["known_copies"]), strategy
        )

    def assert_decision_invariants(self, row):
        state = row["push_fold"]
        dora_type = sim.dora_type_from_indicator(int(row["dora_indicator"]) // 4)
        strategy = state["safety_strategy"]
        for decision in state["decisions"]:
            own = tuple(map(int, decision["own_counts"]))
            public = tuple(map(int, decision["public_counts"]))
            self.assertEqual(sum(own), 14)
            self.assertEqual(sum(public), int(decision["draw_index"]))
            self.assertTrue(all(a + b <= 4 for a, b in zip(own, public)))
            candidates = decision["candidates"]
            self.assertEqual({int(c["tile_type"]) for c in candidates},
                             {tile for tile, count in enumerate(own) if count})
            self.assertEqual(sum(bool(c["chosen"]) for c in candidates), 1)
            self.assertEqual(int(decision["best_shanten"]), self._best_post_shanten(own))
            chosen = next(c for c in candidates if c["chosen"])
            chosen_post = list(own)
            chosen_post[int(chosen["tile_type"])] -= 1
            self.assertEqual(int(decision["post_discard_shanten"]), sim.shanten(tuple(chosen_post)))

            safe = {int(item["threat_seat"]): set(map(int, item["types"]))
                    for item in decision["safe_types"]}
            self.assertEqual(set(safe), set(map(int, decision["threat_seats"])))
            for candidate in candidates:
                tile = int(candidate["tile_type"])
                physical = int(candidate["physical_id"])
                self.assertEqual(int(candidate["known_copies"]), own[tile] + public[tile])
                expected = [safety_tier(
                    tile, safe[threat],
                    is_dora=(physical in sim.RED_FIVE_IDS or tile == dora_type),
                    dora_steps=int(state["dora_danger_steps"]),
                ) for threat in decision["threat_seats"]]
                self.assertEqual(list(map(int, candidate["safety_tiers"])), expected)
                for threat in candidate["ron_hits"]:
                    if int(threat) in safe:
                        self.assertNotIn(tile, safe[int(threat)])

            if decision["mode"] == "full_fold":
                self.assertEqual(
                    self._candidate_key(chosen, strategy),
                    min(self._candidate_key(c, strategy) for c in candidates),
                )
            if decision["mode"] == "half_push":
                self.assertEqual(int(decision["post_discard_shanten"]), int(decision["best_shanten"]))
                self.assertLessEqual(max(map(int, chosen["safety_tiers"]), default=0),
                                     int(state["half_push_max_tier"]))
            if decision["mode_reason"] == "riichi_lock":
                self.assertTrue(decision["own_riichi"])
                self.assertEqual(int(chosen["tile_type"]), int(decision["drawn_physical"]) // 4)
                self.assertEqual(
                    int(chosen["physical_id"]) in sim.RED_FIVE_IDS,
                    int(decision["drawn_physical"]) in sim.RED_FIVE_IDS,
                )
                self.assertEqual(int(decision["post_discard_shanten"]), 0)

    def test_score_service_modes_and_physical_decision_invariants(self):
        # Exercise all policy modes and both multi-threat selectors.  The
        # short runs retain stable, independent physical-table checks.
        scenarios = (
            ("all_push", "mean_safety"),
            ("full_fold", "mean_safety"),
            ("mixed", "mean_safety"),
            ("mixed", "minimax_safety"),
        )
        for mode, strategy in scenarios:
            run, rows = self.run_trials(mode=mode, safety=strategy)
            self.assertEqual(len(rows), 64)
            self.assertGreaterEqual(run["score_requests"], 0)
            for row in rows:
                self.assertIn("push_fold", row)
                self.assert_decision_invariants(row)
                terminal = row["push_fold"]["terminal"]
                if terminal["reason"] in ("ron", "multiple_ron"):
                    self.assertTrue(row["push_fold"]["decisions"])
                    chosen = next(c for c in row["push_fold"]["decisions"][-1]["candidates"] if c["chosen"])
                    self.assertTrue(set(map(int, chosen["ron_hits"])) &
                                    {int(w["seat"]) for w in terminal["winners"]})

    def test_draw_improvement_enters_tenpai_branch_when_present(self):
        _, rows = self.run_trials(mode="mixed", safety="mean_safety")
        observed = 0
        for row in rows:
            prior_post = {}
            for decision in row["push_fold"]["decisions"]:
                seat = int(decision["seat"])
                if prior_post.get(seat) == 1 and int(decision["best_shanten"]) == 0:
                    observed += 1
                    if not decision["own_riichi"] and decision["mode_reason"] != "sticky_fold":
                        self.assertNotEqual(decision["mode_reason"], "one_threat_one_shanten")
                prior_post[seat] = int(decision["post_discard_shanten"])
        self.assertGreater(observed, 0)

    def test_trial_shards_are_identical(self):
        _, whole = self.run_trials(mode="mixed", safety="mean_safety", start=0, trials=16)
        _, left = self.run_trials(mode="mixed", safety="mean_safety", start=0, trials=8)
        _, right = self.run_trials(mode="mixed", safety="mean_safety", start=8, trials=8)
        by_id = {int(row["trial_id"]): row for row in left + right}
        self.assertEqual([int(row["trial_id"]) for row in whole], sorted(by_id))
        for row in whole:
            self.assertEqual(row, by_id[int(row["trial_id"])])

    def test_runner_aggregates_actual_rows_for_every_mode(self):
        for mode in ("all_push", "full_fold", "mixed"):
            _, rows = self.run_trials(mode=mode, safety="mean_safety", trials=20)
            metrics = Path(self.tmp.name) / f"{mode}-metrics.jsonl"
            aggregate = experiments.aggregate_rows(rows, metrics)
            self.assertEqual(aggregate["tables"], 20)
            endings = sum(int(aggregate["diagnostics"].get(reason, 0)) for reason in (
                "ron", "multiple_ron", "tsumo", "exhaustive_draw",
            ))
            self.assertEqual(endings, 20)
            declarations = sum(len(row["push_fold"]["riichi_seats"]) for row in rows)
            self.assertEqual(aggregate["diagnostics"]["riichi_declarations"], declarations)
            self.assertEqual(sum(aggregate["riichi_wait_distribution"].values()), declarations)
            metric_rows = [json.loads(line) for line in metrics.read_text().splitlines() if line]
            self.assertEqual(len(metric_rows), 20)
            self.assertTrue(all(len(row["net_points"]) == 4 for row in metric_rows))

    def test_native_kanchan_outer_tiebreak_matches_python_fixture(self):
        hand = "13m123789p789s11z"
        native_values = []
        python_values = []
        for weight in (0, 1):
            native = subprocess.run(
                [str(self.binary), "analyze", hand, "--policy", "value",
                 "--kanchan-outer-wait-weight", str(weight)],
                check=True, capture_output=True, text=True,
            )
            native_values.append(json.loads(native.stdout)["value_features"]["tiebreak_score"])
            python_values.append(sim.analyze_hand(
                hand, policy="value", future_ukeire_weight=0, improvement_weight=0,
                kanchan_outer_wait_weight=weight,
            )["value_features"]["tiebreak_score"])
        self.assertAlmostEqual(native_values[1] - native_values[0], 1.0)
        self.assertAlmostEqual(python_values[1] - python_values[0], 1.0)


if __name__ == "__main__":
    unittest.main()
