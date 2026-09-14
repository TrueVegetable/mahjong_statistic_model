import json
import subprocess
import tempfile
import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs"))
import hand_efficiency_sim as sim  # noqa: E402


class NativeTileDangerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="tile-danger-test-")
        cls.binary = Path(cls.tmp.name) / "hand_efficiency_sim_cpp"
        subprocess.run(
            ["c++", "-std=c++17", "-O3", str(ROOT / "outputs/hand_efficiency_sim.cpp"), "-o", str(cls.binary)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_trials(self, start, count):
        completed = subprocess.run([
            str(self.binary), "simulate-table", "--trials", str(count),
            "--start-trial", str(start), "--seed", "17", "--tile-danger",
            "--future-ukeire-weight", "0", "--future-ukeire-weight-1-shanten", "0",
            "--future-ukeire-weight-2-shanten", "0", "--improvement-weight", "0",
            "--observations", str(Path(self.tmp.name) / f"{start}-{count}.jsonl"),
        ], check=True, capture_output=True, text=True)
        del completed
        return [json.loads(line) for line in (Path(self.tmp.name) / f"{start}-{count}.jsonl").read_text().splitlines()]

    def test_schema_and_terminal_invariants(self):
        rows = self.run_trials(0, 128)
        self.assertTrue(rows)
        for row in rows:
            danger = row.get("danger")
            if not danger:
                continue
            decisions = danger["decisions"]
            if danger["riichi_seat"] < 0:
                self.assertEqual(decisions, [])
                self.assertIn(danger["terminal_reason"], ("no_riichi_exposure", "pre_riichi_tsumo"))
                continue
            target_waits = set(sim.structural_ukeire_types(tuple(danger["riichi_hand_counts"])))
            target_river_types: set[int] | None = None
            expected_safe: set[int] | None = None
            previous_public: list[int] | None = None
            for decision in decisions:
                own = decision["own_counts"]
                public = decision["public_counts"]
                self.assertEqual(sum(own), 14)
                self.assertEqual(sum(public), decision["draw_index"])
                self.assertTrue(all(a + b <= 4 for a, b in zip(own, public)))
                candidates = decision["candidates"]
                self.assertEqual([c["tile_type"] for c in candidates], sorted(set(c["tile_type"] for c in candidates)))
                self.assertEqual(sum(c["chosen"] for c in candidates), 1)
                self.assertNotEqual(decision["observer_seat"], danger["riichi_seat"])
                self.assertGreater(decision["draw_index"], danger["riichi_draw_index"])
                self.assertEqual(
                    decision["turns_since_riichi"],
                    decision["riichi_player_turn"] - danger["riichi_turn"],
                )

                river_types = {
                    tile_type
                    for tile_type, count in enumerate(decision["riichi_river_counts"])
                    if count
                }
                if target_river_types is None:
                    target_river_types = river_types
                    expected_safe = set(target_river_types)
                else:
                    assert expected_safe is not None
                    assert previous_public is not None
                    expected_safe.update(
                        tile_type
                        for tile_type, (before, after) in enumerate(
                            zip(previous_public, public, strict=True)
                        )
                        if after > before
                    )
                self.assertEqual(set(decision["safe_types"]), expected_safe)

                furiten = bool(target_waits & river_types)
                self.assertEqual(
                    {candidate["tile_type"] for candidate in candidates if candidate["ron_hit"]},
                    set() if furiten else target_waits & {candidate["tile_type"] for candidate in candidates},
                )
                previous_public = public
            if danger["terminal_reason"] == "target_ron":
                self.assertTrue(decisions)
                chosen = next(c for c in decisions[-1]["candidates"] if c["chosen"])
                self.assertTrue(chosen["ron_hit"])
            if danger["terminal_reason"] == "second_riichi":
                self.assertTrue(decisions)
                last = decisions[-1]
                chosen = next(c for c in last["candidates"] if c["chosen"])
                counts = list(last["own_counts"])
                counts[chosen["tile_type"]] -= 1
                self.assertEqual(sim.shanten(tuple(counts)), 0)
                self.assertGreaterEqual(last["live_wall_remaining"], 4)
            if danger["terminal_reason"] == "target_tsumo":
                self.assertTrue(decisions)

    def test_trial_shards_are_byte_identical(self):
        whole = self.run_trials(0, 64)
        left = self.run_trials(0, 32)
        right = self.run_trials(32, 32)
        by_id = {r["trial_id"]: r for r in left + right}
        self.assertEqual([r["trial_id"] for r in whole], sorted(by_id))
        for row in whole:
            self.assertEqual(row, by_id[row["trial_id"]])

    def test_ordinary_mode_omits_danger_and_incompatible_flags_fail(self):
        ordinary = subprocess.run(
            [str(self.binary), "simulate-table", "--trials", "8", "--seed", "17"],
            check=True, capture_output=True, text=True,
        )
        self.assertTrue(all("danger" not in json.loads(line) for line in ordinary.stdout.splitlines()))
        bad = subprocess.run(
            [str(self.binary), "simulate-table", "--trials", "1", "--tile-danger",
             "--tile-availability-turn", "6"], capture_output=True, text=True,
        )
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
