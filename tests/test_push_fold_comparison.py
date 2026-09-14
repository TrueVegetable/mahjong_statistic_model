import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
import summarize_push_fold_comparison as comparison  # noqa: E402


def metric(trial, net, reason, riichi=()):
    return {
        "trial_id": trial, "terminal_reason": reason, "net_points": net,
        "riichi_seats": list(riichi), "payments": [], "winner_seats": [], "loser": -1,
    }


def aggregate():
    danger_rows = []
    for threats in (1, 2, 3):
        danger_rows.append({
            "riichi_opponents": threats, "observer_turn": "all",
            "ron_scope": "riichi_opponents", "tile_level": "all", "tile": "all",
            "safety_profile": "all", "candidate_opportunities": 100,
            "ron_hits": threats, "ron_hit_probability": threats / 100,
            "table_clustered_se": 0.01 * threats, "chosen_opportunities": 20,
            "chosen_ron_hits": threats, "chosen_ron_hit_probability": threats / 20,
            "chosen_table_clustered_se": 0.02 * threats,
        })
    return {
        "tables": 3,
        "diagnostics": {"ron": 1, "multiple_ron": 1, "tsumo": 0,
                        "exhaustive_draw": 1, "riichi_declarations": 4},
        "realized_win_values": {"ron": {"wins": 2, "mean_points": 7700},
                                  "tsumo": {"wins": 0, "mean_points": None}},
        "seat_mean_net_points": [100, -50, -25, -25],
        "seat_net_points_se": [1, 2, 3, 4],
        "seat_outcomes": [{"ron": 1}, {}, {}, {}],
        "riichi_wait_distribution": {"ryanmen": 2},
        "riichi_wait_quality": {"good": 2},
        "danger_rows": danger_rows,
    }


class PushFoldComparisonTests(unittest.TestCase):
    def test_summary_uses_actual_aggregate_and_metrics_schema(self):
        rows = [
            metric(0, [100, -100, 0, 0], "ron", [0]),
            metric(1, [200, -200, 0, 0], "multiple_ron", [1, 2]),
            metric(2, [0, 0, 0, 0], "exhaustive_draw", [3]),
        ]
        summary = comparison.summarize_policy("mixed", aggregate(), rows)
        self.assertEqual(summary["ending_counts"]["multiple_ron"], 1)
        self.assertEqual(summary["ron_ending_rate"], 2 / 3)
        self.assertEqual(summary["riichi_declarations_per_table"], 4 / 3)
        self.assertEqual(summary["mean_ron_points"], 7700)
        cell = summary["danger_by_riichi_opponents"]["2"]
        self.assertEqual(cell["candidate_ron_hit_probability"], 0.02)
        self.assertEqual(cell["chosen_table_clustered_se"], 0.04)

    def test_pairing_is_by_exact_unique_trial_id_and_per_seat(self):
        left = [
            metric(2, [0, 20, 0, -20], "ron", [0]),
            metric(1, [10, 0, -10, 0], "exhaustive_draw", []),
        ]
        right = [
            metric(1, [30, 0, -30, 0], "tsumo", [1]),
            metric(2, [20, 10, 0, -30], "multiple_ron", [0, 2]),
        ]
        paired = comparison.paired_trial_differences(left, right)
        self.assertEqual(paired["paired_tables"], 2)
        self.assertEqual(paired["seat_net_points"][0]["mean"], 20)
        self.assertEqual(paired["seat_net_points"][1]["mean"], -5)
        self.assertEqual(paired["outcome_indicator_differences"]["ron"]["mean"], 0)
        self.assertEqual(paired["outcome_indicator_differences"]["tsumo"]["mean"], 0.5)
        self.assertEqual(paired["riichi_declarations"]["mean"], 1)
        self.assertGreater(paired["seat_net_points"][1]["standard_error"], 0)

    def test_duplicate_or_mismatched_trials_fail(self):
        with self.assertRaises(ValueError):
            comparison.paired_trial_differences(
                [metric(1, [0, 0, 0, 0], "ron"), metric(1, [0, 0, 0, 0], "ron")],
                [metric(1, [0, 0, 0, 0], "ron")],
            )
        with self.assertRaises(ValueError):
            comparison.paired_trial_differences(
                [metric(1, [0, 0, 0, 0], "ron")],
                [metric(2, [0, 0, 0, 0], "ron")],
            )

    def test_cli_writes_markdown_and_optional_json(self):
        rows = [
            metric(0, [0, 0, 0, 0], "ron"),
            metric(1, [0, 0, 0, 0], "multiple_ron"),
            metric(2, [0, 0, 0, 0], "exhaustive_draw"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate_path = root / "aggregate.json"
            metrics_path = root / "metrics.jsonl"
            markdown_path = root / "comparison.md"
            json_path = root / "comparison.json"
            aggregate_path.write_text(json.dumps(aggregate()))
            metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(comparison.main([
                "--input", f"mixed={aggregate_path}", "--metrics", f"mixed={metrics_path}",
                "--output", str(markdown_path), "--json", str(json_path),
            ]), 0)
            text = markdown_path.read_text()
            self.assertIn("All four seats use", text)
            self.assertIn("multiple ron", text)
            self.assertEqual(json.loads(json_path.read_text())["policies"][0]["name"], "mixed")


if __name__ == "__main__":
    unittest.main()
