from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "outputs" / "cross_validate_iishanten.py"
SPEC = importlib.util.spec_from_file_location("cross_validate_iishanten", MODULE_PATH)
assert SPEC and SPEC.loader
cv = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cv
SPEC.loader.exec_module(cv)


def payload(engine: str, *, ev: float = -100.0, event_bump: float = 0.0) -> dict:
    residuals = {
        cell: {"target": target, "fitted": target, "difference": 0.0, "standard_error": 0.001}
        for cell, target in cv.KOBA_TARGETS.items()
    }
    trials = 10_000
    event_rates = {
        "hero_win": 0.20 + event_bump,
        "side_terminal": 0.15,
        "draw": 0.10,
        "dealer_ippatsu_tsumo": 0.03,
    }
    event_counts = {"hero_ron": 2_000, "side_ron": 7_000, "draw": 1_000}
    result = {
        "shape": "ryanmen_ryanmen",
        "strategy": "allout",
        "current_category": "suji37",
        "ev": ev,
        "se": 10.0,
        "fold_ev": -200.0,
        "fold_se": 5.0,
        "ev_difference": ev + 200.0,
        "paired_ev_difference_se": 8.0,
        "break_even_danger": 0.12,
        "event_counts": event_counts,
        "event_rates": event_rates,
        "route_counts": {"continued": 8_000, "none": 2_000},
        "wait_kind_counts": {"ryanmen": 4_000, "kanchan": 2_000, "penchan": 500, "shanpon": 1_500, "tanki": 2_000},
        "current_hit_rates": {"dealer": 0.04, "initial_side": 0.03, "any": 0.06},
        "current_dealer_ippatsu_contribution": -436.0,
        "continuation_ev_after_safe": -90.0,
    }
    return {
        "engine_version": "test-1",
        "engine": engine,
        "seed": 20260809,
        "trials": trials,
        "config": {
            "seat_map": {"dealer": "east", "hero": "south", "side1": "west", "side2": "north"},
            "river_lengths": {"dealer": 9, "hero": 9, "side1": 8, "side2": 8},
            "live_wall_tiles": 36,
            "hero_unseen_positions": 88,
            "action_order": ["hero_current_discard_check", "west_draw_discard", "north_draw_discard", "dealer_draw"],
        },
        "results": [result],
        "calibration": {"residuals": residuals, "weighted_rmse": 0.0},
    }


class CrossValidateTests(unittest.TestCase):
    def test_valid_payload_passes_contract(self) -> None:
        self.assertEqual(cv.validate_document(payload("python")), [])

    def test_break_even_danger_may_be_outside_probability_range(self) -> None:
        document = payload("python")
        document["results"][0]["break_even_danger"] = -0.15
        self.assertEqual(cv.validate_document(document), [])

    def test_entity_timepoint_and_ev_difference_are_enforced(self) -> None:
        document = payload("python")
        document["config"]["live_wall_tiles"] = 34
        document["config"]["hero_unseen_positions"] = 86
        document["config"]["river_lengths"]["side1"] = 9
        document["config"]["action_order"] = ["hero_current_discard_check", "dealer_draw", "west_draw_discard", "north_draw_discard"]
        document["results"][0]["ev_difference"] = 99.0
        errors = cv.validate_document(document)
        self.assertTrue(any("live_wall_tiles" in error for error in errors))
        self.assertTrue(any("hero_unseen_positions" in error for error in errors))
        self.assertTrue(any("river_lengths" in error for error in errors))
        self.assertTrue(any("action_order" in error for error in errors))
        self.assertTrue(any("ev_difference" in error for error in errors))

    def test_equal_river_compatibility_field_is_rejected(self) -> None:
        document = payload("python")
        document["config"]["discard_count_each"] = 9
        errors = cv.validate_document(document)
        self.assertTrue(any("discard_count_each is obsolete" in error for error in errors))

    def test_koba_targets_and_reported_rmse_are_enforced_without_forcing_fit(self) -> None:
        document = payload("python")
        item = document["calibration"]["residuals"]["honor.tanki"]
        item["fitted"] = 0.0256
        item["difference"] = 0.0200
        document["calibration"]["weighted_rmse"] = 0.0
        errors = cv.validate_document(document)
        self.assertTrue(any("weighted_rmse" in error for error in errors))
        document["calibration"]["weighted_rmse"] = 0.0200 / len(cv.BASE_KOBA_CELLS) ** 0.5
        self.assertEqual(cv.validate_document(document), [])

    def test_cross_engine_uses_combined_three_sigma(self) -> None:
        comparisons = cv.compare_documents(payload("python", ev=-100.0), payload("cpp", ev=-125.0))
        ev_comparison = next(item for item in comparisons if item.metric == "ev")
        self.assertTrue(ev_comparison.passed)
        far = cv.compare_documents(payload("python", ev=-100.0), payload("cpp", ev=-200.0))
        self.assertFalse(next(item for item in far if item.metric == "ev").passed)

    def test_markdown_and_csv_summary(self) -> None:
        rows = cv._rows_from_document(payload("python"), "fixture.json")
        self.assertIn("| 引擎 |", cv.markdown_summary(rows))
        self.assertIn("current_hit_any", cv.csv_summary(rows))

    def test_cli_validate_and_summarize_accept_formal_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "result.json"
            source.write_text(json.dumps(payload("python")), encoding="utf-8")
            self.assertEqual(cv.main(["validate", str(source)]), 0)
            self.assertEqual(cv.main(["summarize", str(source), "--format", "csv"]), 0)

    def test_scanner_distinguishes_exact_south_state_from_math_approximation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            outputs.mkdir()
            article = outputs / "statistical-mahjong-the-whys-cn.md"
            article.write_text(
                "数学层使用约 36 张活牌山和约 88 个未见位置。\n"
                "实体模拟按南家精确时点计算，活牌山 36 张，我方未见 88 个位置。\n"
                "庄家河牌 9 张，南家河牌 9 张，西家河牌 8 张，北家河牌 8 张。\n"
                "当前弃牌类别只由庄家自己的 9 张牌河构造。\n"
                "当前弃牌后，西家摸打、北家摸打、庄家下一摸。\n"
                "koba 表给出候选牌成为胜张的贡献，不是每手等待形式比例。\n",
                encoding="utf-8",
            )
            (outputs / "iishanten_wall_sim.py").write_text("LIVE_WALL_TILES = 36\nUNKNOWN_POSITIONS = 88\n", encoding="utf-8")
            findings = cv.scan_repository(Path(directory))
            timing_rules = {finding.rule for finding in findings if "koba" not in finding.rule}
            self.assertEqual(timing_rules, set())
            with article.open("a", encoding="utf-8") as handle:
                handle.write("旧模拟曾写活牌山 34 张，我方未见 86 个位置，四家各九张牌河。\n")
            stale_rules = {finding.rule for finding in cv.scan_repository(Path(directory))}
            self.assertIn("stale-34-live-wall", stale_rules)
            self.assertIn("stale-86-unseen-positions", stale_rules)
            self.assertIn("stale-four-equal-rivers", stale_rules)


if __name__ == "__main__":
    unittest.main()
