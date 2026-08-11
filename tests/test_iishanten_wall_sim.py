import random
import sys
import unittest
import math
from collections import Counter
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs"))
import iishanten_wall_sim as sim  # noqa: E402


class LegalWallReferenceTests(unittest.TestCase):
    def test_all_enumerated_waits_are_legal(self):
        templates = sim.enumerate_wait_templates("test")
        self.assertTrue(all(template.valid() for template in templates))
        ryanmen = [template for template in templates if template.kind == "ryanmen"]
        self.assertEqual(len(ryanmen), 18)
        for template in ryanmen:
            winners = sorted(template.winners)
            self.assertEqual(winners[1] - winners[0], 3)
            self.assertEqual(sim.suit_of(winners[0]), sim.suit_of(winners[1]))

    def test_physical_state_is_exact_136_and_has_south_36_88_timing(self):
        for key in (shape.key for shape in sim.SHAPES):
            state = sim.generate_initial_state(random.Random(100), key)
            state.assert_invariants()
            self.assertEqual(sum(state.physical_counts().values()), 136)
            self.assertEqual(state.physical_counts(), Counter({tile: 4 for tile in range(34)}))
            self.assertEqual(len(state.players[sim.PLAYER_HERO].hand), 14)
            self.assertEqual([len(state.players[name].river) for name in (sim.PLAYER_DEALER, sim.PLAYER_HERO, "side1", "side2")], [9, 8, 8, 8])
            self.assertEqual(len(state.live_wall), 36)
            self.assertEqual(3 * 13 + 13 + len(state.live_wall), 88)
            # Processing the real candidate produces South's ninth river tile.
            state.players[sim.PLAYER_HERO].hand.remove(state.current_candidate)
            state.players[sim.PLAYER_HERO].river.append(state.current_candidate)
            state.assert_after_current_discard()

    def test_current_categories_are_constructed_from_dealer_river(self):
        for category in ("genbutsu", "suji37", "nosuji2378", "nosuji456"):
            state = sim.generate_initial_state(random.Random(200), "ryanmen_kanchan", current_category=category)
            self.assertTrue(sim._category_holds(state.players[sim.PLAYER_DEALER].river, state.current_candidate, category))

    def test_entry_discards_are_bound_and_present(self):
        state = sim.generate_initial_state(random.Random(300), "ryanmen_ryankan")
        for player in state.players.values():
            if player.iishanten is None:
                continue
            for plan in player.iishanten.plans:
                self.assertGreaterEqual(plan.entry_discard, 0)
                self.assertIn(plan.entry_discard, player.hand)
                self.assertNotIn(plan.entry_discard, plan.wait.winners)

    def test_furiten_and_per_riichi_genbutsu_are_separate(self):
        state = sim.generate_initial_state(random.Random(400), "ryanmen_kanchan")
        dealer = state.players[sim.PLAYER_DEALER]
        side = state.players["side1"]
        self.assertFalse(dealer.is_furiten())
        dealer.riichi = True
        side.riichi = True
        safe = next(tile for tile in range(34) if tile not in dealer.wait.winners)
        sim._safe_through(state.players[sim.PLAYER_HERO], dealer, safe)
        self.assertIn(safe, dealer.genbutsu)
        self.assertNotIn(safe, side.genbutsu)

    def test_dynamic_thresholds_retain_turn_nine_values(self):
        self.assertAlmostEqual(sim._threshold("nosuji2378", 9), 0.095)
        self.assertAlmostEqual(sim._threshold("suji37", 9), 0.055)
        self.assertNotEqual(sim._threshold("nosuji2378", 10), sim._threshold("nosuji2378", 18))

    def test_ura_score_uses_mangan_and_haneman_boundaries(self):
        self.assertEqual(sim._hero_score(tsumo=False, ippatsu=False, ura=0), 8000)
        self.assertEqual(sim._hero_score(tsumo=False, ippatsu=False, ura=1), 8000)
        self.assertEqual(sim._hero_score(tsumo=False, ippatsu=False, ura=2), 12000)
        self.assertEqual(sim._hero_score(tsumo=False, ippatsu=True, ura=1), 12000)

    def test_open_and_riichi_side_scores_are_intentionally_distinct(self):
        self.assertEqual((sim.SIDE_RON_LOSS, sim.SIDE_TSUMO_PAYMENT), (4345.0, 4767.0 / 4.0))
        self.assertEqual((sim.RIICHI_SIDE_RON_LOSS, sim.RIICHI_SIDE_TSUMO_PAYMENT), (5300.0, 1800.0))
        cpp = (ROOT / "outputs" / "iishanten_wall_sim.cpp").read_text()
        self.assertIn("constexpr double SIDE_RON_LOSS = 4345.0;", cpp)
        self.assertIn("constexpr double SIDE_TSUMO_PAYMENT = 4767.0 / 4.0;", cpp)
        self.assertIn("constexpr double RIICHI_SIDE_RON_LOSS = 5300.0;", cpp)
        self.assertIn("constexpr double RIICHI_SIDE_TSUMO_PAYMENT = 1800.0;", cpp)

    def test_joint_contribution_denominator_includes_other_wait_forms(self):
        ryanmen = sim.WaitTemplate("ryanmen", frozenset((sim.tile(0, 1), sim.tile(0, 4))), (sim.tile(0, 2), sim.tile(0, 3)), "dealer", 0)
        tanki = sim.WaitTemplate("tanki", frozenset((sim.tile(0, 9),)), (sim.tile(0, 9),), "dealer", 0)
        hits, opportunities = sim.joint_contribution_counts(((ryanmen, set()), (tanki, set())))
        # Two samples each contribute all six 1/9 candidates.  Only 1m in the
        # ryanmen sample is a hit, giving the joint 1/12, not P(hit|ryanmen)=1/6.
        self.assertEqual(opportunities["one_nine.ryanmen"], 12)
        self.assertEqual(hits["one_nine.ryanmen"], 1)
        self.assertAlmostEqual(hits["one_nine.ryanmen"] / opportunities["one_nine.ryanmen"], 1 / 12)

    def test_fixed_seed_run_and_json_shape_are_reproducible(self):
        first = sim.run("ryanmen_kanchan", 25, 991, hero_strategy="suji37", side_mode="full", side_riichi_probability=0.5)
        second = sim.run("ryanmen_kanchan", 25, 991, hero_strategy="suji37", side_mode="full", side_riichi_probability=0.5)
        self.assertEqual(first.payoffs, second.payoffs)
        self.assertEqual(sum(first.events.values()), 25)
        self.assertEqual(set(first.wait_kinds), {"ryanmen", "kanchan", "penchan", "shanpon", "tanki"})

    def test_heads_up_ablation_keeps_side_turns_but_removes_side_wins(self):
        summary = sim.run("ryanmen_kanchan", 100, 7721, hero_strategy="allout", side_mode="heads_up", side_riichi_probability=0.5)
        self.assertEqual(summary.current_side_hits, 0)
        self.assertNotIn("side_tsumo", summary.events)
        # A folded West/North discard may still be won by dealer or South, so
        # `side_ron` remains a possible non-hero-discard terminal label.

        discarder_order = []
        original = sim._resolve_discard

        def record_discarder(wall, discarder, tile, rng, *, ippatsu_dealer):
            discarder_order.append(discarder.name)
            return original(wall, discarder, tile, rng, ippatsu_dealer=ippatsu_dealer)

        with mock.patch.object(sim, "_resolve_discard", side_effect=record_discarder):
            sim.simulate_branch("ryanmen_kanchan", random.Random(9921), hero_strategy="allout", side_mode="heads_up", side_riichi_probability=0.5, current_category="genbutsu")
        self.assertEqual(discarder_order[:2], [sim.PLAYER_HERO, "side1"])

    def test_immediate_fold_cannot_later_win(self):
        summary = sim.run("ryanmen_kanchan", 100, 8821, hero_strategy="fold", side_mode="full", side_riichi_probability=0.5)
        self.assertNotIn("hero_tsumo", summary.events)
        self.assertNotIn("hero_ron", summary.events)

    def test_fold_baseline_selects_before_the_current_discard(self):
        chosen_seed = expected = candidate = None
        for seed in range(9100, 9200):
            rng = random.Random(seed)
            state = sim.generate_initial_state(rng, "ryanmen_kanchan", current_category="nosuji456")
            decision_rng = random.Random()
            decision_rng.setstate(rng.getstate())
            if decision_rng.random() < sim.FOLD_IMPERFECT_TSUMOGIRI:
                continue
            hero = state.players[sim.PLAYER_HERO]
            active = [p for p in state.players.values() if p.name != sim.PLAYER_HERO and p.riichi]
            safest = min(hero.hand, key=lambda tile: sim._joint_danger(hero, active, tile, sim.INITIAL_TURN))
            if safest != state.current_candidate:
                chosen_seed, expected, candidate = seed, safest, state.current_candidate
                break
        self.assertIsNotNone(chosen_seed)

        first_discard = []
        original = sim._resolve_discard

        def record_first(wall, discarder, tile, rng, *, ippatsu_dealer):
            if not first_discard:
                first_discard.append(tile)
            return original(wall, discarder, tile, rng, ippatsu_dealer=ippatsu_dealer)

        with mock.patch.object(sim, "_resolve_discard", side_effect=record_first):
            sim.simulate_branch("ryanmen_kanchan", random.Random(chosen_seed), hero_strategy="fold", side_mode="full", side_riichi_probability=0.5, current_category="nosuji456")
        self.assertEqual(first_discard, [expected])
        self.assertNotEqual(first_discard[0], candidate)

    def test_published_calibration_reports_independent_residuals(self):
        payload = sim.calibration_payload()
        residuals = payload["residuals"]
        basic = [cell for key, cell in residuals.items() if not key.endswith(".total")]
        rmse = math.sqrt(sum(cell["difference"] ** 2 for cell in basic) / len(basic))
        self.assertAlmostEqual(payload["weighted_rmse"], rmse)
        self.assertEqual(payload["validation_trials"], 20_000)
        self.assertEqual(payload["acceptance"]["met"], rmse <= 0.0025 and payload["acceptance"]["observed_max_total_error"] <= 0.005)
        for cell in residuals.values():
            self.assertAlmostEqual(cell["difference"], cell["fitted"] - cell["target"])
            self.assertGreaterEqual(cell["standard_error"], 0.0)
        self.assertEqual(residuals["two_eight.total"]["target"], 0.0720)
        self.assertEqual(residuals["three_seven.total"]["target"], 0.0794)


if __name__ == "__main__":
    unittest.main()
