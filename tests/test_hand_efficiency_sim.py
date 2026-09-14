import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs"))
import hand_efficiency_sim as sim  # noqa: E402


class HandAnalysisTests(unittest.TestCase):
    @staticmethod
    def _dummy_features(discard_type, *, shanten=1, score=10.0):
        return sim.CandidateFeatures(
            discard_type=discard_type,
            shanten=shanten,
            shanten_forms=(),
            ukeire_types=(),
            ukeire_remaining=(),
            ukeire_count=int(score),
            future_ukeire_by_type=(),
            future_ukeire_expected=0.0,
            improvement_types=(),
            improvement_remaining=(),
            improvement_count=0,
            ukeire_score=score,
            ukeire_computed=True,
            base_ukeire_score=score,
        )

    def test_parse_format_and_shanten_forms(self):
        hand = sim.parse_hand("123456789m11p23s")
        self.assertEqual(sim.format_counts(hand), "123456789m11p23s")
        breakdown = sim.shanten_breakdown(hand)
        self.assertEqual(breakdown.minimum, 0)
        self.assertIn("standard", breakdown.best_forms)

        chiitoi = sim.parse_hand("1122m3344p5566s7z")
        self.assertEqual(sim.shanten(chiitoi), 0)
        self.assertIn("chiitoitsu", sim.shanten_breakdown(chiitoi).best_forms)

        kokushi = sim.parse_hand("19m19p19s1234567z")
        self.assertEqual(sim.shanten(kokushi), 0)
        self.assertIn("kokushi", sim.shanten_breakdown(kokushi).best_forms)

    def test_fast_and_detailed_shanten_backends_agree_on_random_hands(self):
        rng = random.Random(441)
        wall = list(range(136))
        for _ in range(100):
            rng.shuffle(wall)
            counts = sim.counts_from_physical(wall[:13])
            self.assertEqual(sim.shanten(counts), sim.shanten_breakdown(counts).minimum)

    def test_ryanmen_ukeire_and_visibility(self):
        hand = sim.parse_hand("123456789m11p23s")
        self.assertEqual(tuple(sim.tile_name(tile) for tile in sim.structural_ukeire_types(hand)), ("1s", "4s"))
        self.assertEqual(sim.wait_shapes(hand, 18), ("ryanmen",))
        self.assertEqual(sim.wait_shapes(hand, 21), ("ryanmen",))

        payload = sim.analyze_hand("123456789m11p23s", visible_text="11s")
        self.assertEqual(payload["ukeire_count"], 6)

    def test_basic_wait_shape_labels(self):
        cases = {
            "123456789m11p24s": (20, ("kanchan",)),
            "123456789m11p12s": (20, ("penchan",)),
            "123456789m11p22s": (9, ("shanpon",)),
            "123456789m123p1s": (18, ("tanki",)),
        }
        for hand_text, (winning_type, expected) in cases.items():
            with self.subTest(hand=hand_text):
                self.assertEqual(sim.shanten(sim.parse_hand(hand_text)), 0)
                self.assertEqual(sim.wait_shapes(sim.parse_hand(hand_text), winning_type), expected)

    def test_special_wait_shape_labels(self):
        chiitoi = sim.parse_hand("1122m3344p5566s7z")
        self.assertEqual(sim.wait_shapes(chiitoi, 33), ("chiitoitsu_tanki",))
        kokushi = sim.parse_hand("19m19p19s1234567z")
        self.assertEqual(sim.wait_shapes(kokushi, 0), ("kokushi_13way",))

    def test_tenpai_snapshot_marks_permanent_furiten(self):
        counts = sim.parse_hand("123456789m11p23s")
        hand = tuple(tile * 4 + copy for tile, count in enumerate(counts) for copy in range(count))
        discarded_1s = 18 * 4
        snapshot = sim._make_tenpai_snapshot(
            hand,
            (discarded_1s,),
            (discarded_1s, 0),
            (),
            turn=5,
            riichi_declared=True,
        )
        self.assertTrue(snapshot.furiten)

    def test_discard_policy_is_shanten_then_ukeire(self):
        payload = sim.analyze_hand("123456789m9p11p23s")
        best = [row for row in payload["discards"] if row["best"]]
        self.assertEqual([row["tile"] for row in best], ["9p"])
        self.assertEqual(best[0]["shanten"], 0)
        self.assertEqual(best[0]["ukeire_count"], 8)

    def test_advanced_ukeire_score_has_one_step_future_and_improvements(self):
        payload = sim.analyze_hand("2378m234789p1s36z")
        self.assertEqual(payload["shanten"]["minimum"], 2)
        self.assertEqual(payload["ukeire_count"], 25)
        self.assertAlmostEqual(payload["future_ukeire_expected"], 14.72)
        self.assertEqual(payload["improvement_ukeire_count"], 45)
        self.assertAlmostEqual(payload["ukeire_score"], 25 + 0.5 * 14.72 + 0.1 * 45)

        weighted_future = sum(row["remaining"] * row["future_ukeire"] for row in payload["ukeire"])
        self.assertAlmostEqual(weighted_future / payload["ukeire_count"], payload["future_ukeire_expected"])

    def test_ukeire_score_weights_are_tunable_and_zero_weights_recover_v1(self):
        hand = "2378m234789p1s36z"
        tuned = sim.analyze_hand(hand, future_ukeire_weight=0.25, improvement_weight=0.2)
        self.assertAlmostEqual(tuned["ukeire_score"], 25 + 0.25 * 14.72 + 0.2 * 45)
        self.assertEqual(tuned["policy"]["future_ukeire_weight"], 0.25)
        self.assertEqual(tuned["policy"]["improvement_weight"], 0.2)

        version_one = sim.analyze_hand(hand, future_ukeire_weight=0.0, improvement_weight=0.0)
        self.assertEqual(version_one["ukeire_score"], version_one["ukeire_count"])

    def test_future_ukeire_weight_can_scale_by_shanten(self):
        evaluator = sim.PureTileEfficiencyEvaluator(
            future_ukeire_weight=0.5,
            future_ukeire_weight_1_shanten=1.25,
            future_ukeire_weight_2_shanten=0.75,
        )
        base = self._dummy_features(0, score=10.0)
        one = replace(base, shanten=1, future_ukeire_expected=8.0)
        two = replace(base, shanten=2, future_ukeire_expected=8.0)
        three = replace(base, shanten=3, future_ukeire_expected=8.0)
        self.assertEqual(evaluator.compose_ukeire_score(one), 20.0)
        self.assertEqual(evaluator.compose_ukeire_score(two), 16.0)
        self.assertEqual(evaluator.compose_ukeire_score(three), 10.0)
        self.assertEqual(
            evaluator.parameters()["future_ukeire_weights_by_shanten"],
            {"1": 1.25, "2": 0.75, "3_plus": 0.0},
        )

    def test_analyze_cli_forwards_shanten_specific_future_weights(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                sim.main([
                    "analyze", "2378m234789p1s36z",
                    "--future-ukeire-weight-1-shanten", "1.25",
                    "--future-ukeire-weight-2-shanten", "0.75",
                ]),
                0,
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(
            payload["policy"]["future_ukeire_weights_by_shanten"],
            {"1": 1.25, "2": 0.75, "3_plus": 0.0},
        )
        self.assertAlmostEqual(payload["ukeire_score"], 25 + 0.75 * 14.72 + 0.1 * 45)

    def test_shanten_still_dominates_advanced_ukeire_score(self):
        payload = sim.analyze_hand(
            "123456789m9p11p23s",
            future_ukeire_weight=100.0,
            improvement_weight=100.0,
        )
        best = [row for row in payload["discards"] if row["best"]]
        self.assertEqual([row["tile"] for row in best], ["9p"])
        self.assertEqual(best[0]["shanten"], 0)

        value_payload = sim.analyze_hand(
            "123456789m9p11p23s",
            policy="custom",
            dora_indicator_text="8p",
            dora_ukeire_weight=10_000.0,
        )
        value_best = [row for row in value_payload["discards"] if row["best"]]
        self.assertEqual([row["tile"] for row in value_best], ["9p"])
        self.assertEqual(value_best[0]["shanten"], 0)

    def test_shanten_reversal_formula_pruning_and_strict_tie(self):
        hand = "46777m1p1233345s6z"
        enabled = sim.ShantenReversalConfig(enabled=True)
        payload = sim.analyze_hand(
            hand,
            dora_indicator_text="9p",
            shanten_reversal=enabled,
            turn=5,
        )
        deep = [row for row in payload["discards"] if row["reversal_deep_evaluated"]]
        self.assertEqual(len(deep), 2)
        self.assertEqual({row["tile"] for row in deep}, {"4m", "6m"})
        chosen = next(row for row in payload["discards"] if row["best"])
        self.assertEqual((chosen["tile"], chosen["shanten"]), ("6m", 2))
        self.assertAlmostEqual(
            chosen["reversal_adjusted_score"],
            chosen["reversal_recovery_count"]
            + 0.5
            * (
                chosen["reversal_future_ukeire_expected"]
                - chosen["reversal_current_ukeire"]
            )
            - 25.0,
        )
        self.assertAlmostEqual(
            chosen["reversal_ukeire_gain"],
            chosen["reversal_future_ukeire_expected"] - chosen["reversal_current_ukeire"],
        )

        strict_score = max(
            row["ukeire_score"] for row in payload["discards"] if row["shanten"] == 1
        )
        raw_reversal_score = chosen["reversal_adjusted_score"] + 25.0
        tied = sim.analyze_hand(
            hand,
            dora_indicator_text="9p",
            shanten_reversal=sim.ShantenReversalConfig(
                enabled=True,
                penalty=raw_reversal_score - strict_score,
            ),
            turn=5,
        )
        self.assertTrue(all(row["shanten"] == 1 for row in tied["discards"] if row["best"]))

    def test_shanten_reversal_turn_budget_and_dora_bonus(self):
        hand = "46777m1p1233345s6z"
        reversal = sim.ShantenReversalConfig(enabled=True)
        at_nine = sim.analyze_hand(hand, shanten_reversal=reversal, turn=9)
        after_nine = sim.analyze_hand(hand, shanten_reversal=reversal, turn=10)
        already_used = sim.analyze_hand(
            hand, shanten_reversal=reversal, turn=5, shanten_reversals_used=1
        )
        self.assertTrue(any(row["is_shanten_reversal"] and row["best"] for row in at_nine["discards"]))
        self.assertTrue(all(row["shanten"] == 1 for row in after_nine["discards"] if row["best"]))
        self.assertTrue(all(row["shanten"] == 1 for row in already_used["discards"] if row["best"]))

        pure = sim.analyze_hand(hand, shanten_reversal=reversal, turn=5)
        value = sim.analyze_hand(
            hand,
            policy="custom",
            dora_indicator_text="3m",
            dora_ukeire_weight=2.0,
            shanten_reversal=reversal,
            turn=5,
        )
        pure_six = next(row for row in pure["discards"] if row["tile"] == "6m")
        value_six = next(row for row in value["discards"] if row["tile"] == "6m")
        self.assertGreater(value_six["dora_ukeire_bonus"], pure_six["dora_ukeire_bonus"])

    def test_custom_evaluator_hook_can_replace_ukeire_tiebreak(self):
        class PreferHighDiscard:
            name = "prefer_high"

            def rank(self, features, context):
                del context
                return (-features.shanten, features.discard_type)

        counts = sim.parse_hand("123456789m9p11p23s")
        physical = tuple(tile * 4 + copy for tile, count in enumerate(counts) for copy in range(count))
        context = sim.DecisionContext(1, physical, (), (), physical[-1], 0)
        discarded, chosen, _ = sim.choose_discard(context, PreferHighDiscard(), random.Random(7))
        self.assertEqual(discarded // 4, chosen.features.discard_type)
        self.assertEqual(chosen.features.shanten, 0)

    def test_value_policy_presets_and_dora_cycles(self):
        pure = sim.make_policy_evaluator("pure")
        value = sim.make_policy_evaluator("value")
        custom = sim.make_policy_evaluator("custom", yakuhai_weight=2.5)
        self.assertIsInstance(pure, sim.PureTileEfficiencyEvaluator)
        self.assertNotIsInstance(pure, sim.HandValueHeuristicEvaluator)
        self.assertEqual(value.dora_ukeire_weight, 1.0)
        self.assertEqual(value.tanyao_weight, 1.0)
        self.assertEqual(custom.dora_ukeire_weight, 0.0)
        self.assertEqual(custom.yakuhai_weight, 2.5)
        with self.assertRaises(ValueError):
            sim.make_policy_evaluator("pure", tanyao_weight=1.0)

        self.assertEqual(sim.dora_type_from_indicator(8), 0)  # 9m -> 1m
        self.assertEqual(sim.dora_type_from_indicator(30), 27)  # North -> East
        self.assertEqual(sim.dora_type_from_indicator(33), 31)  # Red -> White

    def test_dora_yakuhai_and_tanyao_features_are_ordered_tiebreaks(self):
        hand = sim.parse_physical_hand("05m123p123s9m112z5z")
        context = sim.DecisionContext(
            1,
            hand,
            (),
            (),
            hand[-1],
            sim.parse_single_tile_type("4m") * 4,
            player_wind=27,
            round_wind=27,
        )
        value = sim.make_policy_evaluator("value")
        discard_five = value.finalize_features(self._dummy_features(4), context)
        self.assertEqual(discard_five.retained_dora, 2)  # kept red 5m: aka + indicator dora
        self.assertEqual(discard_five.retained_yakuhai_value, 5)  # double-East pair + White
        self.assertEqual(discard_five.ukeire_score, 12.0)

        yakuhai_only = sim.make_policy_evaluator("custom", yakuhai_weight=1.0)
        discard_east = yakuhai_only.finalize_features(self._dummy_features(27), context)
        discard_white = yakuhai_only.finalize_features(self._dummy_features(31), context)
        discard_terminal = yakuhai_only.finalize_features(self._dummy_features(8), context)
        self.assertGreater(discard_terminal.value_tiebreak_score, discard_white.value_tiebreak_score)
        self.assertGreater(discard_white.value_tiebreak_score, discard_east.value_tiebreak_score)

        tanyao_only = sim.make_policy_evaluator("custom", tanyao_weight=1.0)
        discard_1p = tanyao_only.finalize_features(self._dummy_features(9), context)
        discard_2p = tanyao_only.finalize_features(self._dummy_features(10), context)
        self.assertGreater(discard_1p.value_tiebreak_score, discard_2p.value_tiebreak_score)

    def test_value_policy_analysis_exposes_adjusted_and_tie_scores(self):
        payload = sim.analyze_hand(
            "234567m22p345s11z",
            policy="value",
            dora_indicator_text="4m",
            player_wind=27,
            round_wind=27,
        )
        self.assertEqual(payload["dora"], "5m")
        self.assertEqual(payload["value_features"]["retained_dora"], 1)
        self.assertEqual(payload["value_features"]["retained_yakuhai_value"], 4)
        self.assertEqual(payload["ukeire_score"], payload["base_ukeire_score"] + 1)

    def test_tanyao_tiebreak_keeps_simple_tiles_in_an_actual_discard_tie(self):
        hand = "3388m226p25779s34z"
        pure = sim.analyze_hand(
            hand,
            future_ukeire_weight=0.0,
            improvement_weight=0.0,
        )
        custom = sim.analyze_hand(
            hand,
            policy="custom",
            future_ukeire_weight=0.0,
            improvement_weight=0.0,
            tanyao_weight=1.0,
        )
        pure_best = {row["tile"] for row in pure["discards"] if row["best"]}
        custom_best = {row["tile"] for row in custom["discards"] if row["best"]}
        self.assertTrue({"6p", "2s", "5s"}.issubset(pure_best))
        self.assertTrue(custom_best)
        self.assertTrue(custom_best.isdisjoint({"6p", "2s", "5s"}))


class SimulationTests(unittest.TestCase):
    def test_shanten_reversal_is_selected_at_most_once_and_reported(self):
        evaluator = sim.PureTileEfficiencyEvaluator(
            shanten_reversal=sim.ShantenReversalConfig(enabled=True, penalty=0.0)
        )
        aggregate = sim.SimulationAggregate()
        selected = 0
        for trial_id in range(12):
            result = sim.simulate_trial(
                trial_id,
                17,
                evaluator=evaluator,
                config=sim.SimulationConfig(stop_at="tenpai"),
            )
            self.assertLessEqual(len(result.shanten_reversals), 1)
            self.assertTrue(all(record.ukeire_gain > 0 for record in result.shanten_reversals))
            selected += len(result.shanten_reversals)
            aggregate.add(result)
        self.assertGreater(selected, 0)
        self.assertEqual(aggregate.reversals_selected, selected)
        self.assertTrue(set(aggregate.reversal_transitions).issubset({"0->1", "1->2"}))

    def test_multiple_reversals_keep_all_records_and_recoveries(self):
        evaluator = sim.PureTileEfficiencyEvaluator(
            future_ukeire_weight=0.0,
            improvement_weight=0.0,
            shanten_reversal=sim.ShantenReversalConfig(
                enabled=True, penalty=0.0, max_turn=18, max_per_hand=4,
            ),
        )
        result = sim.simulate_trial(
            1, 17, evaluator=evaluator, config=sim.SimulationConfig(stop_at="tenpai")
        )
        self.assertEqual([record.turn for record in result.shanten_reversals], [1, 2, 3, 4])
        self.assertTrue(all(record.recovered for record in result.shanten_reversals))
        self.assertTrue(all(record.recovery_turns is not None for record in result.shanten_reversals))

        aggregate = sim.SimulationAggregate()
        aggregate.add(result)
        self.assertEqual(aggregate.reversals_selected, 4)
        self.assertEqual(aggregate.reversals_recovered, 4)

    def test_trial_seeding_is_stable_and_shard_independent(self):
        self.assertEqual(sim.trial_seed(12, 99), sim.trial_seed(12, 99))
        self.assertNotEqual(sim.trial_seed(12, 99), sim.trial_seed(12, 100))
        first = sim.simulate_trial(3, 991, config=sim.SimulationConfig(stop_at="tenpai"))
        second = sim.simulate_trial(3, 991, config=sim.SimulationConfig(stop_at="tenpai"))
        self.assertEqual(first, second)

    def test_physical_trial_has_unique_allocated_ids(self):
        result = sim.simulate_trial(0, 123, config=sim.SimulationConfig(stop_at="tenpai"))
        self.assertEqual(len(result.initial_hand_physical), 13)
        self.assertEqual(len(set(result.initial_hand_physical)), 13)
        self.assertIn(result.outcome, ("tenpai", "exhausted"))

    def test_batch_streams_jsonl_and_raw_mergeable_counts(self):
        stream = io.StringIO()
        aggregate = sim.run_batch(
            3,
            77,
            config=sim.SimulationConfig(stop_at="tenpai"),
            observation_stream=stream,
        )
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["trial_id"] for row in rows], [0, 1, 2])
        payload = aggregate.to_dict(seed=77, start_trial=0, policy="test", scorer=None)
        self.assertEqual(payload["trials"], 3)
        self.assertIn("tile_danger_at_first_tenpai", payload)
        self.assertIn("raw", payload)

    def test_decision_observations_include_ukeire_score_components(self):
        result = sim.simulate_trial(
            0,
            919,
            config=sim.SimulationConfig(stop_at="tenpai", collect_decisions=True),
        )
        self.assertTrue(result.decisions)
        decision = next(row for row in result.decisions if not row.forced_by_riichi)
        self.assertGreaterEqual(decision.post_discard_ukeire_score, decision.post_discard_ukeire)
        self.assertGreaterEqual(decision.post_discard_future_ukeire, 0.0)
        self.assertGreaterEqual(decision.post_discard_improvement_ukeire, 0)

    def test_aggregate_merge_matches_whole_batch(self):
        config = sim.SimulationConfig(stop_at="tenpai")
        whole = sim.run_batch(6, 101, config=config)
        left = sim.run_batch(2, 101, start_trial=0, config=config)
        right = sim.run_batch(4, 101, start_trial=2, config=config)
        left.merge(right)
        self.assertEqual(left.trials, whole.trials)
        self.assertEqual(left.outcomes, whole.outcomes)
        self.assertEqual(left.wait_signatures, whole.wait_signatures)
        self.assertEqual(left.tile_wait_hands, whole.tile_wait_hands)

    def test_policy_comparison_uses_paired_trial_seeds(self):
        candidate = sim.make_policy_evaluator("custom", tanyao_weight=1.0)
        payload = sim.compare_policy_batches(
            2,
            517,
            candidate=candidate,
            config=sim.SimulationConfig(stop_at="tenpai"),
        )
        self.assertTrue(payload["paired_by_trial_seed"])
        self.assertEqual(payload["baseline"]["trials"], 2)
        self.assertEqual(payload["candidate"]["trials"], 2)
        self.assertEqual(payload["candidate"]["policy"], "custom_hand_value")
        self.assertIn("tenpai_rate", payload["delta_candidate_minus_baseline"])

    def test_policy_comparison_preserves_shanten_specific_future_weights(self):
        candidate = sim.make_policy_evaluator(
            "custom",
            future_ukeire_weight=0.5,
            future_ukeire_weight_1_shanten=1.25,
            future_ukeire_weight_2_shanten=0.75,
        )
        payload = sim.compare_policy_batches(
            1,
            517,
            candidate=candidate,
            config=sim.SimulationConfig(stop_at="tenpai"),
        )
        expected = {"1": 1.25, "2": 0.75, "3_plus": 0.0}
        self.assertEqual(
            payload["baseline"]["policy_parameters"]["future_ukeire_weights_by_shanten"], expected
        )
        self.assertEqual(
            payload["candidate"]["policy_parameters"]["future_ukeire_weights_by_shanten"], expected
        )

    def test_score_hook_collects_ron_and_tsumo_separately(self):
        class ConstantScorer:
            name = "constant"

            def score(self, hand, win_tile, context):
                del hand, win_tile
                points = 3000 if context.is_tsumo else 2000
                return sim.HandScore(points, 2, 30, ("test",), {"total": points})

        aggregate = sim.run_batch(25, 2026, scorer=ConstantScorer())
        payload = aggregate.to_dict(seed=2026, start_trial=0, policy="test", scorer="constant")
        wins = payload["wins"]["hands"]
        self.assertGreater(wins, 0)
        self.assertEqual(payload["score_values_on_realized_winning_hands"]["tsumo"]["count"], wins)
        self.assertEqual(payload["score_values_on_realized_winning_hands"]["tsumo"]["mean_points"], 3000)
        self.assertEqual(payload["score_values_on_realized_winning_hands"]["ron"]["mean_points"], 2000)

    def test_mahjong_repository_score_adapter(self):
        try:
            scorer = sim.MahjongRepositoryScorer()
        except RuntimeError:
            self.skipTest("optional scorer dependency is not installed")
        aggregate = sim.run_batch(25, 606, scorer=scorer)
        payload = aggregate.to_dict(seed=606, start_trial=0, policy="test", scorer=scorer.name)
        scored = payload["score_values_on_realized_winning_hands"]["tsumo"]["count"]
        self.assertEqual(scored, payload["wins"]["hands"])

    def test_native_observation_score_postprocessor(self):
        class ConstantScorer:
            name = "constant"

            def score(self, hand, win_tile, context):
                del hand, win_tile
                points = 3000 if context.is_tsumo else 2000
                return sim.HandScore(points, 2, 30, ("test",), {"total": points})

        row = {
            "engine_version": "closed-hand-efficiency-cpp-test",
            "player_wind": 28,
            "round_wind": 27,
            "dora_indicator": 12,
            "ura_dora_indicator": 13,
            "first_tenpai": {"riichi_declared": True},
            "win": {
                "tile": 20,
                "hand_physical": list(range(14)),
                "ippatsu": True,
                "haitei": False,
                "tenhou": False,
                "chiihou": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            payload = sim.score_observation_file(path, ConstantScorer())
        self.assertEqual(payload["rows"], 1)
        self.assertEqual(payload["wins"], 1)
        self.assertEqual(payload["evaluator_backends"], {"unknown": 1})
        self.assertEqual(payload["score_values_on_realized_winning_hands"]["tsumo"]["mean_points"], 3000)
        self.assertEqual(payload["score_values_on_realized_winning_hands"]["ron"]["mean_points"], 2000)


@unittest.skipUnless(shutil.which("c++"), "C++ compiler is not installed")
class NativeEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls._temporary.name) / "hand_efficiency_sim_cpp"
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O2",
                str(ROOT / "outputs" / "hand_efficiency_sim.cpp"),
                "-o",
                str(cls.binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def _run_json(self, *arguments):
        output = subprocess.check_output([str(self.binary), *arguments], text=True)
        return json.loads(output)

    def test_native_self_test_and_advanced_fixture(self):
        result = subprocess.run([str(self.binary), "--self-test"], check=True, capture_output=True, text=True)
        self.assertIn("ok", result.stdout)
        payload = self._run_json("analyze", "2378m234789p1s36z")
        self.assertEqual(payload["evaluator_backend"], "builtin_fallback")
        self.assertEqual(payload["shanten"], 2)
        self.assertEqual(payload["ukeire_count"], 25)
        self.assertAlmostEqual(payload["future_ukeire_expected"], 14.72)
        self.assertEqual(payload["improvement_ukeire_count"], 45)
        self.assertAlmostEqual(payload["ukeire_score"], 36.86)

        scaled = self._run_json(
            "analyze", "2378m234789p1s36z",
            "--future-ukeire-weight-1-shanten", "1.25",
            "--future-ukeire-weight-2-shanten", "0.75",
        )
        self.assertAlmostEqual(scaled["ukeire_score"], 25 + 0.75 * 14.72 + 0.1 * 45)
        self.assertEqual(
            scaled["policy_parameters"]["future_ukeire_weights_by_shanten"],
            {"1": 1.25, "2": 0.75, "3_plus": 0},
        )

    def test_native_shards_are_reproducible_and_physical(self):
        common = [
            "simulate",
            "--seed",
            "8901",
            "--stop-at",
            "tenpai",
            "--future-ukeire-weight",
            "0",
            "--improvement-weight",
            "0",
        ]
        with tempfile.TemporaryDirectory() as directory:
            whole_path = Path(directory) / "whole.jsonl"
            left_path = Path(directory) / "left.jsonl"
            right_path = Path(directory) / "right.jsonl"
            self._run_json(*common, "--trials", "6", "--observations", str(whole_path))
            self._run_json(*common, "--trials", "2", "--observations", str(left_path))
            self._run_json(
                *common,
                "--trials",
                "4",
                "--start-trial",
                "2",
                "--observations",
                str(right_path),
            )
            whole = [json.loads(line) for line in whole_path.read_text().splitlines()]
            shards = [json.loads(line) for line in (left_path.read_text() + right_path.read_text()).splitlines()]
        self.assertEqual(whole, shards)
        self.assertEqual([row["trial_id"] for row in whole], list(range(6)))
        for row in whole:
            physical = row["initial_hand_physical"]
            self.assertEqual(len(physical), 13)
            self.assertEqual(len(set(physical)), 13)

    def test_native_tile_availability_checkpoint_is_synchronized_and_conserved(self):
        """The turn-N snapshot must represent one shared wall for all four seats."""
        with tempfile.TemporaryDirectory() as directory:
            observations = Path(directory) / "availability.jsonl"
            summary = self._run_json(
                "simulate-table",
                "--trials", "1",
                "--seed", "81723",
                "--no-riichi",
                "--tile-availability-turn", "9",
                "--observations", str(observations),
            )
            rows = [json.loads(line) for line in observations.read_text().splitlines()]

        self.assertEqual(summary["trials"], 1)
        self.assertEqual(summary["config"]["tile_availability_turn"], 9)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["outcome"], "availability_checkpoint")
        checkpoint = row["availability_checkpoint"]
        self.assertEqual(checkpoint["target_own_turn"], 9)
        self.assertEqual(checkpoint["completed_own_turn"], 9)
        self.assertEqual(checkpoint["live_wall_tiles_remaining"], 34)
        self.assertEqual(checkpoint["hidden_dead_wall_tiles_remaining"], 13)
        self.assertEqual(checkpoint["opponent_hand_tiles"], 39)
        self.assertEqual(sum(checkpoint["public_counts"]), 37)  # 36 discards + dora indicator
        self.assertEqual(len(checkpoint["players"]), 4)

        live_wall_counts = None
        hidden_dead_wall_counts = None
        for seat, player in enumerate(checkpoint["players"]):
            self.assertEqual(player["seat"], seat)
            self.assertEqual(sum(player["own_known_counts"]), 13)
            self.assertEqual(player["unseen_tiles"], 86)
            self.assertEqual(player["live_wall_tiles"], 34)
            self.assertEqual(player["hidden_dead_wall_tiles"], 13)
            self.assertEqual(player["opponent_hand_tiles"], 39)
            self.assertEqual(sum(player["unseen_counts"]), 86)
            self.assertEqual(sum(player["live_wall_counts"]), 34)
            self.assertEqual(sum(player["hidden_dead_wall_counts"]), 13)
            self.assertEqual(sum(player["opponent_hand_counts"]), 39)
            self.assertEqual(player["public_counts"], checkpoint["public_counts"])
            for tile in range(34):
                self.assertEqual(
                    player["unseen_counts"][tile],
                    player["live_wall_counts"][tile]
                    + player["hidden_dead_wall_counts"][tile]
                    + player["opponent_hand_counts"][tile],
                    f"seat {seat} tile {tile} compartment mismatch",
                )
                self.assertLessEqual(
                    player["own_known_counts"][tile] + player["public_counts"][tile],
                    4,
                    f"seat {seat} tile {tile} has too many visible copies",
                )
            if live_wall_counts is None:
                live_wall_counts = player["live_wall_counts"]
                hidden_dead_wall_counts = player["hidden_dead_wall_counts"]
            else:
                self.assertEqual(player["live_wall_counts"], live_wall_counts)
                self.assertEqual(player["hidden_dead_wall_counts"], hidden_dead_wall_counts)

    def test_native_value_features_and_paired_comparison(self):
        payload = self._run_json(
            "analyze",
            "234567m22p345s11z",
            "--policy",
            "value",
            "--dora-indicator",
            "4m",
            "--seat-wind",
            "east",
        )
        self.assertEqual(payload["dora"], "5m")
        self.assertEqual(payload["value_features"]["retained_dora"], 1)
        self.assertEqual(payload["value_features"]["retained_yakuhai_value"], 4)

        comparison = self._run_json(
            "compare",
            "--trials",
            "2",
            "--stop-at",
            "tenpai",
            "--policy",
            "custom",
            "--tanyao-weight",
            "1",
        )
        self.assertTrue(comparison["paired_by_trial_seed"])
        self.assertEqual(comparison["baseline"]["policy"], "shanten_then_ukeire_score")
        self.assertEqual(comparison["candidate"]["policy"], "custom_hand_value")


if __name__ == "__main__":
    unittest.main()
