import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))

import hand_efficiency_sim as sim
from hand_push_fold_scoring import SCORE_RESPONSE_FIELDS, PushFoldScorer, ScoringConfig


class FakeScorer:
    name = "fake"

    def __init__(self):
        self.contexts = []
        self.hand_lengths = []

    def score(self, hand, win_tile, context):
        self.hand_lengths.append(len(hand))
        self.contexts.append(context)
        tile = win_tile // 4
        if not context.is_riichi and not context.is_tsumo and tile == 18:
            return sim.HandScore(None, None, None, (), {}, error="no yaku")
        if context.is_tsumo:
            return sim.HandScore(12000, 6, 30, ("Tsumo",), {"total": 12000})
        if context.is_riichi:
            return sim.HandScore(1300, 1, 40, ("Riichi",), {"total": 1300})
        return sim.HandScore(5200 if tile == 21 else 8000, 3, 40, (), {"total": 8000})


def request(hand="123456789m11p23s", *, seat=1):
    physical = list(sim.parse_physical_hand(hand))
    return {
        "kind": "score_request",
        "trial_id": 7,
        "seat": seat,
        "turn": 8,
        "hand_physical": physical,
        "known_physical": physical,
        "round_wind": 27,
        "dora_indicator": 0,
    }


class PushFoldScoringTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeScorer()
        self.service = PushFoldScorer(self.fake)

    def fields(self, response):
        return dict(zip(SCORE_RESPONSE_FIELDS, response))

    def test_native_request_and_decision_scoring_uses_completed_hand(self):
        response = self.service.score_request(request(), ScoringConfig())
        fields = self.fields(response)
        self.assertEqual(len(response), len(SCORE_RESPONSE_FIELDS))
        self.assertEqual(fields["good_wait"], 1)
        self.assertEqual(fields["cheap_all_unseen"], 1)
        self.assertEqual(fields["exact_1300_bad_wait"], 0)
        self.assertEqual(fields["min_dama_ron_points"], 0)
        self.assertTrue(self.fake.hand_lengths)
        self.assertEqual(set(self.fake.hand_lengths), {14})
        self.assertTrue(all(context.player_wind == 28 for context in self.fake.contexts))

    def test_legal_mask_uses_structural_wait_even_when_known_exhausts_it(self):
        base = request()
        base["known_physical"] = list(range(136))
        fields = self.fields(self.service.score_request(base, ScoringConfig()))
        self.assertEqual(fields["good_wait"], 0)
        self.assertEqual(fields["valuable_dama"], 0)
        self.assertTrue(fields["legal_ron_mask"] & (1 << 21))

    def test_whole_wait_category_is_not_reduced_to_unseen_subset(self):
        base = request()
        # 1s is one side of the ryanmen wait.  Exhaust it visibly, leaving
        # only 4s unseen; the complete hand still has a ryanmen structure.
        one_s = 18
        base["known_physical"] = sorted(set(base["known_physical"]) | set(range(one_s, one_s + 4)))
        fields = self.fields(self.service.score_request(base, ScoringConfig()))
        self.assertEqual(fields["good_wait"], 1)

    def test_bad_wait_uses_legacy_exact_1300_and_value_line(self):
        fields = self.fields(self.service.score_request(
            request("1233m123p123s111z"), ScoringConfig(valuable_bad_wait_points=5200)
        ))
        self.assertEqual(fields["good_wait"], 0)
        self.assertEqual(fields["exact_1300_bad_wait"], 1)
        self.assertEqual(fields["valuable_dama"], 1)

    def test_chiitoitsu_is_not_a_legacy_cheap_bad_wait(self):
        fields = self.fields(self.service.score_request(
            request("1122m1122p1122s1z"), ScoringConfig()
        ))
        self.assertEqual(fields["exact_1300_bad_wait"], 0)

    def test_exact_1300_uses_copy_weighted_mean_not_each_copy(self):
        class UnevenRiichiScorer(FakeScorer):
            def score(self, hand, win_tile, context):
                if context.is_riichi:
                    # The two unseen 3m copies score 1000 and 1600, so their
                    # visible-copy mean is exactly 1300.
                    fu = {10: 30, 11: 50}.get(win_tile, 40)
                    return sim.HandScore(0, 1, fu, ("Riichi",), {})
                return super().score(hand, win_tile, context)

        service = PushFoldScorer(UnevenRiichiScorer())
        fields = self.fields(service.score_request(
            request("1233m123p123s111z"), ScoringConfig()
        ))
        self.assertEqual(fields["exact_1300_bad_wait"], 1)

    def test_haneman_tsumo_fallback_requires_no_yaku_ron(self):
        # The fake scorer returns a legal 5200 dama ron for 4s and a 12000
        # tsumo.  That legal but insufficient ron must fail the lock guard.
        fields = self.fields(self.service.score_request(request(), ScoringConfig()))
        self.assertEqual(fields["haneman_lock"], 0)
        self.assertFalse(fields["haneman_ron_mask"] & (1 << 21))

    def test_guards_are_nondealer_normalized_but_actual_minimum_is_dealer(self):
        dealer_request = request("1233m123p123s111z", seat=0)
        self.service.score_request(dealer_request, ScoringConfig())
        contexts = self.fake.contexts
        self.assertTrue(any(not context.is_dealer for context in contexts))
        self.assertTrue(any(context.is_dealer for context in contexts))
        self.assertTrue(all(context.player_wind == 27 for context in contexts))

    def test_realized_win_uses_requested_tsumo_and_preserves_details(self):
        base = request()
        win = 18
        base.update({
            "kind": "score_realized_win",
            "hand_physical": sorted((*base["hand_physical"], win)),
            "win_tile": win,
            "is_riichi": True,
            "is_tsumo": True,
            "ura_dora_indicators": [4],
            "ippatsu": True,
            "haitei": True,
        })
        result = self.service.score_realized_win(base)
        self.assertEqual(result["points"], 12000)
        self.assertEqual(result["han"], 6)
        self.assertEqual(result["cost"]["total"], 12000)
        context = self.fake.contexts[-1]
        self.assertTrue(context.is_tsumo)
        self.assertTrue(context.is_riichi)
        self.assertTrue(context.is_ippatsu)
        self.assertTrue(context.is_haitei)
        self.assertEqual(context.ura_dora_indicators, (4,))

    def test_known_tiles_must_include_hand(self):
        bad = request()
        bad["known_physical"] = []
        with self.assertRaises(ValueError):
            self.service.score_request(bad, ScoringConfig())

    def test_real_scorer_scores_completed_13_plus_win_fixture(self):
        service = PushFoldScorer()
        fields = self.fields(service.score_request(request(), ScoringConfig()))
        self.assertGreater(fields["min_riichi_ron_points"], 0)

    def test_nondealer_normalization_preserves_double_yakuman(self):
        raw = sim.HandScore(
            96000, 26, 0, ("Kokushi Musou",),
            {"total": 96000, "yaku_level": "2x yakuman"},
        )
        normalized = PushFoldScorer._nondealer_normalized(raw, tsumo=False)
        self.assertEqual(normalized.points, 64000)


if __name__ == "__main__":
    unittest.main()
