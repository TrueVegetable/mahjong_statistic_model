// Native continuation engine for --push-fold. Included after the legacy
// danger engine so the old simulation paths remain byte-for-byte isolated.

int push_fold_best_shanten(const std::vector<int>& hand, ShantenCalculator& calculator) {
    Counts counts = counts_from_physical(hand);
    int best = 99;
    for (int tile = 0; tile < TILE_TYPES; ++tile) {
        if (!counts[tile]) continue;
        --counts[tile];
        best = std::min(best, calculator.calculate(counts));
        ++counts[tile];
    }
    return best;
}

int base_safety_tier(int tile, const std::array<bool, TILE_TYPES>& safe) {
    if (safe[tile]) return 0;
    if (tile >= 27) return 2;
    const int start = tile / 9 * 9;
    const int rank = tile % 9 + 1;
    auto anchor = [&](int r) { return safe[start + r - 1]; };
    if (rank == 1) return anchor(4) ? 1 : 6;
    if (rank == 9) return anchor(6) ? 1 : 6;
    if (rank == 2 || rank == 8) return anchor(5) ? 4 : 8;
    if (rank == 3) return anchor(6) ? 5 : 9;
    if (rank == 7) return anchor(4) ? 5 : 9;
    const int low_anchor = rank - 3;
    const int high_anchor = rank + 3;
    const int anchors = int(anchor(low_anchor)) + int(anchor(high_anchor));
    if (anchors == 2) return 3;
    if (anchors == 1) return 7;
    return 10;
}

int adjusted_safety_tier(
    int physical, const std::array<bool, TILE_TYPES>& safe,
    int dora_type, int dora_steps
) {
    int tier = base_safety_tier(physical / 4, safe);
    if (tier != 0 && dora_steps > 0
        && (is_red_five(physical) || physical / 4 == dora_type)) {
        tier += dora_steps;
    }
    return tier;
}

std::vector<int> push_fold_known(
    const std::vector<int>& full_hand, const std::vector<int>& public_river,
    int dora_indicator
) {
    std::vector<int> known = full_hand;
    known.insert(known.end(), public_river.begin(), public_river.end());
    known.push_back(dora_indicator);
    return known;
}

struct PushFoldChoice {
    int discarded = -1;
    PolicyFeatures features;
    bool has_score = false;
    ScoreReply score;
};

int two_threat_intended_min_ron(
    const ScoreReply& score, std::size_t live_wall_remaining,
    std::size_t minimum_live_tiles_for_riichi = 4
) {
    const bool intended_dama = live_wall_remaining < minimum_live_tiles_for_riichi
        || score.exact_1300_bad_wait || score.valuable_dama || score.haneman_lock;
    return intended_dama ? score.min_dama_ron_points : score.min_riichi_ron_points;
}

// Same as push_fold_features, with the table's actual indicator included.
PolicyFeatures push_fold_features_with_dora(
    const std::vector<int>& full_hand, int physical, const std::vector<int>& public_river,
    int dora_indicator, const Config& config, int seat, ShantenCalculator& calculator
) {
    std::vector<int> post = full_hand;
    const auto found = std::find(post.begin(), post.end(), physical);
    if (found == post.end()) throw std::logic_error("push-fold discard not in hand");
    post.erase(found);
    Counts post_counts = counts_from_physical(post);
    Visible external{};
    for (int tile : public_river) ++external[tile / 4];
    ++external[dora_indicator / 4];
    ++external[physical / 4];
    PolicyFeatures features = evaluate_13(
        post_counts, physical / 4, external, config.future,
        config.improvement_weight, calculator
    );
    PolicyFeatures exact = features;
    exact.discard_type = -1;
    apply_value_features(exact, post, dora_indicator, 27 + seat, config.round_wind, config.policy);
    exact.discard_type = physical / 4;
    if (config.policy.kanchan_outer_wait_weight != 0.0 && exact.shanten == 0
        && is_pure_outer_kanchan(post_counts, calculator)) {
        exact.tie_score += config.policy.kanchan_outer_wait_weight;
    }
    return exact;
}

PushFoldChoice choose_push_efficiency(
    std::vector<int>& hand, const std::vector<int>& public_river, int dora_indicator,
    const Config& config, int seat, const std::vector<int>& allowed_physical,
    Rng& rng, ShantenCalculator& calculator
) {
    if (allowed_physical.empty()) throw std::logic_error("no push-fold efficiency candidates");
    std::vector<std::pair<int, PolicyFeatures>> evaluated;
    double best_score = -std::numeric_limits<double>::infinity();
    double best_tie = -std::numeric_limits<double>::infinity();
    for (int physical : allowed_physical) {
        PolicyFeatures features = push_fold_features_with_dora(
            hand, physical, public_river, dora_indicator, config, seat, calculator
        );
        if (features.score > best_score + 1e-12
            || (std::abs(features.score - best_score) <= 1e-12
                && features.tie_score > best_tie + 1e-12)) {
            best_score = features.score;
            best_tie = features.tie_score;
            evaluated.clear();
        }
        if (std::abs(features.score - best_score) <= 1e-12
            && std::abs(features.tie_score - best_tie) <= 1e-12) {
            evaluated.emplace_back(physical, std::move(features));
        }
    }
    auto chosen = evaluated.at(static_cast<std::size_t>(rng.uniform(evaluated.size())));
    const auto found = std::find(hand.begin(), hand.end(), chosen.first);
    hand.erase(found);
    PushFoldChoice result;
    result.discarded = chosen.first;
    result.features = std::move(chosen.second);
    return result;
}

std::pair<double, int> safety_key(const std::vector<int>& tiers, const std::string& strategy) {
    const int worst = tiers.empty() ? 0 : *std::max_element(tiers.begin(), tiers.end());
    const double mean = tiers.empty() ? 0.0
        : double(std::accumulate(tiers.begin(), tiers.end(), 0)) / tiers.size();
    return strategy == "minimax_safety" ? std::make_pair(double(worst), int(std::lround(mean * 1000)))
                                        : std::make_pair(mean, worst);
}

int choose_full_fold(
    std::vector<int>& hand, const std::vector<int>& threats,
    const std::array<std::array<bool, TILE_TYPES>, 4>& safe,
    const Counts& public_counts, int dora_type, const Config& config,
    Rng& rng
) {
    std::vector<int> best;
    std::pair<double, int> best_key{std::numeric_limits<double>::infinity(), 999999};
    int best_known = -1;
    Counts own = counts_from_physical(hand);
    for (int physical : hand) {
        std::vector<int> tiers;
        for (int threat : threats) tiers.push_back(adjusted_safety_tier(
            physical, safe[threat], dora_type, config.dora_danger_steps
        ));
        const auto key = safety_key(tiers, config.safety_strategy);
        const int known = own[physical / 4] + public_counts[physical / 4];
        if (key < best_key || (key == best_key && known > best_known)) {
            best_key = key;
            best_known = known;
            best.clear();
        }
        if (key == best_key && known == best_known) best.push_back(physical);
    }
    const int discarded = best.at(static_cast<std::size_t>(rng.uniform(best.size())));
    hand.erase(std::find(hand.begin(), hand.end(), discarded));
    return discarded;
}

TableTrialResult simulate_table_push_fold_trial(
    const Config& config, std::uint64_t trial_id, ShantenCalculator& calculator
) {
    constexpr std::size_t MIN_LIVE_TILES_FOR_RIICHI = 4;
    TableTrialResult table;
    TrialResult& result = table.winner;
    result.trial_id = trial_id;
    result.seed = trial_seed(config.seed, trial_id);
    Rng wall_rng(splitmix64(result.seed ^ 0x57414C4CULL));
    std::array<std::unique_ptr<Rng>, 4> rngs;
    for (int seat = 0; seat < 4; ++seat) rngs[seat] = std::make_unique<Rng>(
        splitmix64(result.seed ^ 0x504F4C494359ULL ^ splitmix64(seat))
    );
    std::vector<int> wall(PHYSICAL_TILES);
    std::iota(wall.begin(), wall.end(), 0);
    wall_rng.shuffle(wall);
    std::array<std::vector<int>, 4> hands, rivers, initial;
    for (int seat = 0; seat < 4; ++seat) {
        hands[seat].assign(wall.begin() + seat * INITIAL_HAND, wall.begin() + (seat + 1) * INITIAL_HAND);
        initial[seat] = hands[seat];
    }
    const int dead_start = 4 * INITIAL_HAND;
    result.dora_indicator = wall[dead_start];
    result.ura_indicator = wall[dead_start + 1];
    std::vector<int> live(wall.begin() + dead_start + DEAD_WALL, wall.end()), public_river;
    result.initial_physical = initial[0];
    std::sort(result.initial_physical.begin(), result.initial_physical.end());
    result.initial_hand = format_counts(counts_from_physical(initial[0]));

    PushFoldResult payload;
    payload.mode = config.push_fold_mode;
    payload.safety_strategy = config.safety_strategy;
    payload.dora_danger_steps = config.dora_danger_steps;
    payload.half_push_max_tier = config.half_push_max_tier;
    payload.two_riichi_bad_wait_min_ron = config.two_riichi_bad_wait_min_ron;
    std::array<int, 4> turns{}, reversals_used{};
    std::array<bool, 4> riichi{}, sticky_fold{}, valuable_lock{}, haneman_lock{};
    std::array<bool, 4> temporary_furiten{};
    std::array<std::uint64_t, 4> wait_masks{}, legal_ron_masks{};
    std::array<std::array<bool, TILE_TYPES>, 4> safe{};
    int riichi_sticks = 0;
    std::size_t cursor = 0;
    const int dora_type = dora_type_from_indicator(result.dora_indicator / 4);

    auto public_counts_now = [&]() {
        Counts counts{};
        for (int tile : public_river) ++counts[tile / 4];
        ++counts[result.dora_indicator / 4];
        return counts;
    };
    auto known_for = [&](int seat, const std::vector<int>& full) {
        (void)seat;
        return push_fold_known(full, public_river, result.dora_indicator);
    };
    auto score_post = [&](int seat, int turn, const std::vector<int>& post,
                          const std::vector<int>& full_known) {
        return score_service_request(
            config, trial_id, seat, turn, post, full_known, result.dora_indicator
        );
    };
    auto refresh_tenpai = [&](int seat, const ScoreReply* scored) {
        const Counts counts = counts_from_physical(hands[seat]);
        wait_masks[seat] = 0;
        legal_ron_masks[seat] = 0;
        if (calculator.calculate(counts) != 0) return;
        for (int tile : structural_ukeire_types(counts, calculator)) {
            wait_masks[seat] |= std::uint64_t{1} << tile;
        }
        if (riichi[seat]) legal_ron_masks[seat] = wait_masks[seat];
        else if (scored) legal_ron_masks[seat] = scored->legal_ron_mask;
    };
    auto permanent_furiten = [&](int seat) {
        return wait_masks[seat] && is_permanent_furiten(
            counts_from_physical(hands[seat]), rivers[seat], calculator
        );
    };
    auto assert_partition = [&]() {
        std::array<bool, PHYSICAL_TILES> seen{};
        int total = 0;
        auto add = [&](int tile) {
            if (tile < 0 || tile >= PHYSICAL_TILES || seen[tile])
                throw std::logic_error("push-fold physical partition failed");
            seen[tile] = true;
            ++total;
        };
        for (const auto& hand : hands) for (int tile : hand) add(tile);
        for (int tile : public_river) add(tile);
        for (int index = 0; index < DEAD_WALL; ++index) add(wall[dead_start + index]);
        for (std::size_t index = cursor; index < live.size(); ++index) add(live[index]);
        if (total != PHYSICAL_TILES) throw std::logic_error("push-fold physical partition incomplete");
    };
    auto finish = [&](const std::string& reason, std::vector<PushFoldWinner> winners,
                      int loser) -> TableTrialResult {
        assert_partition();
        payload.terminal.reason = reason;
        payload.terminal.winners = std::move(winners);
        payload.terminal.loser = loser;
        payload.terminal.riichi_sticks = riichi_sticks;
        for (int seat = 0; seat < 4; ++seat) {
            payload.terminal.final_hand_counts[seat] = counts_from_physical(hands[seat]);
            if (calculator.calculate(payload.terminal.final_hand_counts[seat]) == 0)
                payload.terminal.tenpai_seats.push_back(seat);
        }
        result.push_fold = std::move(payload);
        result.outcome = reason;
        return table;
    };

    // Initial tenpai can ron before its first draw, but cannot declare riichi.
    for (int seat = 0; seat < 4; ++seat) {
        if (calculator.calculate(counts_from_physical(hands[seat])) != 0) continue;
        std::vector<int> known = hands[seat];
        known.push_back(result.dora_indicator);
        const ScoreReply score = score_post(seat, 0, hands[seat], known);
        refresh_tenpai(seat, &score);
        valuable_lock[seat] = score.valuable_dama;
        haneman_lock[seat] = score.haneman_lock;
        if (haneman_lock[seat]) legal_ron_masks[seat] = score.haneman_ron_mask;
        else if (valuable_lock[seat]) legal_ron_masks[seat] = score.valuable_ron_mask;
    }

    while (cursor < live.size()) {
        const int seat = static_cast<int>(cursor % 4);
        ++turns[seat];
        temporary_furiten[seat] = false;
        const int drawn = live[cursor++];
        hands[seat].push_back(drawn);
        Counts full_counts = counts_from_physical(hands[seat]);
        if (calculator.calculate(full_counts) == -1) {
            PushFoldWinner winner;
            winner.seat = seat;
            winner.hand_physical = hands[seat];
            std::sort(winner.hand_physical.begin(), winner.hand_physical.end());
            winner.winning_tile = drawn;
            winner.is_riichi = riichi[seat];
            winner.is_ippatsu = false;
            winner.is_haitei = cursor == live.size();
            return finish("tsumo", {std::move(winner)}, -1);
        }

        std::vector<int> threats;
        for (int other = 0; other < 4; ++other) if (other != seat && riichi[other]) threats.push_back(other);
        PushFoldDecision row;
        row.seat = seat;
        row.turn = turns[seat];
        row.draw_index = static_cast<int>(cursor);
        row.drawn_physical = drawn;
        row.live_wall_remaining = static_cast<int>(live.size() - cursor);
        row.own_riichi = riichi[seat];
        row.threat_seats = threats;
        row.own_counts = full_counts;
        row.public_counts = public_counts_now();
        row.best_shanten = push_fold_best_shanten(hands[seat], calculator);
        for (int threat : threats) {
            PushFoldSafeTypes entry;
            entry.threat_seat = threat;
            for (int tile = 0; tile < TILE_TYPES; ++tile) if (safe[threat][tile]) entry.types.push_back(tile);
            row.safe_types.push_back(std::move(entry));
        }

        auto ron_hits_for_type = [&](int tile_type) {
            std::vector<int> hits;
            for (int other = 0; other < 4; ++other) {
                if (other == seat || !(wait_masks[other] & (std::uint64_t{1} << tile_type))) continue;
                const bool houtei_yaku = cursor == live.size();
                if (!houtei_yaku
                    && !(legal_ron_masks[other] & (std::uint64_t{1} << tile_type))) continue;
                if (temporary_furiten[other] || permanent_furiten(other)) continue;
                hits.push_back(other);
            }
            return hits;
        };

        // Record one ordinary representative and the red representative when
        // both are held. Downstream 34-type analytics explicitly deduplicate.
        std::map<std::pair<int, int>, int> representatives;
        for (int physical : hands[seat]) {
            const int red_class = is_red_five(physical) ? 1 : 0;
            auto key = std::make_pair(physical / 4, red_class);
            if (!representatives.count(key)) representatives[key] = physical;
        }
        for (const auto& item : representatives) {
            PushFoldCandidate candidate;
            candidate.physical_id = item.second;
            candidate.tile_type = item.second / 4;
            for (int threat : threats) candidate.safety_tiers.push_back(adjusted_safety_tier(
                item.second, safe[threat], dora_type, config.dora_danger_steps
            ));
            candidate.known_copies = row.own_counts[candidate.tile_type] + row.public_counts[candidate.tile_type];
            candidate.ron_hits = ron_hits_for_type(candidate.tile_type);
            row.candidates.push_back(std::move(candidate));
        }

        int discarded = -1;
        PushFoldChoice choice;
        std::vector<int> min_shanten_candidates;
        for (int physical : hands[seat]) {
            std::vector<int> post = hands[seat];
            post.erase(std::find(post.begin(), post.end(), physical));
            if (calculator.calculate(counts_from_physical(post)) == row.best_shanten)
                min_shanten_candidates.push_back(physical);
        }
        const std::vector<int> known = known_for(seat, hands[seat]);
        auto score_candidate = [&](int physical) {
            std::vector<int> post = hands[seat];
            post.erase(std::find(post.begin(), post.end(), physical));
            return score_post(seat, turns[seat], post, known);
        };

        if (riichi[seat]) {
            row.mode = "full_push";
            row.mode_reason = "riichi_lock";
            discarded = drawn;
            hands[seat].pop_back();
        } else if (haneman_lock[seat]) {
            row.mode = "full_push";
            row.mode_reason = "haneman_dama_lock";
            discarded = drawn;
            hands[seat].pop_back();
        } else {
            std::string mode = "full_push";
            std::string reason = threats.empty() ? "no_threat" : "configured_all_push";
            std::vector<int> qualified = min_shanten_candidates;
            if (sticky_fold[seat]) {
                mode = "full_fold";
                reason = "sticky_fold";
            } else if (!threats.empty() && config.push_fold_mode == "full_fold") {
                mode = "full_fold";
                reason = "configured_full_fold";
                sticky_fold[seat] = true;
            } else if (!threats.empty() && config.push_fold_mode == "mixed") {
                if (threats.size() >= 3) {
                    mode = "full_fold"; reason = "three_threats"; sticky_fold[seat] = true;
                } else if (threats.size() == 1) {
                    if (row.best_shanten >= 2) {
                        mode = "full_fold"; reason = "one_threat_two_shanten"; sticky_fold[seat] = true;
                    } else if (row.best_shanten == 1) {
                        mode = "half_push"; reason = "one_threat_one_shanten";
                    } else {
                        std::vector<int> probe_hand = hands[seat];
                        Rng probe_rng = *rngs[seat];
                        PushFoldChoice baseline = choose_push_efficiency(
                            probe_hand, public_river, result.dora_indicator, config, seat,
                            min_shanten_candidates, probe_rng, calculator
                        );
                        const ScoreReply score = score_candidate(baseline.discarded);
                        if (!score.good_wait && score.cheap_all_unseen) {
                            mode = "half_push"; reason = "one_threat_cheap_bad_wait";
                        } else {
                            mode = "full_push";
                            reason = score.good_wait ? "one_threat_good_wait" :
                                score.valuable_dama ? "one_threat_valuable_dama" : "one_threat_medium_bad_wait";
                        }
                    }
                } else {
                    if (row.best_shanten != 0) {
                        mode = "full_fold"; reason = "two_threats_not_tenpai"; sticky_fold[seat] = true;
                    } else {
                        qualified.clear();
                        for (int physical : min_shanten_candidates) {
                            const ScoreReply score = score_candidate(physical);
                            const int intended_min = two_threat_intended_min_ron(
                                score, live.size() - cursor, MIN_LIVE_TILES_FOR_RIICHI
                            );
                            if (score.good_wait || intended_min >= config.two_riichi_bad_wait_min_ron)
                                qualified.push_back(physical);
                        }
                        if (qualified.empty()) {
                            mode = "full_fold"; reason = "two_threats_unqualified_tenpai"; sticky_fold[seat] = true;
                        } else {
                            mode = "full_push"; reason = "two_threats_qualified_tenpai";
                        }
                    }
                }
            }
            row.mode = mode;
            row.mode_reason = reason;
            if (mode == "full_fold") {
                discarded = choose_full_fold(
                    hands[seat], threats, safe, row.public_counts, dora_type, config, *rngs[seat]
                );
            } else {
                if (mode == "half_push") {
                    qualified.clear();
                    for (int physical : min_shanten_candidates) {
                        bool eligible = true;
                        for (int threat : threats) {
                            eligible &= adjusted_safety_tier(
                                physical, safe[threat], dora_type, config.dora_danger_steps
                            ) <= config.half_push_max_tier;
                        }
                        if (eligible) qualified.push_back(physical);
                    }
                    if (qualified.empty()) {
                        row.mode = "full_fold";
                        row.mode_reason += "_failed_safety";
                        sticky_fold[seat] = true;
                        discarded = choose_full_fold(
                            hands[seat], threats, safe, row.public_counts, dora_type, config, *rngs[seat]
                        );
                    }
                }
                if (discarded < 0) {
                    if (valuable_lock[seat]) {
                        std::vector<int> guarded;
                        for (int physical : qualified) {
                            const ScoreReply score = score_candidate(physical);
                            if (score.good_wait || score.valuable_dama || score.haneman_lock)
                                guarded.push_back(physical);
                        }
                        if (guarded.empty()) {
                            if (row.mode == "half_push") {
                                row.mode = "full_fold";
                                row.mode_reason = "valuable_dama_transition_guard_fold";
                                sticky_fold[seat] = true;
                                discarded = choose_full_fold(
                                    hands[seat], threats, safe, row.public_counts,
                                    dora_type, config, *rngs[seat]
                                );
                            } else {
                                discarded = drawn;
                                hands[seat].pop_back();
                                row.mode_reason = "valuable_dama_transition_guard";
                            }
                        } else qualified = std::move(guarded);
                    }
                    if (discarded < 0) {
                        const bool unrestricted_full_push = row.mode == "full_push"
                            && (threats.empty() || row.best_shanten > 0) && !valuable_lock[seat]
                            && qualified.size() == min_shanten_candidates.size();
                        if (unrestricted_full_push) {
                            const auto legacy = choose_discard(
                                hands[seat], public_river, result.dora_indicator, 27 + seat,
                                config.round_wind, config.future, config.improvement_weight,
                                config.policy, config.reversal, turns[seat], reversals_used[seat],
                                *rngs[seat], calculator
                            );
                            discarded = legacy.discarded;
                            choice.features = legacy.features;
                            if (legacy.features.is_reversal) ++reversals_used[seat];
                        } else {
                            choice = choose_push_efficiency(
                                hands[seat], public_river, result.dora_indicator, config, seat,
                                qualified, *rngs[seat], calculator
                            );
                            discarded = choice.discarded;
                        }
                    }
                }
            }
        }

        row.post_discard_shanten = calculator.calculate(counts_from_physical(hands[seat]));
        std::optional<ScoreReply> selected_score;
        if (row.post_discard_shanten == 0) {
            selected_score = score_post(seat, turns[seat], hands[seat], known);
            row.has_selected_score = true;
            row.selected_score = *selected_score;
        }
        auto selected_candidate = std::find_if(row.candidates.begin(), row.candidates.end(),
            [&](const PushFoldCandidate& candidate) { return candidate.physical_id == discarded; });
        if (selected_candidate == row.candidates.end()) {
            selected_candidate = std::find_if(row.candidates.begin(), row.candidates.end(),
                [&](const PushFoldCandidate& candidate) {
                    return candidate.tile_type == discarded / 4
                        && is_red_five(candidate.physical_id) == is_red_five(discarded);
                });
        }
        if (selected_candidate == row.candidates.end())
            throw std::logic_error("chosen push-fold candidate missing from observations");
        selected_candidate->chosen = true;
        payload.decisions.push_back(std::move(row));

        const int discard_type = discarded / 4;
        std::vector<int> ron_winner_seats = ron_hits_for_type(discard_type);
        public_river.push_back(discarded);
        rivers[seat].push_back(discarded);
        if (!ron_winner_seats.empty()) {
            std::vector<PushFoldWinner> winners;
            for (int winner_seat : ron_winner_seats) {
                PushFoldWinner winner;
                winner.seat = winner_seat;
                winner.hand_physical = hands[winner_seat];
                winner.hand_physical.push_back(discarded);
                std::sort(winner.hand_physical.begin(), winner.hand_physical.end());
                winner.winning_tile = discarded;
                winner.is_riichi = riichi[winner_seat];
                winner.is_ippatsu = false;
                winner.is_haitei = cursor == live.size();
                winners.push_back(std::move(winner));
            }
            return finish(winners.size() > 1 ? "multiple_ron" : "ron", std::move(winners), seat);
        }

        // A safely passed discard is genbutsu against every riichi already in force.
        for (int threat = 0; threat < 4; ++threat) if (riichi[threat]) safe[threat][discard_type] = true;

        refresh_tenpai(seat, selected_score ? &*selected_score : nullptr);
        if (!riichi[seat] && selected_score && payload.decisions.back().mode == "full_push") {
            if (selected_score->haneman_lock) {
                haneman_lock[seat] = true;
                valuable_lock[seat] = false;
                legal_ron_masks[seat] = selected_score->haneman_ron_mask;
            } else if (selected_score->valuable_dama) {
                valuable_lock[seat] = true;
                legal_ron_masks[seat] = selected_score->valuable_ron_mask;
            } else if (selected_score->exact_1300_bad_wait) {
                valuable_lock[seat] = false;
                legal_ron_masks[seat] = selected_score->legal_ron_mask;
            } else if (live.size() - cursor >= MIN_LIVE_TILES_FOR_RIICHI) {
                riichi[seat] = true;
                ++riichi_sticks;
                payload.riichi_seats.push_back(seat);
                for (int tile : rivers[seat]) safe[seat][tile / 4] = true;
                legal_ron_masks[seat] = wait_masks[seat];
                PushFoldRiichiEvent event;
                event.seat = seat;
                event.turn = turns[seat];
                event.hand_physical13 = hands[seat];
                std::sort(event.hand_physical13.begin(), event.hand_physical13.end());
                const Counts observer_known = counts_from_physical(hands[seat]);
                const Counts public_known = public_counts_now();
                Counts live_counts{};
                for (std::size_t index = cursor; index < live.size(); ++index)
                    ++live_counts[live[index] / 4];
                for (int tile = 0; tile < TILE_TYPES; ++tile) {
                    if (!(wait_masks[seat] & (std::uint64_t{1} << tile))) continue;
                    event.winning_types.push_back(tile);
                    event.unseen_wait_copies += std::max(
                        0, 4 - int(observer_known[tile]) - int(public_known[tile])
                    );
                    event.live_wait_copies += live_counts[tile];
                }
                event.live_wall_remaining = static_cast<int>(live.size() - cursor);
                event.furiten = permanent_furiten(seat);
                payload.riichi_events.push_back(std::move(event));
            }
        }
        if (riichi[seat]) legal_ron_masks[seat] = wait_masks[seat];
        else if (haneman_lock[seat] && selected_score) legal_ron_masks[seat] = selected_score->haneman_ron_mask;
        else if (valuable_lock[seat] && selected_score) legal_ron_masks[seat] = selected_score->valuable_ron_mask;
        assert_partition();
    }
    return finish("exhaustive_draw", {}, -1);
}
