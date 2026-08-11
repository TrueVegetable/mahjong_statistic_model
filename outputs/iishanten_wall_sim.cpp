#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// High-throughput C++ implementation of outputs/iishanten_wall_sim.py.
// The model is intentionally local: reserved blocks are legal physical tiles,
// while filler tiles are not parsed as complete hands.  The action point is
// South's ninth discard, before that discard is placed in South's river.

namespace {

constexpr int TILE_TYPES = 34;
constexpr int COPIES = 4;
constexpr int INITIAL_TURN = 9;
constexpr int LIVE_WALL_TILES = 36;
constexpr int HERO_UNSEEN_POSITIONS = 88;
constexpr int DEAD_WALL_TILES = 14;
constexpr double DEALER_RON_LOSS = 7500.0;
constexpr double DEALER_TSUMO_PAYMENT = 3400.0;
constexpr double DEALER_IPPATSU_RON_LOSS = 10900.0;
constexpr double DEALER_IPPATSU_TSUMO_PAYMENT = 4600.0;
constexpr double SIDE_RON_LOSS = 4345.0;
constexpr double SIDE_TSUMO_PAYMENT = 4767.0 / 4.0;
constexpr double RIICHI_SIDE_RON_LOSS = 5300.0;
constexpr double RIICHI_SIDE_TSUMO_PAYMENT = 1800.0;
constexpr double HERO_RIICHI_COST = 1000.0;
constexpr double FOLD_IMPERFECT_TSUMOGIRI = 0.02;
constexpr double DEFAULT_SIDE_RIICHI_ON_TENPAI = 0.50;

enum PlayerId { HERO = 0, DEALER = 1, WEST = 2, NORTH = 3 };
enum WaitKind { RYANMEN = 0, KANCHAN = 1, PENCHAN = 2, SHANPON = 3, TANKI = 4 };

const std::array<std::string, 5> WAIT_NAMES = {"ryanmen", "kanchan", "penchan", "shanpon", "tanki"};
const std::array<std::string, 4> PLAYER_NAMES = {"hero", "dealer", "side1", "side2"};
const std::array<std::string, 3> SHAPE_NAMES = {"ryanmen_ryanmen", "ryanmen_ryankan", "ryanmen_kanchan"};
const std::array<std::string, 6> ATTACK_STRATEGIES = {"suji37", "nosuji2378", "allout", "fixed05", "fixed10", "fixed15"};
const std::array<std::string, 4> CURRENT_CATEGORIES = {"genbutsu", "suji37", "nosuji2378", "nosuji456"};

int tile(int suit, int rank) {
    if (suit == 3) return 27 + rank - 1;
    return suit * 9 + rank - 1;
}

int suit_of(int t) { return (t >= 0 && t < 27) ? t / 9 : -1; }
int rank_of(int t) { return (t >= 0 && t < 27) ? t % 9 + 1 : -1; }

std::string tile_name(int t) {
    if (t >= 27) return std::to_string(t - 26) + "z";
    const char suits[] = {'m', 'p', 's'};
    return std::to_string(rank_of(t)) + suits[suit_of(t)];
}

struct WaitTemplate {
    int kind = RYANMEN;
    std::vector<int> winners;
    std::vector<int> reserved_tiles;
    int owner = DEALER;
    int suit = -1;
    std::string source;

    bool valid() const {
        std::array<int, TILE_TYPES> counts{};
        for (int t : reserved_tiles) {
            if (t < 0 || t >= TILE_TYPES) return false;
            ++counts[t];
        }
        if (kind == RYANMEN) {
            if (suit < 0 || winners.size() != 2 || reserved_tiles.size() != 2) return false;
            std::vector<int> w = winners;
            std::sort(w.begin(), w.end());
            return suit_of(w[0]) == suit && suit_of(w[1]) == suit && w[1] - w[0] == 3 &&
                   counts[w[0] + 1] == 1 && counts[w[0] + 2] == 1 &&
                   std::accumulate(counts.begin(), counts.end(), 0) == 2;
        }
        if (kind == KANCHAN) {
            if (suit < 0 || winners.size() != 1 || reserved_tiles.size() != 2) return false;
            const int w = winners[0];
            return suit_of(w) == suit && rank_of(w) >= 2 && rank_of(w) <= 8 &&
                   counts[w - 1] == 1 && counts[w + 1] == 1 &&
                   std::accumulate(counts.begin(), counts.end(), 0) == 2;
        }
        if (kind == PENCHAN) {
            if (suit < 0 || winners.size() != 1 || reserved_tiles.size() != 2) return false;
            const int w = winners[0];
            const bool left = rank_of(w) == 3 && counts[w - 2] == 1 && counts[w - 1] == 1;
            const bool right = rank_of(w) == 7 && counts[w + 1] == 1 && counts[w + 2] == 1;
            return suit_of(w) == suit && (left || right) &&
                   std::accumulate(counts.begin(), counts.end(), 0) == 2;
        }
        if (kind == SHANPON) {
            if (winners.size() != 2 || reserved_tiles.size() != 4) return false;
            return winners[0] != winners[1] && counts[winners[0]] == 2 && counts[winners[1]] == 2 &&
                   std::accumulate(counts.begin(), counts.end(), 0) == 4;
        }
        if (kind == TANKI) {
            if (winners.size() != 1 || reserved_tiles.size() != 1) return false;
            return counts[winners[0]] == 1;
        }
        return false;
    }
};

struct Plan {
    int improvement = -1;
    WaitTemplate wait;
    int entry_discard = -1;
};

struct IishantenTemplate {
    std::string key;
    std::string label;
    int owner = HERO;
    std::vector<int> reserved_tiles;
    std::vector<Plan> plans;
};

struct Weights {
    std::unordered_map<std::string, double> kind;
    std::unordered_map<std::string, double> position;
    std::unordered_map<std::string, double> internal;
    std::unordered_map<std::string, double> location;
    double regularization = 0.0001;
    std::string source = "fixed-common-random full-state maximum-entropy fit";
};

double map_value(const std::unordered_map<std::string, double>& values, const std::string& key) {
    const auto it = values.find(key);
    return it == values.end() ? 0.0 : it->second;
}

std::string tile_class(int t);

std::string wait_position(const WaitTemplate& wait) {
    int w = *std::min_element(wait.winners.begin(), wait.winners.end());
    if (w >= 27) return "honor";
    const int r = rank_of(w);
    if (r == 1 || r == 9) return "terminal";
    if (r == 2 || r == 8) return "outer";
    return "middle";
}

std::string template_location(const WaitTemplate& wait) {
    if (wait.kind == RYANMEN) {
        const auto slash = wait.source.find('/');
        const int first = std::stoi(wait.source.substr(0, slash));
        return "ryanmen_" + std::to_string(std::min(first, 7 - first));
    }
    if (wait.kind == KANCHAN) {
        const int r = std::stoi(wait.source);
        return "kanchan_" + std::to_string(std::min(r, 10 - r));
    }
    return WAIT_NAMES[wait.kind];
}

double template_weight(const WaitTemplate& wait, const Weights& weights) {
    double log_weight = map_value(weights.kind, WAIT_NAMES[wait.kind]);
    if (wait.kind == RYANMEN) {
        const auto slash = wait.source.find('/');
        const int first = std::stoi(wait.source.substr(0, slash));
        const int group = std::min(first, 7 - first);
        if (group != 1) log_weight += map_value(weights.location, "ryanmen:" + std::to_string(group));
    } else if (wait.kind == KANCHAN) {
        const int rank = std::stoi(wait.source);
        const int group = std::min(rank, 10 - rank);
        if (group != 2) log_weight += map_value(weights.location, "kanchan:" + std::to_string(group));
    } else if (wait.kind == TANKI) {
        const std::string klass = tile_class(wait.winners.front());
        if (klass != "honor") log_weight += map_value(weights.location, "tanki:" + klass);
    } else if (wait.kind == SHANPON) {
        for (int winner : wait.winners) {
            const std::string klass = tile_class(winner);
            if (klass != "honor") log_weight += map_value(weights.location, "shanpon:" + klass);
        }
    }
    return std::exp(log_weight);
}

std::vector<WaitTemplate> enumerate_wait_templates() {
    std::vector<WaitTemplate> result;
    result.reserve(640);
    for (int s = 0; s < 3; ++s) {
        for (int first = 1; first <= 6; ++first) {
            const int low = tile(s, first);
            const int high = tile(s, first + 3);
            result.push_back({RYANMEN, {low, high}, {low + 1, low + 2}, DEALER, s,
                               std::to_string(first) + "/" + std::to_string(first + 3)});
        }
        for (int r = 2; r <= 8; ++r) {
            const int w = tile(s, r);
            result.push_back({KANCHAN, {w}, {w - 1, w + 1}, DEALER, s, std::to_string(r)});
        }
        result.push_back({PENCHAN, {tile(s, 3)}, {tile(s, 1), tile(s, 2)}, DEALER, s, "3"});
        result.push_back({PENCHAN, {tile(s, 7)}, {tile(s, 8), tile(s, 9)}, DEALER, s, "7"});
    }
    for (int first = 0; first < TILE_TYPES; ++first) {
        for (int second = first + 1; second < TILE_TYPES; ++second) {
            result.push_back({SHANPON, {first, second}, {first, first, second, second}, DEALER, -1,
                               tile_name(first) + "/" + tile_name(second)});
        }
    }
    for (int w = 0; w < TILE_TYPES; ++w) {
        result.push_back({TANKI, {w}, {w}, DEALER, suit_of(w), tile_name(w)});
    }
    for (const auto& wait : result) {
        if (!wait.valid()) throw std::runtime_error("internal illegal wait template");
    }
    return result;
}

const std::vector<WaitTemplate>& all_wait_templates() {
    static const std::vector<WaitTemplate> templates = enumerate_wait_templates();
    return templates;
}

struct TemplateSampler {
    const std::vector<WaitTemplate>& templates;
    std::vector<double> masses;
    std::array<std::vector<int>, 5> by_kind;
    std::vector<int> all_indices;

    explicit TemplateSampler(const Weights& weights) : templates(all_wait_templates()), masses(templates.size()) {
        all_indices.reserve(templates.size());
        for (std::size_t i = 0; i < templates.size(); ++i) {
            masses[i] = template_weight(templates[i], weights);
            all_indices.push_back(static_cast<int>(i));
            by_kind[templates[i].kind].push_back(static_cast<int>(i));
        }
    }
};

struct Rng {
    std::mt19937_64 engine;
    explicit Rng(std::uint64_t seed) : engine(seed) {}

    std::uint64_t next() { return engine(); }

    double unit() {
        return static_cast<double>(engine() >> 11) * (1.0 / 9007199254740992.0);
    }

    int index(int n) {
        if (n <= 1) return 0;
        return static_cast<int>(engine() % static_cast<std::uint64_t>(n));
    }
};

bool forbidden_winner(const WaitTemplate& wait, const std::array<bool, TILE_TYPES>& forbidden) {
    for (int t : wait.winners) if (forbidden[t]) return true;
    return false;
}

bool reserved_fits(const WaitTemplate& wait, const std::array<int, TILE_TYPES>& used) {
    std::array<int, TILE_TYPES> local{};
    for (int t : wait.reserved_tiles) {
        ++local[t];
        if (used[t] + local[t] > COPIES) return false;
    }
    return true;
}

std::optional<WaitTemplate> sample_wait_template(
    Rng& rng, int owner, const TemplateSampler& sampler, int required_kind,
    const std::array<bool, TILE_TYPES>& forbidden, const std::array<int, TILE_TYPES>& used) {
    const auto& candidates = required_kind >= 0 ? sampler.by_kind[required_kind] : sampler.all_indices;
    double total = 0.0;
    for (int index : candidates) {
        const auto& wait = sampler.templates[index];
        if (forbidden_winner(wait, forbidden) || !reserved_fits(wait, used)) continue;
        total += sampler.masses[index];
    }
    if (!(total > 0.0)) return std::nullopt;
    double target = rng.unit() * total;
    for (int index : candidates) {
        const auto& wait = sampler.templates[index];
        if (forbidden_winner(wait, forbidden) || !reserved_fits(wait, used)) continue;
        target -= sampler.masses[index];
        if (target <= 0.0) {
            WaitTemplate chosen = wait;
            chosen.owner = owner;
            return chosen;
        }
    }
    return std::nullopt;
}

std::optional<IishantenTemplate> make_iishanten_template(
    Rng& rng, const std::string& key, int owner, const std::array<bool, TILE_TYPES>& forbidden,
    const TemplateSampler& sampler) {
    std::vector<int> kinds;
    if (key == "ryanmen_ryanmen") kinds = {RYANMEN, RYANMEN};
    else if (key == "ryanmen_ryankan") kinds = {RYANMEN, KANCHAN, KANCHAN};
    else if (key == "ryanmen_kanchan") kinds = {RYANMEN, KANCHAN};
    else return std::nullopt;

    std::array<int, TILE_TYPES> used{};
    std::vector<WaitTemplate> blocks;
    blocks.reserve(kinds.size());
    for (int kind : kinds) {
        auto block = sample_wait_template(rng, owner, sampler, kind, forbidden, used);
        if (!block) return std::nullopt;
        for (int t : block->reserved_tiles) ++used[t];
        blocks.push_back(*block);
    }
    IishantenTemplate result;
    result.key = key;
    result.owner = owner;
    result.label = key == "ryanmen_ryanmen" ? "两面＋两面" :
                   key == "ryanmen_ryankan" ? "两面＋两嵌" : "两面＋嵌张";
    for (int t = 0; t < TILE_TYPES; ++t) {
        for (int n = 0; n < used[t]; ++n) result.reserved_tiles.push_back(t);
    }
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const WaitTemplate& target = blocks[(i + 1) % blocks.size()];
        for (int improvement : blocks[i].winners) result.plans.push_back({improvement, target, -1});
    }
    return result;
}

std::array<bool, TILE_TYPES> bool_set(const std::vector<int>& tiles) {
    std::array<bool, TILE_TYPES> result{};
    for (int t : tiles) if (t >= 0 && t < TILE_TYPES) result[t] = true;
    return result;
}

std::vector<int> future_wait_winners(const IishantenTemplate& templ) {
    std::array<bool, TILE_TYPES> seen{};
    std::vector<int> result;
    for (const auto& plan : templ.plans) {
        for (int t : plan.wait.winners) {
            if (!seen[t]) {
                seen[t] = true;
                result.push_back(t);
            }
        }
    }
    return result;
}

bool bind_entry_discards(IishantenTemplate& templ, const std::vector<int>& hand,
                         const std::array<bool, TILE_TYPES>& unavailable = {}) {
    std::array<int, TILE_TYPES> reserved{};
    for (int t : templ.reserved_tiles) ++reserved[t];
    std::vector<int> ordered = hand;
    std::sort(ordered.begin(), ordered.end());
    for (auto& plan : templ.plans) {
        int chosen = -1;
        for (int t : ordered) {
            if (!unavailable[t] && !std::binary_search(plan.wait.winners.begin(), plan.wait.winners.end(), t) && reserved[t] == 0) {
                chosen = t;
                break;
            }
        }
        if (chosen < 0) {
            for (int t : ordered) {
                if (!unavailable[t] && !std::binary_search(plan.wait.winners.begin(), plan.wait.winners.end(), t)) {
                    chosen = t;
                    break;
                }
            }
        }
        if (chosen < 0) return false;
        plan.entry_discard = chosen;
    }
    return true;
}

struct Player {
    int id = HERO;
    std::vector<int> hand;
    std::vector<int> river;
    std::optional<WaitTemplate> wait;
    std::optional<IishantenTemplate> iishanten;
    char state = 'F';
    bool riichi = false;
    bool riichi_paid = false;
    std::array<bool, TILE_TYPES> genbutsu{};
    std::array<bool, TILE_TYPES> river_seen{};
    bool folded = false;

    bool furiten() const {
        if (!wait) return false;
        for (int t : wait->winners) {
            if (std::find(river.begin(), river.end(), t) != river.end()) return true;
        }
        return false;
    }
};

struct State {
    std::array<Player, 4> players;
    int dora_indicator = -1;
    std::vector<int> ura_dora;
    std::vector<int> live_wall;
    std::size_t live_index = 0;
    int current_candidate = -1;
    std::string current_category;
    IishantenTemplate hero_template;
    int riichi_sticks = 1;
};

struct Counts {
    std::array<int, TILE_TYPES> count{};
    Counts() { count.fill(COPIES); }
};

bool take_tile(Counts& counts, int t) {
    if (t < 0 || t >= TILE_TYPES || counts.count[t] <= 0) return false;
    --counts.count[t];
    return true;
}

std::optional<int> take_random_type(Counts& counts, Rng& rng, const std::array<bool, TILE_TYPES>& forbidden = {}) {
    std::array<int, TILE_TYPES> candidates{};
    int n = 0;
    for (int t = 0; t < TILE_TYPES; ++t) {
        if (counts.count[t] > 0 && !forbidden[t]) candidates[n++] = t;
    }
    if (n == 0) return std::nullopt;
    const int chosen = candidates[rng.index(n)];
    --counts.count[chosen];
    return chosen;
}

struct Anchors {
    std::array<int, 2> tile = {-1, -1};
    int count = 0;
};

Anchors suji_anchors(int t) {
    Anchors result;
    if (t >= 27 || t < 0) return result;
    const int s = suit_of(t);
    const int r = rank_of(t);
    if (r - 3 >= 1) result.tile[result.count++] = tile(s, r - 3);
    if (r + 3 <= 9) result.tile[result.count++] = tile(s, r + 3);
    return result;
}

int candidate_for(const std::string& category, std::optional<int> override_tile) {
    if (override_tile) return *override_tile;
    return category == "nosuji456" ? tile(1, 4) : tile(1, 3);
}

bool category_holds(const std::vector<int>& dealer_river, int candidate, const std::string& category) {
    std::array<bool, TILE_TYPES> seen{};
    for (int t : dealer_river) seen[t] = true;
    const auto anchors = suji_anchors(candidate);
    if (category == "genbutsu") return seen[candidate];
    if (category == "suji37") {
        for (int i = 0; i < anchors.count; ++i) if (seen[anchors.tile[i]]) return !seen[candidate];
        return false;
    }
    if (category == "nosuji2378" || category == "nosuji456" || category == "fold") {
        if (seen[candidate]) return false;
        for (int i = 0; i < anchors.count; ++i) if (seen[anchors.tile[i]]) return false;
        return true;
    }
    return true;
}

bool add_river(Counts& counts, Rng& rng, Player& player, int length, const std::vector<int>& required,
               const std::array<bool, TILE_TYPES>& explicit_forbidden) {
    for (int t : required) {
        if (!take_tile(counts, t)) return false;
        player.river.push_back(t);
        player.river_seen[t] = true;
    }
    while (static_cast<int>(player.river.size()) < length) {
        auto forbidden = explicit_forbidden;
        if (player.wait) for (int t : player.wait->winners) forbidden[t] = true;
        auto drawn = take_random_type(counts, rng, forbidden);
        if (!drawn) return false;
        player.river.push_back(*drawn);
        player.river_seen[*drawn] = true;
    }
    return true;
}

char sample_side_state(Rng& rng) {
    const double roll = rng.unit();
    if (roll < 0.12) return 'T';
    if (roll < 0.35) return 'H';
    return 'F';
}

bool build_state_once(Rng& rng, const std::string& shape, const std::string& category,
                      std::optional<int> candidate_override, const TemplateSampler& sampler, State& state) {
    Counts counts;
    const int candidate = candidate_for(category, candidate_override);
    if (candidate < 0 || candidate >= TILE_TYPES) return false;

    auto hero_template = make_iishanten_template(rng, shape, HERO, bool_set({candidate}), sampler);
    if (!hero_template) return false;
    const auto hero_future = future_wait_winners(*hero_template);
    if (std::find(hero_future.begin(), hero_future.end(), candidate) != hero_future.end()) return false;

    State local;
    for (int id = 0; id < 4; ++id) local.players[id].id = id;
    local.current_candidate = candidate;
    local.current_category = category;
    local.hero_template = *hero_template;
    local.players[HERO].state = 'H';
    local.players[HERO].iishanten = *hero_template;
    local.players[HERO].hand = hero_template->reserved_tiles;
    for (int t : hero_template->reserved_tiles) if (!take_tile(counts, t)) return false;
    if (!take_tile(counts, candidate)) return false;
    local.players[HERO].hand.push_back(candidate);
    while (local.players[HERO].hand.size() < 14) {
        auto t = take_random_type(counts, rng);
        if (!t) return false;
        local.players[HERO].hand.push_back(*t);
    }

    auto dealer_wait = sample_wait_template(rng, DEALER, sampler, -1, {}, {});
    if (!dealer_wait) return false;
    local.players[DEALER].state = 'T';
    local.players[DEALER].riichi = true;
    local.players[DEALER].riichi_paid = true;
    local.players[DEALER].wait = *dealer_wait;
    local.players[DEALER].hand = dealer_wait->reserved_tiles;
    for (int t : dealer_wait->reserved_tiles) if (!take_tile(counts, t)) return false;

    for (int id : {WEST, NORTH}) {
        Player& player = local.players[id];
        player.state = sample_side_state(rng);
        if (player.state == 'T') {
            auto wait = sample_wait_template(rng, id, sampler, -1, {}, {});
            if (!wait) return false;
            player.wait = *wait;
            player.hand = wait->reserved_tiles;
            for (int t : wait->reserved_tiles) if (!take_tile(counts, t)) return false;
        } else if (player.state == 'H') {
            auto templ = make_iishanten_template(rng, "ryanmen_kanchan", id, {}, sampler);
            if (!templ) return false;
            player.iishanten = *templ;
            player.hand = templ->reserved_tiles;
            for (int t : templ->reserved_tiles) if (!take_tile(counts, t)) return false;
        }
    }

    std::vector<int> dealer_required;
    if (category == "genbutsu") dealer_required.push_back(candidate);
    else if (category == "suji37") {
        const auto anchors = suji_anchors(candidate);
        if (anchors.count == 0) return false;
        dealer_required.push_back(anchors.tile[0]);
    }
    if (!add_river(counts, rng, local.players[DEALER], 9, dealer_required, {})) return false;
    if (!add_river(counts, rng, local.players[HERO], 8, {}, bool_set(hero_future))) return false;

    for (int id : {WEST, NORTH}) {
        std::vector<int> forbidden;
        if (local.players[id].wait) forbidden = local.players[id].wait->winners;
        if (local.players[id].iishanten) {
            const auto future = future_wait_winners(*local.players[id].iishanten);
            forbidden.insert(forbidden.end(), future.begin(), future.end());
        }
        if (!add_river(counts, rng, local.players[id], 8, {}, bool_set(forbidden))) return false;
    }
    if (!category_holds(local.players[DEALER].river, candidate, category)) return false;
    for (const auto& player : local.players) if (player.furiten()) return false;

    auto dora = take_random_type(counts, rng);
    if (!dora) return false;
    local.dora_indicator = *dora;

    for (int id = 0; id < 4; ++id) {
        const std::size_t target = id == HERO ? 14 : 13;
        while (local.players[id].hand.size() < target) {
            auto t = take_random_type(counts, rng);
            if (!t) return false;
            local.players[id].hand.push_back(*t);
        }
    }
    if (!bind_entry_discards(*local.players[HERO].iishanten, local.players[HERO].hand, bool_set({candidate}))) return false;
    local.hero_template = *local.players[HERO].iishanten;
    for (int id : {WEST, NORTH}) {
        if (local.players[id].iishanten && !bind_entry_discards(*local.players[id].iishanten, local.players[id].hand)) return false;
    }

    local.ura_dora.reserve(13);
    for (int i = 0; i < 13; ++i) {
        auto t = take_random_type(counts, rng);
        if (!t) return false;
        local.ura_dora.push_back(*t);
    }
    local.live_wall.reserve(LIVE_WALL_TILES);
    for (int i = 0; i < LIVE_WALL_TILES; ++i) {
        auto t = take_random_type(counts, rng);
        if (!t) return false;
        local.live_wall.push_back(*t);
    }
    local.live_index = 0;
    state = std::move(local);
    return true;
}

bool physical_state_valid(const State& state, bool after_current = false) {
    if (state.players[HERO].hand.size() != (after_current ? 13u : 14u)) return false;
    if (state.players[DEALER].hand.size() != 13u || state.players[WEST].hand.size() != 13u || state.players[NORTH].hand.size() != 13u) return false;
    const std::array<std::size_t, 4> expected_before = {8, 9, 8, 8};
    const std::array<std::size_t, 4> expected_after = {9, 9, 8, 8};
    const auto& expected = after_current ? expected_after : expected_before;
    for (int id = 0; id < 4; ++id) if (state.players[id].river.size() != expected[id]) return false;
    if (state.ura_dora.size() != 13 || state.live_wall.size() - state.live_index != LIVE_WALL_TILES) return false;
    std::array<int, TILE_TYPES> counts{};
    for (const auto& player : state.players) {
        for (int t : player.hand) ++counts[t];
        for (int t : player.river) ++counts[t];
    }
    ++counts[state.dora_indicator];
    for (int t : state.ura_dora) ++counts[t];
    for (std::size_t i = state.live_index; i < state.live_wall.size(); ++i) ++counts[state.live_wall[i]];
    for (int t = 0; t < TILE_TYPES; ++t) if (counts[t] != COPIES) return false;
    if (std::find(state.players[HERO].hand.begin(), state.players[HERO].hand.end(), state.current_candidate) == state.players[HERO].hand.end() && !after_current) return false;
    for (const auto& player : state.players) if (player.furiten()) return false;
    return true;
}

bool generate_state(Rng& rng, const std::string& shape, const std::string& category,
                    std::optional<int> candidate_override, const TemplateSampler& sampler, State& state) {
    for (int attempt = 0; attempt < 20000; ++attempt) {
        if (build_state_once(rng, shape, category, candidate_override, sampler, state) && physical_state_valid(state)) return true;
    }
    return false;
}

void remove_one(std::vector<int>& values, int t) {
    auto it = std::find(values.begin(), values.end(), t);
    if (it == values.end()) throw std::runtime_error("discarded tile is not in hand");
    values.erase(it);
}

void discard(Player& player, int t) {
    remove_one(player.hand, t);
    player.river.push_back(t);
    player.river_seen[t] = true;
}

std::optional<int> draw(State& state) {
    if (state.live_index >= state.live_wall.size()) return std::nullopt;
    return state.live_wall[state.live_index++];
}

void safe_through(Player& opponent, int t) {
    if (!opponent.riichi || !opponent.wait) return;
    if (std::find(opponent.wait->winners.begin(), opponent.wait->winners.end(), t) == opponent.wait->winners.end()) opponent.genbutsu[t] = true;
}

struct RonnerSet {
    std::array<Player*, 4> winner{};
    int count = 0;
};

RonnerSet ronners(State& state, int discarder, int t) {
    RonnerSet result;
    for (auto& player : state.players) {
        if (player.id == discarder || player.folded || !player.wait) continue;
        if (std::find(player.wait->winners.begin(), player.wait->winners.end(), t) != player.wait->winners.end()) {
            if (player.furiten()) throw std::runtime_error("furiten ronner");
            result.winner[result.count++] = &player;
        }
    }
    return result;
}

enum SafetyKind { SAFETY_GENBUTSU = 0, SAFETY_HONOR = 1, SAFETY_DOUBLE_SUJI = 2, SAFETY_SUJI = 3, SAFETY_NOSUJI = 4 };

int number_safety(const Player& opponent, int t) {
    if (opponent.genbutsu[t] || opponent.river_seen[t]) return SAFETY_GENBUTSU;
    if (t >= 27) return SAFETY_HONOR;
    const auto anchors = suji_anchors(t);
    int visible = 0;
    for (int i = 0; i < anchors.count; ++i) if (opponent.genbutsu[anchors.tile[i]] || opponent.river_seen[anchors.tile[i]]) ++visible;
    if (anchors.count > 0 && visible == anchors.count) return SAFETY_DOUBLE_SUJI;
    if (visible > 0) return SAFETY_SUJI;
    return SAFETY_NOSUJI;
}

struct DangerRows {
    std::array<double, 19> no_28;
    std::array<double, 19> no_37;
    std::array<double, 19> no_46;
    std::array<double, 19> no_5;
    std::array<double, 19> suji_37;
};

const DangerRows& danger_rows() {
    static const DangerRows rows = {
        {0.047,0.052,0.058,0.062,0.067,0.071,0.075,0.080,0.086,0.092,0.099,0.106,0.114,0.123,0.133,0.143,0.154,0.167,0.191},
        {0.058,0.063,0.067,0.071,0.075,0.079,0.084,0.089,0.095,0.101,0.108,0.116,0.124,0.133,0.143,0.154,0.166,0.179,0.204},
        {0.057,0.069,0.080,0.089,0.097,0.105,0.113,0.122,0.131,0.141,0.151,0.163,0.175,0.188,0.201,0.217,0.232,0.249,0.278},
        {0.057,0.066,0.077,0.085,0.094,0.102,0.110,0.119,0.128,0.138,0.149,0.160,0.172,0.185,0.199,0.213,0.229,0.247,0.275},
        {0.056,0.053,0.052,0.052,0.053,0.052,0.053,0.053,0.055,0.056,0.057,0.060,0.062,0.066,0.070,0.074,0.080,0.085,0.098}
    };
    return rows;
}

double danger_rate_for_tile(const Player& opponent, int t, int turn) {
    const int safety = number_safety(opponent, t);
    if (safety == SAFETY_GENBUTSU) return 0.0;
    if (t >= 27) return 0.045;
    const int index = std::max(1, std::min(19, turn)) - 1;
    const int rank = rank_of(t);
    const auto& rows = danger_rows();
    if ((safety == SAFETY_SUJI || safety == SAFETY_DOUBLE_SUJI) && (rank == 3 || rank == 7)) return rows.suji_37[index];
    if (rank == 2 || rank == 8) return rows.no_28[index];
    if (rank == 3 || rank == 7) return rows.no_37[index];
    return rank == 5 ? rows.no_5[index] : rows.no_46[index];
}

double joint_danger(const State& state, int discarder, int t, int turn, const std::string& side_mode) {
    double survival = 1.0;
    for (const auto& opponent : state.players) {
        bool in_scope = opponent.id != discarder;
        if (discarder == HERO && side_mode != "full" && (opponent.id == WEST || opponent.id == NORTH)) in_scope = false;
        if (in_scope && opponent.riichi) survival *= 1.0 - danger_rate_for_tile(opponent, t, turn);
    }
    return 1.0 - survival;
}

double threshold_for(const std::string& strategy, int turn) {
    const int index = std::max(1, std::min(19, turn)) - 1;
    if (strategy == "suji37") return danger_rows().suji_37[index];
    if (strategy == "nosuji2378") return std::max(danger_rows().no_28[index], danger_rows().no_37[index]);
    if (strategy == "allout") return 1.0;
    if (strategy == "fixed05") return 0.05;
    if (strategy == "fixed10") return 0.10;
    if (strategy == "fixed15") return 0.15;
    return -1.0;
}

std::pair<int, bool> pick_discard(Player& player, int drawn, const State& state, const std::string& side_mode,
                                  const std::string& strategy, Rng& rng, int turn) {
    std::array<int, 14> legal{};
    int legal_count = 0;
    for (int t : player.hand) {
        if (!player.wait || std::find(player.wait->winners.begin(), player.wait->winners.end(), t) == player.wait->winners.end()) legal[legal_count++] = t;
    }
    if (player.iishanten) {
        std::array<bool, TILE_TYPES> protected_tiles{};
        for (const auto& plan : player.iishanten->plans) if (plan.entry_discard >= 0) protected_tiles[plan.entry_discard] = true;
        std::array<int, 14> preserved{};
        int preserved_count = 0;
        for (int i = 0; i < legal_count; ++i) if (!protected_tiles[legal[i]]) preserved[preserved_count++] = legal[i];
        if (preserved_count > 0) {
            legal = preserved;
            legal_count = preserved_count;
        }
    }
    if (legal_count == 0) {
        legal_count = static_cast<int>(player.hand.size());
        for (int i = 0; i < legal_count; ++i) legal[i] = player.hand[i];
    }

    auto safest = [&]() {
        int best = legal[0];
        double best_danger = joint_danger(state, player.id, best, turn, side_mode);
        for (int i = 1; i < legal_count; ++i) {
            const double danger = joint_danger(state, player.id, legal[i], turn, side_mode);
            if (danger < best_danger) {
                best = legal[i];
                best_danger = danger;
            }
        }
        return best;
    };
    if (strategy == "fold" || player.folded) {
        player.folded = true;
        bool drawn_legal = false;
        for (int i = 0; i < legal_count; ++i) if (legal[i] == drawn) drawn_legal = true;
        if (rng.unit() < FOLD_IMPERFECT_TSUMOGIRI && drawn_legal) return {drawn, true};
        return {safest(), true};
    }
    bool drawn_legal = false;
    for (int i = 0; i < legal_count; ++i) if (legal[i] == drawn) drawn_legal = true;
    if (drawn_legal && joint_danger(state, player.id, drawn, turn, side_mode) <= threshold_for(strategy, turn)) return {drawn, false};
    player.folded = true;
    return {safest(), true};
}

int ura_count(Rng& rng) {
    const int roll = rng.index(34);
    if (roll == 0) return 2;
    if (roll <= 12) return 1;
    return 0;
}

double hero_score(bool tsumo, bool ippatsu, int ura) {
    const int han = 4 + static_cast<int>(tsumo) + static_cast<int>(ippatsu) + ura;
    return han <= 5 ? 8000.0 : han <= 7 ? 12000.0 : 16000.0;
}

void set_folded(Player& player) {
    player.folded = true;
    player.wait.reset();
    player.iishanten.reset();
    player.riichi = false;
}

struct ResolveResult {
    bool terminal = false;
    double delta = 0.0;
    std::string event;
};

ResolveResult resolve_discard(State& state, int discarder, int t, bool dealer_ippatsu, Rng& rng) {
    auto winners = ronners(state, discarder, t);
    if (winners.count > 0) {
        double hero_delta = 0.0;
        for (int i = 0; i < winners.count; ++i) {
            Player* winner = winners.winner[i];
            if (winner->id == HERO) {
                hero_delta += hero_score(false, false, ura_count(rng)) + 1000.0 * state.riichi_sticks;
            } else if (discarder == HERO) {
                if (winner->id == DEALER) hero_delta -= dealer_ippatsu ? DEALER_IPPATSU_RON_LOSS : DEALER_RON_LOSS;
                else hero_delta -= winner->riichi ? RIICHI_SIDE_RON_LOSS : SIDE_RON_LOSS;
            }
        }
        if (discarder == HERO) return {true, hero_delta, "hero_deal_later"};
        for (int i = 0; i < winners.count; ++i) if (winners.winner[i]->id == HERO) return {true, hero_delta, "hero_ron"};
        return {true, 0.0, "side_ron"};
    }
    for (auto& opponent : state.players) if (opponent.id != discarder) safe_through(opponent, t);
    return {};
}

std::optional<Plan> plan_for(const IishantenTemplate& templ, int drawn) {
    for (const auto& plan : templ.plans) if (plan.improvement == drawn) return plan;
    return std::nullopt;
}

bool enter_tenpai(Player& player, int drawn, Rng& rng, double side_riichi_probability) {
    if (!player.iishanten) return false;
    auto plan = plan_for(*player.iishanten, drawn);
    if (!plan) return false;
    if (plan->entry_discard < 0 || std::find(player.hand.begin(), player.hand.end(), plan->entry_discard) == player.hand.end()) {
        throw std::runtime_error("unbound entry discard player=" + std::to_string(player.id) +
                                 " drawn=" + tile_name(drawn) + " entry=" + tile_name(plan->entry_discard));
    }
    if (std::find(plan->wait.winners.begin(), plan->wait.winners.end(), plan->entry_discard) != plan->wait.winners.end()) throw std::runtime_error("entry discard is a winner");
    discard(player, plan->entry_discard);
    player.wait = plan->wait;
    player.wait->owner = player.id;
    player.iishanten.reset();
    player.riichi = player.id == HERO || rng.unit() < side_riichi_probability;
    player.riichi_paid = player.riichi;
    if (player.furiten()) throw std::runtime_error("entered furiten tenpai");
    return true;
}

struct WaitSnapshot {
    std::array<int, 4> kind{};
    int count = 0;
};

WaitSnapshot wait_kind_snapshot(const State& state) {
    WaitSnapshot result;
    for (const auto& player : state.players) if (player.wait) result.kind[result.count++] = player.wait->kind;
    return result;
}

struct Outcome {
    double payoff = 0.0;
    std::string event = "draw_noten";
    std::string route = "continued";
    bool reached_tenpai = false;
    bool current_dealer_hit = false;
    bool current_side_hit = false;
    double current_ippatsu_loss = 0.0;
    double continuation_payoff = 0.0;
    bool current_safe = false;
    WaitSnapshot wait_kinds;
};

double settle_draw(const State& state) {
    int tenpai = 0;
    for (const auto& player : state.players) if (player.wait && !player.folded) ++tenpai;
    const bool hero_tenpai = state.players[HERO].wait.has_value() && !state.players[HERO].folded;
    if (tenpai == 0 || tenpai == 4) return 0.0;
    return hero_tenpai ? 3000.0 / tenpai : -3000.0 / (4 - tenpai);
}

Outcome simulate_from_state(State state, Rng& rng, const std::string& strategy,
                            const std::string& side_mode, double side_riichi_probability) {
    Player& hero = state.players[HERO];
    Player& dealer = state.players[DEALER];
    if (side_mode == "heads_up") {
        // West and North remain physical full-fold players: they still draw,
        // discard, consume wall tiles, and may deal into an active player.
        // Folding removes only their own waits and improvement routes.
        set_folded(state.players[WEST]);
        set_folded(state.players[NORTH]);
    }
    const int candidate = state.current_candidate;
    int current_discard = candidate;
    Rng decision_rng = rng;
    if (strategy == "fold" && decision_rng.unit() >= FOLD_IMPERFECT_TSUMOGIRI) {
        double best_danger = std::numeric_limits<double>::infinity();
        for (int tile : hero.hand) {
            const double danger = joint_danger(state, HERO, tile, INITIAL_TURN, side_mode);
            if (danger < best_danger) {
                best_danger = danger;
                current_discard = tile;
            }
        }
    }
    const bool current_dealer_hit = dealer.wait && std::find(dealer.wait->winners.begin(), dealer.wait->winners.end(), current_discard) != dealer.wait->winners.end();
    bool current_side_hit = false;
    for (int id : {WEST, NORTH}) {
        if (state.players[id].state == 'T' && state.players[id].wait && std::find(state.players[id].wait->winners.begin(), state.players[id].wait->winners.end(), current_discard) != state.players[id].wait->winners.end()) current_side_hit = true;
    }
    if (side_mode != "full") current_side_hit = false;

    const ResolveResult current = resolve_discard(state, HERO, current_discard, true, rng);
    if (current.terminal) {
        return {current.delta, "hero_deal_current", "current", false, current_dealer_hit, current_side_hit,
                current_dealer_hit ? DEALER_IPPATSU_RON_LOSS : 0.0, 0.0, false, wait_kind_snapshot(state)};
    }
    discard(hero, current_discard);
    if (strategy == "fold") set_folded(hero);
    if (!physical_state_valid(state, true)) throw std::runtime_error("post-current-discard invariant failed");

    double payoff = 0.0;
    bool reached_tenpai = false;
    bool dealer_ippatsu = true;
    std::string route = "continued";
    bool current_safe = true;
    int turn = 10;
    const std::array<int, 4> full_order = {WEST, NORTH, DEALER, HERO};

    while (state.live_index < state.live_wall.size()) {
        const int action_count = 4;
        for (int action_index = 0; action_index < action_count; ++action_index) {
            const int id = full_order[action_index];
            auto drawn = draw(state);
            if (!drawn) break;
            Player& player = state.players[id];
            player.hand.push_back(*drawn);
            if (player.wait && std::find(player.wait->winners.begin(), player.wait->winners.end(), *drawn) != player.wait->winners.end()) {
                if (player.id == HERO) {
                    payoff += hero_score(true, false, ura_count(rng)) + 1000.0 * state.riichi_sticks;
                    return {payoff, "hero_tsumo", route, reached_tenpai, false, false, 0.0, payoff, current_safe, wait_kind_snapshot(state)};
                }
                if (player.id == DEALER) {
                    payoff -= dealer_ippatsu ? DEALER_IPPATSU_TSUMO_PAYMENT : DEALER_TSUMO_PAYMENT;
                    return {payoff, dealer_ippatsu ? "dealer_ippatsu_tsumo" : "dealer_tsumo", route, reached_tenpai, false, false, 0.0, payoff, current_safe, wait_kind_snapshot(state)};
                }
                payoff -= player.riichi ? RIICHI_SIDE_TSUMO_PAYMENT : SIDE_TSUMO_PAYMENT;
                return {payoff, "side_tsumo", route, reached_tenpai, false, false, 0.0, payoff, current_safe, wait_kind_snapshot(state)};
            }

            if (player.riichi) {
                discard(player, *drawn);
                const ResolveResult resolved = resolve_discard(state, player.id, *drawn, dealer_ippatsu, rng);
                if (resolved.terminal) return {payoff + resolved.delta, resolved.event, route, reached_tenpai, false, false, 0.0, payoff + resolved.delta, current_safe, wait_kind_snapshot(state)};
                if (player.id == DEALER) {
                    dealer_ippatsu = false;
                    ++turn;
                }
                continue;
            }

            if (player.state == 'H' && enter_tenpai(player, *drawn, rng, side_riichi_probability)) {
                if (player.riichi) {
                    ++state.riichi_sticks;
                    if (player.id == HERO) payoff -= HERO_RIICHI_COST;
                }
                if (player.id == HERO) {
                    reached_tenpai = true;
                    route = WAIT_NAMES[player.wait->kind];
                }
                const int discarded = player.river.back();
                const ResolveResult resolved = resolve_discard(state, player.id, discarded, dealer_ippatsu, rng);
                if (resolved.terminal) return {payoff + resolved.delta, resolved.event, route, reached_tenpai, false, false, 0.0, payoff + resolved.delta, current_safe, wait_kind_snapshot(state)};
                continue;
            }

            std::string player_strategy = strategy;
            if (player.id != HERO) player_strategy = player.state == 'T' ? "nosuji2378" : player.state == 'H' ? "suji37" : "fold";
            const auto picked = pick_discard(player, *drawn, state, side_mode, player_strategy, rng, turn);
            const int discarded = picked.first;
            const bool folded = picked.second;
            discard(player, discarded);
            if (folded) set_folded(player);
            const ResolveResult resolved = resolve_discard(state, player.id, discarded, dealer_ippatsu, rng);
            if (resolved.terminal) {
                if (player.id == HERO) payoff += resolved.delta;
                else payoff += resolved.delta;
                return {payoff, resolved.event, route, reached_tenpai, false, false, 0.0, payoff, current_safe, wait_kind_snapshot(state)};
            }
            if (player.id == DEALER) {
                dealer_ippatsu = false;
                ++turn;
            }
        }
    }

    payoff += settle_draw(state);
    return {payoff, state.players[HERO].wait ? "draw_tenpai" : "draw_noten", route, reached_tenpai, false, false, 0.0, payoff, current_safe, wait_kind_snapshot(state)};
}

Outcome simulate_branch(const std::string& shape, std::uint64_t seed, const std::string& strategy,
                        const std::string& side_mode, double side_riichi_probability,
                        const std::string& category, std::optional<int> candidate_override, const TemplateSampler& sampler) {
    Rng rng(seed);
    State state;
    if (!generate_state(rng, shape, category, candidate_override, sampler, state)) throw std::runtime_error("could not construct a legal physical initial state");
    // Branch randomness is kept separate from the state-construction stream.
    // This lets all strategy branches consume the same post-decision random
    // stream when a shared initial state is used.
    Rng branch_rng(seed ^ 0x9e3779b97f4a7c15ULL);
    return simulate_from_state(std::move(state), branch_rng, strategy, side_mode, side_riichi_probability);
}

struct Summary {
    double ev = 0.0;
    double se = 0.0;
    std::unordered_map<std::string, std::uint64_t> events;
    std::unordered_map<std::string, std::uint64_t> routes;
    std::array<std::uint64_t, 5> wait_kinds{};
    std::uint64_t current_dealer_hits = 0;
    std::uint64_t current_side_hits = 0;
    double current_ippatsu_loss = 0.0;
    double continuation_total = 0.0;
    std::uint64_t current_safe_count = 0;
    std::vector<double> payoffs;
};

Summary summarize(const std::vector<Outcome>& outcomes) {
    Summary result;
    result.payoffs.reserve(outcomes.size());
    for (const auto& outcome : outcomes) {
        result.payoffs.push_back(outcome.payoff);
        result.events[outcome.event]++;
        result.routes[outcome.route]++;
        if (outcome.current_dealer_hit) {
            ++result.current_dealer_hits;
            result.current_ippatsu_loss += DEALER_IPPATSU_RON_LOSS;
        }
        if (outcome.current_side_hit) ++result.current_side_hits;
        if (outcome.current_safe) {
            ++result.current_safe_count;
            result.continuation_total += outcome.continuation_payoff;
        }
        for (int i = 0; i < outcome.wait_kinds.count; ++i) {
            const int kind = outcome.wait_kinds.kind[i];
            if (kind >= 0 && kind < 5) ++result.wait_kinds[kind];
        }
    }
    if (outcomes.empty()) return result;
    for (double x : result.payoffs) result.ev += x;
    result.ev /= static_cast<double>(outcomes.size());
    double variance = 0.0;
    for (double x : result.payoffs) variance += (x - result.ev) * (x - result.ev);
    result.se = std::sqrt(variance / static_cast<double>(outcomes.size()) / static_cast<double>(outcomes.size()));
    return result;
}

std::uint64_t route_seed(std::uint64_t seed, std::uint64_t index) {
    return seed + 1000003ULL * index;
}

Summary run_summary(const std::string& shape, int trials, std::uint64_t seed, const std::string& strategy,
                    const std::string& side_mode, double side_riichi_probability, const std::string& category,
                    std::optional<int> candidate_override, const TemplateSampler& sampler) {
    std::vector<Outcome> outcomes;
    outcomes.reserve(trials);
    for (int i = 0; i < trials; ++i) outcomes.push_back(simulate_branch(shape, route_seed(seed, i), strategy, side_mode, side_riichi_probability, category, candidate_override, sampler));
    return summarize(outcomes);
}

std::pair<double, double> paired_difference(const Summary& attack, const Summary& fold) {
    const std::size_t n = std::min(attack.payoffs.size(), fold.payoffs.size());
    if (n == 0) return {0.0, 0.0};
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += attack.payoffs[i] - fold.payoffs[i];
    mean /= static_cast<double>(n);
    double variance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = attack.payoffs[i] - fold.payoffs[i];
        variance += (d - mean) * (d - mean);
    }
    return {mean, std::sqrt(variance / static_cast<double>(n) / static_cast<double>(n))};
}

double break_even_danger(double continuation, double fold_ev) {
    const double denominator = DEALER_IPPATSU_RON_LOSS + continuation;
    if (std::abs(denominator) < 1e-12) return 0.0;
    // This is the algebraic crossing point.  Values outside [0, 1] are
    // informative: they mean no physical danger probability produces a
    // crossing under the fitted continuation and fold EVs.
    return (continuation - fold_ev) / denominator;
}

std::string tile_class(int t) {
    if (t >= 27) return "honor";
    const int r = rank_of(t);
    if (r == 1 || r == 9) return "one_nine";
    if (r == 2 || r == 8) return "two_eight";
    if (r == 3 || r == 7) return "three_seven";
    return "four_five_six";
}

struct Residual {
    std::string key;
    double target;
    double fitted;
    double difference;
    double standard_error;
};

const std::vector<Residual>& default_residuals() {
    // Independent 20,000-state diagnostics for the compact mirror-symmetric
    // fit.  Residuals are retained rather than hidden with per-tile knobs.
    static const std::vector<Residual> values = {
        {"honor.tanki",0.0056,0.005310855123274208,-0.00028914487672579165,0.00022059428302150612},
        {"honor.shanpon",0.0120,0.012265996352330822,0.00026599635233082175,0.0003706026779560319},
        {"honor.total",0.0176,0.017576851475605032,-0.000023148524394969033,0.00043128665943888714},
        {"one_nine.tanki",0.0047,0.004664423665122782,-0.000035576334877217806,0.00022316528105675523},
        {"one_nine.shanpon",0.0101,0.010410294503189723,0.0003102945031897234,0.00035862901646536294},
        {"one_nine.ryanmen",0.0461,0.04131346674823036,-0.004786533251769642,0.0006030025269382469},
        {"one_nine.total",0.0609,0.05638818491654286,-0.004511815083457139,0.0007362265694838548},
        {"two_eight.tanki",0.0048,0.004485281309876328,-0.00031471869012367117,0.00021859319377225752},
        {"two_eight.shanpon",0.0089,0.009122975091447483,0.00022297509144748352,0.0003286168171230516},
        {"two_eight.kanchan",0.0118,0.01155068803344365,-0.00024931196655634917,0.0003446684120773785},
        {"two_eight.ryanmen",0.0466,0.04165215119317192,-0.004947848806828081,0.0006040978106467116},
        {"two_eight.total",0.0720,0.06681109562793938,-0.005188904372060613,0.0007996889870266644},
        {"three_seven.tanki",0.0038,0.003839979172994316,0.000039979172994315804,0.00020225074453237944},
        {"three_seven.shanpon",0.0066,0.007061656614743784,0.00046165661474378405,0.0002917011408677534},
        {"three_seven.kanchan",0.0118,0.010587061222718791,-0.0012129387772812084,0.0003303089702252966},
        {"three_seven.penchan",0.0097,0.008743003427778019,-0.0009569965722219815,0.0003015419519323806},
        {"three_seven.ryanmen",0.0474,0.0431834945979954,-0.004216505402004599,0.0006104817056291894},
        {"three_seven.total",0.0794,0.07341519503623031,-0.005984804963769685,0.0008358913785545709},
        {"four_five_six.tanki",0.0044,0.003933695300420697,-0.0004663046995793032,0.00016591389033503279},
        {"four_five_six.shanpon",0.0067,0.00642911078350293,-0.0002708892164970701,0.00022737746658039093},
        {"four_five_six.kanchan",0.0083,0.006824637733271008,-0.0014753622667289922,0.00021617505294646534},
        {"four_five_six.ryanmen",0.0762,0.08334112401567725,0.00714112401567725,0.0005045557209977696},
        {"four_five_six.total",0.0956,0.1005285678328719,0.004928567832871891,0.0006168760494804513}
    };
    return values;
}

Weights default_weights() {
    Weights w;
    w.kind = {{"ryanmen",2.103018709966594},{"kanchan",0.7931893428052499},{"penchan",0.5155417096676611},
              {"shanpon",-2.3198482669846596},{"tanki",0.0}};
    w.position = {};
    w.internal = {};
    w.location = {
        {"ryanmen:2",0.0015405314187024607},{"ryanmen:3",0.0329485265810187},
        {"kanchan:3",-0.00028869099824344594},{"kanchan:4",-0.13696828240052153},{"kanchan:5",-1.4625905725590236},
        {"tanki:one_nine",-0.14524019480503936},{"tanki:two_eight",-0.13309023220683117},
        {"tanki:three_seven",-0.29201299611410114},{"tanki:four_five_six",-0.21129329045558723},
        {"shanpon:one_nine",-0.18418856738816006},{"shanpon:two_eight",-0.28592479100670826},
        {"shanpon:three_seven",-0.5103770898962516},{"shanpon:four_five_six",-0.5505124453465684}
    };
    w.regularization = 0.0005;
    w.source = "three-stage constrained fit: 3x20000 legal states; independent n=20000 validation";
    return w;
}

std::string json_quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"') out << "\\\"";
        else if (c == '\\') out << "\\\\";
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec << std::setfill('0');
        else out << c;
    }
    out << '"';
    return out.str();
}

std::string number_json(double value) {
    if (!std::isfinite(value)) return "0";
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::string weights_json(const Weights& weights) {
    const std::array<std::string, 5> kinds = {"ryanmen","kanchan","penchan","shanpon","tanki"};
    const std::array<std::string, 4> positions = {"terminal","outer","middle","honor"};
    auto map_json = [](const auto& map) {
        std::vector<std::pair<std::string,double>> items(map.begin(), map.end());
        std::sort(items.begin(), items.end());
        std::ostringstream out;
        out << "{";
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (i) out << ",";
            out << json_quote(items[i].first) << ":" << number_json(items[i].second);
        }
        out << "}";
        return out.str();
    };
    std::ostringstream out;
    out << "{\"kind_log_weights\":{";
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        if (i) out << ",";
        out << json_quote(kinds[i]) << ":" << number_json(map_value(weights.kind, kinds[i]));
    }
    out << "},\"position_log_weights\":{";
    for (std::size_t i = 0; i < positions.size(); ++i) {
        if (i) out << ",";
        out << json_quote(positions[i]) << ":" << number_json(map_value(weights.position, positions[i]));
    }
    out << "},\"regularization\":" << number_json(weights.regularization)
        << ",\"source\":" << json_quote(weights.source)
        << ",\"internal_log_weights\":" << map_json(weights.internal)
        << ",\"location_log_weights\":" << map_json(weights.location) << "}";
    return out.str();
}

std::string extract_object(const std::string& text, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const std::size_t marker_pos = text.find(marker);
    if (marker_pos == std::string::npos) return {};
    const std::size_t open = text.find('{', marker_pos + marker.size());
    if (open == std::string::npos) return {};
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = open; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) return text.substr(open + 1, i - open - 1);
        }
    }
    return {};
}

std::optional<double> json_number(const std::string& text, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) return std::nullopt;
    try { return std::stod(match[1].str()); } catch (...) { return std::nullopt; }
}

void load_map(const std::string& object, const std::vector<std::string>& keys, std::unordered_map<std::string,double>& destination) {
    for (const auto& key : keys) {
        auto value = json_number(object, key);
        if (value) destination[key] = *value;
    }
}

void load_weights_file(const std::string& path, Weights& weights, std::vector<Residual>& residuals) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open weights file: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string text = buffer.str();
    std::string root = extract_object(text, "weights");
    if (root.empty()) root = text;
    load_map(extract_object(root, "kind_log_weights"), {"ryanmen","kanchan","penchan","shanpon","tanki"}, weights.kind);
    load_map(extract_object(root, "position_log_weights"), {"terminal","outer","middle","honor"}, weights.position);
    load_map(extract_object(root, "internal_log_weights"), {
        "ryanmen:terminal","ryanmen:outer","ryanmen:middle","ryanmen:honor","kanchan:terminal","kanchan:outer","kanchan:middle","kanchan:honor",
        "penchan:terminal","penchan:outer","penchan:middle","penchan:honor","shanpon:terminal","shanpon:outer","shanpon:middle","shanpon:honor",
        "tanki:terminal","tanki:outer","tanki:middle","tanki:honor"}, weights.internal);
    load_map(extract_object(root, "location_log_weights"), {
        "ryanmen:2","ryanmen:3","kanchan:3","kanchan:4","kanchan:5",
        "tanki:one_nine","tanki:two_eight","tanki:three_seven","tanki:four_five_six",
        "shanpon:one_nine","shanpon:two_eight","shanpon:three_seven","shanpon:four_five_six"}, weights.location);
    if (auto value = json_number(root, "regularization")) weights.regularization = *value;
    // Read the provenance string from the nested weights object.  The
    // calibration payload also has a top-level `source` object describing
    // the Koba sample; taking that first key used to emit the bogus label
    // "hanchan" without changing the numerical weights.
    const auto source_pos = root.find("\"source\"");
    if (source_pos != std::string::npos) {
        const auto colon = root.find(':', source_pos);
        const auto first = root.find('"', colon + 1);
        const auto second = root.find('"', first + 1);
        if (first != std::string::npos && second != std::string::npos) weights.source = root.substr(first + 1, second - first - 1);
    }
    const std::string residual_object = extract_object(text, "residuals");
    if (!residual_object.empty()) {
        for (auto& residual : residuals) {
            const std::string item = extract_object(residual_object, residual.key);
            if (item.empty()) continue;
            if (auto value = json_number(item, "fitted")) residual.fitted = *value;
            if (auto value = json_number(item, "standard_error")) residual.standard_error = *value;
            // Targets are kept at the shared validator values.  This avoids
            // propagating the Python payload's historical 0.0721 sum.
            residual.difference = residual.fitted - residual.target;
        }
    }
}

double calibration_rmse(const std::vector<Residual>& residuals) {
    double sum = 0.0;
    int n = 0;
    for (const auto& residual : residuals) if (residual.key.find(".total") == std::string::npos) {
        sum += residual.difference * residual.difference;
        ++n;
    }
    return n ? std::sqrt(sum / n) : 0.0;
}

std::string calibration_json(const Weights& weights, const std::vector<Residual>& residuals) {
    const std::array<std::pair<std::string,double>, 18> targets = {{
        {"honor.tanki",0.0056},{"honor.shanpon",0.0120},{"one_nine.tanki",0.0047},{"one_nine.shanpon",0.0101},{"one_nine.ryanmen",0.0461},
        {"two_eight.tanki",0.0048},{"two_eight.shanpon",0.0089},{"two_eight.kanchan",0.0118},{"two_eight.ryanmen",0.0466},
        {"three_seven.tanki",0.0038},{"three_seven.shanpon",0.0066},{"three_seven.kanchan",0.0118},{"three_seven.penchan",0.0097},{"three_seven.ryanmen",0.0474},
        {"four_five_six.tanki",0.0044},{"four_five_six.shanpon",0.0067},{"four_five_six.kanchan",0.0083},{"four_five_six.ryanmen",0.0762}
    }};
    double max_total_error = 0.0;
    for (const auto& residual : residuals) if (residual.key.find(".total") != std::string::npos) {
        max_total_error = std::max(max_total_error, std::abs(residual.difference));
    }
    const double rmse = calibration_rmse(residuals);
    std::ostringstream out;
    out << "{\"source\":{\"hanchan\":171650,\"riichi\":1300316,\"condition\":\"after riichi declaration discard; excludes furiten and kokushi riichi\"},\"targets\":{";
    for (std::size_t i = 0; i < targets.size(); ++i) {
        if (i) out << ",";
        out << json_quote(targets[i].first) << ":" << number_json(targets[i].second);
    }
    out << "},\"weights\":" << weights_json(weights) << ",\"residuals\":{";
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        if (i) out << ",";
        const auto& r = residuals[i];
        out << json_quote(r.key) << ":{\"target\":" << number_json(r.target)
            << ",\"fitted\":" << number_json(r.fitted)
            << ",\"difference\":" << number_json(r.difference)
            << ",\"standard_error\":" << number_json(r.standard_error) << "}";
    }
    out << "},\"weighted_rmse\":" << number_json(rmse)
        << ",\"validation_trials\":20000,\"acceptance\":{\"basic_rmse_target\":0.0025,\"main_total_error_target\":0.005"
        << ",\"observed_basic_rmse\":" << number_json(rmse)
        << ",\"observed_max_total_error\":" << number_json(max_total_error)
        << ",\"met\":" << (rmse <= 0.0025 && max_total_error <= 0.005 ? "true" : "false") << "}"
        << ",\"note\":\"Targets are candidate-tile winning contributions, not per-hand wait-form shares. The compact mirror-symmetric model keeps its independent-sample residual instead of adding suit- or tile-specific parameters.\"}";
    return out.str();
}

std::string map_counts_json(const std::unordered_map<std::string, std::uint64_t>& values) {
    std::vector<std::pair<std::string,std::uint64_t>> items(values.begin(), values.end());
    std::sort(items.begin(), items.end());
    std::ostringstream out;
    out << "{";
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) out << ",";
        out << json_quote(items[i].first) << ":" << items[i].second;
    }
    out << "}";
    return out.str();
}

std::string result_json(const std::string& shape, const std::string& strategy, const std::string& category,
                        const Summary& attack, const Summary& fold, const std::pair<double,double>& pair, int trials,
                        const std::vector<Residual>& residuals) {
    const double continuation = attack.current_safe_count ? attack.continuation_total / attack.current_safe_count : 0.0;
    const double any_hit = static_cast<double>(attack.current_dealer_hits + attack.current_side_hits) / trials;
    std::ostringstream out;
    out << "{\"shape\":" << json_quote(shape) << ",\"strategy\":" << json_quote(strategy)
        << ",\"current_category\":" << json_quote(category)
        << ",\"ev\":" << number_json(attack.ev) << ",\"se\":" << number_json(attack.se)
        << ",\"fold_ev\":" << number_json(fold.ev) << ",\"fold_se\":" << number_json(fold.se)
        << ",\"ev_difference\":" << number_json(attack.ev - fold.ev)
        << ",\"paired_ev_difference_se\":" << number_json(pair.second)
        << ",\"break_even_danger\":" << number_json(break_even_danger(continuation, fold.ev))
        << ",\"event_counts\":" << map_counts_json(attack.events) << ",\"event_rates\":{";
    std::vector<std::pair<std::string,std::uint64_t>> event_items(attack.events.begin(), attack.events.end());
    std::sort(event_items.begin(), event_items.end());
    for (std::size_t i = 0; i < event_items.size(); ++i) {
        if (i) out << ",";
        out << json_quote(event_items[i].first) << ":" << number_json(static_cast<double>(event_items[i].second) / trials);
    }
    out << "},\"route_counts\":" << map_counts_json(attack.routes) << ",\"wait_kind_counts\":{";
    for (int i = 0; i < 5; ++i) {
        if (i) out << ",";
        out << json_quote(WAIT_NAMES[i]) << ":" << attack.wait_kinds[i];
    }
    out << "},\"current_hit_rates\":{";
    out << "\"dealer\":" << number_json(static_cast<double>(attack.current_dealer_hits) / trials)
        << ",\"initial_side\":" << number_json(static_cast<double>(attack.current_side_hits) / trials)
        << ",\"any\":" << number_json(any_hit) << "}"
        << ",\"current_dealer_ippatsu_contribution\":" << number_json(-attack.current_ippatsu_loss / trials)
        << ",\"continuation_ev_after_safe\":" << number_json(continuation)
        << ",\"calibration_residuals\":{";
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        if (i) out << ",";
        const auto& r = residuals[i];
        out << json_quote(r.key) << ":{\"target\":" << number_json(r.target)
            << ",\"fitted\":" << number_json(r.fitted) << ",\"difference\":" << number_json(r.difference)
            << ",\"standard_error\":" << number_json(r.standard_error) << "}";
    }
    out << "}}";
    return out.str();
}

std::string document_json(std::uint64_t seed, int trials, const std::string& side_mode, double side_riichi_probability,
                          const Weights& weights, const std::vector<Residual>& residuals,
                          const std::vector<std::string>& shapes, const std::vector<std::string>& strategies,
                          const std::vector<std::string>& categories, std::optional<int> candidate_override) {
    const TemplateSampler sampler(weights);
    std::ostringstream out;
    out << "{\"engine_version\":\"cpp-legal-wall-1\",\"engine\":\"cpp\",\"seed\":" << seed
        << ",\"trials\":" << trials << ",\"config\":{"
        << "\"side_mode\":" << json_quote(side_mode) << ",\"side_riichi_probability\":" << number_json(side_riichi_probability)
        << ",\"seat_map\":{\"dealer\":\"east\",\"hero\":\"south\",\"side1\":\"west\",\"side2\":\"north\"}"
        << ",\"river_lengths\":{\"dealer\":9,\"hero\":9,\"side1\":8,\"side2\":8}"
        << ",\"action_order\":[\"hero_current_discard_check\",\"west_draw_discard\",\"north_draw_discard\",\"dealer_draw\"]"
        << ",\"live_wall_tiles\":36,\"hero_unseen_positions\":88,\"weights\":" << weights_json(weights) << "},\"results\":[";
    bool first_result = true;
    bool first_pair = true;
    std::ostringstream pairwise;
    for (const auto& shape : shapes) {
        for (const auto& category : categories) {
            const Summary fold = run_summary(shape, trials, seed, "fold", side_mode, side_riichi_probability, category, candidate_override, sampler);
            std::vector<std::pair<std::string, Summary>> attacks;
            attacks.reserve(strategies.size());
            for (const auto& strategy : strategies) {
                Summary attack = run_summary(shape, trials, seed, strategy, side_mode, side_riichi_probability, category, candidate_override, sampler);
                const auto pair = paired_difference(attack, fold);
                if (!first_result) out << ",";
                first_result = false;
                out << result_json(shape, strategy, category, attack, fold, pair, trials, residuals);
                attacks.emplace_back(strategy, std::move(attack));
            }
            for (std::size_t first = 0; first < attacks.size(); ++first) {
                for (std::size_t second = first + 1; second < attacks.size(); ++second) {
                    const auto comparison = paired_difference(attacks[first].second, attacks[second].second);
                    if (!first_pair) pairwise << ",";
                    first_pair = false;
                    pairwise << "{\"shape\":" << json_quote(shape)
                             << ",\"current_category\":" << json_quote(category)
                             << ",\"strategy_a\":" << json_quote(attacks[first].first)
                             << ",\"strategy_b\":" << json_quote(attacks[second].first)
                             << ",\"ev_difference_a_minus_b\":" << number_json(comparison.first)
                             << ",\"paired_standard_error\":" << number_json(comparison.second) << "}";
                }
            }
        }
    }
    out << "],\"strategy_pairwise\":[" << pairwise.str() << "],\"calibration\":" << calibration_json(weights, residuals) << "}";
    return out.str();
}

std::vector<std::string> split_strategies(const std::string& value) {
    if (value == "all") return std::vector<std::string>(ATTACK_STRATEGIES.begin(), ATTACK_STRATEGIES.end());
    if (value == "category") return {"suji37", "nosuji2378", "allout"};
    if (value == "fixed") return {"fixed05", "fixed10", "fixed15"};
    if (value.find(',') != std::string::npos) {
        std::vector<std::string> result;
        std::size_t begin = 0;
        while (begin <= value.size()) {
            const std::size_t end = value.find(',', begin);
            result.push_back(value.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        return result;
    }
    return {value};
}

std::vector<std::string> split_shapes(const std::string& value) {
    if (value == "all") return std::vector<std::string>(SHAPE_NAMES.begin(), SHAPE_NAMES.end());
    return {value};
}

std::vector<std::string> split_categories(const std::string& value) {
    if (value == "all") return std::vector<std::string>(CURRENT_CATEGORIES.begin(), CURRENT_CATEGORIES.end());
    return {value};
}

void self_test(std::uint64_t seed, const Weights& weights) {
    const TemplateSampler sampler(weights);
    if (all_wait_templates().size() != 640) throw std::runtime_error("wait enumeration size mismatch");
    int ryanmen_count = 0;
    for (const auto& wait : all_wait_templates()) if (wait.kind == RYANMEN) {
        ++ryanmen_count;
        if (wait.winners.size() != 2 || wait.winners[1] - wait.winners[0] != 3 || suit_of(wait.winners[0]) != suit_of(wait.winners[1])) throw std::runtime_error("illegal ryanmen");
    }
    if (ryanmen_count != 18) throw std::runtime_error("ryanmen enumeration mismatch");
    for (const auto& shape : SHAPE_NAMES) {
        Rng first_rng(seed), second_rng(seed);
        State first, second;
        if (!generate_state(first_rng, shape, "nosuji2378", std::nullopt, sampler, first) || !generate_state(second_rng, shape, "nosuji2378", std::nullopt, sampler, second)) throw std::runtime_error("self-test state generation failed");
        if (!physical_state_valid(first) || !physical_state_valid(second)) throw std::runtime_error("self-test physical invariant failed");
        if (first.current_candidate != second.current_candidate || first.dora_indicator != second.dora_indicator || first.live_wall != second.live_wall) throw std::runtime_error("self-test determinism failed");
        if (first.players[HERO].hand.size() != 14 || first.players[DEALER].hand.size() != 13 || first.players[WEST].hand.size() != 13 || first.players[NORTH].hand.size() != 13) throw std::runtime_error("self-test hand size failed");
        for (const auto& player : first.players) {
            if (player.wait && !player.wait->valid()) throw std::runtime_error("self-test wait invalid");
            if (player.iishanten) for (const auto& plan : player.iishanten->plans) {
                if (plan.entry_discard < 0 || std::find(player.hand.begin(), player.hand.end(), plan.entry_discard) == player.hand.end()) throw std::runtime_error("self-test entry binding failed");
            }
        }
    }
    std::cout << "self-test: ok\n";
}

struct Options {
    int trials = 10000;
    std::uint64_t seed = 20260809;
    std::string side_mode = "full";
    std::string shape = "all";
    std::string hero_strategy = "all";
    double side_riichi_probability = DEFAULT_SIDE_RIICHI_ON_TENPAI;
    std::optional<int> current_passed_tile;
    std::string current_category = "nosuji2378";
    std::optional<double> fold_ev;
    std::optional<double> fold_se;
    bool json = false;
    bool self_test = false;
    std::string weights_path;
    std::string write_calibration_path;
};

Options parse_options(int argc, char** argv) {
    Options options;
    auto require_value = [&](int& i, const std::string& name) {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
        return std::string(argv[++i]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--trials") options.trials = std::stoi(require_value(i, arg));
        else if (arg == "--seed") options.seed = std::stoull(require_value(i, arg));
        else if (arg == "--side-mode") options.side_mode = require_value(i, arg);
        else if (arg == "--shape") options.shape = require_value(i, arg);
        else if (arg == "--hero-strategy") options.hero_strategy = require_value(i, arg);
        else if (arg == "--side-riichi-probability") options.side_riichi_probability = std::stod(require_value(i, arg));
        else if (arg == "--current-passed-tile") options.current_passed_tile = std::stoi(require_value(i, arg));
        else if (arg == "--current-category") options.current_category = require_value(i, arg);
        else if (arg == "--fold-ev") options.fold_ev = std::stod(require_value(i, arg));
        else if (arg == "--fold-se") options.fold_se = std::stod(require_value(i, arg));
        else if (arg == "--json") options.json = true;
        else if (arg == "--weights") options.weights_path = require_value(i, arg);
        else if (arg == "--self-test") options.self_test = true;
        else if (arg == "--write-calibration") options.write_calibration_path = require_value(i, arg);
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (options.trials <= 0) throw std::runtime_error("trials must be positive");
    if (options.side_mode != "full" && options.side_mode != "heads_up") throw std::runtime_error("side-mode must be full or heads_up");
    if (options.side_riichi_probability < 0.0 || options.side_riichi_probability > 1.0) throw std::runtime_error("side-riichi-probability must be between 0 and 1");
    if (options.current_passed_tile && (*options.current_passed_tile < 0 || *options.current_passed_tile >= TILE_TYPES)) throw std::runtime_error("current-passed-tile must be between 0 and 33");
    if (options.fold_ev.has_value() != options.fold_se.has_value()) throw std::runtime_error("fold-ev and fold-se must be supplied together");
    if (options.shape != "all" && std::find(SHAPE_NAMES.begin(), SHAPE_NAMES.end(), options.shape) == SHAPE_NAMES.end()) throw std::runtime_error("unknown shape");
    const std::vector<std::string> valid_strategies = {"all","category","fixed","fold","suji37","nosuji2378","allout","fixed05","fixed10","fixed15"};
    for (const auto& strategy : split_strategies(options.hero_strategy)) {
        if (std::find(valid_strategies.begin(), valid_strategies.end(), strategy) == valid_strategies.end() || strategy == "all" || strategy == "category" || strategy == "fixed") {
            throw std::runtime_error("unknown hero strategy");
        }
    }
    const std::vector<std::string> valid_categories = {"all","genbutsu","suji37","nosuji2378","nosuji456","fold"};
    if (std::find(valid_categories.begin(), valid_categories.end(), options.current_category) == valid_categories.end()) throw std::runtime_error("unknown current category");
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        Weights weights = default_weights();
        std::vector<Residual> residuals = default_residuals();
        if (!options.weights_path.empty()) load_weights_file(options.weights_path, weights, residuals);
        if (options.self_test) {
            self_test(options.seed, weights);
            return 0;
        }
        if (!options.write_calibration_path.empty()) {
            std::ofstream output(options.write_calibration_path);
            if (!output) throw std::runtime_error("cannot write calibration file");
            output << calibration_json(weights, residuals) << "\n";
        }
        const auto shapes = split_shapes(options.shape);
        const auto strategies = options.hero_strategy == "fold" ? std::vector<std::string>{"fold"} : split_strategies(options.hero_strategy);
        const auto categories = split_categories(options.current_category);
        const TemplateSampler sampler(weights);
        if (options.json) {
            std::cout << document_json(options.seed, options.trials, options.side_mode, options.side_riichi_probability,
                                       weights, residuals, shapes, strategies, categories, options.current_passed_tile) << "\n";
            return 0;
        }
        for (const auto& shape : shapes) for (const auto& category : categories) {
            const Summary fold = run_summary(shape, options.trials, options.seed, "fold", options.side_mode, options.side_riichi_probability, category, options.current_passed_tile, sampler);
            for (const auto& strategy : strategies) {
                const Summary attack = run_summary(shape, options.trials, options.seed, strategy, options.side_mode, options.side_riichi_probability, category, options.current_passed_tile, sampler);
                const auto pair = paired_difference(attack, fold);
                const double continuation = attack.current_safe_count ? attack.continuation_total / attack.current_safe_count : 0.0;
                std::cout << shape << "\t" << strategy << "\t" << category << "\tEV=" << std::fixed << std::setprecision(2) << attack.ev
                          << "±" << attack.se << "\tΔfold=" << attack.ev - fold.ev << "±" << pair.second
                          << "\tdealer-hit=" << std::setprecision(3) << 100.0 * attack.current_dealer_hits / options.trials
                          << "%\tside-hit=" << 100.0 * attack.current_side_hits / options.trials
                          << "%\tcontinuation=" << std::setprecision(2) << continuation << "\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
