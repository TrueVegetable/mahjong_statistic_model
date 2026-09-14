#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(HAND_EFFICIENCY_USE_TILE_MASKS)
#include <mahjong/calsht_dw.hpp>
#elif defined(HAND_EFFICIENCY_USE_NYANTEN)
#include <nyanten/replacement_number.hpp>
#endif

// Native batch engine for outputs/hand_efficiency_sim.py.  It intentionally
// stops short of han/fu/yaku calculation: --observations emits winning physical
// hands for the Python MahjongRepository scoring adapter.

namespace {

constexpr int TILE_TYPES = 34;
constexpr int PHYSICAL_TILES = 136;
constexpr std::uint64_t DAMA_LOCK_FLAG = std::uint64_t{1} << 63;
constexpr std::uint64_t DAMA_TSUMOGIRI_FLAG = std::uint64_t{1} << 62;
constexpr std::uint64_t DAMA_REJECT_TRANSITION_FLAG = std::uint64_t{1} << 61;
constexpr std::uint64_t DAMA_WAIT_MASK = (std::uint64_t{1} << TILE_TYPES) - 1;
constexpr int INITIAL_HAND = 13;
constexpr int OPPONENT_HANDS = 39;
constexpr int DEAD_WALL = 14;
constexpr int LIVE_WALL = 70;
constexpr std::uint64_t DEFAULT_SEED = 20260813ULL;
constexpr const char* ENGINE_VERSION = "closed-hand-efficiency-cpp-4";
#if defined(HAND_EFFICIENCY_USE_TILE_MASKS)
constexpr const char* EVALUATOR_BACKEND = "necessary_tiles_masks";
#elif defined(HAND_EFFICIENCY_USE_NYANTEN)
constexpr const char* EVALUATOR_BACKEND = "nyanten_shanten";
#else
constexpr const char* EVALUATOR_BACKEND = "builtin_fallback";
#endif

using Counts = std::array<std::uint8_t, TILE_TYPES>;
using Visible = std::array<int, TILE_TYPES>;

template <typename T> std::string numeric_array_json(const std::vector<T>& values);

const std::array<int, 13> TERMINAL_HONORS = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33};
const std::array<std::string, 8> SHAPE_NAMES = {
    "ryanmen", "kanchan", "penchan", "shanpon", "tanki",
    "chiitoitsu_tanki", "kokushi_13way", "kokushi_single",
};

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

std::uint64_t trial_seed(std::uint64_t base_seed, std::uint64_t trial_id) {
    return splitmix64(base_seed ^ splitmix64(trial_id));
}

class Rng {
  public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() {
        state_ += 0x9E3779B97F4A7C15ULL;
        std::uint64_t value = state_;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    }

    std::uint64_t uniform(std::uint64_t bound) {
        if (bound == 0) throw std::invalid_argument("uniform bound must be positive");
        const std::uint64_t threshold = -bound % bound;
        while (true) {
            const std::uint64_t value = next();
            if (value >= threshold) return value % bound;
        }
    }

    template <typename T> void shuffle(std::vector<T>& values) {
        for (std::size_t i = values.size(); i > 1; --i) {
            const std::size_t j = static_cast<std::size_t>(uniform(i));
            std::swap(values[i - 1], values[j]);
        }
    }

  private:
    std::uint64_t state_;
};

int rank_of(int tile) { return tile < 27 ? tile % 9 + 1 : -1; }
int suit_of(int tile) { return tile < 27 ? tile / 9 : -1; }

std::string tile_name(int tile) {
    if (tile < 0 || tile >= TILE_TYPES) throw std::invalid_argument("tile type outside 0..33");
    if (tile >= 27) return std::to_string(tile - 26) + "z";
    const char suits[] = {'m', 'p', 's'};
    return std::to_string(rank_of(tile)) + suits[suit_of(tile)];
}

std::string physical_tile_name(int tile_id) {
    if (tile_id == 16) return "0m";
    if (tile_id == 52) return "0p";
    if (tile_id == 88) return "0s";
    return tile_name(tile_id / 4);
}

bool is_red_five(int tile_id) { return tile_id == 16 || tile_id == 52 || tile_id == 88; }

int dora_type_from_indicator(int indicator) {
    if (indicator < 0 || indicator >= TILE_TYPES) throw std::invalid_argument("invalid dora indicator");
    if (indicator < 27) return indicator / 9 * 9 + (indicator % 9 + 1) % 9;
    if (indicator < 31) return 27 + (indicator - 27 + 1) % 4;
    return 31 + (indicator - 31 + 1) % 3;
}

int yakuhai_value(int tile, int player_wind, int round_wind) {
    if (tile >= 31 && tile < 34) return 1;
    if (tile >= 27 && tile < 31) return int(tile == player_wind) + int(tile == round_wind);
    return 0;
}

bool is_tanyao_tile(int tile) {
    return tile < 27 && tile % 9 >= 1 && tile % 9 <= 7;
}

bool is_dora_neighbor(int tile, int dora) {
    return dora >= 0 && dora < 27 && tile < 27 && tile / 9 == dora / 9
        && std::abs(tile % 9 - dora % 9) == 1;
}

int tile_total(const Counts& counts) {
    return std::accumulate(counts.begin(), counts.end(), 0);
}

Counts counts_from_physical(const std::vector<int>& tiles) {
    Counts counts{};
    for (int tile : tiles) {
        if (tile < 0 || tile >= PHYSICAL_TILES) throw std::invalid_argument("physical tile outside 0..135");
        if (++counts[tile / 4] > 4) throw std::invalid_argument("more than four tile copies");
    }
    return counts;
}

std::string format_counts(const Counts& counts) {
    std::ostringstream out;
    for (int suit = 0; suit < 3; ++suit) {
        bool any = false;
        for (int rank = 0; rank < 9; ++rank) {
            const int tile = suit * 9 + rank;
            for (int copy = 0; copy < counts[tile]; ++copy) {
                out << rank + 1;
                any = true;
            }
        }
        if (any) out << "mps"[suit];
    }
    bool honors = false;
    for (int rank = 0; rank < 7; ++rank) {
        for (int copy = 0; copy < counts[27 + rank]; ++copy) {
            out << rank + 1;
            honors = true;
        }
    }
    if (honors) out << 'z';
    return out.str();
}

Counts parse_hand(const std::string& text) {
    Counts counts{};
    std::vector<int> digits;
    for (char raw : text) {
        const char ch = static_cast<char>(std::tolower(static_cast<unsigned char>(raw)));
        if (std::isspace(static_cast<unsigned char>(ch))) continue;
        if (ch >= '0' && ch <= '9') {
            digits.push_back(ch - '0');
            continue;
        }
        const std::string suits = "mpsz";
        const auto position = suits.find(ch);
        if (position == std::string::npos || digits.empty()) throw std::invalid_argument("invalid mpsz hand");
        for (int digit : digits) {
            int tile = -1;
            if (position == 3) {
                if (digit < 1 || digit > 7) throw std::invalid_argument("honor rank outside 1..7");
                tile = 26 + digit;
            } else {
                if (digit == 0) digit = 5;
                if (digit < 1 || digit > 9) throw std::invalid_argument("suit rank outside 0..9");
                tile = static_cast<int>(position) * 9 + digit - 1;
            }
            if (++counts[tile] > 4) throw std::invalid_argument("more than four copies in hand");
        }
        digits.clear();
    }
    if (!digits.empty()) throw std::invalid_argument("mpsz hand ends without a suit");
    return counts;
}

std::vector<int> parse_physical_hand(const std::string& text) {
    (void)parse_hand(text);  // Perform the shared notation and copy-count validation first.
    std::array<bool, PHYSICAL_TILES> used{};
    std::vector<int> physical;
    std::vector<int> digits;
    for (char raw : text) {
        const char ch = static_cast<char>(std::tolower(static_cast<unsigned char>(raw)));
        if (std::isspace(static_cast<unsigned char>(ch))) continue;
        if (ch >= '0' && ch <= '9') {
            digits.push_back(ch - '0');
            continue;
        }
        const std::size_t suit = std::string("mpsz").find(ch);
        for (int digit : digits) {
            const bool red = suit < 3 && digit == 0;
            if (red) digit = 5;
            const int tile = suit == 3 ? 26 + digit : static_cast<int>(suit) * 9 + digit - 1;
            int tile_id = -1;
            if (red) {
                tile_id = tile * 4;
                if (used[tile_id]) throw std::invalid_argument("duplicate red five");
            } else {
                for (int copy = 0; copy < 4; ++copy) {
                    const int candidate = tile * 4 + copy;
                    if ((tile == 4 || tile == 13 || tile == 22) && copy == 0) continue;
                    if (!used[candidate]) {
                        tile_id = candidate;
                        break;
                    }
                }
                if (tile_id < 0) tile_id = tile * 4;  // fourth implicit five uses the red copy
            }
            if (tile_id < 0 || used[tile_id]) throw std::invalid_argument("duplicate physical tile");
            used[tile_id] = true;
            physical.push_back(tile_id);
        }
        digits.clear();
    }
    return physical;
}

int parse_single_tile_type(const std::string& text) {
    const Counts counts = parse_hand(text);
    if (tile_total(counts) != 1) throw std::invalid_argument("expected exactly one tile");
    for (int tile = 0; tile < TILE_TYPES; ++tile) if (counts[tile]) return tile;
    throw std::logic_error("single tile parser failed");
}

struct HandKey {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
    bool operator==(const HandKey& other) const { return low == other.low && high == other.high; }
};

struct HandKeyHash {
    std::size_t operator()(const HandKey& key) const {
        return static_cast<std::size_t>(splitmix64(key.low ^ splitmix64(key.high)));
    }
};

[[maybe_unused]] HandKey hand_key(const Counts& counts) {
    HandKey key;
    for (int tile = 0; tile < 21; ++tile) key.low |= std::uint64_t(counts[tile]) << (tile * 3);
    for (int tile = 21; tile < TILE_TYPES; ++tile) key.high |= std::uint64_t(counts[tile]) << ((tile - 21) * 3);
    return key;
}

class ShantenCalculator {
  public:
    ShantenCalculator()
#if defined(HAND_EFFICIENCY_USE_TILE_MASKS)
        : masks_(std::make_unique<mahjong::CalshtDW>(INDEX_FILE_PATH))
#endif
    {
#if !defined(HAND_EFFICIENCY_USE_TILE_MASKS) && !defined(HAND_EFFICIENCY_USE_NYANTEN)
        cache_.reserve(1'000'000);
        suit_cache_.reserve(100'000);
        honor_cache_.reserve(10'000);
#endif
    }

    int calculate(const Counts& counts) {
#if defined(HAND_EFFICIENCY_USE_TILE_MASKS)
        const auto [replacement, mode, mask] = (*masks_)(
            integer_counts(counts), tile_total(counts) / 3, 7, mahjong::CalcWait()
        );
        (void)mode;
        (void)mask;
        return replacement - 1;
#elif defined(HAND_EFFICIENCY_USE_NYANTEN)
        return static_cast<int>(Nyanten::calculateReplacementNumber(counts)) - 1;
#else
        const HandKey key = hand_key(counts);
        const auto found = cache_.find(key);
        if (found != cache_.end()) return found->second;
        const int value = std::min({regular(counts), chiitoitsu(counts), kokushi(counts)});
        if (cache_.size() >= 2'000'000) cache_.clear();
        cache_.emplace(key, value);
        return value;
#endif
    }

    std::uint64_t necessary_mask(Counts counts) {
#if defined(HAND_EFFICIENCY_USE_TILE_MASKS)
        const auto [replacement, mode, mask] = (*masks_)(
            integer_counts(counts), tile_total(counts) / 3, 7, mahjong::CalcWait()
        );
        (void)replacement;
        (void)mode;
        return mask;
#else
        const int current = calculate(counts);
        std::uint64_t mask = 0;
        for (int tile = 0; tile < TILE_TYPES; ++tile) {
            if (counts[tile] == 4) continue;
            ++counts[tile];
            if (calculate(counts) < current) mask |= std::uint64_t{1} << tile;
            --counts[tile];
        }
        return mask;
#endif
    }

    std::uint64_t discard_mask(Counts counts) {
#if defined(HAND_EFFICIENCY_USE_TILE_MASKS)
        const auto [replacement, mode, mask] = (*masks_)(
            integer_counts(counts), tile_total(counts) / 3, 7, mahjong::CalcDisc()
        );
        (void)replacement;
        (void)mode;
        return mask;
#else
        int best = 99;
        std::uint64_t mask = 0;
        for (int tile = 0; tile < TILE_TYPES; ++tile) {
            if (!counts[tile]) continue;
            --counts[tile];
            const int value = calculate(counts);
            ++counts[tile];
            if (value < best) {
                best = value;
                mask = 0;
            }
            if (value == best) mask |= std::uint64_t{1} << tile;
        }
        return mask;
#endif
    }

    std::size_t cache_size() const {
#if defined(HAND_EFFICIENCY_USE_TILE_MASKS) || defined(HAND_EFFICIENCY_USE_NYANTEN)
        return 0;
#else
        return cache_.size();
#endif
    }

  private:
#if defined(HAND_EFFICIENCY_USE_TILE_MASKS)
    std::unique_ptr<mahjong::CalshtDW> masks_;

    static std::array<int, TILE_TYPES> integer_counts(const Counts& counts) {
        std::array<int, TILE_TYPES> result{};
        std::copy(counts.begin(), counts.end(), result.begin());
        return result;
    }
#endif
    std::unordered_map<HandKey, int, HandKeyHash> cache_;
    std::unordered_map<std::uint32_t, std::uint64_t> suit_cache_;
    std::unordered_map<std::uint32_t, std::uint64_t> honor_cache_;

    static int chiitoitsu(const Counts& counts) {
        int pairs = 0;
        int kinds = 0;
        for (int count : counts) {
            pairs += count >= 2;
            kinds += count >= 1;
        }
        if (pairs == 7) return -1;
        return 6 - pairs + std::max(0, 7 - kinds);
    }

    static int kokushi(const Counts& counts) {
        int distinct = 0;
        bool pair = false;
        for (int tile : TERMINAL_HONORS) {
            distinct += counts[tile] > 0;
            pair = pair || counts[tile] >= 2;
        }
        return 13 - distinct - (pair ? 1 : 0);
    }

    static int state_bit(int melds, int taatsu, int head) {
        return (melds * 5 + taatsu) * 2 + head;
    }

    static void generate_block_states(
        std::array<std::uint8_t, 9>& counts, int length, bool sequences,
        int start, int melds, int taatsu, int head, std::uint64_t& mask
    ) {
        while (start < length && counts[start] == 0) ++start;
        if (start == length) {
            mask |= std::uint64_t{1} << state_bit(melds, taatsu, head);
            return;
        }
        if (melds < 4 && counts[start] >= 3) {
            counts[start] -= 3;
            generate_block_states(counts, length, sequences, start, melds + 1, taatsu, head, mask);
            counts[start] += 3;
        }
        if (melds < 4 && sequences && start <= 6 && counts[start + 1] && counts[start + 2]) {
            --counts[start]; --counts[start + 1]; --counts[start + 2];
            generate_block_states(counts, length, sequences, start, melds + 1, taatsu, head, mask);
            ++counts[start]; ++counts[start + 1]; ++counts[start + 2];
        }
        if (!head && counts[start] >= 2) {
            counts[start] -= 2;
            generate_block_states(counts, length, sequences, start, melds, taatsu, 1, mask);
            counts[start] += 2;
        }
        if (taatsu < 4 && counts[start] >= 2) {
            counts[start] -= 2;
            generate_block_states(counts, length, sequences, start, melds, taatsu + 1, head, mask);
            counts[start] += 2;
        }
        if (taatsu < 4 && sequences && start <= 7 && counts[start + 1]) {
            --counts[start]; --counts[start + 1];
            generate_block_states(counts, length, sequences, start, melds, taatsu + 1, head, mask);
            ++counts[start]; ++counts[start + 1];
        }
        if (taatsu < 4 && sequences && start <= 6 && counts[start + 2]) {
            --counts[start]; --counts[start + 2];
            generate_block_states(counts, length, sequences, start, melds, taatsu + 1, head, mask);
            ++counts[start]; ++counts[start + 2];
        }
        --counts[start];
        generate_block_states(counts, length, sequences, start, melds, taatsu, head, mask);
        ++counts[start];
    }

    std::uint64_t block_mask(const Counts& source, int offset, int length, bool sequences) {
        std::uint32_t code = 0;
        std::uint32_t place = 1;
        std::array<std::uint8_t, 9> counts{};
        for (int index = 0; index < length; ++index) {
            counts[index] = source[offset + index];
            code += counts[index] * place;
            place *= 5;
        }
        auto& cache = sequences ? suit_cache_ : honor_cache_;
        const auto found = cache.find(code);
        if (found != cache.end()) return found->second;
        std::uint64_t mask = 0;
        generate_block_states(counts, length, sequences, 0, 0, 0, 0, mask);
        cache.emplace(code, mask);
        return mask;
    }

    static std::uint64_t combine_states(std::uint64_t left, std::uint64_t right) {
        std::uint64_t result = 0;
        for (int left_bit = 0; left_bit < 50; ++left_bit) {
            if (!(left & (std::uint64_t{1} << left_bit))) continue;
            const int left_head = left_bit % 2;
            const int left_taatsu = (left_bit / 2) % 5;
            const int left_melds = left_bit / 10;
            for (int right_bit = 0; right_bit < 50; ++right_bit) {
                if (!(right & (std::uint64_t{1} << right_bit))) continue;
                const int head = left_head + right_bit % 2;
                if (head > 1) continue;
                const int melds = left_melds + right_bit / 10;
                if (melds > 4) continue;
                const int taatsu = std::min(4, left_taatsu + (right_bit / 2) % 5);
                result |= std::uint64_t{1} << state_bit(melds, taatsu, head);
            }
        }
        return result;
    }

    int regular(const Counts& source) {
        std::uint64_t states = std::uint64_t{1} << state_bit(0, 0, 0);
        states = combine_states(states, block_mask(source, 0, 9, true));
        states = combine_states(states, block_mask(source, 9, 9, true));
        states = combine_states(states, block_mask(source, 18, 9, true));
        states = combine_states(states, block_mask(source, 27, 7, false));
        int best = 8;
        for (int bit = 0; bit < 50; ++bit) {
            if (!(states & (std::uint64_t{1} << bit))) continue;
            const int head = bit % 2;
            const int taatsu = (bit / 2) % 5;
            const int melds = bit / 10;
            best = std::min(best, 8 - 2 * melds - std::min(taatsu, 4 - melds) - head);
        }
        return best;
    }
};

struct Ukeire {
    std::vector<int> types;
    std::vector<int> remaining;
    int count = 0;
};

std::vector<int> structural_ukeire_types(Counts counts, ShantenCalculator& calculator) {
    std::vector<int> result;
    const std::uint64_t mask = calculator.necessary_mask(counts);
    for (int tile = 0; tile < TILE_TYPES; ++tile) {
        if (mask & (std::uint64_t{1} << tile)) result.push_back(tile);
    }
    return result;
}

Ukeire visible_ukeire(const Counts& hand, const Visible& external, ShantenCalculator& calculator) {
    Ukeire result;
    for (int tile : structural_ukeire_types(hand, calculator)) {
        const int remaining = std::max(0, 4 - int(hand[tile]) - external[tile]);
        if (!remaining) continue;
        result.types.push_back(tile);
        result.remaining.push_back(remaining);
        result.count += remaining;
    }
    return result;
}

struct PostDrawChoice {
    int shanten = 99;
    int ukeire_count = 0;
};

PostDrawChoice best_post_draw_choice(
    Counts hand, int draw_type, const Visible& external, ShantenCalculator& calculator
) {
    if (hand[draw_type] + external[draw_type] >= 4) throw std::logic_error("hypothetical fifth tile");
    ++hand[draw_type];
    const int best_shanten = calculator.calculate(hand);
    const std::uint64_t discards = calculator.discard_mask(hand);
    int best_ukeire = 0;
    for (int discard = 0; discard < TILE_TYPES; ++discard) {
        if (!(discards & (std::uint64_t{1} << discard)) || !hand[discard]) continue;
        --hand[discard];
        Visible next_external = external;
        ++next_external[discard];
        best_ukeire = std::max(best_ukeire, visible_ukeire(hand, next_external, calculator).count);
        ++hand[discard];
    }
    return {best_shanten, best_ukeire};
}

struct AdvancedFeatures {
    std::vector<double> future_by_type;
    double future_expected = 0.0;
    std::vector<int> improvement_types;
    std::vector<int> improvement_remaining;
    int improvement_count = 0;
};

struct PolicyConfig {
    std::string preset = "pure";
    double dora_ukeire_weight = 0.0;
    double dora_tiebreak_weight = 0.0;
    double dora_neighbor_weight = 0.0;
    double yakuhai_weight = 0.0;
    double tanyao_weight = 0.0;
    double kanchan_outer_wait_weight = 0.0;

    bool preserve_red_five() const {
        return dora_ukeire_weight != 0.0 || dora_tiebreak_weight != 0.0;
    }

    std::string name() const {
        if (preset == "pure") return "shanten_then_ukeire_score";
        if (preset == "value") return "shanten_then_value_ukeire";
        return "custom_hand_value";
    }
};

struct FutureUkeireConfig {
    double fallback = 0.5;
    double one_shanten = std::numeric_limits<double>::quiet_NaN();
    double two_shanten = std::numeric_limits<double>::quiet_NaN();

    double weight(int shanten) const {
        if (shanten == 1) return std::isnan(one_shanten) ? fallback : one_shanten;
        if (shanten == 2) return std::isnan(two_shanten) ? fallback : two_shanten;
        return 0.0;
    }
};

struct ReversalConfig {
    bool enabled = false;
    double penalty = 25.0;
    double future_weight = 0.5;
    int max_turn = 9;
    int candidate_limit = 2;
    int max_per_hand = 1;
};

AdvancedFeatures advanced_features(
    const Counts& hand, int hand_shanten, const Visible& external, const Ukeire& ukeire,
    double future_weight, double improvement_weight, ShantenCalculator& calculator
) {
    AdvancedFeatures result;
    const bool future_active = (hand_shanten == 1 || hand_shanten == 2) && future_weight != 0.0;
    const bool improvement_active = (hand_shanten == 1 || hand_shanten == 2) && improvement_weight != 0.0;
    if (!future_active && !improvement_active) {
        return result;
    }
    if (future_active) {
        double weighted = 0.0;
        for (std::size_t index = 0; index < ukeire.types.size(); ++index) {
            const auto choice = best_post_draw_choice(hand, ukeire.types[index], external, calculator);
            result.future_by_type.push_back(choice.ukeire_count);
            weighted += choice.ukeire_count * ukeire.remaining[index];
        }
        if (ukeire.count) result.future_expected = weighted / ukeire.count;
    }
    if (!improvement_active) return result;

    std::array<bool, TILE_TYPES> direct{};
    for (int tile : ukeire.types) direct[tile] = true;
    for (int tile = 0; tile < TILE_TYPES; ++tile) {
        const int remaining = std::max(0, 4 - int(hand[tile]) - external[tile]);
        if (!remaining || direct[tile]) continue;
        const auto choice = best_post_draw_choice(hand, tile, external, calculator);
        if (choice.shanten == hand_shanten && choice.ukeire_count > ukeire.count) {
            result.improvement_types.push_back(tile);
            result.improvement_remaining.push_back(remaining);
            result.improvement_count += remaining;
        }
    }
    return result;
}

struct PolicyFeatures {
    int discard_type = -1;
    int shanten = 99;
    Ukeire ukeire;
    AdvancedFeatures advanced;
    double base_score = -std::numeric_limits<double>::infinity();
    double dora_ukeire_bonus = 0.0;
    double score = -std::numeric_limits<double>::infinity();
    int retained_dora = 0;
    int retained_dora_neighbors = 0;
    int retained_yakuhai_value = 0;
    int retained_tanyao_tiles = 0;
    double tie_score = 0.0;
    int strict_best_shanten = 99;
    bool reversal_eligible = false;
    bool reversal_deep_evaluated = false;
    bool is_reversal = false;
    int reversal_recovery_count = 0;
    double reversal_future_expected = 0.0;
    int reversal_current_ukeire = 0;
    double reversal_ukeire_gain = 0.0;
    double reversal_adjusted_score = -std::numeric_limits<double>::infinity();
};

bool is_pure_outer_kanchan(const Counts& hand, ShantenCalculator& calculator);

PolicyFeatures evaluate_13(
    const Counts& hand, int discard_type, const Visible& external,
    const FutureUkeireConfig& future, double improvement_weight, ShantenCalculator& calculator
) {
    PolicyFeatures features;
    features.discard_type = discard_type;
    features.shanten = calculator.calculate(hand);
    const double future_weight = future.weight(features.shanten);
    features.ukeire = visible_ukeire(hand, external, calculator);
    features.advanced = advanced_features(
        hand, features.shanten, external, features.ukeire,
        future_weight, improvement_weight, calculator
    );
    features.base_score = features.ukeire.count
        + future_weight * features.advanced.future_expected
        + improvement_weight * features.advanced.improvement_count;
    features.score = features.base_score;
    return features;
}

std::optional<std::size_t> representative_discard(
    const std::vector<int>& hand, int discard_type, bool preserve_red
) {
    std::optional<std::size_t> fallback;
    for (std::size_t index = 0; index < hand.size(); ++index) {
        if (hand[index] / 4 != discard_type) continue;
        if (!fallback) fallback = index;
        if (!preserve_red || !is_red_five(hand[index])) return index;
    }
    return fallback;
}

void apply_value_features(
    PolicyFeatures& features, const std::vector<int>& hand, int dora_indicator,
    int player_wind, int round_wind, const PolicyConfig& policy
) {
    const auto discarded = features.discard_type >= 0
        ? representative_discard(hand, features.discard_type, policy.preserve_red_five())
        : std::optional<std::size_t>{};
    const int dora = dora_indicator >= 0 ? dora_type_from_indicator(dora_indicator / 4) : -1;
    for (std::size_t index = 0; index < hand.size(); ++index) {
        if (discarded && index == *discarded) continue;
        const int tile_id = hand[index];
        const int tile = tile_id / 4;
        features.retained_dora += int(is_red_five(tile_id)) + int(dora >= 0 && tile == dora);
        features.retained_dora_neighbors += int(is_dora_neighbor(tile, dora));
        features.retained_yakuhai_value += yakuhai_value(tile, player_wind, round_wind);
        features.retained_tanyao_tiles += int(is_tanyao_tile(tile));
    }
    features.dora_ukeire_bonus = policy.dora_ukeire_weight * features.retained_dora;
    features.score = features.base_score + features.dora_ukeire_bonus;
    features.tie_score =
        policy.dora_tiebreak_weight * features.retained_dora
        + policy.dora_neighbor_weight * features.retained_dora_neighbors
        + policy.yakuhai_weight * features.retained_yakuhai_value
        + policy.tanyao_weight * features.retained_tanyao_tiles;
}

struct DiscardChoice {
    int discarded = -1;
    PolicyFeatures features;
    bool reversal_opportunity = false;
    double strict_score = -std::numeric_limits<double>::infinity();
};

DiscardChoice choose_discard(
    std::vector<int>& hand, const std::vector<int>& river, int dora_indicator,
    int player_wind, int round_wind, const FutureUkeireConfig& future, double improvement_weight,
    const PolicyConfig& policy, const ReversalConfig& reversal, int turn, int reversals_used,
    Rng& rng, ShantenCalculator& calculator
) {
    Counts hand_counts = counts_from_physical(hand);
    std::vector<std::pair<int, Counts>> post_hands;
    const int best_shanten = calculator.calculate(hand_counts);
    const std::uint64_t discards = calculator.discard_mask(hand_counts);
    const bool reversal_active = reversal.enabled && (best_shanten == 0 || best_shanten == 1)
        && turn <= reversal.max_turn && reversals_used < reversal.max_per_hand;
    for (int discard = 0; discard < TILE_TYPES; ++discard) {
        if (!hand_counts[discard]) continue;
        if (!reversal_active && !(discards & (std::uint64_t{1} << discard))) continue;
        --hand_counts[discard];
        post_hands.emplace_back(discard, hand_counts);
        ++hand_counts[discard];
    }

    Visible base_visible{};
    for (int tile : river) ++base_visible[tile / 4];
    ++base_visible[dora_indicator / 4];
    std::vector<PolicyFeatures> candidates;
    std::vector<std::size_t> reversal_indices;
    double strict_score = -std::numeric_limits<double>::infinity();
    double best_tie_score = -std::numeric_limits<double>::infinity();
    for (const auto& item : post_hands) {
        Visible external = base_visible;
        ++external[item.first];
        const int post_shanten = calculator.calculate(item.second);
        PolicyFeatures features;
        if (post_shanten == best_shanten) {
            features = evaluate_13(
                item.second, item.first, external, future, improvement_weight, calculator
            );
        } else {
            features.discard_type = item.first;
            features.shanten = post_shanten;
            if (reversal_active && post_shanten == best_shanten + 1) {
                features.ukeire = visible_ukeire(item.second, external, calculator);
                features.base_score = features.ukeire.count;
                features.score = features.base_score;
                features.reversal_eligible = features.ukeire.count > 0;
            }
        }
        features.strict_best_shanten = best_shanten;
        apply_value_features(features, hand, dora_indicator, player_wind, round_wind, policy);
        if (policy.kanchan_outer_wait_weight != 0.0 && features.shanten == 0
            && is_pure_outer_kanchan(item.second, calculator)) {
            features.tie_score += policy.kanchan_outer_wait_weight;
        }
        if (features.shanten == best_shanten) {
            strict_score = std::max(strict_score, features.score);
        }
        candidates.push_back(std::move(features));
        if (candidates.back().reversal_eligible) {
            reversal_indices.push_back(candidates.size() - 1);
        }
    }

    const bool reversal_opportunity = !reversal_indices.empty();
    std::sort(reversal_indices.begin(), reversal_indices.end(), [&](std::size_t left, std::size_t right) {
        const auto& a = candidates[left];
        const auto& b = candidates[right];
        const double a_score = a.ukeire.count + a.dora_ukeire_bonus;
        const double b_score = b.ukeire.count + b.dora_ukeire_bonus;
        if (std::abs(a_score - b_score) > 1e-12) return a_score > b_score;
        if (std::abs(a.tie_score - b.tie_score) > 1e-12) return a.tie_score > b.tie_score;
        return a.discard_type < b.discard_type;
    });
    if (reversal_indices.size() > static_cast<std::size_t>(reversal.candidate_limit)) {
        reversal_indices.resize(static_cast<std::size_t>(reversal.candidate_limit));
    }
    double best_reversal_score = -std::numeric_limits<double>::infinity();
    int strict_current_ukeire = 0;
    for (const auto& features : candidates) {
        if (features.shanten == best_shanten) {
            strict_current_ukeire = std::max(strict_current_ukeire, features.ukeire.count);
        }
    }
    for (std::size_t index : reversal_indices) {
        auto& features = candidates[index];
        const auto found = std::find_if(post_hands.begin(), post_hands.end(), [&](const auto& item) {
            return item.first == features.discard_type;
        });
        Visible external = base_visible;
        ++external[features.discard_type];
        FutureUkeireConfig reversal_future;
        reversal_future.fallback = reversal.future_weight;
        reversal_future.one_shanten = reversal.future_weight;
        reversal_future.two_shanten = reversal.future_weight;
        features = evaluate_13(found->second, features.discard_type, external, reversal_future, 0.0, calculator);
        features.strict_best_shanten = best_shanten;
        features.reversal_eligible = true;
        features.reversal_deep_evaluated = true;
        features.is_reversal = true;
        features.reversal_recovery_count = features.ukeire.count;
        features.reversal_future_expected = features.advanced.future_expected;
        features.reversal_current_ukeire = strict_current_ukeire;
        features.reversal_ukeire_gain = features.reversal_future_expected - strict_current_ukeire;
        features.base_score = features.ukeire.count
            + reversal.future_weight * features.reversal_ukeire_gain;
        apply_value_features(features, hand, dora_indicator, player_wind, round_wind, policy);
        features.reversal_adjusted_score = features.score - reversal.penalty;
        if (features.reversal_ukeire_gain > 1e-12) {
            best_reversal_score = std::max(best_reversal_score, features.reversal_adjusted_score);
        }
    }
    const bool reversal_wins = best_reversal_score > strict_score + 1e-12;
    std::vector<PolicyFeatures> best;
    double best_score = -std::numeric_limits<double>::infinity();
    best_tie_score = -std::numeric_limits<double>::infinity();
    for (const auto& features : candidates) {
        const bool qualifies_as_reversal = features.reversal_deep_evaluated
            && features.reversal_ukeire_gain > 1e-12;
        if (reversal_wins != qualifies_as_reversal) continue;
        if (!reversal_wins && features.shanten != best_shanten) continue;
        const double score = reversal_wins ? features.reversal_adjusted_score : features.score;
        if (score > best_score + 1e-12
            || (std::abs(score - best_score) <= 1e-12
                && features.tie_score > best_tie_score + 1e-12)) {
            best_score = score;
            best_tie_score = features.tie_score;
            best.clear();
        }
        if (std::abs(score - best_score) <= 1e-12
            && std::abs(features.tie_score - best_tie_score) <= 1e-12) {
            best.push_back(features);
        }
    }
    PolicyFeatures chosen = best.at(static_cast<std::size_t>(rng.uniform(best.size())));
    std::vector<std::size_t> physical;
    for (std::size_t index = 0; index < hand.size(); ++index) {
        if (hand[index] / 4 == chosen.discard_type) physical.push_back(index);
    }
    if (policy.preserve_red_five()) {
        std::vector<std::size_t> ordinary;
        for (std::size_t index : physical) {
            if (!is_red_five(hand[index])) ordinary.push_back(index);
        }
        if (!ordinary.empty()) physical = std::move(ordinary);
    }
    const std::size_t selected = physical.at(static_cast<std::size_t>(rng.uniform(physical.size())));
    const int discarded = hand[selected];
    hand.erase(hand.begin() + static_cast<std::ptrdiff_t>(selected));
    return {discarded, chosen, reversal_opportunity, strict_score};
}

bool all_melds(Counts& counts, int start = 0) {
    while (start < TILE_TYPES && counts[start] == 0) ++start;
    if (start == TILE_TYPES) return true;
    if (counts[start] >= 3) {
        counts[start] -= 3;
        if (all_melds(counts, start)) {
            counts[start] += 3;
            return true;
        }
        counts[start] += 3;
    }
    const int rank = start % 9;
    if (start < 27 && rank <= 6 && counts[start + 1] && counts[start + 2]) {
        --counts[start]; --counts[start + 1]; --counts[start + 2];
        if (all_melds(counts, start)) {
            ++counts[start]; ++counts[start + 1]; ++counts[start + 2];
            return true;
        }
        ++counts[start]; ++counts[start + 1]; ++counts[start + 2];
    }
    return false;
}

bool standard_complete(Counts counts) {
    if (tile_total(counts) % 3 != 2) return false;
    for (int pair = 0; pair < TILE_TYPES; ++pair) {
        if (counts[pair] < 2) continue;
        counts[pair] -= 2;
        const bool valid = all_melds(counts);
        counts[pair] += 2;
        if (valid) return true;
    }
    return false;
}

bool chiitoitsu_complete(const Counts& counts) {
    int pairs = 0;
    for (int count : counts) pairs += count == 2;
    return tile_total(counts) == 14 && pairs == 7;
}

bool kokushi_complete(const Counts& counts) {
    if (tile_total(counts) != 14) return false;
    int total = 0;
    bool pair = false;
    for (int tile : TERMINAL_HONORS) {
        if (!counts[tile]) return false;
        total += counts[tile];
        pair = pair || counts[tile] >= 2;
    }
    return total == 14 && pair;
}

std::vector<int> wait_shapes(const Counts& hand_13, int winning_type, ShantenCalculator& calculator) {
    if (hand_13[winning_type] == 4) return {};
    Counts complete = hand_13;
    ++complete[winning_type];
    if (calculator.calculate(complete) != -1) return {};
    std::array<bool, 8> shapes{};
    if (chiitoitsu_complete(complete)) shapes[5] = true;
    if (kokushi_complete(complete)) {
        int distinct = 0;
        for (int tile : TERMINAL_HONORS) distinct += hand_13[tile] > 0;
        shapes[distinct == 13 ? 6 : 7] = true;
    }

    if (complete[winning_type] >= 2) {
        complete[winning_type] -= 2;
        if (all_melds(complete)) shapes[4] = true;
        complete[winning_type] += 2;
    }
    if (complete[winning_type] >= 3) {
        complete[winning_type] -= 3;
        if (standard_complete(complete)) shapes[3] = true;
        complete[winning_type] += 3;
    }
    if (winning_type < 27) {
        const int suit_start = (winning_type / 9) * 9;
        const int rank = winning_type % 9;
        for (int start_rank = std::max(0, rank - 2); start_rank <= std::min(rank, 6); ++start_rank) {
            const int start = suit_start + start_rank;
            if (!complete[start] || !complete[start + 1] || !complete[start + 2]) continue;
            --complete[start]; --complete[start + 1]; --complete[start + 2];
            const bool valid = standard_complete(complete);
            ++complete[start]; ++complete[start + 1]; ++complete[start + 2];
            if (!valid) continue;
            if (rank == start_rank + 1) shapes[1] = true;
            else if ((start_rank == 0 && rank == 2) || (start_rank == 6 && rank == 6)) shapes[2] = true;
            else shapes[0] = true;
        }
    }
    std::vector<int> result;
    for (int index = 0; index < 8; ++index) if (shapes[index]) result.push_back(index);
    return result;
}

bool is_pure_outer_kanchan(const Counts& hand, ShantenCalculator& calculator) {
    if (calculator.calculate(hand) != 0) return false;
    const auto waits = structural_ukeire_types(hand, calculator);
    if (waits.size() != 1) return false;
    const int rank = rank_of(waits[0]);
    if (rank != 2 && rank != 8) return false;
    const auto shapes = wait_shapes(hand, waits[0], calculator);
    return std::find(shapes.begin(), shapes.end(), 1) != shapes.end();
}

struct WaitInfo {
    int tile_type = -1;
    int visible_remaining = 0;
    int live_remaining = 0;
    std::vector<int> visible_remaining_tile_ids;
    std::vector<int> live_remaining_tile_ids;
    std::vector<int> shapes;
};

struct Snapshot {
    int turn = 0;
    Counts hand{};
    std::vector<int> hand_physical;
    std::string signature;
    std::vector<WaitInfo> waits;
    int visible_ukeire = 0;
    int live_ukeire = 0;
    int live_wall_tiles_remaining = 0;
    bool furiten = false;
    bool riichi_declared = false;
};

bool is_permanent_furiten(
    const Counts& hand, const std::vector<int>& river, ShantenCalculator& calculator
) {
    std::array<bool, TILE_TYPES> wait_types{};
    for (int tile : structural_ukeire_types(hand, calculator)) wait_types[tile] = true;
    return std::any_of(river.begin(), river.end(), [&](int tile) {
        return wait_types[tile / 4];
    });
}

Snapshot make_snapshot(
    const std::vector<int>& hand, const std::vector<int>& visible_river,
    const std::vector<int>& furiten_river, int dora_indicator,
    const std::vector<int>& live_wall, std::size_t cursor, int turn, bool riichi,
    ShantenCalculator& calculator
) {
    Snapshot snapshot;
    snapshot.turn = turn;
    snapshot.hand = counts_from_physical(hand);
    snapshot.hand_physical = hand;
    std::sort(snapshot.hand_physical.begin(), snapshot.hand_physical.end());
    snapshot.riichi_declared = riichi;
    snapshot.live_wall_tiles_remaining = static_cast<int>(live_wall.size() - cursor);
    std::array<bool, PHYSICAL_TILES> known{};
    for (int tile : hand) known[tile] = true;
    for (int tile : visible_river) known[tile] = true;
    known[dora_indicator] = true;
    Visible external{};
    for (int tile : visible_river) ++external[tile / 4];
    ++external[dora_indicator / 4];
    Visible live{};
    for (std::size_t index = cursor; index < live_wall.size(); ++index) ++live[live_wall[index] / 4];
    std::array<bool, 8> signature_shapes{};
    const auto waits = structural_ukeire_types(snapshot.hand, calculator);
    snapshot.furiten = is_permanent_furiten(snapshot.hand, furiten_river, calculator);
    for (int tile : waits) {
        WaitInfo wait;
        wait.tile_type = tile;
        wait.visible_remaining = std::max(0, 4 - int(snapshot.hand[tile]) - external[tile]);
        wait.live_remaining = live[tile];
        for (int copy = 0; copy < 4; ++copy) {
            const int tile_id = tile * 4 + copy;
            if (!known[tile_id]) wait.visible_remaining_tile_ids.push_back(tile_id);
        }
        for (std::size_t index = cursor; index < live_wall.size(); ++index) {
            if (live_wall[index] / 4 == tile) wait.live_remaining_tile_ids.push_back(live_wall[index]);
        }
        if (static_cast<int>(wait.visible_remaining_tile_ids.size()) != wait.visible_remaining
            || static_cast<int>(wait.live_remaining_tile_ids.size()) != wait.live_remaining) {
            throw std::logic_error("wait copy identity count mismatch");
        }
        wait.shapes = wait_shapes(snapshot.hand, tile, calculator);
        for (int shape : wait.shapes) signature_shapes[shape] = true;
        snapshot.visible_ukeire += wait.visible_remaining;
        snapshot.live_ukeire += wait.live_remaining;
        snapshot.waits.push_back(std::move(wait));
    }
    bool first = true;
    for (int shape = 0; shape < 8; ++shape) {
        if (!signature_shapes[shape]) continue;
        if (!first) snapshot.signature += "+";
        snapshot.signature += SHAPE_NAMES[shape];
        first = false;
    }
    return snapshot;
}

struct WinInfo {
    int turn = 0;
    int tile = -1;
    std::vector<int> hand;
    std::vector<int> shapes;
    bool riichi = false;
    bool ippatsu = false;
    bool haitei = false;
    bool tenhou = false;
    bool chiihou = false;
};

struct ScoreReply {
    bool good_wait = false, cheap_all_unseen = false, exact_1300_bad_wait = false;
    bool valuable_dama = false; std::uint64_t valuable_ron_mask = 0;
    bool haneman_lock = false; std::uint64_t haneman_ron_mask = 0, legal_ron_mask = 0;
    int min_dama_ron_points = 0, min_riichi_ron_points = 0;
    double mean_riichi_ron_points = 0.0;
};

struct PushFoldCandidate {
    int physical_id = -1;
    int tile_type = -1;
    std::vector<int> safety_tiers;
    int known_copies = 0;
    bool chosen = false;
    std::vector<int> ron_hits;
};

struct PushFoldSafeTypes {
    int threat_seat = -1;
    std::vector<int> types;
};

struct PushFoldDecision {
    int seat = -1;
    int turn = 0;
    int draw_index = 0;
    int drawn_physical = -1;
    int live_wall_remaining = 0;
    bool own_riichi = false;
    std::vector<int> threat_seats;
    std::string mode;
    std::string mode_reason;
    int best_shanten = 99;
    int post_discard_shanten = 99;
    Counts own_counts{};
    Counts public_counts{};
    std::vector<PushFoldSafeTypes> safe_types;
    std::vector<PushFoldCandidate> candidates;
    bool has_selected_score = false;
    ScoreReply selected_score;
};

struct PushFoldWinner {
    int seat = -1;
    std::vector<int> hand_physical;
    int winning_tile = -1;
    bool is_riichi = false;
    bool is_ippatsu = false;
    bool is_haitei = false;
};

struct PushFoldTerminal {
    std::string reason;
    std::vector<PushFoldWinner> winners;
    int loser = -1;
    int riichi_sticks = 0;
    std::vector<int> tenpai_seats;
    std::array<Counts, 4> final_hand_counts{};
};

struct PushFoldRiichiEvent {
    int seat = -1;
    int turn = 0;
    std::vector<int> hand_physical13;
    std::vector<int> winning_types;
    int unseen_wait_copies = 0;
    int live_wait_copies = 0;
    int live_wall_remaining = 0;
    bool furiten = false;
};

struct PushFoldResult {
    std::string mode;
    std::string safety_strategy;
    int dora_danger_steps = 1;
    int half_push_max_tier = 5;
    int two_riichi_bad_wait_min_ron = 7700;
    std::string ippatsu_model = "excluded";
    std::vector<int> riichi_seats;
    std::vector<PushFoldRiichiEvent> riichi_events;
    std::vector<PushFoldDecision> decisions;
    PushFoldTerminal terminal;
};

struct TrialResult {
    std::uint64_t trial_id = 0;
    std::uint64_t seed = 0;
    std::string outcome;
    std::string initial_hand;
    std::vector<int> initial_physical;
    int dora_indicator = -1;
    int ura_indicator = -1;
    std::optional<Snapshot> first_tenpai;
    std::optional<WinInfo> win;
    struct TableEvent {
        std::string kind;
        int seat = -1;
        int turn = 0;
        std::optional<Snapshot> tenpai;
        std::optional<WinInfo> win;
    };
    std::vector<TableEvent> table_events;
    struct ReversalRecord {
        int turn = 0;
        int from_shanten = 99;
        int to_shanten = 99;
        int discard_type = -1;
        int recovery_count = 0;
        double recovered_ukeire_expected = 0.0;
        int current_ukeire = 0;
        double ukeire_gain = 0.0;
        double strict_score = 0.0;
        double adjusted_score = 0.0;
        bool recovered = false;
        int recovery_turns = 0;
    };
    std::optional<ReversalRecord> reversal;
    int reversal_opportunities = 0;
    struct AvailabilityPlayer {
        int seat = -1;
        Counts own_known_counts{};
        Counts public_counts{};
        Counts unseen_counts{};
        Counts live_wall_counts{};
        Counts hidden_dead_wall_counts{};
        Counts opponent_hand_counts{};
        int unseen_tiles = 0;
        int live_wall_tiles = 0;
        int hidden_dead_wall_tiles = 0;
        int opponent_hand_tiles = 0;
    };
    struct AvailabilityCheckpoint {
        int target_own_turn = -1;
        int completed_own_turn = -1;
        int dora_indicator_type = -1;
        int live_wall_tiles_remaining = 0;
        int hidden_dead_wall_tiles_remaining = 0;
        int opponent_hand_tiles = 0;
        Counts public_counts{};
        std::array<AvailabilityPlayer, 4> players{};
    };
    std::optional<AvailabilityCheckpoint> availability_checkpoint;
    struct DangerCandidate {
        int tile_type = -1;
        bool ron_hit = false;
        bool chosen = false;
    };
    struct DangerDecision {
        int observer_seat = -1, observer_turn = 0, riichi_player_turn = 0, turns_since_riichi = 0;
        int draw_index = 0, live_wall_remaining = 0;
        Counts own_counts{}, public_counts{}, riichi_river_counts{};
        std::vector<int> safe_types;
        std::vector<DangerCandidate> candidates;
    };
    struct Danger {
        int riichi_seat = -1, riichi_turn = -1, riichi_draw_index = -1;
        Counts riichi_hand_counts{};
        std::vector<int> winning_types;
        bool furiten_at_riichi = false;
        std::string terminal_reason = "no_riichi_exposure";
        std::vector<DangerDecision> decisions;
    };
    std::optional<Danger> danger;
    std::optional<PushFoldResult> push_fold;
};

struct Config {
    std::uint64_t trials = 1000;
    std::uint64_t seed = DEFAULT_SEED;
    std::uint64_t start_trial = 0;
    std::string stop_at = "win";
    bool riichi_on_tenpai = true;
    bool continue_table_tenpai = false;
    std::string valuable_dama_directives_path;
    std::unordered_map<std::string, std::uint64_t> valuable_dama_directives;
    int opponent_draws = 3;
    int player_wind = 28;
    int round_wind = 27;
    FutureUkeireConfig future;
    double improvement_weight = 0.1;
    PolicyConfig policy;
    ReversalConfig reversal;
    std::array<bool, 5> policy_weight_set{};
    std::string observations;
    int tile_availability_turn = -1;
    bool tile_danger = false;
    bool push_fold = false;
    std::string push_fold_mode = "mixed";
    std::string safety_strategy = "mean_safety";
    int dora_danger_steps = 1;
    int half_push_max_tier = 5;
    int two_riichi_bad_wait_min_ron = 7700;
    bool score_service = false;
};

ScoreReply score_service_request(
    const Config& config, std::uint64_t trial_id, int seat, int turn,
    const std::vector<int>& hand, const std::vector<int>& known, int dora_indicator
) {
    if (!config.score_service) throw std::logic_error("push-fold value decision requires --score-service");
    std::vector<int> sorted_hand = hand, sorted_known = known;
    std::sort(sorted_hand.begin(), sorted_hand.end());
    std::sort(sorted_known.begin(), sorted_known.end());
    std::ostringstream cache_key;
    cache_key << seat << ':' << dora_indicator << ':' << config.round_wind << ':';
    for (int tile : sorted_hand) cache_key << tile << ',';
    cache_key << ':';
    for (int tile : sorted_known) cache_key << tile << ',';
    static std::unordered_map<std::string, ScoreReply> cache;
    if (const auto found = cache.find(cache_key.str()); found != cache.end()) return found->second;
    std::cout << "{\"kind\":\"score_request\",\"trial_id\":" << trial_id
        << ",\"seat\":" << seat << ",\"turn\":" << turn
        << ",\"hand_physical\":" << numeric_array_json(hand)
        << ",\"known_physical\":" << numeric_array_json(known)
        << ",\"dora_indicator\":" << dora_indicator
        << ",\"round_wind\":" << config.round_wind << "}" << std::endl;
    std::string line;
    if (!std::getline(std::cin, line)) throw std::runtime_error("score service EOF");
    std::istringstream in(line); ScoreReply reply; int good=0, cheap=0, exact=0, valuable=0, haneman=0;
    if (!(in >> good >> cheap >> exact >> valuable >> reply.valuable_ron_mask >> haneman
          >> reply.haneman_ron_mask >> reply.legal_ron_mask >> reply.min_dama_ron_points
          >> reply.min_riichi_ron_points >> reply.mean_riichi_ron_points)) {
        throw std::runtime_error("invalid score service response");
    }
    std::string extra; if (in >> extra) throw std::runtime_error("invalid score service response fields");
    reply.good_wait=good; reply.cheap_all_unseen=cheap; reply.exact_1300_bad_wait=exact;
    reply.valuable_dama=valuable; reply.haneman_lock=haneman;
    cache.emplace(cache_key.str(), reply);
    return reply;
}

std::string dama_directive_key(
    std::uint64_t trial_id, int seat, int turn, const Counts& hand
) {
    std::ostringstream out;
    out << trial_id << ':' << seat << ':' << turn << ':' << format_counts(hand);
    return out.str();
}

void load_dama_directives(Config& config) {
    if (config.valuable_dama_directives_path.empty()) return;
    std::ifstream input(config.valuable_dama_directives_path);
    if (!input) throw std::runtime_error("could not open valuable dama directives");
    std::string key;
    std::uint64_t mask = 0;
    while (input >> key >> mask) config.valuable_dama_directives[key] = mask;
}

void finalize_policy(Config& config) {
    if (config.policy.preset != "pure" && config.policy.preset != "value"
        && config.policy.preset != "custom") {
        throw std::invalid_argument("policy must be pure, value, or custom");
    }
    std::array<double*, 5> weights = {
        &config.policy.dora_ukeire_weight,
        &config.policy.dora_tiebreak_weight,
        &config.policy.dora_neighbor_weight,
        &config.policy.yakuhai_weight,
        &config.policy.tanyao_weight,
    };
    const double default_weight = config.policy.preset == "value" ? 1.0 : 0.0;
    for (std::size_t index = 0; index < weights.size(); ++index) {
        if (!config.policy_weight_set[index]) *weights[index] = default_weight;
        if (!std::isfinite(*weights[index]) || *weights[index] < 0.0) {
            throw std::invalid_argument("policy weights must be finite and nonnegative");
        }
        if (config.policy.preset == "pure" && *weights[index] != 0.0) {
            throw std::invalid_argument("hand-value weights require --policy value or custom");
        }
    }
    const std::array<double, 3> future_weights = {
        config.future.fallback, config.future.one_shanten, config.future.two_shanten,
    };
    if (config.improvement_weight < 0.0 || !std::isfinite(config.improvement_weight)
        || std::any_of(future_weights.begin(), future_weights.end(), [](double weight) {
            return !std::isnan(weight) && (!std::isfinite(weight) || weight < 0.0);
        })) {
        throw std::invalid_argument("ukeire weights must be finite and nonnegative");
    }
    if (!std::isfinite(config.reversal.penalty) || config.reversal.penalty < 0.0
        || !std::isfinite(config.reversal.future_weight) || config.reversal.future_weight < 0.0
        || config.reversal.max_turn < 0 || config.reversal.candidate_limit < 1
        || config.reversal.max_per_hand < 1) {
        throw std::invalid_argument("invalid shanten reversal configuration");
    }
    if (!std::isfinite(config.policy.kanchan_outer_wait_weight)
        || config.policy.kanchan_outer_wait_weight < 0.0) {
        throw std::invalid_argument("kanchan outer wait weight must be finite and nonnegative");
    }
    if (config.policy.preset == "pure" && config.policy.kanchan_outer_wait_weight != 0.0) {
        throw std::invalid_argument("kanchan outer wait weight requires --policy value or custom");
    }
}

TrialResult simulate_trial(const Config& config, std::uint64_t trial_id, ShantenCalculator& calculator) {
    TrialResult result;
    result.trial_id = trial_id;
    result.seed = trial_seed(config.seed, trial_id);
    Rng wall_rng(splitmix64(result.seed ^ 0x57414C4CULL));
    Rng policy_rng(splitmix64(result.seed ^ 0x504F4C494359ULL));
    std::vector<int> wall(PHYSICAL_TILES);
    std::iota(wall.begin(), wall.end(), 0);
    wall_rng.shuffle(wall);
    std::vector<int> hand(wall.begin(), wall.begin() + INITIAL_HAND);
    result.initial_physical = hand;
    std::sort(result.initial_physical.begin(), result.initial_physical.end());
    result.initial_hand = format_counts(counts_from_physical(hand));
    const int dead_start = INITIAL_HAND + OPPONENT_HANDS;
    result.dora_indicator = wall[dead_start];
    result.ura_indicator = wall[dead_start + 1];
    std::vector<int> live_wall(wall.begin() + dead_start + DEAD_WALL, wall.end());
    if (live_wall.size() != LIVE_WALL) throw std::logic_error("invalid live wall size");
    std::vector<int> river;
    std::size_t cursor = static_cast<std::size_t>(config.player_wind - 27);
    int riichi_turn = -1;
    int reversals_used = 0;

    if (calculator.calculate(counts_from_physical(hand)) == 0) {
        result.first_tenpai = make_snapshot(
            hand, river, river, result.dora_indicator, live_wall, cursor, 0, false, calculator
        );
        if (config.stop_at == "tenpai") {
            result.outcome = "tenpai";
            return result;
        }
    }

    int turn = 0;
    while (cursor < live_wall.size()) {
        ++turn;
        const int drawn = live_wall[cursor++];
        hand.push_back(drawn);
        const Counts complete = counts_from_physical(hand);
        if (calculator.calculate(complete) == -1) {
            Counts before = complete;
            --before[drawn / 4];
            WinInfo win;
            win.turn = turn;
            win.tile = drawn;
            win.hand = hand;
            std::sort(win.hand.begin(), win.hand.end());
            win.shapes = wait_shapes(before, drawn / 4, calculator);
            win.riichi = riichi_turn >= 0;
            win.ippatsu = riichi_turn >= 0 && turn == riichi_turn + 1;
            win.haitei = cursor == live_wall.size();
            const bool first_draw = riichi_turn < 0 && turn == 1 && result.first_tenpai.has_value();
            win.tenhou = first_draw && config.player_wind == 27;
            win.chiihou = first_draw && config.player_wind != 27;
            result.win = std::move(win);
            result.outcome = "win";
            return result;
        }

        int discarded = -1;
        if (riichi_turn >= 0) {
            discarded = drawn;
            hand.pop_back();
        } else {
            const auto choice = choose_discard(
                hand, river, result.dora_indicator, config.player_wind, config.round_wind,
                config.future, config.improvement_weight, config.policy, config.reversal,
                turn, reversals_used, policy_rng, calculator
            );
            discarded = choice.discarded;
            result.reversal_opportunities += int(choice.reversal_opportunity);
            if (choice.features.is_reversal) {
                ++reversals_used;
                TrialResult::ReversalRecord record;
                record.turn = turn;
                record.from_shanten = choice.features.strict_best_shanten;
                record.to_shanten = choice.features.shanten;
                record.discard_type = choice.features.discard_type;
                record.recovery_count = choice.features.reversal_recovery_count;
                record.recovered_ukeire_expected = choice.features.reversal_future_expected;
                record.current_ukeire = choice.features.reversal_current_ukeire;
                record.ukeire_gain = choice.features.reversal_ukeire_gain;
                record.strict_score = choice.strict_score;
                record.adjusted_score = choice.features.reversal_adjusted_score;
                result.reversal = record;
            }
        }
        river.push_back(discarded);

        const int post_shanten = calculator.calculate(counts_from_physical(hand));
        if (result.reversal && !result.reversal->recovered && turn > result.reversal->turn
            && post_shanten <= result.reversal->from_shanten) {
            result.reversal->recovered = true;
            result.reversal->recovery_turns = turn - result.reversal->turn;
        }
        const bool tenpai = post_shanten == 0;
        if (tenpai && riichi_turn < 0) {
            const bool declare = config.riichi_on_tenpai;
            if (!result.first_tenpai || result.first_tenpai->turn == 0) {
                result.first_tenpai = make_snapshot(
                    hand, river, river, result.dora_indicator, live_wall, cursor, turn, declare, calculator
                );
            }
            if (declare) riichi_turn = turn;
            if (config.stop_at == "tenpai") {
                result.outcome = "tenpai";
                return result;
            }
        }
        cursor = std::min(live_wall.size(), cursor + static_cast<std::size_t>(config.opponent_draws));
    }
    result.outcome = "exhausted";
    return result;
}

struct TableTrialResult {
    TrialResult winner;
    int winner_seat = -1;
};

TrialResult::AvailabilityCheckpoint make_availability_checkpoint(
    int target_turn, int dora_indicator, const std::array<std::vector<int>, 4>& hands,
    const std::vector<int>& public_river, const std::vector<int>& hidden_dead_wall,
    const std::vector<int>& live_wall, std::size_t cursor
) {
    if (target_turn < 0) throw std::invalid_argument("availability turn must be nonnegative");
    if (cursor > live_wall.size()) throw std::logic_error("live wall cursor outside wall");
    if (dora_indicator < 0 || dora_indicator >= PHYSICAL_TILES) {
        throw std::logic_error("availability checkpoint has invalid dora indicator");
    }
    if (hidden_dead_wall.size() != static_cast<std::size_t>(DEAD_WALL - 1)) {
        throw std::logic_error("hidden dead wall must exclude exactly one dora indicator");
    }

    TrialResult::AvailabilityCheckpoint checkpoint;
    checkpoint.target_own_turn = target_turn;
    checkpoint.completed_own_turn = target_turn;
    checkpoint.dora_indicator_type = dora_indicator / 4;
    checkpoint.live_wall_tiles_remaining = static_cast<int>(live_wall.size() - cursor);
    checkpoint.hidden_dead_wall_tiles_remaining = static_cast<int>(hidden_dead_wall.size());

    for (int tile : public_river) {
        if (tile < 0 || tile >= PHYSICAL_TILES) {
            throw std::logic_error("public river contains invalid physical tile");
        }
        ++checkpoint.public_counts[tile / 4];
    }
    ++checkpoint.public_counts[dora_indicator / 4];
    if (tile_total(checkpoint.public_counts) != static_cast<int>(public_river.size()) + 1) {
        throw std::logic_error("public count total mismatch at availability checkpoint");
    }

    Counts live_counts{};
    for (std::size_t index = cursor; index < live_wall.size(); ++index) {
        const int tile = live_wall[index];
        if (tile < 0 || tile >= PHYSICAL_TILES) {
            throw std::logic_error("live wall contains invalid physical tile");
        }
        ++live_counts[tile / 4];
    }
    Counts hidden_counts = counts_from_physical(hidden_dead_wall);
    if (tile_total(live_counts) != checkpoint.live_wall_tiles_remaining
        || tile_total(hidden_counts) != checkpoint.hidden_dead_wall_tiles_remaining) {
        throw std::logic_error("availability compartment total mismatch");
    }

    std::array<Counts, 4> hand_counts{};
    for (int seat = 0; seat < 4; ++seat) {
        hand_counts[seat] = counts_from_physical(hands[seat]);
        if (tile_total(hand_counts[seat]) != INITIAL_HAND) {
            throw std::logic_error("availability checkpoint player hand is not 13 tiles");
        }
        checkpoint.players[seat].seat = seat;
        checkpoint.players[seat].own_known_counts = hand_counts[seat];
        checkpoint.players[seat].public_counts = checkpoint.public_counts;
        checkpoint.players[seat].live_wall_counts = live_counts;
        checkpoint.players[seat].hidden_dead_wall_counts = hidden_counts;
        checkpoint.players[seat].live_wall_tiles = checkpoint.live_wall_tiles_remaining;
        checkpoint.players[seat].hidden_dead_wall_tiles = checkpoint.hidden_dead_wall_tiles_remaining;
    }

    const int public_tiles = tile_total(checkpoint.public_counts);
    const int live_tiles = checkpoint.live_wall_tiles_remaining;
    const int hidden_tiles = checkpoint.hidden_dead_wall_tiles_remaining;
    int all_hand_tiles = 0;
    for (const Counts& hand : hand_counts) all_hand_tiles += tile_total(hand);
    if (public_tiles + live_tiles + hidden_tiles + all_hand_tiles != PHYSICAL_TILES) {
        throw std::logic_error("availability physical tile partition is not conserved");
    }
    checkpoint.opponent_hand_tiles = all_hand_tiles - INITIAL_HAND;
    if (checkpoint.opponent_hand_tiles < 0) {
        throw std::logic_error("availability opponent hand total is negative");
    }

    for (int seat = 0; seat < 4; ++seat) {
        auto& player = checkpoint.players[seat];
        for (int tile = 0; tile < TILE_TYPES; ++tile) {
            const int own = player.own_known_counts[tile];
            const int public_count = player.public_counts[tile];
            const int unseen = 4 - own - public_count;
            if (unseen < 0) throw std::logic_error("negative unseen tile count");
            player.unseen_counts[tile] = static_cast<std::uint8_t>(unseen);
            int opponents = 0;
            for (int other = 0; other < 4; ++other) {
                if (other != seat) opponents += hand_counts[other][tile];
            }
            player.opponent_hand_counts[tile] = static_cast<std::uint8_t>(opponents);
            if (unseen != int(player.live_wall_counts[tile])
                + int(player.hidden_dead_wall_counts[tile]) + opponents) {
                throw std::logic_error("unseen tile compartment conservation failed");
            }
        }
        player.unseen_tiles = tile_total(player.unseen_counts);
        player.opponent_hand_tiles = tile_total(player.opponent_hand_counts);
        if (player.unseen_tiles != player.live_wall_tiles
            + player.hidden_dead_wall_tiles + player.opponent_hand_tiles) {
            throw std::logic_error("unseen position conservation failed");
        }
        if (player.opponent_hand_tiles != checkpoint.opponent_hand_tiles) {
            throw std::logic_error("opponent hand position count mismatch");
        }
    }
    return checkpoint;
}

TableTrialResult simulate_table_availability_trial(
    const Config& config, std::uint64_t trial_id, ShantenCalculator& calculator
) {
    if (config.tile_availability_turn < 0) {
        throw std::logic_error("availability trial called without a target turn");
    }
    TableTrialResult table;
    TrialResult& result = table.winner;
    result.trial_id = trial_id;
    result.seed = trial_seed(config.seed, trial_id);
    Rng wall_rng(splitmix64(result.seed ^ 0x57414C4CULL));
    std::array<std::unique_ptr<Rng>, 4> policy_rngs;
    for (int seat = 0; seat < 4; ++seat) {
        policy_rngs[seat] = std::make_unique<Rng>(
            splitmix64(result.seed ^ 0x504F4C494359ULL ^ splitmix64(seat))
        );
    }

    std::vector<int> wall(PHYSICAL_TILES);
    std::iota(wall.begin(), wall.end(), 0);
    wall_rng.shuffle(wall);
    std::array<std::vector<int>, 4> hands;
    for (int seat = 0; seat < 4; ++seat) {
        hands[seat].assign(
            wall.begin() + seat * INITIAL_HAND,
            wall.begin() + (seat + 1) * INITIAL_HAND
        );
    }
    const int dead_start = 4 * INITIAL_HAND;
    result.dora_indicator = wall[dead_start];
    result.ura_indicator = wall[dead_start + 1];
    std::vector<int> hidden_dead_wall;
    hidden_dead_wall.reserve(DEAD_WALL - 1);
    hidden_dead_wall.push_back(wall[dead_start + 1]);
    for (int offset = 2; offset < DEAD_WALL; ++offset) {
        hidden_dead_wall.push_back(wall[dead_start + offset]);
    }
    std::vector<int> live_wall(wall.begin() + dead_start + DEAD_WALL, wall.end());
    if (live_wall.size() != LIVE_WALL) throw std::logic_error("invalid shared live wall size");

    // Availability rows have no single winner/player. Keep seat 0's initial
    // hand as a useful stable row-level identity while the checkpoint carries
    // all four player perspectives.
    result.initial_physical = hands[0];
    std::sort(result.initial_physical.begin(), result.initial_physical.end());
    result.initial_hand = format_counts(counts_from_physical(hands[0]));

    std::vector<int> public_river;
    std::array<std::vector<int>, 4> rivers;
    std::array<int, 4> own_turns{};
    std::array<int, 4> reversals_used{};
    std::array<bool, 4> riichi_locked{};
    std::array<bool, 4> locked_dama{};
    std::array<bool, 4> tsumogiri_dama{};

    auto apply_directive = [&](int seat, int turn) {
        const std::string key = dama_directive_key(
            trial_id, seat, turn, counts_from_physical(hands[seat])
        );
        const auto directive = config.valuable_dama_directives.find(key);
        if (directive == config.valuable_dama_directives.end()) return;
        locked_dama[seat] = (directive->second & DAMA_LOCK_FLAG) != 0;
        tsumogiri_dama[seat] = (directive->second & DAMA_TSUMOGIRI_FLAG) != 0;
    };

    for (int seat = 0; seat < 4; ++seat) {
        if (calculator.calculate(counts_from_physical(hands[seat])) == 0) {
            if (config.riichi_on_tenpai) riichi_locked[seat] = true;
            apply_directive(seat, 0);
        }
    }

    auto finish = [&](int completed_turn, std::size_t cursor) {
        TrialResult::AvailabilityCheckpoint checkpoint = make_availability_checkpoint(
            config.tile_availability_turn, result.dora_indicator, hands, public_river,
            hidden_dead_wall, live_wall, cursor
        );
        checkpoint.completed_own_turn = completed_turn;
        result.availability_checkpoint = std::move(checkpoint);
        result.outcome = "availability_checkpoint";
        return table;
    };

    if (config.tile_availability_turn == 0) return finish(0, 0);

    std::size_t cursor = 0;
    while (cursor < live_wall.size()) {
        const int seat = static_cast<int>(cursor % 4);
        ++own_turns[seat];
        const int drawn = live_wall[cursor++];
        hands[seat].push_back(drawn);
        const bool complete = calculator.calculate(counts_from_physical(hands[seat])) == -1;
        const bool locked = riichi_locked[seat] || locked_dama[seat] || tsumogiri_dama[seat];
        int discarded = -1;
        if (complete || locked) {
            // A real win would end the hand. Availability mode intentionally
            // keeps the shared table alive, so discard the draw to retain a
            // well-defined 13-tile hand; locked hands are tsumogiri by rule.
            discarded = drawn;
            hands[seat].pop_back();
        } else {
            const auto choice = choose_discard(
                hands[seat], public_river, result.dora_indicator, 27 + seat,
                config.round_wind, config.future, config.improvement_weight,
                config.policy, config.reversal, own_turns[seat], reversals_used[seat],
                *policy_rngs[seat], calculator
            );
            discarded = choice.discarded;
            if (choice.features.is_reversal) ++reversals_used[seat];
        }
        public_river.push_back(discarded);
        rivers[seat].push_back(discarded);

        const int post_shanten = calculator.calculate(counts_from_physical(hands[seat]));
        if (!locked && post_shanten == 0 && config.riichi_on_tenpai) {
            riichi_locked[seat] = true;
        }
        apply_directive(seat, own_turns[seat]);

        if (seat == 3 && own_turns[seat] == config.tile_availability_turn) {
            for (int observer = 0; observer < 4; ++observer) {
                if (own_turns[observer] != config.tile_availability_turn) {
                    throw std::logic_error("availability checkpoint reached asynchronously");
                }
            }
            return finish(own_turns[seat], cursor);
        }
    }
    throw std::runtime_error("live wall exhausted before tile availability checkpoint");
}

TableTrialResult simulate_table_danger_trial(const Config &config,
                                             std::uint64_t trial_id,
                                             ShantenCalculator &calculator) {
  constexpr std::size_t MIN_LIVE_TILES_FOR_RIICHI = 4;
  TableTrialResult table;
  TrialResult &result = table.winner;
  result.trial_id = trial_id;
  result.seed = trial_seed(config.seed, trial_id);
  Rng wall_rng(splitmix64(result.seed ^ 0x57414C4CULL));
  std::array<std::unique_ptr<Rng>, 4> rngs;
  for (int s = 0; s < 4; ++s)
    rngs[s] = std::make_unique<Rng>(
        splitmix64(result.seed ^ 0x504F4C494359ULL ^ splitmix64(s)));
  std::vector<int> wall(PHYSICAL_TILES);
  std::iota(wall.begin(), wall.end(), 0);
  wall_rng.shuffle(wall);
  std::array<std::vector<int>, 4> hands, rivers, initial;
  for (int s = 0; s < 4; ++s) {
    hands[s].assign(wall.begin() + s * 13, wall.begin() + (s + 1) * 13);
    initial[s] = hands[s];
  }
  const int dead_start = 52;
  result.dora_indicator = wall[dead_start];
  result.ura_indicator = wall[dead_start + 1];
  std::vector<int> live(wall.begin() + dead_start + DEAD_WALL, wall.end()),
      public_river;
  std::array<int, 4> turns{}, reversals{};
  std::size_t cursor = 0;
  std::size_t riichi_public_start = 0;
  TrialResult::Danger danger;
  result.initial_physical = initial[0];
  std::sort(result.initial_physical.begin(), result.initial_physical.end());
  result.initial_hand = format_counts(counts_from_physical(initial[0]));
  auto assert_partition = [&]() {
    std::array<bool, PHYSICAL_TILES> seen{};
    int total = 0;
    auto add = [&](int tile) {
      if (tile < 0 || tile >= PHYSICAL_TILES || seen[tile])
        throw std::logic_error("tile danger physical partition failed");
      seen[tile] = true;
      ++total;
    };
    for (const auto &hand : hands)
      for (int tile : hand)
        add(tile);
    for (int tile : public_river)
      add(tile);
    for (int i = 0; i < DEAD_WALL; ++i)
      add(wall[dead_start + i]);
    for (std::size_t i = cursor; i < live.size(); ++i)
      add(live[i]);
    if (total != PHYSICAL_TILES)
      throw std::logic_error("tile danger physical partition incomplete");
  };
  auto identity = [&](int s) {
    table.winner_seat = s;
    result.initial_physical = initial[s];
    std::sort(result.initial_physical.begin(), result.initial_physical.end());
    result.initial_hand = format_counts(counts_from_physical(initial[s]));
  };
  auto finish = [&](const std::string &reason) {
    assert_partition();
    danger.terminal_reason = reason;
    result.danger = danger;
    result.outcome = reason;
    return table;
  };
  while (cursor < live.size()) {
    const int seat = int(cursor % 4);
    ++turns[seat];
    const int drawn = live[cursor++];
    hands[seat].push_back(drawn);
    Counts full = counts_from_physical(hands[seat]);
    if (calculator.calculate(full) == -1) {
      if (danger.riichi_seat < 0)
        return finish("pre_riichi_tsumo");
      if (seat == danger.riichi_seat) {
        identity(seat);
        return finish("target_tsumo");
      }
      return finish("defender_tsumo");
    }
    if (danger.riichi_seat == seat) {
      hands[seat].pop_back();
      public_river.push_back(drawn);
      rivers[seat].push_back(drawn);
      continue;
    }
    if (danger.riichi_seat >= 0) {
      assert_partition();
      TrialResult::DangerDecision row;
      row.observer_seat = seat;
      row.observer_turn = turns[seat];
      row.riichi_player_turn = turns[danger.riichi_seat];
      row.turns_since_riichi = turns[danger.riichi_seat] - danger.riichi_turn;
      row.draw_index = int(cursor);
      row.live_wall_remaining = int(live.size() - cursor);
      row.own_counts = full;
      for (int x : public_river)
        ++row.public_counts[x / 4];
      ++row.public_counts[result.dora_indicator / 4];
      for (int x : rivers[danger.riichi_seat])
        ++row.riichi_river_counts[x / 4];
      std::array<bool, TILE_TYPES> safe{};
      for (int x : rivers[danger.riichi_seat])
        safe[x / 4] = true;
      for (std::size_t i = riichi_public_start; i < public_river.size(); ++i)
        safe[public_river[i] / 4] = true;
      for (int t = 0; t < TILE_TYPES; ++t)
        if (safe[t])
          row.safe_types.push_back(t);
      const bool furiten = is_permanent_furiten(
          danger.riichi_hand_counts, rivers[danger.riichi_seat], calculator);
      for (int t = 0; t < TILE_TYPES; ++t)
        if (full[t])
          row.candidates.push_back(
              {t,
               !furiten && std::find(danger.winning_types.begin(),
                                     danger.winning_types.end(),
                                     t) != danger.winning_types.end(),
               false});
      const auto choice = choose_discard(
          hands[seat], public_river, result.dora_indicator, 27 + seat,
          config.round_wind, config.future, config.improvement_weight,
          config.policy, config.reversal, turns[seat], reversals[seat],
          *rngs[seat], calculator);
      if (choice.features.is_reversal)
        ++reversals[seat];
      const int discarded = choice.discarded;
      for (auto &c : row.candidates)
        if (c.tile_type == discarded / 4)
          c.chosen = true;
      danger.decisions.push_back(std::move(row));
      if (!furiten &&
          std::find(danger.winning_types.begin(), danger.winning_types.end(),
                    discarded / 4) != danger.winning_types.end()) {
        public_river.push_back(discarded);
        rivers[seat].push_back(discarded);
        assert_partition();
        identity(danger.riichi_seat);
        return finish("target_ron");
      }
      public_river.push_back(discarded);
      rivers[seat].push_back(discarded);
      if (calculator.calculate(counts_from_physical(hands[seat])) == 0 &&
          live.size() - cursor >= MIN_LIVE_TILES_FOR_RIICHI)
        return finish("second_riichi");
      continue;
    }
    const auto choice = choose_discard(
        hands[seat], public_river, result.dora_indicator, 27 + seat,
        config.round_wind, config.future, config.improvement_weight,
        config.policy, config.reversal, turns[seat], reversals[seat],
        *rngs[seat], calculator);
    if (choice.features.is_reversal)
      ++reversals[seat];
    public_river.push_back(choice.discarded);
    rivers[seat].push_back(choice.discarded);
    if (calculator.calculate(counts_from_physical(hands[seat])) == 0 &&
        live.size() - cursor >= MIN_LIVE_TILES_FOR_RIICHI) {
      danger.riichi_seat = seat;
      danger.riichi_turn = turns[seat];
      danger.riichi_draw_index = int(cursor);
      danger.riichi_hand_counts = counts_from_physical(hands[seat]);
      danger.winning_types =
          structural_ukeire_types(danger.riichi_hand_counts, calculator);
      danger.furiten_at_riichi = is_permanent_furiten(danger.riichi_hand_counts,
                                                      rivers[seat], calculator);
      riichi_public_start = public_river.size();
    }
  }
  return finish(danger.riichi_seat < 0 ? "no_riichi_exposure" : "exhausted");
}

#include "hand_push_fold_native.inc"

TableTrialResult simulate_table_trial(
    const Config& config, std::uint64_t trial_id, ShantenCalculator& calculator
) {
    if (config.push_fold) return simulate_table_push_fold_trial(config, trial_id, calculator);
    if (config.tile_danger) return simulate_table_danger_trial(config, trial_id, calculator);
    if (config.tile_availability_turn >= 0) {
        return simulate_table_availability_trial(config, trial_id, calculator);
    }
    TableTrialResult table;
    TrialResult& result = table.winner;
    result.trial_id = trial_id;
    result.seed = trial_seed(config.seed, trial_id);
    Rng wall_rng(splitmix64(result.seed ^ 0x57414C4CULL));
    std::array<std::unique_ptr<Rng>, 4> policy_rngs;
    for (int seat = 0; seat < 4; ++seat) {
        policy_rngs[seat] = std::make_unique<Rng>(
            splitmix64(result.seed ^ 0x504F4C494359ULL ^ splitmix64(seat))
        );
    }

    std::vector<int> wall(PHYSICAL_TILES);
    std::iota(wall.begin(), wall.end(), 0);
    wall_rng.shuffle(wall);
    std::array<std::vector<int>, 4> hands;
    std::array<std::vector<int>, 4> rivers;
    std::array<std::vector<int>, 4> initial_hands;
    for (int seat = 0; seat < 4; ++seat) {
        hands[seat].assign(
            wall.begin() + seat * INITIAL_HAND,
            wall.begin() + (seat + 1) * INITIAL_HAND
        );
        initial_hands[seat] = hands[seat];
    }
    const int dead_start = 4 * INITIAL_HAND;
    result.dora_indicator = wall[dead_start];
    result.ura_indicator = wall[dead_start + 1];
    std::vector<int> live_wall(wall.begin() + dead_start + DEAD_WALL, wall.end());
    if (live_wall.size() != LIVE_WALL) throw std::logic_error("invalid shared live wall size");

    std::vector<int> public_river;
    std::array<int, 4> own_turns{};
    std::array<int, 4> reversals_used{};
    std::array<int, 4> reversal_opportunities{};
    std::array<std::optional<TrialResult::ReversalRecord>, 4> reversals;
    std::array<bool, 4> has_recorded_tenpai{};
    std::array<bool, 4> locked_dama{};
    std::array<bool, 4> tsumogiri_dama{};
    std::array<std::uint64_t, 4> dama_ron_masks{};

    auto set_winner_identity = [&](int seat) {
        table.winner_seat = seat;
        result.initial_physical = initial_hands[seat];
        std::sort(result.initial_physical.begin(), result.initial_physical.end());
        result.initial_hand = format_counts(counts_from_physical(initial_hands[seat]));
        result.reversal = reversals[seat];
        result.reversal_opportunities = reversal_opportunities[seat];
    };

    // Rare initial tenpai is resolved in seat order, matching the first action order.
    for (int seat = 0; seat < 4; ++seat) {
        if (calculator.calculate(counts_from_physical(hands[seat])) != 0) continue;
        Snapshot snapshot = make_snapshot(
            hands[seat], public_river, rivers[seat], result.dora_indicator,
            live_wall, 0, 0, false, calculator
        );
        if (config.continue_table_tenpai) {
            TrialResult::TableEvent event;
            event.kind = "tenpai";
            event.seat = seat;
            event.tenpai = std::move(snapshot);
            result.table_events.push_back(std::move(event));
            has_recorded_tenpai[seat] = true;
            reversals_used[seat] = 0;
            const std::string key = dama_directive_key(
                trial_id, seat, 0, counts_from_physical(hands[seat])
            );
            const auto directive = config.valuable_dama_directives.find(key);
            if (directive != config.valuable_dama_directives.end()) {
                locked_dama[seat] = (directive->second & DAMA_LOCK_FLAG) != 0;
                tsumogiri_dama[seat] = (directive->second & DAMA_TSUMOGIRI_FLAG) != 0;
                dama_ron_masks[seat] = directive->second & DAMA_WAIT_MASK;
            }
            continue;
        }
        set_winner_identity(seat);
        result.first_tenpai = std::move(snapshot);
        result.outcome = "tenpai";
        return table;
    }

    std::size_t cursor = 0;
    while (cursor < live_wall.size()) {
        const int seat = static_cast<int>(cursor % 4);
        ++own_turns[seat];
        const int drawn = live_wall[cursor++];
        hands[seat].push_back(drawn);
        if (calculator.calculate(counts_from_physical(hands[seat])) == -1) {
            if (config.continue_table_tenpai && (has_recorded_tenpai[seat] || locked_dama[seat])) {
                Counts before = counts_from_physical(hands[seat]);
                --before[drawn / 4];
                WinInfo win;
                win.turn = own_turns[seat];
                win.tile = drawn;
                win.hand = hands[seat];
                std::sort(win.hand.begin(), win.hand.end());
                win.shapes = wait_shapes(before, drawn / 4, calculator);
                TrialResult::TableEvent event;
                event.kind = "dama_tsumo";
                event.seat = seat;
                event.turn = own_turns[seat];
                event.win = std::move(win);
                result.table_events.push_back(std::move(event));
                result.outcome = "trajectory";
                return table;
            }
            throw std::logic_error("shared-table player won before a recorded tenpai");
        }
        int discarded = -1;
        const bool was_locked_dama = locked_dama[seat];
        ReversalConfig decision_reversal = config.reversal;
        if (config.continue_table_tenpai && has_recorded_tenpai[seat]) {
            decision_reversal.max_turn = std::max(decision_reversal.max_turn, 99);
        }
        if (was_locked_dama) decision_reversal.enabled = false;
        std::optional<DiscardChoice> choice;
        if (tsumogiri_dama[seat]) {
            discarded = drawn;
            hands[seat].pop_back();
        } else {
            choice = choose_discard(
                hands[seat], public_river, result.dora_indicator, 27 + seat, config.round_wind,
                config.future, config.improvement_weight, config.policy, decision_reversal,
                own_turns[seat], reversals_used[seat], *policy_rngs[seat], calculator
            );
            discarded = choice->discarded;
        }
        if (was_locked_dama && choice) {
            const std::string transition_key = dama_directive_key(
                trial_id, seat, own_turns[seat], counts_from_physical(hands[seat])
            );
            const auto transition = config.valuable_dama_directives.find(transition_key);
            if (transition != config.valuable_dama_directives.end()
                && (transition->second & DAMA_REJECT_TRANSITION_FLAG)) {
                // Python scoring rejected this bad-to-bad change because the
                // new wait lost its yaku or fell below the original value line.
                // Restore the prior 13-tile hand and discard the drawn tile.
                hands[seat].push_back(discarded);
                const auto drawn_it = std::find(hands[seat].begin(), hands[seat].end(), drawn);
                if (drawn_it == hands[seat].end()) {
                    throw std::logic_error("drawn tile missing while rejecting dama transition");
                }
                hands[seat].erase(drawn_it);
                discarded = drawn;
                choice.reset();
            }
        }
        public_river.push_back(discarded);
        rivers[seat].push_back(discarded);
        if (choice) reversal_opportunities[seat] += int(choice->reversal_opportunity);
        if (choice && choice->features.is_reversal) {
            ++reversals_used[seat];
            TrialResult::ReversalRecord record;
            record.turn = own_turns[seat];
            record.from_shanten = choice->features.strict_best_shanten;
            record.to_shanten = choice->features.shanten;
            record.discard_type = choice->features.discard_type;
            record.recovery_count = choice->features.reversal_recovery_count;
            record.recovered_ukeire_expected = choice->features.reversal_future_expected;
            record.current_ukeire = choice->features.reversal_current_ukeire;
            record.ukeire_gain = choice->features.reversal_ukeire_gain;
            record.strict_score = choice->strict_score;
            record.adjusted_score = choice->features.reversal_adjusted_score;
            reversals[seat] = record;
            if (config.continue_table_tenpai && has_recorded_tenpai[seat]) {
                TrialResult::TableEvent event;
                event.kind = "shanten_reversal";
                event.seat = seat;
                event.turn = own_turns[seat];
                result.table_events.push_back(std::move(event));
            }
        }

        for (int dama_seat = 0; dama_seat < 4; ++dama_seat) {
            if (!locked_dama[dama_seat] || dama_seat == seat) continue;
            const int tile_type = discarded / 4;
            if (!(dama_ron_masks[dama_seat] & (std::uint64_t{1} << tile_type))) continue;
            if (is_permanent_furiten(
                    counts_from_physical(hands[dama_seat]), rivers[dama_seat], calculator
                )) continue;
            TrialResult::TableEvent event;
            event.kind = "dama_ron";
            event.seat = dama_seat;
            event.turn = own_turns[dama_seat];
            WinInfo win;
            win.turn = own_turns[dama_seat];
            win.tile = discarded;
            win.hand = hands[dama_seat];
            win.hand.push_back(discarded);
            std::sort(win.hand.begin(), win.hand.end());
            win.shapes = wait_shapes(counts_from_physical(hands[dama_seat]), tile_type, calculator);
            event.win = std::move(win);
            result.table_events.push_back(std::move(event));
            result.outcome = "trajectory";
            return table;
        }

        const int post_shanten = calculator.calculate(counts_from_physical(hands[seat]));
        if (reversals[seat] && !reversals[seat]->recovered
            && own_turns[seat] > reversals[seat]->turn
            && post_shanten <= reversals[seat]->from_shanten) {
            reversals[seat]->recovered = true;
            reversals[seat]->recovery_turns = own_turns[seat] - reversals[seat]->turn;
        }
        if (tsumogiri_dama[seat]) continue;
        if (post_shanten != 0) continue;

        Snapshot snapshot = make_snapshot(
            hands[seat], public_river, rivers[seat], result.dora_indicator,
            live_wall, cursor, own_turns[seat], !config.continue_table_tenpai, calculator
        );
        if (config.continue_table_tenpai) {
            TrialResult::TableEvent event;
            event.kind = "tenpai";
            event.seat = seat;
            event.turn = own_turns[seat];
            event.tenpai = std::move(snapshot);
            result.table_events.push_back(std::move(event));
            has_recorded_tenpai[seat] = true;
            const std::string key = dama_directive_key(
                trial_id, seat, own_turns[seat], counts_from_physical(hands[seat])
            );
            const auto directive = config.valuable_dama_directives.find(key);
            if (directive != config.valuable_dama_directives.end()) {
                locked_dama[seat] = (directive->second & DAMA_LOCK_FLAG) != 0;
                tsumogiri_dama[seat] = (directive->second & DAMA_TSUMOGIRI_FLAG) != 0;
                dama_ron_masks[seat] = directive->second & DAMA_WAIT_MASK;
            }
            // A rejected tenpai starts a fresh, non-recursive allowance for
            // improving the silent hand, including a later shanten reversal.
            reversals_used[seat] = 0;
            continue;
        }

        set_winner_identity(seat);
        result.first_tenpai = std::move(snapshot);
        result.outcome = "tenpai";
        return table;
    }

    result.outcome = config.continue_table_tenpai ? "trajectory_exhausted" : "exhausted";
    return table;
}

struct Aggregate {
    std::uint64_t trials = 0;
    std::map<std::string, std::uint64_t> outcomes;
    std::uint64_t tenpai_hands = 0;
    std::uint64_t ron_eligible = 0;
    std::uint64_t tenpai_turn_sum = 0;
    std::uint64_t wins = 0;
    std::uint64_t win_turn_sum = 0;
    std::map<std::string, double> signatures;
    std::array<double, TILE_TYPES> tile_wait_hands{};
    std::array<std::array<double, 8>, TILE_TYPES> tile_shape_hands{};
    std::array<double, 8> visible_by_shape{};
    std::array<double, 8> live_by_shape{};
    double visible_copies = 0.0;
    double live_copies = 0.0;
    std::uint64_t reversal_opportunities = 0;
    std::uint64_t reversals_selected = 0;
    std::uint64_t reversals_recovered = 0;
    std::uint64_t reversal_turn_sum = 0;
    std::uint64_t reversal_recovery_turn_sum = 0;
    std::array<std::uint64_t, 2> reversal_transitions{};

    void add(const TrialResult& result) {
        ++trials;
        ++outcomes[result.outcome];
        reversal_opportunities += result.reversal_opportunities;
        if (result.reversal) {
            ++reversals_selected;
            reversal_turn_sum += result.reversal->turn;
            if (result.reversal->from_shanten >= 0 && result.reversal->from_shanten <= 1) {
                ++reversal_transitions[result.reversal->from_shanten];
            }
            if (result.reversal->recovered) {
                ++reversals_recovered;
                reversal_recovery_turn_sum += result.reversal->recovery_turns;
            }
        }
        if (result.first_tenpai) {
            const Snapshot& snapshot = *result.first_tenpai;
            ++tenpai_hands;
            tenpai_turn_sum += snapshot.turn;
            ++signatures[snapshot.signature];
            for (const auto& wait : snapshot.waits) {
                const double split = wait.shapes.empty() ? 0.0 : 1.0 / wait.shapes.size();
                for (int shape : wait.shapes) {
                    visible_by_shape[shape] += wait.visible_remaining * split;
                    live_by_shape[shape] += wait.live_remaining * split;
                }
                visible_copies += wait.visible_remaining;
                live_copies += wait.live_remaining;
            }
            if (!snapshot.furiten) {
                ++ron_eligible;
                for (const auto& wait : snapshot.waits) {
                    if (!wait.visible_remaining) continue;
                    ++tile_wait_hands[wait.tile_type];
                    const double split = wait.shapes.empty() ? 0.0 : 1.0 / wait.shapes.size();
                    for (int shape : wait.shapes) tile_shape_hands[wait.tile_type][shape] += split;
                }
            }
        }
        if (result.win) {
            ++wins;
            win_turn_sum += result.win->turn;
        }
    }
};

std::string json_quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec;
                else out << ch;
        }
    }
    out << '"';
    return out.str();
}

std::string number_json(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(15) << value;
    return out.str();
}

template <typename T> std::string numeric_array_json(const std::vector<T>& values) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) out << ',';
        out << values[index];
    }
    out << ']';
    return out.str();
}

std::string shape_array_json(const std::vector<int>& shapes) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        if (index) out << ',';
        out << json_quote(SHAPE_NAMES[shapes[index]]);
    }
    out << ']';
    return out.str();
}

std::string counts_json(const Counts& counts) {
    std::ostringstream out;
    out << '[';
    for (int tile = 0; tile < TILE_TYPES; ++tile) {
        if (tile) out << ',';
        out << int(counts[tile]);
    }
    out << ']';
    return out.str();
}

std::string snapshot_json(const Snapshot& snapshot) {
    std::ostringstream out;
    out << "{\"turn\":" << snapshot.turn
        << ",\"hand\":" << json_quote(format_counts(snapshot.hand))
        << ",\"hand_counts\":" << counts_json(snapshot.hand)
        << ",\"hand_physical\":" << numeric_array_json(snapshot.hand_physical)
        << ",\"signature\":" << json_quote(snapshot.signature)
        << ",\"furiten\":" << (snapshot.furiten ? "true" : "false")
        << ",\"riichi_declared\":" << (snapshot.riichi_declared ? "true" : "false")
        << ",\"visible_ukeire\":" << snapshot.visible_ukeire
        << ",\"live_wall_ukeire\":" << snapshot.live_ukeire
        << ",\"live_wall_tiles_remaining\":" << snapshot.live_wall_tiles_remaining
        << ",\"waits\":[";
    for (std::size_t index = 0; index < snapshot.waits.size(); ++index) {
        if (index) out << ',';
        const auto& wait = snapshot.waits[index];
        out << "{\"tile_type\":" << wait.tile_type
            << ",\"name\":" << json_quote(tile_name(wait.tile_type))
            << ",\"visible_remaining\":" << wait.visible_remaining
            << ",\"live_wall_remaining\":" << wait.live_remaining
            << ",\"visible_remaining_tile_ids\":" << numeric_array_json(wait.visible_remaining_tile_ids)
            << ",\"live_wall_remaining_tile_ids\":" << numeric_array_json(wait.live_remaining_tile_ids)
            << ",\"shapes\":" << shape_array_json(wait.shapes) << '}';
    }
    out << "]}";
    return out.str();
}

std::string availability_player_json(const TrialResult::AvailabilityPlayer& player) {
    std::ostringstream out;
    out << "{\"seat\":" << player.seat
        << ",\"own_known_counts\":" << counts_json(player.own_known_counts)
        << ",\"public_counts\":" << counts_json(player.public_counts)
        << ",\"unseen_counts\":" << counts_json(player.unseen_counts)
        << ",\"live_wall_counts\":" << counts_json(player.live_wall_counts)
        << ",\"hidden_dead_wall_counts\":" << counts_json(player.hidden_dead_wall_counts)
        << ",\"opponent_hand_counts\":" << counts_json(player.opponent_hand_counts)
        << ",\"unseen_tiles\":" << player.unseen_tiles
        << ",\"live_wall_tiles\":" << player.live_wall_tiles
        << ",\"hidden_dead_wall_tiles\":" << player.hidden_dead_wall_tiles
        << ",\"opponent_hand_tiles\":" << player.opponent_hand_tiles << '}';
    return out.str();
}

std::string availability_checkpoint_json(
    const TrialResult::AvailabilityCheckpoint& checkpoint
) {
    std::ostringstream out;
    out << "{\"target_own_turn\":" << checkpoint.target_own_turn
        << ",\"completed_own_turn\":" << checkpoint.completed_own_turn
        << ",\"dora_indicator_type\":" << checkpoint.dora_indicator_type
        << ",\"live_wall_tiles_remaining\":" << checkpoint.live_wall_tiles_remaining
        << ",\"hidden_dead_wall_tiles_remaining\":"
        << checkpoint.hidden_dead_wall_tiles_remaining
        << ",\"opponent_hand_tiles\":" << checkpoint.opponent_hand_tiles
        << ",\"public_counts\":" << counts_json(checkpoint.public_counts)
        << ",\"players\":[";
    for (int seat = 0; seat < 4; ++seat) {
        if (seat) out << ',';
        out << availability_player_json(checkpoint.players[seat]);
    }
    out << "]}";
    return out.str();
}

std::string danger_json(const TrialResult::Danger &danger) {
  std::ostringstream out;
  out << "{\"riichi_seat\":" << danger.riichi_seat
      << ",\"riichi_turn\":" << danger.riichi_turn
      << ",\"riichi_draw_index\":" << danger.riichi_draw_index
      << ",\"riichi_hand_counts\":" << counts_json(danger.riichi_hand_counts)
      << ",\"winning_types\":" << numeric_array_json(danger.winning_types)
      << ",\"furiten_at_riichi\":"
      << (danger.furiten_at_riichi ? "true" : "false")
      << ",\"terminal_reason\":" << json_quote(danger.terminal_reason)
      << ",\"decisions\":[";
  for (std::size_t i = 0; i < danger.decisions.size(); ++i) {
    if (i)
      out << ',';
    const auto &d = danger.decisions[i];
    out << "{\"observer_seat\":" << d.observer_seat
        << ",\"observer_turn\":" << d.observer_turn
        << ",\"riichi_player_turn\":" << d.riichi_player_turn
        << ",\"turns_since_riichi\":" << d.turns_since_riichi
        << ",\"draw_index\":" << d.draw_index
        << ",\"live_wall_remaining\":" << d.live_wall_remaining
        << ",\"own_counts\":" << counts_json(d.own_counts)
        << ",\"public_counts\":" << counts_json(d.public_counts)
        << ",\"safe_types\":" << numeric_array_json(d.safe_types)
        << ",\"riichi_river_counts\":" << counts_json(d.riichi_river_counts)
        << ",\"candidates\":[";
    for (std::size_t j = 0; j < d.candidates.size(); ++j) {
      if (j)
        out << ',';
      const auto &c = d.candidates[j];
      out << "{\"tile_type\":" << c.tile_type
          << ",\"ron_hit\":" << (c.ron_hit ? "true" : "false")
          << ",\"chosen\":" << (c.chosen ? "true" : "false") << '}';
    }
    out << "]}";
  }
  return out << "]}", out.str();
}

std::string score_reply_json(const ScoreReply& score) {
    std::ostringstream out;
    out << "{\"good_wait\":" << (score.good_wait ? "true" : "false")
        << ",\"cheap_all_unseen\":" << (score.cheap_all_unseen ? "true" : "false")
        << ",\"exact_1300_bad_wait\":" << (score.exact_1300_bad_wait ? "true" : "false")
        << ",\"valuable_dama\":" << (score.valuable_dama ? "true" : "false")
        << ",\"valuable_ron_mask\":" << score.valuable_ron_mask
        << ",\"haneman_lock\":" << (score.haneman_lock ? "true" : "false")
        << ",\"haneman_ron_mask\":" << score.haneman_ron_mask
        << ",\"legal_ron_mask\":" << score.legal_ron_mask
        << ",\"min_dama_actual\":" << score.min_dama_ron_points
        << ",\"min_riichi_actual\":" << score.min_riichi_ron_points
        << ",\"mean_riichi_actual\":" << number_json(score.mean_riichi_ron_points) << '}';
    return out.str();
}

std::string push_fold_json(const PushFoldResult& push) {
    std::ostringstream out;
    out << "{\"mode\":" << json_quote(push.mode)
        << ",\"safety_strategy\":" << json_quote(push.safety_strategy)
        << ",\"dora_danger_steps\":" << push.dora_danger_steps
        << ",\"half_push_max_tier\":" << push.half_push_max_tier
        << ",\"two_riichi_bad_wait_min_ron\":" << push.two_riichi_bad_wait_min_ron
        << ",\"ippatsu_model\":" << json_quote(push.ippatsu_model)
        << ",\"riichi_seats\":" << numeric_array_json(push.riichi_seats)
        << ",\"riichi_events\":[";
    for (std::size_t index = 0; index < push.riichi_events.size(); ++index) {
        if (index) out << ',';
        const auto& event = push.riichi_events[index];
        out << "{\"seat\":" << event.seat
            << ",\"turn\":" << event.turn
            << ",\"hand_physical13\":" << numeric_array_json(event.hand_physical13)
            << ",\"hand_physical\":" << numeric_array_json(event.hand_physical13)
            << ",\"winning_types\":" << numeric_array_json(event.winning_types)
            << ",\"unseen_wait_copies\":" << event.unseen_wait_copies
            << ",\"live_wait_copies\":" << event.live_wait_copies
            << ",\"live_wall_remaining\":" << event.live_wall_remaining
            << ",\"furiten\":" << (event.furiten ? "true" : "false") << '}';
    }
    out << ']'
        << ",\"decisions\":[";
    for (std::size_t index = 0; index < push.decisions.size(); ++index) {
        if (index) out << ',';
        const auto& row = push.decisions[index];
        out << "{\"seat\":" << row.seat
            << ",\"turn\":" << row.turn
            << ",\"draw_index\":" << row.draw_index
            << ",\"drawn_physical\":" << row.drawn_physical
            << ",\"live_wall_remaining\":" << row.live_wall_remaining
            << ",\"own_riichi\":" << (row.own_riichi ? "true" : "false")
            << ",\"threat_seats\":" << numeric_array_json(row.threat_seats)
            << ",\"mode\":" << json_quote(row.mode)
            << ",\"mode_reason\":" << json_quote(row.mode_reason)
            << ",\"best_shanten\":" << row.best_shanten
            << ",\"post_discard_shanten\":" << row.post_discard_shanten
            << ",\"own_counts\":" << counts_json(row.own_counts)
            << ",\"public_counts\":" << counts_json(row.public_counts)
            << ",\"safe_types\":[";
        for (std::size_t safe_index = 0; safe_index < row.safe_types.size(); ++safe_index) {
            if (safe_index) out << ',';
            out << "{\"threat_seat\":" << row.safe_types[safe_index].threat_seat
                << ",\"types\":" << numeric_array_json(row.safe_types[safe_index].types) << '}';
        }
        out << "],\"candidates\":[";
        for (std::size_t candidate_index = 0; candidate_index < row.candidates.size(); ++candidate_index) {
            if (candidate_index) out << ',';
            const auto& candidate = row.candidates[candidate_index];
            out << "{\"physical_id\":" << candidate.physical_id
                << ",\"tile_type\":" << candidate.tile_type
                << ",\"safety_tiers\":" << numeric_array_json(candidate.safety_tiers)
                << ",\"known_copies\":" << candidate.known_copies
                << ",\"chosen\":" << (candidate.chosen ? "true" : "false")
                << ",\"ron_hits\":" << numeric_array_json(candidate.ron_hits) << '}';
        }
        out << "],\"selected_score_fields\":";
        if (row.has_selected_score) out << score_reply_json(row.selected_score);
        else out << "null";
        out << '}';
    }
    out << "],\"terminal\":{\"reason\":" << json_quote(push.terminal.reason)
        << ",\"winners\":[";
    for (std::size_t index = 0; index < push.terminal.winners.size(); ++index) {
        if (index) out << ',';
        const auto& winner = push.terminal.winners[index];
        out << "{\"seat\":" << winner.seat
            << ",\"hand_physical14\":" << numeric_array_json(winner.hand_physical)
            << ",\"hand_physical\":" << numeric_array_json(winner.hand_physical)
            << ",\"winning_tile\":" << winner.winning_tile
            << ",\"is_riichi\":" << (winner.is_riichi ? "true" : "false")
            << ",\"is_ippatsu\":" << (winner.is_ippatsu ? "true" : "false")
            << ",\"is_haitei\":" << (winner.is_haitei ? "true" : "false") << '}';
    }
    out << "],\"loser\":" << push.terminal.loser
        << ",\"riichi_sticks\":" << push.terminal.riichi_sticks
        << ",\"tenpai_seats\":" << numeric_array_json(push.terminal.tenpai_seats)
        << ",\"final_hand_counts\":[";
    for (int seat = 0; seat < 4; ++seat) {
        if (seat) out << ',';
        out << counts_json(push.terminal.final_hand_counts[seat]);
    }
    out << "]}}";
    return out.str();
}

std::string policy_parameters_json(const Config& config) {
    std::ostringstream out;
    out << "{\"future_ukeire_weight\":" << number_json(config.future.fallback)
        << ",\"future_ukeire_weights_by_shanten\":{\"1\":" << number_json(config.future.weight(1))
        << ",\"2\":" << number_json(config.future.weight(2))
        << ",\"3_plus\":0}"
        << ",\"improvement_weight\":" << number_json(config.improvement_weight)
        << ",\"advanced_shanten\":[1,2]"
        << ",\"dora_ukeire_weight\":" << number_json(config.policy.dora_ukeire_weight)
        << ",\"dora_tiebreak_weight\":" << number_json(config.policy.dora_tiebreak_weight)
        << ",\"dora_neighbor_weight\":" << number_json(config.policy.dora_neighbor_weight)
        << ",\"yakuhai_weight\":" << number_json(config.policy.yakuhai_weight)
        << ",\"tanyao_weight\":" << number_json(config.policy.tanyao_weight)
        << (config.policy.kanchan_outer_wait_weight != 0.0
            ? ",\"kanchan_outer_wait_weight\":" + number_json(config.policy.kanchan_outer_wait_weight)
            : "")
        << ",\"shanten_reversal\":{\"enabled\":" << (config.reversal.enabled ? "true" : "false")
        << ",\"penalty\":" << number_json(config.reversal.penalty)
        << ",\"recovered_ukeire_weight\":" << number_json(config.reversal.future_weight)
        << ",\"max_turn\":" << config.reversal.max_turn
        << ",\"candidate_limit\":" << config.reversal.candidate_limit
        << ",\"max_per_hand\":" << config.reversal.max_per_hand << "}}";
    return out.str();
}

std::string trial_json(const TrialResult& result, const Config& config) {
    std::ostringstream out;
    out << "{\"engine_version\":" << json_quote(ENGINE_VERSION)
        << ",\"evaluator_backend\":" << json_quote(EVALUATOR_BACKEND)
        << ",\"trial_id\":" << result.trial_id
        << ",\"trial_seed\":" << result.seed
        << ",\"policy\":" << json_quote(config.policy.name())
        << ",\"policy_parameters\":" << policy_parameters_json(config)
        << ",\"outcome\":" << json_quote(result.outcome)
        << ",\"initial_hand\":" << json_quote(result.initial_hand)
        << ",\"initial_hand_physical\":" << numeric_array_json(result.initial_physical)
        << ",\"dora_indicator\":" << result.dora_indicator
        << ",\"ura_dora_indicator\":" << result.ura_indicator
        << ",\"player_wind\":" << config.player_wind
        << ",\"round_wind\":" << config.round_wind
        << ",\"first_tenpai\":";
    if (result.first_tenpai) out << snapshot_json(*result.first_tenpai);
    else out << "null";
    out << ",\"win\":";
    if (!result.win) {
        out << "null";
    } else {
        const auto& win = *result.win;
        out << "{\"turn\":" << win.turn
            << ",\"tile\":" << win.tile
            << ",\"tile_name\":" << json_quote(physical_tile_name(win.tile))
            << ",\"hand_physical\":" << numeric_array_json(win.hand)
            << ",\"wait_shapes\":" << shape_array_json(win.shapes)
            << ",\"riichi\":" << (win.riichi ? "true" : "false")
            << ",\"ippatsu\":" << (win.ippatsu ? "true" : "false")
            << ",\"haitei\":" << (win.haitei ? "true" : "false")
            << ",\"tenhou\":" << (win.tenhou ? "true" : "false")
            << ",\"chiihou\":" << (win.chiihou ? "true" : "false") << '}';
    }
    out << ",\"table_events\":[";
    for (std::size_t index = 0; index < result.table_events.size(); ++index) {
        if (index) out << ',';
        const auto& event = result.table_events[index];
        out << "{\"kind\":" << json_quote(event.kind)
            << ",\"seat\":" << event.seat
            << ",\"turn\":" << event.turn;
        if (event.tenpai) out << ",\"tenpai\":" << snapshot_json(*event.tenpai);
        if (event.win) {
            const auto& win = *event.win;
            out << ",\"win\":{\"turn\":" << win.turn
                << ",\"tile\":" << win.tile
                << ",\"tile_name\":" << json_quote(physical_tile_name(win.tile))
                << ",\"hand_physical\":" << numeric_array_json(win.hand)
                << ",\"wait_shapes\":" << shape_array_json(win.shapes) << '}';
        }
        out << '}';
    }
    out << ']';
    out << ",\"availability_checkpoint\":";
    if (result.availability_checkpoint) {
        out << availability_checkpoint_json(*result.availability_checkpoint);
    } else {
        out << "null";
    }
    if (result.danger) out << ",\"danger\":" << danger_json(*result.danger);
    if (result.push_fold) out << ",\"push_fold\":" << push_fold_json(*result.push_fold);
    out << ",\"shanten_reversal_opportunities\":" << result.reversal_opportunities
        << ",\"shanten_reversals\":[";
    if (result.reversal) {
        const auto& reversal = *result.reversal;
        out << "{\"turn\":" << reversal.turn
            << ",\"from_shanten\":" << reversal.from_shanten
            << ",\"to_shanten\":" << reversal.to_shanten
            << ",\"discard_type\":" << reversal.discard_type
            << ",\"recovery_count\":" << reversal.recovery_count
            << ",\"recovered_ukeire_expected\":" << number_json(reversal.recovered_ukeire_expected)
            << ",\"current_ukeire\":" << reversal.current_ukeire
            << ",\"ukeire_gain\":" << number_json(reversal.ukeire_gain)
            << ",\"strict_score\":" << number_json(reversal.strict_score)
            << ",\"adjusted_reversal_score\":" << number_json(reversal.adjusted_score)
            << ",\"recovered\":" << (reversal.recovered ? "true" : "false")
            << ",\"recovery_turns\":"
            << (reversal.recovered ? std::to_string(reversal.recovery_turns) : "null") << '}';
    }
    out << ']';
    out << '}';
    return out.str();
}

double ratio(double numerator, double denominator) {
    return denominator ? numerator / denominator : std::numeric_limits<double>::quiet_NaN();
}

double binomial_se(double successes, double observations) {
    if (!observations) return std::numeric_limits<double>::quiet_NaN();
    const double probability = successes / observations;
    return std::sqrt(probability * (1.0 - probability) / observations);
}

std::string summary_json(const Aggregate& aggregate, const Config& config, std::size_t cache_size) {
    std::ostringstream out;
    out << "{\"engine_version\":" << json_quote(ENGINE_VERSION)
        << ",\"evaluator_backend\":" << json_quote(EVALUATOR_BACKEND)
        << ",\"seed\":" << config.seed
        << ",\"start_trial\":" << config.start_trial
        << ",\"trials\":" << aggregate.trials
        << ",\"policy\":" << json_quote(config.policy.name())
        << ",\"policy_parameters\":" << policy_parameters_json(config)
        << ",\"scorer\":null"
        << ",\"config\":{\"stop_at\":"
        << json_quote(config.push_fold ? "table_terminal" : config.stop_at)
        << ",\"riichi_on_tenpai\":" << (config.riichi_on_tenpai ? "true" : "false")
        << ",\"opponent_draws_per_turn\":" << config.opponent_draws
        << ",\"player_wind\":" << config.player_wind
        << ",\"round_wind\":" << config.round_wind
        << ",\"tile_availability_turn\":" << config.tile_availability_turn;
    if (config.tile_danger) out << ",\"tile_danger\":true";
    if (config.push_fold) {
        out << ",\"push_fold\":true"
            << ",\"push_fold_mode\":" << json_quote(config.push_fold_mode)
            << ",\"safety_strategy\":" << json_quote(config.safety_strategy)
            << ",\"dora_danger_steps\":" << config.dora_danger_steps
            << ",\"half_push_max_tier\":" << config.half_push_max_tier
            << ",\"two_riichi_bad_wait_min_ron\":" << config.two_riichi_bad_wait_min_ron;
    }
    out << '}';
    out << ",\"outcomes\":{";
    bool first = true;
    for (const auto& item : aggregate.outcomes) {
        if (!first) out << ',';
        out << json_quote(item.first) << ':' << item.second;
        first = false;
    }
    out << "},\"tenpai\":{\"hands\":" << aggregate.tenpai_hands
        << ",\"rate\":" << number_json(ratio(aggregate.tenpai_hands, aggregate.trials))
        << ",\"rate_standard_error\":" << number_json(binomial_se(aggregate.tenpai_hands, aggregate.trials))
        << ",\"mean_first_turn\":" << number_json(ratio(aggregate.tenpai_turn_sum, aggregate.tenpai_hands))
        << ",\"ron_eligible_hands\":" << aggregate.ron_eligible
        << ",\"furiten_hands\":" << aggregate.tenpai_hands - aggregate.ron_eligible << '}'
        << ",\"wins\":{\"hands\":" << aggregate.wins
        << ",\"rate\":" << number_json(ratio(aggregate.wins, aggregate.trials))
        << ",\"rate_standard_error\":" << number_json(binomial_se(aggregate.wins, aggregate.trials))
        << ",\"mean_turn\":" << number_json(ratio(aggregate.win_turn_sum, aggregate.wins)) << '}';
    out << ",\"shanten_reversal\":{\"opportunities\":" << aggregate.reversal_opportunities
        << ",\"selected\":" << aggregate.reversals_selected
        << ",\"selection_rate_per_opportunity\":"
        << number_json(ratio(aggregate.reversals_selected, aggregate.reversal_opportunities))
        << ",\"mean_selection_turn\":"
        << number_json(ratio(aggregate.reversal_turn_sum, aggregate.reversals_selected))
        << ",\"recovered\":" << aggregate.reversals_recovered
        << ",\"recovery_rate\":"
        << number_json(ratio(aggregate.reversals_recovered, aggregate.reversals_selected))
        << ",\"mean_recovery_turns\":"
        << number_json(ratio(aggregate.reversal_recovery_turn_sum, aggregate.reversals_recovered))
        << ",\"transitions\":{\"0->1\":" << aggregate.reversal_transitions[0]
        << ",\"1->2\":" << aggregate.reversal_transitions[1] << "}}";
    out << ",\"wait_signatures\":{";
    first = true;
    for (const auto& item : aggregate.signatures) {
        if (!first) out << ',';
        out << json_quote(item.first) << ":{\"hands\":" << number_json(item.second)
            << ",\"share\":" << number_json(ratio(item.second, aggregate.tenpai_hands))
            << ",\"share_standard_error\":" << number_json(binomial_se(item.second, aggregate.tenpai_hands)) << '}';
        first = false;
    }
    out << "},\"acceptance_copy_share_by_shape\":{";
    first = true;
    for (int shape = 0; shape < 8; ++shape) {
        if (!aggregate.visible_by_shape[shape] && !aggregate.live_by_shape[shape]) continue;
        if (!first) out << ',';
        out << json_quote(SHAPE_NAMES[shape]) << ":{\"visible_copies\":" << number_json(aggregate.visible_by_shape[shape])
            << ",\"visible_share\":" << number_json(ratio(aggregate.visible_by_shape[shape], aggregate.visible_copies))
            << ",\"live_wall_copies\":" << number_json(aggregate.live_by_shape[shape])
            << ",\"live_wall_share\":" << number_json(ratio(aggregate.live_by_shape[shape], aggregate.live_copies)) << '}';
        first = false;
    }
    out << "},\"tile_danger_at_first_tenpai\":{";
    for (int tile = 0; tile < TILE_TYPES; ++tile) {
        if (tile) out << ',';
        out << json_quote(tile_name(tile)) << ":{\"wait_hands\":" << number_json(aggregate.tile_wait_hands[tile])
            << ",\"ron_eligible_tenpai_hands\":" << aggregate.ron_eligible
            << ",\"probability\":" << number_json(ratio(aggregate.tile_wait_hands[tile], aggregate.ron_eligible))
            << ",\"standard_error\":" << number_json(binomial_se(aggregate.tile_wait_hands[tile], aggregate.ron_eligible))
            << ",\"probability_by_shape_fractional\":{";
        bool shape_first = true;
        for (int shape = 0; shape < 8; ++shape) {
            if (!aggregate.tile_shape_hands[tile][shape]) continue;
            if (!shape_first) out << ',';
            out << json_quote(SHAPE_NAMES[shape]) << ':'
                << number_json(ratio(aggregate.tile_shape_hands[tile][shape], aggregate.ron_eligible));
            shape_first = false;
        }
        out << "}}";
    }
    out << "},\"score_values_on_realized_winning_hands\":null"
        << ",\"raw\":{\"tenpai_turn_sum\":" << aggregate.tenpai_turn_sum
        << ",\"win_turn_sum\":" << aggregate.win_turn_sum
        << ",\"visible_wait_copies\":" << number_json(aggregate.visible_copies)
        << ",\"live_wait_copies\":" << number_json(aggregate.live_copies)
        << ",\"reversal_opportunities\":" << aggregate.reversal_opportunities
        << ",\"reversals_selected\":" << aggregate.reversals_selected
        << ",\"reversals_recovered\":" << aggregate.reversals_recovered
        << ",\"reversal_turn_sum\":" << aggregate.reversal_turn_sum
        << ",\"reversal_recovery_turn_sum\":" << aggregate.reversal_recovery_turn_sum
        << ",\"shanten_cache_entries\":" << cache_size << "}}";
    return out.str();
}

std::string comparison_json(
    const Aggregate& baseline, const Config& baseline_config,
    const Aggregate& candidate, const Config& candidate_config, std::size_t cache_size
) {
    auto difference = [](double candidate_value, double baseline_value) {
        if (std::isnan(candidate_value) || std::isnan(baseline_value)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return candidate_value - baseline_value;
    };
    std::ostringstream out;
    out << "{\"paired_by_trial_seed\":true"
        << ",\"baseline\":" << summary_json(baseline, baseline_config, cache_size)
        << ",\"candidate\":" << summary_json(candidate, candidate_config, cache_size)
        << ",\"delta_candidate_minus_baseline\":{"
        << "\"tenpai_rate\":" << number_json(difference(
            ratio(candidate.tenpai_hands, candidate.trials), ratio(baseline.tenpai_hands, baseline.trials)))
        << ",\"mean_first_tenpai_turn\":" << number_json(difference(
            ratio(candidate.tenpai_turn_sum, candidate.tenpai_hands),
            ratio(baseline.tenpai_turn_sum, baseline.tenpai_hands)))
        << ",\"win_rate\":" << number_json(difference(
            ratio(candidate.wins, candidate.trials), ratio(baseline.wins, baseline.trials)))
        << ",\"mean_win_turn\":" << number_json(difference(
            ratio(candidate.win_turn_sum, candidate.wins), ratio(baseline.win_turn_sum, baseline.wins)))
        << ",\"reversals_selected\":" << candidate.reversals_selected
        << "}}";
    return out.str();
}

std::string analysis_json(
    const Counts& hand, const std::vector<int>& physical_hand, const Visible& visible,
    int dora_indicator, const Config& config, ShantenCalculator& calculator
) {
    if (tile_total(hand) != 13) throw std::invalid_argument("C++ analyze currently requires 13 tiles");
    auto features = evaluate_13(
        hand, -1, visible, config.future, config.improvement_weight, calculator
    );
    apply_value_features(
        features, physical_hand, dora_indicator, config.player_wind, config.round_wind, config.policy
    );
    if (config.policy.kanchan_outer_wait_weight != 0.0 && features.shanten == 0
        && is_pure_outer_kanchan(hand, calculator)) {
        features.tie_score += config.policy.kanchan_outer_wait_weight;
    }
    std::ostringstream out;
    out << "{\"engine_version\":" << json_quote(ENGINE_VERSION)
        << ",\"evaluator_backend\":" << json_quote(EVALUATOR_BACKEND)
        << ",\"hand\":" << json_quote(format_counts(hand))
        << ",\"shanten\":" << features.shanten
        << ",\"policy\":" << json_quote(config.policy.name())
        << ",\"policy_parameters\":" << policy_parameters_json(config)
        << ",\"dora_indicator\":"
        << (dora_indicator >= 0 ? json_quote(tile_name(dora_indicator / 4)) : "null")
        << ",\"dora\":"
        << (dora_indicator >= 0 ? json_quote(tile_name(dora_type_from_indicator(dora_indicator / 4))) : "null")
        << ",\"ukeire_count\":" << features.ukeire.count
        << ",\"future_ukeire_expected\":" << number_json(features.advanced.future_expected)
        << ",\"improvement_ukeire_count\":" << features.advanced.improvement_count
        << ",\"base_ukeire_score\":" << number_json(features.base_score)
        << ",\"dora_ukeire_bonus\":" << number_json(features.dora_ukeire_bonus)
        << ",\"ukeire_score\":" << number_json(features.score)
        << ",\"value_features\":{\"retained_dora\":" << features.retained_dora
        << ",\"retained_dora_neighbors\":" << features.retained_dora_neighbors
        << ",\"retained_yakuhai_value\":" << features.retained_yakuhai_value
        << ",\"retained_tanyao_tiles\":" << features.retained_tanyao_tiles
        << ",\"tiebreak_score\":" << number_json(features.tie_score) << '}'
        << ",\"ukeire\":[";
    for (std::size_t index = 0; index < features.ukeire.types.size(); ++index) {
        if (index) out << ',';
        const double future = index < features.advanced.future_by_type.size()
            ? features.advanced.future_by_type[index] : 0.0;
        out << "{\"tile\":" << json_quote(tile_name(features.ukeire.types[index]))
            << ",\"remaining\":" << features.ukeire.remaining[index]
            << ",\"future_ukeire\":" << number_json(future) << '}';
    }
    out << "],\"improvement_ukeire\":[";
    for (std::size_t index = 0; index < features.advanced.improvement_types.size(); ++index) {
        if (index) out << ',';
        out << "{\"tile\":" << json_quote(tile_name(features.advanced.improvement_types[index]))
            << ",\"remaining\":" << features.advanced.improvement_remaining[index] << '}';
    }
    out << "]}";
    return out.str();
}

int wind_index(const std::string& value) {
    const std::array<std::string, 4> winds = {"east", "south", "west", "north"};
    const auto found = std::find(winds.begin(), winds.end(), value);
    if (found == winds.end()) throw std::invalid_argument("wind must be east/south/west/north");
    return 27 + static_cast<int>(found - winds.begin());
}

void self_test() {
    ShantenCalculator calculator;
    const Counts ryanmen = parse_hand("123456789m11p23s");
    if (calculator.calculate(ryanmen) != 0) throw std::runtime_error("ryanmen shanten failed");
    const auto waits = structural_ukeire_types(ryanmen, calculator);
    if (waits != std::vector<int>({18, 21})) throw std::runtime_error("ryanmen ukeire failed");
    if (!is_permanent_furiten(ryanmen, {21 * 4}, calculator)
        || is_permanent_furiten(ryanmen, {20 * 4}, calculator)) {
        throw std::runtime_error("permanent furiten failed");
    }
    if (wait_shapes(ryanmen, 18, calculator) != std::vector<int>({0})) throw std::runtime_error("wait shape failed");
    Counts complete = ryanmen;
    ++complete[18];
    if (calculator.calculate(complete) != -1) throw std::runtime_error("complete hand failed");
    if (calculator.calculate(parse_hand("1122m3344p5566s7z")) != 0) throw std::runtime_error("chiitoitsu failed");
    if (calculator.calculate(parse_hand("19m19p19s1234567z")) != 0) throw std::runtime_error("kokushi failed");

    const Counts advanced = parse_hand("2378m234789p1s36z");
    Visible visible{};
    FutureUkeireConfig future;
    const auto features = evaluate_13(advanced, -1, visible, future, 0.1, calculator);
    if (features.shanten != 2 || features.ukeire.count != 25 ||
        std::abs(features.advanced.future_expected - 14.72) > 1e-9 ||
        features.advanced.improvement_count != 45 || std::abs(features.score - 36.86) > 1e-9) {
        throw std::runtime_error("advanced ukeire fixture failed");
    }
    std::vector<int> reversal_hand = parse_physical_hand("46777m1p1233345s6z");
    ReversalConfig reversal;
    reversal.enabled = true;
    PolicyConfig pure;
    Rng reversal_rng(7);
    const auto reversal_choice = choose_discard(
        reversal_hand, {}, parse_single_tile_type("9p") * 4, 28, 27,
        future, 0.1, pure, reversal, 5, 0, reversal_rng, calculator
    );
    if (!reversal_choice.features.is_reversal || reversal_choice.features.discard_type != 5
        || reversal_choice.features.strict_best_shanten != 1
        || reversal_choice.features.shanten != 2
        || reversal_choice.features.reversal_recovery_count != 49
        || std::abs(reversal_choice.features.reversal_future_expected - 18.26530612244898) > 1e-9
        || reversal_choice.features.reversal_current_ukeire != 12
        || std::abs(reversal_choice.features.reversal_ukeire_gain - 6.26530612244898) > 1e-9) {
        throw std::runtime_error("shanten reversal fixture failed");
    }
    Config availability;
    availability.tile_availability_turn = 1;
    availability.riichi_on_tenpai = false;
    const auto availability_table = simulate_table_trial(availability, 0, calculator);
    if (!availability_table.winner.availability_checkpoint
        || availability_table.winner.outcome != "availability_checkpoint") {
        throw std::runtime_error("availability checkpoint fixture failed");
    }
    const auto& checkpoint = *availability_table.winner.availability_checkpoint;
    if (checkpoint.target_own_turn != 1 || checkpoint.completed_own_turn != 1
        || checkpoint.live_wall_tiles_remaining != 66
        || checkpoint.hidden_dead_wall_tiles_remaining != 13
        || checkpoint.opponent_hand_tiles != 39) {
        throw std::runtime_error("availability checkpoint totals failed");
    }
    for (const auto& player : checkpoint.players) {
        if (player.unseen_tiles != 118 || player.live_wall_tiles != 66
            || player.hidden_dead_wall_tiles != 13 || player.opponent_hand_tiles != 39) {
            throw std::runtime_error("availability player totals failed");
        }
    }
    ScoreReply late_riichi_boundary;
    late_riichi_boundary.min_dama_ron_points = 5200;
    late_riichi_boundary.min_riichi_ron_points = 7700;
    if (two_threat_intended_min_ron(late_riichi_boundary, 4) != 7700
        || two_threat_intended_min_ron(late_riichi_boundary, 3) != 5200) {
        throw std::runtime_error("late riichi intended-action boundary failed");
    }
    late_riichi_boundary.valuable_dama = true;
    if (two_threat_intended_min_ron(late_riichi_boundary, 4) != 5200) {
        throw std::runtime_error("valuable dama intended-action boundary failed");
    }
    std::cout << "self-test: ok\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument(
            "usage: hand_efficiency_sim_cpp analyze|simulate|simulate-table|compare|--self-test"
        );
        const std::string command = argv[1];
        if (command == "--self-test") {
            self_test();
            return 0;
        }
        Config config;
        if (command == "analyze") {
            if (argc < 3) throw std::invalid_argument("analyze requires a 13-tile mpsz hand");
            const Counts hand = parse_hand(argv[2]);
            const std::vector<int> physical_hand = parse_physical_hand(argv[2]);
            Visible visible{};
            int dora_indicator = -1;
            for (int index = 3; index < argc; ++index) {
                const std::string arg = argv[index];
                auto value = [&]() -> std::string {
                    if (++index >= argc) throw std::invalid_argument("missing value for " + arg);
                    return argv[index];
                };
                if (arg == "--visible") {
                    const Counts parsed = parse_hand(value());
                    for (int tile = 0; tile < TILE_TYPES; ++tile) visible[tile] = parsed[tile];
                } else if (arg == "--dora-indicator") {
                    dora_indicator = parse_single_tile_type(value()) * 4;
                } else if (arg == "--seat-wind") config.player_wind = wind_index(value());
                else if (arg == "--round-wind") config.round_wind = wind_index(value());
                else if (arg == "--policy") config.policy.preset = value();
                else if (arg == "--future-ukeire-weight") config.future.fallback = std::stod(value());
                else if (arg == "--future-ukeire-weight-1-shanten") config.future.one_shanten = std::stod(value());
                else if (arg == "--future-ukeire-weight-2-shanten") config.future.two_shanten = std::stod(value());
                else if (arg == "--improvement-weight") config.improvement_weight = std::stod(value());
                else if (arg == "--allow-shanten-reversal") config.reversal.enabled = true;
                else if (arg == "--shanten-reversal-penalty") config.reversal.penalty = std::stod(value());
                else if (arg == "--shanten-reversal-future-weight") config.reversal.future_weight = std::stod(value());
                else if (arg == "--shanten-reversal-max-turn") config.reversal.max_turn = std::stoi(value());
                else if (arg == "--shanten-reversal-candidate-limit") config.reversal.candidate_limit = std::stoi(value());
                else if (arg == "--dora-ukeire-weight") {
                    config.policy.dora_ukeire_weight = std::stod(value()); config.policy_weight_set[0] = true;
                } else if (arg == "--dora-tiebreak-weight") {
                    config.policy.dora_tiebreak_weight = std::stod(value()); config.policy_weight_set[1] = true;
                } else if (arg == "--dora-neighbor-weight") {
                    config.policy.dora_neighbor_weight = std::stod(value()); config.policy_weight_set[2] = true;
                } else if (arg == "--yakuhai-weight") {
                    config.policy.yakuhai_weight = std::stod(value()); config.policy_weight_set[3] = true;
                } else if (arg == "--tanyao-weight") {
                    config.policy.tanyao_weight = std::stod(value()); config.policy_weight_set[4] = true;
                }
                else if (arg == "--kanchan-outer-wait-weight") {
                    config.policy.kanchan_outer_wait_weight = std::stod(value());
                }
                else throw std::invalid_argument("unknown analyze argument: " + arg);
            }
            finalize_policy(config);
            if (dora_indicator >= 0) ++visible[dora_indicator / 4];
            ShantenCalculator calculator;
            std::cout << analysis_json(
                hand, physical_hand, visible, dora_indicator, config, calculator
            ) << '\n';
            return 0;
        }
        const bool compare = command == "compare";
        const bool table_mode = command == "simulate-table";
        if (command != "simulate" && !table_mode && !compare) {
            throw std::invalid_argument("unknown command: " + command);
        }
        if (table_mode) config.stop_at = "tenpai";
        if (compare) config.policy.preset = "value";
        for (int index = 2; index < argc; ++index) {
            const std::string arg = argv[index];
            auto value = [&]() -> std::string {
                if (++index >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[index];
            };
            if (arg == "--trials") config.trials = std::stoull(value());
            else if (arg == "--seed") config.seed = std::stoull(value());
            else if (arg == "--start-trial") config.start_trial = std::stoull(value());
            else if (arg == "--stop-at") config.stop_at = value();
            else if (arg == "--seat-wind") config.player_wind = wind_index(value());
            else if (arg == "--round-wind") config.round_wind = wind_index(value());
            else if (arg == "--policy") config.policy.preset = value();
            else if (arg == "--future-ukeire-weight") config.future.fallback = std::stod(value());
            else if (arg == "--future-ukeire-weight-1-shanten") config.future.one_shanten = std::stod(value());
            else if (arg == "--future-ukeire-weight-2-shanten") config.future.two_shanten = std::stod(value());
            else if (arg == "--improvement-weight") config.improvement_weight = std::stod(value());
            else if (arg == "--allow-shanten-reversal") config.reversal.enabled = true;
            else if (arg == "--shanten-reversal-penalty") config.reversal.penalty = std::stod(value());
            else if (arg == "--shanten-reversal-future-weight") config.reversal.future_weight = std::stod(value());
            else if (arg == "--shanten-reversal-max-turn") config.reversal.max_turn = std::stoi(value());
            else if (arg == "--shanten-reversal-candidate-limit") config.reversal.candidate_limit = std::stoi(value());
            else if (arg == "--dora-ukeire-weight") {
                config.policy.dora_ukeire_weight = std::stod(value()); config.policy_weight_set[0] = true;
            } else if (arg == "--dora-tiebreak-weight") {
                config.policy.dora_tiebreak_weight = std::stod(value()); config.policy_weight_set[1] = true;
            } else if (arg == "--dora-neighbor-weight") {
                config.policy.dora_neighbor_weight = std::stod(value()); config.policy_weight_set[2] = true;
            } else if (arg == "--yakuhai-weight") {
                config.policy.yakuhai_weight = std::stod(value()); config.policy_weight_set[3] = true;
            } else if (arg == "--tanyao-weight") {
                config.policy.tanyao_weight = std::stod(value()); config.policy_weight_set[4] = true;
            }
            else if (arg == "--kanchan-outer-wait-weight") {
                config.policy.kanchan_outer_wait_weight = std::stod(value());
            }
            else if (arg == "--observations") config.observations = value();
            else if (arg == "--tile-availability-turn") {
                config.tile_availability_turn = std::stoi(value());
            }
            else if (arg == "--tile-danger") config.tile_danger = true;
            else if (arg == "--push-fold") config.push_fold = true;
            else if (arg == "--push-fold-mode") config.push_fold_mode = value();
            else if (arg == "--safety-strategy") config.safety_strategy = value();
            else if (arg == "--dora-danger-steps") config.dora_danger_steps = std::stoi(value());
            else if (arg == "--half-push-max-tier") config.half_push_max_tier = std::stoi(value());
            else if (arg == "--two-riichi-bad-wait-min-ron") config.two_riichi_bad_wait_min_ron = std::stoi(value());
            else if (arg == "--score-service") config.score_service = true;
            else if (arg == "--no-riichi") config.riichi_on_tenpai = false;
            else if (arg == "--continue-rejected-tenpai") config.continue_table_tenpai = true;
            else if (arg == "--valuable-dama-directives") config.valuable_dama_directives_path = value();
            else throw std::invalid_argument("unknown simulate argument: " + arg);
        }
        if (config.stop_at != "tenpai" && config.stop_at != "win") throw std::invalid_argument("stop-at must be tenpai or win");
        if (config.tile_availability_turn < -1) {
            throw std::invalid_argument("tile availability turn must be nonnegative");
        }
        if (config.tile_availability_turn > LIVE_WALL / 4) {
            throw std::invalid_argument("tile availability turn must be at most 17");
        }
        if (config.tile_availability_turn >= 0 && !table_mode) {
            throw std::invalid_argument("tile availability requires simulate-table");
        }
        if (config.tile_danger && !table_mode) throw std::invalid_argument("tile danger requires simulate-table");
        if (config.tile_danger && (config.tile_availability_turn >= 0 || config.continue_table_tenpai
            || !config.riichi_on_tenpai || !config.valuable_dama_directives_path.empty())) {
            throw std::invalid_argument("tile danger is incompatible with availability, continuation, no-riichi, and dama directives");
        }
        if (config.push_fold && !table_mode) throw std::invalid_argument("push-fold requires simulate-table");
        if (config.push_fold && !config.score_service) throw std::invalid_argument("push-fold requires --score-service");
        if (config.push_fold && (config.tile_danger || config.tile_availability_turn >= 0
            || config.continue_table_tenpai || !config.valuable_dama_directives_path.empty())) {
            throw std::invalid_argument("push-fold is incompatible with danger, availability, and legacy continuation directives");
        }
        if (config.push_fold_mode != "all_push" && config.push_fold_mode != "full_fold" && config.push_fold_mode != "mixed") throw std::invalid_argument("push-fold-mode must be all_push, full_fold, or mixed");
        if (config.safety_strategy != "mean_safety" && config.safety_strategy != "minimax_safety") throw std::invalid_argument("safety-strategy must be mean_safety or minimax_safety");
        if (config.dora_danger_steps < 0 || config.half_push_max_tier < 0
            || config.two_riichi_bad_wait_min_ron < 0) {
            throw std::invalid_argument("push-fold numeric thresholds must be nonnegative");
        }
        finalize_policy(config);
        load_dama_directives(config);

        if (compare) {
            if (!config.observations.empty()) {
                throw std::invalid_argument("compare does not write observations; run each policy separately for JSONL");
            }
            Config baseline_config = config;
            baseline_config.policy.preset = "pure";
            baseline_config.policy.dora_ukeire_weight = 0.0;
            baseline_config.policy.dora_tiebreak_weight = 0.0;
            baseline_config.policy.dora_neighbor_weight = 0.0;
            baseline_config.policy.yakuhai_weight = 0.0;
            baseline_config.policy.tanyao_weight = 0.0;
            baseline_config.reversal.enabled = false;
            baseline_config.policy_weight_set.fill(false);
            finalize_policy(baseline_config);
            ShantenCalculator calculator;
            Aggregate baseline;
            Aggregate candidate;
            for (std::uint64_t trial = config.start_trial; trial < config.start_trial + config.trials; ++trial) {
                baseline.add(simulate_trial(baseline_config, trial, calculator));
                candidate.add(simulate_trial(config, trial, calculator));
            }
            std::cout << comparison_json(
                baseline, baseline_config, candidate, config, calculator.cache_size()
            ) << '\n';
            return 0;
        }

        std::ofstream observations;
        if (!config.observations.empty()) {
            observations.open(config.observations);
            if (!observations) throw std::runtime_error("could not open observations output");
        }
        ShantenCalculator calculator;
        Aggregate aggregate;
        for (std::uint64_t trial = config.start_trial; trial < config.start_trial + config.trials; ++trial) {
            if (table_mode) {
                const TableTrialResult table = simulate_table_trial(config, trial, calculator);
                aggregate.add(table.winner);
                if (observations) {
                    Config winner_config = config;
                    if (table.winner_seat >= 0) winner_config.player_wind = 27 + table.winner_seat;
                    observations << trial_json(table.winner, winner_config) << '\n';
                }
            } else {
                const TrialResult result = simulate_trial(config, trial, calculator);
                aggregate.add(result);
                if (observations) observations << trial_json(result, config) << '\n';
            }
        }
        std::cout << summary_json(aggregate, config, calculator.cache_size()) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
