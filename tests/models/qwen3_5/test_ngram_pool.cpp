#include "models/qwen3_5/program/speculative/ngram_pool.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::models::qwen3_5::NgramDraftPool;
using ninfer::models::qwen3_5::NgramPoolSpec;

constexpr std::int32_t kQwenTokenDomain = 248077;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<TokenId> propose(const NgramDraftPool& pool, std::span<const TokenId> context,
                             std::size_t limit) {
    std::vector<TokenId> out(limit);
    out.resize(pool.propose(context, out));
    return out;
}

template <class F>
bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

// The same vector is pinned by tests/test_spec_sim.py, so the offline simulator models exactly
// this pool's slots, tags and collisions.
void test_hash_contract_matches_simulator() {
    const std::array<TokenId, 3> window{1, 2, 3};
    const std::uint64_t hash = NgramDraftPool::window_hash(window);
    check(hash == 0x190380fc9abaac46ULL, "window hash differs from the simulator vector");
    check(NgramDraftPool::mix(hash) == 0x2f67e0cd700b3673ULL, "fmix64 differs");
    check(NgramDraftPool::slot(hash, 4U * 1024U * 1024U) == 734835U, "slot differs");
    check(NgramDraftPool::tag(NgramDraftPool::mix(hash)) == 3033U, "tag differs");
    const std::array<TokenId, 4> top{248076, 248076, 248076, 248076};
    const std::uint64_t top_hash = NgramDraftPool::window_hash(top);
    check(NgramDraftPool::slot(top_hash, 1000003U) == 967702U, "top-id slot differs");
    check(NgramDraftPool::tag(NgramDraftPool::mix(top_hash)) == 6786U, "top-id tag differs");
}

void test_walk_reproduces_repeated_span() {
    NgramDraftPool pool({.match_tokens = 3, .entries = 1U << 16, .token_domain = kQwenTokenDomain});
    std::vector<TokenId> block;
    for (TokenId t = 100; t < 140; ++t) { block.push_back(t); }
    std::vector<TokenId> history = block;
    history.insert(history.end(), {7, 7, 7});
    history.insert(history.end(), block.begin(), block.begin() + 3);
    pool.observe(history, 0);

    std::vector<TokenId> expected(block.begin() + 3, block.end());
    expected.insert(expected.end(), {7, 7, 7});
    expected.insert(expected.end(), block.begin(), block.end());
    expected.resize(64);
    check(propose(pool, history, 64) == expected, "walk did not follow the recorded cycle");
    check(propose(pool, history, 5) == std::vector<TokenId>(block.begin() + 3, block.begin() + 8),
          "walk ignored the output bound");
    check(propose(pool, std::span<const TokenId>(history).first(2), 8).empty(),
          "context shorter than n proposed a draft");
    check(propose(pool, history, 0).empty(), "empty output received a draft");
    const std::array<TokenId, 3> unseen{1, 2, 3};
    check(propose(pool, unseen, 8).empty(), "unseen n-gram proposed a draft");
}

void test_incremental_observation_equals_bulk() {
    std::mt19937 rng(3);
    std::uniform_int_distribution<TokenId> token(0, 49);
    std::vector<TokenId> tokens(500);
    for (TokenId& t : tokens) { t = token(rng); }

    NgramDraftPool bulk({.match_tokens = 4, .entries = 997, .token_domain = kQwenTokenDomain});
    bulk.observe(tokens, 0);
    NgramDraftPool incremental(
        {.match_tokens = 4, .entries = 997, .token_domain = kQwenTokenDomain});
    std::uniform_int_distribution<std::size_t> step(1, 8);
    for (std::size_t end = 0; end < tokens.size();) {
        const std::size_t next = std::min(tokens.size(), end + step(rng));
        incremental.observe(std::span<const TokenId>(tokens).first(next), end);
        end = next;
    }
    check(bulk.occupied() == incremental.occupied(), "incremental occupancy differs from bulk");
    bool same = true;
    for (std::size_t end = 4; end <= tokens.size(); ++end) {
        const auto context = std::span<const TokenId>(tokens).first(end);
        same = same && propose(bulk, context, 16) == propose(incremental, context, 16);
    }
    check(same, "incremental observation proposes differently from bulk observation");
}

void test_latest_continuation_wins_and_clear_empties() {
    NgramDraftPool pool({.match_tokens = 2, .entries = 64, .token_domain = kQwenTokenDomain});
    const std::array<TokenId, 3> first{1, 2, 3};
    const std::array<TokenId, 3> second{1, 2, 4};
    pool.observe(first, 0);
    pool.observe(second, 0);
    const std::array<TokenId, 2> key{1, 2};
    check(propose(pool, key, 1) == std::vector<TokenId>{4}, "later continuation did not win");
    pool.clear();
    check(pool.occupied() == 0 && propose(pool, key, 1).empty(), "clear left entries behind");
}

void test_tag_rejects_colliding_window() {
    // One slot: every window collides, and only the tag distinguishes them.
    NgramDraftPool pool({.match_tokens = 2, .entries = 1, .token_domain = kQwenTokenDomain});
    const std::array<TokenId, 3> recorded{1, 2, 5};
    pool.observe(recorded, 0);
    const std::array<TokenId, 2> same{1, 2};
    const std::array<TokenId, 2> other{3, 4};
    check(propose(pool, same, 1) == std::vector<TokenId>{5}, "recorded window missed");
    check(propose(pool, other, 1).empty(), "colliding window with another tag was accepted");
}

void test_domain_and_spec_validation() {
    NgramDraftPool pool({.match_tokens = 2, .entries = 64, .token_domain = 10});
    const std::array<TokenId, 3> good{1, 2, 3};
    pool.observe(good, 0);
    const std::array<TokenId, 4> bad{1, 2, 3, 10};
    check(throws_invalid([&] { pool.observe(bad, 0); }), "out-of-domain continuation accepted");
    check(pool.occupied() == 1, "rejected observation mutated the pool");
    const std::array<TokenId, 2> key{2, 3};
    check(propose(pool, key, 1).empty(), "rejected observation recorded a prefix");
    check(throws_invalid([&] { pool.observe(good, 4); }), "observation past history accepted");
    // Keys are hashed, not stored, so only continuations need to be in the domain.
    const std::array<TokenId, 3> key_outside{-1, 99, 3};
    pool.observe(key_outside, 0);
    const std::array<TokenId, 2> probe{-1, 99};
    check(propose(pool, probe, 1) == std::vector<TokenId>{3}, "out-of-domain key not recorded");

    check(throws_invalid(
              [] { NgramDraftPool({.match_tokens = 0, .entries = 1, .token_domain = 1}); }),
          "zero match length accepted");
    check(throws_invalid(
              [] { NgramDraftPool({.match_tokens = 65, .entries = 1, .token_domain = 1}); }),
          "match length above 64 accepted");
    check(throws_invalid(
              [] { NgramDraftPool({.match_tokens = 1, .entries = 0, .token_domain = 1}); }),
          "zero entries accepted");
    check(throws_invalid(
              [] { NgramDraftPool({.match_tokens = 1, .entries = 1, .token_domain = 1 << 18}); }),
          "token domain above 2^18-1 accepted");
    NgramDraftPool sized(
        {.match_tokens = 24, .entries = 4U << 20, .token_domain = kQwenTokenDomain});
    check(sized.memory_bytes() == 16U << 20, "16 MiB pool does not hold 4 Mi entries");
}

} // namespace

int main() {
    test_hash_contract_matches_simulator();
    test_walk_reproduces_repeated_span();
    test_incremental_observation_equals_bulk();
    test_latest_continuation_wins_and_clear_empties();
    test_tag_rejects_colliding_window();
    test_domain_and_spec_validation();
    if (failures != 0) {
        std::cerr << failures << " n-gram pool check(s) failed\n";
        return 1;
    }
    return 0;
}
