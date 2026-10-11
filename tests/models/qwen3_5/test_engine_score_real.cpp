#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != ninfer::KvCacheStorage::Fp8E4M3Row256) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1).logprobs;
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513).logprobs;
    const std::vector<float> repeated = engine.score_tokens(tokens, 513).logprobs;
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    for (const float value : all) {
        if (!std::isfinite(value) || value > 0.0F) {
            std::cerr << "causal scoring returned an invalid log probability\n";
            return 1;
        }
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(suffix[i]) || suffix[i] > 0.0F) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }

    // Captured outputs. Capturing must not change the log-probabilities; the hidden rows of the
    // suffix window are the full window's rows for the same predictors (both prefill from token
    // 0 in the same chunks); a target replaced by its predictor's argmax is at least as likely
    // as the original target under the same prefix.
    const ninfer::CausalScoreCapture capture{.hidden = true, .argmax = true};
    const ninfer::CausalScore full     = engine.score_tokens(tokens, 1, capture);
    const ninfer::CausalScore tail     = engine.score_tokens(tokens, 513, capture);
    const std::size_t width            = full.hidden_size;
    if (full.logprobs != all || tail.logprobs != suffix || width == 0 ||
        full.hidden.size() != all.size() * width || full.argmax.size() != all.size() ||
        tail.hidden_size != width || tail.hidden.size() != suffix.size() * width ||
        tail.argmax.size() != suffix.size()) {
        std::cerr << "captured causal scoring changed log-probabilities or has a wrong shape\n";
        return 1;
    }
    if (!std::equal(tail.hidden.begin(), tail.hidden.end(),
                    full.hidden.begin() + static_cast<std::ptrdiff_t>(512 * width))) {
        std::cerr << "suffix hidden rows differ from the full window's rows\n";
        return 1;
    }
    for (const std::uint16_t bits : full.hidden) {
        if ((bits & 0x7f80U) == 0x7f80U) {
            std::cerr << "captured hidden state is not finite\n";
            return 1;
        }
    }
    for (const std::size_t target : {std::size_t{600}, std::size_t{1000}, std::size_t{1400}}) {
        const ninfer::TokenId best = full.argmax[target - 1];
        if (best < 0) {
            std::cerr << "captured argmax is negative\n";
            return 1;
        }
        std::vector<ninfer::TokenId> replaced = tokens;
        replaced[target]                      = best;
        const float best_logprob =
            engine.score_tokens(replaced, static_cast<std::uint32_t>(target)).logprobs.front();
        if (best_logprob < all[target - 1] - 1e-3F) {
            std::cerr << "the captured argmax is less likely than the original target\n";
            return 1;
        }
    }
    std::cout << "OK causal_score_real\n";
    return 0;
}
