#pragma once

// Cumulative counters behind GET /metrics, in the flat `name value` subset of
// the Prometheus text format.
//
// The four llamacpp:-prefixed counters reproduce llama.cpp's --metrics
// semantics - computed prefill tokens (prefix-cache hits excluded) billed
// against prefill unit time, committed decode tokens against decode unit
// time - so scrapers that difference llama.cpp counters read this server
// without changes. They are sourced from the Engine's live per-unit totals,
// so they advance during a request like llama.cpp's do, not only at its
// completion. The ninfer:-prefixed series report what llama.cpp cannot:
// speculative draft/acceptance totals and prefix-cache reuse.
//
// The same counters, the Engine's KV and scheduler gauges, the slot table and the most recent
// completed requests also form the JSON snapshot behind GET /monitor/stats, which the page at
// GET /monitor polls; rates are differenced by the page.

#include "serve/generation_service.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

class ServeMetrics {
public:
    // Accumulates one completed request. Called from the same funnel as the
    // request-done log line, so every protocol and both streaming modes count.
    void record(const GenerationOutcome& outcome);

    // Prompt/cache sizes of the most recent completed request, retained for
    // /slots. llama.cpp keeps the last request's counts on an idle slot and
    // scrapers (the fleet dashboard) read them as the resident session
    // depth; the prefix cache genuinely still holds that session, so the
    // retained figure stays truthful until the next completion replaces it.
    struct LastCompleted {
        int prompt_tokens = 0;
        int cached_tokens = 0;
    };
    [[nodiscard]] LastCompleted last_completed() const;

    // Summary of one completed request, newest first in recent_requests().
    struct RecentRequest {
        std::uint64_t sequence  = 0; // 1-based completion order
        int prompt_tokens       = 0;
        int cached_tokens       = 0;
        int completion_tokens   = 0;
        int reasoning_tokens    = 0;
        double ttft_seconds     = 0.0;
        double decode_seconds   = 0.0;
        double total_seconds    = 0.0;
        std::uint64_t drafted   = 0;
        std::uint64_t accepted  = 0;
        ninfer::FinishReason finish_reason = ninfer::FinishReason::None;
    };
    static constexpr std::size_t kRecentRequests = 32;
    [[nodiscard]] std::vector<RecentRequest> recent_requests() const;

    // Static facts of the running server shown by the monitor.
    struct MonitorContext {
        std::string model;
        std::uint32_t max_context        = 0;
        std::uint32_t lanes              = 0; // concurrent requests (--max-concurrency)
        std::uint32_t kv_capacity_tokens = 0;
        std::uint32_t kv_pages           = 0;
        std::uint32_t draft_window       = 0; // 0 without speculative decoding
    };

    // The GET /monitor/stats JSON body. `slots` are the retained-conversation cells of /slots
    // (at least one per lane). The Engine totals are reported relative to `baseline`, the stats
    // at attach, so the startup warmup generation does not count as served traffic.
    // `active_requests` is the same request-lifetime count render() receives.
    [[nodiscard]] std::string render_monitor(const MonitorContext& context,
                                             const ninfer::RuntimeStats& live,
                                             const ninfer::RuntimeStats& baseline,
                                             const std::vector<ninfer::SlotState>& slots,
                                             std::size_t active_requests) const;

    // One complete Prometheus text body, without HTTP framing. The supplied
    // request-lifetime count begins before preparation/submission and survives
    // through response release, so every accepted request remains visible.
    // `live` supplies the four llamacpp token/seconds counters from the Engine's per-unit
    // totals, so scrapers see rates advance during a request; the completion-based sums this
    // class accumulates back the ninfer: series and the idle slot display.
    [[nodiscard]] std::string render(std::uint32_t max_concurrency,
                                     const ninfer::RuntimeStats& live,
                                     std::size_t active_requests) const;

private:
    mutable std::mutex mutex_;
    std::uint64_t requests_total_                    = 0;
    std::uint64_t prompt_tokens_total_               = 0;
    std::uint64_t prefix_cache_hit_tokens_total_     = 0;
    std::uint64_t speculative_draft_tokens_total_    = 0;
    std::uint64_t speculative_accepted_tokens_total_ = 0;
    std::uint64_t ngram_draft_tokens_total_          = 0;
    std::uint64_t ngram_accepted_tokens_total_       = 0;
    LastCompleted last_completed_;
    std::deque<RecentRequest> recent_; // newest first, at most kRecentRequests
};

} // namespace ninfer::serve
