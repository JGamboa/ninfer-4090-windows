#include "serve/serve_metrics.h"

#include "serve/operational_log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>

namespace ninfer::serve {

namespace {

void append_counter(std::string& out, const char* name, std::uint64_t value) {
    char line[160];
    std::snprintf(line, sizeof(line), "%s %llu\n", name,
                  static_cast<unsigned long long>(value));
    out += line;
}

void append_counter(std::string& out, const char* name, double value) {
    char line[160];
    std::snprintf(line, sizeof(line), "%s %.6f\n", name, value);
    out += line;
}

} // namespace

void ServeMetrics::begin_request(std::uint64_t id, int prompt_tokens) {
    const std::lock_guard<std::mutex> lock(mutex_);
    active_[id] = prompt_tokens > 0 ? prompt_tokens : 0;
}

void ServeMetrics::end_request(std::uint64_t id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    active_.erase(id);
}

std::vector<std::pair<std::uint64_t, int>> ServeMetrics::active_snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return {active_.begin(), active_.end()};
}

void ServeMetrics::record(const GenerationOutcome& outcome) {
    const GenerationMetrics& m = outcome.metrics;
    const std::uint64_t cached = m.prefix_cache_hit_tokens;
    const std::uint64_t prompt = outcome.prompt_tokens > 0
                                     ? static_cast<std::uint64_t>(outcome.prompt_tokens)
                                     : 0;

    const std::lock_guard<std::mutex> lock(mutex_);
    requests_total_ += 1;
    prefix_cache_hit_tokens_total_ += cached;
    speculative_draft_tokens_total_ += m.speculative_draft_tokens;
    speculative_accepted_tokens_total_ += m.speculative_accepted_tokens;
    last_completed_.prompt_tokens = static_cast<int>(prompt);
    // Clamped like computed_prefill above: a cache figure reported larger
    // than the prompt must not advertise more resident tokens than exist.
    last_completed_.cached_tokens = static_cast<int>(std::min(cached, prompt));

    RecentRequest recent;
    recent.sequence          = requests_total_;
    recent.prompt_tokens     = last_completed_.prompt_tokens;
    recent.cached_tokens     = last_completed_.cached_tokens;
    recent.completion_tokens = outcome.completion_tokens;
    recent.reasoning_tokens  = outcome.reasoning_tokens;
    recent.ttft_seconds      = m.ttft_seconds;
    recent.decode_seconds    = m.decode_seconds;
    recent.total_seconds     = m.total_seconds;
    recent.drafted           = m.speculative_draft_tokens;
    recent.accepted          = m.speculative_accepted_tokens;
    recent.finish_reason     = outcome.finish_reason;
    recent_.push_front(recent);
    if (recent_.size() > kRecentRequests) { recent_.pop_back(); }
}

std::vector<ServeMetrics::RecentRequest> ServeMetrics::recent_requests() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return {recent_.begin(), recent_.end()};
}

std::string ServeMetrics::render_monitor(const MonitorContext& context,
                                         const ninfer::RuntimeStats& live,
                                         const std::vector<ninfer::SlotState>& slots) const {
    nlohmann::json slot_rows = nlohmann::json::array();
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const ninfer::SlotState& slot = slots[i];
        slot_rows.push_back({{"id", i},
                             {"processing", slot.processing},
                             {"retained", slot.retained},
                             {"prompt_tokens", slot.prompt_tokens},
                             {"cached_tokens", slot.cached_tokens},
                             {"checkpoints", slot.checkpoints.size()}});
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json recent = nlohmann::json::array();
    for (const RecentRequest& r : recent_) {
        recent.push_back({{"sequence", r.sequence},
                          {"prompt_tokens", r.prompt_tokens},
                          {"cached_tokens", r.cached_tokens},
                          {"completion_tokens", r.completion_tokens},
                          {"reasoning_tokens", r.reasoning_tokens},
                          {"ttft_seconds", r.ttft_seconds},
                          {"decode_seconds", r.decode_seconds},
                          {"total_seconds", r.total_seconds},
                          {"drafted", r.drafted},
                          {"accepted", r.accepted},
                          {"finish_reason", finish_reason_name(r.finish_reason)}});
    }
    const nlohmann::json body{
        {"model", context.model},
        {"max_context", context.max_context},
        {"max_concurrency", context.max_concurrency},
        {"draft_window", context.draft_window},
        {"kv",
         {{"capacity_tokens", context.kv_capacity_tokens},
          {"pages", context.kv_pages},
          {"occupied_pages", live.device_main_kv_occupied_pages},
          {"host_occupied_bytes", live.host_kv_occupied_bytes}}},
        {"scheduler",
         {{"in_flight", active_.size()},
          {"running", live.running_requests},
          {"prefilling", live.prefilling_requests},
          {"decode_ready", live.decode_ready_requests},
          {"waiting", live.waiting_requests}}},
        {"totals",
         {{"requests", requests_total_},
          {"prefill_tokens", live.computed_prefill_tokens},
          {"prefill_seconds", live.prefill_seconds_total},
          {"decode_tokens", live.committed_decode_tokens},
          {"decode_seconds", live.decode_seconds_total},
          {"cached_prompt_tokens", prefix_cache_hit_tokens_total_},
          {"drafted", speculative_draft_tokens_total_},
          {"accepted", speculative_accepted_tokens_total_}}},
        {"slots", std::move(slot_rows)},
        {"recent", std::move(recent)}};
    return body.dump();
}

ServeMetrics::LastCompleted ServeMetrics::last_completed() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return last_completed_;
}

std::string ServeMetrics::render(std::uint32_t max_concurrency,
                                 const ninfer::RuntimeStats& live) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t in_flight  = active_.size();
    const std::uint64_t processing = std::min<std::uint64_t>(in_flight, max_concurrency);
    std::string out;
    out.reserve(704);
    append_counter(out, "llamacpp:prompt_tokens_total", live.computed_prefill_tokens);
    append_counter(out, "llamacpp:prompt_seconds_total", live.prefill_seconds_total);
    append_counter(out, "llamacpp:tokens_predicted_total", live.committed_decode_tokens);
    append_counter(out, "llamacpp:tokens_predicted_seconds_total", live.decode_seconds_total);
    append_counter(out, "llamacpp:requests_processing", processing);
    append_counter(out, "llamacpp:requests_deferred", in_flight - processing);
    append_counter(out, "ninfer:requests_total", requests_total_);
    append_counter(out, "ninfer:prefix_cache_hit_tokens_total", prefix_cache_hit_tokens_total_);
    append_counter(out, "ninfer:draft_tokens_total", speculative_draft_tokens_total_);
    append_counter(out, "ninfer:draft_accepted_tokens_total", speculative_accepted_tokens_total_);
    return out;
}

} // namespace ninfer::serve
