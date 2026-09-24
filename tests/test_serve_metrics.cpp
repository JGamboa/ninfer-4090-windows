#include "serve/monitor_page.h"
#include "serve/serve_metrics.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

namespace {

using ninfer::serve::GenerationMetrics;
using ninfer::serve::GenerationOutcome;
using ninfer::serve::ServeMetrics;

int check(bool ok, const char* label) {
    if (!ok) { std::printf("FAIL %s\n", label); }
    return ok ? 0 : 1;
}

std::map<std::string, double> parse(const std::string& body) {
    std::map<std::string, double> values;
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        const auto space = line.find(' ');
        if (space == std::string::npos) { continue; }
        values[line.substr(0, space)] = std::stod(line.substr(space + 1));
    }
    return values;
}

GenerationOutcome outcome(int prompt, std::uint32_t cached, int completion, double prefill_s,
                          double decode_s, std::uint64_t drafted, std::uint64_t accepted) {
    GenerationOutcome out;
    out.prompt_tokens                        = prompt;
    out.completion_tokens                    = completion;
    out.metrics.prefix_cache_hit_tokens      = cached;
    out.metrics.prefill_seconds              = prefill_s;
    out.metrics.decode_seconds               = decode_s;
    out.metrics.speculative_draft_tokens     = drafted;
    out.metrics.speculative_accepted_tokens  = accepted;
    return out;
}

} // namespace

int main() {
    int failures = 0;

    ServeMetrics metrics;
    // The four llamacpp counters flow straight from the Engine's live totals.
    ninfer::RuntimeStats live;
    const auto empty = parse(metrics.render(1, live));
    failures += check(empty.at("llamacpp:prompt_tokens_total") == 0.0, "starts at zero");
    const auto never = metrics.last_completed();
    failures += check(never.prompt_tokens == 0 && never.cached_tokens == 0,
                      "last completed starts at zero");
    failures += check(empty.at("ninfer:requests_total") == 0.0, "requests start at zero");
    failures += check(empty.at("llamacpp:requests_processing") == 0.0, "idle processing");
    failures += check(empty.at("llamacpp:requests_deferred") == 0.0, "idle deferred");

    // Two in-flight requests against one execution lane: FIFO order says the
    // older one processes and the newer one is deferred.
    metrics.begin_request(7, 500);
    metrics.begin_request(8, 900);
    const auto busy = parse(metrics.render(1, live));
    failures += check(busy.at("llamacpp:requests_processing") == 1.0, "one processing");
    failures += check(busy.at("llamacpp:requests_deferred") == 1.0, "one deferred");
    const auto active = metrics.active_snapshot();
    failures += check(active.size() == 2 && active[0].first == 7 && active[0].second == 500,
                      "snapshot FIFO order");
    metrics.end_request(7);
    metrics.end_request(7); // idempotent
    metrics.end_request(8);
    const auto drained = parse(metrics.render(1, live));
    failures += check(drained.at("llamacpp:requests_processing") == 0.0, "drained processing");
    failures += check(drained.at("llamacpp:requests_deferred") == 0.0, "drained deferred");

    // Cold request: whole prompt computed.
    metrics.record(outcome(1000, 0, 200, 0.5, 4.0, 300, 150));
    const auto cold = metrics.last_completed();
    failures += check(cold.prompt_tokens == 1000 && cold.cached_tokens == 0,
                      "last completed after cold request");
    // Warm request: 900 of 1200 prompt tokens served from the prefix cache -
    // only the 300 computed tokens may count toward the prompt counter.
    metrics.record(outcome(1200, 900, 100, 0.1, 2.0, 150, 75));
    const auto warm = metrics.last_completed();
    failures += check(warm.prompt_tokens == 1200 && warm.cached_tokens == 900,
                      "last completed after warm request");

    live.computed_prefill_tokens = 1300;
    live.prefill_seconds_total   = 0.6;
    live.committed_decode_tokens = 300;
    live.decode_seconds_total    = 6.0;
    const auto values = parse(metrics.render(1, live));
    failures += check(values.at("llamacpp:prompt_tokens_total") == 1300.0, "live prefill tokens");
    failures += check(values.at("llamacpp:prompt_seconds_total") == 0.6, "live prefill seconds");
    failures += check(values.at("llamacpp:tokens_predicted_total") == 300.0, "live decode tokens");
    failures += check(values.at("llamacpp:tokens_predicted_seconds_total") == 6.0,
                      "live decode seconds");
    failures += check(values.at("ninfer:requests_total") == 2.0, "request count");
    failures += check(values.at("ninfer:prefix_cache_hit_tokens_total") == 900.0, "cache hits");
    failures += check(values.at("ninfer:draft_tokens_total") == 450.0, "draft tokens");
    failures += check(values.at("ninfer:draft_accepted_tokens_total") == 225.0, "accepted tokens");

    // A cache hit reported larger than the prompt must clamp, not underflow.
    metrics.record(outcome(10, 50, 1, 0.0, 0.1, 0, 0));
    const auto residue = metrics.last_completed();
    failures += check(residue.prompt_tokens == 10 && residue.cached_tokens == 10,
                      "last completed cache clamped to prompt");

    // Monitor snapshot: the static context, live KV/scheduler gauges, Engine totals since the
    // attach baseline (the warmup generation excluded), one row per retained-conversation cell,
    // and the completed requests newest first.
    ServeMetrics::MonitorContext context{"bonsai-27b", 262144, 3, 262144, 4096, 2};
    ninfer::RuntimeStats baseline;
    baseline.computed_prefill_tokens = 60;
    baseline.prefill_seconds_total   = 0.1;
    baseline.committed_decode_tokens = 3;
    baseline.decode_seconds_total    = 0.02;
    live.device_main_kv_occupied_pages = 1400;
    live.running_requests              = 1;
    live.waiting_requests              = 2;
    std::vector<ninfer::SlotState> slots(6);
    slots[0].processing    = true;
    slots[0].prompt_tokens = 86266;
    slots[0].cached_tokens = 85133;
    slots[1].retained      = true;
    const auto monitor =
        nlohmann::json::parse(metrics.render_monitor(context, live, baseline, slots));
    failures += check(monitor.at("model") == "bonsai-27b" && monitor.at("lanes") == 3 &&
                          monitor.at("draft_window") == 2,
                      "monitor context");
    failures += check(monitor.at("kv").at("pages") == 4096 &&
                          monitor.at("kv").at("occupied_pages") == 1400,
                      "monitor kv occupancy");
    failures += check(monitor.at("scheduler").at("running") == 1 &&
                          monitor.at("scheduler").at("waiting") == 2,
                      "monitor scheduler gauges");
    const auto& totals = monitor.at("totals");
    failures += check(totals.at("requests") == 3 && totals.at("prefill_tokens") == 1240 &&
                          totals.at("decode_tokens") == 297 &&
                          std::abs(totals.at("decode_seconds").get<double>() - 5.98) < 1e-9,
                      "monitor totals exclude the warmup baseline");
    // Completed prompts 1000 + 1200 + 10; reuse clamped to each prompt: 0 + 900 + 10.
    failures += check(totals.at("prompt_tokens") == 2210 &&
                          totals.at("cached_prompt_tokens") == 910,
                      "monitor prompt reuse totals");
    const auto& slot_rows = monitor.at("slots");
    failures += check(slot_rows.size() == 6 && slot_rows[0].at("processing") == true &&
                          slot_rows[0].at("prompt_tokens") == 86266 &&
                          slot_rows[1].at("retained") == true,
                      "monitor slot rows");
    const auto& recent = monitor.at("recent");
    failures += check(recent.size() == 3 && recent[0].at("sequence") == 3 &&
                          recent[0].at("cached_tokens") == 10 && recent[2].at("sequence") == 1 &&
                          recent[1].at("drafted") == 150 && recent[1].at("accepted") == 75,
                      "monitor recent requests newest first");

    // The recent list keeps the latest kRecentRequests completions.
    for (int i = 0; i < 40; ++i) metrics.record(outcome(100 + i, 0, 10, 0.1, 0.2, 0, 0));
    const auto kept = metrics.recent_requests();
    failures += check(kept.size() == ServeMetrics::kRecentRequests && kept.front().sequence == 43 &&
                          kept.front().prompt_tokens == 139 &&
                          kept.back().sequence == 43 - ServeMetrics::kRecentRequests + 1,
                      "recent requests bounded");

    // The page compiled into the server is the source page, byte for byte.
    std::ifstream page_file(std::string(NINFER_SOURCE_DIR) + "/src/serve/monitor_page.html",
                            std::ios::binary);
    const std::string page_source{std::istreambuf_iterator<char>(page_file), {}};
    failures += check(!page_source.empty() && ninfer::serve::monitor_page() == page_source,
                      "embedded monitor page matches its source");

    std::printf("%s serve metrics\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
