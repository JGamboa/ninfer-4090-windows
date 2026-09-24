#pragma once

#include <string_view>

namespace ninfer::serve {

// The self-contained HTML page served at GET /monitor. It polls GET /monitor/stats once per
// second and differences the cumulative counters into rates; it loads nothing else.
[[nodiscard]] std::string_view monitor_page() noexcept;

} // namespace ninfer::serve
