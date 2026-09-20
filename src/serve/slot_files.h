#pragma once

// Filename policy for /slots session persistence. Clients name snapshot files; the server
// confines them to the --slot-save-path directory, so names are a single conservative path
// component rather than a path.

#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

inline constexpr std::size_t kSlotFilenameMaxBytes = 128;

// Returns the validated filename, or nullopt when the name is empty, too long, dot-leading, or
// holds anything outside [A-Za-z0-9._-]. Rejecting a leading dot removes "..", ".", and hidden
// files in one rule; the allowlist keeps every separator out.
[[nodiscard]] inline std::optional<std::string> sanitize_slot_filename(std::string_view name) {
    if (name.empty() || name.size() > kSlotFilenameMaxBytes || name.front() == '.') {
        return std::nullopt;
    }
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok) { return std::nullopt; }
    }
    return std::string(name);
}

} // namespace ninfer::serve
