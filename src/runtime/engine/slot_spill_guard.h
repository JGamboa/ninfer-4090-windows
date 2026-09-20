#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace ninfer {

// Fork-local. Keeps an auto-save spill from rolling a slot file back to an older, shallower
// state of the same session. There is no per-session identity at the catalog layer, so a
// session's live continuation and a stale earlier copy of it (left behind in another cell by
// a retaining restore) look identical to the engine; whichever is evicted spills to the file
// it is bound to. On 2026-09-04 a two-day-old 31,505-token copy spilled over a 78,020-token
// file and pi resumed the rolled-back session (D3).
//
// The guard is a per-path high-water mark in process memory. Every explicit save and every
// restore is authoritative and SETS the mark (a client may legitimately save a rewound,
// shallower session). A spill may proceed only if it carries at least as many tokens as the
// mark; a shallower spill is refused and reported so the skip is visible. Equal depth is
// allowed: a restored-then-evicted session spills itself back unchanged, which is harmless.
// Every bound path was bound by a save or restore in this process, so the mark is always
// primed before any spill can target the path.
class SlotSpillGuard {
public:
    void note_authoritative(const std::string& path, std::uint32_t tokens) {
        std::scoped_lock lock(mutex_);
        depth_[path] = tokens;
    }

    // nullopt: the spill may be written. Otherwise the deeper token count on record, which the
    // spill would have overwritten.
    [[nodiscard]] std::optional<std::uint32_t> blocks(const std::string& path,
                                                      std::uint32_t tokens) const {
        std::scoped_lock lock(mutex_);
        const auto it = depth_.find(path);
        if (it == depth_.end() || tokens >= it->second) { return std::nullopt; }
        return it->second;
    }

    void note_spilled(const std::string& path, std::uint32_t tokens) {
        std::scoped_lock lock(mutex_);
        std::uint32_t& depth = depth_[path];
        if (tokens > depth) { depth = tokens; }
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::uint32_t> depth_;
};

} // namespace ninfer
