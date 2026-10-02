// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <string_view>

#include "common/logging/log.h"
#include "common/types.h"

namespace VideoCore::TraceWindow {

// Texture trace windows (Debug.texture_trace_trigger / texture_trace_seconds).
// A window opens whenever the game's own output contains the trigger text and stays open for the
// configured number of seconds, so the same scene can be traced each time it comes back and the
// windows compared with each other. Without a trigger the trace is always on (window 1).

struct State {
    std::mutex mutex;
    std::string trigger;
    s64 length_ns = 0;
    std::atomic<bool> configured{false};
    std::atomic<bool> always_on{false};
    std::atomic<s64> end_ns{0};
    std::atomic<u32> index{0};
};

inline State& Get() {
    static State state;
    return state;
}

inline s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void Configure(std::string trigger, int seconds) {
    auto& state = Get();
    std::scoped_lock lk{state.mutex};
    state.trigger = std::move(trigger);
    state.length_ns = static_cast<s64>(seconds > 0 ? seconds : 1) * 1'000'000'000;
    state.index = state.trigger.empty() ? 1 : 0;
    state.end_ns = 0;
    state.always_on = state.trigger.empty();
    state.configured = true;
    if (!state.trigger.empty()) {
        LOG_INFO(Render_Vulkan, "TexTrace windows: \"{}\" opens a {} s window",
                 state.trigger, seconds > 0 ? seconds : 1);
    }
}

/// Called with every line the game writes to its stdout/stderr.
inline void OnGuestOutput(std::string_view line) {
    auto& state = Get();
    if (!state.configured.load(std::memory_order_relaxed)) {
        return;
    }
    std::scoped_lock lk{state.mutex};
    if (state.trigger.empty() || line.find(state.trigger) == std::string_view::npos) {
        return;
    }
    const u32 index = state.index.fetch_add(1) + 1;
    state.end_ns = NowNs() + state.length_ns;
    LOG_INFO(Render_Vulkan, "TexTrace window {} open", index);
}

/// Number of the open window, or 0 when no window is open.
inline u32 Current() {
    auto& state = Get();
    if (!state.configured.load(std::memory_order_relaxed)) {
        return 0;
    }
    if (state.always_on.load(std::memory_order_relaxed)) {
        return 1;
    }
    if (NowNs() >= state.end_ns.load(std::memory_order_relaxed)) {
        return 0;
    }
    return state.index.load(std::memory_order_relaxed);
}

} // namespace VideoCore::TraceWindow
