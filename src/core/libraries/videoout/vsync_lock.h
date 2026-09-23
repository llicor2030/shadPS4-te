// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <functional>
#include <memory>

#include "common/types.h"

namespace Libraries::VideoOut {

// Keeps the emulated vblank in a fixed phase relative to the host display's vblank.
//
// The present thread paces the guest with its own timer (host CPU clock) while the display
// scans out on the GPU's pixel clock. With Mailbox/Immediate presentation nothing ties the two
// together, so the present time slowly drifts across the host vsync boundary; while it sits on
// the boundary, frames are alternately shown twice and dropped for a long stretch.
//
// VsyncLock watches the host vblank and trims the timer interval by a few ppm so that each
// present happens `margin` before the next host vblank. Only active on Windows, only for
// Mailbox/Immediate, and only when the host refresh rate is (a multiple of) the guest rate.
//
// Settings (config.json "GPU", overridable per game):
//   vsync_lock                 enable (default true)
//   vsync_lock_margin_ms       how long before the host vblank to present (1 - 50, default 6)
//   vsync_lock_log             log phase statistics every 10 seconds (default false)
//   vsync_drop_missed_vblanks  false restores upstream's catch-up bursts in the present timer
//                              (independent of the lock itself; default true)
class VsyncLock {
public:
    using NativeWindowGetter = std::function<void*()>;

    VsyncLock(std::chrono::nanoseconds guest_period, bool present_mode_allows_lock,
              NativeWindowGetter get_native_window);
    ~VsyncLock();

    VsyncLock(const VsyncLock&) = delete;
    VsyncLock& operator=(const VsyncLock&) = delete;

    // Interval the present thread should use for the iteration that has just started.
    std::chrono::nanoseconds NextInterval();

    bool IsEnabled() const {
        return enabled;
    }

    // Whether the present thread should drop missed intervals instead of catching up.
    bool DropMissedIntervals() const {
        return drop_missed;
    }

    // Host vblank estimate shared with the watcher thread (defined in vsync_lock.cpp).
    struct Shared;

private:
    std::chrono::nanoseconds nominal;
    std::shared_ptr<Shared> shared;

    bool enabled = false;
    bool drop_missed = true;
    bool log_stats = false;
    double margin_ns = 6'000'000.0;

    bool locked = false;
    double err_avg = 0.0;
    bool lockable_reported = true;
    u32 good_frames = 0;
    u32 bad_frames = 0;

    // Statistics (vsync_lock_log)
    u32 stat_frames = 0;
    double stat_err_sum = 0.0;
    double stat_err_abs_max = 0.0;
    std::chrono::steady_clock::time_point stat_begin{};
};

} // namespace Libraries::VideoOut
