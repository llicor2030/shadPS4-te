// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

#include "common/logging/log.h"
#include "common/thread.h"
#include "core/libraries/videoout/vsync_lock.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Libraries::VideoOut {

namespace {

using Clock = std::chrono::steady_clock;

constexpr double NsPerMs = 1'000'000.0;

double ToNs(Clock::time_point t) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count());
}

std::string ReadEnv(const char* name) {
#ifdef _WIN32
    char buffer[64]{};
    const DWORD len = GetEnvironmentVariableA(name, buffer, sizeof(buffer));
    if (len == 0 || len >= sizeof(buffer)) {
        return {};
    }
    return std::string(buffer, len);
#else
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string{};
#endif
}

bool EnvFlag(const char* name, bool default_value) {
    const std::string value = ReadEnv(name);
    if (value.empty()) {
        return default_value;
    }
    const char c = value[0];
    return !(c == '0' || c == 'n' || c == 'N' || c == 'f' || c == 'F');
}

double EnvDouble(const char* name, double default_value) {
    const std::string value = ReadEnv(name);
    if (value.empty()) {
        return default_value;
    }
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    return end == value.c_str() ? default_value : parsed;
}

} // namespace

// State shared between the present thread and the (detached) vblank watcher thread.
struct VsyncLock::Shared {
    std::atomic<bool> stop{false};
    VsyncLock::NativeWindowGetter get_native_window;

    std::mutex mutex;
    // Host vblank estimate, all in steady_clock nanoseconds.
    double phase = 0.0;  // estimated time of the most recent host vblank
    double period = 0.0; // estimated host refresh period, 0 while unknown
    double last_sample = 0.0;
    u32 samples = 0;

    // Estimator internals (watcher thread only).
    double init_first = 0.0;
    double init_prev = 0.0;
    u32 init_count = 0;
    u32 outliers = 0;

    void Reset() {
        std::scoped_lock lk{mutex};
        phase = period = last_sample = 0.0;
        samples = 0;
        init_count = 0;
        outliers = 0;
    }

    void OnVblank(double t) {
        std::scoped_lock lk{mutex};
        last_sample = t;

        if (period <= 0.0) {
            // Initial period: average of 30 consistent intervals.
            if (init_count == 0) {
                init_first = init_prev = t;
                init_count = 1;
                return;
            }
            const double dt = t - init_prev;
            init_prev = t;
            if (dt < 2.0 * NsPerMs || dt > 110.0 * NsPerMs) {
                init_count = 0; // missed or spurious vblank, start over
                return;
            }
            if (init_count >= 2) {
                const double mean = (init_prev - dt - init_first) / (init_count - 1);
                if (std::abs(dt - mean) > mean * 0.1) {
                    init_count = 0;
                    return;
                }
            }
            if (++init_count > 30) {
                period = (t - init_first) / (init_count - 1);
                phase = t;
                samples = 0;
                outliers = 0;
            }
            return;
        }

        const double since = t - phase;
        if (since < period * 0.5) {
            return; // duplicate wake-up
        }
        const double n = std::round(since / period);
        if (n > 120.0) {
            period = 0.0; // long gap (display off, mode change): re-measure
            init_count = 0;
            return;
        }
        const double predicted = phase + n * period;
        const double err = t - predicted;
        if (std::abs(err) > period * 0.25) {
            if (++outliers > 8) {
                period = 0.0;
                init_count = 0;
            }
            return;
        }
        outliers = 0;
        // Phase follows quickly, period very slowly (it only differs by ppm-level drift).
        phase = predicted + 0.1 * err;
        period += 0.002 * err / n;
        ++samples;
    }

    // Returns false if there is no usable estimate.
    bool Get(double now, double& out_phase, double& out_period) {
        std::scoped_lock lk{mutex};
        if (period <= 0.0 || samples < 60 || now - last_sample > 250.0 * NsPerMs) {
            return false;
        }
        out_phase = phase;
        out_period = period;
        return true;
    }
};

#ifdef _WIN32

namespace {

// Minimal D3DKMT declarations (gdi32.dll exports); avoids depending on d3dkmthk.h.
namespace kmt {
using HANDLE_T = UINT;
struct OPENADAPTERFROMHDC {
    HDC hDc;
    HANDLE_T hAdapter;
    LUID AdapterLuid;
    UINT VidPnSourceId;
};
struct WAITFORVERTICALBLANKEVENT {
    HANDLE_T hAdapter;
    HANDLE_T hDevice;
    UINT VidPnSourceId;
};
struct CLOSEADAPTER {
    HANDLE_T hAdapter;
};
using OpenAdapterFromHdcFn = LONG(APIENTRY*)(OPENADAPTERFROMHDC*);
using WaitForVerticalBlankEventFn = LONG(APIENTRY*)(const WAITFORVERTICALBLANKEVENT*);
using CloseAdapterFn = LONG(APIENTRY*)(const CLOSEADAPTER*);
} // namespace kmt

template <typename Fn>
Fn GetGdiProc(HMODULE gdi, const char* name) {
    if (gdi == nullptr) {
        return nullptr;
    }
    // Go through a generic function pointer to silence -Wcast-function-type.
    return reinterpret_cast<Fn>(reinterpret_cast<void (*)()>(GetProcAddress(gdi, name)));
}

void VblankWatcher(std::shared_ptr<VsyncLock::Shared> shared) {
    Common::SetCurrentThreadName("shadPS4:VblankWatcher");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);

    HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
    if (gdi == nullptr) {
        gdi = LoadLibraryW(L"gdi32.dll");
    }
    const auto open_adapter =
        GetGdiProc<kmt::OpenAdapterFromHdcFn>(gdi, "D3DKMTOpenAdapterFromHdc");
    const auto wait_vblank =
        GetGdiProc<kmt::WaitForVerticalBlankEventFn>(gdi, "D3DKMTWaitForVerticalBlankEvent");
    const auto close_adapter = GetGdiProc<kmt::CloseAdapterFn>(gdi, "D3DKMTCloseAdapter");
    if (!open_adapter || !wait_vblank || !close_adapter) {
        LOG_WARNING(Lib_VideoOut, "VsyncLock: D3DKMT vblank API unavailable, lock disabled");
        return;
    }

    HWND hwnd = nullptr;
    HMONITOR monitor = nullptr;
    bool have_adapter = false;
    kmt::WAITFORVERTICALBLANKEVENT wait_args{};
    auto last_check = Clock::now() - std::chrono::hours(1);

    const auto close_current = [&] {
        if (have_adapter) {
            const kmt::CLOSEADAPTER close_args{wait_args.hAdapter};
            close_adapter(&close_args);
            have_adapter = false;
        }
    };

    while (!shared->stop.load(std::memory_order_relaxed)) {
        const auto now = Clock::now();
        if (!have_adapter || now - last_check >= std::chrono::seconds(1)) {
            last_check = now;
            if (hwnd == nullptr && shared->get_native_window) {
                hwnd = static_cast<HWND>(shared->get_native_window());
            }
            HMONITOR current = nullptr;
            if (hwnd != nullptr && IsWindow(hwnd)) {
                current = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
            } else {
                current = MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
            }
            if (current != monitor || !have_adapter) {
                close_current();
                shared->Reset();
                monitor = current;

                MONITORINFOEXW info{};
                info.cbSize = sizeof(info);
                HDC hdc = nullptr;
                if (GetMonitorInfoW(current, &info)) {
                    hdc = CreateDCW(nullptr, info.szDevice, nullptr, nullptr);
                }
                if (hdc != nullptr) {
                    kmt::OPENADAPTERFROMHDC open_args{};
                    open_args.hDc = hdc;
                    if (open_adapter(&open_args) == 0) {
                        wait_args.hAdapter = open_args.hAdapter;
                        wait_args.hDevice = 0;
                        wait_args.VidPnSourceId = open_args.VidPnSourceId;
                        have_adapter = true;
                    }
                    DeleteDC(hdc);
                }
                if (!have_adapter) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
            }
        }

        if (wait_vblank(&wait_args) != 0) {
            close_current();
            shared->Reset();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        shared->OnVblank(ToNs(Clock::now()));
    }
    close_current();
}

} // namespace

#endif // _WIN32

VsyncLock::VsyncLock(std::chrono::nanoseconds guest_period,
                     [[maybe_unused]] bool present_mode_allows_lock,
                     NativeWindowGetter get_native_window)
    : nominal(guest_period), shared(std::make_shared<Shared>()) {
    shared->get_native_window = std::move(get_native_window);
    drop_missed = EnvFlag("SHADPS4_VSYNC_DROP_MISSED", true);

#ifdef _WIN32
    enabled = present_mode_allows_lock && EnvFlag("SHADPS4_VSYNC_LOCK", true);
#else
    enabled = false;
#endif
    if (!enabled) {
        return;
    }

    margin_ns = std::clamp(EnvDouble("SHADPS4_VSYNC_LOCK_MARGIN_MS", 6.0), 0.5, 50.0) * NsPerMs;
    log_stats = EnvFlag("SHADPS4_VSYNC_LOCK_LOG", false);
    stat_begin = Clock::now();

    LOG_INFO(Lib_VideoOut, "VsyncLock: enabled, margin {:.2f} ms", margin_ns / NsPerMs);

#ifdef _WIN32
    // Detached: D3DKMTWaitForVerticalBlankEvent cannot be interrupted, so never join it.
    std::thread(VblankWatcher, shared).detach();
#endif
}

VsyncLock::~VsyncLock() {
    shared->stop.store(true, std::memory_order_relaxed);
}

std::chrono::nanoseconds VsyncLock::NextInterval() {
    if (!enabled) {
        return nominal;
    }

    const auto now_tp = Clock::now();
    const double now = ToNs(now_tp);
    const double guest = static_cast<double>(nominal.count());

    const auto unlock = [&](const char* reason) {
        if (locked) {
            LOG_INFO(Lib_VideoOut, "VsyncLock: lost ({})", reason);
        }
        locked = false;
        good_frames = bad_frames = 0;
        err_avg = 0.0;
        return nominal;
    };

    double phase = 0.0;
    double host = 0.0;
    if (!shared->Get(now, phase, host)) {
        return unlock("no host vblank");
    }

    // Guest period must be N host periods (N >= 1) within 0.5%.
    const double n = std::max(1.0, std::round(guest / host));
    const double effective = n * host;
    const bool lockable = std::abs(effective - guest) <= guest * 0.005;
    if (lockable != lockable_reported) {
        lockable_reported = lockable;
        if (!lockable) {
            LOG_INFO(Lib_VideoOut,
                     "VsyncLock: host {:.3f} Hz is not a multiple of guest {:.3f} Hz, not locking",
                     1e9 / host, 1e9 / guest);
        }
    }
    if (!lockable) {
        return unlock("refresh rate mismatch");
    }

    // Where are we inside the host refresh cycle, and where do we want to be?
    double since = std::fmod(now - phase, host);
    if (since < 0.0) {
        since += host;
    }
    double target = host - std::fmod(margin_ns, host);
    double err = since - target; // > 0: this iteration started too late
    if (err > host * 0.5) {
        err -= host;
    } else if (err <= -host * 0.5) {
        err += host;
    }

    // Proportional correction, slew-limited: 0.5% while acquiring, 500 ppm once locked.
    const double max_step = effective * (locked ? 0.0005 : 0.005);
    const double correction = std::clamp(err * 0.1, -max_step, max_step);

    // Lock state is judged on the smoothed error so that sleep jitter does not flap it.
    err_avg += (err - err_avg) * 0.05;
    const double abs_avg = std::abs(err_avg);
    const double abs_err = std::abs(err);
    if (!locked) {
        good_frames = abs_avg < 0.5 * NsPerMs ? good_frames + 1 : 0;
        if (good_frames >= 60) {
            locked = true;
            bad_frames = 0;
            LOG_INFO(Lib_VideoOut,
                     "VsyncLock: locked to host {:.4f} Hz (x{:.0f}), margin {:.2f} ms", 1e9 / host,
                     n, margin_ns / NsPerMs);
        }
    } else {
        bad_frames = abs_avg > 2.0 * NsPerMs ? bad_frames + 1 : 0;
        if (bad_frames >= 30) {
            // Stay active, but allow the faster acquisition slew again.
            LOG_INFO(Lib_VideoOut, "VsyncLock: phase error {:.2f} ms, re-acquiring",
                     err_avg / NsPerMs);
            locked = false;
            good_frames = bad_frames = 0;
        }
    }

    if (log_stats) {
        ++stat_frames;
        stat_err_sum += err;
        stat_err_abs_max = std::max(stat_err_abs_max, abs_err);
        if (now_tp - stat_begin >= std::chrono::seconds(10)) {
            LOG_INFO(Lib_VideoOut,
                     "VsyncLock: host {:.5f} Hz, {}, phase err mean {:+.3f} ms, max {:.3f} ms",
                     1e9 / host, locked ? "locked" : "acquiring",
                     stat_err_sum / stat_frames / NsPerMs, stat_err_abs_max / NsPerMs);
            stat_frames = 0;
            stat_err_sum = 0.0;
            stat_err_abs_max = 0.0;
            stat_begin = now_tp;
        }
    }

    const auto interval = static_cast<s64>(std::llround(effective - correction));
    return std::chrono::nanoseconds(interval);
}

} // namespace Libraries::VideoOut
