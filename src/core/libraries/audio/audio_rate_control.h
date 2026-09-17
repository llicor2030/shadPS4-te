// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include "common/logging/log.h"
#include "common/types.h"

namespace Libraries::AudioOut {

// Dynamic rate control for a queued audio output.
//
// The output thread hands the backend one guest buffer per buffer period of the host CPU clock,
// while the device consumes at its own crystal (for HDMI audio: the GPU's pixel clock). The two
// differ by tens of ppm, so the backend queue slowly runs dry (dropouts) or fills up (the queue
// gets dropped). A title that mixes a fixed amount per vblank also produces 0.1% less than
// real time while the guest vblank follows a 59.94 Hz display.
//
// The playback ratio is trimmed by at most +-0.5% (under 9 cents of pitch) so that the lowest
// queue level seen just before new data arrives stays at the smallest value that survives one
// guest period of device pulls plus scheduling jitter. Controlling the low point rather than the
// average keeps it safe whether guest buffers are larger or smaller than device pulls. The trim
// is updated four times a second in steps of a few ppm.
//
// Environment variables:
//   SHADPS4_AUDIO_DRC=0              disable (upstream behaviour)
//   SHADPS4_AUDIO_DRC_MARGIN_MS=2    jitter allowance added to the queue target (0 - 100)
//   SHADPS4_AUDIO_DRC_LOG=1          log ratio, queue level and dropouts every 10 seconds
class AudioRateControl {
public:
    static constexpr double MaxDeviation = 0.005;

    // `guest_frames`: frames per guest buffer. `device_frames`: frames the device takes per
    // pull. Returns the queue level to refill to after running dry, in frames (0 when disabled).
    u32 Configure(u32 guest_frames, u32 device_frames, u32 sample_rate) {
        enabled = EnvFlag("SHADPS4_AUDIO_DRC", true);
        log_stats = EnvFlag("SHADPS4_AUDIO_DRC_LOG", false);
        if (!enabled) {
            target = 0;
            return 0;
        }
        const double margin_ms =
            std::clamp(EnvDouble("SHADPS4_AUDIO_DRC_MARGIN_MS", 2.0), 0.0, 100.0);
        const u32 margin = static_cast<u32>(std::lround(margin_ms * sample_rate / 1000.0));
        const u32 pull = std::max(device_frames, 1u);
        // Worst case between two guest buffers: the pulls that fit into one guest period plus
        // the jitter, plus the one already due. Whatever the guest buffer does not cover has to
        // be waiting in the queue already.
        const u32 pulls = (guest_frames + margin) / pull + 1;
        const s64 needed = static_cast<s64>(pulls) * pull - guest_frames + margin;
        target = static_cast<u32>(std::max<s64>(needed, std::max<u32>(margin, 1)));
        scale = static_cast<double>(std::max(guest_frames, pull));
        window_min = -1.0;
        return target;
    }

    bool Enabled() const {
        return enabled;
    }

    u32 Target() const {
        return target;
    }

    // `queued_frames`: frames still waiting in the backend, sampled right before new data is
    // queued (0 after running dry). `now_us`: a monotonic timestamp. Returns the playback ratio.
    double Update(double queued_frames, u64 now_us) {
        if (!enabled) {
            return 1.0;
        }
        if (window_min < 0.0 || queued_frames < window_min) {
            window_min = queued_frames;
        }
        if (window_begin_us == 0) {
            window_begin_us = now_us;
        }
        if (now_us - window_begin_us >= WindowUs) {
            // Low point above target: the device can take a little more; below: a little less.
            const double error = std::clamp((window_min - target) / scale, -1.0, 1.0);
            integral = std::clamp(integral + Ki * error, -MaxDeviation, MaxDeviation);
            ratio = 1.0 + std::clamp(Kp * error + integral, -MaxDeviation, MaxDeviation);
            last_min = window_min;
            window_min = -1.0;
            window_begin_us = now_us;
        }

        if (log_stats) {
            if (log_begin_us == 0) {
                log_begin_us = now_us;
            } else if (now_us - log_begin_us >= 10'000'000) {
                LOG_INFO(Lib_AudioOut,
                         "Audio DRC: ratio {:.6f} ({:+.0f} ppm), queue low {:.0f}/{} frames, "
                         "dropouts {}, queue drops {}",
                         ratio, (ratio - 1.0) * 1e6, last_min, target, underruns, clears);
                log_begin_us = now_us;
                underruns = 0;
                clears = 0;
            }
        }
        return ratio;
    }

    void NoteUnderrun() {
        ++underruns;
    }

    void NoteClear() {
        ++clears;
    }

    // Avoids re-applying the ratio for sub-ppm changes.
    static bool Differs(double a, double b) {
        return std::abs(a - b) > 2e-6;
    }

private:
    static constexpr u64 WindowUs = 250'000;
    static constexpr double Kp = 0.005;
    static constexpr double Ki = 0.0002;

    static std::string ReadEnv(const char* name) {
#ifdef _WIN32
        char* value = nullptr;
        size_t length = 0;
        if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
            return {};
        }
        std::string result(value);
        std::free(value);
        return result;
#else
        const char* value = std::getenv(name);
        return value ? std::string(value) : std::string{};
#endif
    }

    static bool EnvFlag(const char* name, bool default_value) {
        const std::string value = ReadEnv(name);
        if (value.empty()) {
            return default_value;
        }
        const char c = value[0];
        return !(c == '0' || c == 'n' || c == 'N' || c == 'f' || c == 'F');
    }

    static double EnvDouble(const char* name, double default_value) {
        const std::string value = ReadEnv(name);
        if (value.empty()) {
            return default_value;
        }
        char* end = nullptr;
        const double parsed = std::strtod(value.c_str(), &end);
        return end == value.c_str() ? default_value : parsed;
    }

    bool enabled = false;
    bool log_stats = false;
    u32 target = 0;
    double scale = 1.0;
    double ratio = 1.0;
    double integral = 0.0;
    double window_min = -1.0;
    double last_min = 0.0;
    u64 window_begin_us = 0;

    u64 log_begin_us = 0;
    u32 underruns = 0;
    u32 clears = 0;
};

} // namespace Libraries::AudioOut
