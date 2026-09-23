// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"

namespace Libraries::AudioOut {

// OpenAL Soft reads its configuration once per process, on the first OpenAL call: alsoft.ini
// in %AppData% and next to the executable, then the file named by ALSOFT_CONF. WASAPI exclusive
// mode and the update period have no API of their own, so they are handed over through a file
// generated from config.json ("Audio", overridable per game). Must run before any OpenAL call.
//
//   openal_exclusive_mode   WASAPI exclusive mode (Windows; other apps cannot use the device)
//   openal_period_frames    mixer update size in frames (64 - 8192, 0 = OpenAL Soft default)
//
// Nothing is written or set while both are at their defaults, and an ALSOFT_CONF the user set
// themselves is left alone.
inline void ApplyOpenALConfig() {
    const bool exclusive = EmulatorSettings.IsOpenALExclusiveMode();
    const u32 period = EmulatorSettings.GetOpenALPeriodFrames();
    if (!exclusive && period == 0) {
        return;
    }
    if (std::getenv("ALSOFT_CONF") != nullptr) {
        LOG_WARNING(Lib_AudioOut, "ALSOFT_CONF is already set; OpenAL device options not applied");
        return;
    }
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "alsoft_shadps4.ini";
    {
        std::ofstream file{path, std::ios::out | std::ios::trunc};
        if (!file) {
            LOG_ERROR(Lib_AudioOut, "Failed to write OpenAL config");
            return;
        }
        if (period != 0) {
            file << "[general]\nperiod_size = " << std::clamp(period, 64u, 8192u) << "\n";
        }
        if (exclusive) {
            file << "[wasapi]\nexclusive-mode = true\n";
        }
    }
#ifdef _WIN32
    _wputenv_s(L"ALSOFT_CONF", path.wstring().c_str());
#else
    setenv("ALSOFT_CONF", path.string().c_str(), 1);
#endif
    LOG_INFO(Lib_AudioOut, "OpenAL device options: exclusive mode {}, period {} frames", exclusive,
             period);
}

} // namespace Libraries::AudioOut
