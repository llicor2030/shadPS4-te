// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

#include "common/types.h"
#include "core/memory.h"

namespace VideoCore::HostWriteTrace {

// Texture trace diagnostics: writes into guest memory that the emulator makes itself (GPU data
// written back to memory, fences, DMA done on the CPU, PM4 memory writes). Writes through the
// backing memory bypass page protection, so the page tracking never sees them.

using Observer = void (*)(void* context, std::string_view what, VAddr addr, VAddr ctx_addr,
                          u64 ctx_size, std::span<const u8> before, std::span<const u8> after);

struct State {
    std::atomic<Observer> observer{nullptr};
    std::atomic<void*> context{nullptr};
};

inline State& Get() {
    static State state;
    return state;
}

inline void Register(Observer observer, void* context) {
    auto& state = Get();
    state.context = context;
    state.observer = observer;
}

inline void Unregister(void* context) {
    auto& state = Get();
    if (state.context == context) {
        state.observer = nullptr;
        state.context = nullptr;
    }
}

struct LabelInfo {
    std::string_view what;
    VAddr addr = 0;
    u64 size = 0;
};

/// Label of the labelled scopes opened on this thread (see Label).
inline thread_local LabelInfo current_label{};

/// Names the labelled scopes (backing-memory writes) opened on this thread while it lives.
class Label {
public:
    explicit Label(std::string_view what, VAddr addr = 0, u64 size = 0) : previous{current_label} {
        current_label = {what, addr, size};
    }
    ~Label() {
        current_label = previous;
    }
    Label(const Label&) = delete;
    Label& operator=(const Label&) = delete;

private:
    LabelInfo previous;
};

/// Copies the bytes a write is about to change and reports them with the new bytes when the
/// scope ends, if they differ. Reads go through the backing memory and never fault.
class Scope {
public:
    Scope(std::string_view what_, VAddr addr_, u64 size_, bool labelled_ = false) {
        if (size_ == 0 || !Get().observer.load(std::memory_order_acquire)) {
            return;
        }
        what = what_;
        addr = addr_;
        labelled = labelled_;
        before.resize(size_);
        Core::Memory::Instance()->CopySparseMemory(addr, before.data(), before.size());
        active = true;
    }
    ~Scope() {
        if (!active) {
            return;
        }
        auto& state = Get();
        const Observer observer = state.observer.load(std::memory_order_acquire);
        if (!observer) {
            return;
        }
        std::vector<u8> after(before.size());
        Core::Memory::Instance()->CopySparseMemory(addr, after.data(), after.size());
        if (std::memcmp(before.data(), after.data(), before.size()) == 0) {
            return;
        }
        const LabelInfo& label = current_label;
        if (labelled && !label.what.empty()) {
            observer(state.context.load(), label.what, addr, label.addr, label.size, before, after);
        } else {
            observer(state.context.load(), what, addr, 0, 0, before, after);
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    std::string_view what;
    VAddr addr = 0;
    bool labelled = false;
    bool active = false;
    std::vector<u8> before;
};

} // namespace VideoCore::HostWriteTrace
