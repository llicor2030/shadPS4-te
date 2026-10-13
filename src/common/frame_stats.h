// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/thread.h"
#include "common/types.h"
#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#endif

// Diagnostics: once a second, logs the guest frame rate with the time the emulator spent in
// operations that can stall a frame. Enabled with the frame_stats setting.
namespace Common::FrameStats {

enum class Kind : std::size_t {
    Pipeline,       ///< Compiling a new graphics or compute pipeline (GPU thread).
    EnsureResident, ///< Allocating and binding buffer memory on first use (GPU thread).
    Upload,         ///< Refreshing a texture from guest memory (GPU thread).
    Readback,       ///< Reading GPU-written memory back for the CPU (caller waits).
    ReadbackGpu,    ///< The part of a readback run on the GPU thread (copy and full GPU wait).
    Draw,           ///< Draw and indirect draw commands (GPU thread).
    Dispatch,       ///< Direct and indirect dispatch commands (GPU thread).
    Submit,         ///< Submitting a command buffer to the host GPU.
    GpuThreadWait,  ///< Waiting for the host GPU on the GPU thread.
    OtherWait,      ///< Waiting for the host GPU on any other thread.
    DrawSkipped,    ///< Draws dropped by FilterDraw.
    DrawPipeline,   ///< Draw phase: graphics pipeline lookup (and compilation).
    DrawResources,  ///< Draw phase: render state and shader resources (textures, buffers).
    DrawBuffers,    ///< Draw phase: render targets, vertex and index buffers, barriers.
    DrawSetup,      ///< Draw phase: descriptors, dynamic state, render pass.
    DrawRecord,     ///< Draw phase: recording the draw and resetting the bindings.
    TargetLookup,   ///< PrepareRenderState: finding the color and depth target images.
    BufferBinds,    ///< BindBuffers: shader buffers of one stage.
    TextureBinds,   ///< BindTextures: shader images and samplers of one stage.
    VertexBinds,    ///< BindVertexBuffers.
    RenderPass,     ///< Rasterizer::BeginRendering: render target views and render pass state.
    ObtainBuffer,   ///< BufferCache::ObtainBuffer: every call, from any caller (sampled time).
    ObtainCheck,    ///< ObtainBuffer: choosing the path, including IsRegionGpuModified (sampled).
    ObtainMap,      ///< ObtainBuffer stream path: StreamBuffer::Map (sampled).
    ObtainCopy,     ///< ObtainBuffer stream path: CopySparseMemory (sampled).
    ObtainCommit,   ///< ObtainBuffer stream path: StreamBuffer::Commit (sampled).
    ObtainCache,    ///< ObtainBuffer cache path: arena, residency and synchronization (sampled).
    GpuIdleWait,    ///< A guest submission waiting in WaitGpuIdle (after sceGnmSubmitDone).
    ReadbackFinish, ///< Scheduler::Finish in BufferCache::DownloadMemory (host GPU wait).
    VertexSampled,  ///< BindVertexBuffers: every call, with its phases timed (sampled).
    VertexInputs,   ///< BindVertexBuffers phase: vertex input state, setVertexInputEXT (sampled).
    VertexRanges,   ///< BindVertexBuffers phase: building and merging the guest ranges (sampled).
    VertexObtain,   ///< BindVertexBuffers phase: obtaining the buffer of one range (sampled).
    VertexAccess,   ///< BindVertexBuffers phase: IsBufferAccessed for one range (sampled).
    VertexHostBind, ///< BindVertexBuffers phase: host buffer lists and bindVertexBuffers (sampled).
    IndexBinds,     ///< BindIndexBuffer.
    FileRead,       ///< Reading a guest file (ReadFile in the kernel library), any thread.
    FileInvalidate, ///< The part of a file read that invalidates the destination's GPU caches.
    Count,
};

struct Counter {
    std::atomic<u64> count{0};
    std::atomic<u64> ns{0};
};

inline std::atomic_bool enabled{false};
inline std::array<Counter, static_cast<std::size_t>(Kind::Count)> counters{};
inline std::atomic<u64> flips{0};
inline std::atomic<u64> frame_index{0}; ///< Flips since start, never reset.
inline std::atomic<u64> gpu_idle_ns{0};
inline std::atomic<u64> access_violations{0};
inline std::atomic<u64> flip_pending_queries{0};
inline std::atomic<u64> occlusion_query_dumps{0};
inline std::atomic<u64> file_read_bytes{0};

/// Set on the GPU command thread, to tell its waits for the host GPU from those of other threads.
inline thread_local bool on_gpu_thread = false;

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Returns a start time, or 0 when the statistics are disabled.
inline u64 Begin() {
    return enabled.load(std::memory_order_relaxed) ? NowNs() : 0;
}

/// Measures one operation of the given kind for the lifetime of the object.
class Scope {
public:
    explicit Scope(Kind kind_) : kind{kind_}, start{Begin()} {}
    ~Scope() {
        if (start != 0) {
            auto& counter = counters[static_cast<std::size_t>(kind)];
            counter.count.fetch_add(1, std::memory_order_relaxed);
            counter.ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Kind kind;
    u64 start;
};

/// Measures a wait for the host GPU, attributed to the GPU thread or to the other threads.
class WaitScope : public Scope {
public:
    WaitScope() : Scope{on_gpu_thread ? Kind::GpuThreadWait : Kind::OtherWait} {}
};

/// Picks one call in SAMPLE_RATE at random, so that frequent short operations can be timed
/// without reading the clock on every call.
inline constexpr u64 SAMPLE_RATE = 64;

inline bool Sample() {
    thread_local u64 state = 0x9e3779b97f4a7c15ULL;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return (state >> 32) % SAMPLE_RATE == 0;
}

/// Waits until the stores issued so far are complete. A sampled phase starts and ends with
/// this, so that stores which complete late (write-combined stores to mapped GPU memory) are
/// counted in the phase that issued them.
inline void DrainStores() {
#if defined(_M_X64) || defined(__x86_64__)
    _mm_mfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

/// Counts every call of an operation, and for one call in SAMPLE_RATE measures the whole call
/// and its phases (each Mark adds the time since the previous mark). Sampled times and phase
/// counts are multiplied by SAMPLE_RATE, so they are estimates. Sampled calls also wait for
/// their stores at every mark (DrainStores), which makes them slower than the other calls.
class SampledPhases {
public:
    explicit SampledPhases(Kind total_) : total{total_} {
        if (!enabled.load(std::memory_order_relaxed)) {
            return;
        }
        counters[static_cast<std::size_t>(total)].count.fetch_add(1, std::memory_order_relaxed);
        if (Sample()) {
            DrainStores();
            start = NowNs();
            last = start;
        }
    }
    ~SampledPhases() {
        if (start != 0) {
            counters[static_cast<std::size_t>(total)].ns.fetch_add((NowNs() - start) * SAMPLE_RATE,
                                                                   std::memory_order_relaxed);
        }
    }
    SampledPhases(const SampledPhases&) = delete;
    SampledPhases& operator=(const SampledPhases&) = delete;

    void Mark(Kind phase) {
        if (start == 0) {
            return;
        }
        DrainStores();
        const u64 now = NowNs();
        auto& counter = counters[static_cast<std::size_t>(phase)];
        counter.count.fetch_add(SAMPLE_RATE, std::memory_order_relaxed);
        counter.ns.fetch_add((now - last) * SAMPLE_RATE, std::memory_order_relaxed);
        last = now;
    }

private:
    Kind total;
    u64 start = 0;
    u64 last = 0;
};

/// Splits one operation into consecutive phases: each Mark adds the time since the previous mark.
class Phases {
public:
    Phases() : last{Begin()} {}
    void Mark(Kind kind) {
        if (last == 0) {
            return;
        }
        const u64 now = NowNs();
        auto& counter = counters[static_cast<std::size_t>(kind)];
        counter.count.fetch_add(1, std::memory_order_relaxed);
        counter.ns.fetch_add(now - last, std::memory_order_relaxed);
        last = now;
    }

private:
    u64 last;
};

/// Draws whose state matches the previous draw, counted for the draws that reached recording.
enum class Repeat : std::size_t {
    Draws,         ///< Draws counted.
    Pipeline,      ///< Same pipeline object.
    Targets,       ///< Same color and depth target images.
    Resources,     ///< Same buffer, image and sampler descriptors.
    All,           ///< All three of the above.
    Small,         ///< 6 or fewer indices or vertices.
    Textures,      ///< Same image and sampler descriptors.
    Buffers,       ///< Same buffer descriptors, ignoring offsets.
    GuestBuffers,  ///< Same guest shader buffers (V# address and size).
    VertexBuffers, ///< Same guest vertex buffer ranges.
    Count,
};
inline std::array<std::atomic<u64>, static_cast<std::size_t>(Repeat::Count)> repeats{};

inline u64 Mix(u64 hash, u64 value) {
    return hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2));
}

/// Guest inputs of the draw being bound on this thread, compared by CountDrawRepeats.
inline thread_local u64 draw_guest_buffers = 0;
inline thread_local u64 draw_vertex_buffers = 0;

/// Called when a draw or dispatch starts binding its resources.
inline void ResetDrawInputs() {
    draw_guest_buffers = 0;
    draw_vertex_buffers = 0;
}

inline void AddGuestBuffer(u64 address, u64 size) {
    if (enabled.load(std::memory_order_relaxed)) {
        draw_guest_buffers = Mix(Mix(draw_guest_buffers, address), size);
    }
}

inline void AddVertexBuffer(u64 address, u64 size) {
    if (enabled.load(std::memory_order_relaxed)) {
        draw_vertex_buffers = Mix(Mix(draw_vertex_buffers, address), size);
    }
}

inline std::atomic<u64> stream_copies{0};
inline std::atomic<u64> stream_bytes{0};
inline std::atomic<u64> user_data_copies{0};
inline std::atomic<u64> user_data_bytes{0};

/// A read-only buffer copied into the stream buffer by ObtainBuffer.
inline void CountStreamCopy(u64 size) {
    if (enabled.load(std::memory_order_relaxed)) {
        stream_copies.fetch_add(1, std::memory_order_relaxed);
        stream_bytes.fetch_add(size, std::memory_order_relaxed);
    }
}

/// Stream copies of the tracked addresses, by what the last copy of the same range (address and
/// size) on the same thread was.
enum class Reuse : std::size_t {
    Tracked,           ///< Copies looked at.
    TrackedBytes,      ///< Their bytes.
    SameFrame,         ///< Same range copied before in this frame, same contents.
    SameFrameBytes,    ///< Their bytes.
    EarlierFrame,      ///< Last copied in an earlier frame, same contents.
    EarlierFrameBytes, ///< Their bytes.
    Changed,           ///< Copied before, different contents.
    Count,
};
inline std::array<std::atomic<u64>, static_cast<std::size_t>(Reuse::Count)> reuse{};

/// One address in REUSE_SAMPLE is tracked (every copy of it), chosen by a multiplicative hash of
/// the address (the top bits, which also vary for aligned addresses).
inline constexpr u64 REUSE_SAMPLE = 16;

inline bool IsTrackedAddress(u64 address) {
    return (address * 0x9e3779b97f4a7c15ULL) >> 60 < 16 / REUSE_SAMPLE;
}

/// Looks at a stream copy of a tracked address: read(dst) must copy the size bytes at address
/// from guest memory into dst.
template <typename Read>
inline void CheckStreamReuse(u64 address, u64 size, Read&& read) {
    if (!enabled.load(std::memory_order_relaxed) || !IsTrackedAddress(address)) {
        return;
    }
    struct Last {
        u64 contents;
        u64 frame;
    };
    thread_local std::unordered_map<u64, Last> last_copies;
    thread_local std::vector<u8> scratch;
    scratch.resize(size);
    read(scratch.data());
    u64 contents = size;
    u64 i = 0;
    for (; i + sizeof(u64) <= size; i += sizeof(u64)) {
        u64 word;
        std::memcpy(&word, scratch.data() + i, sizeof(word));
        contents = Mix(contents, word);
    }
    for (; i < size; ++i) {
        contents = Mix(contents, scratch[i]);
    }
    const u64 frame = frame_index.load(std::memory_order_relaxed);
    const auto add = [](Reuse kind, u64 value) {
        reuse[static_cast<std::size_t>(kind)].fetch_add(value, std::memory_order_relaxed);
    };
    add(Reuse::Tracked, 1);
    add(Reuse::TrackedBytes, size);
    if (last_copies.size() >= (1U << 20)) {
        last_copies.clear();
    }
    const auto [it, is_new] = last_copies.try_emplace(Mix(address, size), Last{contents, frame});
    if (is_new) {
        return;
    }
    if (it->second.contents != contents) {
        add(Reuse::Changed, 1);
    } else if (it->second.frame == frame) {
        add(Reuse::SameFrame, 1);
        add(Reuse::SameFrameBytes, size);
    } else {
        add(Reuse::EarlierFrame, 1);
        add(Reuse::EarlierFrameBytes, size);
    }
    it->second = Last{contents, frame};
}

/// Flattened user data copied into the stream buffer by BindBuffers.
inline void CountUserData(u64 size) {
    if (enabled.load(std::memory_order_relaxed)) {
        user_data_copies.fetch_add(1, std::memory_order_relaxed);
        user_data_bytes.fetch_add(size, std::memory_order_relaxed);
    }
}

/// Compares a draw with the previous one on the same thread.
inline void CountDrawRepeats(const void* pipeline, u64 targets, u64 resources, u32 num_indices,
                             u64 textures, u64 buffers) {
    thread_local const void* last_pipeline = nullptr;
    thread_local u64 last_targets = 0;
    thread_local u64 last_resources = 0;
    thread_local u64 last_textures = 0;
    thread_local u64 last_buffers = 0;
    thread_local u64 last_guest_buffers = 0;
    thread_local u64 last_vertex_buffers = 0;
    thread_local bool has_last = false;
    const bool same_pipeline = has_last && pipeline == last_pipeline;
    const bool same_targets = has_last && targets == last_targets;
    const bool same_resources = has_last && resources == last_resources;
    const auto add = [](Repeat repeat, bool value) {
        if (value) {
            repeats[static_cast<std::size_t>(repeat)].fetch_add(1, std::memory_order_relaxed);
        }
    };
    add(Repeat::Draws, true);
    add(Repeat::Pipeline, same_pipeline);
    add(Repeat::Targets, same_targets);
    add(Repeat::Resources, same_resources);
    add(Repeat::All, same_pipeline && same_targets && same_resources);
    add(Repeat::Small, num_indices <= 6);
    add(Repeat::Textures, has_last && textures == last_textures);
    add(Repeat::Buffers, has_last && buffers == last_buffers);
    add(Repeat::GuestBuffers, has_last && draw_guest_buffers == last_guest_buffers);
    add(Repeat::VertexBuffers, has_last && draw_vertex_buffers == last_vertex_buffers);
    last_guest_buffers = draw_guest_buffers;
    last_vertex_buffers = draw_vertex_buffers;
    last_pipeline = pipeline;
    last_targets = targets;
    last_resources = resources;
    last_textures = textures;
    last_buffers = buffers;
    has_last = true;
}

/// The PM4 opcode being processed on this thread (0x100 is added for compute queues).
inline thread_local u32 current_opcode = 0xffffffffU;

/// What the GPU thread is doing inside the current PM4 packet, for the readbacks it causes.
enum class Context : u32 {
    Packet,      ///< The packet handler itself (reading or writing guest memory directly).
    Pipeline,    ///< Pipeline lookup (shader programs, user data, specialization).
    Resources,   ///< Rasterizer::BindResources (shader buffers and textures).
    VertexIndex, ///< BindVertexBuffers and BindIndexBuffer.
    Command,     ///< A command queued from another thread (Liverpool::ProcessCommands).
};

inline const char* ContextName(u32 context) {
    constexpr std::array<const char*, 5> names{"packet", "pipeline", "resources", "vertex",
                                               "command"};
    return context < names.size() ? names[context] : "?";
}

inline thread_local Context current_context = Context::Packet;

/// Sets the context of the calling thread for the lifetime of the object.
class ContextScope {
public:
    explicit ContextScope(Context context) : active{enabled.load(std::memory_order_relaxed)} {
        if (active) {
            previous = current_context;
            current_context = context;
        }
    }
    ~ContextScope() {
        if (active) {
            current_context = previous;
        }
    }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;

private:
    bool active;
    Context previous = Context::Packet;
};

/// Readbacks by fault address, and by the PM4 opcode and context of the GPU thread.
struct ReadbackStats {
    static constexpr std::size_t MAX_KEYS = 4096;
    std::mutex mutex;
    std::unordered_map<u64, u32> addresses;
    std::unordered_map<u32, u32> opcodes;
    /// By "address source": the PM4 opcode and context on the GPU thread, else the thread name.
    std::unordered_map<std::string, u32> sources;
    u64 on_gpu_thread = 0;
    u64 on_other_threads = 0;
    u64 for_writes = 0;
    u64 downloads = 0;
    u64 download_bytes = 0;
    u64 download_ranges = 0;
};
inline ReadbackStats readback_stats;

/// A readback requested by BufferCache::ReadMemory.
inline void CountReadback(u64 address, bool is_write, bool on_gpu_thread_) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    const std::string source =
        on_gpu_thread_
            ? fmt::format("{:#x} {:#x}/{}{}", address, current_opcode & 0xffffU,
                          ContextName(static_cast<u32>(current_context)), is_write ? " write" : "")
            : fmt::format("{:#x} {}{}", address, Common::GetCurrentThreadName(),
                          is_write ? " write" : "");
    std::scoped_lock lock{readback_stats.mutex};
    ++(on_gpu_thread_ ? readback_stats.on_gpu_thread : readback_stats.on_other_threads);
    if (readback_stats.sources.size() < ReadbackStats::MAX_KEYS ||
        readback_stats.sources.contains(source)) {
        ++readback_stats.sources[source];
    }
    readback_stats.for_writes += is_write;
    if (readback_stats.addresses.size() < ReadbackStats::MAX_KEYS ||
        readback_stats.addresses.contains(address)) {
        ++readback_stats.addresses[address];
    }
    if (on_gpu_thread_) {
        const u32 key = (current_opcode & 0xffffU) | (static_cast<u32>(current_context) << 16);
        ++readback_stats.opcodes[key];
    }
}

/// Readbacks by fault address: whether the accessed bytes themselves were written by the GPU.
struct ReadbackTargets {
    std::mutex mutex;
    std::unordered_map<u64, std::pair<u32, u32>> by_address; ///< {GPU-written, not}
    u64 gpu_written = 0;
    u64 not_gpu_written = 0;
};
inline ReadbackTargets readback_targets;

/// Called by BufferCache::ReadMemory with whether [address, address + size) intersects the
/// ranges the GPU has written (gpu_modified_ranges) at the time of the readback.
inline void CountReadbackTarget(u64 address, bool gpu_written) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    std::scoped_lock lock{readback_targets.mutex};
    ++(gpu_written ? readback_targets.gpu_written : readback_targets.not_gpu_written);
    auto& by_address = readback_targets.by_address;
    if (by_address.size() < ReadbackStats::MAX_KEYS || by_address.contains(address)) {
        auto& counts = by_address[address];
        ++(gpu_written ? counts.first : counts.second);
    }
}

/// A download of GPU-modified ranges by BufferCache::DownloadMemory.
inline void CountDownload(u64 bytes, u64 ranges) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    std::scoped_lock lock{readback_stats.mutex};
    ++readback_stats.downloads;
    readback_stats.download_bytes += bytes;
    readback_stats.download_ranges += ranges;
}

inline void CountFlip() {
    if (enabled.load(std::memory_order_relaxed)) {
        flips.fetch_add(1, std::memory_order_relaxed);
        frame_index.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void CountAccessViolation() {
    if (enabled.load(std::memory_order_relaxed)) {
        access_violations.fetch_add(1, std::memory_order_relaxed);
    }
}

/// Adds the time the GPU thread waited for work since start (a value from Begin()).
inline void AddGpuIdle(u64 start) {
    if (start != 0) {
        gpu_idle_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
    }
}

/// Enables the statistics and starts the thread that logs them once a second.
/// CPU time of the threads named with Common::SetCurrentThreadName (the guest threads and the
/// emulator's own), read once a second.
struct ThreadCpuTimes {
    /// Reads the CPU time of one thread: {nanoseconds, CPU cycles}, 0 when not available.
    using Reader = std::function<std::pair<u64, u64>()>;
    struct Entry {
        std::string name;
        Reader read;
        u64 last_ns = 0;
        u64 last_cycles = 0;
    };
    static constexpr std::size_t MAX_THREADS = 1024;
    std::mutex mutex;
    std::unordered_map<u64, Entry> threads; ///< By OS thread id.
};
inline ThreadCpuTimes thread_cpu_times;

/// Starts reading the CPU time of a thread. A later call with the same OS thread id (a renamed
/// thread, or a new thread that got the id of an exited one) replaces the entry.
inline void RegisterThread(u64 id, const char* name, ThreadCpuTimes::Reader read) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    const auto [ns, cycles] = read();
    std::scoped_lock lock{thread_cpu_times.mutex};
    auto& threads = thread_cpu_times.threads;
    if (threads.size() >= ThreadCpuTimes::MAX_THREADS && !threads.contains(id)) {
        return;
    }
    threads.insert_or_assign(id, ThreadCpuTimes::Entry{name, std::move(read), ns, cycles});
}

/// Returns the efficiency class of the logical processor the calling thread runs on (set by the
/// platform code; on Windows, SYSTEM_CPU_SET_INFORMATION::EfficiencyClass).
using CoreClassReader = u32 (*)();
inline std::atomic<CoreClassReader> core_class_reader{nullptr};
inline constexpr std::size_t MAX_CORE_CLASSES = 4;
inline std::array<std::atomic<u64>, MAX_CORE_CLASSES> gpu_thread_core_samples{};

/// For one draw in SAMPLE_RATE, counts the class of the core the GPU thread is running on.
inline void SampleGpuThreadCore() {
    if (!enabled.load(std::memory_order_relaxed) || !Sample()) {
        return;
    }
    const CoreClassReader reader = core_class_reader.load(std::memory_order_relaxed);
    if (reader == nullptr) {
        return;
    }
    const u32 core_class = std::min<u32>(reader(), MAX_CORE_CLASSES - 1);
    gpu_thread_core_samples[core_class].fetch_add(1, std::memory_order_relaxed);
}

/// Draws and dispatches grouped by what they run (pipeline and render target size, or compute
/// pipeline), for the busiest groups of each second.
struct WorkGroups {
    static constexpr std::size_t MAX_GROUPS = 4096;
    struct Group {
        std::string description;
        u64 count = 0;
        u64 elements = 0; ///< Indices times instances for draws, workgroups for dispatches.
    };
    std::mutex mutex;
    std::unordered_map<u64, Group> draws;
    std::unordered_map<u64, Group> dispatches;
};
inline WorkGroups work_groups;

/// Counts one draw or dispatch in its group. describe() is called only for a group's first
/// call in a second.
inline void CountWork(std::unordered_map<u64, WorkGroups::Group>& groups, u64 key, u64 elements,
                      auto&& describe) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    std::scoped_lock lock{work_groups.mutex};
    auto it = groups.find(key);
    if (it == groups.end()) {
        if (groups.size() >= WorkGroups::MAX_GROUPS) {
            return;
        }
        it = groups.emplace(key, WorkGroups::Group{describe(), 0, 0}).first;
    }
    ++it->second.count;
    it->second.elements += elements;
}

inline void CountDrawGroup(u64 key, u64 elements, auto&& describe) {
    CountWork(work_groups.draws, key, elements, describe);
}

inline void CountDispatchGroup(u64 key, u64 elements, auto&& describe) {
    CountWork(work_groups.dispatches, key, elements, describe);
}

inline void CountFileRead(s64 bytes) {
    if (enabled.load(std::memory_order_relaxed) && bytes > 0) {
        file_read_bytes.fetch_add(static_cast<u64>(bytes), std::memory_order_relaxed);
    }
}

/// An occlusion query result written by an EVENT_WRITE ZPASS_DONE (PixelPipeStatDump) packet.
inline void CountOcclusionQueryDump() {
    if (enabled.load(std::memory_order_relaxed)) {
        occlusion_query_dumps.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void CountFlipPendingQuery() {
    if (enabled.load(std::memory_order_relaxed)) {
        flip_pending_queries.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void Start() {
    enabled.store(true, std::memory_order_relaxed);
    std::thread{[] {
        u64 last = NowNs();
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds{1});
            const u64 now = NowNs();
            const double seconds = static_cast<double>(now - last) / 1e9;
            last = now;
            const auto take = [](Kind kind) {
                auto& counter = counters[static_cast<std::size_t>(kind)];
                const u64 count = counter.count.exchange(0, std::memory_order_relaxed);
                const u64 ns = counter.ns.exchange(0, std::memory_order_relaxed);
                return std::pair{count, static_cast<double>(ns) / 1e6};
            };
            const auto [pipelines, pipeline_ms] = take(Kind::Pipeline);
            const auto [resident, resident_ms] = take(Kind::EnsureResident);
            const auto [uploads, upload_ms] = take(Kind::Upload);
            const auto [readbacks, readback_ms] = take(Kind::Readback);
            const double readback_gpu_ms = take(Kind::ReadbackGpu).second;
            const auto [draws, draw_ms] = take(Kind::Draw);
            const auto [dispatches, dispatch_ms] = take(Kind::Dispatch);
            const auto [submits, submit_ms] = take(Kind::Submit);
            const auto [gpu_waits, gpu_wait_ms] = take(Kind::GpuThreadWait);
            const auto [other_waits, other_wait_ms] = take(Kind::OtherWait);
            const double fps =
                static_cast<double>(flips.exchange(0, std::memory_order_relaxed)) / seconds;
            const double idle_ms =
                static_cast<double>(gpu_idle_ns.exchange(0, std::memory_order_relaxed)) / 1e6;
            const double busy = 100.0 - idle_ms / (seconds * 1000.0) * 100.0;
            LOG_INFO(Debug,
                     "FrameStats: {:.1f} fps, GPU thread busy {:.0f}%, draws {} ({:.1f} ms), "
                     "dispatches {} ({:.1f} ms), submits {} ({:.1f} ms), GPU thread waits {} "
                     "({:.1f} ms), other waits {} ({:.1f} ms), pipelines {} ({:.1f} ms), "
                     "resident {} ({:.1f} ms), uploads {} ({:.1f} ms), readbacks {} ({:.1f} ms, "
                     "GPU thread {:.1f} ms), access violations {}, occlusion queries {}",
                     fps, busy, draws, draw_ms, dispatches, dispatch_ms, submits, submit_ms,
                     gpu_waits, gpu_wait_ms, other_waits, other_wait_ms, pipelines, pipeline_ms,
                     resident, resident_ms, uploads, upload_ms, readbacks, readback_ms,
                     readback_gpu_ms, access_violations.exchange(0, std::memory_order_relaxed),
                     occlusion_query_dumps.exchange(0, std::memory_order_relaxed));
            const auto [skipped, skipped_ms] = take(Kind::DrawSkipped);
            const auto [lookups, lookup_ms] = take(Kind::DrawPipeline);
            const auto [resource_draws, resource_ms] = take(Kind::DrawResources);
            const auto [buffer_draws, buffer_ms] = take(Kind::DrawBuffers);
            const auto [state_draws, state_ms] = take(Kind::DrawSetup);
            const auto [recorded, record_ms] = take(Kind::DrawRecord);
            LOG_INFO(Debug,
                     "FrameStats draws: skipped {} ({:.1f} ms), pipeline {} ({:.1f} ms), "
                     "resources {} ({:.1f} ms), buffers {} ({:.1f} ms), state {} ({:.1f} ms), "
                     "record {} ({:.1f} ms)",
                     skipped, skipped_ms, lookups, lookup_ms, resource_draws, resource_ms,
                     buffer_draws, buffer_ms, state_draws, state_ms, recorded, record_ms);
            const auto take_repeat = [](Repeat repeat) {
                return repeats[static_cast<std::size_t>(repeat)].exchange(
                    0, std::memory_order_relaxed);
            };
            const u64 compared = take_repeat(Repeat::Draws);
            const auto percent = [compared](u64 value) {
                return compared == 0
                           ? 0.0
                           : static_cast<double>(value) * 100.0 / static_cast<double>(compared);
            };
            const u64 same_pipeline = take_repeat(Repeat::Pipeline);
            const u64 same_targets = take_repeat(Repeat::Targets);
            const u64 same_resources = take_repeat(Repeat::Resources);
            const u64 same_all = take_repeat(Repeat::All);
            const u64 small_draws = take_repeat(Repeat::Small);
            const u64 same_textures = take_repeat(Repeat::Textures);
            const u64 same_buffers = take_repeat(Repeat::Buffers);
            LOG_INFO(Debug,
                     "FrameStats repeats: draws {}, same pipeline {:.0f}%, same targets {:.0f}%, "
                     "same resources {:.0f}%, all three {:.0f}%, 6 or fewer indices {:.0f}%, "
                     "same textures {:.0f}%, same buffers ignoring offsets {:.0f}%",
                     compared, percent(same_pipeline), percent(same_targets),
                     percent(same_resources), percent(same_all), percent(small_draws),
                     percent(same_textures), percent(same_buffers));
            const auto [target_lookups, target_ms] = take(Kind::TargetLookup);
            const auto [buffer_binds, buffer_bind_ms] = take(Kind::BufferBinds);
            const auto [texture_binds, texture_bind_ms] = take(Kind::TextureBinds);
            const auto [vertex_binds, vertex_bind_ms] = take(Kind::VertexBinds);
            const auto [render_passes, render_pass_ms] = take(Kind::RenderPass);
            LOG_INFO(Debug,
                     "FrameStats binds: render targets {} ({:.1f} ms), shader buffers {} ({:.1f} "
                     "ms), shader textures {} ({:.1f} ms), vertex buffers {} ({:.1f} ms), "
                     "render pass {} ({:.1f} ms)",
                     target_lookups, target_ms, buffer_binds, buffer_bind_ms, texture_binds,
                     texture_bind_ms, vertex_binds, vertex_bind_ms, render_passes, render_pass_ms);
            const auto [obtains, obtain_ms] = take(Kind::ObtainBuffer);
            const double check_ms = take(Kind::ObtainCheck).second;
            const double map_ms = take(Kind::ObtainMap).second;
            const double copy_ms = take(Kind::ObtainCopy).second;
            const double commit_ms = take(Kind::ObtainCommit).second;
            const double cache_ms = take(Kind::ObtainCache).second;
            const u64 copies = stream_copies.exchange(0, std::memory_order_relaxed);
            const u64 copied = stream_bytes.exchange(0, std::memory_order_relaxed);
            const u64 ud_copies = user_data_copies.exchange(0, std::memory_order_relaxed);
            const u64 ud_copied = user_data_bytes.exchange(0, std::memory_order_relaxed);
            const u64 same_guest = take_repeat(Repeat::GuestBuffers);
            const u64 same_vertex = take_repeat(Repeat::VertexBuffers);
            LOG_INFO(Debug,
                     "FrameStats buffers: obtain {} (~{:.1f} ms), check ~{:.1f} ms, stream "
                     "copies {} ({} KB, map ~{:.1f} ms, copy ~{:.1f} ms, commit ~{:.1f} ms), "
                     "cache path {} (~{:.1f} ms), user data copies {} ({} KB), same guest "
                     "buffers {:.0f}%, same vertex buffers {:.0f}%",
                     obtains, obtain_ms, check_ms, copies, copied / 1024, map_ms, copy_ms,
                     commit_ms, obtains > copies ? obtains - copies : 0, cache_ms, ud_copies,
                     ud_copied / 1024, percent(same_guest), percent(same_vertex));
            const auto take_reuse = [](Reuse kind) {
                return reuse[static_cast<std::size_t>(kind)].exchange(0, std::memory_order_relaxed);
            };
            const u64 tracked = take_reuse(Reuse::Tracked);
            const u64 tracked_bytes = take_reuse(Reuse::TrackedBytes);
            const u64 same_frame = take_reuse(Reuse::SameFrame);
            const u64 same_frame_bytes = take_reuse(Reuse::SameFrameBytes);
            const u64 earlier_frame = take_reuse(Reuse::EarlierFrame);
            const u64 earlier_frame_bytes = take_reuse(Reuse::EarlierFrameBytes);
            const u64 changed = take_reuse(Reuse::Changed);
            const u64 seen = same_frame + earlier_frame + changed;
            const u64 first = tracked > seen ? tracked - seen : 0;
            const auto share = [](u64 value, u64 total) {
                return total == 0 ? 0.0
                                  : static_cast<double>(value) * 100.0 / static_cast<double>(total);
            };
            LOG_INFO(Debug,
                     "FrameStats reuse: tracked copies {} ({} KB), same contents as an earlier "
                     "copy in this frame {:.0f}% ({:.0f}% of bytes), same contents as a copy in "
                     "an earlier frame {:.0f}% ({:.0f}% of bytes), contents changed {:.0f}%, "
                     "first copy {:.0f}%",
                     tracked, tracked_bytes / 1024, share(same_frame, tracked),
                     share(same_frame_bytes, tracked_bytes), share(earlier_frame, tracked),
                     share(earlier_frame_bytes, tracked_bytes), share(changed, tracked),
                     share(first, tracked));
            const auto [idle_waits, idle_wait_ms] = take(Kind::GpuIdleWait);
            const auto [finishes, finish_ms] = take(Kind::ReadbackFinish);
            std::vector<std::pair<u64, u32>> top_addresses;
            std::vector<std::pair<u32, u32>> top_opcodes;
            std::vector<std::pair<std::string, u32>> top_sources;
            u64 rb_gpu_thread = 0;
            u64 rb_other = 0;
            u64 rb_writes = 0;
            u64 rb_downloads = 0;
            u64 rb_bytes = 0;
            u64 rb_ranges = 0;
            {
                std::scoped_lock lock{readback_stats.mutex};
                top_addresses.assign(readback_stats.addresses.begin(),
                                     readback_stats.addresses.end());
                top_opcodes.assign(readback_stats.opcodes.begin(), readback_stats.opcodes.end());
                top_sources.assign(readback_stats.sources.begin(), readback_stats.sources.end());
                rb_gpu_thread = std::exchange(readback_stats.on_gpu_thread, 0);
                rb_other = std::exchange(readback_stats.on_other_threads, 0);
                rb_writes = std::exchange(readback_stats.for_writes, 0);
                rb_downloads = std::exchange(readback_stats.downloads, 0);
                rb_bytes = std::exchange(readback_stats.download_bytes, 0);
                rb_ranges = std::exchange(readback_stats.download_ranges, 0);
                readback_stats.addresses.clear();
                readback_stats.opcodes.clear();
                readback_stats.sources.clear();
            }
            const auto by_count = [](const auto& a, const auto& b) { return a.second > b.second; };
            std::ranges::sort(top_addresses, by_count);
            std::ranges::sort(top_opcodes, by_count);
            std::string addresses_text;
            for (std::size_t k = 0; k < std::min<std::size_t>(top_addresses.size(), 6); ++k) {
                addresses_text +=
                    fmt::format(" {:#x} x{}", top_addresses[k].first, top_addresses[k].second);
            }
            std::string opcodes_text;
            for (std::size_t k = 0; k < std::min<std::size_t>(top_opcodes.size(), 6); ++k) {
                opcodes_text +=
                    fmt::format(" {:#x}/{} x{}", top_opcodes[k].first & 0xffffU,
                                ContextName(top_opcodes[k].first >> 16), top_opcodes[k].second);
            }
            LOG_INFO(Debug,
                     "FrameStats readbacks: GPU thread {}, other threads {}, for writes {}, "
                     "downloads {} ({} KB, {} ranges), host GPU wait {} ({:.1f} ms), "
                     "WaitGpuIdle {} ({:.1f} ms), {} addresses, top:{}, opcodes:{}",
                     rb_gpu_thread, rb_other, rb_writes, rb_downloads, rb_bytes / 1024, rb_ranges,
                     finishes, finish_ms, idle_waits, idle_wait_ms, top_addresses.size(),
                     addresses_text, opcodes_text);
            std::ranges::sort(top_sources, by_count);
            std::string sources_text;
            for (std::size_t k = 0; k < std::min<std::size_t>(top_sources.size(), 12); ++k) {
                sources_text +=
                    fmt::format(" [{}] x{},", top_sources[k].first, top_sources[k].second);
            }
            LOG_INFO(Debug, "FrameStats readback sources: {} kinds,{}", top_sources.size(),
                     sources_text);
            std::vector<std::pair<u64, std::pair<u32, u32>>> targets;
            u64 targets_gpu = 0;
            u64 targets_other = 0;
            {
                std::scoped_lock lock{readback_targets.mutex};
                targets.assign(readback_targets.by_address.begin(),
                               readback_targets.by_address.end());
                targets_gpu = std::exchange(readback_targets.gpu_written, 0);
                targets_other = std::exchange(readback_targets.not_gpu_written, 0);
                readback_targets.by_address.clear();
            }
            std::ranges::sort(targets, [](const auto& a, const auto& b) {
                return a.second.first + a.second.second > b.second.first + b.second.second;
            });
            std::string targets_text;
            for (std::size_t k = 0; k < std::min<std::size_t>(targets.size(), 12); ++k) {
                targets_text +=
                    fmt::format(" [{:#x} {}/{}]", targets[k].first, targets[k].second.first,
                                targets[k].second.first + targets[k].second.second);
            }
            LOG_INFO(Debug,
                     "FrameStats readback targets: accessed bytes GPU-written in {} of {}, by "
                     "address (GPU-written/all):{}",
                     targets_gpu, targets_gpu + targets_other, targets_text);
            std::vector<std::pair<std::string, std::pair<u64, u64>>> draw_groups;
            std::vector<std::pair<std::string, std::pair<u64, u64>>> dispatch_groups;
            {
                std::scoped_lock lock{work_groups.mutex};
                for (const auto& [key, group] : work_groups.draws) {
                    draw_groups.push_back({group.description, {group.count, group.elements}});
                }
                for (const auto& [key, group] : work_groups.dispatches) {
                    dispatch_groups.push_back({group.description, {group.count, group.elements}});
                }
                work_groups.draws.clear();
                work_groups.dispatches.clear();
            }
            const auto by_group_count = [](const auto& a, const auto& b) {
                return a.second.first > b.second.first;
            };
            std::ranges::sort(draw_groups, by_group_count);
            std::ranges::sort(dispatch_groups, by_group_count);
            const auto groups_text = [](const auto& groups, std::size_t limit) {
                std::string text;
                for (std::size_t k = 0; k < std::min<std::size_t>(groups.size(), limit); ++k) {
                    text += fmt::format(" [{}] x{} ({}),", groups[k].first, groups[k].second.first,
                                        groups[k].second.second);
                }
                return text;
            };
            LOG_INFO(Debug, "FrameStats draw groups: {} groups,{}", draw_groups.size(),
                     groups_text(draw_groups, 10));
            LOG_INFO(Debug, "FrameStats dispatch groups: {} groups,{}", dispatch_groups.size(),
                     groups_text(dispatch_groups, 8));
            const auto [file_reads, file_read_ms] = take(Kind::FileRead);
            const double file_invalidate_ms = take(Kind::FileInvalidate).second;
            LOG_INFO(Debug, "FrameStats files: reads {} ({} KB, {:.1f} ms), invalidate {:.1f} ms",
                     file_reads, file_read_bytes.exchange(0, std::memory_order_relaxed) / 1024,
                     file_read_ms, file_invalidate_ms);
            const auto [vertex_calls, vertex_ms] = take(Kind::VertexSampled);
            const double inputs_ms = take(Kind::VertexInputs).second;
            const double ranges_ms = take(Kind::VertexRanges).second;
            const auto [vertex_ranges, vertex_obtain_ms] = take(Kind::VertexObtain);
            const double access_ms = take(Kind::VertexAccess).second;
            const double host_bind_ms = take(Kind::VertexHostBind).second;
            const auto [index_binds, index_bind_ms] = take(Kind::IndexBinds);
            LOG_INFO(Debug,
                     "FrameStats vertex: calls {} (~{:.1f} ms), inputs ~{:.1f} ms, ranges "
                     "~{:.1f} ms, obtain ~{} (~{:.1f} ms), access ~{:.1f} ms, bind ~{:.1f} ms, "
                     "index buffers {} ({:.1f} ms)",
                     vertex_calls, vertex_ms, inputs_ms, ranges_ms, vertex_ranges, vertex_obtain_ms,
                     access_ms, host_bind_ms, index_binds, index_bind_ms);
            struct ThreadUse {
                std::string name;
                u64 ns;
                u64 cycles;
            };
            std::vector<ThreadUse> thread_uses;
            {
                std::scoped_lock lock{thread_cpu_times.mutex};
                for (auto& [id, entry] : thread_cpu_times.threads) {
                    const auto [ns, cycles] = entry.read();
                    const u64 used_ns = ns > entry.last_ns ? ns - entry.last_ns : 0;
                    const u64 used_cycles =
                        cycles > entry.last_cycles ? cycles - entry.last_cycles : 0;
                    entry.last_ns = ns;
                    entry.last_cycles = cycles;
                    if (used_ns != 0 || used_cycles != 0) {
                        thread_uses.push_back({entry.name, used_ns, used_cycles});
                    }
                }
            }
            std::ranges::sort(thread_uses, [](const ThreadUse& a, const ThreadUse& b) {
                return a.cycles != b.cycles ? a.cycles > b.cycles : a.ns > b.ns;
            });
            std::string threads_text;
            for (std::size_t k = 0; k < std::min<std::size_t>(thread_uses.size(), 8); ++k) {
                threads_text += fmt::format(" {} {:.1f} ms ({} Mcycles),", thread_uses[k].name,
                                            static_cast<double>(thread_uses[k].ns) / 1e6,
                                            thread_uses[k].cycles / 1000000);
            }
            std::array<u64, MAX_CORE_CLASSES> core_samples{};
            u64 core_sample_total = 0;
            for (std::size_t k = 0; k < MAX_CORE_CLASSES; ++k) {
                core_samples[k] = gpu_thread_core_samples[k].exchange(0, std::memory_order_relaxed);
                core_sample_total += core_samples[k];
            }
            std::string cores_text;
            for (std::size_t k = 0; k < MAX_CORE_CLASSES; ++k) {
                if (core_samples[k] != 0) {
                    cores_text += fmt::format(" class {} {:.0f}%", k,
                                              static_cast<double>(core_samples[k]) * 100.0 /
                                                  static_cast<double>(core_sample_total));
                }
            }
            LOG_INFO(Debug,
                     "FrameStats threads: {} running,{} flip pending queries {}, GPU thread "
                     "cores:{}",
                     thread_uses.size(), threads_text,
                     flip_pending_queries.exchange(0, std::memory_order_relaxed), cores_text);
        }
    }}.detach();
}

} // namespace Common::FrameStats
