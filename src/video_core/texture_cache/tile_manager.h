// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/texture_cache/depth_staging.h"

namespace VideoCore {

struct ImageInfo;
struct Image;
class StreamBuffer;

TilingFormat GetTilingFormat(const ImageInfo& info, vk::Format host_format);
u64 GuestToHostBytes(u64 guest_bytes, u32 guest_bytes_per_pixel, const TilingFormat& tiling_format);

class TileManager {
    static constexpr size_t NUM_BPPS = 6;
    static constexpr size_t NUM_SAMPLE_COUNTS = 4;
    static constexpr size_t NUM_DEPTH_CONVERSIONS = 3;
    static constexpr size_t NUM_PIPELINE_LAYOUTS = 2;

public:
    using ScratchBuffer = std::pair<vk::Buffer, VmaAllocation>;
    using Result = std::pair<vk::Buffer, u32>;

    explicit TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         StreamBuffer& stream_buffer);
    ~TileManager();

    void TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                   Buffer& out_buffer, u64 out_offset, u64 copy_size);

    Result DetileImage(Buffer& in_buffer, u32 in_offset, const ImageInfo& info,
                       vk::Format host_format);

private:
    vk::Pipeline GetTilingPipeline(const ImageInfo& info, bool is_tiler,
                                   DepthConversion depth_conversion, bool is_linear);
    // Scratch buffers are pooled by capacity and handed back once the GPU has finished with
    // them, so a detile or tile pass does not create and destroy a multi-megabyte device
    // allocation every time an image is refreshed.
    ScratchBuffer GetScratchBuffer(vk::DeviceSize size);
    void ReleaseScratchBuffer(ScratchBuffer buffer, vk::DeviceSize size);
    static vk::DeviceSize ScratchCapacity(vk::DeviceSize size);

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    StreamBuffer& stream_buffer;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pl_layout;
    std::array<vk::UniquePipeline, AmdGpu::NUM_TILE_MODES * NUM_BPPS * NUM_SAMPLE_COUNTS *
                                       NUM_DEPTH_CONVERSIONS * NUM_PIPELINE_LAYOUTS>
        detilers{};
    std::array<vk::UniquePipeline, AmdGpu::NUM_TILE_MODES * NUM_BPPS * NUM_SAMPLE_COUNTS *
                                       NUM_DEPTH_CONVERSIONS * NUM_PIPELINE_LAYOUTS>
        tilers{};
    static constexpr size_t MAX_POOLED_SCRATCH_BUFFERS = 16;
    std::vector<std::pair<vk::DeviceSize, ScratchBuffer>> free_scratch_buffers;
};

} // namespace VideoCore
