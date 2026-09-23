// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <cstddef>
#include <limits>
#include <string_view>

#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/depth_format.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/tile_manager.h"

#include "video_core/host_shaders/tiling_comp.h"

#include <magic_enum/magic_enum.hpp>
#include <vk_mem_alloc.h>

namespace VideoCore {

struct TilingInfo {
    u32 bank_swizzle;
    u32 num_slices;
    u32 num_mips;
    u32 num_texels;
    std::array<ImageInfo::MipInfo, 16> mips;
};
static_assert(sizeof(TilingInfo) == 272);
static_assert(offsetof(TilingInfo, mips) == 16);

[[nodiscard]] constexpr u32 BitsPerPixelId(const u32 num_bits) {
    switch (num_bits) {
    case 8:
        return 0;
    case 16:
        return 1;
    case 32:
        return 2;
    case 64:
        return 3;
    case 96:
        return 4;
    case 128:
        return 5;
    default:
        UNREACHABLE_MSG("Unsupported image bits per pixel {}", num_bits);
    }
}

void ValidateStorageDescriptor(const Vulkan::Instance& instance, const vk::DeviceSize offset,
                               const vk::DeviceSize range, const std::string_view name) {
    ASSERT_MSG(range > 0, "{} storage descriptor has an empty range", name);
    ASSERT_MSG(offset % instance.StorageMinAlignment() == 0,
               "{} storage descriptor offset {:#x} is not aligned to {:#x}", name, offset,
               instance.StorageMinAlignment());
    ASSERT_MSG(range <= instance.StorageMaxSize(),
               "{} storage descriptor range {:#x} exceeds the device limit {:#x}", name, range,
               instance.StorageMaxSize());
}

// Texels are spread over a 2D grid of 64-wide groups: a single row would exceed the device's
// maximum group count for images above 4M texels.
static Dispatch2D ComputeTilingDispatch(const Vulkan::Instance& instance, const u64 num_texels,
                                        const std::string_view name) {
    const auto& max_group_count = instance.MaxComputeWorkGroupCount();
    const auto dispatch =
        TryComputeDispatch2D(num_texels, 64, max_group_count[0], max_group_count[1]);
    ASSERT_MSG(dispatch.has_value(),
               "{} dispatch for {} texels exceeds device workgroup limits {}x{}", name, num_texels,
               max_group_count[0], max_group_count[1]);
    return dispatch.value_or(Dispatch2D{});
}

static u32 TexelCount(const ImageInfo& info, const std::string_view name) {
    const u64 num_texels = info.guest_size / (info.num_bits / 8);
    ASSERT_MSG(num_texels <= std::numeric_limits<u32>::max(),
               "{} texel count {} exceeds the shader index range", name, num_texels);
    return static_cast<u32>(num_texels);
}

TilingFormat GetTilingFormat(const ImageInfo& info, const vk::Format host_format) {
    const u32 guest_bytes_per_pixel = info.num_bits / 8;
    if (!info.props.is_depth) {
        return {.host_bytes_per_pixel = guest_bytes_per_pixel,
                .depth_conversion = DepthConversion::None};
    }

    // Buffer-image copies address a depth aspect using the host format's depth plane. When a
    // combined D16/S8 attachment falls back to D24/S8 or D32/S8, the staging layout and values
    // must be widened before the copy instead of treating the guest's 16-bit data as 32-bit.
    return GetDepthTilingFormat(info.pixel_format, host_format, guest_bytes_per_pixel);
}

u64 GuestToHostBytes(const u64 guest_bytes, const u32 guest_bytes_per_pixel,
                     const TilingFormat& tiling_format) {
    const auto host_bytes = TryGuestToHostBytes(guest_bytes, guest_bytes_per_pixel, tiling_format);
    ASSERT_MSG(host_bytes.has_value(),
               "Invalid host image byte size conversion: guest_bytes={}, guest_bpp={}, host_bpp={}",
               guest_bytes, guest_bytes_per_pixel, tiling_format.host_bytes_per_pixel);
    return host_bytes.value_or(0);
}

TileManager::TileManager(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, StreamBuffer& stream_buffer_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, stream_buffer{stream_buffer_} {
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};

    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto desc_layout_result = device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(desc_layout_result.result == vk::Result::eSuccess,
               "Failed to create descriptor set layout: {}",
               vk::to_string(desc_layout_result.result));
    desc_layout = std::move(desc_layout_result.value);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 0U,
        .pPushConstantRanges = nullptr,
    };
    auto [layout_result, layout] = device.createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "Failed to create pipeline layout: {}",
               vk::to_string(layout_result));
    pl_layout = std::move(layout);
}

TileManager::~TileManager() = default;

vk::Pipeline TileManager::GetTilingPipeline(const ImageInfo& info, const bool is_tiler,
                                            const DepthConversion depth_conversion,
                                            const bool is_linear) {
    ASSERT_MSG(std::has_single_bit(info.num_samples), "Invalid sample count {}", info.num_samples);
    const u32 sample_id = std::countr_zero(info.num_samples);
    ASSERT_MSG(sample_id < NUM_SAMPLE_COUNTS, "Unsupported sample count {}", info.num_samples);
    const u32 pipeline_id = u32(info.tile_mode) * NUM_BPPS + BitsPerPixelId(info.num_bits);
    const u32 pl_id = ((pipeline_id * NUM_SAMPLE_COUNTS + sample_id) * NUM_DEPTH_CONVERSIONS +
                       u32(depth_conversion)) *
                          NUM_PIPELINE_LAYOUTS +
                      u32(is_linear);
    auto& tiling_pipelines = is_tiler ? tilers : detilers;
    if (auto pipeline = *tiling_pipelines[pl_id]; pipeline != VK_NULL_HANDLE) {
        return pipeline;
    }

    const auto device = instance.GetDevice();
    const auto micro_tile_mode = AmdGpu::GetMicroTileMode(info.tile_mode);
    const u32 linear_bits_per_pixel =
        depth_conversion == DepthConversion::None ? info.num_bits : 32;
    std::vector<std::string> defines = {
        fmt::format("BITS_PER_PIXEL={}", info.num_bits),
        fmt::format("LINEAR_BITS_PER_PIXEL={}", linear_bits_per_pixel),
        fmt::format("DEPTH_CONVERSION={}", u32(depth_conversion)),
        fmt::format("NUM_SAMPLES={}", info.num_samples),
        fmt::format("ARRAY_MODE={}", u32(info.array_mode)),
        fmt::format("MICRO_TILE_MODE={}", u32(micro_tile_mode)),
        fmt::format("MICRO_TILE_THICKNESS={}", AmdGpu::GetMicroTileThickness(info.array_mode)),
    };
    if (is_linear) {
        defines.emplace_back("IS_LINEAR=1");
    }
    if (AmdGpu::IsMacroTiled(info.array_mode)) {
        const auto macro_tile_mode =
            AmdGpu::CalculateMacrotileMode(info.tile_mode, info.num_bits, info.num_samples);
        const u32 num_banks = AmdGpu::GetNumBanks(macro_tile_mode);
        defines.emplace_back(
            fmt::format("PIPE_CONFIG={}", u32(AmdGpu::GetPipeConfig(info.tile_mode))));
        defines.emplace_back(fmt::format("BANK_WIDTH={}", AmdGpu::GetBankWidth(macro_tile_mode)));
        defines.emplace_back(fmt::format("BANK_HEIGHT={}", AmdGpu::GetBankHeight(macro_tile_mode)));
        defines.emplace_back(fmt::format("NUM_BANKS={}", num_banks));
        defines.emplace_back(fmt::format("NUM_BANK_BITS={}", std::bit_width(num_banks) - 1));
        defines.emplace_back(fmt::format(
            "TILE_SPLIT_BYTES={}", AmdGpu::CalculateTileSplit(info.tile_mode, info.array_mode,
                                                              micro_tile_mode, info.num_bits)));
        defines.emplace_back(
            fmt::format("MACRO_TILE_ASPECT={}", AmdGpu::GetMacrotileAspect(macro_tile_mode)));
    }
    if (is_tiler) {
        defines.emplace_back(fmt::format("IS_TILER=1"));
    }

    const auto& module = Vulkan::Compile(HostShaders::TILING_COMP,
                                         vk::ShaderStageFlagBits::eCompute, device, defines);
    const auto module_name = fmt::format("{}_{} {}", magic_enum::enum_name(info.tile_mode),
                                         info.num_bits, is_tiler ? "tiler" : "detiler");
    LOG_INFO(Render_Vulkan, "Compiling shader {}", module_name);
    for (const auto& def : defines) {
        LOG_INFO(Render_Vulkan, "#define {}", def);
    }
    Vulkan::SetObjectName(device, module, module_name);
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .stage = shader_ci,
        .layout = *pl_layout,
    };
    auto [result, pipeline] =
        device.createComputePipelineUnique(VK_NULL_HANDLE, compute_pipeline_ci);
    ASSERT_MSG(result == vk::Result::eSuccess, "Detiler pipeline creation failed {}",
               vk::to_string(result));
    tiling_pipelines[pl_id] = std::move(pipeline);
    device.destroyShaderModule(module);
    return *tiling_pipelines[pl_id];
}

std::pair<const Buffer*, u64> TileManager::DetileImage(const VideoCore::Buffer* in_buffer,
                                                       u64 in_offset, const ImageInfo& info,
                                                       const vk::Format host_format) {
    // A promoted depth image (D16 held as D24/D32) also goes through the compute pass when it is
    // linear: the pass widens every texel to the host format's depth plane.
    const auto tiling_format = GetTilingFormat(info, host_format);
    if (!NeedsTilingPipeline(info.props.is_tiled, tiling_format.depth_conversion)) {
        return {in_buffer, in_offset};
    }
    const bool is_linear = !info.props.is_tiled;
    const u64 output_size = GuestToHostBytes(info.guest_size, info.num_bits / 8, tiling_format);

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    params.num_mips = info.resources.levels;
    ASSERT_MSG(params.num_mips <= params.mips.size(), "Detiler has {} mips, maximum is {}",
               params.num_mips, params.mips.size());
    params.num_texels = TexelCount(info, "Detiler");
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max((mip_info.height + 3) / 4, 1U);
        }
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const auto staging = runtime.GetStagingPool().Request(output_size, MemoryType::DeviceLocal,
                                                          instance.StorageMinAlignment());

    scheduler.EndRendering();
    runtime.FlushBarriers();

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute,
                        GetTilingPipeline(info, false, tiling_format.depth_conversion, is_linear));

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = in_buffer->Handle(),
        .offset = in_offset,
        .range = info.guest_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = staging.buffer->Handle(),
        .offset = staging.offset,
        .range = output_size,
    };
    ValidateStorageDescriptor(instance, tiled_buffer_info.offset, tiled_buffer_info.range,
                              "Detiler input");
    ValidateStorageDescriptor(instance, linear_buffer_info.offset, linear_buffer_info.range,
                              "Detiler output");

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);

    const auto [dim_x, dim_y] = ComputeTilingDispatch(instance, params.num_texels, "Detiler");
    cmdbuf.dispatch(dim_x, dim_y, 1);

    runtime.AccessBuffer(staging.buffer, staging.offset, output_size,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderWrite);

    return {staging.buffer, staging.offset};
}

void TileManager::TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                            const VideoCore::Buffer* out_buffer, u64 out_offset) {
    const auto& info = in_image.info;
    const auto tiling_format = GetTilingFormat(info, in_image.GetImageFormat());
    if (!NeedsTilingPipeline(info.props.is_tiled, tiling_format.depth_conversion)) {
        for (auto& copy : buffer_copies) {
            copy.bufferOffset += out_offset;
        }
        runtime.DownloadImage(&in_image, out_buffer, buffer_copies);
        return;
    }
    // Promoted depth is read back in its host texel size and narrowed by the compute pass.
    const bool is_linear = !info.props.is_tiled;
    const u32 guest_bytes_per_pixel = info.num_bits / 8;
    const u64 host_size = GuestToHostBytes(info.guest_size, guest_bytes_per_pixel, tiling_format);

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    params.num_mips = static_cast<u32>(buffer_copies.size());
    ASSERT_MSG(params.num_mips <= params.mips.size(), "Tiler has {} mips, maximum is {}",
               params.num_mips, params.mips.size());
    params.num_texels = TexelCount(info, "Tiler");
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max((mip_info.height + 3) / 4, 1U);
        }
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const auto staging = runtime.GetStagingPool().Request(host_size, MemoryType::DeviceLocal,
                                                          instance.StorageMinAlignment());
    for (auto& copy : buffer_copies) {
        copy.bufferOffset =
            GuestToHostBytes(copy.bufferOffset, guest_bytes_per_pixel, tiling_format) +
            staging.offset;
    }

    runtime.DownloadImage(&in_image, staging.buffer, buffer_copies);
    runtime.FlushBarriers();

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute,
                        GetTilingPipeline(info, true, tiling_format.depth_conversion, is_linear));

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = out_buffer->Handle(),
        .offset = out_offset,
        .range = info.guest_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = staging.buffer->Handle(),
        .offset = staging.offset,
        .range = host_size,
    };
    ValidateStorageDescriptor(instance, tiled_buffer_info.offset, tiled_buffer_info.range,
                              "Tiler output");
    ValidateStorageDescriptor(instance, linear_buffer_info.offset, linear_buffer_info.range,
                              "Tiler input");

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);

    const auto [dim_x, dim_y] = ComputeTilingDispatch(instance, params.num_texels, "Tiler");
    cmdbuf.dispatch(dim_x, dim_y, 1);

    runtime.AccessBuffer(out_buffer, out_offset, info.guest_size,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderWrite);
}

} // namespace VideoCore
