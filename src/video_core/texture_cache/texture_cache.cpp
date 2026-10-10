// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#include <xxhash.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/hash.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

#include <thread>
#include "video_core/amdgpu/liverpool.h"
#include "video_core/texture_cache/host_write_trace.h"

namespace VideoCore {

static constexpr u32 MAX_IMAGES = std::numeric_limits<u16>::max();
static constexpr u32 MAX_IMAGE_VIEWS = std::numeric_limits<u16>::max();
static constexpr u32 MAX_SAMPLERS = std::numeric_limits<u16>::max();

static void ConvertDepthDownloadToGuest(u8* data, const u64 num_texels,
                                        const DepthConversion conversion) {
    if (conversion == DepthConversion::None) {
        return;
    }

    for (u64 texel = 0; texel < num_texels; ++texel) {
        u32 host_value;
        std::memcpy(&host_value, data + texel * sizeof(host_value), sizeof(host_value));

        u16 guest_value;
        if (conversion == DepthConversion::D16ToD24) {
            guest_value = D24ToD16(host_value);
        } else {
            const float depth = std::bit_cast<float>(host_value);
            const float clamped_depth = std::isnan(depth) ? 0.0f : std::clamp(depth, 0.0f, 1.0f);
            guest_value = static_cast<u16>(
                std::lround(clamped_depth * static_cast<float>(std::numeric_limits<u16>::max())));
        }
        std::memcpy(data + texel * sizeof(guest_value), &guest_value, sizeof(guest_value));
    }
}

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                           BufferCache& buffer_cache_, PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, slot_images{MAX_IMAGES},
      slot_image_views{MAX_IMAGE_VIEWS}, slot_samplers{MAX_SAMPLERS},
      blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, runtime, buffer_cache.GetStreamBuffer()},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()} {

    if (texture_trace) {
        HostWriteTrace::Register(
            [](void* self, std::string_view what, VAddr addr, VAddr ctx_addr, u64 ctx_size,
               std::span<const u8> before, std::span<const u8> after) {
                static_cast<TextureCache*>(self)->OnHostWrite(what, addr, ctx_addr, ctx_size,
                                                              before, after);
            },
            this);
    }

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    const s64 min_pressure_floor =
        std::clamp<s64>(device_local_memory / 4, 256_MB, DEFAULT_PRESSURE_GC_MEMORY);
    const s64 min_critical_floor =
        std::clamp<s64>(device_local_memory / 2, 512_MB, DEFAULT_CRITICAL_GC_MEMORY);
    pressure_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      min_pressure_floor));
    critical_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      min_critical_floor));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
}

TextureCache::~TextureCache() {
    HostWriteTrace::Unregister(this);
}

void TextureCache::OnHostWrite(std::string_view what, VAddr addr, VAddr ctx_addr, u64 ctx_size,
                               std::span<const u8> before, std::span<const u8> after) {
    if (!TraceOpen()) {
        return;
    }
    constexpr u64 HostWriteSummaryTicks = 1000;
    const u64 size = before.size();
    u64 changed = 0;
    for (u64 i = 0; i < size; ++i) {
        changed += before[i] != after[i];
    }
    const u64 tick = scheduler.CurrentTick();
    const std::string context =
        ctx_size ? fmt::format(" for {:#x}+{:#x}", ctx_addr, ctx_size) : std::string{};
    std::string summary;
    {
        std::scoped_lock lk{trace_mutex};
        auto& stats = host_write_stats[std::string{what}];
        ++stats.writes;
        stats.bytes += changed;
        auto& site = host_write_sites[addr];
        if (site.count++ == 0) {
            site.first_tick = tick;
        }
        site.kind = what;
        site.size = std::max(site.size, size);
        site.last_tick = tick;
        site.last_value = 0;
        std::memcpy(&site.last_value, after.data(), std::min<u64>(size, sizeof(u64)));
        host_write_max_size = std::max(host_write_max_size, size);
        if (tick - host_write_last_summary >= HostWriteSummaryTicks) {
            host_write_last_summary = tick;
            for (const auto& [name, entry] : host_write_stats) {
                fmt::format_to(std::back_inserter(summary),
                               "{}{}: {} writes, {} bytes changed, {} changed cached images",
                               summary.empty() ? "" : "; ", name, entry.writes, entry.bytes,
                               entry.image_writes);
            }
            fmt::format_to(std::back_inserter(summary), "; {} write sites remembered",
                           host_write_sites.size());
        }
    }
    if (!summary.empty()) {
        LOG_INFO(Render_Vulkan, "TexTrace t={} host writes summary: {}", tick, summary);
    }

    // The page table of the cache belongs to the GPU thread.
    if (!liverpool || std::this_thread::get_id() != liverpool->GetGpuCommandProcessorThread()) {
        LOG_INFO(Render_Vulkan,
                 "TexTrace t={} host write {} at {:#x}+{:#x}{}: {} bytes changed (not on the GPU "
                 "thread, cached images not looked up) w={}",
                 tick, what, addr, size, context, changed, TraceWindow::Current());
        return;
    }
    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(addr, size,
                         [&](ImageId image_id, Image&) { image_ids.push_back(image_id); });
    for (const ImageId image_id : image_ids) {
        const Image& image = slot_images[image_id];
        const VAddr image_begin = image.info.guest_address;
        const VAddr begin = std::max<VAddr>(addr, image_begin);
        const VAddr end = std::min<VAddr>(addr + size, image_begin + image.info.guest_size);
        u64 image_changed = 0;
        u64 first = 0;
        u64 last = 0;
        for (VAddr a = begin; a < end; ++a) {
            if (before[a - addr] != after[a - addr]) {
                if (image_changed == 0) {
                    first = a - image_begin;
                }
                last = a - image_begin;
                ++image_changed;
            }
        }
        if (image_changed == 0) {
            continue;
        }
        {
            std::scoped_lock lk{trace_mutex};
            ++host_write_stats[std::string{what}].image_writes;
        }
        TraceImage(image, "host write",
                   fmt::format(" {} at {:#x}+{:#x}{}: {} bytes of the image changed at "
                               "+{:#x}..+{:#x}",
                               what, addr, size, context, image_changed, first, last));
    }
}

void TextureCache::ReportHostWritesInImage(const Image& image) {
    const VAddr begin = image.info.guest_address;
    const VAddr end = begin + image.info.guest_size;
    std::vector<std::pair<VAddr, HostWriteSite>> sites;
    {
        std::scoped_lock lk{trace_mutex};
        const VAddr from = begin > host_write_max_size ? begin - host_write_max_size : 0;
        for (auto it = host_write_sites.lower_bound(from);
             it != host_write_sites.end() && it->first < end; ++it) {
            if (it->first + it->second.size > begin) {
                sites.emplace_back(it->first, it->second);
            }
        }
    }
    if (sites.empty()) {
        return;
    }
    std::ranges::sort(sites, [](const auto& a, const auto& b) {
        return a.second.last_tick > b.second.last_tick;
    });
    constexpr size_t MaxListed = 8;
    std::string list;
    for (size_t i = 0; i < std::min(sites.size(), MaxListed); ++i) {
        const auto& [addr, site] = sites[i];
        fmt::format_to(
            std::back_inserter(list), "{}{} at {:#x}+{:#x} (image {:+#x}) x{} t={}..{} last {:#x}",
            list.empty() ? "" : "; ", site.kind, addr, site.size, static_cast<s64>(addr - begin),
            site.count, site.first_tick, site.last_tick, site.last_value);
    }
    TraceImage(image, "host writes in range",
               fmt::format(" {} site(s), most recent first: {}{}", sites.size(), list,
                           sites.size() > MaxListed ? "; ..." : ""));
}

void TextureCache::ProcessDownloadImages() {
    std::unique_lock lk{download_images_mutex};
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, true);
    }
    download_images.clear();
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    const u32 guest_bytes_per_pixel = image.info.num_bits / 8;
    const u64 num_texels = u64{image.info.pitch} * image.info.size.height * image.info.size.depth *
                           image.info.resources.layers;
    const u64 download_size = num_texels * guest_bytes_per_pixel;
    ASSERT(download_size <= image.info.guest_size);
    ASSERT_MSG(image.info.num_samples == 1,
               "Image readback requires a single-sample source, got {} samples",
               image.info.num_samples);
    // A promoted depth image (D16 held as D24/D32) is read back in its host texel size and
    // narrowed in place before it reaches guest memory.
    const auto tiling_format = GetTilingFormat(image.info, image.GetImageFormat());
    const u64 host_download_size =
        GuestToHostBytes(download_size, guest_bytes_per_pixel, tiling_format);
    const auto depth_conversion = tiling_format.depth_conversion;
    const auto download =
        runtime.GetStagingPool().Request(host_download_size, MemoryType::HostCached, 16, !sync);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    if (sync) {
        scheduler.Finish();
        download.Invalidate();
        ConvertDepthDownloadToGuest(download.mapped, num_texels, depth_conversion);
        const HostWriteTrace::Label trace_label{"image readback"};
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image.info.guest_address),
                                                  download.mapped, download_size);
    } else {
        scheduler.DeferPriorityOperation([this, device_addr = image.info.guest_address, download,
                                          download_size, num_texels, depth_conversion] {
            download.Invalidate();
            ConvertDepthDownloadToGuest(download.mapped, num_texels, depth_conversion);
            const HostWriteTrace::Label trace_label{"image readback"};
            Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr),
                                                      download.mapped, download_size);
            runtime.GetStagingPool().FreeDeferred(download);
        });
    }
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);

    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(pages_start, pages_end - pages_start,
                         [&](ImageId image_id, Image&) { image_ids.push_back(image_id); });

    for (const auto image_id : image_ids) {
        Image& image = slot_images[image_id];
        std::scoped_lock lk{image.mutex};

        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
            if (TraceOpen()) {
                TraceImage(image, "cpu write", fmt::format(" at {:#x}+{:#x}", addr, size));
            }
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
            if (TraceOpen()) {
                TraceImage(image, "untrack head", fmt::format(" at {:#x}+{:#x}", addr, size));
            }
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
            if (TraceOpen()) {
                TraceImage(image, "untrack tail", fmt::format(" at {:#x}+{:#x}", addr, size));
            }
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
            if (TraceOpen()) {
                TraceImage(image, "maybe dirty", fmt::format(" at {:#x}+{:#x}", addr, size));
            }
        }
    }
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            if (TraceOpen()) {
                // Diagnostics only: the write covers part of this image but leaves it unmarked.
                TraceImage(image, "gpu write (not marked)",
                           fmt::format(" at {:#x}+{:#x}", address, max_size));
            }
            return;
        }
        // Ensure image is reuploaded when accessed again.
        if (TraceOpen()) {
            TraceImage(image, "gpu write", fmt::format(" at {:#x}+{:#x}", address, max_size));
        }
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    SmallVector<ImageId, 16> deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    if (TraceOpen() && !deleted_images.empty()) {
        LOG_INFO(Render_Vulkan, "TexTrace t={} unmap {:#x}+{:#x} frees {} image(s)",
                 scheduler.CurrentTick(), cpu_addr, size, deleted_images.size());
    }
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        new_info.UpdateSize();
        const auto new_image_id = slot_images.Insert(instance, runtime, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        if (new_info.num_samples == 1 &&
            (new_info.resources.layers > cache_image.info.resources.layers ||
             new_info.resources.levels > cache_image.info.resources.levels)) {
            RefreshImage(new_image);
        }
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;
        runtime.CopyColorAndDepth(&cache_image, &new_image);
        if (texture_trace) {
            TraceCopy(new_image, cache_image, "copy (depth overlap)");
        }

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    static constexpr u64 NUM_FRAMES_BEFORE_REMOVAL = 32;

    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NUM_FRAMES_BEFORE_REMOVAL;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool pow2_padding_only =
            image_info.props.is_pow2 != cache_image.info.props.is_pow2 &&
            image_info.tile_mode == cache_image.info.tile_mode &&
            image_info.size == cache_image.info.size &&
            image_info.pitch == cache_image.info.pitch && image_info.resources.levels == 1 &&
            cache_image.info.resources.levels == 1 && image_info.resources.layers == 1 &&
            cache_image.info.resources.layers == 1;

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size || pow2_padding_only) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Same address, same block layout, but the overlap fits neither a view nor an expansion
        // (for example a cube map placed where a 2D image with mips used to live after the guest
        // reused that memory). Treat it like the pool-allocation case above: drop the stale image
        // and let the caller create a new one instead of crashing.
        LOG_WARNING(Render_Vulkan,
                    "Unresolvable image overlap at {:#x}, recreating: cached {}x{} mips {} layers "
                    "{} ({:#x} bytes) vs requested {}x{} mips {} layers {} ({:#x} bytes)",
                    image_info.guest_address, cache_image.info.size.width,
                    cache_image.info.size.height, cache_image.info.resources.levels,
                    cache_image.info.resources.layers, cache_image.info.guest_size,
                    image_info.size.width, image_info.size.height, image_info.resources.levels,
                    image_info.resources.layers, image_info.guest_size);
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }
        return {merged_image_id, -1, -1};
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    runtime.CopyMip(&cache_image, &merged_image, mip, slice);
                    if (texture_trace) {
                        TraceCopy(merged_image, cache_image,
                                  fmt::format("copy into mip {} layer {}", mip, slice));
                    }
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id = slot_images.Insert(instance, runtime, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    // The new image now holds what guest memory holds. The old image's contents only take
    // precedence when they are newer than memory, i.e. rendered by the GPU and not overwritten
    // since. A dirty old image is stale: memory at this address may already hold a different
    // texture (a pool allocator placing a larger texture where a smaller one lived), and copying
    // it would put the previous texture into the new one's first mips until it is reloaded.
    if (True(src_image.flags & ImageFlagBits::GpuModified) &&
        False(src_image.flags & ImageFlagBits::Dirty)) {
        runtime.CopyImage(&src_image, &new_image);
        if (texture_trace) {
            TraceCopy(new_image, src_image, "copy (expand)");
        }
    }

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    if (TraceOpen()) {
        TraceImage(new_image, "expanded",
                   fmt::format(" from uid={} {:#x}+{:#x} mips {} flags {:#x}", src_image.image_uid,
                               src_image.info.guest_address, src_image.info.guest_size,
                               src_image.info.resources.levels, u32(src_image.flags)));
    }
    FreeImage(image_id);
    TrackImage(new_image_id);
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        image_id = slot_images.Insert(instance, runtime, slot_image_views, info);
        RegisterImage(image_id);
        if (TraceOpen()) {
            TraceImage(slot_images[image_id], "new");
        }
    } else if (TraceOpen()) {
        const auto& found = slot_images[image_id].info;
        if (found.guest_address != info.guest_address || found.pixel_format != info.pixel_format ||
            found.tile_mode != info.tile_mode || found.guest_size != info.guest_size) {
            TraceImage(slot_images[image_id], "reused for",
                       fmt::format(" {:#x}+{:#x} {}x{} {} tile {}", info.guest_address,
                                   info.guest_size, info.size.width, info.size.height,
                                   vk::to_string(info.pixel_format), u32(info.tile_mode)));
        }
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    SmallVector<ImageId, 4> image_ids;
    ForEachImageWithAddress(address, [&](ImageId image_id, Image& image) {
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.flags |= ImageFlagBits::GpuModified;
        if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);
    if (texture_trace && desc.type == BindingType::Storage) {
        MarkTraceRendered(image);
    }
    return image.FindView(desc.view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);
    if (texture_trace) {
        MarkTraceRendered(image);
    }

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaType::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaType::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    UpdateImage(image_id);
    if (texture_trace) {
        MarkTraceRendered(image);
    }

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaType::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    AssociateStencil(image_id, desc.info);

    return image.FindView(desc.view_info, false);
}

void TextureCache::AssociateStencil(ImageId depth_id, const ImageInfo& depth_info) {
    if (depth_info.stencil_addr == 0) {
        return;
    }
    ImageId stencil_id{};
    ForEachImageInRegion(
        depth_info.stencil_addr, depth_info.stencil_size, [&](ImageId image_id, Image& image) {
            if (image.info.guest_address != depth_info.stencil_addr) {
                return;
            }
            if (image.info.pixel_format == vk::Format::eUndefined ||
                Vulkan::LiverpoolToVK::IsFormatStencilCompatible(image.info.pixel_format)) {
                stencil_id = image_id;
            }
        });
    if (!stencil_id) {
        ImageInfo info{};
        info.guest_address = depth_info.stencil_addr;
        info.guest_size = depth_info.stencil_size;
        info.size = depth_info.size;
        stencil_id = slot_images.Insert(instance, runtime, slot_image_views, info);
        RegisterImage(stencil_id);
    }
    Image& stencil_image = slot_images[stencil_id];
    TouchImage(stencil_image);
    stencil_image.AssociateDepth(depth_id, slot_images[depth_id].image_uid);
}

void TextureCache::RefreshImage(Image& image) {
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }
    // Texture trace: hash of what guest memory holds now. No upload line after the refresh line
    // means the refresh was skipped because the contents looked unchanged.
    const u64 trace_mem_hash =
        texture_trace
            ? XXH3_64bits(std::bit_cast<const u8*>(image.info.guest_address), image.info.guest_size)
            : 0;
    if (TraceOpen()) {
        TraceImage(image, "refresh", fmt::format(" mem={:016x}", trace_mem_hash));
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    // An image only becomes MaybeCpuDirty when every page it touches was unprotected by a write
    // next to it, so it spans at most two tracker pages. Once those pages are unprotected the image
    // itself can be rewritten without any fault, and the hash is the only thing that notices.
    // It has to cover the whole image: a new sprite placed where an old one lived usually shares
    // the old one's transparent top-left corner, so a partial hash keeps the stale contents.
    const auto addr = std::bit_cast<const u8*>(image.info.guest_address);
    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & (ImageFlagBits::CpuDirty | ImageFlagBits::GpuDirty))) {
        const u64 hash = XXH3_64bits(addr, image.info.guest_size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
    }
    // Remember what guest memory held when these contents were taken, so a later
    // MaybeCpuDirty check compares against the uploaded data and not an older version.
    // GPU-dirty contents come from the buffer cache, not guest memory: hash lazily instead.
    // Page watchers work in PageManager pages: two of them bound what can be MaybeCpuDirty.
    constexpr u64 MaybeDirtyMaxSize = 2 * PageManager::GetNextPageAddr(0);
    image.hash =
        image.info.guest_size <= MaybeDirtyMaxSize && False(image.flags & ImageFlagBits::GpuDirty)
            ? XXH3_64bits(addr, image.info.guest_size)
            : 0;

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);
    const auto tiling_format = GetTilingFormat(image.info, image.GetImageFormat());
    const u32 guest_bytes_per_pixel = image.info.num_bits / 8;

    SmallVector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];
        const u32 extent_width = mip_pitch ? std::min<u32>(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min<u32>(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = GuestToHostBytes(mip_offset, guest_bytes_per_pixel, tiling_format),
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        if (TraceOpen()) {
            TraceImage(image, "refresh skipped (gpu contents kept)");
        }
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }
    if (texture_trace) {
        // Record what the image now holds. ObtainBufferForImage below takes the data from the
        // buffer cache instead of guest memory when any part of the range was written by the GPU,
        // and then the memory hash does not describe the uploaded data.
        const bool from_gpu_buffer =
            buffer_cache.IsRegionGpuModified(image.info.guest_address, image.info.guest_size);
        std::string content = fmt::format("{:016x} from {}", trace_mem_hash,
                                          from_gpu_buffer ? "gpu-buffer" : "memory");
        if (image_copies.size() < num_mips) {
            content += fmt::format(" mips {}/{}", image_copies.size(), num_mips);
        }
        if (TraceOpen()) {
            TraceImage(image, "upload", " content " + content);
            ReportHostWritesInImage(image);
        }
        SetTraceContent(image, std::move(content));
        SetTraceUpload(image, trace_mem_hash, !from_gpu_buffer && image_copies.size() == num_mips);
    }

    scheduler.EndRendering();

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    const auto [buffer, offset] =
        tile_manager.DetileImage(in_buffer, in_offset, image.info, image.GetImageFormat());
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    runtime.UploadImage(&image, buffer, image_copies);
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sharp,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     const bool is_depth) {
    // Compare and plain uses of one S# need separate samplers.
    const u64 hash = HashCombine(XXH3_64bits(&sharp, sizeof(sharp)), is_depth);
    const auto [it, new_sampler] = samplers.try_emplace(hash);
    if (new_sampler) {
        it->second = slot_samplers.Insert(instance, sharp, border_color_base, is_depth);
        Sampler& sampler = slot_samplers[it->second];
        sampler.hash = hash;
        sampler_lru_cache.Insert(sampler, gc_tick);
    }

    Sampler& sampler = slot_samplers[it->second];
    sampler_lru_cache.Touch(sampler, gc_tick);
    return sampler.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image_lru_cache.Insert(image, gc_tick);
    const auto& info = image.info;
    ASSERT_MSG((info.guest_address & 0xff) == 0, "Trying to register an unaligned image");
    ForEachPage(info.guest_address, info.guest_size, [this, image_id, info](u64 page) {
        page_table[page].entries.emplace_back(BucketEntry{
            .key = u32(info.guest_address >> 8),
            .size = info.guest_size,
            .id = image_id,
        });
    });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    image_lru_cache.Free(image);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        ASSERT_MSG(page_it, "Unregistering unregistered page={:#x}", page << Traits::PAGE_BITS);
        auto& entries = page_it->entries;
        const auto vector_it = std::ranges::find(entries, image_id, &BucketEntry::id);
        ASSERT_MSG(vector_it != entries.end(), "Unregistering unregistered image in page={:#x}",
                   page << Traits::PAGE_BITS);
        entries.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    std::scoped_lock lk{image.mutex};
    if (image.IsUntracked()) {
        tracker.UpdatePageWatchers(image_begin, image.info.guest_size, PageOp::Track);
    } else {
        if (image_begin < image.track_addr) {
            tracker.UpdatePageWatchers(image_begin, image.track_addr - image_begin, PageOp::Track);
        }
        if (image.track_addr_end < image_end) {
            tracker.UpdatePageWatchers(image.track_addr_end, image_end - image.track_addr_end,
                                       PageOp::Track);
        }
    }
    image.track_addr = image_begin;
    image.track_addr_end = image_end;
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (image.IsUntracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    if (size != 0) {
        tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
    }
    image.track_addr = 0;
    image.track_addr_end = 0;
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (image.IsUntracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    tracker.UpdatePageWatchers(image_begin, size, PageOp::Untrack);

    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image.IsUntracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
}

void TextureCache::GarbageCollectImages() {
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](Image& image) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        if (tiled && download) {
            // This is a workaround for now. We can't handle non-linear image downloads.
            return false;
        }
        if (download && !pressured) {
            return false;
        }
        const auto image_id = slot_images.GetSlotId(image);
        if (download) {
            DownloadImageMemory(image_id);
        }
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    image_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        image_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](Sampler& sampler) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        sampler_lru_cache.Free(sampler);
        samplers.erase(sampler.hash);
        const auto sampler_id = slot_samplers.GetSlotId(sampler);
        slot_samplers.Erase(sampler_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    GarbageCollectImages();
    GarbageCollectSamplers();
    ++gc_tick;
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(image.IsUntracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.Erase(image_view_id);
            }
        }
        slot_images.Erase(image_id);
    });
}

bool TextureCache::IsTextureTraceEnabled() {
    const bool enabled = EmulatorSettings.IsTextureTrace();
    if (enabled) {
        TraceWindow::Configure(EmulatorSettings.GetTextureTraceTrigger(),
                               EmulatorSettings.GetTextureTraceSeconds());
    }
    return enabled;
}

void TextureCache::TraceImage(const Image& image, std::string_view event, std::string_view detail) {
    if (event.starts_with("untrack") || event.starts_with("gpu write") || event == "cpu write" ||
        event == "maybe dirty") {
        // Stale-texture check: the last few ways this image's memory stopped being watched or
        // was written, printed with a stale report.
        constexpr size_t MaxHistory = 4;
        std::scoped_lock lk{trace_mutex};
        auto& history = trace_states[image.image_uid].history;
        if (history.size() == MaxHistory) {
            history.erase(history.begin());
        }
        history.push_back(fmt::format("t={} {}{}", scheduler.CurrentTick(), event, detail));
    }
    // Every event while a window is open, with no per-image limit, so that each window holds the
    // complete story of the textures it uses and windows can be compared with each other.
    const auto& info = image.info;
    LOG_INFO(Render_Vulkan,
             "TexTrace t={} {} uid={} {:#x}+{:#x} {}x{}x{} {} tile {} mips {} layers {} "
             "flags {:#x}{} w={}",
             scheduler.CurrentTick(), event, image.image_uid, info.guest_address, info.guest_size,
             info.size.width, info.size.height, info.size.depth, vk::to_string(info.pixel_format),
             u32(info.tile_mode), info.resources.levels, info.resources.layers, u32(image.flags),
             detail, TraceWindow::Current());
}

void TextureCache::SetTraceContent(const Image& image, std::string content) {
    std::scoped_lock lk{trace_mutex};
    auto& state = trace_states[image.image_uid];
    state.content = std::move(content);
    state.rendered = false;
}

void TextureCache::MarkTraceRendered(const Image& image) {
    std::scoped_lock lk{trace_mutex};
    trace_states[image.image_uid].rendered = true;
}

void TextureCache::SetTraceUpload(const Image& image, u64 mem_hash, bool from_memory) {
    std::scoped_lock lk{trace_mutex};
    auto& state = trace_states[image.image_uid];
    state.from_memory = from_memory;
    state.upload_hash = mem_hash;
    state.upload_tick = scheduler.CurrentTick();
    state.reported_hash = 0;
    state.reported_gpu = false;
}

void TextureCache::CheckStaleTextures(std::span<const TraceBinding> bindings) {
    // Diagnostics only. A texture the cache treats as current (no Dirty flag, never written by the
    // GPU, filled entirely from guest memory) must still match guest memory; if it does not, a
    // write reached that memory without invalidating the image and the GPU shows old contents.
    constexpr u64 CheckIntervalTicks = 30;
    const u64 tick = scheduler.CurrentTick();

    struct Candidate {
        ImageId image_id;
        u64 upload_hash;
    };
    std::vector<Candidate> candidates;
    {
        std::scoped_lock lk{trace_mutex};
        for (const auto& binding : bindings) {
            if (binding.kind == "rt") {
                continue;
            }
            const Image& image = slot_images[binding.image_id];
            if (image.info.guest_address == 0 || image.info.num_samples > 1 ||
                True(image.flags & (ImageFlagBits::Dirty | ImageFlagBits::GpuModified))) {
                continue;
            }
            const auto it = trace_states.find(image.image_uid);
            if (it == trace_states.end()) {
                continue;
            }
            auto& state = it->second;
            if (!state.from_memory || state.rendered ||
                (state.last_check_tick != 0 && tick - state.last_check_tick < CheckIntervalTicks)) {
                continue;
            }
            state.last_check_tick = tick;
            candidates.push_back({binding.image_id, state.upload_hash});
        }
    }

    // Guest memory is hashed without trace_mutex held: a read can fault into the page tracker,
    // which logs through TraceImage and takes that lock.
    for (const auto& candidate : candidates) {
        const Image& image = slot_images[candidate.image_id];
        const VAddr address = image.info.guest_address;
        const u64 size = image.info.guest_size;
        const bool gpu_range = buffer_cache.IsRegionGpuModified(address, size);
        const u64 now_hash = gpu_range ? 0 : XXH3_64bits(std::bit_cast<const u8*>(address), size);
        if (!gpu_range && now_hash == candidate.upload_hash) {
            continue;
        }
        // A write caught by the tracker while hashing sets a Dirty flag before memory changes.
        if (True(image.flags & ImageFlagBits::Dirty)) {
            continue;
        }
        std::string detail;
        {
            std::scoped_lock lk{trace_mutex};
            auto& state = trace_states[image.image_uid];
            if (gpu_range ? state.reported_gpu : state.reported_hash == now_hash) {
                continue;
            }
            std::string history;
            for (const auto& entry : state.history) {
                fmt::format_to(std::back_inserter(history), "{}{}", history.empty() ? "" : "; ",
                               entry);
            }
            if (gpu_range) {
                state.reported_gpu = true;
                detail = fmt::format(" range written by the GPU (buffer cache), image not "
                                     "GpuDirty; uploaded {:016x} at t={}; history: {}",
                                     state.upload_hash, state.upload_tick,
                                     history.empty() ? "none" : history);
            } else {
                state.reported_hash = now_hash;
                detail = fmt::format(" uploaded {:016x} at t={} now {:016x}; history: {}",
                                     state.upload_hash, state.upload_tick, now_hash,
                                     history.empty() ? "none" : history);
            }
        }
        TraceImage(image, gpu_range ? "stale? (gpu)" : "stale", detail);
    }
}

void TextureCache::TraceCopy(const Image& dst, const Image& src, std::string_view what) {
    std::string content;
    {
        std::scoped_lock lk{trace_mutex};
        // Copy the source state: inserting the destination may rehash the map.
        const TraceState src_state = trace_states[src.image_uid];
        auto& dst_state = trace_states[dst.image_uid];
        dst_state.content += fmt::format(" + {} of uid {} ({}{})", what, src.image_uid,
                                         src_state.content.empty() ? "none" : src_state.content,
                                         src_state.rendered ? ", rendered" : "");
        dst_state.rendered |= src_state.rendered;
        dst_state.from_memory = false;
        content = dst_state.content;
    }
    if (TraceOpen()) {
        TraceImage(dst, what,
                   fmt::format(" from uid={} {:#x}+{:#x} flags {:#x} content {}", src.image_uid,
                               src.info.guest_address, src.info.guest_size, u32(src.flags),
                               content));
    }
}

static std::string_view SwizzleName(const AmdGpu::CompSwizzle swizzle) {
    switch (swizzle) {
    case AmdGpu::CompSwizzle::Zero:
        return "0";
    case AmdGpu::CompSwizzle::One:
        return "1";
    case AmdGpu::CompSwizzle::Red:
        return "R";
    case AmdGpu::CompSwizzle::Green:
        return "G";
    case AmdGpu::CompSwizzle::Blue:
        return "B";
    case AmdGpu::CompSwizzle::Alpha:
        return "A";
    default:
        return "?";
    }
}

void TextureCache::TraceDraw(u64 pipeline_hash, std::span<const TraceBinding> bindings) {
    const u32 window = TraceWindow::Current();
    if (!texture_trace || window == 0 || bindings.empty()) {
        return;
    }
    CheckStaleTextures(bindings);
    // One line per distinct draw (pipeline plus every image it reads or renders to, with the
    // contents each image holds) per window. The same draw is listed again in the next window, so
    // windows can be compared directly.
    std::string line;
    {
        std::scoped_lock lk{trace_mutex};
        for (const auto& binding : bindings) {
            const Image& image = slot_images[binding.image_id];
            const auto& info = image.info;
            const auto& view = binding.view;
            const auto& state = trace_states[image.image_uid];
            fmt::format_to(
                std::back_inserter(line),
                " | {}{} uid={} {:#x}+{:#x} {}x{} {} mips {} view {} {},{},{},{} mips "
                "{}+{} content {}{}",
                binding.kind, binding.slot, image.image_uid, info.guest_address, info.guest_size,
                info.size.width, info.size.height, vk::to_string(info.pixel_format),
                info.resources.levels, vk::to_string(view.format), SwizzleName(view.mapping.r),
                SwizzleName(view.mapping.g), SwizzleName(view.mapping.b),
                SwizzleName(view.mapping.a), view.range.base.level, view.range.extent.levels,
                state.content.empty() ? "none" : state.content, state.rendered ? " rendered" : "");
        }
        if (trace_draw_window != window) {
            trace_draw_window = window;
            trace_draws_seen.clear();
        }
        const u64 key = HashCombine(pipeline_hash, XXH3_64bits(line.data(), line.size()));
        if (!trace_draws_seen.insert(key).second) {
            return;
        }
    }
    LOG_INFO(Render_Vulkan, "TexTrace t={} draw w={} pl={:016x}{}", scheduler.CurrentTick(), window,
             pipeline_hash, line);
}

} // namespace VideoCore
