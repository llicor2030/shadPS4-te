// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ajm_aac.h"
#include "ajm_at9.h"
#include "ajm_instance.h"
#include "ajm_mp3.h"
#include "ajm_result.h"

#include <magic_enum/magic_enum.hpp>

#include "core/memory.h"

namespace Libraries::Ajm {

// AJMDBG: FNV-1a 64, used to fingerprint each 4 KiB sub-block of the ring.
static u64 AjmRingHash64(const u8* p, u64 size) {
    u64 h = 0xcbf29ce484222325ULL;
    for (u64 i = 0; i < size; i++) {
        h = (h ^ static_cast<u64>(p[i])) * 0x100000001b3ULL;
    }
    return h;
}

u8 GetPCMSize(AjmFormatEncoding format) {
    switch (format) {
    case AjmFormatEncoding::S16:
        return sizeof(s16);
    case AjmFormatEncoding::S32:
        return sizeof(s32);
    case AjmFormatEncoding::Float:
        return sizeof(float);
    default:
        UNREACHABLE();
    }
}

AjmInstance::AjmInstance(AjmCodecType codec_type, AjmInstanceFlags flags) : m_flags(flags) {
    switch (codec_type) {
    case AjmCodecType::At9Dec: {
        m_codec = std::make_unique<AjmAt9Decoder>(
            AjmFormatEncoding(flags.format), AjmAt9CodecFlags(flags.codec), u32(flags.channels));
        break;
    }
    case AjmCodecType::Mp3Dec: {
        m_codec = std::make_unique<AjmMp3Decoder>(
            AjmFormatEncoding(flags.format), AjmMp3CodecFlags(flags.codec), u32(flags.channels));
        break;
    }
    case AjmCodecType::M4aacDec: {
        m_codec = std::make_unique<AjmAacDecoder>(
            AjmFormatEncoding(flags.format), AjmAacCodecFlags(flags.codec), u32(flags.channels));
        break;
    }
    default:
        UNREACHABLE_MSG("Unimplemented codec type {}", magic_enum::enum_name(codec_type));
    }
}

void AjmInstance::DbgDump(u32 instance_id) const {
    // AJMDBG: the guest tore this voice down. Everything the emulator knew
    // about it at that moment, in one line.
    LOG_INFO(Lib_Ajm,
             "[AJMDBG] DESTROY inst={:#x} jobs={} intotal={} total={} ringbase={:#x} flags={:#x}",
             instance_id, m_dbg_jobs_total, m_dbg_in_total, m_total_samples, m_dbg_ring_base,
             m_flags.raw);
}

void AjmInstance::Reset() {
    m_total_samples = 0;
    m_gapless.Reset();
    m_codec->Reset();
    // AJMDBG: fold the bookkeeping per stream as well.
    m_dbg_in_total = 0;
    m_dbg_ring_base = 0;
    m_dbg_jobs = 0;
    m_dbg_ring_hash.fill(0);
}

void AjmInstance::ExecuteJob(AjmJob& job) {
    const auto control_flags = job.flags.control_flags;
    job.output.p_result->result = 0;
    if (True(control_flags & AjmJobControlFlags::Reset)) {
        LOG_INFO(Lib_Ajm, "[AJMDBG] RESET inst={:#x} (was total={} intotal={} ringbase={:#x})",
                 job.instance_id, m_total_samples, m_dbg_in_total, m_dbg_ring_base);
        Reset();
    }
    if (job.input.init_params.has_value()) {
        LOG_INFO(Lib_Ajm, "[AJMDBG] INIT  inst={:#x}", job.instance_id);
        auto& params = job.input.init_params.value();
        m_codec->Initialize(&params, sizeof(params));
    }
    if (job.input.resample_parameters.has_value()) {
        LOG_ERROR(Lib_Ajm, "Unimplemented: resample parameters");
        m_resample_parameters = job.input.resample_parameters.value();
    }
    if (job.input.format.has_value()) {
        LOG_ERROR(Lib_Ajm, "Unimplemented: format parameters");
        m_format = job.input.format.value();
    }
    if (job.input.gapless_decode.has_value()) {
        auto& params = job.input.gapless_decode.value();

        const auto samples_processed =
            m_gapless.init.total_samples - m_gapless.current.total_samples;
        if (params.total_samples != 0 || params.skip_samples == 0) {
            if (params.total_samples >= samples_processed) {
                const auto sample_difference =
                    s64(m_gapless.init.total_samples) - params.total_samples;

                m_gapless.init.total_samples = params.total_samples;
                m_gapless.current.total_samples -= sample_difference;
            } else {
                LOG_WARNING(Lib_Ajm, "ORBIS_AJM_RESULT_INVALID_PARAMETER");
                job.output.p_result->result |= ORBIS_AJM_RESULT_INVALID_PARAMETER;
            }
        }

        const auto samples_skipped = m_gapless.init.skip_samples - m_gapless.current.skip_samples;
        if (params.skip_samples != 0 || params.total_samples == 0) {
            if (params.skip_samples >= samples_skipped) {
                const auto sample_difference =
                    s32(m_gapless.init.skip_samples) - params.skip_samples;

                m_gapless.init.skip_samples = params.skip_samples;
                m_gapless.current.skip_samples -= sample_difference;
            } else {
                LOG_WARNING(Lib_Ajm, "ORBIS_AJM_RESULT_INVALID_PARAMETER");
                job.output.p_result->result |= ORBIS_AJM_RESULT_INVALID_PARAMETER;
            }
        }
    }

    std::span<u8> in_buf(job.input.buffer);
    SparseOutputBuffer out_buf(job.output.buffers);
    auto in_size = in_buf.size();
    auto out_size = out_buf.Size();
    u32 frames_decoded = 0;

    if (!job.input.buffer.empty()) {
        for (;;) {
            if (m_flags.gapless_loop && m_gapless.IsEnd()) {
                m_gapless.Reset();
                m_total_samples = 0;
            }
            if (!HasEnoughSpace(out_buf)) {
                LOG_TRACE(Lib_Ajm, "ORBIS_AJM_RESULT_NOT_ENOUGH_ROOM ({} < {})", out_buf.Size(),
                          m_codec->GetNextFrameSize(m_gapless));
                job.output.p_result->result |= ORBIS_AJM_RESULT_NOT_ENOUGH_ROOM;
            }
            if (in_buf.size() < m_codec->GetMinimumInputSize()) {
                job.output.p_result->result |= ORBIS_AJM_RESULT_PARTIAL_INPUT;
            }
            if (job.output.p_result->result != 0) {
                break;
            }
            const auto result = m_codec->ProcessData(in_buf, out_buf, m_gapless);
            if (result.is_reset) {
                m_total_samples = 0;
            } else {
                m_total_samples += result.samples_written;
            }
            frames_decoded += result.frames_decoded;
            if (result.result != 0) {
                job.output.p_result->result |= result.result;
                job.output.p_result->internal_result = result.internal_result;
                break;
            }
            if (False(job.flags.run_flags & AjmJobRunFlags::MultipleFrames)) {
                break;
            }
        }
    }

    if (job.output.p_mframe) {
        job.output.p_mframe->num_frames = frames_decoded;
    }
    if (job.output.p_stream) {
        job.output.p_stream->input_consumed = in_size - in_buf.size();
        job.output.p_stream->output_written = out_size - out_buf.Size();
        job.output.p_stream->total_decoded_samples = m_total_samples;
    }

    if (job.output.p_format != nullptr) {
        *job.output.p_format = m_codec->GetFormat();
    }
    if (job.output.p_gapless_decode != nullptr) {
        *job.output.p_gapless_decode = m_gapless.current;
    }
    if (job.output.p_codec_info != nullptr) {
        m_codec->GetInfo(job.output.p_codec_info);
    }

    // AJMDBG ----------------------------------------------------------------
    // If intotal tops out at the size of the leading partial read (27,368 for
    // mode_303) then not one of the 128 KiB refills was ever credited. If it
    // goes past that, the credit happened and the bytes did not. ringchg says
    // which part of the ring window actually changed.
    {
        const u64 addr = job.dbg_chunks.empty() ? 0 : job.dbg_chunks.front().addr;
        const u32 nchunks = static_cast<u32>(job.dbg_chunks.size());
        m_dbg_in_total += static_cast<u64>(in_size);
        if (addr != 0 && m_dbg_ring_base == 0) {
            m_dbg_ring_base = addr;
        }

        u32 changed = 0;
        s32 first_changed = -1;
        if (m_dbg_ring_base != 0 && (m_dbg_jobs % 16) == 0) {
            auto* memory = Core::Memory::Instance();
            for (u32 i = 0; i < 32; i++) {
                const VAddr sub = m_dbg_ring_base + static_cast<u64>(i) * 4096;
                if (!memory->IsValidMapping(sub, 4096)) {
                    continue;
                }
                const u64 h = AjmRingHash64(reinterpret_cast<const u8*>(sub), 4096);
                if (h != m_dbg_ring_hash[i]) {
                    m_dbg_ring_hash[i] = h;
                    changed++;
                    if (first_changed < 0) {
                        first_changed = static_cast<s32>(i);
                    }
                }
            }
        }

        // AJMDBG: which output sidebands the guest asked for, and what was
        // written back into them. The guest reads these to decide how far the
        // stream has got, so a wrong value here is a wrong decision there.
        m_dbg_inst_id = job.instance_id;
        m_dbg_jobs_total++;
        char sb[8];
        u32 sbn = 0;
        if (job.output.p_stream != nullptr) {
            sb[sbn++] = 'S';
        }
        if (job.output.p_format != nullptr) {
            sb[sbn++] = 'F';
        }
        if (job.output.p_codec_info != nullptr) {
            sb[sbn++] = 'C';
        }
        if (job.output.p_gapless_decode != nullptr) {
            sb[sbn++] = 'G';
        }
        if (job.output.p_mframe != nullptr) {
            sb[sbn++] = 'M';
        }
        if (sbn == 0) {
            sb[sbn++] = '-';
        }
        sb[sbn] = '\0';

        LOG_INFO(Lib_Ajm,
                 "[AJMDBG] job inst={:#x} addr={:#x} chunks={} in={} consumed={} intotal={} "
                 "out={} written={} frames={} result={:#x} total={} ringbase={:#x} ringchg={} "
                 "first={} jflags={:#x} run={:#x} ctl={:#x} sb={} sIn={} sOut={} sTot={}",
                 job.instance_id, addr, nchunks, in_size, in_size - in_buf.size(), m_dbg_in_total,
                 out_size, out_size - out_buf.Size(), frames_decoded, job.output.p_result->result,
                 m_total_samples, m_dbg_ring_base, changed, first_changed, job.flags.raw,
                 static_cast<u64>(job.flags.run_flags), static_cast<u64>(job.flags.control_flags),
                 static_cast<const char*>(sb),
                 job.output.p_stream != nullptr ? job.output.p_stream->input_consumed : -1,
                 job.output.p_stream != nullptr ? job.output.p_stream->output_written : -1,
                 job.output.p_stream != nullptr
                     ? static_cast<s64>(job.output.p_stream->total_decoded_samples)
                     : -1);

        // The rare ones get their own line so the common job line stays short.
        if (job.output.p_format != nullptr || job.output.p_gapless_decode != nullptr ||
            job.output.p_codec_info != nullptr) {
            const auto* ci =
                reinterpret_cast<const AjmSidebandDecAt9CodecInfo*>(job.output.p_codec_info);
            LOG_INFO(Lib_Ajm,
                     "[AJMDBG] sbx inst={:#x} fmt_ch={} fmt_freq={} fmt_enc={} gap_total={} "
                     "gap_skip={} gap_skipped={} at9_sfs={} at9_fps={} at9_next={} at9_fs={}",
                     job.instance_id,
                     job.output.p_format != nullptr ? s64(job.output.p_format->num_channels) : -1,
                     job.output.p_format != nullptr ? s64(job.output.p_format->sampl_freq) : -1,
                     job.output.p_format != nullptr
                         ? s64(static_cast<u32>(job.output.p_format->sample_encoding))
                         : -1,
                     job.output.p_gapless_decode != nullptr
                         ? s64(job.output.p_gapless_decode->total_samples)
                         : -1,
                     job.output.p_gapless_decode != nullptr
                         ? s64(job.output.p_gapless_decode->skip_samples)
                         : -1,
                     job.output.p_gapless_decode != nullptr
                         ? s64(job.output.p_gapless_decode->skipped_samples)
                         : -1,
                     ci != nullptr ? s64(ci->super_frame_size) : -1,
                     ci != nullptr ? s64(ci->frames_in_super_frame) : -1,
                     ci != nullptr ? s64(ci->next_frame_size) : -1,
                     ci != nullptr ? s64(ci->frame_samples) : -1);
        }
        m_dbg_jobs++;
    }
}

bool AjmInstance::HasEnoughSpace(const SparseOutputBuffer& output) const {
    if (m_gapless.IsEnd()) {
        return true;
    }
    return output.Size() >= m_codec->GetNextFrameSize(m_gapless);
}

} // namespace Libraries::Ajm
