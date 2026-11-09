#include "build_sbk.h"

#include <array>
#include <algorithm>
#include <climits>
#include <cstring>
#include <optional>
#include <span>
#include <sstream>
#include <unordered_map>

#include "common/log/log.h"
#include "common/util/BinaryReader.h"
#include "common/util/BinaryWriter.h"
#include "common/util/FileUtil.h"

namespace sbk {
namespace {

// ── Pitch helpers ─────────────────────────────────────────────────────────────
// Table copied from game/sound/989snd/util.cpp (NotePitchTable).
// 12 semitone entries followed by 128 fine-tuning entries.
// clang-format off
static const u16 kPitchTable[] = {
    0x8000, 0x879C, 0x8FAC, 0x9837, 0xA145, 0xAADC, 0xB504, 0xBFC8, 0xCB2F, 0xD744, 0xE411, 0xF1A1,
    0x8000, 0x800E, 0x801D, 0x802C, 0x803B, 0x804A, 0x8058, 0x8067, 0x8076, 0x8085, 0x8094, 0x80A3,
    0x80B1, 0x80C0, 0x80CF, 0x80DE, 0x80ED, 0x80FC, 0x810B, 0x811A, 0x8129, 0x8138, 0x8146, 0x8155,
    0x8164, 0x8173, 0x8182, 0x8191, 0x81A0, 0x81AF, 0x81BE, 0x81CD, 0x81DC, 0x81EB, 0x81FA, 0x8209,
    0x8218, 0x8227, 0x8236, 0x8245, 0x8254, 0x8263, 0x8272, 0x8282, 0x8291, 0x82A0, 0x82AF, 0x82BE,
    0x82CD, 0x82DC, 0x82EB, 0x82FA, 0x830A, 0x8319, 0x8328, 0x8337, 0x8346, 0x8355, 0x8364, 0x8374,
    0x8383, 0x8392, 0x83A1, 0x83B0, 0x83C0, 0x83CF, 0x83DE, 0x83ED, 0x83FD, 0x840C, 0x841B, 0x842A,
    0x843A, 0x8449, 0x8458, 0x8468, 0x8477, 0x8486, 0x8495, 0x84A5, 0x84B4, 0x84C3, 0x84D3, 0x84E2,
    0x84F1, 0x8501, 0x8510, 0x8520, 0x852F, 0x853E, 0x854E, 0x855D, 0x856D, 0x857C, 0x858B, 0x859B,
    0x85AA, 0x85BA, 0x85C9, 0x85D9, 0x85E8, 0x85F8, 0x8607, 0x8617, 0x8626, 0x8636, 0x8645, 0x8655,
    0x8664, 0x8674, 0x8683, 0x8693, 0x86A2, 0x86B2, 0x86C1, 0x86D1, 0x86E0, 0x86F0, 0x8700, 0x870F,
    0x871F, 0x872E, 0x873E, 0x874E, 0x875D, 0x876D, 0x877D, 0x878C};
// clang-format on

// Mirror of compute_sample_rate from decompiler/data/extract_sbk.cpp.
// Returns effective WAV playback rate for a Tone played at default m_note=60.
// PS1 mode (center_note >= 0): base rate 44100 Hz; PS2 mode (< 0): base rate 48000 Hz.
static s32 note_to_hz(s8 center_note, s8 center_fine) {
  const bool ps1 = (center_note >= 0);
  const u16 cn = ps1 ? (u16)(u8)center_note : (u16)(u8)(-center_note);
  const u16 cf = (u16)(s32)center_fine;

  s32 _fine = (s32)cf;
  s32 _fine2 = _fine;
  if (_fine < 0)
    _fine2 += 127;
  _fine2 /= 128;
  s32 _note = 60 + _fine2 - (s32)cn;
  s32 val3 = _note / 6;
  if (_note < 0)
    val3--;
  s32 off2 = _fine - _fine2 * 128;
  s32 val2 = (val3 < 0) ? -1 : 0;
  if (val3 < 0)
    val3--;
  val2 = (val3 / 2) - val2;
  s32 val = val2 - 2;
  s32 off1 = _note - val2 * 12;
  if (off1 < 0 || (off1 == 0 && off2 < 0)) {
    off1 += 12;
    val = val2 - 3;
  }
  if (off2 < 0) {
    off1 = off1 - 1 + _fine2;
    off2 += (_fine2 + 1) * 128;
  }
  if (off1 < 0 || off1 > 11 || off2 < 0 || off2 > 127)
    return ps1 ? 44100 : 48000;

  s32 ret = ((s32)kPitchTable[off1] * (s32)kPitchTable[off2 + 12]) / 0x10000;
  if (val < 0)
    ret = (ret + (1 << (-val - 1))) >> -val;
  return std::max(1, ps1 ? (s32)(44100 * ret / 0x1000) : (s32)(48000 * ret / 0x1000));
}

// Scan PS1-mode center_note [0..127] for the value whose playback rate is nearest to hz.
static s8 hz_to_note(s32 hz) {
  s8 best = 60;
  s32 best_diff = std::abs(note_to_hz(60, 0) - hz);
  for (int cn = 0; cn <= 127; cn++) {
    s32 diff = std::abs(note_to_hz((s8)cn, 0) - hz);
    if (diff < best_diff) {
      best_diff = diff;
      best = (s8)cn;
    }
  }
  return best;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
static constexpr u32 fcc(std::string_view s) {
  return (u32)(u8)s[3] << 24 | (u32)(u8)s[2] << 16 | (u32)(u8)s[1] << 8 | (u32)(u8)s[0];
}

static std::vector<u8> bw_bytes(BinaryWriter& w) {
  std::vector<u8> v(w.get_size());
  memcpy(v.data(), w.get_data(), v.size());
  return v;
}

// ── WAV reader ────────────────────────────────────────────────────────────────
struct WavData {
  std::vector<s16> samples;
  s32 sample_rate = 44100;
};

static std::optional<WavData> read_wav(const fs::path& path) {
  auto raw = file_util::read_binary_file(path);
  if (raw.size() < 44) {
    lg::error("[build_sbk] {} too small to be a WAV file", path.filename().string());
    return std::nullopt;
  }

  BinaryReader r({raw.data(), raw.size()});

  if (r.read<u32>() != fcc("RIFF")) {
    lg::error("[build_sbk] {} missing RIFF header", path.filename().string());
    return std::nullopt;
  }
  r.read<u32>();  // file size
  if (r.read<u32>() != fcc("WAVE")) {
    lg::error("[build_sbk] {} is not a WAVE file", path.filename().string());
    return std::nullopt;
  }

  u16 nch = 0, bits = 0;
  s32 rate = 0;
  std::vector<u8> data_chunk;

  while (r.bytes_left() >= 8) {
    u32 id = r.read<u32>();
    u32 sz = r.read<u32>();
    if (id == fcc("fmt ")) {
      u16 fmt = r.read<u16>();
      if (fmt != 1) {
        lg::error("[build_sbk] {} is not PCM (format tag {})", path.filename().string(), fmt);
        return std::nullopt;
      }
      nch = r.read<u16>();
      rate = (s32)r.read<u32>();
      r.read<u32>();  // byte_rate
      r.read<u16>();  // block_align
      bits = r.read<u16>();
      if (sz > 16)
        r.ffwd(sz - 16);
    } else if (id == fcc("data")) {
      data_chunk.resize(sz);
      for (u32 i = 0; i < sz && r.bytes_left(); i++)
        data_chunk[i] = r.read<u8>();
    } else {
      if (r.bytes_left() >= sz)
        r.ffwd(sz);
      else
        break;
    }
  }

  if (nch == 0 || bits != 16 || data_chunk.empty()) {
    lg::error("[build_sbk] {} must be 16-bit PCM (got {} channels, {} bits per sample)",
              path.filename().string(), nch, bits);
    return std::nullopt;
  }

  const auto* p = reinterpret_cast<const s16*>(data_chunk.data());
  const size_t n = data_chunk.size() / ((size_t)nch * 2);
  std::vector<s16> mono;
  mono.reserve(n);
  for (size_t i = 0; i < n; i++) {
    s32 mix = 0;
    for (u16 c = 0; c < nch; c++)
      mix += p[i * nch + c];
    mono.push_back((s16)(mix / nch));
  }
  return WavData{std::move(mono), rate};
}

// ── SPU ADPCM encoder ─────────────────────────────────────────────────────────
// Each output block is 16 bytes: [shift|filter, flags, 14 nibble bytes] → 28 samples.
// Selects filter (0-4) and shift (0-12) per block to minimise total quantisation error.
// Uses proper decoded-value feedback between blocks for accuracy.
std::vector<u8> encode_spu_adpcm(const std::vector<s16>& samples, bool loop = false) {
  static constexpr s32 f1[5] = {0, 60, 115, 98, 122};
  static constexpr s32 f2[5] = {0, 0, -52, -55, -60};

  std::vector<s16> buf = samples;
  while (buf.size() % 28 != 0)
    buf.push_back(0);
  if (buf.empty())
    buf.resize(28, 0);

  std::vector<u8> out;
  out.reserve((buf.size() / 28) * 16);
  s32 prev[2] = {0, 0};

  const int nblk = (int)buf.size() / 28;
  for (int blk = 0; blk < nblk; blk++) {
    const s16* src = buf.data() + blk * 28;

    // Try all filter × shift combinations; keep the one with lowest summed error.
    int best_f = 0, best_sh = 0;
    s64 best_err = INT64_MAX;
    for (int fi = 0; fi < 5; fi++) {
      // Compute residuals using original samples as intra-block feedback (approximation).
      s32 deltas[28];
      s32 tp[2] = {prev[0], prev[1]};
      for (int s = 0; s < 28; s++) {
        s32 pred = (tp[0] * f1[fi] + tp[1] * f2[fi] + 32) / 64;
        deltas[s] = (s32)src[s] - pred;
        tp[1] = tp[0];
        tp[0] = src[s];
      }
      for (int sh = 0; sh <= 12; sh++) {
        const int rsh = 12 - sh;
        s64 err = 0;
        for (int s = 0; s < 28; s++) {
          s32 nibble = std::clamp(deltas[s] >> rsh, -8, 7);
          s32 decoded = ((s32)(s16)(nibble << 12)) >> sh;
          err += std::abs(deltas[s] - decoded);
        }
        if (err < best_err) {
          best_err = err;
          best_f = fi;
          best_sh = sh;
        }
      }
    }

    u8 block[16] = {};
    block[0] = (u8)((best_f << 4) | best_sh);
    if (!loop) {
      block[1] = (blk == nblk - 1) ? 0x01u : 0x00u;
    } else if (blk == nblk - 1) {
      block[1] = 0x03u;  // LoopEnd + LoopRepeat
    } else if (blk == 1 && nblk > 2) {
      block[1] = 0x06u;  // LoopStart + LoopRepeat
    } else if (blk == 0 && nblk <= 2) {
      block[1] = 0x06u;  // LoopStart at beginning for very short samples
    } else {
      block[1] = 0x00u;
    }

    s32 ep[2] = {prev[0], prev[1]};
    const int rsh = 12 - best_sh;
    for (int s = 0; s < 28; s++) {
      s32 pred = (ep[0] * f1[best_f] + ep[1] * f2[best_f] + 32) / 64;
      s32 nibble = std::clamp(((s32)src[s] - pred) >> rsh, -8, 7);
      if (s % 2 == 0)
        block[2 + s / 2] = (u8)(nibble & 0xF);
      else
        block[2 + s / 2] |= (u8)((nibble & 0xF) << 4);
      s32 dec = ((s32)(s16)(nibble << 12)) >> best_sh;
      ep[1] = ep[0];
      ep[0] = std::clamp(pred + dec, (s32)-32768, (s32)32767);
    }
    prev[0] = ep[0];
    prev[1] = ep[1];
    for (u8 b : block)
      out.push_back(b);
  }
  return out;
}

// ── Per-grain and per-sound data ──────────────────────────────────────────────
struct GrainData {
  u32 type = 1;  // GrainType enum value
  s32 delay = 0;
  // TONE / TONE2 fields (type 1 or 9)
  std::vector<u8> adpcm;
  s8 priority = 0;
  s8 vol = 127;
  s8 center_note = 60;
  s8 center_fine = 0;
  s16 pan = 0;
  s8 map_low = 0;
  s8 map_high = 127;
  s8 pb_low = 0;
  s8 pb_high = 0;
  u16 adsr1 = 0x8f7f;
  u16 adsr2 = 0xdfff;
  u16 tone_flags = 0;
  // Control grain params (all non-TONE, non-RAND_DELAY types)
  s16 ctrl[4] = {};
  // RAND_DELAY (type 26)
  s32 rand_delay_amount = 0;
  // LFOParams
  u8 lfo_which = 0, lfo_target = 0, lfo_target_extra = 0, lfo_shape = 0;
  u16 lfo_duty_cycle = 0, lfo_depth = 0, lfo_flags = 0, lfo_start_offset = 0;
  u32 lfo_step_size = 0;
  bool loop = false;
  s32 play_vol = 0, play_pan = 0;
  s8 play_reg[4] = {};
  s32 play_sound_id = -1;
  std::string play_snd_name;

  bool is_tone() const { return type == 1 || type == 9; }
  bool is_lfo() const { return type == 4; }
  bool is_playsound() const { return type == 5 || type == 6 || type == 8; }
};

struct SoundData {
  std::string name;
  s8 vol = 127;
  s8 volgroup = 0;
  s16 pan = 90;
  s8 instance_limit = 0;
  u16 flags = 0;
  std::vector<GrainData> grains;
};

std::vector<SoundData> encode_sounds(const std::vector<SoundSpec>& specs) {
  std::vector<SoundData> out;
  for (const auto& spec : specs) {
    SoundData sd;
    sd.vol = 127;
    sd.pan = 90;

    std::vector<GrainData> tone_grains;
    for (const auto& p : spec.variants) {
      auto wav = read_wav(p);
      if (!wav) {
        lg::warn("[build_sbk] Skipping {}", p.filename().string());
        continue;
      }
      GrainData gd;
      gd.type = 1;  // TONE
      gd.center_note = hz_to_note(wav->sample_rate);
      gd.adpcm = encode_spu_adpcm(wav->samples);
      lg::info("[build_sbk] {} — {} samples, {}Hz → center_note={}", p.filename().string(),
               wav->samples.size(), wav->sample_rate, (int)gd.center_note);
      tone_grains.push_back(std::move(gd));
    }

    const u32 nv = (u32)tone_grains.size();
    if (nv == 0)
      continue;

    if (nv > 1) {
      GrainData rp;
      rp.type = 25;  // RAND_PLAY
      rp.ctrl[0] = (s16)nv;
      rp.ctrl[1] = 1;
      rp.ctrl[2] = (s16)nv;
      rp.ctrl[3] = 0;
      sd.grains.push_back(std::move(rp));
    }
    for (auto& g : tone_grains)
      sd.grains.push_back(std::move(g));

    out.push_back(std::move(sd));
  }
  return out;
}

std::vector<u8> build_sblk_entries(const std::vector<SoundData>& sounds,
                                   u32 grain_index_base,
                                   u32 sample_base) {
  BinaryWriter w;
  // Sound entries (12 bytes each)
  u32 grain_cursor = grain_index_base;
  for (const auto& snd : sounds) {
    const u32 ng = (u32)snd.grains.size();
    w.add<s8>(snd.vol);
    w.add<s8>(snd.volgroup);
    w.add<s16>(snd.pan);
    w.add<s8>((s8)ng);
    w.add<s8>(snd.instance_limit);
    w.add<u16>(snd.flags);
    w.add<u32>(grain_cursor * 40);  // first_sfx_grain (byte offset in grain table)
    grain_cursor += ng;
  }
  // grain entries (40 bytes each)
  u32 soff = sample_base;
  for (const auto& snd : sounds) {
    for (const auto& g : snd.grains) {
      w.add<u32>(g.type);
      w.add<s32>(g.delay);
      if (g.is_tone()) {
        // tone struct, 20 bytes of fields + 12 bytes padding
        w.add<s8>(g.priority);
        w.add<s8>(g.vol);
        w.add<s8>(g.center_note);
        w.add<s8>(g.center_fine);
        w.add<s16>(g.pan);
        w.add<s8>(g.map_low);
        w.add<s8>(g.map_high);
        w.add<s8>(g.pb_low);
        w.add<s8>(g.pb_high);
        w.add<u16>(g.adsr1);
        w.add<u16>(g.adsr2);
        w.add<u16>(g.tone_flags);
        w.add<u32>(soff);
        w.add<u32>(0);  // reserved
        w.add<u32>(0);  // pad
        w.add<u32>(0);  // pad
        soff += (u32)g.adpcm.size();
      } else if (g.type == 26) {  // RAND_DELAY
        w.add<s32>(g.rand_delay_amount);
        for (int i = 0; i < 28; i++)
          w.add<u8>(0);
      } else if (g.is_lfo()) {
        w.add<u8>(g.lfo_which);
        w.add<u8>(g.lfo_target);
        w.add<u8>(g.lfo_target_extra);
        w.add<u8>(g.lfo_shape);
        w.add<u16>(g.lfo_duty_cycle);
        w.add<u16>(g.lfo_depth);
        w.add<u16>(g.lfo_flags);
        w.add<u16>(g.lfo_start_offset);
        w.add<u32>(g.lfo_step_size);
        for (int i = 0; i < 16; i++) {
          w.add<u8>(0);
        }
      } else if (g.is_playsound()) {
        w.add<s32>(g.play_vol);
        w.add<s32>(g.play_pan);
        for (int i = 0; i < 4; i++) {
          w.add<s8>(g.play_reg[i]);
        }
        w.add<s32>(g.play_sound_id);
        for (int i = 0; i < 16; i++) {
          w.add<u8>(0);
        }
      } else {
        // Control grain: ControlParams (8 bytes) + 24 bytes padding
        for (int i = 0; i < 4; i++)
          w.add<s16>(g.ctrl[i]);
        for (int i = 0; i < 24; i++)
          w.add<u8>(0);
      }
    }
  }
  return bw_bytes(w);
}

std::vector<u8> build_sblk(const std::vector<SoundData>& sounds, u32 bank_id) {
  const u32 n = (u32)sounds.size();
  u32 total_grains = 0;
  u32 total_samples = 0;
  for (const auto& s : sounds) {
    total_grains += (u32)s.grains.size();
    for (const auto& g : s.grains)
      if (g.is_tone())
        total_samples += (u32)g.adpcm.size();
  }

  constexpr u32 kHeaderSize = 60;
  const u32 first_sound = kHeaderSize;
  const u32 first_grain = kHeaderSize + n * 12;

  BinaryWriter w;
  w.add<u32>(fcc("SBlk"));
  w.add<u32>(1);                  // version
  w.add<u32>(0);                  // flags
  w.add<u32>(bank_id);            // BankID
  w.add<s8>(0);                   // BankNum
  w.add<s8>(0);                   // pad
  w.add<s16>(0);                  // pad
  w.add<s16>(0);                  // pad
  w.add<s16>((s16)n);             // num_sounds
  w.add<s16>((s16)total_grains);  // num_grains
  w.add<s16>(0);                  // num_vags
  w.add<u32>(first_sound);
  w.add<u32>(first_grain);
  w.add<u32>(0);              // VagsInSR
  w.add<u32>(total_samples);  // VagDataSize
  w.add<u32>(0);              // SRAMAllocSize
  w.add<u32>(0);              // NextBlock
  w.add<u32>(0);              // block_names_offset
  w.add<u32>(0);              // SFXUD

  auto entries = build_sblk_entries(sounds, 0, 0);
  w.add_data(entries.data(), entries.size());
  return bw_bytes(w);
}

// ── FileAttributes wrapper ────────────────────────────────────────────────────
std::vector<u8> wrap_fa(const std::vector<u8>& bank, const std::vector<u8>& samples) {
  // FA header: {u32 type, u32 nchunks, nchunks×{u32 offset, u32 size}}
  constexpr u32 kFAHeader = 8 + 2 * 8;  // 24 bytes
  const u32 bank_off = kFAHeader;
  const u32 samp_off = kFAHeader + (u32)bank.size();

  BinaryWriter w;
  w.add<u32>(3);  // type = SFX bank
  w.add<u32>(2);  // num_chunks
  w.add<u32>(bank_off);
  w.add<u32>((u32)bank.size());
  w.add<u32>(samp_off);
  w.add<u32>((u32)samples.size());
  w.add_data(const_cast<u8*>(bank.data()), bank.size());
  w.add_data(const_cast<u8*>(samples.data()), samples.size());
  return bw_bytes(w);
}

// ── Jak 1 on-disc name table ──────────────────────────────────────────────────
std::vector<u8> build_name_table(const std::vector<std::string>& names) {
  BinaryWriter w;
  w.add_cstr_len("CUSTOM", 16);
  w.add<u32>(0);
  w.add<u32>((u32)names.size());
  for (const auto& n : names) {
    std::string trimmed = n.substr(0, 15);
    w.add_cstr_len(trimmed.c_str(), 16);
    w.add<u32>(0);
  }
  while (w.get_size() % 2048)
    w.add<u8>(0);
  return bw_bytes(w);
}

// ── FA header offset scanner (mirrors decompiler logic) ───────────────────────
size_t find_fa_offset(std::span<const u8> data) {
  auto valid = [](u32 t, u32 nc) { return (t == 1 || t == 3) && nc >= 2 && nc <= 4; };
  if (data.size() >= 8) {
    u32 t, nc;
    memcpy(&t, data.data(), 4);
    memcpy(&nc, data.data() + 4, 4);
    if (valid(t, nc))
      return 0;
  }
  for (size_t off = 2048; off + 8 <= data.size(); off += 2048) {
    u32 t, nc;
    memcpy(&t, data.data() + off, 4);
    memcpy(&nc, data.data() + off + 4, 4);
    if (valid(t, nc))
      return off;
  }
  return SIZE_MAX;
}

void write_file(const fs::path& path, const std::vector<u8>& data) {
  file_util::create_dir_if_needed(path.parent_path());
  auto fp = file_util::open_file(path.string().c_str(), "wb");
  if (!fp) {
    lg::error("[build_sbk] Cannot open {} for writing", path.string());
    return;
  }
  fwrite(data.data(), 1, data.size(), fp);
  fclose(fp);
}

// ── Metadata parsing helpers ──────────────────────────────────────────────────

u32 grain_type_from_name(const std::string& name) {
  static const std::unordered_map<std::string, u32> kMap = {
      {"NULL_GRAIN", 0},
      {"TONE", 1},
      {"XREF_ID", 2},
      {"XREF_NUM", 3},
      {"LFO_SETTINGS", 4},
      {"STARTCHILDSOUND", 5},
      {"STOPCHILDSOUND", 6},
      {"PLUGIN_MESSAGE", 7},
      {"BRANCH", 8},
      {"TONE2", 9},
      {"CONTROL_NULL", 20},
      {"LOOP_START", 21},
      {"LOOP_END", 22},
      {"LOOP_CONTINUE", 23},
      {"STOP", 24},
      {"RAND_PLAY", 25},
      {"RAND_DELAY", 26},
      {"RAND_PB", 27},
      {"PB", 28},
      {"ADD_PB", 29},
      {"SET_REGISTER", 30},
      {"SET_REGISTER_RAND", 31},
      {"INC_REGISTER", 32},
      {"DEC_REGISTER", 33},
      {"TEST_REGISTER", 34},
      {"MARKER", 35},
      {"GOTO_MARKER", 36},
      {"GOTO_RANDOM_MARKER", 37},
      {"WAIT_FOR_ALL_VOICES", 38},
      {"PLAY_CYCLE", 39},
      {"ADD_REGISTER", 40},
      {"KEY_OFF_VOICES", 41},
      {"KILL_VOICES", 42},
      {"ON_STOP_MARKER", 43},
      {"COPY_REGISTER", 44},
  };
  auto it = kMap.find(name);
  return it != kMap.end() ? it->second : 0;
}

// Extract a decimal or hex integer value for "key=VALUE" from line; returns def if not found.
static s64 kv_int(const std::string& line, const std::string& key, s64 def = 0) {
  const std::string tok = key + "=";
  auto pos = line.find(tok);
  if (pos == std::string::npos)
    return def;
  pos += tok.size();
  const char* s = line.c_str() + pos;
  char* end = nullptr;
  s64 v = strtoll(s, &end, 0);
  return (end == s) ? def : v;
}

// Extract [A,B] from "key=[A,B]"; ignores surrounding spaces.
static std::pair<s64, s64> kv_pair(const std::string& line, const std::string& key) {
  const std::string tok = key + "=[";
  auto pos = line.find(tok);
  if (pos == std::string::npos)
    return {0, 0};
  pos += tok.size();
  const char* s = line.c_str() + pos;
  char* end = nullptr;
  s64 a = strtoll(s, &end, 0);
  s = end;
  while (*s == ',' || *s == ' ')
    s++;
  s64 b = strtoll(s, &end, 0);
  return {a, b};
}

// Extract key=[v0, v1, v2, v3] from the line (4 s16 values).
static std::array<s16, 4> kv_array4(const std::string& line, const std::string& key) {
  std::array<s16, 4> result = {};
  const std::string tok = key + "=[";
  auto pos = line.find(tok);
  if (pos == std::string::npos)
    return result;
  pos += tok.size();
  const char* s = line.c_str() + pos;
  for (int i = 0; i < 4; i++) {
    char* end = nullptr;
    result[i] = (s16)strtol(s, &end, 0);
    s = end;
    while (*s == ',' || *s == ' ')
      s++;
    if (*s == ']')
      break;
  }
  return result;
}

// Extract param=[p0, p1, p2, p3] from the line.
static std::array<s16, 4> kv_params(const std::string& line) {
  std::array<s16, 4> result = {};
  const std::string tok = "param=[";
  auto pos = line.find(tok);
  if (pos == std::string::npos)
    return result;
  pos += tok.size();
  const char* s = line.c_str() + pos;
  for (int i = 0; i < 4; i++) {
    char* end = nullptr;
    result[i] = (s16)strtol(s, &end, 0);
    s = end;
    while (*s == ',' || *s == ' ')
      s++;
    if (*s == ']')
      break;
  }
  return result;
}

// Return the first double-quoted string on line, or empty string.
static std::string extract_quoted(const std::string& line) {
  auto a = line.find('"');
  if (a == std::string::npos)
    return "";
  auto b = line.find('"', a + 1);
  if (b == std::string::npos)
    return line.substr(a + 1);
  return line.substr(a + 1, b - a - 1);
}

// Return the filename after "-> " on a grain line.
static std::string extract_wav_ref(const std::string& line) {
  auto pos = line.find("-> ");
  if (pos == std::string::npos)
    return "";
  pos += 3;
  while (pos < line.size() && line[pos] == ' ')
    pos++;
  size_t end = pos;
  while (end < line.size() && line[end] != '\n' && line[end] != '\r')
    end++;
  return line.substr(pos, end - pos);
}

// Parse a metadata.txt produced by extract_sbk into SoundData entries.
// WAV files referenced by TONE grains are loaded from wav_dir.
static std::vector<SoundData> parse_sbk_metadata(const fs::path& metadata_path,
                                                 const fs::path& wav_dir) {
  std::string text = file_util::read_text_file(metadata_path);
  std::vector<SoundData> sounds;
  SoundData* cur = nullptr;

  std::istringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    if (line[0] == '[') {
      // [NNN] "name"  vol=V  volgroup=VG  pan=P  instlimit=I  flags=0xF
      sounds.emplace_back();
      cur = &sounds.back();
      cur->name = extract_quoted(line);
      cur->vol = (s8)kv_int(line, "vol", 127);
      cur->volgroup = (s8)kv_int(line, "volgroup", 0);
      cur->pan = (s16)kv_int(line, "pan", 90);
      cur->instance_limit = (s8)kv_int(line, "instlimit", 0);
      cur->flags = (u16)kv_int(line, "flags", 0);
    } else if (cur && line.size() >= 4 && line[0] == ' ' && line[1] == ' ') {
      // "  grain[N] TYPE  ..."
      auto bracket_end = line.find("] ");
      if (bracket_end == std::string::npos)
        continue;
      size_t type_start = bracket_end + 2;
      size_t type_end = line.find(' ', type_start);
      if (type_end == std::string::npos)
        type_end = line.size();
      std::string type_name = line.substr(type_start, type_end - type_start);

      GrainData g;
      g.type = grain_type_from_name(type_name);
      g.delay = (s32)kv_int(line, "delay", 0);

      if (g.is_tone()) {
        g.priority = (s8)kv_int(line, "priority", 0);
        g.vol = (s8)kv_int(line, "vol", 127);
        g.center_note = (s8)kv_int(line, "note", 60);
        g.center_fine = (s8)kv_int(line, "fine", 0);
        g.pan = (s16)kv_int(line, "pan", 0);
        auto maplh = kv_pair(line, "map");
        g.map_low = (s8)maplh.first;
        g.map_high = (s8)maplh.second;
        auto pblh = kv_pair(line, "pb");
        g.pb_low = (s8)pblh.first;
        g.pb_high = (s8)pblh.second;
        auto adsr = kv_pair(line, "adsr");
        g.adsr1 = (u16)adsr.first;
        g.adsr2 = (u16)adsr.second;
        g.tone_flags = (u16)kv_int(line, "flags", 0);
        g.loop = kv_int(line, "loop", 0) != 0;
        // sample_offset is ignored — recomputed by build_sblk_entries

        auto wav_ref = extract_wav_ref(line);
        if (!wav_ref.empty()) {
          auto wav_path = wav_dir / wav_ref;
          auto wav = read_wav(wav_path);
          if (wav) {
            g.adpcm = encode_spu_adpcm(wav->samples, g.loop);
            lg::info("[build_sbk] {} — {} samples{}", wav_ref, wav->samples.size(),
                     g.loop ? " [loop]" : "");
          } else {
            lg::warn("[build_sbk] could not load wav {}", wav_path.string());
          }
        }
      } else if (g.type == 26) {  // RAND_DELAY
        g.rand_delay_amount = (s32)kv_int(line, "amount", 0);
      } else if (g.is_lfo()) {
        if (line.find("which_lfo=") != std::string::npos) {
          g.lfo_which = (u8)kv_int(line, "which_lfo", 0);
          g.lfo_target = (u8)kv_int(line, "target", 0);
          g.lfo_target_extra = (u8)kv_int(line, "target_extra", 0);
          g.lfo_shape = (u8)kv_int(line, "shape", 0);
          g.lfo_duty_cycle = (u16)kv_int(line, "duty_cycle", 0);
          g.lfo_depth = (u16)kv_int(line, "depth", 0);
          g.lfo_flags = (u16)kv_int(line, "flags", 0);
          g.lfo_start_offset = (u16)kv_int(line, "start_offset", 0);
          g.lfo_step_size = (u32)kv_int(line, "step_size", 0);
        }
      } else if (g.is_playsound()) {
        if (line.find("sound_id=") != std::string::npos) {
          g.play_vol = (s32)kv_int(line, "vol", 0);
          g.play_pan = (s32)kv_int(line, "pan", 0);
          auto reg = kv_array4(line, "reg");
          for (int i = 0; i < 4; i++) {
            g.play_reg[i] = (s8)reg[i];
          }
          g.play_sound_id = (s32)kv_int(line, "sound-id", -1);
          g.play_snd_name = extract_quoted(line);
        } else {
          auto p = kv_params(line);
          g.play_vol = (s32)((u32)(u16)(s16)p[0] | ((u32)(u16)(s16)p[1] << 16));
          g.play_pan = (s32)((u32)(u16)(s16)p[2] | ((u32)(u16)(s16)p[3] << 16));
          g.play_sound_id = 0;
        }
      } else {
        auto params = kv_params(line);
        for (int i = 0; i < 4; i++)
          g.ctrl[i] = params[i];
      }

      cur->grains.push_back(std::move(g));
    }
  }

  // resolve name-based child sound references (v2 banks use snd_name, v1 uses sound_id index).
  std::unordered_map<std::string, s32> name_to_idx;
  for (size_t i = 0; i < sounds.size(); i++) {
    name_to_idx[sounds[i].name] = (s32)i;
  }
  for (auto& snd : sounds) {
    for (auto& g : snd.grains) {
      if (!g.is_playsound() || g.play_snd_name.empty()) {
        continue;
      }
      auto it = name_to_idx.find(g.play_snd_name);
      if (it != name_to_idx.end()) {
        g.play_sound_id = it->second;
        lg::info("[build_sbk] resolved STARTCHILDSOUND '{}' to sound_id={}", g.play_snd_name,
                 g.play_sound_id);
      } else {
        lg::warn("[build_sbk] cannot resolve STARTCHILDSOUND name '{}' — sound not in bank",
                 g.play_snd_name);
      }
    }
  }

  return sounds;
}

// ── Internal append helper ────────────────────────────────────────────────────
static void do_append_sbk(const fs::path& input,
                          const std::vector<SoundData>& new_sounds,
                          const fs::path& output) {
  auto file_data = file_util::read_binary_file(input);
  if (file_data.empty()) {
    lg::error("[build_sbk] Cannot read {}", input.string());
    return;
  }

  std::span<const u8> span(file_data);
  const size_t fa_off = find_fa_offset(span);
  if (fa_off == SIZE_MAX) {
    lg::error("[build_sbk] Cannot find FileAttributes header in {}", input.string());
    return;
  }

  // preserve any on-disc prefix (Jak 1 name table) before the FA header.
  std::vector<u8> prefix(file_data.begin(), file_data.begin() + (std::ptrdiff_t)fa_off);

  // parse FA header.
  auto fa = span.subspan(fa_off);
  BinaryReader fa_r(fa);
  fa_r.read<u32>();  // type
  const u32 nchunks = fa_r.read<u32>();
  if (nchunks < 2) {
    lg::error("[build_sbk] Expected ≥2 FA chunks, got {}", nchunks);
    return;
  }
  struct Chunk {
    u32 off, sz;
  };
  std::vector<Chunk> chunks(nchunks);
  for (auto& c : chunks) {
    c.off = fa_r.read<u32>();
    c.sz = fa_r.read<u32>();
  }
  const std::span<const u8> bank_span = fa.subspan(chunks[0].off, chunks[0].sz);
  const std::span<const u8> samp_span = fa.subspan(chunks[1].off, chunks[1].sz);

  BinaryReader br(bank_span);
  if (br.read<u32>() != fcc("SBlk")) {
    lg::error("[build_sbk] Bank chunk is not an SBlk");
    return;
  }
  const u32 version = br.read<u32>();
  if (version >= 2) {
    lg::error("[build_sbk] --append only supports V1 SBlk banks (this is V{})", version);
    return;
  }
  br.read<u32>();  // flags
  br.read<u32>();  // BankID
  br.read<s8>();
  br.read<s8>();
  br.read<s16>();
  br.read<s16>();
  const u32 n_old = (u32)(u16)(s16)br.read<s16>();
  const u32 n_old_grains = (u32)(u16)(s16)br.read<s16>();
  br.read<s16>();  // NumVAGs
  const u32 first_sound = br.read<u32>();
  const u32 first_grain = br.read<u32>();

  if (first_sound > bank_span.size() || first_grain > bank_span.size() ||
      first_grain < first_sound) {
    lg::error("[build_sbk] Malformed SBlk: first_sound={} first_grain={} bank_size={}", first_sound,
              first_grain, bank_span.size());
    return;
  }

  const std::vector<u8> old_sounds(bank_span.data() + first_sound, bank_span.data() + first_grain);
  const std::vector<u8> old_grains(bank_span.data() + first_grain,
                                   bank_span.data() + bank_span.size());

  const u32 n_new = (u32)new_sounds.size();
  u32 total_new_grains = 0;
  u32 new_vag_size = (u32)samp_span.size();
  for (const auto& s : new_sounds) {
    total_new_grains += (u32)s.grains.size();
    for (const auto& g : s.grains)
      if (g.is_tone())
        new_vag_size += (u32)g.adpcm.size();
  }

  auto entries = build_sblk_entries(new_sounds, n_old_grains, (u32)samp_span.size());
  const u32 new_sound_bytes = n_new * 12;
  std::vector<u8> new_sound_data(entries.begin(), entries.begin() + new_sound_bytes);
  std::vector<u8> new_grain_data(entries.begin() + new_sound_bytes, entries.end());

  const u32 new_n_total = n_old + n_new;
  const u32 new_n_grains = n_old_grains + total_new_grains;
  const u32 new_first_grain = first_sound + new_n_total * 12;

  std::vector<u8> new_bank(60 + old_sounds.size() + new_sound_data.size() + old_grains.size() +
                           new_grain_data.size());
  memcpy(new_bank.data(), bank_span.data(), 60);

  auto patch = [&](size_t off, auto val) { memcpy(new_bank.data() + off, &val, sizeof(val)); };
  patch(22, (s16)new_n_total);
  patch(24, (s16)new_n_grains);
  patch(32, new_first_grain);
  patch(40, new_vag_size);

  size_t cur = 60;
  auto append_vec = [&](const std::vector<u8>& v) {
    memcpy(new_bank.data() + cur, v.data(), v.size());
    cur += v.size();
  };
  append_vec(old_sounds);
  append_vec(new_sound_data);
  append_vec(old_grains);
  append_vec(new_grain_data);

  // new sample data = old samples + new adpcm.
  std::vector<u8> new_samples(samp_span.begin(), samp_span.end());
  for (const auto& s : new_sounds)
    for (const auto& g : s.grains)
      if (g.is_tone())
        new_samples.insert(new_samples.end(), g.adpcm.begin(), g.adpcm.end());

  auto sbk = wrap_fa(new_bank, new_samples);

  // re-prepend the Jak 1 name table if the input had one.
  if (!prefix.empty())
    prefix.insert(prefix.end(), sbk.begin(), sbk.end());
  else
    prefix = std::move(sbk);

  write_file(output, prefix);
  lg::info("[build_sbk] Appended {} sounds ({} total) → {}", n_new, new_n_total, output.string());
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Public API
// ═════════════════════════════════════════════════════════════════════════════

void create_sbk(const std::vector<SoundSpec>& specs,
                const fs::path& output,
                const BuildOptions& opts) {
  auto sounds = encode_sounds(specs);
  if (sounds.empty()) {
    lg::error("[build_sbk] No sounds encoded, nothing to write");
    return;
  }

  std::vector<u8> samples;
  for (const auto& s : sounds)
    for (const auto& g : s.grains)
      if (g.is_tone())
        samples.insert(samples.end(), g.adpcm.begin(), g.adpcm.end());

  auto bank = build_sblk(sounds, opts.bank_id);
  auto sbk = wrap_fa(bank, samples);

  if (opts.jak1_format) {
    std::vector<std::string> names;
    names.reserve(specs.size());
    for (const auto& spec : specs)
      names.push_back(spec.variants.empty() ? "" : spec.variants[0].stem().string().substr(0, 15));
    auto prefix = build_name_table(names);
    prefix.insert(prefix.end(), sbk.begin(), sbk.end());
    sbk = std::move(prefix);
  }

  write_file(output, sbk);
  lg::info("[build_sbk] wrote {} sounds to {}", sounds.size(), output.string());
}

void append_sbk(const fs::path& input,
                const std::vector<SoundSpec>& specs,
                const fs::path& output) {
  auto new_sounds = encode_sounds(specs);
  if (new_sounds.empty()) {
    lg::error("[build_sbk] No sounds to append");
    return;
  }
  do_append_sbk(input, new_sounds, output);
}

void create_sbk_from_dir(const fs::path& dir, const fs::path& output, const BuildOptions& opts) {
  auto meta_path = dir / "metadata.txt";
  if (!fs::exists(meta_path)) {
    lg::error("[build_sbk] no metadata.txt found in {}", dir.string());
    return;
  }

  auto sounds = parse_sbk_metadata(meta_path, dir);
  if (sounds.empty()) {
    lg::error("[build_sbk] no sounds parsed from {}", meta_path.string());
    return;
  }

  std::vector<u8> samples;
  for (const auto& s : sounds)
    for (const auto& g : s.grains)
      if (g.is_tone())
        samples.insert(samples.end(), g.adpcm.begin(), g.adpcm.end());

  auto bank = build_sblk(sounds, opts.bank_id);
  auto sbk = wrap_fa(bank, samples);

  if (opts.jak1_format) {
    std::vector<std::string> names;
    names.reserve(sounds.size());
    for (const auto& snd : sounds)
      names.push_back(snd.name.substr(0, 15));
    auto prefix = build_name_table(names);
    prefix.insert(prefix.end(), sbk.begin(), sbk.end());
    sbk = std::move(prefix);
  }

  write_file(output, sbk);
  lg::info("[build_sbk] wrote {} sounds to {}", sounds.size(), output.string());
}

void append_sbk_from_dir(const fs::path& input, const fs::path& dir, const fs::path& output) {
  auto meta_path = dir / "metadata.txt";
  if (!fs::exists(meta_path)) {
    lg::error("[build_sbk] no metadata.txt found in {}", dir.string());
    return;
  }

  auto new_sounds = parse_sbk_metadata(meta_path, dir);
  if (new_sounds.empty()) {
    lg::error("[build_sbk] no sounds parsed from {}", meta_path.string());
    return;
  }

  do_append_sbk(input, new_sounds, output);
}

}  // namespace sbk
