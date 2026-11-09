#include "extract_sbk.h"

#include <array>
#include <algorithm>
#include <cstring>
#include <span>
#include <unordered_map>
#include <unordered_set>

#include "common/audio/audio_formats.h"
#include "common/log/log.h"
#include "common/util/BinaryReader.h"
#include "common/util/FileUtil.h"

#include "fmt/format.h"

namespace decompiler {
namespace {

// clang-format off
const u16 kNotePitchTable[] = {
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

s32 compute_sample_rate(s8 center_note, s8 center_fine) {
  const bool ps1_mode = (center_note >= 0);
  const u16 cn = ps1_mode ? (u16)(u8)center_note : (u16)(u8)(-center_note);
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
  s32 offset2 = _fine - _fine2 * 128;
  s32 val2 = (val3 < 0) ? -1 : 0;
  if (val3 < 0)
    val3--;
  val2 = (val3 / 2) - val2;
  s32 val = val2 - 2;
  s32 offset1 = _note - val2 * 12;
  if (offset1 < 0 || (offset1 == 0 && offset2 < 0)) {
    offset1 += 12;
    val = val2 - 3;
  }
  if (offset2 < 0) {
    offset1 = offset1 - 1 + _fine2;
    offset2 += (_fine2 + 1) * 128;
  }
  if (offset1 < 0 || offset1 > 11 || offset2 < 0 || offset2 > 127)
    return ps1_mode ? 44100 : 48000;

  s32 ret = ((s32)kNotePitchTable[offset1] * (s32)kNotePitchTable[offset2 + 12]) / 0x10000;
  if (val < 0)
    ret = (ret + (1 << (-val - 1))) >> -val;

  if (ps1_mode)
    return std::max(1, (s32)(44100 * ret / 0x1000));
  return std::max(1, (s32)(48000 * ret / 0x1000));
}

constexpr u32 fcc(std::string_view s) {
  return s[3] << 24 | s[2] << 16 | s[1] << 8 | s[0];
}

size_t adpcm_sample_size(const u8* data, size_t max_bytes) {
  size_t pos = 0;
  while (pos + 16 <= max_bytes) {
    u8 flags = data[pos + 1];
    pos += 16;
    if (flags & 1)
      return pos;
  }
  return pos;
}

bool adpcm_has_loop(const u8* data, size_t size) {
  for (size_t pos = 0; pos + 16 <= size; pos += 16) {
    u8 flags = data[pos + 1];
    if ((flags & 0x03) == 0x03) {
      return true;
    }
    if (flags & 1) {
      break;
    }
  }
  return false;
}

// Decode raw SPU ADPCM bytes to 16-bit PCM.
// Each 16-byte block: [shift|filter, flags, 14 data bytes] → 28 samples.
std::vector<s16> decode_spu_adpcm(const u8* data, size_t size) {
  static constexpr s32 f1[5] = {0, 60, 115, 98, 122};
  static constexpr s32 f2[5] = {0, 0, -52, -55, -60};

  std::vector<s16> samples;
  s32 prev[2] = {0, 0};

  for (size_t pos = 0; pos + 16 <= size; pos += 16) {
    u8 shift = data[pos] & 0xF;
    u8 filter = std::min<u8>(4, data[pos] >> 4);

    for (int i = 0; i < 14; i++) {
      u8 b = data[pos + 2 + i];

      // Each byte encodes two samples: low nibble first, then high nibble.
      for (int nib = 0; nib < 2; nib++) {
        s32 raw = (nib == 0) ? (b & 0xF) : (b >> 4);
        s32 s = (s32)(s16)(raw << 12) >> shift;
        s += (prev[0] * f1[filter] + prev[1] * f2[filter] + 32) / 64;
        s = std::clamp(s, (s32)-32768, (s32)32767);
        samples.push_back((s16)s);
        prev[1] = prev[0];
        prev[0] = s;
      }
    }
  }
  return samples;
}

// Full per-grain data collected for both WAV extraction and metadata output.
struct FullGrainInfo {
  u32 type = 0;
  s32 delay = 0;
  // TONE / TONE2 fields
  s8 priority = 0, vol = 0, center_note = 60, center_fine = 0;
  s16 pan = 0;
  s8 map_low = 0, map_high = 127, pb_low = 0, pb_high = 0;
  u16 adsr1 = 0, adsr2 = 0, tone_flags = 0;
  u32 sample_offset = 0;
  // ControlParams (RAND_PLAY, PLAY_CYCLE, SET_REGISTER, etc.)
  s16 ctrl[4] = {};
  // RAND_DELAY
  s32 rand_delay_amount = 0;
  // LFOParams
  u8 lfo_which = 0, lfo_target = 0, lfo_target_extra = 0, lfo_shape = 0;
  u16 lfo_duty_cycle = 0, lfo_depth = 0, lfo_flags = 0, lfo_start_offset = 0;
  u32 lfo_step_size = 0;
  s32 play_vol = 0, play_pan = 0;
  s8 play_reg[4] = {};
  s32 play_sound_id = -1;
  std::string play_snd_name;  // non-empty for name-based child sound references
  // Resolved WAV output filename (TONE grains only, relative to bank output dir)
  std::string wav_filename;

  bool has_loop = false;

  bool is_tone() const { return type == 1 || type == 9; }
  bool is_lfo() const { return type == 4; }
  bool is_playsound() const { return type == 5 || type == 6 || type == 8; }
};

struct SoundFullInfo {
  int index = 0;
  std::string name;
  s8 vol = 0, volgroup = 0;
  s16 pan = 0;
  s8 instance_limit = 0;
  u16 flags = 0;
  std::vector<FullGrainInfo> grains;
};

static const char* grain_type_name(u32 type) {
  switch (type) {
    case 0:
      return "NULL_GRAIN";
    case 1:
      return "TONE";
    case 2:
      return "XREF_ID";
    case 3:
      return "XREF_NUM";
    case 4:
      return "LFO_SETTINGS";
    case 5:
      return "STARTCHILDSOUND";
    case 6:
      return "STOPCHILDSOUND";
    case 7:
      return "PLUGIN_MESSAGE";
    case 8:
      return "BRANCH";
    case 9:
      return "TONE2";
    case 20:
      return "CONTROL_NULL";
    case 21:
      return "LOOP_START";
    case 22:
      return "LOOP_END";
    case 23:
      return "LOOP_CONTINUE";
    case 24:
      return "STOP";
    case 25:
      return "RAND_PLAY";
    case 26:
      return "RAND_DELAY";
    case 27:
      return "RAND_PB";
    case 28:
      return "PB";
    case 29:
      return "ADD_PB";
    case 30:
      return "SET_REGISTER";
    case 31:
      return "SET_REGISTER_RAND";
    case 32:
      return "INC_REGISTER";
    case 33:
      return "DEC_REGISTER";
    case 34:
      return "TEST_REGISTER";
    case 35:
      return "MARKER";
    case 36:
      return "GOTO_MARKER";
    case 37:
      return "GOTO_RANDOM_MARKER";
    case 38:
      return "WAIT_FOR_ALL_VOICES";
    case 39:
      return "PLAY_CYCLE";
    case 40:
      return "ADD_REGISTER";
    case 41:
      return "KEY_OFF_VOICES";
    case 42:
      return "KILL_VOICES";
    case 43:
      return "ON_STOP_MARKER";
    case 44:
      return "COPY_REGISTER";
    default:
      return "UNKNOWN";
  }
}

// Parse all grains from a V1 SBlk grain table (40 bytes per grain).
static std::vector<FullGrainInfo> parse_grains_v1(BinaryReader grains, int num_grains) {
  std::vector<FullGrainInfo> result;
  for (int i = 0; i < num_grains; i++) {
    u32 pos = grains.get_seek();
    FullGrainInfo g;
    g.type = grains.read<u32>();
    g.delay = grains.read<s32>();
    if (g.is_tone()) {
      g.priority = grains.read<s8>();
      g.vol = grains.read<s8>();
      g.center_note = grains.read<s8>();
      g.center_fine = grains.read<s8>();
      g.pan = grains.read<s16>();
      g.map_low = grains.read<s8>();
      g.map_high = grains.read<s8>();
      g.pb_low = grains.read<s8>();
      g.pb_high = grains.read<s8>();
      g.adsr1 = grains.read<u16>();
      g.adsr2 = grains.read<u16>();
      g.tone_flags = grains.read<u16>();
      g.sample_offset = grains.read<u32>();
    } else if (g.type == 26) {  // RAND_DELAY
      g.rand_delay_amount = grains.read<s32>();
    } else if (g.is_lfo()) {
      g.lfo_which = grains.read<u8>();
      g.lfo_target = grains.read<u8>();
      g.lfo_target_extra = grains.read<u8>();
      g.lfo_shape = grains.read<u8>();
      g.lfo_duty_cycle = grains.read<u16>();
      g.lfo_depth = grains.read<u16>();
      g.lfo_flags = grains.read<u16>();
      g.lfo_start_offset = grains.read<u16>();
      g.lfo_step_size = grains.read<u32>();
    } else if (g.is_playsound()) {
      g.play_vol = grains.read<s32>();
      g.play_pan = grains.read<s32>();
      for (auto& r : g.play_reg) {
        r = grains.read<s8>();
      }
      g.play_sound_id = grains.read<s32>();
      char snd_name[17] = {};
      for (int n = 0; n < 16; n++) {
        snd_name[n] = (char)grains.read<u8>();
      }
      g.play_snd_name = snd_name;
    } else {
      for (auto& p : g.ctrl)
        p = grains.read<s16>();
    }
    grains.set_seek(pos + 0x28);
    result.push_back(std::move(g));
  }
  return result;
}

// Parse all grains from a V2 SBlk (8-byte opcode+delay; Tone data in separate grain_data blob).
static std::vector<FullGrainInfo> parse_grains_v2(BinaryReader grains,
                                                  BinaryReader grain_data,
                                                  int num_grains) {
  std::vector<FullGrainInfo> result;
  for (int i = 0; i < num_grains; i++) {
    u32 opcode = grains.read<u32>();
    FullGrainInfo g;
    g.type = opcode >> 24;
    g.delay = grains.read<s32>();
    u32 value = opcode & 0xFFFFFF;
    if (g.is_tone()) {
      auto r = grain_data.at(value);
      g.priority = r.read<s8>();
      g.vol = r.read<s8>();
      g.center_note = r.read<s8>();
      g.center_fine = r.read<s8>();
      g.pan = r.read<s16>();
      g.map_low = r.read<s8>();
      g.map_high = r.read<s8>();
      g.pb_low = r.read<s8>();
      g.pb_high = r.read<s8>();
      g.adsr1 = r.read<u16>();
      g.adsr2 = r.read<u16>();
      g.tone_flags = r.read<u16>();
      g.sample_offset = r.read<u32>();
    } else if (g.type == 26) {  // RAND_DELAY
      g.rand_delay_amount = (s32)value + 1;
    } else if (g.is_lfo()) {
      auto r = grain_data.at(value);
      g.lfo_which = r.read<u8>();
      g.lfo_target = r.read<u8>();
      g.lfo_target_extra = r.read<u8>();
      g.lfo_shape = r.read<u8>();
      g.lfo_duty_cycle = r.read<u16>();
      g.lfo_depth = r.read<u16>();
      g.lfo_flags = r.read<u16>();
      g.lfo_start_offset = r.read<u16>();
      g.lfo_step_size = r.read<u32>();
    } else if (g.is_playsound()) {
      auto r = grain_data.at(value);
      g.play_vol = r.read<s32>();
      g.play_pan = r.read<s32>();
      for (auto& p : g.play_reg) {
        p = r.read<s8>();
      }
      g.play_sound_id = r.read<s32>();
      char snd_name[17] = {};
      for (int n = 0; n < 16; n++) {
        snd_name[n] = (char)r.read<u8>();
      }
      g.play_snd_name = snd_name;
    } else {
      g.ctrl[0] = (s8)(opcode & 0xFF);
      g.ctrl[1] = (s8)((opcode >> 8) & 0xFF);
      g.ctrl[2] = (s8)((opcode >> 16) & 0xFF);
    }
    result.push_back(std::move(g));
  }
  return result;
}

static void write_sfx_metadata(const std::vector<SoundFullInfo>& sounds,
                               const fs::path& output_path,
                               const std::string& bank_name,
                               u32 version,
                               u32 bank_id,
                               s16 num_grains_hdr) {
  std::string out;
  out += fmt::format("# {}  version={}  bank_id=0x{:08x}\n", bank_name, version, bank_id);
  out += fmt::format("# sounds={}  grains={}\n\n", sounds.size(), num_grains_hdr);
  for (const auto& snd : sounds) {
    out +=
        fmt::format("[{:03d}] \"{}\"  vol={}  volgroup={}  pan={}  instlimit={}  flags=0x{:04x}\n",
                    snd.index, snd.name, (int)snd.vol, (int)snd.volgroup, (int)snd.pan,
                    (int)snd.instance_limit, snd.flags);
    for (size_t gi = 0; gi < snd.grains.size(); gi++) {
      const auto& g = snd.grains[gi];
      out += fmt::format("  grain[{}] {}  delay={}", gi, grain_type_name(g.type), g.delay);
      if (g.is_tone()) {
        out += fmt::format(
            "  priority={}  vol={}  note={}  fine={}  pan={}"
            "  map=[{},{}]  pb=[{},{}]  adsr=[0x{:04x},0x{:04x}]  flags=0x{:04x}"
            "  offset=0x{:08x}",
            (int)g.priority, (int)g.vol, (int)g.center_note, (int)g.center_fine, (int)g.pan,
            (int)g.map_low, (int)g.map_high, (int)g.pb_low, (int)g.pb_high, g.adsr1, g.adsr2,
            g.tone_flags, g.sample_offset);
        if (g.has_loop) {
          out += "  loop=1";
        }
        if (!g.wav_filename.empty())
          out += fmt::format("  -> {}", g.wav_filename);
      } else if (g.type == 26) {
        out += fmt::format("  amount={}", g.rand_delay_amount);
      } else if (g.is_lfo()) {
        out += fmt::format(
            "  which_lfo={}  target={}  target_extra={} shape={}"
            "  duty_cycle={}  depth={}  flags=0x{:04x}  start_offset={}  step_size={}",
            g.lfo_which, g.lfo_target, g.lfo_target_extra, g.lfo_shape, g.lfo_duty_cycle,
            g.lfo_depth, g.lfo_flags, g.lfo_start_offset, g.lfo_step_size);
      } else if (g.is_playsound()) {
        out += fmt::format("  vol={}  pan={}  reg=[{},{},{},{}]  sound_id={}", g.play_vol,
                           g.play_pan, (int)g.play_reg[0], (int)g.play_reg[1], (int)g.play_reg[2],
                           (int)g.play_reg[3], g.play_sound_id);
        if (!g.play_snd_name.empty()) {
          out += fmt::format("  sndname=\"{}\"", g.play_snd_name);
        }
      } else {
        out += fmt::format("  param=[{}, {}, {}, {}]", g.ctrl[0], g.ctrl[1], g.ctrl[2], g.ctrl[3]);
      }
      out += '\n';
    }
    if (!snd.grains.empty())
      out += '\n';
  }
  file_util::write_text_file(output_path, out);
}

void write_sample_wav(const u8* sample_data,
                      size_t sample_data_size,
                      u32 offset,
                      s32 sample_rate,
                      const fs::path& out_path) {
  if (offset >= sample_data_size) {
    lg::warn("[sbk] sample offset 0x{:x} out of bounds (data size 0x{:x})", offset,
             sample_data_size);
    return;
  }
  size_t size = adpcm_sample_size(sample_data + offset, sample_data_size - offset);
  if (size == 0)
    return;
  auto pcm = decode_spu_adpcm(sample_data + offset, size);
  if (!pcm.empty()) {
    write_wave_file(pcm, {}, sample_rate, out_path);
  }
}

// Parse the SFXBlock name table and return a map from sound index → name.
std::unordered_map<u32, std::string> parse_sfx_names(std::span<const u8> bank_data,
                                                     u32 block_names_offset) {
  std::unordered_map<u32, std::string> result;
  if (block_names_offset == 0 || block_names_offset >= bank_data.size())
    return result;

  // These structs match the binary layout used by the 989snd loader.
  struct SFXBlockNames {
    u32 BlockName[2];           // 8 bytes  — block name (padded)
    u32 SFXNameTableOffset;     // 4 bytes  — offset from start of SFXBlockNames
    u32 VAGNameTableOffset;     // 4 bytes
    u32 VAGImportsTableOffset;  // 4 bytes
    u32 VAGExportsTableOffset;  // 4 bytes
    s16 SFXHashOffsets[32];     // 64 bytes — hash bucket start indices into name table
    s16 VAGHashOffsets[32];     // 64 bytes
  };  // 152 bytes total

  struct SFXName {
    u32 Name[4];   // 16 bytes — null-padded ASCII name
    s16 Index;     // 2 bytes  — sound index this name maps to
    s16 reserved;  // 2 bytes
  };  // 20 bytes total

  static_assert(sizeof(SFXBlockNames) == 152);
  static_assert(sizeof(SFXName) == 20);

  if (block_names_offset + sizeof(SFXBlockNames) > bank_data.size())
    return result;

  SFXBlockNames names_hdr;
  memcpy(&names_hdr, bank_data.data() + block_names_offset, sizeof(names_hdr));

  u32 name_table_base = block_names_offset + names_hdr.SFXNameTableOffset;
  if (name_table_base >= bank_data.size())
    return result;

  size_t max_entries = (bank_data.size() - name_table_base) / sizeof(SFXName);
  const auto* name_table = reinterpret_cast<const SFXName*>(bank_data.data() + name_table_base);

  // Walk each hash bucket chain until an empty entry (Name[0] == 0).
  std::unordered_set<u32> seen_indices;
  for (s16 bucket : names_hdr.SFXHashOffsets) {
    if (bucket < 0 || (size_t)bucket >= max_entries)
      continue;
    const SFXName* entry = name_table + bucket;
    while (entry->Name[0] != 0 && (size_t)(entry - name_table) < max_entries) {
      char buf[17];
      buf[16] = '\0';
      memcpy(buf, entry->Name, 16);
      std::string name(buf);
      auto null_pos = name.find('\0');
      if (null_pos != std::string::npos)
        name.resize(null_pos);

      u32 idx = (u32)(u16)entry->Index;
      if (!name.empty() && !seen_indices.count(idx)) {
        result[idx] = name;
        seen_indices.insert(idx);
      }
      ++entry;
    }
  }
  return result;
}

// Parse the on-disc SBK name table that precedes the FileAttributes header.
// Returns a vector of sound names in order (0-indexed, matching SBlk sound indices).
std::vector<std::string> parse_disc_name_table(std::span<const u8> file_data) {
  if (file_data.size() < 24)
    return {};

  BinaryReader reader(file_data);
  reader.ffwd(16);     // bank name
  reader.read<u32>();  // reserved
  u32 count = reader.read<u32>();

  if (count == 0 || count > 100000)
    return {};

  std::vector<std::string> names;
  names.reserve(count);

  for (u32 i = 0; i < count; i++) {
    auto arr = reader.read<std::array<char, 16>>();
    reader.read<u32>();  // falloff_params

    std::string name(arr.data(), 16);
    auto null_pos = name.find('\0');
    if (null_pos != std::string::npos)
      name.resize(null_pos);
    names.push_back(std::move(name));
  }

  return names;
}

// Find the FileAttributes header offset.
// Jak 2/3: FA header is at byte 0 (no on-disc name table prefix).
// Jak 1: FA header is at a 2048-byte sector boundary after the on-disc name table.
// Returns SIZE_MAX if no valid header is found.
size_t find_fa_offset(std::span<const u8> file_data) {
  auto is_valid_fa = [](u32 type, u32 num_chunks) {
    return (type == 1 || type == 3) && num_chunks >= 2 && num_chunks <= 4;
  };
  // Check offset 0 first (Jak 2/3 format)
  if (file_data.size() >= 8) {
    u32 type, num_chunks;
    memcpy(&type, file_data.data(), 4);
    memcpy(&num_chunks, file_data.data() + 4, 4);
    if (is_valid_fa(type, num_chunks))
      return 0;
  }
  // Scan 2048-byte boundaries (Jak 1 format)
  for (size_t offset = 2048; offset + 8 <= file_data.size(); offset += 2048) {
    u32 type, num_chunks;
    memcpy(&type, file_data.data() + offset, 4);
    memcpy(&num_chunks, file_data.data() + offset + 4, 4);
    if (is_valid_fa(type, num_chunks))
      return offset;
  }
  return SIZE_MAX;
}

// Extract samples and write a metadata text file from an SBlk (SFX) bank.
void extract_sfx_block(std::span<const u8> bank_data,
                       std::span<const u8> sample_data,
                       const fs::path& output_dir,
                       const std::string& bank_name,
                       const std::vector<std::string>& disc_names = {}) {
  BinaryReader data(bank_data);

  if (data.read<u32>() != fcc("SBlk")) {
    lg::warn("[SBK] Expected SBlk FourCC");
    return;
  }

  u32 version = data.read<u32>();
  u32 flags = data.read<u32>();
  u32 bank_id = data.read<u32>();
  data.read<s8>();  // BankNum
  data.read<s8>();
  data.read<s16>();
  data.read<s16>();  // padding

  s16 num_sounds = data.read<s16>();
  s16 num_grains_hdr = data.read<s16>();
  data.read<s16>();  // NumVAGs
  u32 first_sound = data.read<u32>();
  u32 first_grain = data.read<u32>();
  data.read<u32>();  // VagsInSR
  data.read<u32>();  // VagDataSize
  data.read<u32>();  // SRAMAllocSize
  data.read<u32>();  // NextBlock

  u32 grain_data_offset = 0;
  if (version >= 2)
    grain_data_offset = data.read<u32>();

  u32 block_names_offset = data.read<u32>();
  data.read<u32>();  // SFXUD

  auto sound_names = parse_sfx_names(bank_data, (flags & 0x100) ? block_names_offset : 0);
  file_util::create_dir_if_needed(output_dir);

  // Parse all sounds and grains, assigning WAV filenames to TONE grains.
  std::vector<SoundFullInfo> all_sounds;
  all_sounds.reserve(num_sounds);

  data.set_seek(first_sound);
  for (int i = 0; i < num_sounds; i++) {
    SoundFullInfo snd;
    snd.index = i;
    snd.vol = data.read<s8>();
    snd.volgroup = data.read<s8>();
    snd.pan = data.read<s16>();
    s8 num_grains = data.read<s8>();
    snd.instance_limit = data.read<s8>();
    snd.flags = data.read<u16>();
    u32 first_sfx_grain = data.read<u32>();

    auto it = sound_names.find((u32)i);
    if (it != sound_names.end()) {
      snd.name = it->second;
    } else if ((size_t)i < disc_names.size() && !disc_names[i].empty()) {
      snd.name = disc_names[i];
    } else {
      snd.name = fmt::format("sfx_{:04d}", i);
    }

    if (num_grains > 0) {
      auto grains_reader = data.at(first_grain + first_sfx_grain);
      if (version < 2) {
        snd.grains = parse_grains_v1(grains_reader, num_grains);
      } else if (grain_data_offset > 0) {
        snd.grains = parse_grains_v2(grains_reader, data.at(grain_data_offset), num_grains);
      }
    }

    // Assign WAV filenames to TONE grains (single → name.wav; multi → name_0.wav, etc.).
    std::vector<size_t> tone_indices;
    for (size_t gi = 0; gi < snd.grains.size(); gi++) {
      if (snd.grains[gi].is_tone())
        tone_indices.push_back(gi);
    }
    for (size_t vi = 0; vi < tone_indices.size(); vi++) {
      auto& g = snd.grains[tone_indices[vi]];
      g.wav_filename =
          (tone_indices.size() == 1) ? snd.name + ".wav" : fmt::format("{}_{}.wav", snd.name, vi);
    }

    all_sounds.push_back(std::move(snd));
  }

  // annotate child sound grains with the referenced sound name
  for (auto& snd : all_sounds) {
    for (auto& g : snd.grains) {
      if (!g.is_playsound() || g.play_sound_id < 0 || !g.play_snd_name.empty()) {
        continue;
      }
      if ((size_t)g.play_sound_id < all_sounds.size()) {
        g.play_snd_name = all_sounds[g.play_sound_id].name;
      }
    }
  }

  for (auto& snd : all_sounds) {
    for (auto& g : snd.grains) {
      if (!g.is_tone() || g.sample_offset >= sample_data.size()) {
        continue;
      }
      size_t sz = adpcm_sample_size(sample_data.data() + g.sample_offset,
                                    sample_data.size() - g.sample_offset);
      g.has_loop = adpcm_has_loop(sample_data.data() + g.sample_offset, sz);
    }
  }

  // Write WAV files.
  for (const auto& snd : all_sounds) {
    for (const auto& g : snd.grains) {
      if (!g.is_tone() || g.wav_filename.empty())
        continue;
      s32 rate = compute_sample_rate(g.center_note, g.center_fine);
      write_sample_wav(sample_data.data(), sample_data.size(), g.sample_offset, rate,
                       output_dir / g.wav_filename);
    }
  }

  // Write metadata text file alongside the WAVs.
  write_sfx_metadata(all_sounds, output_dir / "metadata.txt", bank_name, version, bank_id,
                     num_grains_hdr);

  lg::info("[SBK] Extracted {} SFX sounds to {}", num_sounds, output_dir.string());
}

// Extract instrument tone samples from an SBv2 (music) bank.
void extract_music_bank(std::span<const u8> bank_data,
                        std::span<const u8> sample_data,
                        const fs::path& output_dir) {
  BinaryReader data(bank_data);

  if (data.read<u32>() != fcc("SBv2")) {
    lg::warn("[sbk] expected SBv2 fcc");
    return;
  }

  data.read<u32>();  // Version
  data.read<u32>();  // Flags
  data.read<u32>();  // BankID
  data.read<s8>();   // BankNum
  // Padding: 1 byte + 1 short (matches 989snd loader byte sequence)
  data.read<s8>();
  data.read<s16>();

  data.read<s16>();  // NumSounds
  s16 num_progs = data.read<s16>();
  data.read<s16>();  // NumTones
  data.read<s16>();  // NumVAGs

  data.read<u32>();  // FirstSound
  u32 first_prog = data.read<u32>();
  data.read<u32>();  // FirstTone (per-program FirstTone overrides this)
  data.read<u32>();  // VagsInSR
  data.read<u32>();  // VagDataSize

  file_util::create_dir_if_needed(output_dir);

  // Each program entry is 8 bytes: s8 NumTones, s8 Vol, s16 Pan, u32 FirstTone
  data.set_seek(first_prog);
  for (int p = 0; p < num_progs; p++) {
    s8 num_tones = data.read<s8>();
    data.read<s8>();   // Vol
    data.read<s16>();  // Pan
    u32 first_tone = data.read<u32>();

    // Tone structs: 24 bytes each. Layout: Priority(0), Vol(1), CenterNote(2), CenterFine(3),
    // Pan(4), MapLow(6), MapHigh(7), PBLow(8), PBHigh(9), ADSR1(10), ADSR2(12), Flags(14),
    // SampleOffset(16), reserved(20).
    auto tones = data.at(first_tone);
    for (int t = 0; t < num_tones; t++) {
      tones.ffwd(2);  // Priority, Vol
      s8 center_note = tones.read<s8>();
      s8 center_fine = tones.read<s8>();
      tones.ffwd(12);  // Pan through Flags
      u32 sample_offset = tones.read<u32>();
      tones.read<u32>();  // reserved

      s32 rate = compute_sample_rate(center_note, center_fine);
      auto filename = fmt::format("prog_{:03d}_tone_{:03d}.wav", p, t);
      write_sample_wav(sample_data.data(), sample_data.size(), sample_offset, rate,
                       output_dir / filename);
    }
  }

  lg::info("[sbk] extracted {} programs to {}", num_progs, output_dir.string());
}

void extract_sbk(const fs::path& sbk_path, const fs::path& output_dir) {
  auto file_data = file_util::read_binary_file(sbk_path);
  if (file_data.empty()) {
    lg::warn("[sbk] could not read {}", sbk_path.string());
    return;
  }

  std::span<const u8> file_span(file_data);

  size_t fa_offset = find_fa_offset(file_span);
  if (fa_offset == SIZE_MAX) {
    lg::warn("[sbk] could not find FileAttributes header in {}", sbk_path.string());
    return;
  }

  // Jak 1: FA header is preceded by an on-disc name table (provides per-sound names).
  // Jak 2/3: FA header is at offset 0 — no name table prefix.
  std::vector<std::string> disc_names;
  if (fa_offset > 0) {
    disc_names = parse_disc_name_table(file_span);
  }

  // FileAttributes header: u32 type, u32 num_chunks, then num_chunks * {u32 offset, u32 size}
  // All chunk offsets are relative to the start of the FileAttributes section.
  auto fa_span = file_span.subspan(fa_offset);
  BinaryReader reader(fa_span);
  u32 fa_type = reader.read<u32>();
  u32 num_chunks = reader.read<u32>();

  if (fa_type != 1 && fa_type != 3) {
    lg::warn("[sbk] unknown file type {} in {}", fa_type, sbk_path.string());
    return;
  }
  if (num_chunks < 2) {
    lg::warn("[sbk] {} only has {} chunk(s)", sbk_path.string(), num_chunks);
    return;
  }

  struct ChunkInfo {
    u32 offset, size;
  };
  std::vector<ChunkInfo> chunks(num_chunks);
  for (auto& c : chunks) {
    c.offset = reader.read<u32>();
    c.size = reader.read<u32>();
  }

  auto bank_data = fa_span.subspan(chunks[0].offset, chunks[0].size);
  auto sample_data = fa_span.subspan(chunks[1].offset, chunks[1].size);

  BinaryReader bank_reader(bank_data);
  u32 fourcc_val = bank_reader.read<u32>();

  // Output goes into a subdirectory named after the SBK stem (e.g. COMMON.SBK → COMMON/)
  auto bank_out_dir = output_dir / sbk_path.stem().string();

  if (fourcc_val == fcc("SBlk")) {
    lg::info("[sbk] {} — sfx block", sbk_path.filename().string());
    extract_sfx_block(bank_data, sample_data, bank_out_dir, sbk_path.stem().string(), disc_names);
  } else if (fourcc_val == fcc("SBv2")) {
    if (num_chunks < 3) {
      lg::warn("[sbk] SBv2 bank {} is missing its midi chunk", sbk_path.string());
      return;
    }
    lg::info("[sbk] {} — music bank", sbk_path.filename().string());
    extract_music_bank(bank_data, sample_data, bank_out_dir);
  } else {
    lg::warn("[sbk] unrecognised bank fcc 0x{:08x} in {}", fourcc_val, sbk_path.string());
  }
}

}  // namespace

void extract_sbk_files(const fs::path& input_dir, const fs::path& output_dir) {
  if (!fs::exists(input_dir)) {
    lg::warn("[sbk] input directory {} does not exist", input_dir.string());
    return;
  }

  file_util::create_dir_if_needed(output_dir);

  for (const auto& entry : fs::directory_iterator(input_dir)) {
    auto ext = entry.path().extension().string();
    std::ranges::transform(ext, ext.begin(), tolower);
    if (ext == ".sbk") {
      extract_sbk(entry.path(), output_dir);
    }
  }
}

}  // namespace decompiler
