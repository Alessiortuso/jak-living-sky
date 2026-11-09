#pragma once

#include <vector>

#include "common/common_types.h"
#include "common/util/FileUtil.h"

namespace sbk {

struct BuildOptions {
  u32 bank_id = 0;
  bool jak1_format = false;  // prepend 2048-byte on-disc name table (Jak 1 SBK layout)
};

// One logical sound with one or more variant WAV files.
// Each variant becomes a TONE grain; for multiple variants a RAND_PLAY grain is auto-inserted.
struct SoundSpec {
  std::vector<fs::path> variants;
};

// Create a new SBK from a list of sound specs (JSON manifest path).
void create_sbk(const std::vector<SoundSpec>& sounds,
                const fs::path& output,
                const BuildOptions& opts = {});

// Append sound specs as new sounds to an existing V1 SBlk bank.
void append_sbk(const fs::path& input,
                const std::vector<SoundSpec>& sounds,
                const fs::path& output);

// Create a new SBK from a metadata directory (output of extract_sbk).
// The directory must contain metadata.txt and the WAV files it references.
void create_sbk_from_dir(const fs::path& dir,
                         const fs::path& output,
                         const BuildOptions& opts = {});

// Append sounds from a metadata directory to an existing V1 SBlk bank.
void append_sbk_from_dir(const fs::path& input,
                          const fs::path& dir,
                          const fs::path& output);

}  // namespace sbk
