#pragma once

#include "common/util/FileUtil.h"

namespace decompiler {

// extract all sound banks into wav files under output_dir.
// SBlk (SFX) banks write one wav per named sound.
// SBv2 (music) banks write one wav per instrument tone.
void extract_sbk_files(const fs::path& input_dir, const fs::path& output_dir);

}  // namespace decompiler
