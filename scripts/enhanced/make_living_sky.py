"""Jak Living Sky: the mod's ZIP for OpenGOAL's launcher (mods.json).

    python scripts/enhanced/make_living_sky.py --bin build/bin --version v0.1

The programs and data/ at the ZIP's root, as the launcher wants them; it unpacks the ZIP into its own
mod folder and runs the extractor itself. Only files tracked by git go in, so nothing from the game
can slip into it, and nothing that only Jak 2, Jak 3 or the left-out effects use.
"""

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "Jak-Living-Sky"
DATA = ["decompiler/config", "goal_src", "game/assets", "game/graphics/opengl_renderer/shaders", "custom_assets"]
LEFT_OUT = ("custom_assets/jak2/", "custom_assets/jak3/", "game/assets/enhanced/art/",
            "game/assets/enhanced/bigmap/", "game/assets/enhanced/grass_paint/")
PROGRAMS = ("gk.exe", "goalc.exe", "extractor.exe", "decompiler.exe")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, default=ROOT / "build/bin")
    parser.add_argument("--version", default="dev")
    parser.add_argument("--out", type=Path, default=ROOT / "out/release")
    args = parser.parse_args()

    dest = args.out / NAME
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for exe in PROGRAMS:
        shutil.copy2(args.bin / exe, dest / exe)

    count = 0
    for rel in git("ls-files", "-z", *DATA).decode().split("\0"):
        if rel and not rel.startswith(LEFT_OUT):
            target = dest / "data" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
            count += 1

    # no objects added to the world: the models aren't part of this mod
    placed = dest / "data/game/assets/enhanced/placed.json"
    games = json.loads(placed.read_text(encoding="utf-8"))
    placed.write_text(json.dumps({game: [] for game in games}), encoding="utf-8", newline="\n")

    (dest / "data/launcher").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / ".github/scripts/releases/error-code-metadata.json",
                 dest / "data/launcher/error-code-metadata.json")
    (dest / "data/log").mkdir(exist_ok=True)
    # the launcher compiles the game again when this changes
    rev = git("rev-parse", "--short", "HEAD").decode().strip()
    (dest / "data/VERSION").write_text(f"{args.version}-{rev}", encoding="ascii")
    # the launcher's single-game build (common/util/enhanced_light.h)
    (dest / "data/enhanced-light").write_text("Jak Living Sky\n", encoding="ascii")

    zip_path = args.out / f"{NAME}-{args.version}.zip"
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(dest.rglob("*")):
            z.write(f, f.relative_to(dest))
    print(f"{count} data files; {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
