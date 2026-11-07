#pragma once

#include <string>

/*!
 * Jak Living Sky: scenes made in the game (the Enhanced menu's Cutscenes editor): camera shots and voice lines
 * with their subtitles on one timeline, saved in custom_assets/<game>/cutscenes/<name>.json (metres, seconds).
 *
 * The game plays one by name (pc-cutscene-play) and, every frame it runs, asks where the camera is and what
 * subtitle shows (pc-cutscene-poll); the voices are started from here, on the same clock, so the shots and the
 * subtitles keep time with them whatever the frame rate. In the editor the camera is flown with the pad by the
 * game, which tells where it is (pc-cutscene-report): "add shot" takes what it sees.
 */
namespace enhanced_cutscene {

// what's going on: none, a scene played by the game, the editor's free camera, the editor previewing
enum Mode { kNone = 0, kPlay = 1, kEdit = 2, kPreview = 3 };

// the game asks for a scene; false if there's none by that name
bool play(const std::string& name);
// skipped, or over
void stop();

struct Frame {
  int mode = kNone;
  float pos[3] = {};   // the camera, game units
  float look[3] = {};  // what it looks at, game units
  bool has_cam = false;  // (play, preview: a shot to look through; none: the game's own camera)
  bool jump = false;   // (edit: the game's free camera to be put here, once)
  bool letterbox = true;
  bool hold_jak = true;
  std::string speaker;  // utf-8
  std::string text;     // utf-8
};
// once a frame, from the game's thread: the state now (starting the voices due)
Frame poll();
// the free camera, from the game (edit), game units
void report_camera(const float* pos, const float* look);

// the Enhanced menu's Cutscenes section (the render thread)
void draw_menu();

}  // namespace enhanced_cutscene

// (in kmachine.cpp: the game's mp3 player) start a file under custom_assets/<game>/audio, from start_s in
bool enhanced_play_mp3(const std::string& file, unsigned volume, float start_s);
void enhanced_stop_mp3(const std::string& file);
float enhanced_mp3_length(const std::string& file);
