#pragma once

#include <string>
#include <vector>

#include "common/common_types.h"

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
  bool scan = false;
  bool dump = false;    // (edit: the game to send the skeletons of the characters near, joint_dump)    // (edit: the game to list the characters near the camera, actors_begin / actor_anim)
  std::string speaker;  // utf-8
  std::string text;     // utf-8
};
// once a frame, from the game's thread: the state now (starting the voices due)
Frame poll();
// the free camera, from the game (edit), game units: where it is, what it looks at, the first solid thing
// along its view (where a character is put by "where the camera aims")
void report_camera(const float* pos, const float* look, const float* aim);

// the characters placed now (after poll, same frame): where (game units) and which way they face (degrees,
// round the vertical: 0 looks down +z)
struct Place {
  std::string actor;
  float pos[3] = {};
  float yaw = 0.f;
};
bool place(int i, Place* out);
// (the game's list) where a character near the camera is now, and which way it faces (game units, degrees)
void actor_at(const std::string& actor, const float* pos, float yaw);
// (edit) a character or thing held with R2 and moved with the pad: its place at the cursor (game units,
// degrees): that key moved, or a new one
void grab(const std::string& actor, const float* pos, float yaw);

// a body part of a character turned (degrees round its own x y z) and moved (game units, in its parent's
// space) on top of its animation, now (after poll, same frame)
struct Pose {
  std::string actor;
  std::string joint;
  float rot[3] = {};
  float off[3] = {};
};
bool pose(int i, Pose* out);
// (the game's list) a body part of a character near the camera
void actor_joint(const std::string& actor, const std::string& joint);
// (edit) the body part picked in the menu, for the pad (false: none, R2 takes the whole character)
bool pose_selected(std::string* actor, std::string* joint);
// (edit) a body part turned or moved with the pad: its key at the cursor (made or changed)
void pose_grab(const std::string& actor, const std::string& joint, const float* rot, const float* off);
// (edit) a body part picked with the pad (X on it): the one the pad works on, shown in the menu (empty: none)
void pose_pick(const std::string& actor, const std::string& joint);

// (edit) the pad's own cursor, moved by turning the controller itself (its gyroscope): -1..1 across and up the
// screen; Start held puts it back in the middle (Start doesn't reach the game while editing)
bool editing();
// (edit) back one change (-1), or on again (1): the D-pad's left and right
void undo_redo(int dir);

// a character's face shapes set by hand now (-1..1 each, up to 16), after poll
struct Face {
  std::string actor;
  float w[16] = {};
};
bool face(int i, Face* out);
// (the game's list) how many face shapes a character near the camera has
void actor_face(const std::string& actor, int count);
void gyro(const float* rad_per_s, float dt);
void start_button(bool down);
// where it is (false: no gyroscope, it stays in the middle)
bool cursor(float* x, float* y);

// the characters played as puppets now (after poll, same frame): by the game's name ("jak" for Jak), one of
// its animations by name, how far into it (seconds, at its own speed), looping or held on its last frame
struct Puppet {
  std::string actor;
  std::string anim;
  float secs = 0.f;
  bool loop = true;
};
bool puppet(int i, Puppet* out);
// what the game made of the i-th: 0 its character not found, 1 its animation not found, 2 played
void puppet_status(int i, int status);
// the characters near the camera and their animations, from the game (when asked by poll's scan)
void actors_begin();
void actor_anim(const std::string& actor, const std::string& anim);

// the Enhanced menu's Cutscenes section (the render thread)
void draw_menu();
// the timeline bar at the bottom of the screen while editing (the render thread, the menu closed)
void draw_overlay();
// (edit) the pad on the timeline: buttons pressed this frame and held (the game's bits), the sticks (-1..1, up
// positive); true while the timeline has it (the camera and the characters are left alone)
bool timeline_pad(u32 pressed, u32 held, float lx, float ly, float rx);

// captures for the one making the scenes (Claude): pictures of the scene, a sheet of each face shape, the
// characters' skeletons, under custom_assets/<game>/cutscenes/captures/<scene>
// (the render thread, once a frame: a picture wanted now, where: empty none)
std::string capture_path();
// (the game, when the scan's dump is asked) a character's joint: its index, its parent's (-1), its name, its
// place in the world (16 floats, game units)
void joint_dump(const std::string& actor, int j, int parent, const std::string& name, const float* m);

}  // namespace enhanced_cutscene

// (in kmachine.cpp: the game's mp3 player) start a file under custom_assets/<game>/audio, from start_s in
bool enhanced_play_mp3(const std::string& file, unsigned volume, float start_s);
void enhanced_stop_mp3(const std::string& file);
float enhanced_mp3_length(const std::string& file);
std::vector<float> enhanced_mp3_envelope(const std::string& file, float fps);
