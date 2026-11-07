#include "enhanced_cutscene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <atomic>
#include <map>
#include <mutex>
#include <vector>

#include "common/log/log.h"
#include "common/util/FileUtil.h"
#include "common/util/json_util.h"
#include "common/versions/versions.h"

#include "game/graphics/opengl_renderer/EnhancedPost.h"
#include "game/runtime.h"

#include "third-party/imgui/imgui.h"

namespace enhanced_cutscene {
namespace {

constexpr float kMeter = 4096.f;

struct Shot {
  float t = 0.f;
  float pos[3] = {};   // metres
  float look[3] = {};  // metres: a point it looks at
  bool smooth = false;  // moving on to the next shot, not cut to it
};

struct Sub {
  float t = 0.f;  // from its line's start
  std::string text;
};

struct Line {
  float t = 0.f;
  std::string file;  // under custom_assets/<game>/audio, e.g. voice/fisher-ask.mp3
  std::string speaker;
  int volume = 200;
  float len = 0.f;  // its mp3's length
  std::vector<Sub> subs;
};

// from t on, a character plays an animation (none: back to its own)
struct AnimKey {
  float t = 0.f;
  std::string actor;  // the game's name for it ("jak" for Jak: Daxter follows him)
  std::string anim;   // one of its art group's animations; empty: its own again
  bool loop = true;
  float speed = 1.f;
};

// at t a character is at pos (metres), facing yaw (degrees); smooth: sliding on to its next key
struct MoveKey {
  float t = 0.f;
  std::string actor;
  float pos[3] = {};
  float yaw = 0.f;
  bool smooth = false;
};

// at t a body part is turned (degrees round its own x y z) and moved (metres, in its parent's space) on top
// of its animation; smooth: on to its next key
struct PoseKey {
  float t = 0.f;
  std::string actor;
  std::string joint;
  float rot[3] = {};
  float off[3] = {};
  bool smooth = true;
};

// at t a character's face shapes are set by hand (-1..1 each); smooth: on to its next key
struct FaceKey {
  float t = 0.f;
  std::string actor;
  float w[16] = {};
  bool smooth = true;
};

struct Scene {
  std::string name;
  float length = 10.f;
  bool letterbox = true;
  bool hold_jak = true;
  std::vector<Shot> shots;
  std::vector<Line> lines;
  std::vector<AnimKey> anims;
  std::vector<MoveKey> moves;
  std::vector<PoseKey> poses;
  std::vector<FaceKey> faces;
};

using Clock = std::chrono::steady_clock;

std::mutex g_mutex;
Scene g_scene;          // the one played or edited
bool g_scene_loaded = false;
int g_mode = kNone;
Clock::time_point g_start;  // when its time 0 was (play, preview)
float g_from = 0.f;          // (preview: starting from)
std::vector<bool> g_started;  // each line's voice started
float g_cursor = 0.f;         // the editor's time
bool g_jump = false;          // (edit) the free camera put somewhere
float g_jump_pos[3] = {};
float g_jump_look[3] = {};
float g_cam_pos[3] = {};      // (edit) where the game's free camera is, metres
float g_cam_look[3] = {};
bool g_cam_known = false;
float g_cam_aim[3] = {};      // (edit) the first solid thing along its view, metres
std::vector<Place> g_places;  // (poll) the characters placed now
std::vector<Pose> g_poses;    // (poll) the body parts turned or moved now
std::vector<Face> g_faces;    // (poll) the faces set by hand now
std::map<std::string, int> g_actor_face;       // how many face shapes each character near has
std::map<std::string, int> g_actor_face_next;  // (with the list)
std::map<std::string, std::vector<std::string>> g_actor_joints;       // each character's body parts
std::map<std::string, std::vector<std::string>> g_actor_joints_next;  // (with the list)
std::string g_sel_actor;      // (edit) the body part picked for the pad
std::string g_sel_joint;
std::string g_pick_actor;     // (a pick from the pad, for the menu's character)
std::atomic<bool> g_edit_now{false};  // (editing: read by the input thread)
float g_cur_x = 0.f;          // the gyro cursor, -1..1
float g_cur_y = 0.f;
bool g_have_gyro = false;
bool g_start_held = false;
std::string g_held;           // (edit) what R2 holds now, for the menu
Clock::time_point g_held_at;
struct ActorAt {
  float pos[3] = {};  // metres
  float yaw = 0.f;    // degrees
};
std::map<std::string, ActorAt> g_actor_at;       // where the characters near the camera are
std::map<std::string, ActorAt> g_actor_at_next;  // (filled with the list, swapped in with it)
bool g_dirty = false;

// the editor's history (D-pad left / right, the menu's Undo / Redo): the scene as it was after each change,
// one change being what's done before a short stop (a stick held, a number dragged: one)
Clock::time_point g_last_change;
unsigned g_rev = 0;            // a change's count
unsigned g_committed_rev = 0;  // (the last one in the history)
Scene g_committed;             // the scene as the history last saw it
std::vector<Scene> g_undo;
std::vector<Scene> g_redo;
constexpr size_t kHistory = 100;

void changed() {
  g_dirty = true;
  g_rev++;
  g_last_change = Clock::now();
}

// (a scene picked, made, or played: a history of its own)
void history_reset() {
  g_undo.clear();
  g_redo.clear();
  g_committed = g_scene;
  g_committed_rev = g_rev;
}

// the change done (after a short stop, or when asked) into the history
void history_commit(bool now) {
  if (g_rev == g_committed_rev) {
    return;
  }
  if (!now && Clock::now() - g_last_change < std::chrono::milliseconds(400)) {
    return;
  }
  g_undo.push_back(g_committed);
  if (g_undo.size() > kHistory) {
    g_undo.erase(g_undo.begin());
  }
  g_redo.clear();
  g_committed = g_scene;
  g_committed_rev = g_rev;
}
std::vector<Puppet> g_puppets;  // (poll) the characters played now
std::vector<int> g_puppet_status;  // (the game's answer for each: puppet_status)
Clock::time_point g_edit_clock = Clock::now();  // (edit) loops play on from the cursor, live
// the characters near the camera, and their animations (the game's list, edit)
std::vector<std::pair<std::string, std::vector<std::string>>> g_actors;
// (the game's new list, filled apart and swapped in whole at the next poll: the menu, drawn meanwhile, never
// saw it half done - an empty list moved what's under it, a click landed on a Teleport button)
std::vector<std::pair<std::string, std::vector<std::string>>> g_actors_next;
bool g_actors_ready = false;
bool g_scan_wanted = false;
Clock::time_point g_last_scan;

fs::path dir() {
  return file_util::get_jak_project_dir() / "custom_assets" / game_version_names[g_game_version] / "cutscenes";
}

std::vector<std::string> scene_names() {
  std::vector<std::string> out;
  std::error_code ec;
  for (const auto& d : fs::directory_iterator(dir(), ec)) {
    if (d.path().extension() == ".json") {
      out.push_back(d.path().stem().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> voice_files() {
  std::vector<std::string> out;
  std::error_code ec;
  const auto audio = file_util::get_jak_project_dir() / "custom_assets" / game_version_names[g_game_version] / "audio";
  for (const auto& sub : {"voice", "sound_replacements"}) {
    for (const auto& d : fs::directory_iterator(audio / sub, ec)) {
      if (d.path().extension() == ".mp3" || d.path().extension() == ".wav") {
        out.push_back(std::string(sub) + "/" + d.path().filename().string());
      }
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

void sort_scene(Scene& s) {
  std::stable_sort(s.shots.begin(), s.shots.end(), [](const Shot& a, const Shot& b) { return a.t < b.t; });
  std::stable_sort(s.lines.begin(), s.lines.end(), [](const Line& a, const Line& b) { return a.t < b.t; });
  for (auto& l : s.lines) {
    std::stable_sort(l.subs.begin(), l.subs.end(), [](const Sub& a, const Sub& b) { return a.t < b.t; });
  }
  std::stable_sort(s.anims.begin(), s.anims.end(), [](const AnimKey& a, const AnimKey& b) { return a.t < b.t; });
  std::stable_sort(s.moves.begin(), s.moves.end(), [](const MoveKey& a, const MoveKey& b) { return a.t < b.t; });
  std::stable_sort(s.poses.begin(), s.poses.end(), [](const PoseKey& a, const PoseKey& b) { return a.t < b.t; });
  std::stable_sort(s.faces.begin(), s.faces.end(), [](const FaceKey& a, const FaceKey& b) { return a.t < b.t; });
}

bool load(const std::string& name, Scene* out) {
  const auto file = dir() / (name + ".json");
  if (!fs::exists(file)) {
    return false;
  }
  try {
    const auto j = parse_commented_json(file_util::read_text_file(file), file.string());
    Scene s;
    s.name = name;
    s.length = j.value("length", 10.f);
    s.letterbox = j.value("letterbox", true);
    s.hold_jak = j.value("hold_jak", true);
    for (const auto& js : j.value("shots", nlohmann::json::array())) {
      Shot sh;
      sh.t = js.value("t", 0.f);
      for (int k = 0; k < 3; k++) {
        sh.pos[k] = js["pos"][k].get<float>();
        sh.look[k] = js["look"][k].get<float>();
      }
      sh.smooth = js.value("smooth", false);
      s.shots.push_back(sh);
    }
    for (const auto& jl : j.value("lines", nlohmann::json::array())) {
      Line l;
      l.t = jl.value("t", 0.f);
      l.file = jl.value("file", std::string());
      l.speaker = jl.value("speaker", std::string());
      l.volume = jl.value("volume", 200);
      l.len = jl.value("len", 0.f);
      for (const auto& jsub : jl.value("subs", nlohmann::json::array())) {
        l.subs.push_back({jsub.value("t", 0.f), jsub.value("text", std::string())});
      }
      if (l.len <= 0.f && !l.file.empty()) {
        l.len = enhanced_mp3_length(l.file);
      }
      s.lines.push_back(l);
    }
    for (const auto& ja : j.value("anims", nlohmann::json::array())) {
      AnimKey k;
      k.t = ja.value("t", 0.f);
      k.actor = ja.value("actor", std::string());
      k.anim = ja.value("anim", std::string());
      k.loop = ja.value("loop", true);
      k.speed = ja.value("speed", 1.f);
      s.anims.push_back(k);
    }
    for (const auto& jm : j.value("moves", nlohmann::json::array())) {
      MoveKey k;
      k.t = jm.value("t", 0.f);
      k.actor = jm.value("actor", std::string());
      for (int c = 0; c < 3; c++) {
        k.pos[c] = jm["pos"][c].get<float>();
      }
      k.yaw = jm.value("yaw", 0.f);
      k.smooth = jm.value("smooth", false);
      s.moves.push_back(k);
    }
    for (const auto& jp : j.value("poses", nlohmann::json::array())) {
      PoseKey k;
      k.t = jp.value("t", 0.f);
      k.actor = jp.value("actor", std::string());
      k.joint = jp.value("joint", std::string());
      for (int c = 0; c < 3; c++) {
        k.rot[c] = jp["rot"][c].get<float>();
        k.off[c] = jp["off"][c].get<float>();
      }
      k.smooth = jp.value("smooth", true);
      s.poses.push_back(k);
    }
    for (const auto& jf : j.value("faces", nlohmann::json::array())) {
      FaceKey k;
      k.t = jf.value("t", 0.f);
      k.actor = jf.value("actor", std::string());
      const auto w = jf.value("w", nlohmann::json::array());
      for (size_t c = 0; c < w.size() && c < 16; c++) {
        k.w[c] = w[c].get<float>();
      }
      k.smooth = jf.value("smooth", true);
      s.faces.push_back(k);
    }
    sort_scene(s);
    *out = s;
    return true;
  } catch (std::exception& e) {
    lg::error("[cutscene] {}: {}", file.string(), e.what());
    return false;
  }
}

void save(const Scene& s) {
  nlohmann::json j;
  j["length"] = s.length;
  j["letterbox"] = s.letterbox;
  j["hold_jak"] = s.hold_jak;
  j["shots"] = nlohmann::json::array();
  for (const auto& sh : s.shots) {
    j["shots"].push_back({{"t", sh.t},
                          {"pos", {sh.pos[0], sh.pos[1], sh.pos[2]}},
                          {"look", {sh.look[0], sh.look[1], sh.look[2]}},
                          {"smooth", sh.smooth}});
  }
  j["lines"] = nlohmann::json::array();
  for (const auto& l : s.lines) {
    nlohmann::json jl = {{"t", l.t}, {"file", l.file}, {"speaker", l.speaker}, {"volume", l.volume},
                         {"len", l.len}, {"subs", nlohmann::json::array()}};
    for (const auto& sub : l.subs) {
      jl["subs"].push_back({{"t", sub.t}, {"text", sub.text}});
    }
    j["lines"].push_back(jl);
  }
  j["faces"] = nlohmann::json::array();
  for (const auto& k : s.faces) {
    nlohmann::json w = nlohmann::json::array();
    for (float x : k.w) {
      w.push_back(x);
    }
    j["faces"].push_back({{"t", k.t}, {"actor", k.actor}, {"w", w}, {"smooth", k.smooth}});
  }
  j["poses"] = nlohmann::json::array();
  for (const auto& k : s.poses) {
    j["poses"].push_back({{"t", k.t},
                          {"actor", k.actor},
                          {"joint", k.joint},
                          {"rot", {k.rot[0], k.rot[1], k.rot[2]}},
                          {"off", {k.off[0], k.off[1], k.off[2]}},
                          {"smooth", k.smooth}});
  }
  j["moves"] = nlohmann::json::array();
  for (const auto& k : s.moves) {
    j["moves"].push_back({{"t", k.t},
                          {"actor", k.actor},
                          {"pos", {k.pos[0], k.pos[1], k.pos[2]}},
                          {"yaw", k.yaw},
                          {"smooth", k.smooth}});
  }
  j["anims"] = nlohmann::json::array();
  for (const auto& k : s.anims) {
    j["anims"].push_back(
        {{"t", k.t}, {"actor", k.actor}, {"anim", k.anim}, {"loop", k.loop}, {"speed", k.speed}});
  }
  file_util::create_dir_if_needed(dir());
  file_util::write_text_file(dir() / (s.name + ".json"), j.dump(2));
}

float elapsed() {
  return g_from + std::chrono::duration<float>(Clock::now() - g_start).count();
}

// the camera at time t (metres)
bool camera_at(const Scene& s, float t, float* pos, float* look) {
  if (s.shots.empty()) {
    return false;
  }
  size_t i = 0;
  while (i + 1 < s.shots.size() && s.shots[i + 1].t <= t) {
    i++;
  }
  const Shot& a = s.shots[i];
  if (a.smooth && i + 1 < s.shots.size() && s.shots[i + 1].t > a.t && t > a.t) {
    const Shot& b = s.shots[i + 1];
    float u = std::clamp((t - a.t) / (b.t - a.t), 0.f, 1.f);
    u = u * u * (3.f - 2.f * u);
    for (int k = 0; k < 3; k++) {
      pos[k] = a.pos[k] + (b.pos[k] - a.pos[k]) * u;
      look[k] = a.look[k] + (b.look[k] - a.look[k]) * u;
    }
  } else {
    std::memcpy(pos, a.pos, sizeof(float) * 3);
    std::memcpy(look, a.look, sizeof(float) * 3);
  }
  return true;
}

// the subtitle showing at time t
void subtitle_at(const Scene& s, float t, std::string* speaker, std::string* text) {
  for (const auto& l : s.lines) {
    if (t < l.t || t > l.t + std::max(l.len, 0.5f)) {
      continue;
    }
    const Sub* cur = nullptr;
    for (const auto& sub : l.subs) {
      if (l.t + sub.t <= t) {
        cur = &sub;
      }
    }
    if (cur && !cur->text.empty()) {
      *speaker = l.speaker;
      *text = cur->text;
    }
  }
}

// each character's animation at time t: its last key by then
void puppets_at(const Scene& s, float t, float live = 0.f) {
  g_puppets.clear();
  std::vector<const AnimKey*> cur;
  for (const auto& k : s.anims) {
    if (k.t > t || k.actor.empty()) {
      continue;
    }
    bool found = false;
    for (auto& c : cur) {
      if (c->actor == k.actor) {
        c = &k;
        found = true;
      }
    }
    if (!found) {
      cur.push_back(&k);
    }
  }
  for (const auto* k : cur) {
    if (!k->anim.empty()) {
      g_puppets.push_back({k->actor, k->anim, (t - k->t + (k->loop ? live : 0.f)) * k->speed, k->loop});
    }
  }
}

// each character's place at time t: its last key by then (on towards its next, if it slides)
void places_at(const Scene& s, float t) {
  g_places.clear();
  std::vector<std::string> done;
  for (const auto& k : s.moves) {
    if (k.actor.empty() || std::find(done.begin(), done.end(), k.actor) != done.end()) {
      continue;
    }
    done.push_back(k.actor);
    const MoveKey* a = nullptr;
    const MoveKey* b = nullptr;
    for (const auto& m : s.moves) {
      if (m.actor != k.actor) {
        continue;
      }
      if (m.t <= t) {
        a = &m;
      } else if (!b) {
        b = &m;
      }
    }
    if (!a) {
      continue;  // (before its first key: where it is)
    }
    Place p;
    p.actor = a->actor;
    float u = 0.f;
    if (a->smooth && b && b->t > a->t) {
      u = std::clamp((t - a->t) / (b->t - a->t), 0.f, 1.f);
    }
    for (int c = 0; c < 3; c++) {
      p.pos[c] = (a->pos[c] + (b && u > 0.f ? (b->pos[c] - a->pos[c]) * u : 0.f)) * kMeter;
    }
    float yaw = a->yaw;
    if (u > 0.f) {
      float d = std::fmod(b->yaw - a->yaw + 540.f, 360.f) - 180.f;  // the short way round
      yaw += d * u;
    }
    p.yaw = yaw;
    g_places.push_back(p);
  }
}

// each body part's turn and move at time t: its last key by then (on towards its next, if it slides)
void poses_at(const Scene& s, float t) {
  g_poses.clear();
  std::vector<std::string> done;
  for (const auto& k : s.poses) {
    const std::string id = k.actor + "\n" + k.joint;
    if (k.actor.empty() || k.joint.empty() || std::find(done.begin(), done.end(), id) != done.end()) {
      continue;
    }
    done.push_back(id);
    const PoseKey* a = nullptr;
    const PoseKey* b = nullptr;
    for (const auto& m : s.poses) {
      if (m.actor != k.actor || m.joint != k.joint) {
        continue;
      }
      if (m.t <= t) {
        a = &m;
      } else if (!b) {
        b = &m;
      }
    }
    if (!a) {
      continue;
    }
    float u = 0.f;
    if (a->smooth && b && b->t > a->t) {
      u = std::clamp((t - a->t) / (b->t - a->t), 0.f, 1.f);
      u = u * u * (3.f - 2.f * u);
    }
    Pose p;
    p.actor = a->actor;
    p.joint = a->joint;
    for (int c = 0; c < 3; c++) {
      p.rot[c] = a->rot[c] + (u > 0.f ? (b->rot[c] - a->rot[c]) * u : 0.f);
      p.off[c] = (a->off[c] + (u > 0.f ? (b->off[c] - a->off[c]) * u : 0.f)) * kMeter;
    }
    g_poses.push_back(p);
  }
}

// each character's face at time t: its last key by then (on towards its next, if it slides)
void faces_at(const Scene& s, float t) {
  g_faces.clear();
  std::vector<std::string> done;
  for (const auto& k : s.faces) {
    if (k.actor.empty() || std::find(done.begin(), done.end(), k.actor) != done.end()) {
      continue;
    }
    done.push_back(k.actor);
    const FaceKey* a = nullptr;
    const FaceKey* b = nullptr;
    for (const auto& m : s.faces) {
      if (m.actor != k.actor) {
        continue;
      }
      if (m.t <= t) {
        a = &m;
      } else if (!b) {
        b = &m;
      }
    }
    if (!a) {
      continue;
    }
    float u = 0.f;
    if (a->smooth && b && b->t > a->t) {
      u = std::clamp((t - a->t) / (b->t - a->t), 0.f, 1.f);
      u = u * u * (3.f - 2.f * u);
    }
    Face f;
    f.actor = a->actor;
    for (int c = 0; c < 16; c++) {
      f.w[c] = a->w[c] + (u > 0.f ? (b->w[c] - a->w[c]) * u : 0.f);
    }
    g_faces.push_back(f);
  }
}

void stop_voices() {
  for (size_t i = 0; i < g_scene.lines.size() && i < g_started.size(); i++) {
    if (g_started[i]) {
      enhanced_stop_mp3(g_scene.lines[i].file);
    }
  }
  g_started.assign(g_scene.lines.size(), false);
}

void start_clock(float from) {
  g_from = from;
  g_start = Clock::now();
  g_started.assign(g_scene.lines.size(), false);
}

}  // namespace

bool play(const std::string& name) {
  std::lock_guard<std::mutex> lock(g_mutex);
  Scene s;
  if (g_scene_loaded && g_dirty && g_scene.name == name) {
    // (being edited, not saved yet: as it is in the editor)
    s = g_scene;
  } else if (!load(name, &s)) {
    return false;
  }
  if (g_mode != kNone) {
    stop_voices();
  }
  g_scene = s;
  g_scene_loaded = true;
  g_dirty = false;
  g_mode = kPlay;
  history_reset();
  start_clock(0.f);
  return true;
}

void stop() {
  std::lock_guard<std::mutex> lock(g_mutex);
  stop_voices();
  g_mode = g_mode == kPreview ? kEdit : kNone;
}

Frame poll() {
  std::lock_guard<std::mutex> lock(g_mutex);
  Frame f;
  f.mode = g_mode;
  g_edit_now = g_mode == kEdit;
  history_commit(false);
  if (g_actors_ready) {
    g_actors.swap(g_actors_next);
    g_actor_at.swap(g_actor_at_next);
    g_actor_joints.swap(g_actor_joints_next);
    g_actor_face.swap(g_actor_face_next);
    g_actors_ready = false;
  }
  if (g_mode == kNone) {
    g_puppets.clear();
    g_places.clear();
    g_poses.clear();
    g_faces.clear();
    return f;
  }
  f.letterbox = g_scene.letterbox;
  f.hold_jak = g_scene.hold_jak;
  if (g_mode == kEdit) {
    // the free camera's own; put somewhere when asked
    if (g_jump) {
      g_jump = false;
      f.jump = true;
      for (int k = 0; k < 3; k++) {
        f.pos[k] = g_jump_pos[k] * kMeter;
        f.look[k] = g_jump_look[k] * kMeter;
      }
    }
    subtitle_at(g_scene, g_cursor, &f.speaker, &f.text);
    puppets_at(g_scene, g_cursor, std::chrono::duration<float>(Clock::now() - g_edit_clock).count());
    places_at(g_scene, g_cursor);
    poses_at(g_scene, g_cursor);
    faces_at(g_scene, g_cursor);
    // the characters near the camera, every so often while the animations are being set
    if (g_scan_wanted && Clock::now() - g_last_scan > std::chrono::milliseconds(1500)) {
      g_last_scan = Clock::now();
      f.scan = true;
    }
    g_scan_wanted = false;
    return f;
  }
  const float t = elapsed();
  if (g_mode == kPreview) {
    g_cursor = std::min(t, g_scene.length);
  }
  if (t >= g_scene.length) {
    stop_voices();
    if (g_mode == kPreview) {
      g_mode = kEdit;
      f.mode = kEdit;
      puppets_at(g_scene, g_cursor);
      places_at(g_scene, g_cursor);
      poses_at(g_scene, g_cursor);
      faces_at(g_scene, g_cursor);
    } else {
      g_mode = kNone;
      f.mode = kNone;
      g_puppets.clear();
      g_places.clear();
      g_poses.clear();
      g_faces.clear();
    }
    return f;
  }
  // the voices due, each from where it is now (a preview may start halfway through one)
  for (size_t i = 0; i < g_scene.lines.size(); i++) {
    const auto& l = g_scene.lines[i];
    if (!g_started[i] && t >= l.t && t < l.t + std::max(l.len, 0.1f)) {
      g_started[i] = true;
      enhanced_play_mp3(l.file, l.volume, t - l.t);
    }
  }
  float pos[3], look[3];
  if (camera_at(g_scene, t, pos, look)) {
    f.has_cam = true;
    for (int k = 0; k < 3; k++) {
      f.pos[k] = pos[k] * kMeter;
      f.look[k] = look[k] * kMeter;
    }
  }
  subtitle_at(g_scene, t, &f.speaker, &f.text);
  puppets_at(g_scene, t);
  places_at(g_scene, t);
  poses_at(g_scene, t);
  faces_at(g_scene, t);
  return f;
}

bool face(int i, Face* out) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (i < 0 || i >= (int)g_faces.size()) {
    return false;
  }
  *out = g_faces[i];
  return true;
}

void actor_face(const std::string& actor, int count) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_actor_face_next[actor] = count;
}

bool pose(int i, Pose* out) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (i < 0 || i >= (int)g_poses.size()) {
    return false;
  }
  *out = g_poses[i];
  return true;
}

void actor_joint(const std::string& actor, const std::string& joint) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_actor_joints_next[actor].push_back(joint);
}

bool pose_selected(std::string* actor, std::string* joint) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_mode != kEdit || g_sel_actor.empty() || g_sel_joint.empty()) {
    return false;
  }
  *actor = g_sel_actor;
  *joint = g_sel_joint;
  return true;
}

void undo_redo(int dir) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_scene_loaded) {
    return;
  }
  history_commit(true);
  auto& from = dir < 0 ? g_undo : g_redo;
  auto& to = dir < 0 ? g_redo : g_undo;
  if (from.empty()) {
    return;
  }
  to.push_back(g_scene);
  const std::string name = g_scene.name;
  g_scene = from.back();
  g_scene.name = name;
  from.pop_back();
  g_dirty = true;
  g_rev++;
  g_committed = g_scene;
  g_committed_rev = g_rev;
  g_cursor = std::min(g_cursor, g_scene.length);
}

bool editing() {
  return g_edit_now.load();
}

void gyro(const float* rad_per_s, float dt) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_have_gyro = true;
  if (g_mode != kEdit || dt <= 0.f || dt > 0.1f) {
    return;
  }
  if (g_start_held) {
    g_cur_x = 0.f;
    g_cur_y = 0.f;
    return;
  }
  // (about 20 degrees of turn from the middle to the side)
  constexpr float kGain = 1.f / 0.35f;
  g_cur_x = std::clamp(g_cur_x - rad_per_s[1] * dt * kGain, -1.f, 1.f);
  g_cur_y = std::clamp(g_cur_y + rad_per_s[0] * dt * kGain, -1.f, 1.f);
}

void start_button(bool down) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_start_held = down;
  if (down) {
    g_cur_x = 0.f;
    g_cur_y = 0.f;
  }
}

bool cursor(float* x, float* y) {
  std::lock_guard<std::mutex> lock(g_mutex);
  *x = g_cur_x;
  *y = g_cur_y;
  return g_have_gyro;
}

void pose_pick(const std::string& actor, const std::string& joint) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_sel_actor = actor;
  g_sel_joint = joint;
  g_pick_actor = actor;
}

void pose_grab(const std::string& actor, const std::string& joint, const float* rot, const float* off) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_mode != kEdit || actor.empty() || joint.empty()) {
    return;
  }
  g_held = actor + ": " + joint;
  g_held_at = Clock::now();
  PoseKey* key = nullptr;
  for (auto& k : g_scene.poses) {
    if (k.actor == actor && k.joint == joint && std::fabs(k.t - g_cursor) < 0.02f) {
      key = &k;
    }
  }
  if (!key) {
    PoseKey k;
    k.t = g_cursor;
    k.actor = actor;
    k.joint = joint;
    g_scene.poses.push_back(k);
    sort_scene(g_scene);
    for (auto& m : g_scene.poses) {
      if (m.actor == actor && m.joint == joint && std::fabs(m.t - g_cursor) < 0.02f) {
        key = &m;
      }
    }
  }
  if (key) {
    for (int c = 0; c < 3; c++) {
      key->rot[c] = rot[c];
      key->off[c] = off[c] / kMeter;
    }
    changed();
  }
}

bool place(int i, Place* out) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (i < 0 || i >= (int)g_places.size()) {
    return false;
  }
  *out = g_places[i];
  return true;
}

void grab(const std::string& actor, const float* pos, float yaw) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_mode != kEdit || actor.empty()) {
    return;
  }
  g_held = actor;
  g_held_at = Clock::now();
  MoveKey* key = nullptr;
  for (auto& k : g_scene.moves) {
    if (k.actor == actor && std::fabs(k.t - g_cursor) < 0.02f) {
      key = &k;
    }
  }
  if (!key) {
    MoveKey k;
    k.t = g_cursor;
    k.actor = actor;
    g_scene.moves.push_back(k);
    sort_scene(g_scene);
    for (auto& m : g_scene.moves) {
      if (m.actor == actor && std::fabs(m.t - g_cursor) < 0.02f) {
        key = &m;
      }
    }
  }
  if (key) {
    for (int c = 0; c < 3; c++) {
      key->pos[c] = pos[c] / kMeter;
    }
    key->yaw = yaw;
    changed();
  }
}

void actor_at(const std::string& actor, const float* pos, float yaw) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ActorAt a;
  for (int k = 0; k < 3; k++) {
    a.pos[k] = pos[k] / kMeter;
  }
  a.yaw = yaw;
  g_actor_at_next[actor] = a;
}

void puppet_status(int i, int status) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (i < 0 || i >= 16) {
    return;
  }
  if ((int)g_puppet_status.size() <= i) {
    g_puppet_status.resize(i + 1, -1);
  }
  g_puppet_status[i] = status;
}

bool puppet(int i, Puppet* out) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (i < 0 || i >= (int)g_puppets.size()) {
    return false;
  }
  *out = g_puppets[i];
  return true;
}

void actors_begin() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_actors_next.clear();
  g_actor_at_next.clear();
  g_actor_joints_next.clear();
  g_actor_face_next.clear();
  g_actors_ready = true;
}

void actor_anim(const std::string& actor, const std::string& anim) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_actors_next.empty() || g_actors_next.back().first != actor) {
    g_actors_next.push_back({actor, {}});
  }
  if (!anim.empty()) {
    g_actors_next.back().second.push_back(anim);
  }
}

void report_camera(const float* pos, const float* look, const float* aim) {
  std::lock_guard<std::mutex> lock(g_mutex);
  for (int k = 0; k < 3; k++) {
    g_cam_pos[k] = pos[k] / kMeter;
    g_cam_look[k] = look[k] / kMeter;
    g_cam_aim[k] = aim[k] / kMeter;
  }
  g_cam_known = true;
}

// ---------------------------------------------------------------------------------------------------------
// the editor

namespace {

char g_new_name[64] = "";
int g_pick = -1;

void begin_edit() {
  // the free camera starts where the game's camera is (the renderer's last frame)
  g_mode = kEdit;
  const auto& cam = g_enhanced_camera;
  if (cam.valid) {
    for (int k = 0; k < 3; k++) {
      g_jump_pos[k] = cam.trans[k] / kMeter;
      g_jump_look[k] = (cam.trans[k] + cam.rot[k][2] * 10.f * kMeter) / kMeter;
    }
    g_jump = true;
  }
}

// the timeline: the scene's length across, shots (blue ticks) and lines (orange bars), the cursor
void timeline() {
  auto* dl = ImGui::GetWindowDrawList();
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  const float w = std::max(200.f, ImGui::GetContentRegionAvail().x);
  const float h = 54.f;
  ImGui::InvisibleButton("##timeline", ImVec2(w, h));
  const float len = std::max(g_scene.length, 0.1f);
  auto x_of = [&](float t) { return p0.x + w * std::clamp(t / len, 0.f, 1.f); };
  if (ImGui::IsItemActive() && g_mode != kPreview) {
    g_cursor = std::clamp((ImGui::GetIO().MousePos.x - p0.x) / w * len, 0.f, len);
  }
  dl->AddRectFilled(p0, ImVec2(p0.x + w, p0.y + h), IM_COL32(30, 30, 36, 255), 4.f);
  for (int s = 0; s <= (int)len; s++) {
    const float x = x_of((float)s);
    dl->AddLine(ImVec2(x, p0.y + h - (s % 5 == 0 ? 10.f : 5.f)), ImVec2(x, p0.y + h), IM_COL32(120, 120, 130, 255));
  }
  for (const auto& l : g_scene.lines) {
    dl->AddRectFilled(ImVec2(x_of(l.t), p0.y + 26.f), ImVec2(x_of(l.t + l.len), p0.y + 40.f),
                      IM_COL32(220, 140, 40, 200), 3.f);
    for (const auto& sub : l.subs) {
      const float x = x_of(l.t + sub.t);
      dl->AddLine(ImVec2(x, p0.y + 24.f), ImVec2(x, p0.y + 42.f), IM_COL32(255, 230, 160, 255), 1.5f);
    }
  }
  for (const auto& k : g_scene.poses) {
    const float x = x_of(k.t);
    dl->AddCircleFilled(ImVec2(x, p0.y + 52.f), 2.5f, IM_COL32(200, 120, 255, 255));
  }
  for (const auto& k : g_scene.moves) {
    const float x = x_of(k.t);
    dl->AddRectFilled(ImVec2(x - 3.f, p0.y + 43.f), ImVec2(x + 3.f, p0.y + 49.f),
                      k.smooth ? IM_COL32(255, 220, 90, 255) : IM_COL32(220, 170, 40, 255));
  }
  for (const auto& k : g_scene.anims) {
    const float x = x_of(k.t);
    dl->AddTriangleFilled(ImVec2(x, p0.y + 42.f), ImVec2(x - 4.f, p0.y + 50.f), ImVec2(x + 4.f, p0.y + 50.f),
                          k.anim.empty() ? IM_COL32(140, 140, 140, 255) : IM_COL32(90, 220, 110, 255));
  }
  for (const auto& sh : g_scene.shots) {
    const float x = x_of(sh.t);
    dl->AddRectFilled(ImVec2(x - 2.f, p0.y + 4.f), ImVec2(x + 2.f, p0.y + 22.f),
                      sh.smooth ? IM_COL32(90, 200, 255, 255) : IM_COL32(60, 120, 255, 255));
  }
  const float cx = x_of(g_cursor);
  dl->AddLine(ImVec2(cx, p0.y), ImVec2(cx, p0.y + h), IM_COL32(255, 60, 60, 255), 2.f);
}

// a character's animations, from the game's list (none if it isn't near)
const std::vector<std::string>* anims_of(const std::string& actor) {
  for (const auto& a : g_actors) {
    if (a.first == actor) {
      return &a.second;
    }
  }
  return nullptr;
}

// an animation picked from a character's, filtered by what's typed
bool anim_combo(const char* id, const std::string& actor, std::string* anim) {
  static char filter[64] = "";
  bool changed = false;
  ImGui::SetNextItemWidth(230);
  if (ImGui::BeginCombo(id, anim->empty() ? "(its own animation)" : anim->c_str(), ImGuiComboFlags_HeightLarge)) {
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##filter", "filter", filter, sizeof(filter));
    if (ImGui::Selectable("(its own animation)", anim->empty())) {
      anim->clear();
      changed = true;
    }
    if (const auto* list = anims_of(actor)) {
      for (const auto& a : *list) {
        if (filter[0] && a.find(filter) == std::string::npos) {
          continue;
        }
        if (ImGui::Selectable(a.c_str(), a == *anim)) {
          *anim = a;
          changed = true;
        }
      }
    } else {
      ImGui::TextDisabled("(not near the camera: fly closer in Edit)");
    }
    ImGui::EndCombo();
  }
  return changed;
}

void animations_menu() {
  static int s_actor = 0;
  if (g_mode == kEdit) {
    g_scan_wanted = true;
  }
  if (g_actors.empty()) {
    ImGui::TextDisabled(g_mode == kEdit ? "(no characters near the camera: fly closer)"
                                        : "(Edit first: the characters near the camera are listed)");
  } else {
    s_actor = std::clamp(s_actor, 0, (int)g_actors.size() - 1);
    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo("Character", g_actors[s_actor].first.c_str())) {
      for (int i = 0; i < (int)g_actors.size(); i++) {
        char label[96];
        std::snprintf(label, sizeof(label), "%s (%d)", g_actors[i].first.c_str(),
                      (int)g_actors[i].second.size());
        if (ImGui::Selectable(label, i == s_actor)) {
          s_actor = i;
        }
      }
      ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Add an animation at the cursor")) {
      AnimKey k;
      k.t = g_cursor;
      k.actor = g_actors[s_actor].first;
      if (!g_actors[s_actor].second.empty()) {
        k.anim = g_actors[s_actor].second[0];
      }
      g_scene.anims.push_back(k);
      sort_scene(g_scene);
      changed();
    }
  }
  ImGui::TextDisabled("(each key: from its time on, the character plays it; \"its own\" gives it back its own)");
  int gone = -1;
  for (size_t i = 0; i < g_scene.anims.size(); i++) {
    auto& k = g_scene.anims[i];
    ImGui::PushID(5000 + (int)i);
    ImGui::SetNextItemWidth(80);
    if (ImGui::DragFloat("##at", &k.t, 0.02f, 0.f, g_scene.length, "%.2f s")) {
      changed();
    }
    ImGui::SameLine();
    ImGui::Text("%s", k.actor.c_str());
    ImGui::SameLine();
    if (anim_combo("##anim", k.actor, &k.anim)) {
      changed();
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("loop", &k.loop)) {
      changed();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    if (ImGui::DragFloat("##speed", &k.speed, 0.01f, 0.1f, 4.f, "x%.2f")) {
      changed();
    }
    ImGui::SameLine();
    if (ImGui::Button("Go")) {
      g_cursor = k.t;
    }
    // (what the game made of it, if it's the one playing now)
    for (size_t p = 0; p < g_puppets.size(); p++) {
      if (g_puppets[p].actor == k.actor && g_puppets[p].anim == k.anim && !k.anim.empty() && k.t <= g_cursor + 0.001f) {
        const int st = p < g_puppet_status.size() ? g_puppet_status[p] : -1;
        ImGui::SameLine();
        if (st == 2) {
          ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.f), "playing");
        } else if (st == 1) {
          ImGui::TextColored(ImVec4(1.f, 0.5f, 0.3f, 1.f), "animation not found");
        } else if (st == 0) {
          ImGui::TextColored(ImVec4(1.f, 0.5f, 0.3f, 1.f), "character not found");
        }
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete")) {
      gone = (int)i;
    }
    ImGui::PopID();
  }
  if (gone >= 0) {
    g_scene.anims.erase(g_scene.anims.begin() + gone);
    changed();
  }
}

// where a character is now (the game's list), metres and degrees
bool actor_now(const std::string& actor, float* pos, float* yaw) {
  auto it = g_actor_at.find(actor);
  if (it == g_actor_at.end()) {
    return false;
  }
  for (int k = 0; k < 3; k++) {
    pos[k] = it->second.pos[k];
  }
  *yaw = it->second.yaw;
  return true;
}

// which way from a to b, degrees (the game's: 0 looks down +z)
float yaw_towards(const float* a, const float* b) {
  return std::atan2(b[0] - a[0], b[2] - a[2]) * 57.29578f;
}

void moves_menu() {
  static int s_actor = 0;
  if (g_mode == kEdit) {
    g_scan_wanted = true;
  }
  if (g_actors.empty()) {
    ImGui::TextDisabled(g_mode == kEdit ? "(no characters near the camera: fly closer)"
                                        : "(Edit first: the characters near the camera are listed)");
  } else {
    s_actor = std::clamp(s_actor, 0, (int)g_actors.size() - 1);
    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo("Character##move", g_actors[s_actor].first.c_str())) {
      for (int i = 0; i < (int)g_actors.size(); i++) {
        if (ImGui::Selectable(g_actors[i].first.c_str(), i == s_actor)) {
          s_actor = i;
        }
      }
      ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Add a place at the cursor (where it is now)")) {
      MoveKey k;
      k.t = g_cursor;
      k.actor = g_actors[s_actor].first;
      if (!actor_now(k.actor, k.pos, &k.yaw) && g_cam_known) {
        std::memcpy(k.pos, g_cam_aim, sizeof(k.pos));
      }
      g_scene.moves.push_back(k);
      sort_scene(g_scene);
      changed();
    }
  }
  ImGui::TextDisabled("(each key: the character is there at its time; \"slide\" takes it on to its next key)");
  ImGui::TextDisabled(
      "Pad: R2 held takes what the camera aims at; left stick moves it (L1: up, down), right stick turns it");
  if (!g_held.empty() && Clock::now() - g_held_at < std::chrono::milliseconds(300)) {
    ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "Holding %s (its place at %.2f s)", g_held.c_str(), g_cursor);
  }
  int gone = -1;
  for (size_t i = 0; i < g_scene.moves.size(); i++) {
    auto& k = g_scene.moves[i];
    ImGui::PushID(8000 + (int)i);
    ImGui::Separator();
    ImGui::SetNextItemWidth(80);
    if (ImGui::DragFloat("##mt", &k.t, 0.02f, 0.f, g_scene.length, "%.2f s")) {
      changed();
    }
    ImGui::SameLine();
    ImGui::Text("%s", k.actor.c_str());
    ImGui::SameLine();
    if (ImGui::Checkbox("slide", &k.smooth)) {
      changed();
    }
    ImGui::SameLine();
    if (ImGui::Button("Go")) {
      g_cursor = k.t;
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete")) {
      gone = (int)i;
    }
    ImGui::Indent(20.f);
    ImGui::SetNextItemWidth(240);
    if (ImGui::DragFloat3("x y z (m)", k.pos, 0.02f, 0.f, 0.f, "%.2f")) {
      changed();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::DragFloat("facing", &k.yaw, 0.5f, -360.f, 360.f, "%.0f deg")) {
      changed();
    }
    if (ImGui::Button("Where it is now")) {
      float y;
      if (actor_now(k.actor, k.pos, &y)) {
        k.yaw = y;
        changed();
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Where the camera aims") && g_cam_known) {
      std::memcpy(k.pos, g_cam_aim, sizeof(k.pos));
      changed();
    }
    ImGui::SameLine();
    if (ImGui::Button("Face the camera") && g_cam_known) {
      k.yaw = yaw_towards(k.pos, g_cam_pos);
      changed();
    }
    ImGui::SameLine();
    if (ImGui::Button("Face where the camera aims") && g_cam_known) {
      k.yaw = yaw_towards(k.pos, g_cam_aim);
      changed();
    }
    ImGui::Unindent(20.f);
    ImGui::PopID();
  }
  if (gone >= 0) {
    g_scene.moves.erase(g_scene.moves.begin() + gone);
    changed();
  }
}

void poses_menu() {
  static int s_actor = 0;
  static char filter[64] = "";
  if (g_mode == kEdit) {
    g_scan_wanted = true;
  }
  if (g_actors.empty()) {
    ImGui::TextDisabled(g_mode == kEdit ? "(no characters near the camera: fly closer)"
                                        : "(Edit first: the characters near the camera are listed)");
  } else {
    // (picked with the pad: its character here)
    if (!g_pick_actor.empty()) {
      for (int i = 0; i < (int)g_actors.size(); i++) {
        if (g_actors[i].first == g_pick_actor) {
          s_actor = i;
        }
      }
      g_pick_actor.clear();
    }
    s_actor = std::clamp(s_actor, 0, (int)g_actors.size() - 1);
    const std::string& actor = g_actors[s_actor].first;
    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo("Character##pose", actor.c_str())) {
      for (int i = 0; i < (int)g_actors.size(); i++) {
        if (ImGui::Selectable(g_actors[i].first.c_str(), i == s_actor)) {
          s_actor = i;
          g_sel_joint.clear();
        }
      }
      ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo("Body part", g_sel_joint.empty() ? "(none: R2 takes the whole character)"
                                                             : g_sel_joint.c_str(),
                          ImGuiComboFlags_HeightLarge)) {
      ImGui::SetNextItemWidth(-1);
      ImGui::InputTextWithHint("##jfilter", "filter", filter, sizeof(filter));
      if (ImGui::Selectable("(none: R2 takes the whole character)", g_sel_joint.empty())) {
        g_sel_joint.clear();
      }
      auto it = g_actor_joints.find(actor);
      if (it != g_actor_joints.end()) {
        for (const auto& j : it->second) {
          if (filter[0] && j.find(filter) == std::string::npos) {
            continue;
          }
          if (ImGui::Selectable(j.c_str(), j == g_sel_joint)) {
            g_sel_joint = j;
          }
        }
      }
      ImGui::EndCombo();
    }
    g_sel_actor = g_sel_joint.empty() ? std::string() : actor;
  }
  ImGui::TextDisabled("Pad: L1 + X picks the body part under the cross (turn the controller to move the cross,");
  ImGui::TextDisabled("Start held puts it back in the middle); L1 + circle: none picked any more.");
  ImGui::TextDisabled(g_have_gyro ? "(gyroscope: yes)" : "(no gyroscope seen yet: the cross stays in the middle)");
  ImGui::TextDisabled("Pad, a body part picked: R2 held moves it (left stick; L1: up, down), L2 held turns it");
  ImGui::TextDisabled("in place (left stick, right stick); without one, R2 takes the whole character.");
  int gone = -1;
  for (size_t i = 0; i < g_scene.poses.size(); i++) {
    auto& k = g_scene.poses[i];
    ImGui::PushID(9000 + (int)i);
    ImGui::Separator();
    ImGui::SetNextItemWidth(80);
    if (ImGui::DragFloat("##pt", &k.t, 0.02f, 0.f, g_scene.length, "%.2f s")) {
      changed();
    }
    ImGui::SameLine();
    ImGui::Text("%s: %s", k.actor.c_str(), k.joint.c_str());
    ImGui::SameLine();
    if (ImGui::Checkbox("slide", &k.smooth)) {
      changed();
    }
    ImGui::SameLine();
    if (ImGui::Button("Go")) {
      g_cursor = k.t;
    }
    ImGui::SameLine();
    if (ImGui::Button("Pick")) {
      g_sel_joint = k.joint;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
      for (int c = 0; c < 3; c++) {
        k.rot[c] = 0.f;
        k.off[c] = 0.f;
      }
      changed();
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete")) {
      gone = (int)i;
    }
    ImGui::Indent(20.f);
    ImGui::SetNextItemWidth(220);
    if (ImGui::DragFloat3("turn (deg)", k.rot, 0.5f, -360.f, 360.f, "%.0f")) {
      changed();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220);
    if (ImGui::DragFloat3("move (m)", k.off, 0.005f, -5.f, 5.f, "%.3f")) {
      changed();
    }
    ImGui::Unindent(20.f);
    ImGui::PopID();
  }
  if (gone >= 0) {
    g_scene.poses.erase(g_scene.poses.begin() + gone);
    changed();
  }
}

void faces_menu() {
  static int s_actor = 0;
  if (g_mode == kEdit) {
    g_scan_wanted = true;
  }
  std::vector<std::string> with;
  for (const auto& a : g_actors) {
    if (g_actor_face.count(a.first)) {
      with.push_back(a.first);
    }
  }
  if (with.empty()) {
    ImGui::TextDisabled(g_mode == kEdit ? "(no character with face shapes near the camera)"
                                        : "(Edit first: the characters near the camera are listed)");
  } else {
    s_actor = std::clamp(s_actor, 0, (int)with.size() - 1);
    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo("Character##face", with[s_actor].c_str())) {
      for (int i = 0; i < (int)with.size(); i++) {
        if (ImGui::Selectable(with[i].c_str(), i == s_actor)) {
          s_actor = i;
        }
      }
      ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Add a face at the cursor")) {
      FaceKey k;
      k.t = g_cursor;
      k.actor = with[s_actor];
      // (from the face it has then, if any)
      for (const auto& f : g_faces) {
        if (f.actor == k.actor) {
          std::memcpy(k.w, f.w, sizeof(k.w));
        }
      }
      g_scene.faces.push_back(k);
      sort_scene(g_scene);
      changed();
    }
  }
  ImGui::TextDisabled("(each shape: 0 as it is, 1 all the way; one of them opens the mouth - try them)");
  int gone = -1;
  for (size_t i = 0; i < g_scene.faces.size(); i++) {
    auto& k = g_scene.faces[i];
    ImGui::PushID(11000 + (int)i);
    ImGui::Separator();
    ImGui::SetNextItemWidth(80);
    if (ImGui::DragFloat("##ft", &k.t, 0.02f, 0.f, g_scene.length, "%.2f s")) {
      changed();
    }
    ImGui::SameLine();
    ImGui::Text("%s", k.actor.c_str());
    ImGui::SameLine();
    if (ImGui::Checkbox("slide", &k.smooth)) {
      changed();
    }
    ImGui::SameLine();
    if (ImGui::Button("Go")) {
      g_cursor = k.t;
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete")) {
      gone = (int)i;
    }
    const int n = g_actor_face.count(k.actor) ? std::min(16, g_actor_face[k.actor]) : 16;
    ImGui::Indent(20.f);
    for (int s = 0; s < n; s++) {
      char label[32];
      std::snprintf(label, sizeof(label), "shape %d", s + 1);
      ImGui::SetNextItemWidth(200);
      if (ImGui::SliderFloat(label, &k.w[s], -1.f, 1.f, "%.2f")) {
        changed();
      }
      if (s % 2 == 0 && s + 1 < n) {
        ImGui::SameLine();
      }
    }
    ImGui::Unindent(20.f);
    ImGui::PopID();
  }
  if (gone >= 0) {
    g_scene.faces.erase(g_scene.faces.begin() + gone);
    changed();
  }
}

}  // namespace

void draw_menu() {
  if (!ImGui::CollapsingHeader("Cutscenes")) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  ImGui::PushID("cutscenes");
  // which scene
  const auto names = scene_names();
  const char* cur = g_scene_loaded ? g_scene.name.c_str() : "(none)";
  if (ImGui::BeginCombo("Scene", cur)) {
    for (const auto& n : names) {
      if (ImGui::Selectable(n.c_str(), g_scene_loaded && n == g_scene.name)) {
        Scene s;
        if (load(n, &s)) {
          stop_voices();
          g_scene = s;
          g_scene_loaded = true;
          g_dirty = false;
          g_cursor = 0.f;
          history_reset();
        }
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SetNextItemWidth(160);
  ImGui::InputText("##newname", g_new_name, sizeof(g_new_name));
  ImGui::SameLine();
  if (ImGui::Button("New scene") && g_new_name[0]) {
    stop_voices();
    g_scene = Scene();
    g_scene.name = g_new_name;
    g_scene_loaded = true;
    changed();
    g_cursor = 0.f;
    history_reset();
  }
  if (!g_scene_loaded) {
    ImGui::TextDisabled("Pick a scene, or name a new one.");
    ImGui::PopID();
    return;
  }
  ImGui::SameLine();
  {
    // (the history: the D-pad's left and right do the same, in Edit)
    history_commit(false);
    ImGui::BeginDisabled(g_undo.empty() && g_rev == g_committed_rev);
    if (ImGui::Button("Undo")) {
      g_mutex.unlock();
      undo_redo(-1);
      g_mutex.lock();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(g_redo.empty());
    if (ImGui::Button("Redo")) {
      g_mutex.unlock();
      undo_redo(1);
      g_mutex.lock();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("(%d / %d)", (int)g_undo.size(), (int)g_redo.size());
    ImGui::SameLine();
  }
  if (ImGui::Button(g_dirty ? "Save *" : "Save")) {
    sort_scene(g_scene);
    save(g_scene);
    g_dirty = false;
  }

  // edit / preview / stop
  if (g_mode == kNone || g_mode == kPlay) {
    if (ImGui::Button("Edit (fly the camera with the pad)", ImVec2(280, 0))) {
      stop_voices();
      begin_edit();
    }
  } else {
    if (g_mode == kEdit && ImGui::Button("Preview from the cursor", ImVec2(200, 0))) {
      g_mode = kPreview;
      start_clock(g_cursor);
    }
    if (g_mode == kPreview && ImGui::Button("Stop preview", ImVec2(200, 0))) {
      stop_voices();
      g_mode = kEdit;
    }
    ImGui::SameLine();
    if (ImGui::Button("Leave the editor")) {
      stop_voices();
      g_mode = kNone;
    }
    ImGui::TextDisabled(
        "Pad: left stick moves, right stick looks; L1 held: only across the view; R1 held: holds on to the\n"
        "point aimed at (left stick: towards it, right stick: round it); Square slow, Circle fast");
  }

  ImGui::SetNextItemWidth(120);
  if (ImGui::InputFloat("Length (s)", &g_scene.length, 0.5f, 5.f, "%.1f")) {
    g_scene.length = std::max(g_scene.length, 0.5f);
    changed();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Black bars", &g_scene.letterbox)) {
    changed();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Hold Jak", &g_scene.hold_jak)) {
    changed();
  }
  timeline();
  ImGui::SetNextItemWidth(-1);
  if (g_mode != kPreview) {
    ImGui::SliderFloat("##cursor", &g_cursor, 0.f, g_scene.length, "%.2f s");
  } else {
    ImGui::Text("%.2f s", g_cursor);
  }

  // shots (a copied one stays on the clipboard across scenes, to be pasted at any point of any of them)
  static Shot s_clip;
  static bool s_clip_full = false;
  if (ImGui::TreeNodeEx("Shots", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::Button("Add a shot at the cursor (what the camera sees)") && g_cam_known) {
      Shot sh;
      sh.t = g_cursor;
      std::memcpy(sh.pos, g_cam_pos, sizeof(sh.pos));
      std::memcpy(sh.look, g_cam_look, sizeof(sh.look));
      g_scene.shots.push_back(sh);
      sort_scene(g_scene);
      changed();
    }
    if (s_clip_full) {
      if (ImGui::Button("Paste the copied shot at the cursor")) {
        Shot sh = s_clip;
        sh.t = g_cursor;
        g_scene.shots.push_back(sh);
        sort_scene(g_scene);
        changed();
      }
    } else {
      ImGui::TextDisabled("(Copy a shot to paste it at the cursor, in this scene or another)");
    }
    if (!g_cam_known) {
      ImGui::TextDisabled("(Edit first: the shots are taken from the free camera)");
    }
    int gone = -1;
    for (size_t i = 0; i < g_scene.shots.size(); i++) {
      auto& sh = g_scene.shots[i];
      ImGui::PushID((int)i);
      ImGui::SetNextItemWidth(80);
      if (ImGui::DragFloat("##t", &sh.t, 0.02f, 0.f, g_scene.length, "%.2f s")) {
        changed();
      }
      ImGui::SameLine();
      if (ImGui::Checkbox("move to next", &sh.smooth)) {
        changed();
      }
      ImGui::SameLine();
      if (ImGui::Button("Go")) {
        g_cursor = sh.t;
        std::memcpy(g_jump_pos, sh.pos, sizeof(sh.pos));
        std::memcpy(g_jump_look, sh.look, sizeof(sh.look));
        g_jump = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Set from camera") && g_cam_known) {
        std::memcpy(sh.pos, g_cam_pos, sizeof(sh.pos));
        std::memcpy(sh.look, g_cam_look, sizeof(sh.look));
        changed();
      }
      ImGui::SameLine();
      if (ImGui::Button("Copy")) {
        s_clip = sh;
        s_clip_full = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Paste") && s_clip_full) {
        // the copied framing onto this shot (its time stays)
        std::memcpy(sh.pos, s_clip.pos, sizeof(sh.pos));
        std::memcpy(sh.look, s_clip.look, sizeof(sh.look));
        sh.smooth = s_clip.smooth;
        changed();
      }
      ImGui::SameLine();
      if (ImGui::Button("Delete")) {
        gone = (int)i;
      }
      ImGui::PopID();
    }
    if (gone >= 0) {
      g_scene.shots.erase(g_scene.shots.begin() + gone);
      changed();
    }
    ImGui::TreePop();
  }

  // voice lines and their subtitles
  if (ImGui::TreeNodeEx("Voice lines", ImGuiTreeNodeFlags_DefaultOpen)) {
    // (the mp3s there, looked for again whenever a line is added)
    static std::vector<std::string> files;
    if (files.empty()) {
      files = voice_files();
    }
    if (ImGui::Button("Add a line at the cursor")) {
      files = voice_files();
      Line l;
      l.t = g_cursor;
      l.speaker = "DAXTER";
      g_scene.lines.push_back(l);
      sort_scene(g_scene);
      changed();
    }
    int gone = -1;
    for (size_t i = 0; i < g_scene.lines.size(); i++) {
      auto& l = g_scene.lines[i];
      ImGui::PushID(1000 + (int)i);
      ImGui::Separator();
      ImGui::SetNextItemWidth(80);
      if (ImGui::DragFloat("##lt", &l.t, 0.02f, 0.f, g_scene.length, "%.2f s")) {
        changed();
      }
      ImGui::SameLine();
      ImGui::SetNextItemWidth(220);
      if (ImGui::BeginCombo("##file", l.file.empty() ? "(pick an mp3)" : l.file.c_str())) {
        for (const auto& f : files) {
          if (ImGui::Selectable(f.c_str(), f == l.file)) {
            l.file = f;
            l.len = enhanced_mp3_length(f);
            changed();
          }
        }
        ImGui::EndCombo();
      }
      ImGui::SameLine();
      char spk[64];
      std::snprintf(spk, sizeof(spk), "%s", l.speaker.c_str());
      ImGui::SetNextItemWidth(110);
      if (ImGui::InputText("##spk", spk, sizeof(spk))) {
        l.speaker = spk;
        changed();
      }
      ImGui::SameLine();
      ImGui::Text("%.1f s", l.len);
      ImGui::SameLine();
      if (ImGui::Button("Delete line")) {
        gone = (int)i;
      }
      // its subtitles
      int sub_gone = -1;
      for (size_t k = 0; k < l.subs.size(); k++) {
        auto& sub = l.subs[k];
        ImGui::PushID((int)k);
        ImGui::Indent(20.f);
        ImGui::SetNextItemWidth(80);
        if (ImGui::DragFloat("##st", &sub.t, 0.02f, 0.f, std::max(l.len, 0.1f), "+%.2f s")) {
          changed();
        }
        ImGui::SameLine();
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s", sub.text.c_str());
        ImGui::SetNextItemWidth(-90);
        if (ImGui::InputText("##text", buf, sizeof(buf))) {
          sub.text = buf;
          changed();
        }
        ImGui::SameLine();
        if (ImGui::Button("x")) {
          sub_gone = (int)k;
        }
        ImGui::Unindent(20.f);
        ImGui::PopID();
      }
      if (sub_gone >= 0) {
        l.subs.erase(l.subs.begin() + sub_gone);
        changed();
      }
      ImGui::Indent(20.f);
      if (ImGui::Button("Add a subtitle at the cursor")) {
        l.subs.push_back({std::max(0.f, g_cursor - l.t), ""});
        sort_scene(g_scene);
        changed();
      }
      ImGui::Unindent(20.f);
      ImGui::PopID();
    }
    if (gone >= 0) {
      g_scene.lines.erase(g_scene.lines.begin() + gone);
      changed();
    }
    ImGui::TreePop();
  }

  // the characters' animations
  if (ImGui::TreeNodeEx("Animations", ImGuiTreeNodeFlags_DefaultOpen)) {
    animations_menu();
    ImGui::TreePop();
  }

  // where the characters are, and which way they face
  if (ImGui::TreeNodeEx("Places", ImGuiTreeNodeFlags_DefaultOpen)) {
    moves_menu();
    ImGui::TreePop();
  }

  // their faces: the mouth opened by hand, ...
  if (ImGui::TreeNodeEx("Face", ImGuiTreeNodeFlags_DefaultOpen)) {
    faces_menu();
    ImGui::TreePop();
  }

  // their body parts, turned and moved on top of their animations
  if (ImGui::TreeNodeEx("Body parts", ImGuiTreeNodeFlags_DefaultOpen)) {
    poses_menu();
    ImGui::TreePop();
  }
  ImGui::PopID();
}

}  // namespace enhanced_cutscene
