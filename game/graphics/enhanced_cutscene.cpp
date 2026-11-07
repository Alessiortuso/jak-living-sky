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
  std::string actor;  // who says it: their mouth follows its loudness (empty: nobody's)
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
  // each speaker's face shape that opens the mouth (-1 none), how far loud speech opens it
  std::map<std::string, std::pair<int, float>> mouths;
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
std::atomic<bool> g_edit_now{false};  // (editing, or its preview: read by the input thread)
bool g_tl_focus_now = false;           // (the timeline has the pad: Start plays)
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
      l.actor = jl.value("actor", std::string());
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
    for (const auto& [a, jm] : j.value("mouths", nlohmann::json::object()).items()) {
      s.mouths[a] = {jm.value("shape", -1), jm.value("gain", 1.f)};
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
    nlohmann::json jl = {{"t", l.t}, {"file", l.file}, {"speaker", l.speaker}, {"actor", l.actor}, {"volume", l.volume},
                         {"len", l.len}, {"subs", nlohmann::json::array()}};
    for (const auto& sub : l.subs) {
      jl["subs"].push_back({{"t", sub.t}, {"text", sub.text}});
    }
    j["lines"].push_back(jl);
  }
  j["mouths"] = nlohmann::json::object();
  for (const auto& [a, m] : s.mouths) {
    j["mouths"][a] = {{"shape", m.first}, {"gain", m.second}};
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

const std::vector<float>& lips_of(const std::string& file);
extern std::string g_cap_face_actor;
extern int g_cap_face_shape;
void capture_step(Frame& f);
void capture_start(bool faces_only);
bool capturing();
std::string capture_status();

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
  auto face_of = [&](const std::string& actor) -> Face& {
    for (auto& f : g_faces) {
      if (f.actor == actor) return f;
    }
    Face f;
    f.actor = actor;
    g_faces.push_back(f);
    return g_faces.back();
  };
  // the speakers' lips
  for (const auto& l : s.lines) {
    if (l.actor.empty() || t < l.t || t >= l.t + l.len) continue;
    auto m = s.mouths.find(l.actor);
    if (m == s.mouths.end() || m->second.first < 0 || m->second.first >= 16) continue;
    const auto& env = lips_of(l.file);
    const size_t i = (size_t)((t - l.t) * 60.f);
    if (i < env.size()) face_of(l.actor).w[m->second.first] += env[i] * m->second.second;
  }
  // (a capture's face sheet: that one shape alone)
  if (!g_cap_face_actor.empty()) {
    Face& f = face_of(g_cap_face_actor);
    for (float& w : f.w) w = 0.f;
    if (g_cap_face_shape >= 0 && g_cap_face_shape < 16) f.w[g_cap_face_shape] = 1.f;
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
  g_edit_now = g_mode == kEdit || g_mode == kPreview;
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
    capture_step(f);
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

void undo_redo_locked(int dir);

void undo_redo(int dir) {
  std::lock_guard<std::mutex> lock(g_mutex);
  undo_redo_locked(dir);
}

void undo_redo_locked(int dir) {
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
  // (the timeline's: play from the cursor, or stop)
  if (down && g_mode == kPreview) {
    stop_voices();
    g_mode = kEdit;
    return;
  }
  if (down && g_mode == kEdit && g_tl_focus_now) {
    g_mode = kPreview;
    start_clock(g_cursor);
    return;
  }
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
  // each speaker's mouth: the shape its lines open, how far
  for (const auto& l : g_scene.lines) {
    if (l.actor.empty()) continue;
    auto& m = g_scene.mouths[l.actor];
    if (m.second == 0.f && m.first == 0) m = {-1, 1.f};
  }
  for (auto& [a, m] : g_scene.mouths) {
    ImGui::PushID(a.c_str());
    ImGui::Text("%s's lips:", a.c_str());
    ImGui::SameLine();
    int shape = m.first + 1;
    ImGui::SetNextItemWidth(90);
    if (ImGui::InputInt("mouth shape (0 none)", &shape)) {
      m.first = std::clamp(shape, 0, 16) - 1;
      changed();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    if (ImGui::SliderFloat("how open", &m.second, 0.f, 2.f, "%.2f")) {
      changed();
    }
    ImGui::PopID();
  }
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

  // captures for Claude (pictures of the scene, the face shapes, the skeletons)
  if (g_mode == kEdit) {
    if (capturing()) {
      ImGui::TextColored(ImVec4(1.f, 0.8f, 0.3f, 1.f), "%s (hands off the pad)", capture_status().c_str());
    } else {
      if (ImGui::Button("Capture for Claude (pictures, face shapes, skeletons)")) {
        capture_start(false);
      }
      ImGui::SameLine();
      if (ImGui::Button("Faces only")) {
        capture_start(true);
      }
    }
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
      ImGui::SetNextItemWidth(140);
      if (ImGui::BeginCombo("##lips", l.actor.empty() ? "(lips: nobody)" : l.actor.c_str())) {
        if (ImGui::Selectable("(lips: nobody)", l.actor.empty())) {
          l.actor.clear();
          changed();
        }
        for (const auto& a : g_actors) {
          if (ImGui::Selectable(a.first.c_str(), a.first == l.actor)) {
            l.actor = a.first;
            changed();
          }
        }
        ImGui::EndCombo();
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


// ---------------------------------------------------------------------------------------------------------
// the timeline bar: at the bottom of the screen while editing, driven by the pad (Triangle: the timeline, or
// the camera and the characters as before). A row for the camera's shots, one for the voices, one for each
// character (its animations as blocks, its places, body parts and faces as marks), the cursor across them

namespace {

constexpr u32 kPadStart = 1u << 3, kPadUp = 1u << 4, kPadDown = 1u << 6, kPadL2 = 1u << 8, kPadR2 = 1u << 9,
              kPadL1 = 1u << 10, kPadR1 = 1u << 11, kPadTriangle = 1u << 12, kPadCircle = 1u << 13,
              kPadX = 1u << 14, kPadSquare = 1u << 15;

bool g_tl_focus = false;        // the pad on the timeline
int g_tl_row = 0;               // the row picked
float g_tl_view0 = 0.f;         // the time shown at the left
float g_tl_view = 0.f;          // how much time is shown (0: all of it)
Clock::time_point g_tl_last = Clock::now();
// the list opened on a row (a character's animations, the voices' mp3s)
bool g_tl_list = false;
int g_tl_list_kind = 0;         // 0 animations, 1 mp3s
int g_tl_list_at = 0;
std::vector<std::string> g_tl_items;
std::string g_tl_list_actor;
std::string g_tl_note;          // a word on what was just done
Clock::time_point g_tl_note_at;

struct Row {
  int kind = 0;  // 0 camera, 1 voices, 2 a character
  std::string actor;
};

// the rows: the camera, the voices, the characters with keys, then those near the camera
std::vector<Row> tl_rows() {
  std::vector<Row> rows = {{0, ""}, {1, ""}};
  auto add = [&](const std::string& a) {
    if (a.empty()) {
      return;
    }
    for (const auto& r : rows) {
      if (r.kind == 2 && r.actor == a) {
        return;
      }
    }
    rows.push_back({2, a});
  };
  for (const auto& k : g_scene.anims) add(k.actor);
  for (const auto& k : g_scene.moves) add(k.actor);
  for (const auto& k : g_scene.poses) add(k.actor);
  for (const auto& k : g_scene.faces) add(k.actor);
  for (const auto& a : g_actors) add(a.first);
  return rows;
}

void tl_note(const std::string& s) {
  g_tl_note = s;
  g_tl_note_at = Clock::now();
}

// the times of a row's keys
std::vector<float> tl_times(const Row& r) {
  std::vector<float> out;
  if (r.kind == 0) {
    for (const auto& s : g_scene.shots) out.push_back(s.t);
  } else if (r.kind == 1) {
    for (const auto& l : g_scene.lines) {
      out.push_back(l.t);
      for (const auto& s : l.subs) out.push_back(l.t + s.t);
    }
  } else {
    for (const auto& k : g_scene.anims) if (k.actor == r.actor) out.push_back(k.t);
    for (const auto& k : g_scene.moves) if (k.actor == r.actor) out.push_back(k.t);
    for (const auto& k : g_scene.poses) if (k.actor == r.actor) out.push_back(k.t);
    for (const auto& k : g_scene.faces) if (k.actor == r.actor) out.push_back(k.t);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// the keys of a row at the cursor gone
int tl_delete(const Row& r) {
  const float t = g_cursor;
  auto close_to = [&](float x) { return std::fabs(x - t) < 0.06f; };
  size_t before = 0, after = 0;
  if (r.kind == 0) {
    before = g_scene.shots.size();
    std::erase_if(g_scene.shots, [&](const Shot& s) { return close_to(s.t); });
    after = g_scene.shots.size();
  } else if (r.kind == 1) {
    before = g_scene.lines.size();
    std::erase_if(g_scene.lines, [&](const Line& l) { return close_to(l.t); });
    after = g_scene.lines.size();
  } else {
    before = g_scene.anims.size() + g_scene.moves.size() + g_scene.poses.size() + g_scene.faces.size();
    std::erase_if(g_scene.anims, [&](const AnimKey& k) { return k.actor == r.actor && close_to(k.t); });
    std::erase_if(g_scene.moves, [&](const MoveKey& k) { return k.actor == r.actor && close_to(k.t); });
    std::erase_if(g_scene.poses, [&](const PoseKey& k) { return k.actor == r.actor && close_to(k.t); });
    std::erase_if(g_scene.faces, [&](const FaceKey& k) { return k.actor == r.actor && close_to(k.t); });
    after = g_scene.anims.size() + g_scene.moves.size() + g_scene.poses.size() + g_scene.faces.size();
  }
  if (after != before) {
    changed();
  }
  return (int)(before - after);
}

void tl_list_open(int kind, const std::string& actor) {
  g_tl_items.clear();
  if (kind == 0) {
    g_tl_items.push_back("(its own animation)");
    for (const auto& a : g_actors) {
      if (a.first == actor) {
        for (const auto& n : a.second) g_tl_items.push_back(n);
      }
    }
  } else {
    g_tl_items = voice_files();
  }
  if (g_tl_items.empty()) {
    tl_note(kind == 0 ? "no animations: fly closer to it" : "no mp3s in custom_assets/jak1/audio");
    return;
  }
  g_tl_list = true;
  g_tl_list_kind = kind;
  g_tl_list_at = 0;
  g_tl_list_actor = actor;
}

void tl_list_pick() {
  const std::string& item = g_tl_items[g_tl_list_at];
  if (g_tl_list_kind == 0) {
    AnimKey k;
    k.t = g_cursor;
    k.actor = g_tl_list_actor;
    k.anim = g_tl_list_at == 0 ? std::string() : item;
    // (one there already: changed)
    bool done = false;
    for (auto& a : g_scene.anims) {
      if (a.actor == k.actor && std::fabs(a.t - k.t) < 0.06f) {
        a.anim = k.anim;
        done = true;
      }
    }
    if (!done) {
      g_scene.anims.push_back(k);
    }
    tl_note(g_tl_list_actor + ": " + item);
  } else {
    Line l;
    l.t = g_cursor;
    l.file = item;
    l.len = enhanced_mp3_length(item);
    l.speaker = "DAXTER";
    g_scene.lines.push_back(l);
    tl_note("voice " + item + " (its speaker and subtitles: the menu)");
  }
  sort_scene(g_scene);
  changed();
  g_tl_list = false;
}

}  // namespace

bool timeline_pad(u32 pressed, u32 held, float lx, float ly, float rx) {
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto now = Clock::now();
  const float dt = std::clamp(std::chrono::duration<float>(now - g_tl_last).count(), 0.f, 0.1f);
  g_tl_last = now;
  if (g_mode != kEdit || !g_scene_loaded) {
    g_tl_focus = false;
    g_tl_focus_now = false;
    g_tl_list = false;
    return false;
  }
  if (pressed & kPadTriangle) {
    if (g_tl_list) {
      g_tl_list = false;
    } else {
      g_tl_focus = !g_tl_focus;
      tl_note(g_tl_focus ? "timeline" : "camera");
    }
  }
  g_tl_focus_now = g_tl_focus;
  if (!g_tl_focus) {
    return false;
  }
  const auto rows = tl_rows();
  g_tl_row = std::clamp(g_tl_row, 0, (int)rows.size() - 1);
  const Row& row = rows[g_tl_row];

  // the list opened on a row
  if (g_tl_list) {
    const int n = (int)g_tl_items.size();
    if (pressed & kPadDown) g_tl_list_at = (g_tl_list_at + 1) % n;
    if (pressed & kPadUp) g_tl_list_at = (g_tl_list_at + n - 1) % n;
    static float s_scroll = 0.f;
    s_scroll -= ly * dt * 12.f;
    while (s_scroll > 1.f) { g_tl_list_at = std::min(n - 1, g_tl_list_at + 1); s_scroll -= 1.f; }
    while (s_scroll < -1.f) { g_tl_list_at = std::max(0, g_tl_list_at - 1); s_scroll += 1.f; }
    if (pressed & kPadX) tl_list_pick();
    if (pressed & kPadCircle) g_tl_list = false;
    return true;
  }

  // time: the left stick (square slow, circle fast); the view: the right stick zooms
  const float len = std::max(g_scene.length, 0.5f);
  if (g_tl_view <= 0.f || g_tl_view > len) g_tl_view = len;
  const float speed = (held & kPadSquare) ? 0.2f : (held & kPadCircle) ? 3.f : 1.f;
  if (lx != 0.f) {
    g_cursor = std::clamp(g_cursor + lx * dt * speed * std::max(1.f, g_tl_view * 0.25f), 0.f, len);
  }
  if (rx != 0.f) {
    g_tl_view = std::clamp(g_tl_view * (1.f - rx * dt * 1.5f), 1.f, len);
  }
  // (the cursor kept in view)
  if (g_cursor < g_tl_view0 + g_tl_view * 0.1f) g_tl_view0 = g_cursor - g_tl_view * 0.1f;
  if (g_cursor > g_tl_view0 + g_tl_view * 0.9f) g_tl_view0 = g_cursor - g_tl_view * 0.9f;
  g_tl_view0 = std::clamp(g_tl_view0, 0.f, std::max(0.f, len - g_tl_view));

  // (the D-pad's left and right: back a change, on again, as everywhere)
  constexpr u32 kPadRight = 1u << 5, kPadLeft = 1u << 7;
  if (pressed & kPadLeft) {
    undo_redo_locked(-1);
    tl_note("undo");
  }
  if (pressed & kPadRight) {
    undo_redo_locked(1);
    tl_note("redo");
  }
  // rows: the D-pad's up and down
  if (pressed & kPadDown) g_tl_row = std::min((int)rows.size() - 1, g_tl_row + 1);
  if (pressed & kPadUp) g_tl_row = std::max(0, g_tl_row - 1);

  // L1 / R1: the row's key before / after
  if (pressed & (kPadL1 | kPadR1)) {
    const auto times = tl_times(row);
    float best = g_cursor;
    for (float x : times) {
      if ((pressed & kPadL1) && x < g_cursor - 0.01f) best = x;
      if ((pressed & kPadR1) && x > g_cursor + 0.01f) {
        best = x;
        break;
      }
    }
    g_cursor = best;
  }

  // X: a key here (a shot from the camera; a voice; a character's place now)
  if (pressed & kPadX) {
    if (row.kind == 0) {
      if (g_cam_known) {
        Shot sh;
        sh.t = g_cursor;
        std::memcpy(sh.pos, g_cam_pos, sizeof(sh.pos));
        std::memcpy(sh.look, g_cam_look, sizeof(sh.look));
        std::erase_if(g_scene.shots, [&](const Shot& s) { return std::fabs(s.t - g_cursor) < 0.06f; });
        g_scene.shots.push_back(sh);
        sort_scene(g_scene);
        changed();
        tl_note("shot: what the camera sees");
      }
    } else if (row.kind == 1) {
      tl_list_open(1, "");
    } else {
      MoveKey k;
      k.t = g_cursor;
      k.actor = row.actor;
      if (actor_now(k.actor, k.pos, &k.yaw)) {
        std::erase_if(g_scene.moves,
                      [&](const MoveKey& m) { return m.actor == k.actor && std::fabs(m.t - g_cursor) < 0.06f; });
        g_scene.moves.push_back(k);
        sort_scene(g_scene);
        changed();
        tl_note(row.actor + ": its place here");
      } else {
        tl_note("not near the camera: fly closer to it");
      }
    }
  }
  // R2: a character's animations
  if ((pressed & kPadR2) && row.kind == 2) {
    tl_list_open(0, row.actor);
  }
  // square: the row's keys here gone
  if (pressed & kPadSquare) {
    const int n = tl_delete(row);
    tl_note(n ? "deleted" : "nothing here to delete");
  }
  return true;
}

void draw_overlay() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_scene_loaded || (g_mode != kEdit && g_mode != kPreview)) {
    return;
  }
  auto* dl = ImGui::GetForegroundDrawList();
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  const float fs = std::max(16.f, ds.y / 45.f);  // (readable on a small screen)
  auto* font = ImGui::GetFont();
  const auto rows = tl_rows();
  const int shown = std::min((int)rows.size(), 7);
  const int first = std::clamp(g_tl_row - shown + 1, 0, std::max(0, (int)rows.size() - shown));
  const float rh = fs * 1.5f;
  const float h = fs * 3.6f + rh * shown;
  const float x0 = ds.x * 0.02f, x1 = ds.x * 0.98f, y0 = ds.y - h - ds.y * 0.02f;
  const float lw = fs * 9.f;  // the names' column
  const float tx0 = x0 + lw, tx1 = x1 - fs * 0.5f;
  dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y0 + h), IM_COL32(12, 14, 20, g_tl_focus ? 225 : 150), fs * 0.4f);
  if (g_tl_focus) {
    dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y0 + h), IM_COL32(255, 200, 80, 255), fs * 0.4f, 0, 2.f);
  }
  const float len = std::max(g_scene.length, 0.5f);
  const float v = (g_tl_view <= 0.f || g_tl_view > len) ? len : g_tl_view;
  const float v0 = std::clamp(g_tl_view0, 0.f, std::max(0.f, len - v));
  auto x_of = [&](float t) { return tx0 + (tx1 - tx0) * (t - v0) / v; };
  auto in = [&](float t) { return t >= v0 - 0.001f && t <= v0 + v + 0.001f; };

  // the header: the scene, the time, what the pad does now
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s   %.2f / %.1f s   %s", g_scene.name.c_str(), g_cursor, g_scene.length,
                g_mode == kPreview ? "PLAYING (Start: stop)" : g_tl_focus ? "TIMELINE" : "CAMERA");
  dl->AddText(font, fs, ImVec2(x0 + fs * 0.5f, y0 + fs * 0.3f), IM_COL32(255, 255, 255, 255), buf);
  const char* hint =
      g_tl_list ? "D-pad / left stick: choose   X: put it at the cursor   Circle: close"
      : g_tl_focus
          ? "Left stick: time   D-pad: row   L1/R1: keys   X: add   R2: animations   Square: delete   "
            "Start: play   Triangle: camera"
          : "Triangle: timeline   (camera, R2/L2, L1+X... as always)";
  dl->AddText(font, fs * 0.8f, ImVec2(x0 + fs * 0.5f, y0 + fs * 1.4f), IM_COL32(190, 190, 200, 255), hint);
  if (!g_tl_note.empty() && Clock::now() - g_tl_note_at < std::chrono::seconds(3)) {
    const ImVec2 sz = font->CalcTextSizeA(fs * 0.85f, FLT_MAX, 0.f, g_tl_note.c_str());
    dl->AddText(font, fs * 0.85f, ImVec2(x1 - sz.x - fs * 0.5f, y0 + fs * 0.35f), IM_COL32(255, 210, 90, 255),
                g_tl_note.c_str());
  }

  // the ruler
  const float ry = y0 + fs * 2.5f;
  const float step = v > 40.f ? 5.f : v > 12.f ? 1.f : 0.5f;
  for (float t = std::ceil(v0 / step) * step; t <= v0 + v; t += step) {
    const float x = x_of(t);
    const bool big = std::fmod(t, step * 5.f) < 0.001f;
    dl->AddLine(ImVec2(x, ry + fs * (big ? 0.2f : 0.6f)), ImVec2(x, ry + fs), IM_COL32(120, 120, 135, 255));
    if (big) {
      std::snprintf(buf, sizeof(buf), "%.0f", t);
      dl->AddText(font, fs * 0.7f, ImVec2(x + 2.f, ry - fs * 0.15f), IM_COL32(150, 150, 165, 255), buf);
    }
  }

  // the rows
  for (int i = 0; i < shown; i++) {
    const Row& r = rows[first + i];
    const float yy = ry + fs * 1.1f + rh * i;
    const bool picked = g_tl_focus && first + i == g_tl_row;
    if (picked) {
      dl->AddRectFilled(ImVec2(x0 + 2.f, yy), ImVec2(x1 - 2.f, yy + rh - 2.f), IM_COL32(255, 200, 80, 45));
    }
    dl->AddRectFilled(ImVec2(tx0, yy + 2.f), ImVec2(tx1, yy + rh - 4.f), IM_COL32(30, 32, 42, 255), 3.f);
    const char* name = r.kind == 0 ? "Camera" : r.kind == 1 ? "Voices" : r.actor.c_str();
    dl->AddText(font, fs * 0.85f, ImVec2(x0 + fs * 0.5f, yy + rh * 0.2f),
                picked ? IM_COL32(255, 220, 120, 255) : IM_COL32(220, 220, 230, 255), name);
    const float my = yy + rh * 0.5f;
    if (r.kind == 0) {
      for (const auto& s : g_scene.shots) {
        if (!in(s.t)) continue;
        const float x = x_of(s.t);
        dl->AddRectFilled(ImVec2(x - 3.f, yy + 4.f), ImVec2(x + 3.f, yy + rh - 6.f),
                          s.smooth ? IM_COL32(110, 200, 255, 255) : IM_COL32(60, 120, 255, 255));
      }
    } else if (r.kind == 1) {
      for (const auto& l : g_scene.lines) {
        const float a = std::max(l.t, v0), b = std::min(l.t + std::max(l.len, 0.3f), v0 + v);
        if (b <= a) continue;
        dl->AddRectFilled(ImVec2(x_of(a), yy + 5.f), ImVec2(x_of(b), yy + rh - 7.f), IM_COL32(220, 140, 40, 210),
                          3.f);
        dl->PushClipRect(ImVec2(x_of(a), yy), ImVec2(x_of(b), yy + rh), true);
        const auto slash = l.file.find_last_of('/');
        dl->AddText(font, fs * 0.7f, ImVec2(x_of(a) + 4.f, yy + rh * 0.25f), IM_COL32(30, 20, 10, 255),
                    l.file.substr(slash == std::string::npos ? 0 : slash + 1).c_str());
        dl->PopClipRect();
      }
    } else {
      // its animations: blocks, each up to its next
      std::vector<const AnimKey*> ks;
      for (const auto& k : g_scene.anims) if (k.actor == r.actor) ks.push_back(&k);
      for (size_t k = 0; k < ks.size(); k++) {
        const float a0 = ks[k]->t, a1 = k + 1 < ks.size() ? ks[k + 1]->t : len;
        const float a = std::max(a0, v0), b = std::min(a1, v0 + v);
        if (b <= a || ks[k]->anim.empty()) continue;
        dl->AddRectFilled(ImVec2(x_of(a), yy + 5.f), ImVec2(x_of(b) - 1.f, yy + rh - 7.f),
                          IM_COL32(70, 170, 95, 220), 3.f);
        dl->PushClipRect(ImVec2(x_of(a), yy), ImVec2(x_of(b) - 2.f, yy + rh), true);
        dl->AddText(font, fs * 0.7f, ImVec2(x_of(a) + 4.f, yy + rh * 0.25f), IM_COL32(10, 30, 15, 255),
                    ks[k]->anim.c_str());
        dl->PopClipRect();
      }
      for (const auto& k : g_scene.moves) {
        if (k.actor != r.actor || !in(k.t)) continue;
        const float x = x_of(k.t);
        dl->AddQuadFilled(ImVec2(x, my - 6.f), ImVec2(x + 6.f, my), ImVec2(x, my + 6.f), ImVec2(x - 6.f, my),
                          IM_COL32(255, 215, 80, 255));
      }
      for (const auto& k : g_scene.poses) {
        if (k.actor != r.actor || !in(k.t)) continue;
        dl->AddCircleFilled(ImVec2(x_of(k.t), yy + rh - 7.f), 3.5f, IM_COL32(200, 120, 255, 255));
      }
      for (const auto& k : g_scene.faces) {
        if (k.actor != r.actor || !in(k.t)) continue;
        const float x = x_of(k.t);
        dl->AddTriangleFilled(ImVec2(x, yy + 4.f), ImVec2(x - 4.f, yy + 11.f), ImVec2(x + 4.f, yy + 11.f),
                              IM_COL32(255, 130, 170, 255));
      }
    }
  }
  if ((int)rows.size() > shown) {
    std::snprintf(buf, sizeof(buf), "%d/%d", g_tl_row + 1, (int)rows.size());
    dl->AddText(font, fs * 0.7f, ImVec2(x0 + fs * 0.5f, y0 + h - fs * 0.9f), IM_COL32(150, 150, 165, 255), buf);
  }

  // the cursor
  if (in(g_cursor)) {
    const float cx = x_of(g_cursor);
    dl->AddLine(ImVec2(cx, ry), ImVec2(cx, y0 + h - 4.f), IM_COL32(255, 70, 70, 255), 2.f);
    dl->AddTriangleFilled(ImVec2(cx - 6.f, ry), ImVec2(cx + 6.f, ry), ImVec2(cx, ry + 8.f), IM_COL32(255, 70, 70, 255));
  }

  // the list, over the bar
  if (g_tl_list) {
    const int n = (int)g_tl_items.size();
    const int vis = std::min(n, 10);
    const int top = std::clamp(g_tl_list_at - vis / 2, 0, std::max(0, n - vis));
    const float lh = fs * 1.25f, lw2 = fs * 18.f;
    const float lx0 = std::clamp(x_of(g_cursor) - lw2 * 0.5f, x0, x1 - lw2), ly1 = y0 - fs * 0.4f;
    const float ly0 = ly1 - lh * vis - fs * 1.6f;
    dl->AddRectFilled(ImVec2(lx0, ly0), ImVec2(lx0 + lw2, ly1), IM_COL32(18, 20, 28, 240), fs * 0.3f);
    dl->AddRect(ImVec2(lx0, ly0), ImVec2(lx0 + lw2, ly1), IM_COL32(255, 200, 80, 255), fs * 0.3f, 0, 2.f);
    std::snprintf(buf, sizeof(buf), "%s  (%d/%d)",
                  g_tl_list_kind == 0 ? g_tl_list_actor.c_str() : "Voices", g_tl_list_at + 1, n);
    dl->AddText(font, fs * 0.85f, ImVec2(lx0 + fs * 0.5f, ly0 + fs * 0.3f), IM_COL32(255, 220, 120, 255), buf);
    for (int i = 0; i < vis; i++) {
      const int at = top + i;
      const float yy = ly0 + fs * 1.4f + lh * i;
      if (at == g_tl_list_at) {
        dl->AddRectFilled(ImVec2(lx0 + 3.f, yy), ImVec2(lx0 + lw2 - 3.f, yy + lh), IM_COL32(255, 200, 80, 70));
      }
      dl->AddText(font, fs * 0.85f, ImVec2(lx0 + fs * 0.6f, yy + lh * 0.1f), IM_COL32(230, 230, 240, 255),
                  g_tl_items[at].c_str());
    }
  }
}


// ---------------------------------------------------------------------------------------------------------
// lips: each voice's loudness, the speaker's mouth opened with it

namespace {

constexpr float kLipFps = 60.f;
std::map<std::string, std::vector<float>> g_env;  // each mp3's mouth, 0..1, kLipFps a second

const std::vector<float>& lips_of(const std::string& file) {
  auto it = g_env.find(file);
  if (it != g_env.end()) {
    return it->second;
  }
  auto rms = enhanced_mp3_envelope(file, kLipFps);
  std::vector<float> out(rms.size());
  if (!rms.empty()) {
    // loud as its loud parts (not its peaks), a little quiet under it read as shut, smoothed: quick to open,
    // slower to close, as a mouth does
    auto sorted = rms;
    std::sort(sorted.begin(), sorted.end());
    const float ref = std::max(1e-4f, sorted[(size_t)(sorted.size() * 0.9)]);
    float s = 0.f;
    for (size_t i = 0; i < rms.size(); i++) {
      const float v = std::clamp((rms[i] / ref - 0.12f) / 0.88f, 0.f, 1.f);
      s += (v - s) * (v > s ? 0.55f : 0.22f);
      out[i] = s;
    }
  }
  return g_env.emplace(file, std::move(out)).first->second;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// captures

namespace {

struct CapStep {
  int kind = 0;  // 0 the scene at t, 1 a face shape (-1: none), 2 the skeletons
  float t = 0.f;
  std::string actor;
  int shape = -1;
  std::string file;
};
std::vector<CapStep> g_cap;
size_t g_cap_i = 0;
int g_cap_wait = 0;
fs::path g_cap_dir;
std::string g_cap_request;     // (the render thread takes it)
std::string g_cap_face_actor;  // (a face shape shown alone, for the sheet)
int g_cap_face_shape = -1;
bool g_cap_dump_now = false;
struct Joint {
  int j = 0;
  int parent = -1;
  std::string name;
  float m[16] = {};
};
std::map<std::string, std::vector<Joint>> g_skel;

fs::path cap_dir() {
  return dir() / "captures" / g_scene.name;
}

// a face's head, metres (its skeleton's "head" joint), and which way it faces
bool head_of(const std::string& actor, float* pos) {
  auto it = g_skel.find(actor);
  if (it == g_skel.end()) {
    return false;
  }
  for (const auto& jt : it->second) {
    std::string n = jt.name;
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    if (n == "head" || n == "headb" || n == "neckb") {
      for (int k = 0; k < 3; k++) pos[k] = jt.m[12 + k] / kMeter;
      return true;
    }
  }
  return false;
}

// a face, from its skeleton: where its mouth is, between its eyes, and the way it looks (flat), metres
bool face_view(const std::string& actor, float* mouth, float* eyes, float* fwd) {
  auto it = g_skel.find(actor);
  float head[3];
  if (it == g_skel.end() || !head_of(actor, head)) {
    return false;
  }
  float e[3] = {}, m[3] = {};
  int ne = 0;
  bool got_mouth = false;
  for (const auto& jt : it->second) {
    std::string n = jt.name;
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    if (n == "leye" || n == "reye" || n == "eyel" || n == "eyer") {
      for (int k = 0; k < 3; k++) e[k] += jt.m[12 + k] / kMeter;
      ne++;
    }
    if (!got_mouth && (n == "jaw" || n == "mouth")) {
      for (int k = 0; k < 3; k++) m[k] = jt.m[12 + k] / kMeter;
      got_mouth = true;
    }
  }
  if (ne == 0) {
    return false;
  }
  for (int k = 0; k < 3; k++) e[k] /= (float)ne;
  if (!got_mouth) {
    for (int k = 0; k < 3; k++) m[k] = e[k];
    m[1] -= 0.25f;
  }
  float d[3] = {e[0] - head[0], 0.f, e[2] - head[2]};
  const float len = std::sqrt(d[0] * d[0] + d[2] * d[2]);
  if (len < 1e-4f) {
    return false;
  }
  for (int k = 0; k < 3; k++) {
    fwd[k] = d[k] / len;
    mouth[k] = m[k];
    eyes[k] = e[k];
  }
  return true;
}

void capture_start(bool faces_only = false) {
  g_cap.clear();
  g_cap_i = 0;
  g_cap_wait = 0;
  g_skel.clear();
  g_cap_dir = cap_dir();
  std::error_code ec;
  if (!faces_only) {
    fs::remove_all(g_cap_dir, ec);
  }
  fs::create_directories(g_cap_dir, ec);
  g_cap.push_back({2});
  // the scene: every second, and each shot just after it begins
  std::vector<float> ts;
  for (float x = 0.f; x < g_scene.length; x += 1.f) ts.push_back(x);
  for (const auto& s : g_scene.shots) ts.push_back(std::min(g_scene.length - 0.05f, s.t + 0.3f));
  std::sort(ts.begin(), ts.end());
  ts.erase(std::unique(ts.begin(), ts.end(), [](float a, float b) { return std::fabs(a - b) < 0.2f; }), ts.end());
  if (faces_only) ts.clear();
  for (float x : ts) {
    char name[64];
    std::snprintf(name, sizeof(name), "scene_%06.2f.png", x);
    g_cap.push_back({0, x, "", -1, name});
  }
  // each speaker's face shapes, one at a time (and none)
  std::vector<std::string> faces;
  for (const auto& l : g_scene.lines) {
    if (!l.actor.empty() && std::find(faces.begin(), faces.end(), l.actor) == faces.end()) faces.push_back(l.actor);
  }
  for (const auto& [a, n] : g_actor_face) {
    if (std::find(faces.begin(), faces.end(), a) == faces.end()) faces.push_back(a);
  }
  for (const auto& a : faces) {
    const int n = g_actor_face.count(a) ? std::min(16, g_actor_face[a]) : 0;
    for (int k = -1; k < n; k++) {
      char name[128];
      std::snprintf(name, sizeof(name), "face_%s_%02d.png", a.c_str(), k + 1);
      g_cap.push_back({1, 0.f, a, k, name});
    }
  }
  tl_note("capturing: hands off the pad");
}

void capture_write_skeletons() {
  for (const auto& [actor, joints] : g_skel) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& jt : joints) {
      nlohmann::json m = nlohmann::json::array();
      for (float x : jt.m) m.push_back(x);
      j.push_back({{"index", jt.j}, {"parent", jt.parent}, {"name", jt.name}, {"world", m}});
    }
    file_util::write_text_file(g_cap_dir / ("skeleton_" + actor + ".json"), j.dump(1));
  }
}

void capture_write_index() {
  nlohmann::json j;
  j["scene"] = g_scene.name;
  j["length"] = g_scene.length;
  j["note"] = "scene_T.png: the scene at T seconds, through its shot; face_ACTOR_NN.png: that face shape at 1 "
              "(00: none), the scene's time 0; skeleton_ACTOR.json: joints, world matrices in game units "
              "(4096 a metre), rows: x, y, z axes then the place";
  nlohmann::json faces = nlohmann::json::object();
  for (const auto& [a, n] : g_actor_face) faces[a] = n;
  j["face_shapes"] = faces;
  nlohmann::json at = nlohmann::json::object();
  for (const auto& [a, v] : g_actor_at) at[a] = {{"pos", {v.pos[0], v.pos[1], v.pos[2]}}, {"yaw", v.yaw}};
  j["characters"] = at;
  file_util::write_text_file(g_cap_dir / "index.json", j.dump(2));
}

// once a frame (poll, editing): the capture's next step
void capture_step(Frame& f) {
  if (g_cap_i >= g_cap.size()) {
    return;
  }
  const CapStep& s = g_cap[g_cap_i];
  if (g_cap_wait == 0) {
    g_cap_face_actor.clear();
    g_cap_face_shape = -1;
    if (s.kind == 0) {
      g_cursor = s.t;
      float pos[3], look[3];
      if (camera_at(g_scene, s.t, pos, look)) {
        std::memcpy(g_jump_pos, pos, sizeof(pos));
        std::memcpy(g_jump_look, look, sizeof(look));
        g_jump = true;
      }
    } else if (s.kind == 1) {
      g_cursor = 0.f;
      g_cap_face_actor = s.actor;
      g_cap_face_shape = s.shape;
      // (in front of its face, its mouth and eyes in the middle)
      float mouth[3], eyes[3], fwd[3];
      if (face_view(s.actor, mouth, eyes, fwd)) {
        for (int k = 0; k < 3; k++) {
          const float mid = (mouth[k] + eyes[k]) * 0.5f;
          g_jump_pos[k] = mid + fwd[k] * 1.5f + (k == 1 ? 0.1f : 0.f);
          g_jump_look[k] = mid;
        }
        g_jump = true;
      }
    } else {
      g_cap_dump_now = true;
      f.dump = true;
    }
  }
  g_cap_wait++;
  const int settle = s.kind == 0 ? 40 : s.kind == 1 ? 10 : 3;
  if (g_cap_wait == settle) {
    if (s.kind == 2) {
      capture_write_skeletons();
    } else {
      g_cap_request = (g_cap_dir / s.file).string();
    }
  }
  if (g_cap_wait >= settle + 4) {
    g_cap_wait = 0;
    g_cap_i++;
    if (g_cap_i >= g_cap.size()) {
      g_cap_face_actor.clear();
      capture_write_index();
      tl_note("captures done: " + g_cap_dir.string());
      lg::info("[cutscene] captures done: {}", g_cap_dir.string());
    }
  }
}

}  // namespace

std::string capture_path() {
  std::lock_guard<std::mutex> lock(g_mutex);
  std::string p;
  p.swap(g_cap_request);
  return p;
}

void joint_dump(const std::string& actor, int j, int parent, const std::string& name, const float* m) {
  std::lock_guard<std::mutex> lock(g_mutex);
  Joint jt;
  jt.j = j;
  jt.parent = parent;
  jt.name = name;
  std::memcpy(jt.m, m, sizeof(jt.m));
  g_skel[actor].push_back(jt);
}

namespace {
bool capturing() {
  return g_cap_i < g_cap.size();
}
std::string capture_status() {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "capturing %d / %d", (int)g_cap_i + 1, (int)g_cap.size());
  return buf;
}
}  // namespace

}  // namespace enhanced_cutscene
