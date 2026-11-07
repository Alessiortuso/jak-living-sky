#include "enhanced_cutscene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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

struct Scene {
  std::string name;
  float length = 10.f;
  bool letterbox = true;
  bool hold_jak = true;
  std::vector<Shot> shots;
  std::vector<Line> lines;
  std::vector<AnimKey> anims;
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
bool g_dirty = false;
std::vector<Puppet> g_puppets;  // (poll) the characters played now
// the characters near the camera, and their animations (the game's list, edit)
std::vector<std::pair<std::string, std::vector<std::string>>> g_actors;
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
void puppets_at(const Scene& s, float t) {
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
      g_puppets.push_back({k->actor, k->anim, (t - k->t) * k->speed, k->loop});
    }
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
  if (!load(name, &s)) {
    return false;
  }
  if (g_mode != kNone) {
    stop_voices();
  }
  g_scene = s;
  g_scene_loaded = true;
  g_dirty = false;
  g_mode = kPlay;
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
  if (g_mode == kNone) {
    g_puppets.clear();
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
    puppets_at(g_scene, g_cursor);
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
    } else {
      g_mode = kNone;
      f.mode = kNone;
      g_puppets.clear();
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
  return f;
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
  g_actors.clear();
}

void actor_anim(const std::string& actor, const std::string& anim) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_actors.empty() || g_actors.back().first != actor) {
    g_actors.push_back({actor, {}});
  }
  if (!anim.empty()) {
    g_actors.back().second.push_back(anim);
  }
}

void report_camera(const float* pos, const float* look) {
  std::lock_guard<std::mutex> lock(g_mutex);
  for (int k = 0; k < 3; k++) {
    g_cam_pos[k] = pos[k] / kMeter;
    g_cam_look[k] = look[k] / kMeter;
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
      g_dirty = true;
    }
  }
  ImGui::TextDisabled("(each key: from its time on, the character plays it; \"its own\" gives it back its own)");
  int gone = -1;
  for (size_t i = 0; i < g_scene.anims.size(); i++) {
    auto& k = g_scene.anims[i];
    ImGui::PushID(5000 + (int)i);
    ImGui::SetNextItemWidth(80);
    if (ImGui::DragFloat("##at", &k.t, 0.02f, 0.f, g_scene.length, "%.2f s")) {
      g_dirty = true;
    }
    ImGui::SameLine();
    ImGui::Text("%s", k.actor.c_str());
    ImGui::SameLine();
    if (anim_combo("##anim", k.actor, &k.anim)) {
      g_dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("loop", &k.loop)) {
      g_dirty = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    if (ImGui::DragFloat("##speed", &k.speed, 0.01f, 0.1f, 4.f, "x%.2f")) {
      g_dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Go")) {
      g_cursor = k.t;
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete")) {
      gone = (int)i;
    }
    ImGui::PopID();
  }
  if (gone >= 0) {
    g_scene.anims.erase(g_scene.anims.begin() + gone);
    g_dirty = true;
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
    g_dirty = true;
    g_cursor = 0.f;
  }
  if (!g_scene_loaded) {
    ImGui::TextDisabled("Pick a scene, or name a new one.");
    ImGui::PopID();
    return;
  }
  ImGui::SameLine();
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
    g_dirty = true;
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Black bars", &g_scene.letterbox)) {
    g_dirty = true;
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Hold Jak", &g_scene.hold_jak)) {
    g_dirty = true;
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
      g_dirty = true;
    }
    if (s_clip_full) {
      if (ImGui::Button("Paste the copied shot at the cursor")) {
        Shot sh = s_clip;
        sh.t = g_cursor;
        g_scene.shots.push_back(sh);
        sort_scene(g_scene);
        g_dirty = true;
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
        g_dirty = true;
      }
      ImGui::SameLine();
      if (ImGui::Checkbox("move to next", &sh.smooth)) {
        g_dirty = true;
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
        g_dirty = true;
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
        g_dirty = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Delete")) {
        gone = (int)i;
      }
      ImGui::PopID();
    }
    if (gone >= 0) {
      g_scene.shots.erase(g_scene.shots.begin() + gone);
      g_dirty = true;
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
      g_dirty = true;
    }
    int gone = -1;
    for (size_t i = 0; i < g_scene.lines.size(); i++) {
      auto& l = g_scene.lines[i];
      ImGui::PushID(1000 + (int)i);
      ImGui::Separator();
      ImGui::SetNextItemWidth(80);
      if (ImGui::DragFloat("##lt", &l.t, 0.02f, 0.f, g_scene.length, "%.2f s")) {
        g_dirty = true;
      }
      ImGui::SameLine();
      ImGui::SetNextItemWidth(220);
      if (ImGui::BeginCombo("##file", l.file.empty() ? "(pick an mp3)" : l.file.c_str())) {
        for (const auto& f : files) {
          if (ImGui::Selectable(f.c_str(), f == l.file)) {
            l.file = f;
            l.len = enhanced_mp3_length(f);
            g_dirty = true;
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
        g_dirty = true;
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
          g_dirty = true;
        }
        ImGui::SameLine();
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s", sub.text.c_str());
        ImGui::SetNextItemWidth(-90);
        if (ImGui::InputText("##text", buf, sizeof(buf))) {
          sub.text = buf;
          g_dirty = true;
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
        g_dirty = true;
      }
      ImGui::Indent(20.f);
      if (ImGui::Button("Add a subtitle at the cursor")) {
        l.subs.push_back({std::max(0.f, g_cursor - l.t), ""});
        sort_scene(g_scene);
        g_dirty = true;
      }
      ImGui::Unindent(20.f);
      ImGui::PopID();
    }
    if (gone >= 0) {
      g_scene.lines.erase(g_scene.lines.begin() + gone);
      g_dirty = true;
    }
    ImGui::TreePop();
  }

  // the characters' animations
  if (ImGui::TreeNodeEx("Animations", ImGuiTreeNodeFlags_DefaultOpen)) {
    animations_menu();
    ImGui::TreePop();
  }
  ImGui::PopID();
}

}  // namespace enhanced_cutscene
