"""Jak Living Sky: Dark Jak's transformation, an animation of Jak's own skeleton made from numbers (no
Blender, no clips): from his standing pose (eichar-stance-loop's first frame) he hunches, trembling, then
bursts open (back arched, head thrown back, arms flung wide) and settles. Written as the nap's glb is
(scripts/enhanced/nap_anims.py: Jak's skeleton, one tiny triangle, the animation), for
(build-actor "darkjak-turn" :master-art-group "eichar-ag") in game.gp; target2.gc plays it.

    python scripts/enhanced/darkjak_anim.py <eichar-lod0.glb (decompiler rip)> <jak-nap.glb> <out.glb>

Rotations are put on in the world (glTF: Y up, Jak facing +Z, his left +X) about each joint, each joint's
local worked out again under its parent, so the joints' own axes never matter.
"""
import json
import math
import struct
import sys

import numpy as np

FPS = 30.0
SECONDS = 0.6


def read_glb(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    off = 20 + n
    bl = struct.unpack("<I", b[off:off + 4])[0]
    return j, b[off + 8:off + 8 + bl]


def acc(j, binary, i):
    a = j["accessors"][i]
    bv = j["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
    return np.frombuffer(binary, dtype=np.float32, count=a["count"] * n,
                         offset=bv.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(a["count"], n)


def q2m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m2q(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def trs(t, q, s):
    m = np.eye(4)
    m[:3, :3] = q2m(q) @ np.diag(s)
    m[:3, 3] = t
    return m


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def smooth(t, a, b):
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    x = (t - a) / (b - a)
    return x * x * (3 - 2 * x)


def deltas(t, f):
    """each joint's turn in the world at time t: h hunched and trembling, u burst open"""
    h = smooth(t, 0.0, 0.12) * (1 - smooth(t, 0.24, 0.32))
    u = smooth(t, 0.24, 0.32) * (1 - smooth(t, 0.4, 0.6))
    jit = h * (1 if f % 2 else -1) * 2.5
    return h, {
        "upper_body": rot("x", 18 * h - 16 * u),
        "chest": rot("x", 14 * h - 14 * u + jit),
        "neckA": rot("x", 14 * h - 20 * u - jit),
        "neckB": rot("x", 8 * h - 12 * u),
        "Lshould": rot("z", -8 * h + 22 * u), "Rshould": rot("z", 8 * h - 22 * u),
        # (an arm hanging down turned about +X swings back: forward is the other way)
        "Larm": rot("z", 65 * u) @ rot("x", -35 * h), "Rarm": rot("z", -65 * u) @ rot("x", -35 * h),
        "Lforarm": rot("x", -50 * h), "Rforarm": rot("x", -50 * h),
        # legs a little apart (more as he bursts), knees bent a little: not a trunk, not squashed down
        "Lthigh": rot("x", -14 * h) @ rot("z", 6 * h + 10 * u), "Rthigh": rot("x", -14 * h) @ rot("z", -6 * h - 10 * u),
        "Lknee": rot("x", 24 * h), "Rknee": rot("x", 24 * h),
        "Lankle": rot("x", -10 * h) @ rot("z", -6 * h - 10 * u), "Rankle": rot("x", -10 * h) @ rot("z", 6 * h + 10 * u),
    }


def main():
    eich, nap, out = sys.argv[1:4]
    je, be = read_glb(eich)
    jn, bn = read_glb(nap)
    # Jak's standing pose: eichar-stance-loop's first frame, by joint name
    an = [a for a in je["animations"] if a["name"] == "eichar-stance-loop"][0]
    pose = {}
    for ch in an["channels"]:
        name = je["nodes"][ch["target"]["node"]]["name"]
        v = acc(je, be, an["samplers"][ch["sampler"]]["output"])[0]
        pose.setdefault(name, {})[ch["target"]["path"]] = v.astype(float)
    nodes = jn["nodes"]
    joints = jn["skins"][0]["joints"]
    parent = {}
    for i in joints:
        for c in nodes[i].get("children", []):
            parent[c] = i
    base = {}
    for i in joints:
        nd = nodes[i]
        p = pose.get(nd["name"], {})
        base[i] = (np.array(p.get("translation", nd.get("translation", [0, 0, 0])), float),
                   np.array(p.get("rotation", nd.get("rotation", [0, 0, 0, 1])), float),
                   np.array(p.get("scale", nd.get("scale", [1, 1, 1])), float))
    order = []

    def visit(i):
        order.append(i)
        for c in nodes[i].get("children", []):
            if c in base:
                visit(c)

    for i in joints:
        if parent.get(i) not in base:
            visit(i)
    n = int(round(SECONDS * FPS)) + 1
    frames = []
    for f in range(n):
        h, d = deltas(f / FPS, f)
        world, local = {}, {}
        for i in order:
            tl, ql, sl = base[i]
            if nodes[i]["name"] == "main":
                tl = tl + np.array([0, -0.04 * h, 0])
            m = trs(tl, ql, sl)
            p = world.get(parent.get(i), np.eye(4))
            w = p @ m
            r = d.get(nodes[i]["name"])
            if r is not None:
                w = w.copy()
                w[:3, :3] = r @ w[:3, :3]
                m = np.linalg.inv(p) @ w
            world[i] = w
            local[i] = m
        frames.append(local)
    # the nap's glb with this one animation in place of its own
    blob = bytearray(bn)
    accs, views = jn["accessors"], jn["bufferViews"]

    def add(arr, kind, count, extra=None):
        while len(blob) % 4:
            blob.append(0)
        data = np.asarray(arr, np.float32).tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        a = {"bufferView": len(views) - 1, "componentType": 5126, "count": count, "type": kind}
        if extra:
            a.update(extra)
        accs.append(a)
        return len(accs) - 1

    tacc = add([f / FPS for f in range(n)], "SCALAR", n, {"min": [0.0], "max": [(n - 1) / FPS]})
    samplers, channels = [], []
    for i in joints:
        ts, qs, ss = [], [], []
        for local in frames:
            m = local[i]
            s = np.linalg.norm(m[:3, :3], axis=0)
            q = m2q(m[:3, :3] / s)
            if qs and np.dot(q, qs[-1]) < 0:
                q = -q
            ts.append(m[:3, 3])
            qs.append(q)
            ss.append(s)
        for path, kind, vals in (("translation", "VEC3", ts), ("rotation", "VEC4", qs), ("scale", "VEC3", ss)):
            samplers.append({"input": tacc, "interpolation": "LINEAR",
                             "output": add(np.array(vals).ravel(), kind, len(vals))})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": i, "path": path}})
    jn["animations"] = [{"name": "eichar-darkjak-turn", "samplers": samplers, "channels": channels}]
    nodes[0]["name"] = "darkjak-turn"
    for m in jn["meshes"]:
        m["name"] = "darkjak-turn"
    jn["buffers"] = [{"byteLength": len(blob)}]
    js = json.dumps(jn, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    while len(blob) % 4:
        blob.append(0)
    with open(out, "wb") as fo:
        fo.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        fo.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fo.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))
    print("ok", n, "frames")


main()
