# Jak Living Sky: the dark eco pump in Abandoned Caves (the fisherman's mission): a steel tank on a stone base,
# riveted flanges, welded seams, purple bands glowing, a domed top with a piston, a pressure gauge, two pipes
# bent down into the pool, a valve wheel. A glb for build-actor (game.gp).
#
# Its parts go in three materials named for the renderer's own looks (EnhancedMaterials.h): "steel" and "brass"
# shine as metal, the base stays dull. No texture reaches the game for these: the colours, wear included (plates
# uneven, rust running from under the bands, grime at the foot), are painted into the vertices (merc doubles
# them, so they're halved here). Units: metres (then SCALE), +Z its front (the pipes' side), Y up.
#
#   python scripts/enhanced/darkpump_model.py custom_assets/jak1/models
import io
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image

OUT = sys.argv[1]

METAL = (0.30, 0.29, 0.31)
RUST = (0.62, 0.32, 0.14)
DARK = (0.16, 0.14, 0.17)
GLOW = (1.00, 0.40, 1.00)
BRASS = (0.80, 0.62, 0.26)
SCALE = 1.6  # (bigger than first made: a machine you can't miss)

STEEL, BRASS_M, BASE = 'darkpump-steel', 'darkpump-brass', 'darkpump-base'


def tube(path, radii, colour_at, sides=16, caps=True):
    """a tube along path (N x 3), radius per point, colour per point (k: 0..N-1)"""
    P = np.array(path, float)
    N = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    n = np.cross(T[0], ref)
    n /= np.linalg.norm(n)
    V, Nm, C, I = [], [], [], []
    for i in range(N):
        n = n - np.dot(n, T[i]) * T[i]
        n /= np.linalg.norm(n)
        b = np.cross(T[i], n)
        for s in range(sides + 1):
            a = 2 * math.pi * s / sides
            d = math.cos(a) * n + math.sin(a) * b
            V.append(P[i] + d * radii[i])
            Nm.append(d)
            C.append(colour_at(i))
    for i in range(N - 1):
        for s in range(sides):
            a = i * (sides + 1) + s
            c = a + sides + 1
            I += [a, a + 1, c, a + 1, c + 1, c]
    if caps:
        for end, sgn in ((0, -1), (N - 1, 1)):
            centre = len(V)
            V.append(P[end])
            Nm.append(T[end] * sgn)
            C.append(colour_at(end))
            ring = end * (sides + 1)
            for s in range(sides):
                if sgn < 0:
                    I += [centre, ring + s + 1, ring + s]
                else:
                    I += [centre, ring + s, ring + s + 1]
    return V, Nm, C, I


def box(lo, hi, colour):
    lo, hi = np.array(lo, float), np.array(hi, float)
    V, Nm, C, I = [], [], [], []
    faces = [((0, 1), 2, 1), ((0, 1), 2, -1), ((0, 2), 1, 1), ((0, 2), 1, -1), ((1, 2), 0, 1), ((1, 2), 0, -1)]
    for (u, v), w, sgn in faces:
        base = len(V)
        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = np.zeros(3)
            p[u] = (lo, hi)[du][u]
            p[v] = (lo, hi)[dv][v]
            p[w] = hi[w] if sgn > 0 else lo[w]
            V.append(p)
            nn = np.zeros(3)
            nn[w] = sgn
            Nm.append(nn)
            C.append(colour)
        # (winding outward)
        e1 = V[base + 1] - V[base]
        e2 = V[base + 2] - V[base]
        if np.dot(np.cross(e1, e2), Nm[base]) > 0:
            I += [base, base + 1, base + 2, base, base + 2, base + 3]
        else:
            I += [base, base + 2, base + 1, base, base + 3, base + 2]
    return V, Nm, C, I


def bez(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.array, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


parts = []  # (material, (V, N, C, I))
# the base: a dark slab on a rusty foot
parts.append((BASE, box((-1.5, 0.0, -1.5), (1.5, 0.35, 1.5), DARK)))
parts.append((BASE, box((-1.6, 0.0, -1.6), (1.6, 0.12, 1.6), RUST)))
# the tank: steel plates, two purple bands glowing, thick flanges top and bottom
tank_y = np.linspace(0.35, 2.75, 49)


def tank_colour(i):
    y = tank_y[i]
    if 1.08 < y < 1.22 or 2.08 < y < 2.22:
        return GLOW
    return METAL


parts.append((STEEL, tube([(0, y, 0) for y in tank_y], [1.0] * len(tank_y), tank_colour, sides=40)))
for y0 in (0.35, 2.62):
    parts.append((STEEL, tube([(0, y0, 0), (0, y0 + 0.14, 0)], [1.08, 1.08], lambda i: DARK, sides=40)))
# rivets round the flanges, and above and below each purple band
for y, r in ((0.42, 1.09), (2.69, 1.09), (1.0, 1.01), (1.3, 1.01), (2.0, 1.01), (2.3, 1.01)):
    for k in range(20):
        a = 2 * math.pi * (k + 0.5) / 20
        d = np.array([math.cos(a), 0, math.sin(a)])
        c = np.array([0, y, 0]) + d * r
        parts.append((STEEL, tube([c, c + d * 0.05], [0.035, 0.03], lambda i: METAL, sides=6)))
# welded seams down the plates
for k in range(4):
    a = 2 * math.pi * (k + 0.5) / 4
    d = np.array([math.cos(a), 0, math.sin(a)])
    parts.append((STEEL, tube([d * 1.0 + (0, 0.5, 0), d * 1.0 + (0, 2.6, 0)], [0.025, 0.025], lambda i: DARK, sides=5)))
# the dome on top, and its piston
dome = [(0, 2.76 + 0.55 * math.sin(a), 0) for a in np.linspace(0, math.pi / 2, 14)]
dome_r = [1.0 * math.cos(a) + 0.02 for a in np.linspace(0, math.pi / 2, 14)]
parts.append((STEEL, tube(dome, dome_r, lambda i: METAL, sides=40, caps=False)))
parts.append((BRASS_M, tube([(0, 3.2, 0), (0, 3.9, 0)], [0.22, 0.22], lambda i: BRASS, sides=16)))
parts.append((STEEL, tube([(0, 3.9, 0), (0, 4.05, 0)], [0.38, 0.38], lambda i: DARK, sides=16)))
# a pressure gauge on the side: a brass rim round a pale face
gd = np.array([0.707, 0, 0.707])
g = np.array([0, 1.75, 0]) + gd * 0.98
parts.append((BRASS_M, tube([g, g + gd * 0.12], [0.2, 0.2], lambda i: BRASS, sides=16)))
parts.append((BASE, tube([g + gd * 0.12, g + gd * 0.125], [0.16, 0.16], lambda i: (0.85, 0.82, 0.72), sides=16)))
# two pipes out of the tank's front, bending down into the pool, purple at the joints
for sx in (-0.45, 0.45):
    # (long enough to reach over the island's edge, down into the pool)
    path = bez((sx, 1.6, 0.85), (sx, 1.8, 3.5), (sx * 1.4, 1.6, 5.8), (sx * 1.5, -1.5, 6.6), 40)
    parts.append((STEEL, tube(path, [0.2] * 40, lambda i: GLOW if i in (0, 1, 19, 20, 38, 39) else METAL, sides=16)))
    # brass flanges where it leaves the tank, and a third of the way along
    parts.append((BRASS_M, tube([(sx, 1.6, 0.9), (sx, 1.6, 1.08)], [0.3, 0.3], lambda i: BRASS, sides=16)))
    tdir = (path[13] - path[11]) / np.linalg.norm(path[13] - path[11])
    parts.append((BRASS_M, tube([path[12] - tdir * 0.06, path[12] + tdir * 0.06], [0.27, 0.27], lambda i: BRASS,
                                sides=16)))
# the valve wheel on the front, between the pipes
wheel = [(0.4 * math.cos(a), 0.9 + 0.4 * math.sin(a), 1.15) for a in np.linspace(0, 2 * math.pi, 33)]
parts.append((BRASS_M, tube(wheel, [0.06] * 33, lambda i: BRASS, sides=8, caps=False)))
for a in (0, math.pi / 2):
    d = np.array([math.cos(a), math.sin(a), 0]) * 0.4
    parts.append((BRASS_M, tube([(-d[0], 0.9 - d[1], 1.15), (d[0], 0.9 + d[1], 1.15)], [0.04, 0.04], lambda i: BRASS,
                                sides=6)))
parts.append((STEEL, tube([(0, 0.9, 1.0), (0, 0.9, 1.15)], [0.08, 0.08], lambda i: DARK, sides=8)))


def weather(V, C):
    """wear painted into the vertices: the plates uneven, rust running down from under the bands, grime and rust
    round the foot"""
    V = np.asarray(V, float)
    C = np.asarray(C, float).copy()
    glow = np.all(np.isclose(C, GLOW), axis=1)
    ang = np.arctan2(V[:, 2], V[:, 0])
    y = V[:, 1]
    h = np.sin(V[:, 0] * 13.1 + V[:, 1] * 7.7) * np.sin(V[:, 2] * 11.3 - V[:, 1] * 5.3)
    C *= (0.9 + 0.12 * h)[:, None]
    # streaks: thin, at a few angles, from under each band down a while
    streak = np.clip(np.sin(ang * 9 + 1.3) * np.sin(ang * 4 - 0.6), 0, 1) ** 3
    run = np.zeros(len(V))
    for top in (1.08, 2.08):
        d = top - y
        run = np.maximum(run, np.where((d > 0) & (d < 0.75), 1 - d / 0.75, 0))
    rust = np.clip(streak * run * 1.6 + np.clip((0.75 - y) / 0.5, 0, 1) * 0.55, 0, 1)
    C = C * (1 - rust[:, None]) + np.array(RUST) * rust[:, None]
    # grime low down
    grime = np.clip((0.9 - y) / 0.6, 0, 1) * 0.35
    C *= (1 - grime)[:, None]
    C[glow] = GLOW
    return C


# merged by material: one primitive each
groups = {}
for mat, (v, n, c, i) in parts:
    g_ = groups.setdefault(mat, [[], [], [], []])
    o = len(g_[0])
    g_[0] += list(v)
    g_[1] += list(n)
    g_[2] += list(c)
    g_[3] += [k + o for k in i]
prims = []
for mat, (v, n, c, i) in groups.items():
    V_ = np.array(v, np.float32)
    C_ = weather(V_, c) if mat == STEEL else np.array(c, float)
    N_ = np.array(n, np.float32)
    N_ /= np.linalg.norm(N_, axis=1)[:, None]
    C4_ = np.ones((len(V_), 4), np.float32)
    C4_[:, :3] = C_ * 0.5  # (merc doubles them)
    glow_ = np.all(np.isclose(C_, GLOW), axis=1)
    C4_[glow_, :3] = GLOW  # (the purple bands at full strength, glowing against the cave's dark)
    prims.append((mat, (V_ * SCALE).astype(np.float32), N_, np.zeros((len(V_), 2), np.float32), C4_,
                  np.array(i, np.uint32)))


def png_white():
    bio = io.BytesIO()
    Image.new('RGB', (4, 4), (255, 255, 255)).save(bio, 'PNG')
    return bio.getvalue()


def write_glb(path, name):
    """one primitive per material, each with its own (white) picture named for it"""
    data = bytearray()
    views, accs, mats, texs, imgs, prim_js = [], [], [], [], [], []

    def add(blob, target=None):
        while len(data) % 4:
            data.append(0)
        v = {'buffer': 0, 'byteOffset': len(data), 'byteLength': len(blob)}
        if target:
            v['target'] = target
        views.append(v)
        data.extend(blob)
        return len(views) - 1

    for k, (mat, V_, N_, UV_, C4_, I_) in enumerate(prims):
        a0 = len(accs)
        accs.append({'bufferView': add(V_.tobytes(), 34962), 'componentType': 5126, 'count': len(V_), 'type': 'VEC3',
                     'min': V_.min(0).tolist(), 'max': V_.max(0).tolist()})
        accs.append({'bufferView': add(N_.tobytes(), 34962), 'componentType': 5126, 'count': len(V_), 'type': 'VEC3'})
        accs.append({'bufferView': add(UV_.tobytes(), 34962), 'componentType': 5126, 'count': len(V_), 'type': 'VEC2'})
        accs.append({'bufferView': add(I_.tobytes(), 34963), 'componentType': 5125, 'count': len(I_), 'type': 'SCALAR'})
        accs.append({'bufferView': add(C4_.tobytes(), 34962), 'componentType': 5126, 'count': len(V_), 'type': 'VEC4'})
        imgs.append({'bufferView': add(png_white()), 'mimeType': 'image/png', 'name': mat})
        texs.append({'source': k, 'sampler': 0})
        mats.append({'name': mat, 'pbrMetallicRoughness': {'baseColorTexture': {'index': k}, 'metallicFactor': 0.0,
                                                           'roughnessFactor': 0.6}})
        prim_js.append({'attributes': {'POSITION': a0, 'NORMAL': a0 + 1, 'TEXCOORD_0': a0 + 2, 'COLOR_0': a0 + 4},
                        'indices': a0 + 3, 'material': k})
    while len(data) % 4:
        data.append(0)
    g = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
         'nodes': [{'name': name, 'mesh': 0}], 'meshes': [{'name': name, 'primitives': prim_js}],
         'materials': mats, 'textures': texs, 'samplers': [{'magFilter': 9729, 'minFilter': 9987}], 'images': imgs,
         'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(g).encode()
    while len(js) % 4:
        js += b' '
    out = (b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) + struct.pack('<I4s', len(js), b'JSON') +
           js + struct.pack('<I4s', len(data), b'BIN\0') + bytes(data))
    open(path, 'wb').write(out)


# (the same model twice: custom_levels for build-actor, common/<name>-lod0 so the extractor puts it in GAME)
write_glb(os.path.join(OUT, 'custom_levels', 'darkpump.glb'), 'darkpump')
write_glb(os.path.join(OUT, 'common', 'darkpump-lod0.glb'), 'darkpump')
print('ok', sum(len(p[1]) for p in prims), 'vertices', sum(len(p[5]) for p in prims) // 3, 'triangles',
      [p[0] for p in prims])
