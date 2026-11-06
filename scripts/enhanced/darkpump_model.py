# Jak Living Sky: the dark eco pump in Abandoned Caves (the fisherman's mission): a rusty tank on a base,
# purple bands glowing, a domed top with a piston, two pipes bent down into the pool, a valve wheel. A glb for
# build-actor (game.gp), coloured in its vertices (textures don't reach the game for these; merc doubles them,
# so they're halved here). Units: metres, +Z its front (the pipes' side), Y up.
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


def ring_path(n, r, y=0.0):
    return [(r * math.cos(2 * math.pi * k / n), y, r * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]


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


parts = []
# the base: a dark slab, rusty edges
parts.append(box((-1.5, 0.0, -1.5), (1.5, 0.35, 1.5), DARK))
parts.append(box((-1.6, 0.0, -1.6), (1.6, 0.12, 1.6), RUST))
# the tank: rust and metal rings, two purple bands glowing
tank_y = np.linspace(0.35, 2.75, 25)


def tank_colour(i):
    y = tank_y[i]
    if 1.05 < y < 1.25 or 2.05 < y < 2.25:
        return GLOW
    # (rust streaks: more of it low down, where the dark eco splashes)
    return RUST if (i % 4 == 0 or (i < 8 and i % 2 == 0)) else METAL


parts.append(tube([(0, y, 0) for y in tank_y], [1.0] * len(tank_y), tank_colour, sides=24))
# the dome on top, and its piston
dome = [(0, 2.75 + 0.55 * math.sin(a), 0) for a in np.linspace(0, math.pi / 2, 10)]
dome_r = [1.0 * math.cos(a) + 0.02 for a in np.linspace(0, math.pi / 2, 10)]
parts.append(tube(dome, dome_r, lambda i: METAL, sides=24, caps=False))
parts.append(tube([(0, 3.2, 0), (0, 3.9, 0)], [0.22, 0.22], lambda i: BRASS))
parts.append(tube([(0, 3.9, 0), (0, 4.05, 0)], [0.38, 0.38], lambda i: DARK))
# two pipes out of the tank's front, bending down (into the pool), purple at the joints
for sx in (-0.45, 0.45):
    # (long enough to reach over the island's edge, down into the pool)
    path = bez((sx, 1.6, 0.85), (sx, 1.8, 3.5), (sx * 1.4, 1.6, 5.8), (sx * 1.5, -1.5, 6.6), 24)
    parts.append(tube(path, [0.2] * 24, lambda i: GLOW if i in (0, 1, 11, 12, 22, 23) else (RUST if i % 5 == 0 else METAL),
                      sides=12))
    # a flange where it leaves the tank
    parts.append(tube([(sx, 1.6, 0.9), (sx, 1.6, 1.08)], [0.3, 0.3], lambda i: BRASS, sides=12))
# the valve wheel on the front, between the pipes
wheel = [(x, y + 0.9, 1.15) for x, y, _ in [(0.4 * math.cos(a), 0.4 * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 25)]]
parts.append(tube(wheel, [0.06] * 25, lambda i: BRASS, sides=8, caps=False))
for a in (0, math.pi / 2):
    d = np.array([math.cos(a), math.sin(a), 0]) * 0.4
    parts.append(tube([(-d[0], 0.9 - d[1], 1.15), (d[0], 0.9 + d[1], 1.15)], [0.04, 0.04], lambda i: BRASS, sides=6))
parts.append(tube([(0, 0.9, 1.0), (0, 0.9, 1.15)], [0.08, 0.08], lambda i: DARK, sides=8))

V, Nm, C, I = [], [], [], []
for v, n, c, i in parts:
    o = len(V)
    V += list(v)
    Nm += list(n)
    C += list(c)
    I += [k + o for k in i]
V = np.array(V, np.float32) * SCALE
Nm = np.array(Nm, np.float32)
Nm /= np.linalg.norm(Nm, axis=1)[:, None]
C4 = np.ones((len(V), 4), np.float32)
C4[:, :3] = np.array(C, np.float32) * 0.5  # (merc doubles them)
# (the purple bands at full strength, glowing against the cave's dark)
glow = np.all(np.isclose(np.array(C, np.float32), GLOW), axis=1)
C4[glow, :3] = GLOW
UV = np.zeros((len(V), 2), np.float32)
I = np.array(I, np.uint32)

bio = io.BytesIO()
Image.new('RGB', (4, 4), (255, 255, 255)).save(bio, 'PNG')
tex = bio.getvalue()


def write_glb(path, name):
    blobs = [V.tobytes(), Nm.tobytes(), UV.tobytes(), I.tobytes(), tex, C4.tobytes()]
    views, data = [], b''
    for bl in blobs:
        while len(data) % 4:
            data += b'\0'
        views.append({'buffer': 0, 'byteOffset': len(data), 'byteLength': len(bl)})
        data += bl
    while len(data) % 4:
        data += b'\0'
    for k in range(3):
        views[k]['target'] = 34962
    views[3]['target'] = 34963
    accs = [{'bufferView': 0, 'componentType': 5126, 'count': len(V), 'type': 'VEC3',
             'min': V.min(0).tolist(), 'max': V.max(0).tolist()},
            {'bufferView': 1, 'componentType': 5126, 'count': len(V), 'type': 'VEC3'},
            {'bufferView': 2, 'componentType': 5126, 'count': len(V), 'type': 'VEC2'},
            {'bufferView': 3, 'componentType': 5125, 'count': len(I), 'type': 'SCALAR'},
            {'bufferView': 5, 'componentType': 5126, 'count': len(V), 'type': 'VEC4'}]
    g = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
         'nodes': [{'name': name, 'mesh': 0}],
         'meshes': [{'name': name, 'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'TEXCOORD_0': 2,
                                                                  'COLOR_0': 4}, 'indices': 3, 'material': 0}]}],
         'materials': [{'name': name + '-tex', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0},
                                                                       'metallicFactor': 0.0,
                                                                       'roughnessFactor': 0.6}}],
         'textures': [{'source': 0, 'sampler': 0}], 'samplers': [{'magFilter': 9729, 'minFilter': 9987}],
         'images': [{'bufferView': 4, 'mimeType': 'image/png', 'name': name + '-tex'}],
         'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(g).encode()
    while len(js) % 4:
        js += b' '
    out = (b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) + struct.pack('<I4s', len(js), b'JSON') +
           js + struct.pack('<I4s', len(data), b'BIN\0') + data)
    open(path, 'wb').write(out)


# (the same model twice: custom_levels for build-actor, common/<name>-lod0 so the extractor puts it in GAME)
write_glb(os.path.join(OUT, 'custom_levels', 'darkpump.glb'), 'darkpump')
write_glb(os.path.join(OUT, 'common', 'darkpump-lod0.glb'), 'darkpump')
print('ok', len(V), 'vertices', len(I) // 3, 'triangles')
