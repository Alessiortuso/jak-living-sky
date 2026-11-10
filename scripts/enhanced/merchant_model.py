# Jak Living Sky: the white eco merchant and his stall, one in each village (the shop moved off the warp gates).
# A fat little trader in a patched ochre cloak and hood, a red scarf, a rope belt with a pouch, a sack and a blue
# bottle; in the hood's shadow only his eyes, normal ones (white, brown pupils, lids that blink); long elf ears out of
# the hood's sides; two short bony legs under the belly, big shoes. Behind him his stall: rough posts lashed with
# rope, a red and cream striped awning, a plank counter with pots, crates, a basket of fruit, and the shop's sign: a
# white eco orb in a rope net hanging from an arm. (from the concept chosen, mercante-concept/concept-5.png)
#
# A skinned glb for build-actor (game.gp): joints main, body (breathing), head, the lids, the ears, the arms; one
# animation, "idle" (a loop: breathing, blinks, an ear twitch, the hands moving). Every vertex on one joint. No
# texture reaches the game: the colours are painted into the vertices (merc doubles them: halved here). Materials
# named for the renderer's looks (EnhancedMaterials.h). Units: metres, Y up, +Z his front.
#
#   python scripts/enhanced/merchant_model.py custom_assets/jak1/models
import io
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image

OUT = sys.argv[1]

OCHRE = (0.80, 0.56, 0.20)
OCHRE_D = (0.55, 0.36, 0.12)
TEAL = (0.33, 0.55, 0.52)
RED = (0.72, 0.20, 0.13)
CREAM = (0.93, 0.86, 0.68)
ROPE = (0.78, 0.63, 0.40)
BROWN = (0.36, 0.20, 0.10)
GLOVE = (0.30, 0.17, 0.10)
BONE = (0.86, 0.76, 0.60)
SKIN = (0.96, 0.66, 0.50)
BLACK = (0.03, 0.02, 0.02)
WHITE = (0.97, 0.96, 0.92)
PUPIL = (0.30, 0.17, 0.08)
LID = (0.10, 0.06, 0.04)
BLUE = (0.16, 0.42, 0.74)
WOOD = (0.50, 0.32, 0.17)
WOOD_D = (0.36, 0.22, 0.11)
CLAY = (0.80, 0.42, 0.22)
GREENPOT = (0.30, 0.55, 0.48)
FRUIT = (0.98, 0.72, 0.15)
GLOW = (1.0, 1.0, 1.0)

# joints: name, parent, bind position (world). "align" and "prejoint" first, as the game's own skeletons (and the
# board's): without them build-actor and the extractor number the joints one apart, each part on its neighbour's
JOINTS = [
    ('align', -1, (0, 0, 0)),
    ('prejoint', 0, (0, 0, 0)),
    ('main', 1, (0, 0, 0)),
    ('body', 2, (0, 0.45, 0)),
    ('head', 3, (0, 1.38, 0.02)),
    ('lid_l', 4, (0.12, 1.60, 0.40)),
    ('lid_r', 4, (-0.12, 1.60, 0.40)),
    ('ear_l', 4, (0.30, 1.62, 0.02)),
    ('ear_r', 4, (-0.30, 1.62, 0.02)),
    ('arm_l', 3, (0.52, 1.20, 0.05)),
    ('arm_r', 3, (-0.52, 1.20, 0.05)),
]
J = {n: i for i, (n, _, _) in enumerate(JOINTS)}

parts = []  # (material, V, N, C, I, joint)


def add(mat, mesh, joint):
    parts.append((mat,) + mesh + (J[joint],))


def ellipsoid(c, r, colour, seg=18, ring=12, keep=None):
    """an ellipsoid (centre, radii), colour a function of the unit direction (or a colour); keep(d) -> bool leaves
    out the triangles not kept (an opening)"""
    V, N, C, I = [], [], [], []
    cf = colour if callable(colour) else (lambda d: colour)
    for i in range(ring + 1):
        th = math.pi * i / ring
        for j in range(seg + 1):
            ph = 2 * math.pi * j / seg
            d = np.array([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)])
            V.append(np.array(c) + d * np.array(r))
            n = d / np.array(r)
            N.append(n / np.linalg.norm(n))
            C.append(cf(d))
    for i in range(ring):
        for j in range(seg):
            a = i * (seg + 1) + j
            b = a + seg + 1
            for t in ((a, b, a + 1), (a + 1, b, b + 1)):
                if keep is None or keep(np.mean([V[k] - np.array(c) for k in t], axis=0) / np.array(r)):
                    I += list(t)
    return V, N, C, I


def tube(path, radii, colour, sides=12, caps=True):
    P = np.array(path, float)
    n_ = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    nv = np.cross(T[0], ref)
    nv /= np.linalg.norm(nv)
    cf = colour if callable(colour) else (lambda k: colour)
    V, N, C, I = [], [], [], []
    for i in range(n_):
        nv = nv - np.dot(nv, T[i]) * T[i]
        nv /= np.linalg.norm(nv)
        b = np.cross(T[i], nv)
        for s in range(sides + 1):
            a = 2 * math.pi * s / sides
            d = math.cos(a) * nv + math.sin(a) * b
            V.append(P[i] + d * radii[i])
            N.append(d)
            C.append(cf(i))
    for i in range(n_ - 1):
        for s in range(sides):
            a = i * (sides + 1) + s
            c = a + sides + 1
            I += [a, a + 1, c, a + 1, c + 1, c]
    if caps:
        for end, sgn in ((0, -1), (n_ - 1, 1)):
            centre = len(V)
            V.append(P[end])
            N.append(T[end] * sgn)
            C.append(cf(end))
            for s in range(sides):
                a = end * (sides + 1) + s
                I += [centre, a + 1, a] if sgn > 0 else [centre, a, a + 1]
    return V, N, C, I


def box(lo, hi, colour):
    lo, hi = np.array(lo, float), np.array(hi, float)
    V, N, C, I = [], [], [], []
    for axis in range(3):
        for sgn in (-1, 1):
            u, v = [a for a in range(3) if a != axis]
            base = len(V)
            for k in range(4):
                p = np.where(sgn > 0, hi, lo).astype(float).copy()
                p[axis] = hi[axis] if sgn > 0 else lo[axis]
                p[u] = hi[u] if k in (1, 2) else lo[u]
                p[v] = hi[v] if k in (2, 3) else lo[v]
                V.append(p)
                nn = np.zeros(3)
                nn[axis] = sgn
                N.append(nn)
                C.append(colour)
            if (sgn > 0) == ((axis == 1)):
                I += [base, base + 1, base + 2, base, base + 2, base + 3]
            else:
                I += [base, base + 2, base + 1, base, base + 3, base + 2]
    return V, N, C, I


def lathe(profile, centre, colour, sides=14):
    """a pot: profile [(radius, y)] turned about the y axis at centre"""
    path = [(centre[0], centre[1] + y, centre[2]) for r, y in profile]
    return tube(path, [r for r, y in profile], colour, sides=sides)


def patches(base):
    """the cloak's colour: ochre, darker in the folds, teal and red patches stitched on"""
    def f(d):
        h = math.sin(d[0] * 5.1 + d[1] * 3.3) * math.sin(d[2] * 4.7 - d[1] * 2.9)
        if 0.55 < d[2] and -0.55 < d[1] < 0.05 and d[0] < -0.15:
            return TEAL
        if d[0] > 0.35 and -0.2 < d[1] < 0.4 and d[2] > 0.1:
            return RED
        if d[2] < -0.5 and 0.0 < d[1] < 0.5 and d[0] > 0.0:
            return TEAL
        if d[1] < -0.62:
            return OCHRE_D
        k = 0.9 + 0.12 * h
        return tuple(min(1.0, c * k) for c in base)
    return f


# --- the merchant ---
# shoes and legs (on main)
for sx in (1, -1):
    add('merchant-shoe', ellipsoid((0.17 * sx, 0.08, 0.05), (0.12, 0.08, 0.19), BROWN, 12, 8), 'main')
    add('merchant-legs', tube([(0.16 * sx, 0.13, 0.0), (0.155 * sx, 0.3, 0.0), (0.15 * sx, 0.5, 0.0)],
                              [0.04, 0.032, 0.036], BONE, 8), 'main')
    # the bony knees
    add('merchant-legs', ellipsoid((0.155 * sx, 0.3, 0.005), (0.048, 0.045, 0.048), BONE, 8, 6), 'main')

# the body: a big pear, its hem ragged (darker) low down
add('merchant-cloth', ellipsoid((0, 0.98, 0), (0.62, 0.56, 0.55), patches(OCHRE), 24, 16), 'body')
# the underside's shadow, between the legs
add('merchant-cloth', ellipsoid((0, 0.47, 0), (0.42, 0.06, 0.38), OCHRE_D, 16, 6), 'body')
# the ragged hem: teeth round the bottom
for k in range(22):
    a = 2 * math.pi * k / 22
    x, z = math.cos(a), math.sin(a)
    add('merchant-cloth', tube([(0.5 * x, 0.6, 0.47 * z), (0.46 * x, 0.46, 0.43 * z)], [0.07, 0.01],
                               OCHRE_D if k % 3 else OCHRE, 5), 'body')
# rope belt round the waist
ring = [(0.63 * math.cos(2 * math.pi * k / 32), 0.95 + 0.02 * math.sin(4 * math.pi * k / 32),
         0.56 * math.sin(2 * math.pi * k / 32)) for k in range(33)]
add('merchant-belt', tube(ring, [0.035] * 33, ROPE, 8, caps=False), 'body')
# pouch (front left), sack (front middle), blue bottle in a net (front right)
add('merchant-pouch', box((0.22, 0.72, 0.42), (0.44, 0.94, 0.56), BROWN), 'body')
add('merchant-pouch', box((0.24, 0.86, 0.555), (0.42, 0.95, 0.575), (0.28, 0.15, 0.07)), 'body')
add('merchant-cloth', ellipsoid((-0.02, 0.78, 0.55), (0.1, 0.12, 0.08), CREAM, 10, 8), 'body')
add('merchant-cloth', tube([(-0.02, 0.88, 0.55), (-0.02, 0.95, 0.56)], [0.03, 0.03], ROPE, 6), 'body')
add('merchant-bottle', ellipsoid((-0.38, 0.8, 0.45), (0.12, 0.13, 0.11), BLUE, 12, 8), 'body')
add('merchant-bottle', tube([(-0.38, 0.92, 0.45), (-0.38, 1.0, 0.45)], [0.04, 0.035], ROPE, 6), 'body')

# the scarf: thick round the neck, its ends hanging down his front left
neck = [(0.34 * math.cos(2 * math.pi * k / 24), 1.36, 0.32 * math.sin(2 * math.pi * k / 24) + 0.02) for k in range(25)]
add('merchant-scarf', tube(neck, [0.11] * 25, RED, 10, caps=False), 'body')
add('merchant-scarf', tube([(0.2, 1.32, 0.33), (0.3, 1.1, 0.5), (0.34, 0.85, 0.56)], [0.08, 0.075, 0.07],
                           lambda k: RED if k < 2 else (0.62, 0.16, 0.1), 8), 'body')

# sleeves and gloved hands (on the arms): out from the shoulders, the hands open at his sides
for sx, arm in ((1, 'arm_l'), (-1, 'arm_r')):
    add('merchant-sleeve', tube([(0.52 * sx, 1.2, 0.05), (0.68 * sx, 1.08, 0.1), (0.8 * sx, 0.95, 0.16)],
                                [0.13, 0.12, 0.13], lambda k: OCHRE if k < 2 else OCHRE_D, 10), arm)
    add('merchant-glove', ellipsoid((0.86 * sx, 0.9, 0.2), (0.08, 0.07, 0.09), GLOVE, 10, 8), arm)
    for f in range(3):
        a = (f - 1) * 0.45
        tip = (0.86 * sx + 0.1 * sx * math.cos(a), 0.86, 0.2 + 0.12 * math.sin(a) + 0.04)
        add('merchant-glove', tube([(0.86 * sx, 0.88, 0.22), tip], [0.03, 0.022], GLOVE, 6), arm)
    add('merchant-glove', tube([(0.86 * sx, 0.92, 0.25), (0.84 * sx, 0.98, 0.32)], [0.03, 0.022], GLOVE, 6), arm)

# the hood: a round cowl open at the front, its tip up and back, a red patch on top; inside, all shadow
add('merchant-hood', ellipsoid((0, 1.66, 0.0), (0.4, 0.38, 0.4),
                               lambda d: RED if d[1] > 0.75 and d[0] > -0.2 else patches(OCHRE)(d), 20, 14,
                               keep=lambda d: not (d[2] > 0.55 and -0.75 < d[1] < 0.55 and abs(d[0]) < 0.72)), 'head')
add('merchant-hood', tube([(0, 1.98, -0.08), (0.02, 2.12, -0.2), (0.06, 2.18, -0.34)], [0.16, 0.08, 0.02],
                          lambda k: RED, 10), 'head')
# the rim round the opening, a little thicker
rim = []
for k in range(29):
    a = 2 * math.pi * k / 28
    rim.append((0.29 * math.sin(a), 1.63 + 0.25 * math.cos(a), 0.3 + 0.03 * math.cos(a)))
add('merchant-hood', tube(rim, [0.045] * 29, OCHRE_D, 8, caps=False), 'head')
# the shadow inside: a dark face filling the opening
add('merchant-shadow', ellipsoid((0, 1.62, 0.22), (0.3, 0.27, 0.16), BLACK, 16, 10), 'head')

# the eyes: white, a brown pupil, a dark lid (the lids on their own joints, open: rolled up out of sight)
for sx, lid in ((1, 'lid_l'), (-1, 'lid_r')):
    c = (0.12 * sx, 1.6, 0.36)
    add('merchant-white', ellipsoid(c, (0.085, 0.09, 0.05), WHITE, 14, 10), 'head')
    add('merchant-pupil', ellipsoid((0.12 * sx - 0.01 * sx, 1.595, 0.402), (0.038, 0.045, 0.012), PUPIL, 10, 6),
        'head')
    add('merchant-pupil', ellipsoid((0.12 * sx - 0.005 * sx, 1.61, 0.413), (0.012, 0.012, 0.004), WHITE, 6, 4),
        'head')
    # the lid: the upper half of a shell just over the eye, closing down over it when rolled forward
    add('merchant-lid', ellipsoid(c, (0.092, 0.097, 0.058), LID, 14, 10, keep=lambda d: d[1] > 0.25), lid)

# the ears: long, flat, pointed, out sideways and a little up and back
for sx, ear in ((1, 'ear_l'), (-1, 'ear_r')):
    path = [(0.3 * sx, 1.62, 0.02), (0.5 * sx, 1.68, -0.02), (0.7 * sx, 1.76, -0.06), (0.86 * sx, 1.84, -0.1)]
    V, N, C, I = tube(path, [0.1, 0.075, 0.045, 0.006], lambda k: SKIN if k else (0.86, 0.55, 0.42), 10)
    V = [np.array(v) * np.array([1, 1, 1]) for v in V]
    # flattened front to back
    base = np.array(path[0])
    V = [base + (v - base) * np.array([1.0, 1.0, 0.45]) for v in V]
    add('merchant-skin', (V, N, C, I), ear)

# --- the stall (on main): behind him ---
SZ = -1.05  # the counter's middle
for x in (-1.15, 1.15):
    for z in (SZ - 0.45, SZ + 0.45):
        h = 2.3 if z > SZ else 2.55
        add('merchant-wood', tube([(x, 0, z), (x + 0.03, h * 0.5, z), (x - 0.02, h, z)], [0.08, 0.075, 0.07],
                                  lambda k: WOOD if k % 2 else WOOD_D, 8), 'main')
        # rope lashings
        for y in (0.85, h - 0.42):
            add('merchant-rope', tube([(x + 0.09 * math.cos(a), y + 0.02 * a, z + 0.09 * math.sin(a))
                                       for a in np.linspace(0, 4 * math.pi, 20)], [0.02] * 20, ROPE, 5,
                                      caps=False), 'main')
# the counter: planks, a front of boards, a cloth hanging over it
add('merchant-wood', box((-1.2, 0.85, SZ - 0.45), (1.2, 0.95, SZ + 0.45), WOOD), 'main')
for k in range(6):
    x0 = -1.15 + k * 0.385
    add('merchant-wood', box((x0, 0.05, SZ + 0.38), (x0 + 0.37, 0.85, SZ + 0.44), WOOD if k % 2 else WOOD_D), 'main')
add('merchant-awning', box((-0.35, 0.5, SZ + 0.45), (0.35, 0.96, SZ + 0.47), RED), 'main')
for k in range(4):
    x0 = -0.3 + k * 0.17
    add('merchant-awning', box((x0, 0.52, SZ + 0.47), (x0 + 0.1, 0.6, SZ + 0.48), CREAM), 'main')
# the awning: stripes red and cream, sloping down to the front, ragged teeth at its edge
for k in range(8):
    x0 = -1.3 + k * 0.325
    col = RED if k % 2 == 0 else CREAM
    V, N, C, I = box((x0, 0, -0.62), (x0 + 0.325, 0.04, 0.62), col)
    rot = math.radians(14)
    V = [np.array([v[0], v[1] * math.cos(rot) - v[2] * math.sin(rot) + 2.42,
                   v[1] * math.sin(rot) + v[2] * math.cos(rot) + SZ]) for v in V]
    add('merchant-awning', (V, N, C, I), 'main')
    add('merchant-awning', tube([(x0 + 0.16, 2.27, SZ + 0.6), (x0 + 0.16, 2.13, SZ + 0.6)], [0.12, 0.01], col, 4),
        'main')
# on the counter: pots, a crate, a basket of fruit
add('merchant-pot', lathe([(0.12, 0), (0.2, 0.12), (0.18, 0.26), (0.09, 0.34), (0.11, 0.38)], (-0.8, 0.95, SZ),
                          lambda k: CLAY if k != 2 else CREAM), 'main')
add('merchant-pot', lathe([(0.08, 0), (0.13, 0.08), (0.12, 0.18), (0.06, 0.24), (0.07, 0.27)], (-0.45, 0.95, SZ - 0.1),
                          lambda k: (0.9, 0.5, 0.15)), 'main')
add('merchant-pot', lathe([(0.1, 0), (0.15, 0.1), (0.13, 0.24), (0.07, 0.3), (0.08, 0.33)], (-0.1, 0.95, SZ - 0.15),
                          lambda k: GREENPOT if k != 2 else CREAM), 'main')
add('merchant-wood', box((0.25, 0.95, SZ - 0.3), (0.6, 1.25, SZ + 0.05), WOOD_D), 'main')
add('merchant-wood', box((0.27, 1.25, SZ - 0.28), (0.58, 1.27, SZ + 0.03), WOOD), 'main')
add('merchant-rope', lathe([(0.15, 0), (0.22, 0.12), (0.23, 0.16)], (0.85, 0.95, SZ), ROPE), 'main')
for k in range(5):
    a = 2 * math.pi * k / 5
    add('merchant-fruit', ellipsoid((0.85 + 0.1 * math.cos(a), 1.13, SZ + 0.1 * math.sin(a)), (0.07, 0.07, 0.07),
                                    FRUIT, 8, 6), 'main')
# crates stacked at its side
add('merchant-wood', box((1.3, 0, SZ - 0.3), (1.8, 0.5, SZ + 0.2), WOOD), 'main')
add('merchant-wood', box((1.35, 0.5, SZ - 0.25), (1.75, 0.9, SZ + 0.15), WOOD_D), 'main')
# the sign: an arm out from the front right post, a rope, a net round a glowing white eco orb
add('merchant-wood', tube([(1.12, 2.1, SZ + 0.45), (1.5, 2.2, SZ + 0.6), (1.85, 2.25, SZ + 0.7)], [0.06, 0.055, 0.05],
                          WOOD, 8), 'main')
add('merchant-rope', tube([(1.82, 2.25, SZ + 0.7), (1.82, 1.82, SZ + 0.7)], [0.015, 0.015], ROPE, 5), 'main')
add('merchant-orb', ellipsoid((1.82, 1.6, SZ + 0.7), (0.2, 0.2, 0.2), GLOW, 14, 10), 'main')
for k in range(6):
    a = math.pi * k / 6
    add('merchant-rope', tube([(1.82 + 0.215 * math.cos(a) * math.sin(t), 1.6 + 0.215 * math.cos(t),
                                SZ + 0.7 + 0.215 * math.sin(a) * math.sin(t)) for t in np.linspace(0, math.pi, 10)],
                              [0.012] * 10, ROPE, 4, caps=False), 'main')

# --- the idle animation: 4 s, keys every 1/30 s ---
FPS = 30
T = 4.0
NF = int(T * FPS) + 1
times = np.arange(NF) / FPS


def qx(a):
    return (math.sin(a / 2), 0, 0, math.cos(a / 2))


def qz(a):
    return (0, 0, math.sin(a / 2), math.cos(a / 2))


def qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def blink(t):
    """0 open, 1 shut: two blinks, quick"""
    out = 0.0
    for b in (1.3, 3.1, 3.35):
        out = max(out, max(0.0, 1 - abs(t - b) / 0.09))
    return out


anim = {}  # joint -> (translations, rotations)
for i, (name, parent, pos) in enumerate(JOINTS):
    local = np.array(pos) - (np.array(JOINTS[parent][2]) if parent >= 0 else 0)
    tr, ro = [], []
    for t in times:
        w = 2 * math.pi * t / T
        p = local.copy()
        q = (0, 0, 0, 1)
        if name == 'body':
            p = p + np.array([0, 0.015 * math.sin(2 * w), 0])
            q = qz(0.025 * math.sin(w))
        elif name == 'head':
            q = qmul(qz(-0.04 * math.sin(w + 0.6)), qx(0.04 * math.sin(2 * w)))
        elif name.startswith('lid'):
            # open: rolled up and back (out of sight in the hood); shut: down over the eye
            q = qx(-1.25 + 1.55 * blink(t))
        elif name.startswith('ear'):
            s = 1 if name == 'ear_l' else -1
            tw = max(0.0, 1 - abs(t - 2.2) / 0.15) * (1 if name == 'ear_l' else 0.4)
            q = qz(s * (0.05 * math.sin(w) + 0.25 * tw))
        elif name.startswith('arm'):
            s = 1 if name == 'arm_l' else -1
            q = qz(s * 0.08 * math.sin(w + (0 if s > 0 else math.pi)))
        tr.append(p)
        ro.append(q)
    anim[i] = (np.array(tr, np.float32), np.array(ro, np.float32))


def png_white():
    bio = io.BytesIO()
    Image.new('RGB', (4, 4), (255, 255, 255)).save(bio, 'PNG')
    return bio.getvalue()


def write_glb(path, name):
    data = bytearray()
    views, accs = [], []

    def view(blob, target=None):
        while len(data) % 4:
            data.append(0)
        v = {'buffer': 0, 'byteOffset': len(data), 'byteLength': len(blob)}
        if target:
            v['target'] = target
        views.append(v)
        data.extend(blob)
        return len(views) - 1

    def acc(arr, ctype, typ, target=None, minmax=False):
        a = {'bufferView': view(arr.tobytes(), target), 'componentType': ctype, 'count': len(arr), 'type': typ}
        if minmax:
            a['min'] = arr.min(0).tolist()
            a['max'] = arr.max(0).tolist()
        accs.append(a)
        return len(accs) - 1

    # merged by material
    groups = {}
    for mat, V, N, C, I, j in parts:
        g = groups.setdefault(mat, [[], [], [], [], []])
        o = len(g[0])
        g[0] += [np.array(v, float) for v in V]
        g[1] += [np.array(n, float) for n in N]
        g[2] += list(C)
        g[3] += [k + o for k in I]
        g[4] += [j] * len(V)
    mats, texs, imgs, prims = [], [], [], []
    for k, (mat, (V, N, C, I, Jn)) in enumerate(groups.items()):
        V = np.array(V, np.float32)
        N = np.array(N, np.float32)
        N /= np.maximum(np.linalg.norm(N, axis=1), 1e-6)[:, None]
        C4 = np.ones((len(V), 4), np.float32)
        C4[:, :3] = np.array(C, np.float32) * 0.5
        if mat == 'merchant-orb':
            C4[:, :3] = 1.0  # (full strength: it glows)
        Jn4 = np.zeros((len(V), 4), np.uint8)
        Jn4[:, 0] = Jn
        W4 = np.zeros((len(V), 4), np.float32)
        W4[:, 0] = 1
        a_pos = acc(V, 5126, 'VEC3', 34962, True)
        a_nrm = acc(N, 5126, 'VEC3', 34962)
        a_uv = acc(np.zeros((len(V), 2), np.float32), 5126, 'VEC2', 34962)
        a_col = acc(C4, 5126, 'VEC4', 34962)
        a_j = acc(Jn4, 5121, 'VEC4', 34962)
        a_w = acc(W4, 5126, 'VEC4', 34962)
        a_i = acc(np.array(I, np.uint32), 5125, 'SCALAR', 34963)
        imgs.append({'bufferView': view(png_white()), 'mimeType': 'image/png', 'name': mat})
        texs.append({'source': k, 'sampler': 0})
        mats.append({'name': mat, 'pbrMetallicRoughness': {'baseColorTexture': {'index': k}, 'metallicFactor': 0.0,
                                                           'roughnessFactor': 0.7}})
        prims.append({'attributes': {'POSITION': a_pos, 'NORMAL': a_nrm, 'TEXCOORD_0': a_uv, 'COLOR_0': a_col,
                                     'JOINTS_0': a_j, 'WEIGHTS_0': a_w}, 'indices': a_i, 'material': k})
    # nodes: 0 the mesh, 1.. the joints
    nodes = [{'name': name, 'mesh': 0, 'skin': 0}]
    for i, (jn, parent, pos) in enumerate(JOINTS):
        local = np.array(pos) - (np.array(JOINTS[parent][2]) if parent >= 0 else 0)
        nodes.append({'name': jn, 'translation': local.tolist(), 'children': []})
    for i, (jn, parent, pos) in enumerate(JOINTS):
        if parent >= 0:
            nodes[1 + parent]['children'].append(1 + i)
    for n_ in nodes:
        if 'children' in n_ and not n_['children']:
            del n_['children']
    ibm = np.zeros((len(JOINTS), 4, 4), np.float32)
    for i, (_, _, pos) in enumerate(JOINTS):
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = -np.array(pos)
        ibm[i] = m.T  # (column-major)
    a_ibm = acc(ibm.reshape(-1, 16), 5126, 'MAT4')
    skin = {'joints': list(range(1, 1 + len(JOINTS))), 'inverseBindMatrices': a_ibm, 'skeleton': 1}
    a_t = acc(times.astype(np.float32), 5126, 'SCALAR', None, True)
    samplers, channels = [], []
    for i in range(len(JOINTS)):
        tr, ro = anim[i]
        samplers.append({'input': a_t, 'output': acc(tr, 5126, 'VEC3'), 'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1, 'target': {'node': 1 + i, 'path': 'translation'}})
        samplers.append({'input': a_t, 'output': acc(ro, 5126, 'VEC4'), 'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1, 'target': {'node': 1 + i, 'path': 'rotation'}})
    while len(data) % 4:
        data.append(0)
    g = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0, 1]}], 'nodes': nodes,
         'meshes': [{'name': name, 'primitives': prims}], 'skins': [skin],
         'animations': [{'name': 'idle', 'samplers': samplers, 'channels': channels}],
         'materials': mats, 'textures': texs, 'samplers': [{'magFilter': 9729, 'minFilter': 9987}], 'images': imgs,
         'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(g).encode()
    while len(js) % 4:
        js += b' '
    out = (b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) + struct.pack('<I4s', len(js), b'JSON') +
           js + struct.pack('<I4s', len(data), b'BIN\0') + bytes(data))
    open(path, 'wb').write(out)
    return sum(len(p[0]) for p in groups.values()), sum(len(p[3]) for p in groups.values()) // 3


# (the same model twice: custom_levels for build-actor, common/<name>-lod0 so the extractor puts it in GAME)
write_glb(os.path.join(OUT, 'custom_levels', 'merchant.glb'), 'merchant')
nv, nt = write_glb(os.path.join(OUT, 'common', 'merchant-lod0.glb'), 'merchant')
print('ok', nv, 'vertices', nt, 'triangles')
