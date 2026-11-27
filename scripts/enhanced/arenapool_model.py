# Jak Living Sky: the mud pools of the merchant's first arena fight ("FUOCO INCROCIATO", misty-obs.gc), one under
# each quicksand lurker: a copy of a real one of Misty's, the "lonely island" (mud-11, its quicksand lurker in the
# middle), at its own size: the ring of rock round it (Misty's background, cut out 14 m round, its own textures and
# colours) and its mud (the water-anim's mesh, turned as the level turns it). Its origin the mud's middle, on the mud.
# Seen only (opaque); its collision a plain ring of the rock's shape (inner and outer edges and height taken from the
# rock, 32 slices round), in an invisible node of its own. A glb for build-actor (game.gp, :gen-mesh #t) and the same
# as common/<name>-lod0 for the extractor (GAME.fr3).
#
#   python scripts/enhanced/arenapool_model.py custom_assets/jak1/models [misty's decompiled level folder]
import json
import math
import os
import struct
import sys

import numpy as np

OUT = sys.argv[1]
MISTY = (sys.argv[2] if len(sys.argv) > 2 else
         'C:/Users/aless/features/jak1/mods/Jak Living Sky/jak-living-sky/data/decompiler_out/jak1/levels/misty/')
NAME = 'arenapool'
CX, CY, CZ = -114.8, 16.4, 881.8  # mud-11's place (its mud's height)
ROT = -7281.78 / 65536 * 2 * math.pi  # its rotoffset
RAD = 14.0  # the rock cut out round it
SLICES = 32
# (its colours brighter than Misty's own: a model's lit by the arena's lights as well, Misty's ground has its light
# in its colours alone; as they were it came out near black)
BRIGHT = 1.8


def read_glb(path):
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[12:16])[0]
    return json.loads(b[20:20 + n]), b[20 + n + 8:]


def accessor(g, bin_, i):
    a = g['accessors'][i]
    v = g['bufferViews'][a['bufferView']]
    n = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1, 'VEC4': 4}[a['type']]
    dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a['componentType']]
    arr = np.frombuffer(bin_, dt, count=a['count'] * n, offset=v.get('byteOffset', 0) + a.get('byteOffset', 0))
    return arr.reshape(-1, n) if n > 1 else arr


parts = {}  # material name -> (positions, uvs, colours) of its triangles' corners
images = {}  # material name -> its picture's uri


def add_part(mat, P, UV, C, uri):
    p = parts.setdefault(mat, [[], [], []])
    p[0].append(P)
    p[1].append(UV)
    p[2].append(C)
    images.setdefault(mat, uri)


# --- the rock: Misty's background round the pool --------------------------------------------------------------
g, bin_ = read_glb(MISTY + 'misty-background.glb')
for nd in g['nodes']:
    if 'mesh' not in nd:
        continue
    T = np.array(nd.get('translation', [0, 0, 0]), np.float32)
    for pr in g['meshes'][nd['mesh']]['primitives']:
        if 'indices' not in pr or 'material' not in pr:
            continue
        P = accessor(g, bin_, pr['attributes']['POSITION']) + T
        I = accessor(g, bin_, pr['indices']).reshape(-1, 3)
        c = P[I].mean(1)
        # (only the rock over the mud's level, or about it: under it, in Misty in the mud, in the arena it hung in
        # shards and lay on the sand; a skirt of its own instead, below)
        V3 = P[I]
        nrm = np.cross(V3[:, 1] - V3[:, 0], V3[:, 2] - V3[:, 0])
        flat = np.abs(nrm[:, 1]) > 0.7 * np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)
        keep = ((np.hypot(c[:, 0] - CX, c[:, 2] - CZ) < RAD) & (c[:, 1] > CY - 6) & (c[:, 1] < CY + 12) &
                (V3[:, :, 1].max(1) > CY - 0.2))
        if not keep.any():
            continue
        I = I[keep]
        mat = g['materials'][pr['material']]
        tex = g['textures'][mat['pbrMetallicRoughness']['baseColorTexture']['index']]
        uri = g['images'][tex['source']]['uri']
        UV = accessor(g, bin_, pr['attributes']['TEXCOORD_0'])
        C = (accessor(g, bin_, pr['attributes']['COLOR_0']) if 'COLOR_0' in pr['attributes']
             else np.full((len(P), 4), 0.5, np.float32)).copy()
        C[:, :3] = np.clip(C[:, :3] * BRIGHT, 0, 1)
        add_part(mat.get('name', 'rock'), (P[I] - np.array([CX, CY, CZ])).reshape(-1, 3), UV[I].reshape(-1, 2),
                 C[I].reshape(-1, C.shape[1]), uri)
rock = np.concatenate([np.concatenate(v[0]) for k, v in parts.items()])

# --- the bank: the island's side, a slope of Misty's ground from just under its rim (at its outermost rock, all
# round) out 1.8 m and down into the arena's sand: no gap under the rock, nothing hanging outside it, no wall
SKSEG = 96
BANK_OUT, BANK_BOTTOM = 1.8, -1.3
ang_r = np.arctan2(rock[:, 2], rock[:, 0])
rad_r = np.hypot(rock[:, 0], rock[:, 2])
edge, rim = [], []
for k in range(SKSEG):
    a0 = -math.pi + 2 * math.pi * k / SKSEG
    m = np.abs((ang_r - a0 + math.pi) % (2 * math.pi) - math.pi) <= math.pi / SKSEG * 1.5
    if m.any():
        mt_ = m & (rock[:, 1] > 0.1)  # (the top's outline: the sides go far down, under the sand)
        e_ = float(rad_r[mt_].max()) if mt_.any() else float(rad_r[m].max())
        near = m & (rad_r > e_ - 1.0)
        edge.append(e_)
        rim.append(float(np.percentile(rock[near, 1], 20)) if near.any() else 0.0)
    else:
        edge.append(None)
        rim.append(None)
for lst in (edge, rim):
    known = [v for v in lst if v is not None]
    for k in range(SKSEG):
        if lst[k] is None:
            lst[k] = sum(known) / len(known)
# (at the rock's outermost, smoothed; its top just under the rim there)
edge = [0.97 * max(edge[(k + j) % SKSEG] for j in range(-1, 2)) for k in range(SKSEG)]
edge = [sum(edge[(k + j) % SKSEG] for j in range(-2, 3)) / 5.0 for k in range(SKSEG)]  # (a smooth outline, no spikes)
rim = [0.0] * SKSEG  # (at the mud's level, under the rock's top: the island's sides go far down, not a rim)
ground_uri = None
for m_ in g['materials']:
    if m_.get('name') == 'mis-ground':
        ground_uri = g['images'][g['textures'][m_['pbrMetallicRoughness']['baseColorTexture']['index']]['source']]['uri']
        break
bank_c = np.concatenate(parts['mis-ground'][2]).mean(0)
SP, SUV = [], []
for k in range(SKSEG):
    a0 = -math.pi + 2 * math.pi * k / SKSEG
    a1 = -math.pi + 2 * math.pi * (k + 1) / SKSEG
    k1 = (k + 1) % SKSEG
    t0 = [edge[k] * math.cos(a0), rim[k], edge[k] * math.sin(a0)]
    t1 = [edge[k1] * math.cos(a1), rim[k1], edge[k1] * math.sin(a1)]
    b0 = [(edge[k] + BANK_OUT) * math.cos(a0), BANK_BOTTOM, (edge[k] + BANK_OUT) * math.sin(a0)]
    b1 = [(edge[k1] + BANK_OUT) * math.cos(a1), BANK_BOTTOM, (edge[k1] + BANK_OUT) * math.sin(a1)]
    for q in (t0, b0, t1, t1, b0, b1):
        SP.append(q)
        SUV.append([q[0] / 4.0, q[2] / 4.0])  # (the ground's picture laid flat, as Misty's)
SP = np.array(SP, np.float32)
BANK = False  # (the bank left out: the rock's own sides go down under the sand)
if BANK:
    add_part('mis-ground', SP, np.array(SUV, np.float32), np.tile(bank_c, (len(SP), 1)).astype(np.float32), ground_uri)

# --- the collision: a ring of the rock's shape --------------------------------------------------------------------
top = rock[rock[:, 1] > 0.3]  # (the rock over the mud)
ang = np.arctan2(top[:, 2], top[:, 0])
rad = np.hypot(top[:, 0], top[:, 2])
inner, outer, height = [], [], []
for k in range(SLICES):
    a0 = -math.pi + 2 * math.pi * k / SLICES
    m = np.abs((ang - a0 + math.pi) % (2 * math.pi) - math.pi) <= math.pi / SLICES * 1.5
    if m.any():
        inner.append(float(rad[m].min()))
        outer.append(float(rad[m].max()))
        height.append(float(np.percentile(top[m, 1], 80)))
    else:
        inner.append(None)
        outer.append(None)
        height.append(None)
for lst in (inner, outer, height):  # (gaps: from their neighbours)
    known = [v for v in lst if v is not None]
    for k in range(SLICES):
        if lst[k] is None:
            lst[k] = sum(known) / len(known)
# --- the sides: every outer open edge of the rock carried on down, in its own picture and colours, out a little and
# into the arena's sand (Misty's rock under the mud's level is left out above: there it'd have shown as holes)
SIDE_BOTTOM, SIDE_OUT = -2.5, 0.4
for mname in [k_ for k_ in parts if not k_.startswith('arenapool-') and 'grass' not in k_]:  # (not the grass tufts)
    P_ = np.concatenate(parts[mname][0])
    U_ = np.concatenate(parts[mname][1])
    C_ = np.concatenate(parts[mname][2])
    key = [tuple(np.round(v, 2)) for v in P_]
    cnt = {}
    for t in range(0, len(P_), 3):
        for i0, i1 in ((t, t + 1), (t + 1, t + 2), (t + 2, t)):
            e = tuple(sorted((key[i0], key[i1])))
            cnt.setdefault(e, []).append((i0, i1))
    allk = {}
    for m2 in parts:  # (an edge shared with another material's triangle isn't open)
        if m2 == mname or m2.startswith('arenapool-') or 'grass' in m2:
            continue
        Q_ = np.concatenate(parts[m2][0])
        kq = [tuple(np.round(v, 2)) for v in Q_]
        for t in range(0, len(Q_), 3):
            for i0, i1 in ((t, t + 1), (t + 1, t + 2), (t + 2, t)):
                allk[tuple(sorted((kq[i0], kq[i1])))] = 1
    SP, SU, SC = [], [], []
    for e, lst in cnt.items():
        if len(lst) != 1 or e in allk:
            continue
        i0, i1 = lst[0]
        v0, v1 = P_[i0], P_[i1]
        if max(v0[1], v1[1]) < -0.9:
            continue
        mid = (v0 + v1) / 2
        am = math.atan2(mid[2], mid[0])
        f = (am + math.pi) / (2 * math.pi) * SLICES
        k0 = int(f) % SLICES
        w = f - int(f)
        i_ = inner[k0] * (1 - w) + inner[(k0 + 1) % SLICES] * w
        o_ = outer[k0] * (1 - w) + outer[(k0 + 1) % SLICES] * w
        if math.hypot(mid[0], mid[2]) < 0.5 * (i_ + o_):
            continue  # (the hole's edge, under the mud)
        ln = float(np.linalg.norm(v1 - v0))
        if ln < 1e-4:
            continue
        sc = float(np.linalg.norm(U_[i1] - U_[i0])) / ln
        du = U_[i1] - U_[i0]
        nu = np.array([-du[1], du[0]]) / max(float(np.linalg.norm(du)), 1e-9)
        bs = []
        for v, u_, c_ in ((v0, U_[i0], C_[i0]), (v1, U_[i1], C_[i1])):
            d = np.array([v[0], 0, v[2]]) / max(math.hypot(v[0], v[2]), 1e-6)
            h = float(v[1]) - SIDE_BOTTOM
            bs.append((np.array([v[0] + d[0] * SIDE_OUT, SIDE_BOTTOM, v[2] + d[2] * SIDE_OUT]), u_ + nu * h * sc, c_))
        t0, t1 = (v0, U_[i0], C_[i0]), (v1, U_[i1], C_[i1])
        for q in (t0, bs[0], t1, t1, bs[0], bs[1]):
            SP.append(q[0]); SU.append(q[1]); SC.append(q[2])
    if SP:
        add_part(mname, np.array(SP, np.float32), np.array(SU, np.float32), np.array(SC, np.float32), images[mname])
        print('sides', mname, len(SP) // 6)

# --- the mud: its own surface (texture, colours, the scale of its picture: Misty's, as the level turns it) on a shape of
# the ring's: out to the rock's middle all round (under it), so the hole is covered and none of it shows outside
g, bin_ = read_glb(MISTY + 'water-anim-misty-mud-lonely-island-lod0.glb')
ca, sa = math.cos(ROT), math.sin(ROT)
pr = g['meshes'][0]['primitives'][0]
P = accessor(g, bin_, pr['attributes']['POSITION'])
I = accessor(g, bin_, pr['indices']).reshape(-1)
Q = P[I].copy()
x, z = Q[:, 0].copy(), Q[:, 2].copy()
Q[:, 0] = x * ca + z * sa
Q[:, 2] = -x * sa + z * ca
UVo = accessor(g, bin_, pr['attributes']['TEXCOORD_0'])[I]
Co = accessor(g, bin_, pr['attributes']['COLOR_0'])[I]
# (its picture's mapping, u and v from x and z, as the level's)
A = np.stack([Q[:, 0], Q[:, 2], np.ones(len(Q))], 1)
uv_fit, *_ = np.linalg.lstsq(A, UVo, rcond=None)
mat = g['materials'][pr['material']]
tex = g['textures'][mat['pbrMetallicRoughness']['baseColorTexture']['index']]
MUDSEG = 96
ring = []
for k in range(MUDSEG):
    t = 2 * math.pi * k / MUDSEG  # (from -pi as the slices)
    f = t / (2 * math.pi) * SLICES
    k0 = int(f) % SLICES
    k1 = (k0 + 1) % SLICES
    w = f - int(f)
    i_ = inner[k0] * (1 - w) + inner[k1] * w
    o_ = outer[k0] * (1 - w) + outer[k1] * w
    r_ = i_ + 0.45 * (o_ - i_)
    a_ = -math.pi + t
    ring.append([r_ * math.cos(a_), -0.3, r_ * math.sin(a_)])  # (under the real mud: its floor, mud-coloured)
MP = []
for k in range(MUDSEG):
    MP += [[0.0, -0.3, 0.0], ring[(k + 1) % MUDSEG], ring[k]]
MP = np.array(MP, np.float32)
MUV = (np.stack([MP[:, 0], MP[:, 2], np.ones(len(MP))], 1) @ uv_fit).astype(np.float32)
MC = np.tile(Co.mean(0), (len(MP), 1)).astype(np.float32)
add_part('arenapool-' + mat.get('name', 'mud'), MP, MUV, MC, g['images'][tex['source']]['uri'])

CV, CI = [], []
for k in range(SLICES):
    a0 = -math.pi + 2 * math.pi * k / SLICES
    c_, s_ = math.cos(a0), math.sin(a0)
    for r_, y_ in ((inner[k], -0.6), (inner[k], height[k]), (outer[k], height[k]), (outer[k] + 0.6, -0.6)):
        CV.append([r_ * c_, y_, r_ * s_])
for k in range(SLICES):
    a, b = 4 * k, 4 * ((k + 1) % SLICES)
    for j in range(3):  # inner wall, top, outer slope (both faces)
        q = [a + j, b + j, b + j + 1, a + j + 1]
        CI += [q[0], q[1], q[2], q[0], q[2], q[3], q[0], q[2], q[1], q[0], q[3], q[2]]
CV = np.array(CV, np.float32)
CI = np.array(CI, np.uint32)


def write_glb(path):
    data = bytearray()
    views, accs, mats, texs, imgs, prims = [], [], [], [], [], []

    def add(blob, target=None):
        while len(data) % 4:
            data.append(0)
        vw = {'buffer': 0, 'byteOffset': len(data), 'byteLength': len(blob)}
        if target:
            vw['target'] = target
        views.append(vw)
        data.extend(blob)
        return len(views) - 1

    def acc(arr, typ, ct, target, mm=False):
        a = {'bufferView': add(arr.tobytes(), target), 'componentType': ct, 'count': len(arr), 'type': typ}
        if mm:
            a['min'] = arr.min(0).tolist()
            a['max'] = arr.max(0).tolist()
        accs.append(a)
        return len(accs) - 1

    for k, (mat, (Ps, UVs, Cs)) in enumerate(parts.items()):
        P = np.concatenate(Ps).astype(np.float32)
        UV = np.concatenate(UVs).astype(np.float32)
        C = np.concatenate(Cs).astype(np.float32)
        if C.shape[1] == 3:
            C = np.concatenate([C, np.ones((len(C), 1), np.float32)], 1)
        e1, e2 = P[1::3] - P[0::3], P[2::3] - P[0::3]
        n = np.cross(e1, e2)
        n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-9)
        N = np.repeat(n, 3, axis=0).astype(np.float32)
        I = np.arange(len(P), dtype=np.uint32)
        imgs.append({'uri': images[mat], 'name': mat})
        texs.append({'source': k, 'sampler': 0})
        # (as Misty's own: alpha-tested, not blended)
        mats.append({'name': mat, 'alphaMode': 'MASK', 'alphaCutoff': 0.149, 'doubleSided': True,
                     'pbrMetallicRoughness': {'baseColorTexture': {'index': k}, 'metallicFactor': 0.0,
                                              'roughnessFactor': 0.9}})
        J4 = np.zeros((len(P), 4), np.uint8)
        J4[:, 0] = 2  # (all on main)
        W4 = np.zeros((len(P), 4), np.float32)
        W4[:, 0] = 1.0
        prims.append({'attributes': {'POSITION': acc(P, 'VEC3', 5126, 34962, True), 'NORMAL': acc(N, 'VEC3', 5126, 34962),
                                     'TEXCOORD_0': acc(UV, 'VEC2', 5126, 34962), 'COLOR_0': acc(C, 'VEC4', 5126, 34962),
                                     'JOINTS_0': acc(J4, 'VEC4', 5121, 34962), 'WEIGHTS_0': acc(W4, 'VEC4', 5126, 34962)},
                      'indices': acc(I, 'SCALAR', 5125, 34963), 'material': k})
    # the collision ring: a node of its own, not drawn
    CN = np.tile(np.array([0, 1, 0], np.float32), (len(CV), 1))
    col_prim = {'attributes': {'POSITION': acc(CV, 'VEC3', 5126, 34962, True), 'NORMAL': acc(CN, 'VEC3', 5126, 34962)},
                'indices': acc(CI, 'SCALAR', 5125, 34963)}
    # a skeleton of align, prejoint and main and a still "idle", as prop_import.py's: without align and prejoint the
    # extractor (GAME.fr3) and build-actor number the joints one apart, and the model is drawn on another's joints
    # (Jak's)
    ibm = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    a_ibm = acc(ibm.reshape(-1, 16), 'MAT4', 5126, None)
    a_t = acc(np.array([0.0, 1.0], np.float32), 'SCALAR', 5126, None, True)
    a_samplers, a_channels = [], []
    for k in range(3):
        a_samplers.append({'input': a_t, 'output': acc(np.zeros((2, 3), np.float32), 'VEC3', 5126, None),
                           'interpolation': 'LINEAR'})
        a_channels.append({'sampler': len(a_samplers) - 1, 'target': {'node': 1 + k, 'path': 'translation'}})
        a_samplers.append({'input': a_t, 'output': acc(np.tile(np.array([0, 0, 0, 1.0], np.float32), (2, 1)), 'VEC4',
                                                       5126, None), 'interpolation': 'LINEAR'})
        a_channels.append({'sampler': len(a_samplers) - 1, 'target': {'node': 1 + k, 'path': 'rotation'}})
    while len(data) % 4:
        data.append(0)
    g = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0, 1, 4]}],
         'nodes': [{'name': NAME, 'mesh': 0, 'skin': 0, 'extras': {'set_collision': 1, 'ignore': 1}},
                   {'name': 'align', 'children': [2]}, {'name': 'prejoint', 'children': [3]}, {'name': 'main'},
                   {'name': NAME + '-collide', 'mesh': 1, 'extras': {'set_invisible': 1}}],
         'skins': [{'joints': [1, 2, 3], 'inverseBindMatrices': a_ibm, 'skeleton': 1}],
         'animations': [{'name': 'idle', 'samplers': a_samplers, 'channels': a_channels}],
         'meshes': [{'name': NAME, 'primitives': prims}, {'name': NAME + '-collide', 'primitives': [col_prim]}],
         'materials': mats, 'textures': texs, 'samplers': [{'magFilter': 9729, 'minFilter': 9987}], 'images': imgs,
         'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(g).encode()
    while len(js) % 4:
        js += b' '
    out = (b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) + struct.pack('<I4s', len(js), b'JSON') +
           js + struct.pack('<I4s', len(data), b'BIN\0') + bytes(data))
    open(path, 'wb').write(out)


# (the same model twice: custom_levels for build-actor, common/<name>-lod0 so the extractor puts it in GAME)
write_glb(os.path.join(OUT, 'custom_levels', NAME + '.glb'))
write_glb(os.path.join(OUT, 'common', NAME + '-lod0.glb'))
print('ok', {k: sum(len(p) for p in v[0]) // 3 for k, v in parts.items()}, 'triangles;', len(CV), 'collision verts',
      'rock top %.1f m over the mud' % float(np.percentile(top[:, 1], 80)))
