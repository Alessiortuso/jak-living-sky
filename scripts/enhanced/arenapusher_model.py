# Jak Living Sky: the pillars of the arena's second fight ("CAMPIONE: OSSOBELLO", misty-obs.gc): Snowy Mountain's
# Precursor pistons (snowpusher, the yellow eco vent's room), stood on end. Its own triangles and pictures are taken
# from the decompiler's export of it (decompiler_out/jak1/levels/snow/snowpusher-lod0.glb: its long side, the way it
# pushes, made the upward one), its foot at its origin; drawn as it is, collided with as a box round it.
#
#   python -I scripts/enhanced/arenapusher_model.py custom_assets/jak1/models <snowpusher-lod0.glb>
#
# Writes custom_levels/arenapusher.glb (build-actor) and common/arenapusher-lod0.glb (the extractor: GAME.fr3); the
# skeleton as arenapool's (align, prejoint, main, a still "idle": see arenapool_model.py).
import json
import os
import struct
import sys

import numpy as np


def opaque(uri):
    # (its picture with no see-through parts: alpha-tested, Misty's way, its holes let the sand through)
    import base64
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1]))).convert('RGBA')
    im.putalpha(255)
    out = io.BytesIO()
    im.save(out, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()

OUT = sys.argv[1]
SRC = sys.argv[2]
NAME = 'arenapusher'
SHADE = 1.0  # its vertex colours times this

d = open(SRC, 'rb').read()
n = struct.unpack('<I', d[12:16])[0]
g = json.loads(d[20:20 + n])
BIN = 20 + n + 8


def acc(i, ct, k):
    a = g['accessors'][i]
    v = g['bufferViews'][a['bufferView']]
    o = BIN + v.get('byteOffset', 0) + a.get('byteOffset', 0)
    st = v.get('byteStride', np.dtype(ct).itemsize * k)
    return np.array([np.frombuffer(d[o + j * st:o + j * st + np.dtype(ct).itemsize * k], ct) for j in range(a['count'])])


prim0 = g['meshes'][0]['primitives'][0]
POS = acc(prim0['attributes']['POSITION'], np.float32, 3)
UVS = acc(prim0['attributes']['TEXCOORD_0'], np.float32, 2)
ca = g['accessors'][prim0['attributes']['COLOR_0']]
COL = acc(prim0['attributes']['COLOR_0'], {5126: np.float32, 5121: np.uint8, 5123: np.uint16}[ca['componentType']], 4)
COL = COL.astype(np.float32) / {5126: 1.0, 5121: 255.0, 5123: 65535.0}[ca['componentType']]

parts = []  # (material name, picture's uri, positions, uvs, colours)
for p in g['meshes'][0]['primitives']:
    mat = g['materials'][p['material']]
    if 'baseColorTexture' not in mat.get('pbrMetallicRoughness', {}):
        continue
    ic = g['accessors'][p['indices']]['componentType']
    I = acc(p['indices'], {5123: np.uint16, 5125: np.uint32}[ic], 1).reshape(-1)
    tex = g['textures'][mat['pbrMetallicRoughness']['baseColorTexture']['index']]
    uri = opaque(g['images'][tex['source']]['uri'])
    P = POS[I]
    # stood on end: its face (the rounded stone with the metal groove, x min: what comes out of the wall) on top, its
    # back (x max, in the wall) its foot at 0, its middle over its origin (a turn: its triangles still face out)
    Q = np.stack([P[:, 1] - 2.925, 8.02 - P[:, 0], P[:, 2]], 1).astype(np.float32)
    UV = UVS[I]
    C = np.clip(COL[I] * SHADE, 0, 1)
    C[:, 3] = 1.0
    parts.append((NAME + '-' + mat['name'].replace('-envmap', ''), uri, Q, UV.astype(np.float32), C.astype(np.float32)))

ALL = np.concatenate([q for _, _, q, _, _ in parts])
lo, hi = ALL.min(0), ALL.max(0)
# the collision: a box round it (its corners shared)
x0, y0, z0 = lo
x1, y1, z1 = hi
CV = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
               [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], np.float32)
CI = np.array([0, 2, 1, 0, 3, 2,  4, 5, 6, 4, 6, 7,  0, 1, 5, 0, 5, 4,  1, 2, 6, 1, 6, 5,
               2, 3, 7, 2, 7, 6,  3, 0, 4, 3, 4, 7], np.uint32).reshape(-1, 3)
# (each face out: collision is one-sided, a box facing in lets Jak through)
_ct = CV[CI]
_out = np.einsum('ij,ij->i', np.cross(_ct[:, 1] - _ct[:, 0], _ct[:, 2] - _ct[:, 0]), _ct.mean(1) - CV.mean(0)) < 0
CI[_out] = CI[_out][:, [0, 2, 1]]
CI = CI.reshape(-1).astype(np.uint32)


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

    def ac(arr, typ, ct, target, mm=False):
        a = {'bufferView': add(arr.tobytes(), target), 'componentType': ct, 'count': len(arr), 'type': typ}
        if mm:
            a['min'] = arr.min(0).tolist()
            a['max'] = arr.max(0).tolist()
        accs.append(a)
        return len(accs) - 1

    for k, (mname, uri, P, UV, C) in enumerate(parts):
        e1, e2 = P[1::3] - P[0::3], P[2::3] - P[0::3]
        nn = np.cross(e1, e2)
        nn /= np.maximum(np.linalg.norm(nn, axis=1)[:, None], 1e-9)
        N = np.repeat(nn, 3, axis=0).astype(np.float32)
        J4 = np.zeros((len(P), 4), np.uint8)
        J4[:, 0] = 2  # (all on main)
        W4 = np.zeros((len(P), 4), np.float32)
        W4[:, 0] = 1.0
        imgs.append({'uri': uri, 'name': mname})
        texs.append({'source': k, 'sampler': 0})
        mats.append({'name': mname, 'alphaMode': 'MASK', 'alphaCutoff': 0.149, 'doubleSided': True,
                     'pbrMetallicRoughness': {'baseColorTexture': {'index': k}, 'metallicFactor': 0.0,
                                              'roughnessFactor': 0.9}})
        prims.append({'attributes': {'POSITION': ac(P, 'VEC3', 5126, 34962, True), 'NORMAL': ac(N, 'VEC3', 5126, 34962),
                                     'TEXCOORD_0': ac(UV, 'VEC2', 5126, 34962), 'COLOR_0': ac(C, 'VEC4', 5126, 34962),
                                     'JOINTS_0': ac(J4, 'VEC4', 5121, 34962), 'WEIGHTS_0': ac(W4, 'VEC4', 5126, 34962)},
                      'indices': ac(np.arange(len(P), dtype=np.uint32), 'SCALAR', 5125, 34963), 'material': k})
    CN = np.tile(np.array([0, 1, 0], np.float32), (len(CV), 1))
    col = {'attributes': {'POSITION': ac(CV, 'VEC3', 5126, 34962, True), 'NORMAL': ac(CN, 'VEC3', 5126, 34962)},
           'indices': ac(CI, 'SCALAR', 5125, 34963)}
    ibm = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    a_ibm = ac(ibm.reshape(-1, 16), 'MAT4', 5126, None)
    a_t = ac(np.array([0.0, 1.0], np.float32), 'SCALAR', 5126, None, True)
    smp, chn = [], []
    for k in range(3):
        smp.append({'input': a_t, 'output': ac(np.zeros((2, 3), np.float32), 'VEC3', 5126, None), 'interpolation': 'LINEAR'})
        chn.append({'sampler': len(smp) - 1, 'target': {'node': 1 + k, 'path': 'translation'}})
        smp.append({'input': a_t, 'output': ac(np.tile(np.array([0, 0, 0, 1.0], np.float32), (2, 1)), 'VEC4', 5126, None),
                    'interpolation': 'LINEAR'})
        chn.append({'sampler': len(smp) - 1, 'target': {'node': 1 + k, 'path': 'rotation'}})
    while len(data) % 4:
        data.append(0)
    gl = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0, 1, 4]}],
          'nodes': [{'name': NAME, 'mesh': 0, 'skin': 0, 'extras': {'set_collision': 1, 'ignore': 1}},
                    {'name': 'align', 'children': [2]}, {'name': 'prejoint', 'children': [3]}, {'name': 'main'},
                    {'name': NAME + '-collide', 'mesh': 1, 'extras': {'set_invisible': 1}}],
          'skins': [{'joints': [1, 2, 3], 'inverseBindMatrices': a_ibm, 'skeleton': 1}],
          'animations': [{'name': 'idle', 'samplers': smp, 'channels': chn}],
          'meshes': [{'name': NAME, 'primitives': prims}, {'name': NAME + '-collide', 'primitives': [col]}],
          'materials': mats, 'textures': texs, 'samplers': [{'magFilter': 9729, 'minFilter': 9987}], 'images': imgs,
          'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(gl).encode()
    while len(js) % 4:
        js += b' '
    out = (b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) + struct.pack('<I4s', len(js), b'JSON') +
           js + struct.pack('<I4s', len(data), b'BIN\0') + bytes(data))
    open(path, 'wb').write(out)


write_glb(os.path.join(OUT, 'custom_levels', NAME + '.glb'))
write_glb(os.path.join(OUT, 'common', NAME + '-lod0.glb'))
print('ok', sum(len(p[2]) for p in parts) // 3, 'triangles; size', (hi - lo).round(2), 'from', lo.round(2), 'to', hi.round(2))
