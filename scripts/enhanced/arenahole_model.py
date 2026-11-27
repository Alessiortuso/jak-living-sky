# Jak Living Sky: the holes the pistons of the arena's second fight rise out of, three a hole (arenapusher, misty-obs.gc): a frame of
# Precursor metal (the pistons' own picture) lying on the sand round each one's foot, its inside walls dark, going down:
# with the piston's top flush in it (down), a hole with the piston in it. Its origin at the sand's level, in the middle;
# the piston (5.85 x 4.19 m) fits in it, a hand's breadth to spare. Collided with: its rim (Jak steps over it).
#
#   python -I scripts/enhanced/arenahole_model.py custom_assets/jak1/models
#
# Writes custom_levels/arenahole.glb (build-actor) and common/arenahole-lod0.glb (the extractor: GAME.fr3); the skeleton
# as arenapool's (see arenapool_model.py). Its picture from arenapusher.glb (made first: arenapusher_model.py).
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
NAME = 'arenahole'
PUSHER = os.path.join(OUT, 'custom_levels', 'arenapusher.glb')
IX, IZ = 5.85 / 2 + 0.12, (3 * 4.19 + 2 * 0.25) / 2 + 0.12  # its opening (half sizes): three pistons side by side
RIM = 0.7       # the rim's width
TOP = 0.22      # its top over the sand
DOWN = 1.2      # its inside walls going down under it
FLOOR = 0.18    # its black bottom, over the sand (its bumps under it: they showed through it), under the rim's top
TILE = 1.5

d = open(PUSHER, 'rb').read()
n = struct.unpack('<I', d[12:16])[0]
g = json.loads(d[20:20 + n])
img = next(i for i in g['images'] if 'metal-edge' in i['name'])
uri = opaque(img['uri'])

P, UV, C = [], [], []


def quad(a, b, c, dd, shade):
    # a b c d round it, facing out the way a->b->c turns (counter-clockwise seen from that side)
    for t in ((a, b, c), (a, c, dd)):
        for v in t:
            P.append(v)
            # (its picture laid by the two widest of its axes)
            ax = np.argsort(np.abs(np.cross(np.subtract(b, a), np.subtract(dd, a))))[:2]
            UV.append((v[ax[0]] / TILE, v[ax[1]] / TILE))
            C.append((shade, shade, shade, 1.0))


OX, OZ = IX + RIM, IZ + RIM
# the rim's top: four strips round the opening (y up: wound counter-clockwise seen from above)
corners_in = [(-IX, -IZ), (IX, -IZ), (IX, IZ), (-IX, IZ)]
corners_out = [(-OX, -OZ), (OX, -OZ), (OX, OZ), (-OX, OZ)]
for k in range(4):
    i0, i1 = corners_in[k], corners_in[(k + 1) % 4]
    o0, o1 = corners_out[k], corners_out[(k + 1) % 4]
    quad((o0[0], TOP, o0[1]), (i0[0], TOP, i0[1]), (i1[0], TOP, i1[1]), (o1[0], TOP, o1[1]), 0.5)
    # its outside: down into the sand
    quad((o1[0], TOP, o1[1]), (o1[0], -0.8, o1[1]), (o0[0], -0.8, o0[1]), (o0[0], TOP, o0[1]), 0.4)
    # its inside: the hole's walls, dark further down
    quad((i0[0], TOP, i0[1]), (i0[0], -DOWN, i0[1]), (i1[0], -DOWN, i1[1]), (i1[0], TOP, i1[1]), 0.0)
# its bottom: black, a little over the sand (the sand hidden: a hole, the pistons coming up out of it). Not collided with
quad((-IX, FLOOR, -IZ), (-IX, FLOOR, IZ), (IX, FLOOR, IZ), (IX, FLOOR, -IZ), 0.0)

P = np.array(P, np.float32)
UV = np.array(UV, np.float32)
C = np.array(C, np.float32)
# (faces checked out: each one turned to face away from the opening's middle, or up for the top)
T = P.reshape(-1, 3, 3)
nn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
mid = T.mean(1)
flip = np.zeros(len(T), bool)
for t in range(len(T)):
    if abs(nn[t, 1]) > np.linalg.norm(nn[t, [0, 2]]):
        flip[t] = nn[t, 1] < 0
    else:
        out = np.array([mid[t, 0], 0, mid[t, 2]])
        inner = abs(mid[t, 0]) <= IX + 0.01 and abs(mid[t, 2]) <= IZ + 0.01
        # (the hole's walls face its middle; the outside faces away)
        flip[t] = (np.dot(nn[t], out) > 0) == inner
T[flip] = T[flip][:, [0, 2, 1]]
P = T.reshape(-1, 3)
U = UV.reshape(-1, 3, 2)
U[flip] = U[flip][:, [0, 2, 1]]
UV = U.reshape(-1, 2)
Cc = C.reshape(-1, 3, 4)
C = Cc.reshape(-1, 4)

# the collision: the rim (its top and outside; not its inside walls, nor its bottom), its corners shared
keep = []
for t in range(len(T)):
    inner = abs(T[t].mean(0)[0]) <= IX + 0.01 and abs(T[t].mean(0)[2]) <= IZ + 0.01
    if not inner:
        keep.append(t)
CP = T[keep].reshape(-1, 3)
CV, CI = np.unique(np.round(CP, 4), axis=0, return_inverse=True)
CV = CV.astype(np.float32)
CI = CI.reshape(-1).astype(np.uint32)


def write_glb(path):
    data = bytearray()
    views, accs = [], []

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

    e1, e2 = P[1::3] - P[0::3], P[2::3] - P[0::3]
    nrm = np.cross(e1, e2)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1)[:, None], 1e-9)
    N = np.repeat(nrm, 3, axis=0).astype(np.float32)
    J4 = np.zeros((len(P), 4), np.uint8)
    J4[:, 0] = 2
    W4 = np.zeros((len(P), 4), np.float32)
    W4[:, 0] = 1.0
    prim = {'attributes': {'POSITION': ac(P, 'VEC3', 5126, 34962, True), 'NORMAL': ac(N, 'VEC3', 5126, 34962),
                           'TEXCOORD_0': ac(UV, 'VEC2', 5126, 34962), 'COLOR_0': ac(C, 'VEC4', 5126, 34962),
                           'JOINTS_0': ac(J4, 'VEC4', 5121, 34962), 'WEIGHTS_0': ac(W4, 'VEC4', 5126, 34962)},
            'indices': ac(np.arange(len(P), dtype=np.uint32), 'SCALAR', 5125, 34963), 'material': 0}
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
    mat = {'name': NAME + '-metal', 'alphaMode': 'MASK', 'alphaCutoff': 0.149, 'doubleSided': True,
           'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0.0, 'roughnessFactor': 0.9}}
    gl = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0, 1, 4]}],
          'nodes': [{'name': NAME, 'mesh': 0, 'skin': 0, 'extras': {'set_collision': 1, 'ignore': 1}},
                    {'name': 'align', 'children': [2]}, {'name': 'prejoint', 'children': [3]}, {'name': 'main'},
                    {'name': NAME + '-collide', 'mesh': 1, 'extras': {'set_invisible': 1}}],
          'skins': [{'joints': [1, 2, 3], 'inverseBindMatrices': a_ibm, 'skeleton': 1}],
          'animations': [{'name': 'idle', 'samplers': smp, 'channels': chn}],
          'meshes': [{'name': NAME, 'primitives': [prim]}, {'name': NAME + '-collide', 'primitives': [col]}],
          'materials': [mat], 'textures': [{'source': 0, 'sampler': 0}],
          'samplers': [{'magFilter': 9729, 'minFilter': 9987}], 'images': [{'uri': uri, 'name': mat['name']}],
          'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(gl).encode()
    while len(js) % 4:
        js += b' '
    out = (b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) + struct.pack('<I4s', len(js), b'JSON') +
           js + struct.pack('<I4s', len(data), b'BIN\0') + bytes(data))
    open(path, 'wb').write(out)


write_glb(os.path.join(OUT, 'custom_levels', NAME + '.glb'))
write_glb(os.path.join(OUT, 'common', NAME + '-lod0.glb'))
print('ok', len(P) // 3, 'triangles;', len(CV), 'collision corners; flipped', int(flip.sum()))
