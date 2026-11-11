# Jak Living Sky: the white eco merchant, from the models made with Higgsfield (image to 3D, from the concept chosen:
# mercante-concept/seduto-1.png and banchetto.png): him on his tall stool, his stall beside him. Made ready for
# build-actor (game.gp) and the extractor (the same glb as custom_levels/merchant.glb and common/merchant-lod0.glb):
#
# - scaled to Jak's world (him 4 m on his stool, seated as tall as Jak; the stall 5.2 m high), standing on y 0, facing +Z, his stall to his
#   left (-X), his stool at the origin;
# - the textures down from 2048 to 512, named for the renderer's looks ("merchant-cloth": cloth; the stall's wood);
# - a skeleton: align and prejoint first (as the game's own and the JET-Board's: without them build-actor and the
#   extractor number the joints one apart), main (the stool and the stall), body (breathing), head, the ears, the
#   lids (two small dark lids over his painted eyes, closing down when he blinks), his two short legs (swinging);
#   every vertex on one joint, parts told apart by where they are;
# - shading painted into the vertices (ambient occlusion: the folds, the hood, under the belly and the awning darker),
#   the textures' colours toned (the generator's loud orange to an ochre, a little more contrast);
# - one animation, "idle": a 4.8 s loop: the little legs swinging, breathing, two blinks, the head tilting, an ear
#   twitching.
#
#   python scripts/enhanced/merchant_import.py custom_assets/jak1/models
import io
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image

OUT = sys.argv[1]
SRC = os.path.join(OUT, 'source')


def load(path):
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + n])
    bs = 20 + n + 8

    def acc(i):
        a = j['accessors'][i]
        bv = j['bufferViews'][a['bufferView']]
        ct = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a['componentType']]
        nc = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
        return np.frombuffer(b, dtype=ct, count=a['count'] * nc,
                             offset=bs + bv.get('byteOffset', 0) + a.get('byteOffset', 0)).reshape(-1, nc).copy()

    p = j['meshes'][0]['primitives'][0]
    V = acc(p['attributes']['POSITION']).astype(np.float64)
    N = acc(p['attributes']['NORMAL']).astype(np.float64)
    UV = acc(p['attributes']['TEXCOORD_0']).astype(np.float64)
    I = acc(p['indices']).ravel().astype(np.uint32)
    img = j['images'][j['textures'][j['materials'][p['material']]['pbrMetallicRoughness']['baseColorTexture']['index']]
                      ['source']]
    bv = j['bufferViews'][img['bufferView']]
    tex = Image.open(io.BytesIO(b[bs + bv.get('byteOffset', 0): bs + bv.get('byteOffset', 0) + bv['byteLength']]))
    return V, N, UV, I, tex.convert('RGB')


mV, mN, mUV, mI, mTex = load(os.path.join(SRC, 'merchant-meshy.glb'))
sV, sN, sUV, sI, sTex = load(os.path.join(SRC, 'stall-meshy.glb'))

# --- the merchant's parts (in his source's units: y -0.95..0.95, his front +Z) ---
SEAT_Y = -0.06  # the stool's seat, under him
y, x, z = mV[:, 1], mV[:, 0], mV[:, 2]
legs = (y < SEAT_Y) & (y > -0.4) & (z > 0.17)
head = (y > 0.56) & ~legs
ears = head & (np.abs(x) > 0.27) & (y > 0.62)
body = (y >= SEAT_Y) & ~head
stool = ~(legs | head | body)


# his painted eyes: the white of them, on the face, from the texture
def tex_at(tex, uv):
    a = np.asarray(tex) / 255.0
    H, W = a.shape[:2]
    return a[np.clip((uv[:, 1] % 1) * H, 0, H - 1).astype(int), np.clip((uv[:, 0] % 1) * W, 0, W - 1).astype(int)]


col = tex_at(mTex, mUV)
white = (col.min(1) > 0.72) & (y > 0.62) & (np.abs(x) < 0.2) & (z > 0.1)
# (painted only: few vertices fall on them, so the face's triangles sampled inside, a grid of points each)
tri = mI.reshape(-1, 3)
tc = mV[tri].mean(1)
ft = tri[(tc[:, 1] > 0.6) & (np.abs(tc[:, 0]) < 0.22) & (tc[:, 2] > 0.05)]
bary = np.array([(a, b2, 1 - a - b2) for a in np.linspace(0, 1, 7) for b2 in np.linspace(0, 1, 7) if a + b2 <= 1])
pts = np.einsum('kj,tjc->tkc', bary, mV[ft]).reshape(-1, 3)
puv = np.einsum('kj,tjc->tkc', bary, mUV[ft]).reshape(-1, 2)
pw = tex_at(mTex, puv).min(1) > 0.7
eyes = []
for sgn in (1, -1):
    m = pw & (np.sign(pts[:, 0]) == sgn)
    if m.sum() < 3:
        raise SystemExit('no eye found on side %d' % sgn)
    P = pts[m]
    eyes.append((P.mean(0), P.min(0), P.max(0)))
print('eyes', [(e[0].round(3), (e[2] - e[1]).round(3)) for e in eyes])

# --- to the game: metres, on the ground ---
MS = 4.0 / (mV[:, 1].max() - mV[:, 1].min())  # (seated, his head as high as Jak's)
m_off = np.array([0.0, -mV[:, 1].min() * MS, 0.0])
SS = 5.2 / (sV[:, 1].max() - sV[:, 1].min())
stall_w = (sV[:, 0].max() - sV[:, 0].min()) * SS
s_off = np.array([-(stall_w / 2 + 1.25), -sV[:, 1].min() * SS, -0.8])


def M(p):
    return np.asarray(p) * MS + m_off


JOINTS = [  # name, parent, bind position (world, metres)
    ('align', -1, (0, 0, 0)),
    ('prejoint', 0, (0, 0, 0)),
    ('main', 1, (0, 0, 0)),
    ('body', 2, tuple(M((0, SEAT_Y, 0)))),
    ('head', 3, tuple(M((0, 0.56, 0)))),
    ('ear_l', 4, tuple(M((0.27, 0.72, 0)))),
    ('ear_r', 4, tuple(M((-0.27, 0.72, 0)))),
    ('lid_l', 4, tuple(M((eyes[0][0][0], eyes[0][2][1] + 0.01, eyes[0][2][2] + 0.01)))),
    ('lid_r', 4, tuple(M((eyes[1][0][0], eyes[1][2][1] + 0.01, eyes[1][2][2] + 0.01)))),
    ('leg_l', 2, tuple(M((0.12, SEAT_Y, 0.3)))),
    ('leg_r', 2, tuple(M((-0.12, SEAT_Y, 0.3)))),
]
J = {n: i for i, (n, _, _) in enumerate(JOINTS)}

mJ = np.full(len(mV), J['main'])
mJ[body] = J['body']
mJ[head] = J['head']
mJ[ears & (x > 0)] = J['ear_l']
mJ[ears & (x < 0)] = J['ear_r']
mJ[legs & (x > 0)] = J['leg_l']
mJ[legs & (x < 0)] = J['leg_r']
print('parts', {n: int((mJ == i).sum()) for n, i in J.items()})

# the lids: small dark half-discs just in front of each eye, hanging down from the joint (scaled to nothing but a
# line while open)
lid_V, lid_N, lid_I, lid_J = [], [], [], []
for k, (c, lo, hi) in enumerate(eyes):
    jn = J['lid_l' if k == 0 else 'lid_r']
    w = (hi[0] - lo[0]) * 0.6 + 0.012
    h = (hi[1] - lo[1]) * 1.15 + 0.02
    top = np.array([c[0], hi[1] + 0.01, hi[2] + 0.012])
    base = len(lid_V)
    seg = 12
    lid_V.append(top)
    lid_N.append((0, 0, 1))
    for s in range(seg + 1):
        a = math.pi * s / seg
        lid_V.append(top + np.array([-w * math.cos(a), -h * math.sin(a), 0.0]))
        lid_N.append((0, 0, 1))
    for s in range(seg):
        lid_I += [base, base + 1 + s, base + 2 + s]
    lid_J += [jn] * (seg + 2)
lid_V = np.array(lid_V)

# --- one glb: two primitives (him and his lids on his texture, the stall on its own) ---
mVw = M(mV)
lidVw = M(lid_V)
sVw = sV * SS + s_off


def png(img, size=512):
    bio = io.BytesIO()
    img.resize((size, size), Image.LANCZOS).save(bio, 'PNG', optimize=True)
    return bio.getvalue()


# the lids sample the hood's dark inside: a uv on the black of his face (the darkest texel near the eyes)
face = (y > 0.6) & (np.abs(x) < 0.12) & (z > 0.1) & ~white
dark_uv = mUV[face][np.argmin(col[face].sum(1))]

prims = [
    ('merchant-cloth', np.vstack([mVw, lidVw]), np.vstack([mN, np.tile([0, 0, 1.0], (len(lidVw), 1))]),
     np.vstack([mUV, np.tile(dark_uv, (len(lidVw), 1))]),
     np.concatenate([mI, np.array(lid_I, np.uint32) + len(mVw)]), np.concatenate([mJ, lid_J]), mTex),
    ('merchant-stall', sVw, sN, sUV, sI, np.full(len(sVw), J['main']), sTex),
]


# --- shading painted into the vertices: ambient occlusion (the folds of the cloth, the hood's inside, under the
# scarf and the belly, between the planks, under the awning darker: what Jak's own textures have painted in) ---
def bake_ao(prims, cell=0.03, dirs_n=40, reach=0.55):
    """each vertex: how much of the sky over it (the hemisphere round its normal) the model leaves open. Its
    surfaces sampled into a grid of cells; rays marched from just off each vertex"""
    pts = []
    bary = np.array([(a, b2, 1 - a - b2) for a in np.linspace(0, 1, 5) for b2 in np.linspace(0, 1, 5) if a + b2 <= 1])
    for _, V, _, _, I, _, _ in prims:
        T_ = V[I.reshape(-1, 3)]
        pts.append(np.einsum('kj,tjc->tkc', bary, T_).reshape(-1, 3))
    pts = np.vstack(pts)
    lo = pts.min(0) - 0.6
    dims = np.ceil((pts.max(0) + 0.6 - lo) / cell).astype(int) + 1
    occ = np.zeros(dims, bool)
    c = ((pts - lo) / cell).astype(int)
    occ[c[:, 0], c[:, 1], c[:, 2]] = True
    rng = np.random.default_rng(7)
    # directions: a fixed spread over the sphere, kept to each vertex's hemisphere (cosine weighted)
    d = rng.normal(size=(dirs_n * 2, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    steps = np.arange(2.5, reach / cell) * cell
    out = []
    for _, V, N, _, _, _, _ in prims:
        Nn = N / np.maximum(np.linalg.norm(N, axis=1), 1e-6)[:, None]
        ao = np.zeros(len(V))
        for s in range(0, len(V), 2000):
            v = V[s:s + 2000]
            n_ = Nn[s:s + 2000]
            cos = n_ @ d.T  # (n, dirs)
            w = np.clip(cos, 0, None)
            hit = np.zeros_like(w)
            for st in steps:
                p = v[:, None, :] + n_[:, None, :] * (cell * 1.5) + d[None, :, :] * st
                q = ((p - lo) / cell).astype(int)
                q = np.clip(q, 0, dims - 1)
                o = occ[q[..., 0], q[..., 1], q[..., 2]]
                # nearer things shade more
                hit = np.maximum(hit, o * (1 - st / reach) ** 0.6)
            ao[s:s + 2000] = 1 - (hit * w).sum(1) / np.maximum(w.sum(1), 1e-6)
        out.append(ao)
    return out


AO = bake_ao(prims)
SHADE = []
for (mat, V, N, UV, I, Jn, tex), ao in zip(prims, AO):
    s = 0.42 + 0.58 * np.clip(ao, 0, 1) ** 1.3
    # worn: a soft mottling over it all (sun-faded, dusty, grimy in places), from smooth noise in space (so no seams
    # where the texture's islands meet)
    ph = np.array([[1.7, 0.3, 2.9], [3.1, 1.9, 0.7], [0.5, 2.3, 1.3]])
    mot = sum(np.sin(V @ (np.array(f_) * k_) + k_) for k_, f_ in zip((2.3, 4.1, 7.3), ph)) / 3.0
    s *= 0.93 + 0.08 * mot
    if mat == 'merchant-cloth':
        # a touch lighter toward his head, as lit from above; his hem dusty; his lids as dark as the hood anyway
        h = (V[:, 1] - V[:, 1].min()) / (V[:, 1].max() - V[:, 1].min())
        s *= 0.9 + 0.14 * h
        s[len(mV):] = 1.0
    else:
        # the stall's foot darker (the ground's dirt and shade)
        s *= 0.82 + 0.18 * np.clip(V[:, 1] / 0.8, 0, 1)
    SHADE.append(np.clip(s, 0, 1.1))
    print(mat, 'shade', round(float(s.min()), 2), round(float(s.mean()), 2), round(float(s.max()), 2))


def grade(img):
    """the colours less flat and loud than the generator's: its orange toned to an ochre, a little more contrast,
    a little less saturation"""
    from PIL import ImageEnhance
    a = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    # the loud oranges (red high, green half, blue low) toward a warm ochre, a little browner
    orange = np.clip((r - b - 0.35) * 3, 0, 1) * np.clip(1 - np.abs(g / np.maximum(r, 1e-3) - 0.6) * 2.5, 0, 1)
    a[..., 0] = r * (1 - 0.15 * orange)
    a[..., 1] = g * (1 - 0.05 * orange)
    a[..., 2] = b + (0.06 + 0.12 * b) * orange
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(0.84)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    return img


prims = [(mat, V, N, UV, I, Jn, grade(tex)) for (mat, V, N, UV, I, Jn, tex) in prims]

# --- the idle: 4.8 s ---
FPS = 30
T = 4.8
NF = int(T * FPS) + 1
times = np.arange(NF) / FPS


def qx(a):
    return (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))


def qz(a):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def blink(t):
    out = 0.0
    for b in (1.4, 3.6, 3.85):
        out = max(out, max(0.0, 1 - abs(t - b) / 0.1))
    return out


anim = {}
for i, (name, parent, pos) in enumerate(JOINTS):
    local = np.array(pos) - (np.array(JOINTS[parent][2]) if parent >= 0 else 0)
    tr, ro, sc = [], [], []
    for t in times:
        w = 2 * math.pi * t / T
        q = (0.0, 0.0, 0.0, 1.0)
        s = (1.0, 1.0, 1.0)
        if name == 'body':
            br = 0.5 + 0.5 * math.sin(2 * w)
            s = (1 + 0.012 * br, 1 + 0.02 * br, 1 + 0.012 * br)
            q = qz(0.02 * math.sin(w))
        elif name == 'head':
            q = qmul(qz(-0.05 * math.sin(w + 0.7)), qx(0.04 * math.sin(2 * w + 0.3)))
        elif name.startswith('leg'):
            # swinging, the two out of step, four swings a loop
            ph = 0 if name == 'leg_l' else math.pi
            q = qx(0.35 * math.sin(4 * w + ph) - 0.1)
        elif name.startswith('ear'):
            sg = 1 if name == 'ear_l' else -1
            tw = max(0.0, 1 - abs(t - 2.5) / 0.16) * (1.0 if name == 'ear_l' else 0.35)
            q = qz(sg * (0.04 * math.sin(w) + 0.22 * tw))
        elif name.startswith('lid'):
            s = (1.0, 0.04 + 0.96 * blink(t), 1.0)
        tr.append(local)
        ro.append(q)
        sc.append(s)
    anim[i] = (np.array(tr, np.float32), np.array(ro, np.float32), np.array(sc, np.float32))


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

    mats, texs, imgs, out_prims = [], [], [], []
    for k, (mat, V, N, UV, I, Jn, tex) in enumerate(prims):
        V = V.astype(np.float32)
        N = N.astype(np.float32)
        N /= np.maximum(np.linalg.norm(N, axis=1), 1e-6)[:, None]
        C4 = np.ones((len(V), 4), np.float32)
        C4[:, :3] = 0.5 * SHADE[k][:, None]  # (merc doubles them: 0.5 is the texture as it is)
        J4 = np.zeros((len(V), 4), np.uint8)
        J4[:, 0] = Jn
        W4 = np.zeros((len(V), 4), np.float32)
        W4[:, 0] = 1
        a = [acc(V, 5126, 'VEC3', 34962, True), acc(N, 5126, 'VEC3', 34962),
             acc(UV.astype(np.float32), 5126, 'VEC2', 34962), acc(C4, 5126, 'VEC4', 34962),
             acc(J4, 5121, 'VEC4', 34962), acc(W4, 5126, 'VEC4', 34962),
             acc(I.astype(np.uint32), 5125, 'SCALAR', 34963)]
        imgs.append({'bufferView': view(png(tex)), 'mimeType': 'image/png', 'name': mat})
        texs.append({'source': k, 'sampler': 0})
        mats.append({'name': mat, 'pbrMetallicRoughness': {'baseColorTexture': {'index': k}, 'metallicFactor': 0.0,
                                                           'roughnessFactor': 0.8}})
        out_prims.append({'attributes': {'POSITION': a[0], 'NORMAL': a[1], 'TEXCOORD_0': a[2], 'COLOR_0': a[3],
                                         'JOINTS_0': a[4], 'WEIGHTS_0': a[5]}, 'indices': a[6], 'material': k})
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
        mm = np.eye(4, dtype=np.float32)
        mm[:3, 3] = -np.array(pos)
        ibm[i] = mm.T
    a_ibm = acc(ibm.reshape(-1, 16), 5126, 'MAT4')
    skin = {'joints': list(range(1, 1 + len(JOINTS))), 'inverseBindMatrices': a_ibm, 'skeleton': 1}
    a_t = acc(times.astype(np.float32), 5126, 'SCALAR', None, True)
    samplers, channels = [], []
    for i in range(len(JOINTS)):
        for arr, path_, typ in zip(anim[i], ('translation', 'rotation', 'scale'), ('VEC3', 'VEC4', 'VEC3')):
            samplers.append({'input': a_t, 'output': acc(arr, 5126, typ), 'interpolation': 'LINEAR'})
            channels.append({'sampler': len(samplers) - 1, 'target': {'node': 1 + i, 'path': path_}})
    while len(data) % 4:
        data.append(0)
    g = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0, 1]}], 'nodes': nodes,
         'meshes': [{'name': name, 'primitives': out_prims}], 'skins': [skin],
         'animations': [{'name': 'idle', 'samplers': samplers, 'channels': channels}],
         'materials': mats, 'textures': texs, 'samplers': [{'magFilter': 9729, 'minFilter': 9987}], 'images': imgs,
         'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(g).encode()
    while len(js) % 4:
        js += b' '
    open(path, 'wb').write(b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) +
                           struct.pack('<I4s', len(js), b'JSON') + js + struct.pack('<I4s', len(data), b'BIN\0') +
                           bytes(data))


write_glb(os.path.join(OUT, 'custom_levels', 'merchant.glb'), 'merchant')
write_glb(os.path.join(OUT, 'common', 'merchant-lod0.glb'), 'merchant')
# and him alone on his stool, no stall (the arena's middle): his texture the same picture, shared in the game
prims = [p_ for p_ in prims if p_[0] != 'merchant-stall']
write_glb(os.path.join(OUT, 'custom_levels', 'merchantsolo.glb'), 'merchantsolo')
write_glb(os.path.join(OUT, 'common', 'merchantsolo-lod0.glb'), 'merchantsolo')
print('ok', sum(len(p[1]) for p in prims), 'vertices', sum(len(p[4]) for p in prims) // 3, 'triangles',
      'stall offset', s_off.round(2), 'scales', round(MS, 3), round(SS, 3))
