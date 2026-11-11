# Jak Living Sky: a still prop from a model made with Higgsfield (image to 3D) made ready for build-actor and the
# extractor, as the merchant (merchant_import.py) but with no moving parts:
# - scaled to a height (metres), standing on y 0, centred on x and z, facing +Z as made;
# - its texture down to 512, named for the renderer's looks (the material name given);
# - shading painted into the vertices (ambient occlusion, a soft wear), the generator's colours toned;
# - a skeleton of align, prejoint and main (as the game's own: without align and prejoint build-actor and the
#   extractor number the joints one apart), an "idle" animation that holds still.
# Written as custom_levels/<name>.glb and common/<name>-lod0.glb.
#
#   python scripts/enhanced/prop_import.py <source.glb> <name> <height m> <material> [models dir]
import io
import json
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageEnhance

SRC, NAME, HEIGHT, MAT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
OUT = sys.argv[5] if len(sys.argv) > 5 else 'custom_assets/jak1/models'


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
    img = j['images'][j['textures'][j['materials'][p['material']]['pbrMetallicRoughness']['baseColorTexture']['index']]
                      ['source']]
    bv = j['bufferViews'][img['bufferView']]
    tex = Image.open(io.BytesIO(b[bs + bv.get('byteOffset', 0): bs + bv.get('byteOffset', 0) + bv['byteLength']]))
    return (acc(p['attributes']['POSITION']).astype(np.float64), acc(p['attributes']['NORMAL']).astype(np.float64),
            acc(p['attributes']['TEXCOORD_0']).astype(np.float64), acc(p['indices']).ravel().astype(np.uint32),
            tex.convert('RGB'))


V, N, UV, I, tex = load(SRC)
s = HEIGHT / (V[:, 1].max() - V[:, 1].min())
V = (V - np.array([(V[:, 0].max() + V[:, 0].min()) / 2, V[:, 1].min(), (V[:, 2].max() + V[:, 2].min()) / 2])) * s


def bake_ao(V, N, I, cell=0.03, dirs_n=40, reach=0.5):
    bary = np.array([(a, b2, 1 - a - b2) for a in np.linspace(0, 1, 5) for b2 in np.linspace(0, 1, 5) if a + b2 <= 1])
    pts = np.einsum('kj,tjc->tkc', bary, V[I.reshape(-1, 3)]).reshape(-1, 3)
    lo = pts.min(0) - 0.6
    dims = np.ceil((pts.max(0) + 0.6 - lo) / cell).astype(int) + 1
    occ = np.zeros(dims, bool)
    c = ((pts - lo) / cell).astype(int)
    occ[c[:, 0], c[:, 1], c[:, 2]] = True
    d = np.random.default_rng(7).normal(size=(dirs_n * 2, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    steps = np.arange(2.5, reach / cell) * cell
    Nn = N / np.maximum(np.linalg.norm(N, axis=1), 1e-6)[:, None]
    ao = np.zeros(len(V))
    for k in range(0, len(V), 2000):
        v, n_ = V[k:k + 2000], Nn[k:k + 2000]
        w = np.clip(n_ @ d.T, 0, None)
        hit = np.zeros_like(w)
        for st in steps:
            q = np.clip(((v[:, None, :] + n_[:, None, :] * cell * 1.5 + d[None] * st - lo) / cell).astype(int), 0, dims - 1)
            hit = np.maximum(hit, occ[q[..., 0], q[..., 1], q[..., 2]] * (1 - st / reach) ** 0.6)
        ao[k:k + 2000] = 1 - (hit * w).sum(1) / np.maximum(w.sum(1), 1e-6)
    return ao


ao = bake_ao(V, N, I)
shade = 0.42 + 0.58 * np.clip(ao, 0, 1) ** 1.3
ph = np.array([[1.7, 0.3, 2.9], [3.1, 1.9, 0.7], [0.5, 2.3, 1.3]])
shade *= 0.93 + 0.08 * sum(np.sin(V @ (f_ * k_) + k_) for k_, f_ in zip((2.3, 4.1, 7.3), ph)) / 3.0
shade *= 0.85 + 0.15 * np.clip(V[:, 1] / 0.6, 0, 1)  # (its foot darker: the ground's shade)

a = np.asarray(tex).astype(np.float32) / 255.0
img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
img = ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(0.88)).enhance(1.12).resize((512, 512), Image.LANCZOS)
bio = io.BytesIO()
img.save(bio, 'PNG', optimize=True)
png = bio.getvalue()


def write(path):
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
        x = {'bufferView': view(arr.tobytes(), target), 'componentType': ctype, 'count': len(arr), 'type': typ}
        if minmax:
            x['min'] = arr.min(0).tolist()
            x['max'] = arr.max(0).tolist()
        accs.append(x)
        return len(accs) - 1

    Nn = (N / np.maximum(np.linalg.norm(N, axis=1), 1e-6)[:, None]).astype(np.float32)
    C4 = np.ones((len(V), 4), np.float32)
    C4[:, :3] = 0.5 * np.clip(shade, 0, 1.1)[:, None]
    J4 = np.zeros((len(V), 4), np.uint8)
    J4[:, 0] = 2  # (main)
    W4 = np.zeros((len(V), 4), np.float32)
    W4[:, 0] = 1
    at = [acc(V.astype(np.float32), 5126, 'VEC3', 34962, True), acc(Nn, 5126, 'VEC3', 34962),
          acc(UV.astype(np.float32), 5126, 'VEC2', 34962), acc(C4, 5126, 'VEC4', 34962),
          acc(J4, 5121, 'VEC4', 34962), acc(W4, 5126, 'VEC4', 34962), acc(I, 5125, 'SCALAR', 34963)]
    imgv = view(png)
    nodes = [{'name': NAME, 'mesh': 0, 'skin': 0}, {'name': 'align', 'children': [2]},
             {'name': 'prejoint', 'children': [3]}, {'name': 'main'}]
    ibm = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    a_ibm = acc(ibm.reshape(-1, 16), 5126, 'MAT4')
    a_t = acc(np.array([0.0, 1.0], np.float32), 5126, 'SCALAR', None, True)
    samplers, channels = [], []
    for k in range(3):
        samplers.append({'input': a_t, 'output': acc(np.zeros((2, 3), np.float32), 5126, 'VEC3'),
                         'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1, 'target': {'node': 1 + k, 'path': 'translation'}})
        samplers.append({'input': a_t, 'output': acc(np.tile(np.array([0, 0, 0, 1.0], np.float32), (2, 1)), 5126,
                                                     'VEC4'), 'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1, 'target': {'node': 1 + k, 'path': 'rotation'}})
    while len(data) % 4:
        data.append(0)
    g = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0, 1]}], 'nodes': nodes,
         'meshes': [{'name': NAME, 'primitives': [{'attributes': {'POSITION': at[0], 'NORMAL': at[1],
                                                                  'TEXCOORD_0': at[2], 'COLOR_0': at[3],
                                                                  'JOINTS_0': at[4], 'WEIGHTS_0': at[5]},
                                                   'indices': at[6], 'material': 0}]}],
         'skins': [{'joints': [1, 2, 3], 'inverseBindMatrices': a_ibm, 'skeleton': 1}],
         'animations': [{'name': 'idle', 'samplers': samplers, 'channels': channels}],
         'materials': [{'name': MAT, 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0.0,
                                                              'roughnessFactor': 0.8}}],
         'textures': [{'source': 0, 'sampler': 0}], 'samplers': [{'magFilter': 9729, 'minFilter': 9987}],
         'images': [{'bufferView': imgv, 'mimeType': 'image/png', 'name': MAT}],
         'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(g).encode()
    while len(js) % 4:
        js += b' '
    open(path, 'wb').write(b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) +
                           struct.pack('<I4s', len(js), b'JSON') + js + struct.pack('<I4s', len(data), b'BIN\0') +
                           bytes(data))


write(os.path.join(OUT, 'custom_levels', NAME + '.glb'))
write(os.path.join(OUT, 'common', NAME + '-lod0.glb'))
print('ok', NAME, len(V), 'vertices', len(I) // 3, 'triangles', 'size', (V.max(0) - V.min(0)).round(2))
