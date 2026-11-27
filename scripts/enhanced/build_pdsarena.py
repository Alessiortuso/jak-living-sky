# Jak Living Sky: the merchant's JET-Board fight. From Kuitar5's The Forgotten Lands (used with the author's
# permission): its boss, the PDS sentinel, with its arena's actors, moved onto Misty's arena (docs/pds-arena-port.md).
# Writes the small extra level "pdsarena" (custom_assets/jak1/levels/pdsarena: jsonc with the actors, gd, a tiny glb
# far below so Jak is never inside it) and copies the boss models into custom_assets/jak1/models/custom_levels.
#
#   python scripts/enhanced/build_pdsarena.py <the-forgotten-lands checkout>
import base64
import json
import math
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'custom_assets/jak1/levels/pdsarena'
MODELS = ROOT / 'custom_assets/jak1/models/custom_levels'

VALLEY_C = (2075.05, 7.93, 1852.18)  # the mod's arena (metres)
MISTY_C = (-197.28, 28.0, 958.28)  # Misty's arena floor
ROT = 67.0  # degrees about y: the gems' stands clear of Misty's silo side (its structures fill 0-122 degrees round the arena, 40-49 m out)
TRAP_LIFT = 1.0  # metres: above Misty's raised ring round its groove (28.52 m, 14-22 m out), even a squashed tile (-0.16)
PATH_SCALE = 0.7  # the sentinel's flight round the arena, pulled in (66 m out is beyond Misty's walls)

DROP_ETYPES = {'lab-door', 'tfl-commentary-node', 'pds-plat-postfight-rectangle', 'pds-plat-postfight-square'}
MODEL_NAMES = ['pds-sentinel', 'pds-gem-green', 'pds-gem-yellow', 'pds-gem-blue', 'pds-trap-central', 'pds-trap-inner',
               'pds-trap-mid-left', 'pds-trap-mid-right', 'pds-trap-out-left', 'pds-trap-out-right', 'pds-plat-circle',
               'pds-plat-square', 'pds-plat-rectangle', 'pds-swingpole', 'pds-cam']


def rot_xz(dx, dz, deg):
    a = math.radians(deg)
    return dx * math.cos(a) + dz * math.sin(a), -dx * math.sin(a) + dz * math.cos(a)


def move(p, scale=1.0):
    dx, dz = (p[0] - VALLEY_C[0]) * scale, (p[2] - VALLEY_C[2]) * scale
    rx, rz = rot_xz(dx, dz, ROT)
    return [round(MISTY_C[0] + rx, 3), round(p[1] - VALLEY_C[1] + MISTY_C[1], 3), round(MISTY_C[2] + rz, 3)]


def turn_quat(q):
    # q' = (rotation about y by ROT) * q
    a = math.radians(ROT) / 2
    ry = (0.0, math.sin(a), 0.0, math.cos(a))
    x1, y1, z1, w1 = ry
    x2, y2, z2, w2 = q
    return [round(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, 4), round(w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, 4),
            round(w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, 4), round(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, 4)]


raw = (SRC / 'custom_assets/jak1/levels/valley/valley.jsonc').read_text(encoding='utf-8')
txt = re.sub(r',(\s*[\]}])', r'\1', re.sub(r'//[^\n]*', '', raw))
valley = json.loads(txt)

actors = []
dropped = set()
for a in valley['actors']:
    e = a['etype']
    name = a['lump'].get('name', '')
    keep = (e.startswith('pds') or e == 'pds-cam') and e not in DROP_ETYPES and 'commentary' not in name
    if not keep:
        if e.startswith('pds') or e in DROP_ETYPES:
            dropped.add(name)
        continue
    actors.append(a)
names = {a['lump']['name'] for a in actors}

out_actors = []
for a in actors:
    b = json.loads(json.dumps(a))
    sentinel = b['etype'] == 'pds-sentinel'
    b['trans'] = move(b['trans'])
    b['bsphere'] = move(b['bsphere'][:3]) + [b['bsphere'][3]]
    b['quat'] = turn_quat(b['quat'])
    # the fight's floor (its traps) a little above Misty's own, flush at 28 m: there the two showed through each
    # other and Jak stood on Misty's, so the traps never felt him ('ridden') nor lit up
    if b['etype'].startswith('pds-trap') or b['etype'] == 'pds-master':  # (the master: where enemies land, pds-nav.gc too)
        b['trans'][1] = round(b['trans'][1] + TRAP_LIFT, 3)
        b['bsphere'][1] = round(b['bsphere'][1] + TRAP_LIFT, 3)
    # (no task of the mod's: the merchant's fight keeps its own state)
    if b.get('game_task') not in (0, None):
        b['game_task'] = 0
    lump = b['lump']
    for k, v in list(lump.items()):
        if isinstance(v, list) and v and v[0] == 'vector3m':
            lump[k] = ['vector3m'] + [move(p, PATH_SCALE if sentinel and k == 'path' else 1.0) for p in v[1:]]
        elif isinstance(v, list) and v and v[0] == 'movie-pos':
            p = v[1]
            lump[k] = ['movie-pos', move(p) + [round(p[3] + ROT, 3)]]
        elif isinstance(v, list) and v and v[0] == 'string':
            lump[k] = ['string'] + [s for s in v[1:] if s in names]
        elif isinstance(v, str) and k in ('next-actor', 'prev-actor', 'alt-actor', 'path-actor', 'nav-mesh-actor'):
            if v not in names:
                del lump[k]
        if k == 'custom-nav':
            lump[k] = ['symbol', 'pds-arena']
        if k == 'name':
            lump[k] = v.replace('valley-', 'pdsarena-')
    for k in ('next-actor', 'prev-actor', 'alt-actor', 'path-actor', 'nav-mesh-actor'):
        if isinstance(lump.get(k), str):
            lump[k] = lump[k].replace('valley-', 'pdsarena-')
    for k, v in list(lump.items()):
        if isinstance(v, list) and v and v[0] == 'string':
            lump[k] = ['string'] + [s.replace('valley-', 'pdsarena-') for s in v[1:]]
    out_actors.append(b)

# On Misty's silo side (about 24-105 degrees round, as polar() measures them) its structures stand inside the arena:
# from the ground ~39 m out, and its wooden docks 20-30 m over the floor reach in to 30 m. The barrier round the fight
# (wall and laser grid, below) comes in in front of them there; what of the fight was in them moves in front of them
# too (the sentinel's flight, the yellow gem's way up: the rectangle and the swinging pole to it, the lift's top;
# the blue gem's: a pole and a circle's top), its jumps kept about as they were.
def polar(p):
    """(degrees round from +z towards +x, metres out, metres over Misty's arena floor)"""
    return (math.degrees(math.atan2(p[0] - MISTY_C[0], p[2] - MISTY_C[2])) % 360,
            math.hypot(p[0] - MISTY_C[0], p[2] - MISTY_C[2]), p[1] - MISTY_C[1])


def at_polar(deg, r, h):
    a = math.radians(deg)
    return [round(MISTY_C[0] + r * math.sin(a), 3), round(MISTY_C[1] + h, 3), round(MISTY_C[2] + r * math.cos(a), 3)]


def turn_quat_by(q, deg):
    a = math.radians(deg) / 2
    x1, y1, z1, w1 = 0.0, math.sin(a), 0.0, math.cos(a)
    x2, y2, z2, w2 = q
    return [round(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, 4), round(w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, 4),
            round(w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, 4), round(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, 4)]


def move_actor(b, deg, r):
    """an actor (its spot, its path's points, its sphere) to deg round, r out; turned as much as it went round"""
    d0 = polar(b['trans'])[0]
    b['trans'] = at_polar(deg, r, polar(b['trans'])[2])
    b['bsphere'] = b['trans'] + [b['bsphere'][3]]
    b['quat'] = turn_quat_by(b['quat'], deg - d0)
    pth = b['lump'].get('path')
    if pth:
        b['lump']['path'] = ['vector3m'] + [at_polar(deg, r, polar(q)[2]) for q in pth[1:]]


def set_path_point(b, i, deg, r):
    pth = b['lump']['path']
    pth[1 + i] = at_polar(deg, r, polar(pth[1 + i])[2])


_by_name = {b['lump']['name']: b for b in out_actors}
# the yellow gem's (7 degrees): its rectangle from 37 to 18 degrees (and in, 28 m); the lift up to it tops at 35
# degrees (was 48); the pole swings to 18 degrees (was 31), over the rectangle as before
move_actor(_by_name['pdsarena-pds-plat-rectangle-05'], 18.0, 28.0)
set_path_point(_by_name['pdsarena-pds-plat-square-04'], 1, 35.0, 22.0)
set_path_point(_by_name['pdsarena-pds-swingpole-04'], 2, 18.0, 34.0)
# the blue gem's (127 degrees): the pole at 103 degrees in to 31.5 m (was 35.1), the circle rising to it tops 31 m
# out at 93 degrees (was 38.7 m, 91)
move_actor(_by_name['pdsarena-pds-swingpole-02'], 103.0, 31.5)
set_path_point(_by_name['pdsarena-pds-plat-circle-02'], 1, 93.0, 31.0)
# the sentinel's flight: its three points on that side in front of Misty's structures
_sp = _by_name['pdsarena-pds-sentinel-01']['lump']['path']
for _i in range(1, len(_sp)):
    _d, _r, _h = polar(_sp[_i])
    for _deg, _rmax in ((37, 38.0), (66, 33.0), (97, 42.0)):
        if abs(_d - _deg) < 2 and _r > _rmax:
            _sp[_i] = at_polar(_d, _rmax, _h)

level = {
    'long_name': 'pdsarena',
    'iso_name': 'PDSARENA',
    'nickname': 'pds',
    'gltf_file': 'custom_assets/jak1/levels/pdsarena/pdsarena.glb',
    'automatic_wall_detection': True,
    'automatic_wall_angle': 45.0,
    'double_sided_collide': False,
    'base_id': 95000,  # (free: Brume's 253 actors take 60000-60252, their saved states were ours)
    'art_groups': [],
    'custom_models': MODEL_NAMES,
    'textures': [['misty-vis-alpha']],  # (Misty's sky page: also what makes the builder read Misty's data below)
    # Misty's texture remap, sky and tpages (filled in from its bsp): without tpages the level logs in none and the
    # drawing crashes
    'tex_remap': 'misty',
    'sky': 'misty',
    'tpages': [],
    'ambients': [],
    'actors': out_actors,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'pdsarena.jsonc').write_text(
    '// Jak Living Sky: the PDS sentinel fight (from The Forgotten Lands, by Kuitar5, with permission) on Misty\'s arena.\n'
    '// Made by scripts/enhanced/build_pdsarena.py from the mod\'s valley.jsonc: edit there.\n' +
    json.dumps(level, indent=2), encoding='utf-8')

# villagep-obs: target-warp-in is there (as in each village's DGO), and this level takes the village's place beside
# Misty: without it Jak's warp-in state is left pointing into the unloaded village
gd = ['"villagep-obs.o"', '"pdsarena-obs.o"', '"pds-nav.o"', '"pds-boss-part.o"', '"pds-boss-setup.o"', '"pds-boss-extras.o"',
      '"pds-boss-states.o"', '"eichar-pole+0-ag.go"', '"redring-ag.go"'] + ['"%s-ag.go"' % m for m in MODEL_NAMES]
(OUT / 'pdsarena.gd').write_text(
    ';; Jak Living Sky: the PDS sentinel fight\'s level (build_pdsarena.py)\n'
    '("PDS.DGO"\n (\n  ' + '\n  '.join(gd + ['"pdsarena.go"']) + '\n  )\n )\n', encoding='utf-8')

# a tiny mesh 200 m under the arena: the level's own ground, never reached (Jak is in Misty)
cx, cy, cz = MISTY_C
V = np.array([[cx - 0.5, cy - 200, cz - 0.5], [cx + 0.5, cy - 200, cz - 0.5], [cx + 0.5, cy - 200, cz + 0.5],
              [cx - 0.5, cy - 200, cz + 0.5]], np.float32)
N = np.tile(np.array([0, 1, 0], np.float32), (4, 1))
C = np.full((4, 4), 0.5, np.float32)
I = np.array([0, 2, 1, 0, 3, 2], np.uint16)
data = bytearray()
views, accs = [], []
for arr, typ, ct, tgt in ((V, 'VEC3', 5126, 34962), (N, 'VEC3', 5126, 34962), (C, 'VEC4', 5126, 34962),
                          (I, 'SCALAR', 5123, 34963)):
    while len(data) % 4:
        data.append(0)
    views.append({'buffer': 0, 'byteOffset': len(data), 'byteLength': arr.nbytes, 'target': tgt})
    data.extend(arr.tobytes())
    a_ = {'bufferView': len(views) - 1, 'componentType': ct, 'count': len(arr) if typ != 'SCALAR' else arr.size,
          'type': typ}
    if typ == 'VEC3' and len(accs) == 0:
        a_['min'] = V.min(0).tolist()
        a_['max'] = V.max(0).tolist()
    accs.append(a_)
nodes = [{'name': 'pdsarena', 'mesh': 0}]
meshes = [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'COLOR_0': 2}, 'indices': 3}]}]
materials, textures, images, samplers = [], [], [], []

# the gems' stands: the valley's own "Gem-receptacle" (part of its ground, not an actor), moved and turned as the
# actors are, copied as it is (Draco-compressed, the builder reads it) with only its own data
vb = (SRC / 'custom_assets/jak1/levels/valley/valley.glb').read_bytes()
vn = struct.unpack('<I', vb[12:16])[0]
vj = json.loads(vb[20:20 + vn])
vbin = vb[20 + vn + 8:]
view_map, acc_map, tex_map, img_map, smp_map, mat_map = {}, {}, {}, {}, {}, {}


def copy_view(i):
    if i not in view_map:
        v = dict(vj['bufferViews'][i])
        chunk = vbin[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']]
        while len(data) % 4:
            data.append(0)
        v['byteOffset'] = len(data)
        v['buffer'] = 0
        data.extend(chunk)
        views.append(v)
        view_map[i] = len(views) - 1
    return view_map[i]


def copy_acc(i):
    if i not in acc_map:
        a = dict(vj['accessors'][i])
        if 'bufferView' in a:
            a['bufferView'] = copy_view(a['bufferView'])
        accs.append(a)
        acc_map[i] = len(accs) - 1
    return acc_map[i]


def copy_tex(i):
    if i not in tex_map:
        t = dict(vj['textures'][i])
        if 'source' in t:
            si = t['source']
            if si not in img_map:
                im = dict(vj['images'][si])
                if 'bufferView' in im:
                    im['bufferView'] = copy_view(im['bufferView'])
                images.append(im)
                img_map[si] = len(images) - 1
            t['source'] = img_map[si]
        if 'sampler' in t:
            sp = t['sampler']
            if sp not in smp_map:
                samplers.append(vj['samplers'][sp])
                smp_map[sp] = len(samplers) - 1
            t['sampler'] = smp_map[sp]
        textures.append(t)
        tex_map[i] = len(textures) - 1
    return tex_map[i]


def remap_tex_refs(o):
    if isinstance(o, dict):
        return {k: (copy_tex(v) if k == 'index' and isinstance(v, int) else remap_tex_refs(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [remap_tex_refs(x) for x in o]
    return o


def copy_mat(i):
    if i not in mat_map:
        # (no specular/envmap: that made it a TIE instance, which enemies' ground probes crashed on once it had no
        # collision; plain ground drawing instead)
        m = {k: v for k, v in vj['materials'][i].items() if k != 'extensions'}
        materials.append(remap_tex_refs(m))
        mat_map[i] = len(materials) - 1
    return mat_map[i]


# the mod's own stands are tripods whose legs reach ~70 m out, through Misty's walls (51 m): instead a straight column
# under each gem, in Misty's own Precursor metal (its arena's: from Misty's ground, as the decompiler extracts it,
# with their envmaps), a platform on top to climb onto
MISTY_GLB = ROOT.parent / 'features/jak1/mods/Jak Living Sky/jak-living-sky/data/decompiler_out/jak1/levels/misty/misty-background.glb'
if not MISTY_GLB.exists():
    MISTY_GLB = Path('C:/Users/aless/features/jak1/mods/Jak Living Sky/jak-living-sky/data/decompiler_out/jak1/levels/misty/misty-background.glb')
mb = MISTY_GLB.read_bytes()
mn = struct.unpack('<I', mb[12:16])[0]
mj = json.loads(mb[20:20 + mn])
mimg = {}


def import_misty_tex(i):
    t = dict(mj['textures'][i])
    si = t['source']
    if si not in mimg:
        im = dict(mj['images'][si])
        raw_png = base64.b64decode(im.pop('uri').split(',', 1)[1])
        while len(data) % 4:
            data.append(0)
        views.append({'buffer': 0, 'byteOffset': len(data), 'byteLength': len(raw_png)})
        data.extend(raw_png)
        im['bufferView'] = len(views) - 1
        im['mimeType'] = 'image/png'
        images.append(im)
        mimg[si] = len(images) - 1
    t['source'] = mimg[si]
    samplers.append({'magFilter': 9729, 'minFilter': 9729, 'wrapS': 10497, 'wrapT': 10497})
    t['sampler'] = len(samplers) - 1
    textures.append(t)
    return len(textures) - 1


def misty_mat(name):
    for m in mj['materials']:
        if m.get('name') == name:
            def refs(o):
                if isinstance(o, dict):
                    return {k: (import_misty_tex(v) if k == 'index' and isinstance(v, int) else refs(v)) for k, v in o.items()}
                if isinstance(o, list):
                    return [refs(x) for x in o]
                return o
            materials.append(refs(m))
            return len(materials) - 1
    raise SystemExit('no Misty material ' + name)


shaft_mat = misty_mat('cmn-precursor-metal-plain-01-envmap')
edge_mat = misty_mat('cmn-precursor-metal-edge-01-envmap')
top_mat = misty_mat('cmn-precursor-stonehengetop-envmap')

def add_mesh(name, verts, norms, uvs, idx, mat, extras=None):
    def put(arr, typ, ct, tgt, mm=False):
        while len(data) % 4:
            data.append(0)
        views.append({'buffer': 0, 'byteOffset': len(data), 'byteLength': arr.nbytes, 'target': tgt})
        data.extend(arr.tobytes())
        a_ = {'bufferView': len(views) - 1, 'componentType': ct, 'count': len(arr) if typ != 'SCALAR' else arr.size,
              'type': typ}
        if mm:
            a_['min'] = arr.min(0).tolist()
            a_['max'] = arr.max(0).tolist()
        accs.append(a_)
        return len(accs) - 1
    attrs = {'POSITION': put(np.array(verts, np.float32), 'VEC3', 5126, 34962, True),
             'NORMAL': put(np.array(norms, np.float32), 'VEC3', 5126, 34962)}
    if uvs is not None:
        attrs['TEXCOORD_0'] = put(np.array(uvs, np.float32), 'VEC2', 5126, 34962)
    attrs['COLOR_0'] = put(np.full((len(verts), 4), 0.5, np.float32), 'VEC4', 5126, 34962)
    prim = {'attributes': attrs, 'indices': put(np.array(idx, np.uint32), 'SCALAR', 5125, 34963)}
    if mat is not None:
        prim['material'] = mat
    meshes.append({'name': name, 'primitives': [prim]})
    nd_ = {'name': name, 'mesh': len(meshes) - 1}
    if extras:
        nd_['extras'] = extras
    nodes.append(nd_)


def lathe(cx, cz, rings, seg=20, outward=True):
    """a surface of revolution: rings = [(y, radius, v)], bottom to top"""
    V, Nn, U, I = [], [], [], []
    for (y, r, v) in rings:
        for k in range(seg + 1):
            ang = 2 * math.pi * k / seg
            dx, dz = math.sin(ang), math.cos(ang)
            V.append([cx + r * dx, y, cz + r * dz])
            Nn.append([dx if outward else -dx, 0.0, dz if outward else -dz])
            U.append([k / seg * 3.0, v])
    for i in range(len(rings) - 1):
        for k in range(seg):
            a0 = i * (seg + 1) + k
            b0 = a0 + seg + 1
            if outward:
                I += [a0, b0, a0 + 1, a0 + 1, b0, b0 + 1]
            else:
                I += [a0, a0 + 1, b0, a0 + 1, b0 + 1, b0]
    return V, Nn, U, I


for b_ in out_actors:
    if b_['etype'] != 'pds-gem':
        continue
    gx, gy, gz = b_['trans']
    top = gy - 4.48 - 0.4  # (the gem's bottom is 4.48 m under its origin: the platform just under it)
    floor = MISTY_C[1]
    # the shaft
    V, Nn, U, I = lathe(gx, gz, [(floor - 0.5, 1.0, 0.0), (top - 1.6, 1.0, (top - floor) / 4)])
    add_mesh('pds-gem-column', V, Nn, U, I, shaft_mat)
    # its flare up to the platform's edge, and the edge's rim (0.35 m up, round the top)
    V, Nn, U, I = lathe(gx, gz, [(top - 1.6, 1.0, 0.0), (top - 0.35, 3.0, 1.0), (top, 3.0, 1.35)])
    add_mesh('pds-gem-column-edge', V, Nn, U, I, edge_mat)
    # the platform's top, a little inside the rim (and the rim's inner side down to it)
    V, Nn, U, I = lathe(gx, gz, [(top, 3.0, 0.0), (top - 0.1, 2.75, 0.25)], outward=False)
    add_mesh('pds-gem-column-rim', V, Nn, U, I, edge_mat)
    seg = 20
    ty = top - 0.1
    V = [[gx, ty, gz]] + [[gx + 2.75 * math.sin(2 * math.pi * k / seg), ty, gz + 2.75 * math.cos(2 * math.pi * k / seg)]
                          for k in range(seg)]
    Nn = [[0.0, 1.0, 0.0]] * len(V)
    U = [[0.5, 0.5]] + [[0.5 + 0.5 * math.sin(2 * math.pi * k / seg), 0.5 + 0.5 * math.cos(2 * math.pi * k / seg)]
                        for k in range(seg)]
    I = []
    for k in range(seg):
        I += [0, 1 + (k + 1) % seg, 1 + k]
    add_mesh('pds-gem-column-top', V, Nn, U, I, top_mat)

# the floor under the traps (brown, the valley's own "Prec_mainSD_darker": what shows where a trap's tile squashes, not
# Misty's floor; it hides the platforms waiting below too): a whole hexagon 0.3 m under them (the valley's
# "PDS-trap-floor-under" is open in the middle), the traps' size and turn
dark_mat = None
for nd in vj['nodes']:
    if nd.get('name') == 'PDS-trap-floor-under':
        dark_mat = copy_mat(vj['meshes'][nd['mesh']]['primitives'][0]['material'])
fy = MISTY_C[1] + TRAP_LIFT - 0.3
hexv = [(-24.2884, -42.0688), (-48.5769, 0.0), (-24.2884, 42.0688), (24.2884, 42.0688), (48.5769, 0.0), (24.2884, -42.0688)]
V = [[MISTY_C[0], fy, MISTY_C[2]]]
for hx, hz in hexv:
    rx, rz = rot_xz(hx * 1.02, hz * 1.02, ROT)
    V.append([MISTY_C[0] + rx, fy, MISTY_C[2] + rz])
Nn = [[0.0, 1.0, 0.0]] * len(V)
U = [[(v[0] - MISTY_C[0]) / 8.0, (v[2] - MISTY_C[2]) / 8.0] for v in V]
I = []
for k in range(6):
    I += [0, 1 + (k + 1) % 6, 1 + k]
add_mesh('pds-floor-under', V, Nn, U, I, dark_mat, {'set_collision': 1, 'ignore': 1})

# the barrier round the fight: an invisible wall the fight is left only by winning or losing (from the platforms' and
# columns' tops Misty's structures could be reached) and a laser grid where it is. 48.5 m out (just behind the gems'
# columns' tops, 47.8 m; the merchant's gate, 52.7 m, outside), from under the ground up to 82 m over the floor;
# where Misty's structures are inside that above 15 m on its silo side (its docks) it comes in in front of them,
# 0.8 m off (lower down, and all round elsewhere, it stays the circle: through Misty's lower structures there and
# its two wooden masts):
# measured from Misty's background (as the decompiler extracts it) round in 3-degree slices, up in 2 m bands, each
# band's radius the least of the bands under it (so never a ledge out to stand on, only steps in, overhead).
# All the fight's enemies are born within 41 m, Jak arrives 45 m out.
B_R0, B_BIN, B_Y0, B_CELL, B_MARG = 48.5, 3.0, -1.0, 2.0, 0.8
B_FREE = 15.0  # (under it, the circle all round, through Misty's lower structures, as asked: the floor whole)
B_SIDE = (0.0, 135.0)  # (the silo's side only: elsewhere the circle whole, through Misty's two wooden masts)
B_NB, B_NC = int(round(360 / B_BIN)), 42  # (bands from 1 m under the floor to 83 m over it)


def _misty_points():
    mbin = mb[20 + mn + 8:]
    bary = np.array([[1 - u / 4 - v / 4, u / 4, v / 4] for u in range(5) for v in range(5 - u)], np.float32)
    out = []
    for nd_ in mj['nodes']:
        if 'mesh' not in nd_:
            continue
        T_ = np.array(nd_.get('translation', [0.0, 0.0, 0.0]), np.float32)
        for pr_ in mj['meshes'][nd_['mesh']]['primitives']:
            if 'indices' not in pr_:
                continue

            def get(i):
                a_ = mj['accessors'][i]
                v_ = mj['bufferViews'][a_['bufferView']]
                n_ = {'VEC3': 3, 'SCALAR': 1}[a_['type']]
                dt_ = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a_['componentType']]
                arr = np.frombuffer(mbin, dt_, count=a_['count'] * n_,
                                    offset=v_.get('byteOffset', 0) + a_.get('byteOffset', 0))
                return arr.reshape(-1, n_) if n_ > 1 else arr
            P_ = get(pr_['attributes']['POSITION']) + T_
            tri = P_[get(pr_['indices']).reshape(-1, 3)]
            dx, dz = tri[:, :, 0] - MISTY_C[0], tri[:, :, 2] - MISTY_C[2]
            near = (np.abs(dx).min(1) < 52) & (np.abs(dz).min(1) < 52)
            tri = tri[near]
            if not len(tri):
                continue
            S = np.einsum('kc,ncd->nkd', bary, tri).reshape(-1, 3)
            rr = np.hypot(S[:, 0] - MISTY_C[0], S[:, 2] - MISTY_C[2])
            hh = S[:, 1] - MISTY_C[1]
            keep = (rr > 25) & (rr < B_R0 + B_MARG + 1.6) & (hh > 0.3) & (hh < 90)
            out.append(S[keep])
    S = np.concatenate(out)
    return np.stack([np.degrees(np.arctan2(S[:, 0] - MISTY_C[0], S[:, 2] - MISTY_C[2])) % 360,
                     np.hypot(S[:, 0] - MISTY_C[0], S[:, 2] - MISTY_C[2]), S[:, 1] - MISTY_C[1]], 1)


_mp = _misty_points()
BR = np.full((B_NB, B_NC), B_R0)
for i in range(B_NB):
    dd = (_mp[:, 0] - (i + 0.5) * B_BIN + 180) % 360 - 180
    win = np.abs(dd) <= B_BIN / 2 + np.degrees(1.5 / _mp[:, 1])  # (and 1.5 m either side)
    for j in range(B_NC):
        lo = B_Y0 + B_CELL * j
        m = win & (_mp[:, 2] >= max(lo, B_FREE)) & (_mp[:, 2] < lo + B_CELL) & \
            (_mp[:, 0] >= B_SIDE[0]) & (_mp[:, 0] <= B_SIDE[1])
        if m.any():
            BR[i, j] = min(B_R0, float(_mp[m, 1].min()) - B_MARG)
    BR[i, 0] = min(BR[i, 0], BR[i, 1])  # (the band at the ground: what stands on it)
    for j in range(1, B_NC):
        BR[i, j] = min(BR[i, j], BR[i, j - 1])
BR = np.floor(BR * 2) / 2
print('barrier: in front of Misty at', sorted({round(i * B_BIN) for i in range(B_NB) if BR[i].min() < B_R0}), 'degrees')


def b_radius(deg, h):
    return BR[int((deg % 360) // B_BIN) % B_NB, int(min(B_NC - 1, max(0, (h - B_Y0) // B_CELL)))]


def _pt(deg, r, y):
    a = math.radians(deg)
    return [MISTY_C[0] + r * math.sin(a), MISTY_C[1] + y, MISTY_C[2] + r * math.cos(a)]


def _tan(deg):
    a = math.radians(deg)
    return [math.cos(a), 0.0, -math.sin(a)]


def _rad(deg):
    a = math.radians(deg)
    return [math.sin(a), 0.0, math.cos(a)]


# the wall (invisible, solid both ways): each slice's band faces, the steps' undersides, the sides between slices
V, Nn, I = [], [], []


def _quad(p0, p1, p2, p3):
    b_ = len(V)
    V.extend([p0, p1, p2, p3])
    Nn.extend([[0.0, 1.0, 0.0]] * 4)
    I.extend([b_, b_ + 1, b_ + 2, b_, b_ + 2, b_ + 3, b_, b_ + 2, b_ + 1, b_, b_ + 3, b_ + 2])


for i in range(B_NB):
    a0, a1 = i * B_BIN, (i + 1) * B_BIN
    for j in range(B_NC):
        y0_, y1_ = B_Y0 + B_CELL * j, B_Y0 + B_CELL * (j + 1)
        r_ = BR[i, j]
        _quad(_pt(a0, r_, y0_), _pt(a1, r_, y0_), _pt(a1, r_, y1_), _pt(a0, r_, y1_))
        if j + 1 < B_NC and BR[i, j + 1] < r_:
            r2 = BR[i, j + 1]
            _quad(_pt(a0, r2, y1_), _pt(a1, r2, y1_), _pt(a1, r_, y1_), _pt(a0, r_, y1_))
        rn = BR[(i + 1) % B_NB, j]
        if rn != r_:
            _quad(_pt(a1, min(r_, rn), y0_), _pt(a1, max(r_, rn), y0_), _pt(a1, max(r_, rn), y1_),
                  _pt(a1, min(r_, rn), y1_))
add_mesh('pds-arena-wall', V, Nn, None, I, None, {'set_invisible': 1})

# the laser grid on it: beams 2 m apart, each a thin strip with a glow across it (alpha-blended: the level's
# translucent geometry), upright ones and rings, round the steps too. Beams, not a grid texture (that one's fine
# lines shimmered, seen aslant or from afar)
LASER_CELL = 2.0
LASER_W = 0.3  # a beam's width (its glow; the white-hot core in the middle)
_w = 32
_x = np.abs((np.arange(_w, dtype=np.float32) + 0.5) / _w * 2.0 - 1.0)
_core = np.exp(-(_x / 0.18) ** 2)
_glow = np.exp(-(_x / 0.55) ** 2) * (1.0 - _x)
_img = np.zeros((4, _w, 4), np.float32)
_img[..., 0] = 255.0
_img[..., 1] = 50.0 + 205.0 * _core
_img[..., 2] = 40.0 + 215.0 * _core
_img[..., 3] = 255.0 * np.clip(0.75 * _glow + _core, 0.0, 1.0)
import io
from PIL import Image
_buf = io.BytesIO()
Image.fromarray(_img.round().astype(np.uint8), 'RGBA').save(_buf, 'PNG')
_png = _buf.getvalue()
while len(data) % 4:
    data.append(0)
views.append({'buffer': 0, 'byteOffset': len(data), 'byteLength': len(_png)})
data.extend(_png)
images.append({'name': 'pds-laser-beam', 'bufferView': len(views) - 1, 'mimeType': 'image/png'})
samplers.append({'magFilter': 9729, 'minFilter': 9729, 'wrapS': 33071, 'wrapT': 10497})
textures.append({'source': len(images) - 1, 'sampler': len(samplers) - 1})
materials.append({'name': 'pds-laser-beam', 'alphaMode': 'BLEND', 'doubleSided': True,
                  'pbrMetallicRoughness': {'baseColorTexture': {'index': len(textures) - 1}}})
laser_mat = len(materials) - 1
V, Nn, U, I = [], [], [], []


def _beam(p0, p1, side, nrm):
    """a strip from p0 to p1, LASER_W wide along side (a unit vector), facing nrm"""
    b_ = len(V)
    for p_, v_ in ((p0, 0.0), (p1, 1.0)):
        for s_, u_ in ((-0.5, 0.0), (0.5, 1.0)):
            V.append([p_[0] + side[0] * LASER_W * s_, p_[1] + side[1] * LASER_W * s_, p_[2] + side[2] * LASER_W * s_])
            Nn.append(nrm)
            U.append([u_, v_])
    I.extend([b_, b_ + 1, b_ + 2, b_ + 1, b_ + 3, b_ + 2])


_ytop = B_Y0 + B_CELL * B_NC - 1.0  # (82 m over the floor)
# upright beams round (at B_R0, 2 m apart): up each band, in along a step's underside where the barrier steps in
_n_up = int(round(2 * math.pi * B_R0 / LASER_CELL))
for k in range(_n_up):
    deg = 360.0 * k / _n_up
    i = int(deg // B_BIN)
    j0 = 0
    for j in range(B_NC):
        y1_ = min(_ytop, B_Y0 + B_CELL * (j + 1))
        if j + 1 == B_NC or BR[i, j + 1] != BR[i, j]:
            _beam(_pt(deg, BR[i, j], B_Y0 + B_CELL * j0), _pt(deg, BR[i, j], y1_), _tan(deg), [-c for c in _rad(deg)])
            if j + 1 < B_NC:
                _beam(_pt(deg, BR[i, j], y1_), _pt(deg, BR[i, j + 1], y1_), _tan(deg), [0.0, -1.0, 0.0])
            j0 = j + 1
# rings every 2 m from the floor up: round each slice's band, across the sides between slices
for jr in range(1, 42):
    y_ = float(LASER_CELL * jr)
    j = int((y_ - B_Y0) // B_CELL)
    for i in range(B_NB):
        a0, a1 = i * B_BIN, (i + 1) * B_BIN
        r_ = BR[i, j]
        _beam(_pt(a0, r_, y_), _pt(a1, r_, y_), [0.0, 1.0, 0.0], [-c for c in _rad((a0 + a1) / 2)])
        rn = BR[(i + 1) % B_NB, j]
        if rn != r_:
            _beam(_pt(a1, r_, y_), _pt(a1, rn, y_), [0.0, 1.0, 0.0], _tan(a1))
# the corners where the barrier steps round (between slices): upright beams up them
for i in range(B_NB):
    a1 = (i + 1) * B_BIN
    i2 = (i + 1) % B_NB
    for j in range(B_NC):
        if BR[i, j] != BR[i2, j]:
            y0_, y1_ = B_Y0 + B_CELL * j, min(_ytop, B_Y0 + B_CELL * (j + 1))
            for r_ in (BR[i, j], BR[i2, j]):
                _beam(_pt(a1, r_, y0_), _pt(a1, r_, y1_), _tan(a1), _tan(a1))
add_mesh('pds-laser-grid', V, Nn, U, I, laser_mat, {'set_collision': 1, 'ignore': 1})

# the fight's things, where they go, against it (half a metre to spare round each, and 2 m over a platform for Jak)
_ext = {'pds-plat-circle': 4.0, 'pds-plat-square': 5.0, 'pds-plat-rectangle': 6.5, 'pds-swingpole': 3.5}
for b_ in out_actors:
    e_ = _ext.get(b_['etype'])
    if e_ is None or 'path-actor' in b_['lump']:
        continue
    for q in b_['lump'].get('path', ['', b_['trans']])[1:]:
        d_, r_, h_ = polar(q)
        for da in np.linspace(-1.0, 1.0, 9) * math.degrees(e_ / max(r_, 1.0)):
            for dy in (0.0, 2.0):
                if b_radius(d_ + da, h_ + dy) < r_ + e_ * math.cos(math.radians(da)) + 0.5:
                    print('WARNING barrier: %s at %d degrees, %.1f m out, %.1f m up: the barrier at %.1f m' %
                          (b_['lump']['name'], d_ + da, r_, h_ + dy, b_radius(d_ + da, h_ + dy)))


# the traps' outer edge down to Misty's ground (the tiles looked afloat, 1.2 m over it), in the brown between the
# tiles (not the shiny envmapped metal: white glare): the
# tiles as drawn (measured from their models) end 42.9-43.9 m out at their top (29.0 m), 43.9-44.3 m at their bottom
# (28.78 m), corners at ROT + 173 - 60k degrees; the brown floor under them only reaches 42.9 m. So a flat ring just
# under the tiles' bottom from 42.9 m, where the floor under them ends (under them: no gap whichever way it's seen, nor when an outer tile drops; not
# solid, the traps work as before) to 44.6 m, then a slope 3 m out down into Misty's ground (27.65-27.88 m all round,
# measured from its background), walkable. (Distances: from the middle to the hexagon's sides.)
COS30 = math.cos(math.radians(30))
RING_Y = MISTY_C[1] + TRAP_LIFT - 0.26


def hex_band(ap0, y0, ap1, y1, u1):
    V, Nn, U, I = [], [], [], []
    for k in range(7):
        ang = math.radians(173.0 - 60.0 * (k % 6))
        for ap, y_, v_ in ((ap0, y0, 0.0), (ap1, y1, u1)):
            r_ = ap / COS30
            V.append([MISTY_C[0] + r_ * math.cos(ang), y_, MISTY_C[2] + r_ * math.sin(ang)])
            Nn.append([0.0, 1.0, 0.0])
            U.append([k * 13.0, v_])
    for k in range(6):
        a0 = 2 * k
        I += [a0, a0 + 2, a0 + 1, a0 + 1, a0 + 2, a0 + 3]
    # one face, up (two on top of each other flicker)
    e1, e2 = np.subtract(V[I[1]], V[I[0]]), np.subtract(V[I[2]], V[I[0]])
    if np.cross(e1, e2)[1] < 0:
        I = [I[t + 1] if t % 3 == 1 else (I[t - 1] if t % 3 == 2 else I[t]) for t in range(len(I))]
    return V, Nn, U, I


V, Nn, U, I = hex_band(42.9, RING_Y, 44.6, RING_Y, 0.5)
add_mesh('pds-floor-ring', V, Nn, U, I, dark_mat, {'set_collision': 1, 'ignore': 1})
V, Nn, U, I = hex_band(44.6, RING_Y, 44.6 + 3.0, MISTY_C[1] - 0.6, 1.0)
add_mesh('pds-floor-skirt', V, Nn, U, I, dark_mat)

while len(data) % 4:
    data.append(0)
g = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': list(range(len(nodes)))}], 'nodes': nodes,
     'meshes': meshes, 'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
if materials:
    g.update({'materials': materials, 'textures': textures, 'images': images})
    if samplers:
        g['samplers'] = samplers
if any('KHR_draco_mesh_compression' in pr.get('extensions', {}) for m_ in meshes for pr in m_['primitives']):
    g['extensionsUsed'] = ['KHR_draco_mesh_compression']
    g['extensionsRequired'] = ['KHR_draco_mesh_compression']
js = json.dumps(g).encode()
while len(js) % 4:
    js += b' '
(OUT / 'pdsarena.glb').write_bytes(b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) +
                                   struct.pack('<I4s', len(js), b'JSON') + js +
                                   struct.pack('<I4s', len(data), b'BIN\0') + bytes(data))

def add_align(b):
    """An 'align' bone first in the skin (our build-actor adds align AND prejoint when there's no align, the mod's
    builder only align: every joint would land one further on, the sentinel's parts on their neighbours' bones)."""
    n = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + n])
    rest = b[20 + n:]
    bn = struct.unpack('<I', rest[:4])[0]
    bin_ = bytearray(rest[8:8 + bn])
    if not j.get('skins'):
        return b
    skin = j['skins'][0]
    names = [j['nodes'][k].get('name') for k in skin['joints']]
    if names[0] == 'align':
        return b
    joints = skin['joints']
    children_of = {i: set(nd.get('children', [])) for i, nd in enumerate(j['nodes'])}
    roots = [k for k in joints if not any(k in children_of[o] for o in joints)]
    new = len(j['nodes'])
    j['nodes'].append({'name': 'align', 'children': roots})
    parent = [i for i, c in children_of.items() if set(roots) & c]
    for pi in parent:
        nd = j['nodes'][pi]
        nd['children'] = [c for c in nd['children'] if c not in roots] + [new]
    if not parent:
        for sc in j['scenes']:
            sc['nodes'] = [x for x in sc['nodes'] if x not in roots] + [new]
    skin['joints'] = [new] + joints

    def read(acc_i):
        a = j['accessors'][acc_i]
        v = j['bufferViews'][a['bufferView']]
        ncomp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
        dt = {5121: np.uint8, 5123: np.uint16, 5126: np.float32}[a['componentType']]
        isz = np.dtype(dt).itemsize * ncomp
        stride = v.get('byteStride', isz)
        off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        out = np.zeros((a['count'], ncomp), dt)
        for i in range(a['count']):
            out[i] = np.frombuffer(bytes(bin_[off + i * stride: off + i * stride + isz]), dt)
        return out

    def append(arr, like_i):
        while len(bin_) % 4:
            bin_.append(0)
        j['bufferViews'].append({'buffer': 0, 'byteOffset': len(bin_), 'byteLength': arr.nbytes})
        bin_.extend(arr.tobytes())
        a = dict(j['accessors'][like_i])
        a.pop('byteOffset', None)
        a['bufferView'] = len(j['bufferViews']) - 1
        a['count'] = len(arr)
        j['accessors'].append(a)
        return len(j['accessors']) - 1

    if 'inverseBindMatrices' in skin:
        ibm = read(skin['inverseBindMatrices'])
        ibm = np.vstack([np.eye(4, dtype=np.float32).reshape(1, 16), ibm])
        skin['inverseBindMatrices'] = append(ibm, skin['inverseBindMatrices'])
    meshes = {nd['mesh'] for nd in j['nodes'] if 'mesh' in nd and nd.get('skin') == 0}
    done = {}
    for mi in meshes:
        for prim in j['meshes'][mi]['primitives']:
            ai = prim['attributes'].get('JOINTS_0')
            if ai is None:
                continue
            if ai not in done:
                arr = read(ai)
                done[ai] = append((arr + 1).astype(arr.dtype), ai)
            prim['attributes']['JOINTS_0'] = done[ai]
    j['buffers'][0]['byteLength'] = len(bin_)
    js = json.dumps(j).encode()
    while len(js) % 4:
        js += b' '
    while len(bin_) % 4:
        bin_.append(0)
    return (b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(bin_)) + struct.pack('<I4s', len(js), b'JSON') + js
            + struct.pack('<I4s', len(bin_), b'BIN\0') + bytes(bin_))


for m in MODEL_NAMES:
    raw_glb = (SRC / 'custom_assets/jak1/models/custom_levels' / (m + '.glb')).read_bytes()
    (MODELS / (m + '.glb')).write_bytes(add_align(raw_glb))
print('ok', len(out_actors), 'actors; dropped', sorted(dropped))
