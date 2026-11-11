# Jak Living Sky: the merchant's JET-Board fight. From Kuitar5's The Forgotten Lands (used with the author's
# permission): its boss, the PDS sentinel, with its arena's actors, moved onto Misty's arena (docs/pds-arena-port.md).
# Writes the small extra level "pdsarena" (custom_assets/jak1/levels/pdsarena: jsonc with the actors, gd, a tiny glb
# far below so Jak is never inside it) and copies the boss models into custom_assets/jak1/models/custom_levels.
#
#   python scripts/enhanced/build_pdsarena.py <the-forgotten-lands checkout>
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
ROT = 52.0  # degrees about y: the outer traps off Misty's diagonal structures
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

level = {
    'long_name': 'pdsarena',
    'iso_name': 'PDSARENA',
    'nickname': 'pds',
    'gltf_file': 'custom_assets/jak1/levels/pdsarena/pdsarena.glb',
    'automatic_wall_detection': True,
    'automatic_wall_angle': 45.0,
    'double_sided_collide': False,
    'base_id': 60100,
    'art_groups': [],
    'custom_models': MODEL_NAMES,
    'textures': [],
    'ambients': [],
    'actors': out_actors,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'pdsarena.jsonc').write_text(
    '// Jak Living Sky: the PDS sentinel fight (from The Forgotten Lands, by Kuitar5, with permission) on Misty\'s arena.\n'
    '// Made by scripts/enhanced/build_pdsarena.py from the mod\'s valley.jsonc: edit there.\n' +
    json.dumps(level, indent=2), encoding='utf-8')

gd = ['"pdsarena-obs.o"', '"pds-nav.o"', '"pds-boss-part.o"', '"pds-boss-setup.o"', '"pds-boss-extras.o"',
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
while len(data) % 4:
    data.append(0)
g = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'name': 'pdsarena', 'mesh': 0}],
     'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'COLOR_0': 2}, 'indices': 3}]}],
     'buffers': [{'byteLength': len(data)}], 'bufferViews': views, 'accessors': accs}
js = json.dumps(g).encode()
while len(js) % 4:
    js += b' '
(OUT / 'pdsarena.glb').write_bytes(b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data)) +
                                   struct.pack('<I4s', len(js), b'JSON') + js +
                                   struct.pack('<I4s', len(data), b'BIN\0') + bytes(data))

for m in MODEL_NAMES:
    shutil.copy(SRC / 'custom_assets/jak1/models/custom_levels' / (m + '.glb'), MODELS / (m + '.glb'))
print('ok', len(out_actors), 'actors; dropped', sorted(dropped))
