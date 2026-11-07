"""fisher-intro: the fisherman's gestures with his lines, his face (brows, blinks), Jak listening, the shots on
them. The user's own shots stay as they are; the ones on the characters added here."""
import numpy as np

FISHER = 'v4-fisherman'
JAK = 'jak'
CAVE = np.array([-1583.0, 185.2, 288.5])      # the cavern's mouth (the user's shot at 30 s looks at it)
VILLAGE = np.array([-1815.0, 205.0, 395.0])   # (the shot at 6.5 s)
GATE = np.array([-2032.6, 183.0, 203.5])      # the warp gate in the sage's hut (10.68 s)

# the fisherman's face shapes (from the captures' face sheet; 0-based): 1 opens the mouth, 6 smiles,
# 8 frowns, 9 and 10 raise the brows, 3 and 14 close the eyes
MOUTH, SMILE, FROWN, BROW_A, BROW_B, LID_A, LID_B = 1, 6, 8, 9, 10, 3, 14


def PLAN(A, cap, scene):
    fs = A.Skeleton(cap / f'skeleton_{FISHER}.json')
    js = A.Skeleton(cap / f'skeleton_{JAK}.json')
    f = A.Author(FISHER, fs)
    jak_head = js.pos('neckB') + np.array([0, 0.25, 0])
    jak_chest = js.pos('neckB') - np.array([0, 0.35, 0])
    chest = 'chest'

    ident = np.array([0, 0, 0, 1.0])
    joints = ['Rshoulder', 'Relbow', 'Lshoulder', 'Lelbow', 'head', 'chest']

    def P(*parts):
        out = {j: ident for j in joints}
        for p in parts:
            out.update(p)
        return out

    # --- hands ---
    def r_to(right, up, fwd, at=chest):
        return f.arm('R', f.body(at, right, up, fwd))

    def l_to(right, up, fwd, at=chest):
        return f.arm('L', f.body(at, right, up, fwd))

    def r_point(target, reach=0.95):
        S = fs.pos('Rshoulder')
        d = target - S
        d = d / np.linalg.norm(d)
        a = np.linalg.norm(fs.pos('Relbow') - S) + np.linalg.norm(fs.pos('Rhand') - fs.pos('Relbow'))
        return f.arm('R', S + d * a * reach, pole=S - f.up + f.right * 0.3)

    def l_point(target, reach=0.95):
        S = fs.pos('Lshoulder')
        d = target - S
        d = d / np.linalg.norm(d)
        a = np.linalg.norm(fs.pos('Lelbow') - S) + np.linalg.norm(fs.pos('Lhand') - fs.pos('Lelbow'))
        return f.arm('L', S + d * a * reach, pole=S - f.up - f.right * 0.3)

    look_jak = f.look(jak_head)
    look_jak_nod = f.look(jak_head, extra=f.nod(10))
    look_cave = f.look(CAVE, amount=0.7)
    look_down = f.look(f.body('chest', 0.15, -1.2, 0.6), amount=0.8)

    talk_a = merge2(r_to(0.30, -0.12, 0.38), l_to(-0.30, -0.10, 0.36))
    talk_b = merge2(r_to(0.26, -0.02, 0.42), l_to(-0.33, -0.16, 0.33))
    talk_c = merge2(r_to(0.34, -0.18, 0.34), l_to(-0.26, -0.04, 0.42))
    wave_1 = r_to(0.42, 0.42, 0.22, at='Rshoulder')
    wave_2 = r_to(0.30, 0.46, 0.26, at='Rshoulder')
    open_arms = merge2(r_to(0.55, -0.05, 0.35, at='Rshoulder'), l_to(-0.55, -0.05, 0.35, at='Lshoulder'))
    shrug = merge2(r_to(0.42, -0.30, 0.30, at='Rshoulder'), l_to(-0.42, -0.30, 0.30, at='Lshoulder'))
    hand_chest_l = l_to(0.02, 0.05, 0.20)
    fist_r = r_to(0.16, 0.05, 0.40)
    fist_r_down = r_to(0.17, -0.10, 0.42)
    basket_l = l_point(f.body('chest', -0.5, -1.3, 0.5), reach=0.85)

    K = []

    def key(t, *parts):
        K.extend(A.pose_keys(FISHER, t, P(*parts)))

    # line 1 (0.2 - 22.5): who he is, the empty village, the warp gate, here you are
    key(0.0, look_jak)
    key(0.25, look_jak)
    key(0.75, wave_1, look_jak)          # "EHI, RAGAZZI!"
    key(1.05, wave_2, look_jak)
    key(1.35, wave_1, look_jak)
    key(1.65, wave_2, look_jak)
    key(1.95, wave_1, look_jak)
    key(2.5, talk_a, look_jak)           # "HO CAPTATO UN SEGNALE..."
    key(3.3, talk_b, look_jak_nod)
    key(4.2, talk_a, look_jak)
    key(5.1, talk_c, look_jak_nod)
    key(6.0, talk_a, look_jak)
    key(7.2, l_point(VILLAGE, 0.9), talk_a_r(talk_a), f.look(VILLAGE, amount=0.6))   # the village
    key(9.4, talk_b, look_jak)
    key(11.6, l_point(GATE, 0.9), talk_a_r(talk_a), f.look(GATE, amount=0.6))       # the warp gate
    key(14.4, talk_c, look_jak)
    key(15.6, talk_a, look_jak_nod)
    key(16.8, talk_b, look_jak)
    key(17.7, talk_a, look_jak)          # "...SPERANDO CHE ARRIVASSE QUALCUNO..."
    key(18.7, talk_c, look_jak_nod)
    key(19.6, talk_b, look_jak)
    key(20.5, open_arms, look_jak)       # "...ECCOVI QUI!"
    key(22.2, open_arms, look_jak)
    key(23.0, talk_a, look_jak)
    # line 2 (23.09 - 52.9)
    key(24.3, r_point(CAVE), talk_l(talk_a), look_cave)     # "...SCENDERE NELLE CAVERNE"
    key(26.8, r_point(CAVE), talk_l(talk_a), look_cave)
    key(27.9, talk_b, look_jak)
    key(29.0, r_point(CAVE), talk_l(talk_b), look_cave)     # "...I FIUMI: NASCONO PROPRIO DA QUELLE CAVERNE"
    key(31.6, r_point(CAVE), talk_l(talk_b), look_cave)
    key(32.6, talk_a, look_jak)
    key(33.6, basket_l, talk_r(talk_a), look_down)          # "L'ACQUA E' TORBIDA, VIOLA, E I MIEI PESCI..."
    key(34.4, basket_l, talk_r(talk_a), f.look(f.body('chest', 0.15, -1.2, 0.6), amount=0.8, extra=f.shake(8)))
    key(35.1, basket_l, talk_r(talk_a), f.look(f.body('chest', 0.15, -1.2, 0.6), amount=0.8, extra=f.shake(-8)))
    key(35.8, basket_l, talk_r(talk_a), look_down)
    key(37.0, talk_c, look_jak)
    key(38.5, fist_r, talk_l(talk_a), look_jak)              # "E' ECO OSCURO,
    key(38.95, fist_r_down, talk_l(talk_a), look_jak_nod)    #  NE SONO CERTO!"
    key(39.5, fist_r, talk_l(talk_a), look_jak)
    key(40.0, fist_r_down, talk_l(talk_a), look_jak_nod)
    key(40.9, hand_chest_l, talk_r(talk_a), look_jak)        # "CI ANDREI IO A VEDERE,
    key(42.0, hand_chest_l, talk_r(talk_b), look_jak)
    key(42.7, shrug, f.look(jak_head, extra=f.shake(6)))     #  MA QUELLE CAVERNE PULLULANO DI LURKER!"
    key(44.2, shrug, look_jak)
    key(45.1, r_point(jak_chest, 0.9), talk_l(talk_a), look_jak)   # "GIA' CHE SCENDI LAGGIU',
    key(46.3, r_point(jak_chest, 0.9), talk_l(talk_a), look_jak)
    key(46.9, r_point(CAVE), talk_l(talk_a), look_cave)            #  SCOPRI DA DOVE ARRIVA QUELL'ECO OSCURO!"
    key(48.2, talk_b, look_jak)
    key(48.8, talk_a, look_jak_nod)      # "SE RIUSCIRETE NELL'IMPRESA
    key(49.7, talk_c, look_jak)
    key(50.4, hand_chest_l, talk_r(talk_a), look_jak_nod)   #  VI RICOMPENSERO' SICURAMENTE
    key(51.6, open_arms, look_jak)                          #  CON UNA BATTERIA!"
    key(52.8, talk_a, look_jak)
    key(53.25, look_jak)
    scene['poses'] = K

    # Jak: listening, his head to the fisherman, a nod at "ECCOVI QUI" and at the end
    if js.has('neckB'):
        j = A.Author(JAK, js, face_fwd=jak_fwd(scene, js))
        fhead = fs.pos('head')
        jl = j.look(fhead, joint='neckB', amount=0.8, max_deg=30, chest_share=0)
        jn = j.look(fhead, joint='neckB', amount=0.8, max_deg=30, chest_share=0, extra=j.nod(12))
        for t, p in [(0.0, jl), (20.8, jl), (21.15, jn), (21.5, jl), (52.2, jl), (52.55, jn), (53.0, jl)]:
            scene['poses'] += A.pose_keys(JAK, t, p)

    # the face: brows with what he says, blinks every few seconds, the smile; the lips from the voice
    brows = [(0.0, {}), (0.3, {BROW_A: 0.8, BROW_B: 0.6}), (2.2, {}), (20.4, {SMILE: 0.9, BROW_A: 0.4}),
             (22.6, {}), (32.9, {FROWN: 0.8}), (37.0, {FROWN: 0.4}), (38.4, {FROWN: 1.0}), (40.6, {}),
             (42.6, {BROW_A: 0.7, BROW_B: 0.7}), (44.4, {FROWN: 0.5}), (48.5, {}), (50.3, {SMILE: 0.8}),
             (53.2, {SMILE: 0.4})]
    blinks = [1.9, 5.4, 8.6, 12.9, 16.2, 19.4, 24.0, 27.6, 31.2, 34.0, 37.4, 41.8, 46.0, 49.3, 52.4]
    times = sorted({t for t, _ in brows} | {b + d for b in blinks for d in (0.0, 0.07, 0.17)})

    def brow_at(t):
        cur = {}
        for bt, w in brows:
            if bt <= t + 1e-6:
                cur = w
        return cur

    faces = []
    for t in times:
        w = [0.0] * 16
        for k, v in brow_at(t).items():
            w[k] = v
        for b in blinks:
            if abs(t - (b + 0.07)) < 1e-6:
                w[LID_A] = 1.0
                w[LID_B] = 1.0
        faces.append({"t": round(t, 3), "actor": FISHER, "w": [round(x, 3) for x in w], "smooth": True})
    scene['faces'] = faces
    scene['mouths'] = {FISHER: {"shape": MOUTH, "gain": 1.0}}
    for l in scene['lines']:
        l['actor'] = FISHER

    # the shots on the characters (the user's shots kept)
    H = fs.pos('head')
    fwd, right, up = f.fwd, f.right, f.up
    rs = fs.pos('Rshoulder')
    react = shot(rs + right * 1.0 - fwd * 0.8 + up * 0.1, jak_head + up * 0.1)
    close = shot(H + fwd * 2.3 + right * 0.5 - up * 0.25, H - up * 0.12)
    medium = shot(H + fwd * 3.2 + right * 0.8 - up * 0.45, H - up * 0.4)
    two = next(s for s in scene['shots'] if abs(s['t'] - 32.77) < 0.05)
    user = [s for s in scene['shots'] if round(s['t'], 2) in (0.0, 6.5, 10.68, 17.55, 23.09, 30.09, 32.77)]
    mine = [dict(react, t=20.6), dict(close, t=38.39), dict(two, t=40.75), dict(medium, t=48.51),
            dict(react, t=51.6)]
    scene['shots'] = sorted(user + mine, key=lambda s: s['t'])
    scene['anims'] = []


def merge2(a, b):
    out = dict(a)
    out.update(b)
    return out


def talk_a_r(talk):
    return {k: v for k, v in talk.items() if k.startswith('R')}


def talk_l(talk):
    return {k: v for k, v in talk.items() if k.startswith('L')}


def talk_r(talk):
    return {k: v for k, v in talk.items() if k.startswith('R')}


def shot(pos, look):
    return {"pos": [round(float(x), 3) for x in pos], "look": [round(float(x), 3) for x in look], "smooth": False}


def jak_fwd(scene, js):
    import math
    for m in scene.get('moves', []):
        if m['actor'] == 'jak':
            y = math.radians(m['yaw'])
            return np.array([math.sin(y), 0, math.cos(y)])
    return np.array([0, 0, 1.0])
