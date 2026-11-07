"""Jak Living Sky: gestures for a scene made in the game's cutscene editor, worked out from the characters'
skeletons (the editor's "Capture for Claude": captures/<scene>/skeleton_<actor>.json, world matrices in game
units, rows: the joint's x, y, z axes then its place).

A gesture is a place a hand goes to (two-bone IK: shoulder, elbow), a way the head looks, a turn of the chest;
each becomes, for each joint, a turn in the joint's own space (a quaternion, x y z w: the game multiplies a
joint's animated turn by it, so it acts in the joint's space), keyed on the scene's timeline and sliding from
one key to the next. The game's quaternions are the usual ones (quaternion->matrix: a 90 degree turn round z
takes x to y; a matrix's rows are the turned axes).

  python cutscene_author.py <captures dir> <scene.json> <plan.py> [more scene.json copies]

The plan (a Python file) has PLAN(a) -> scene changes, `a` being this module's Author for that skeleton.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

M = 4096.0


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qnorm(q):
    return q / np.linalg.norm(q)


def qaxis(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return np.array([*(axis * math.sin(h)), math.cos(h)])


def qarc(u, v, amount=1.0, max_deg=180.0):
    """the turn taking direction u to v (the short way), a part of it, no more than max_deg"""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    ang = math.degrees(math.atan2(s, float(np.dot(u, v))))
    if s < 1e-6:
        return np.array([0, 0, 0, 1.0])
    return qaxis(axis / s, min(ang * amount, max_deg))


def qrot(q, v):
    p = np.array([*v, 0.0])
    return qmul(qmul(q, p), qconj(q))[:3]


def q_from_rows(m3):
    """a joint's turn from its world matrix's rows (its axes)"""
    r = np.array(m3).T  # columns: where x, y, z go
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return qnorm(np.array(q))


class Skeleton:
    def __init__(self, path):
        self.joints = json.loads(Path(path).read_text())
        self.by = {j['name']: j for j in self.joints}

    def has(self, name):
        return name in self.by and any(self.by[name]['world'][12:15])

    def pos(self, name):
        w = self.by[name]['world']
        return np.array([w[12], w[13], w[14]]) / M

    def rows(self, name):
        w = self.by[name]['world']
        out = []
        for k in range(3):
            v = np.array(w[k * 4:k * 4 + 3])
            out.append(v / np.linalg.norm(v))
        return out

    def turn(self, name):
        return q_from_rows(self.rows(name))


class Author:
    """one character: its skeleton, which way it looks; poses: joint -> local quaternion"""

    def __init__(self, actor, skel, face_fwd=None):
        self.actor = actor
        self.s = skel
        self.up = np.array([0, 1.0, 0])
        if face_fwd is None:
            eyes = [n for n in ('Leye', 'Reye', 'eyeL', 'eyeR') if skel.has(n)]
            head = 'head' if skel.has('head') else 'neckB'
            e = np.mean([skel.pos(n) for n in eyes], axis=0)
            d = e - skel.pos(head)
            d[1] = 0
            face_fwd = d / np.linalg.norm(d)
        self.fwd = np.asarray(face_fwd, float)
        self.right = np.cross(self.fwd, self.up)
        self.right /= np.linalg.norm(self.right)

    def body(self, at, right=0.0, up=0.0, fwd=0.0):
        """a point in the body's space from a joint: so far right, up, forward (metres)"""
        return self.s.pos(at) + self.right * right + self.up * up + self.fwd * fwd

    def local(self, joint, world_turn, before=None):
        """a world turn applied at a joint -> its own-space quaternion (`before`: what its parents turned)"""
        qj = self.s.turn(joint)
        if before is not None:
            qj = qmul(before, qj)
        return qnorm(qmul(qmul(qconj(qj), world_turn), qj))

    def arm(self, side, target, pole=None):
        """two-bone IK: the hand (wrist) to target; elbow bending its own way (or towards pole)"""
        sj, ej, hj = f'{side}shoulder', f'{side}elbow', f'{side}hand'
        S, E, H = self.s.pos(sj), self.s.pos(ej), self.s.pos(hj)
        a, b = np.linalg.norm(E - S), np.linalg.norm(H - E)
        T = np.asarray(target, float)
        d = T - S
        dist = float(np.clip(np.linalg.norm(d), abs(a - b) + 1e-3, (a + b) * 0.995))
        dn = d / np.linalg.norm(d)
        # the bend's side: where the elbow points now (or the pole), off the shoulder-to-target line
        p = (np.asarray(pole, float) - S) if pole is not None else (E - S)
        p = p - dn * np.dot(p, dn)
        if np.linalg.norm(p) < 1e-5:
            p = -self.up - dn * np.dot(-self.up, dn)
        p /= np.linalg.norm(p)
        cos_a = (a * a + dist * dist - b * b) / (2 * a * dist)
        alpha = math.acos(max(-1.0, min(1.0, cos_a)))
        E2 = S + a * (math.cos(alpha) * dn + math.sin(alpha) * p)
        H2 = S + dn * dist
        r1 = qarc(E - S, E2 - S)
        fore = qrot(r1, H - E)
        r2 = qarc(fore, H2 - E2)
        return {sj: self.local(sj, r1), ej: self.local(ej, r2, before=r1)}

    def look(self, target, joint='head', amount=0.85, max_deg=40.0, chest_share=0.25, extra=None):
        """the head (and a little of the chest) turned to look at target; extra: a further world turn"""
        H = self.s.pos(joint)
        want = np.asarray(target, float) - H
        arc = qarc(self.fwd_of(joint), want, amount, max_deg)
        if extra is not None:
            arc = qmul(extra, arc)
        out = {}
        if chest_share > 0 and self.s.has('chest'):
            # (the chest takes a part of it: a quaternion's part, by its angle)
            ang = 2 * math.degrees(math.acos(min(1.0, abs(arc[3]))))
            ax = arc[:3] / (np.linalg.norm(arc[:3]) or 1)
            c = qaxis(ax, ang * chest_share) if ang > 0.01 else np.array([0, 0, 0, 1.0])
            out['chest'] = self.local('chest', c)
            rest = qmul(arc, qconj(c))
            out[joint] = self.local(joint, rest, before=c)
        else:
            out[joint] = self.local(joint, arc)
        return out

    def fwd_of(self, joint):
        return self.fwd

    def nod(self, deg):
        """a world turn: the head down (positive) round the body's right"""
        return qaxis(self.right, deg)

    def shake(self, deg):
        return qaxis(self.up, deg)


def pose_keys(actor, t, pose, smooth=True):
    keys = []
    for joint, q in pose.items():
        keys.append({"t": round(float(t), 3), "actor": actor, "joint": joint, "rot": [0, 0, 0], "off": [0, 0, 0],
                     "smooth": smooth, "q": [round(float(x), 5) for x in q]})
    return keys


def merge(*poses):
    out = {}
    for p in poses:
        out.update(p)
    return out


def main():
    cap = Path(sys.argv[1])
    scene_path = Path(sys.argv[2])
    spec = importlib.util.spec_from_file_location('plan', sys.argv[3])
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    plan.PLAN(sys.modules[__name__], cap, scene)
    text = json.dumps(scene, indent=2, ensure_ascii=False)
    for p in [scene_path] + [Path(x) for x in sys.argv[4:]]:
        p.write_text(text, encoding='utf-8')
    print('poses', len(scene.get('poses', [])), 'faces', len(scene.get('faces', [])), 'shots', len(scene['shots']))


if __name__ == '__main__':
    main()
