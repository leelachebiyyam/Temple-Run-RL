#!/usr/bin/env python3
"""Export the Temple Run 2 runner ("Guy Dangerous") for raylib.

Source of truth (shipped data, nothing invented):

    recon/unity_assets/allbundles/assets/Bundles/20_03/Characters/0/
        Guy_Dangerous/0/characterdata_guy_dangerous.unity3d
        _characterdata_shared.unity3d

The guy bundle holds one ``Mesh`` (``GuyDangerous``, 1351 verts / 1930 tris,
25-bone skin with per-vertex indices+weights), four ``Material``s, three
``Texture2D``s, 135 ``Transform``/``GameObject`` pairs (the bone hierarchy of
the prefab ``guy_dangerous`` plus two more baked prefab instances) and two
legacy ``AnimationClip``s (``Run01``, ``Idle01``).  The shared bundle holds the
remaining nine clips used by every runner (``Jump01``, ``Slide01``,
``SlideEnter01``, ``RunStumble01``, ``Death*``, ``ZiplineIdle01``).

Outputs (see ``resources/templerun2/character/README.md`` for the contract):

    resources/templerun2/character/guy.glb          skinned mesh + skeleton + clips
    resources/templerun2/character/guy.obj          static fallback mesh
    resources/templerun2/character/guy.mtl          static fallback material (map_Kd guy.png)
    resources/templerun2/character/guy.png          albedo (1024x1024, from the bundle)
    resources/templerun2/character/animations.json  clip + bone catalogue
    resources/templerun2/character/README.md        conventions / provenance

Conventions
-----------
* Vertices are Unity coordinates with **X negated** (the standard left-handed
  Unity -> right-handed glTF/raylib conversion; UnityPy's OBJ writer does the
  same).  Winding is reversed so front faces stay front (CCW).  Bone locals are
  converted with the same ``S*L*S`` mirror: translation ``-x``, rotation
  ``(x, -y, -z, w)``, scale unchanged.
* Y is up, +Z is the character's forward (measured from the shipped mesh, not
  assumed: the nose is the head's +Z extreme at z = +0.123 and the toes are the
  feet's +Z extreme at z = +0.132, against back-of-head z = -0.084).  Feet are at
  y = 0, the mesh origin is the prefab root ``guy_dangerous``; units are Unity
  metres (the mesh is 1.24 m tall).
* glTF ``TEXCOORD_0`` is written bottom-up (``v' = 1 - v``); the OBJ fallback
  keeps Unity's ``vt`` values verbatim.
* ``animations.json`` sets ``yaw_offset_deg = 0`` so the runner faces the track's
  ``forward = (sin yaw, cos yaw)`` (the shipped props/track pieces are
  ``-Z``-forward and use 180).

Animation: every shipped legacy clip is decoded - bone rotations come from
``m_CompressedRotationCurves`` (AssetStudio's ``PackedQuatVector.UnpackQuats``;
times are centisecond deltas) and positions/scales from the plain
``AnimationCurve`` keyframes - resampled onto the clip's own ``m_SampleRate``
grid and written into ``guy.glb`` as a glTF animation (LINEAR samplers).  No
motion is synthesised.

Usage:
    recon/unity_assets/.venv/bin/python tools/export_character.py
"""
from __future__ import annotations

import argparse
import io
import json
import math
import struct
import sys
from pathlib import Path

import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLES = ROOT / "recon/unity_assets/allbundles"
DEFAULT_OUT = ROOT / "resources/templerun2/character"

CHAR_DIR = "assets/Bundles/20_03/Characters/0"
GUY_BUNDLE = CHAR_DIR + "/Guy_Dangerous/0/characterdata_guy_dangerous.unity3d"
SHARED_BUNDLE = CHAR_DIR + "/_characterdata_shared.unity3d"

PREFAB_ROOT = "guy_dangerous"
MESH_NAME = "GuyDangerous"
MATERIAL_NAME = "guy_dangerous"
DEFAULT_FPS = 30.0

# clip object name -> state-ish name used by the runtime
CLIP_NAMES = {
    "Run01": "run",
    "Jump01": "jump",
    "Slide01": "slide",
    "SlideEnter01": "slide_enter",
    "Idle01": "idle",
    "RunStumble01": "stumble",
    "Death01": "death",
    "DeathFalling01": "death_falling",
    "DeathWaterfall01": "death_waterfall",
    "DeathCartHead01": "death_cart_head",
    "ZiplineIdle01": "zipline_idle",
}
# which clip is the usable one for each gameplay state (first existing wins)
CLIP_ORDER = ["Run01", "Idle01", "Jump01", "Slide01", "SlideEnter01",
              "RunStumble01", "Death01", "DeathFalling01", "DeathWaterfall01",
              "DeathCartHead01", "ZiplineIdle01"]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def fmt(x: float) -> str:
    if not math.isfinite(x):
        x = 0.0
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def quat_to_rows(x: float, y: float, z: float, w: float):
    """Row-major 3x3 rotation matrix from a Unity quaternion (x, y, z, w)."""
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)),
        (2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)),
        (2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)),
    )


IDENTITY4 = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def mat_mul(a, b):
    return tuple(tuple(sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4))
                 for r in range(4))


def mat_invert(m):
    """General 4x4 inverse (row-major, column-vector convention)."""
    a = [list(row) for row in m]
    inv = [list(row) for row in IDENTITY4]
    for i in range(4):
        pivot = max(range(i, 4), key=lambda r: abs(a[r][i]))
        if abs(a[pivot][i]) < 1e-12:
            raise ValueError("singular matrix")
        a[i], a[pivot] = a[pivot], a[i]
        inv[i], inv[pivot] = inv[pivot], inv[i]
        d = a[i][i]
        a[i] = [v / d for v in a[i]]
        inv[i] = [v / d for v in inv[i]]
        for r in range(4):
            if r == i:
                continue
            f = a[r][i]
            if f:
                a[r] = [a[r][c] - f * a[i][c] for c in range(4)]
                inv[r] = [inv[r][c] - f * inv[i][c] for c in range(4)]
    return tuple(tuple(row) for row in inv)


def xform(m, v):
    x, y, z = v
    return (m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3])


def unity_matrix4x4(mat) -> tuple:
    """UnityPy Matrix4x4f -> row-major 4x4 tuple."""
    return (
        (mat.e00, mat.e01, mat.e02, mat.e03),
        (mat.e10, mat.e11, mat.e12, mat.e13),
        (mat.e20, mat.e21, mat.e22, mat.e23),
        (mat.e30, mat.e31, mat.e32, mat.e33),
    )


def mirror_x(m):
    """S * M * S for S = diag(-1, 1, 1, 1)."""
    s = (-1.0, 1.0, 1.0, 1.0)
    return tuple(tuple(s[r] * m[r][c] * s[c] for c in range(4)) for r in range(4))


def decompose(m):
    """Row-major TRS matrix -> (translation, quaternion xyzw, scale)."""
    t = (m[0][3], m[1][3], m[2][3])
    cols = [(m[0][i], m[1][i], m[2][i]) for i in range(3)]
    sx = math.sqrt(sum(v * v for v in cols[0])) or 1.0
    sy = math.sqrt(sum(v * v for v in cols[1])) or 1.0
    sz = math.sqrt(sum(v * v for v in cols[2])) or 1.0
    r = [[cols[c][r] / (sx, sy, sz)[c] for c in range(3)] for r in range(3)]
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0:
        s_ = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s_
        x = (r[2][1] - r[1][2]) / s_
        y = (r[0][2] - r[2][0]) / s_
        z = (r[1][0] - r[0][1]) / s_
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s_ = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        w = (r[2][1] - r[1][2]) / s_
        x = 0.25 * s_
        y = (r[0][1] + r[1][0]) / s_
        z = (r[0][2] + r[2][0]) / s_
    elif r[1][1] > r[2][2]:
        s_ = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        w = (r[0][2] - r[2][0]) / s_
        x = (r[0][1] + r[1][0]) / s_
        y = 0.25 * s_
        z = (r[1][2] + r[2][1]) / s_
    else:
        s_ = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        w = (r[1][0] - r[0][1]) / s_
        x = (r[0][2] + r[2][0]) / s_
        y = (r[1][2] + r[2][1]) / s_
        z = 0.25 * s_
    return t, (x, y, z, w), (sx, sy, sz)


# --------------------------------------------------------------------------- #
# legacy compressed-curve decoding
#
# Unity's legacy AnimationClip stores bone rotations in CompressedAnimationCurve:
#   m_Path   : bone path, e.g. "Bone Pelvis/Bone Spine"
#   m_Times  : PackedIntVector of *delta* times in centiseconds (prefix-summed)
#   m_Values : PackedQuatVector, 3 flag bits + 3 x (9 or 10) bits per quaternion
#   m_Slopes : PackedFloatVector, 4 slope components per key
# This mirrors AssetStudio's
# AssetStudio/Classes/AnimationClip.cs (PackedQuatVector.UnpackQuats) and
# AssetStudioUtility/ModelConverter.cs (times[i] = prefix(data) * 0.01f).
# --------------------------------------------------------------------------- #
def _unpack_ints(pv):
    """PackedIntVector / PackedBitVector -> list[int] (AssetStudio UnpackInts)."""
    bitsize = pv.m_BitSize
    data = pv.m_Data
    n = pv.m_NumItems
    out = [0] * n
    index_pos = 0
    bit_pos = 0
    for i in range(n):
        bits = 0
        value = 0
        while bits < bitsize:
            value |= (data[index_pos] >> bit_pos) << bits
            num = min(bitsize - bits, 8 - bit_pos)
            bit_pos += num
            bits += num
            if bit_pos == 8:
                index_pos += 1
                bit_pos = 0
        out[i] = value & ((1 << bitsize) - 1)
    return out


def _unpack_quats(pv):
    """PackedQuatVector -> list[(x, y, z, w)] (AssetStudio UnpackQuats)."""
    data = pv.m_Data
    out = []
    index_pos = 0
    bit_pos = 0
    for _ in range(pv.m_NumItems):
        flags = 0
        bits = 0
        while bits < 3:
            flags |= (data[index_pos] >> bit_pos) << bits
            num = min(3 - bits, 8 - bit_pos)
            bit_pos += num
            bits += num
            if bit_pos == 8:
                index_pos += 1
                bit_pos = 0
        flags &= 7

        q = [0.0, 0.0, 0.0, 0.0]
        total = 0.0
        for j in range(4):
            if (flags & 3) != j:
                bit_size = 9 if ((flags & 3) + 1) % 4 == j else 10
                x = 0
                bits = 0
                while bits < bit_size:
                    x |= (data[index_pos] >> bit_pos) << bits
                    num = min(bit_size - bits, 8 - bit_pos)
                    bit_pos += num
                    bits += num
                    if bit_pos == 8:
                        index_pos += 1
                        bit_pos = 0
                x &= (1 << bit_size) - 1
                q[j] = x / (0.5 * ((1 << bit_size) - 1)) - 1.0
                total += q[j] * q[j]

        last = flags & 3
        q[last] = math.sqrt(max(0.0, 1.0 - total))
        if flags & 4:
            q[last] = -q[last]
        out.append((q[0], q[1], q[2], q[3]))
    return out


# --------------------------------------------------------------------------- #
# unit-space conversions and curve evaluation
#
# The export frame is Unity with X negated, so a Unity local transform L maps to
# S*L*S with S = diag(-1, 1, 1):
#   translation (x, y, z) -> (-x, y, z)
#   rotation   (x, y, z, w) -> (x, -y, -z, w)   (same as AssetStudio's FBX path)
#   scale      unchanged
# --------------------------------------------------------------------------- #
def mirror_translation(v):
    return (-v[0], v[1], v[2])


def mirror_rotation(q):
    return (q[0], -q[1], -q[2], q[3])


def quat_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_neg(a):
    return (-a[0], -a[1], -a[2], -a[3])


def slerp(a, b, t):
    d = quat_dot(a, b)
    if d < 0.0:
        b = quat_neg(b)
        d = -d
    if d > 0.9995:
        r = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
             a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t)
        n = math.sqrt(sum(v * v for v in r)) or 1.0
        return (r[0] / n, r[1] / n, r[2] / n, r[3] / n)
    theta = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / s
    wb = math.sin(t * theta) / s
    return (a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb,
            a[2] * wa + b[2] * wb, a[3] * wa + b[3] * wb)


def _segment(times, t):
    """Index i so that times[i] <= t < times[i+1], clamped to a valid segment."""
    if t <= times[0]:
        return 0
    if t >= times[-1]:
        return len(times) - 2
    lo, hi = 0, len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid
    return lo


def sample_vector(keys, t):
    """Cubic-Hermite evaluation of a Unity AnimationCurve<Vector3>.

    ``keys`` is a sorted list of (time, value, inSlope, outSlope); Unity stores
    the tangents, so the classic Hermite basis reproduces the shipped curve.
    """
    if len(keys) == 1:
        return keys[0][1]
    times = [k[0] for k in keys]
    i = _segment(times, t)
    t0, v0, _, m0 = keys[i]
    t1, v1, m1, _ = keys[i + 1]
    dt = t1 - t0
    if dt <= 1e-9:
        return v1
    u = (t - t0) / dt
    u2 = u * u
    u3 = u2 * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    return tuple(h00 * v0[c] + h10 * dt * m0[c] + h01 * v1[c] + h11 * dt * m1[c]
                 for c in range(3))


def sample_quats(keys, t):
    """Piecewise-slerp evaluation of a decoded quaternion curve.

    ``keys`` is a sorted, sign-continuous list of (time, (x, y, z, w)).
    """
    if len(keys) == 1:
        return keys[0][1]
    times = [k[0] for k in keys]
    i = _segment(times, t)
    t0, q0 = keys[i]
    t1, q1 = keys[i + 1]
    dt = t1 - t0
    if dt <= 1e-9:
        return q1
    return slerp(q0, q1, (t - t0) / dt)


# --------------------------------------------------------------------------- #
# bundle access
# --------------------------------------------------------------------------- #
def load_env(bundles_dir: Path, rel: str):
    path = bundles_dir / rel
    if not path.exists():
        raise SystemExit(f"missing bundle: {path}")
    return UnityPy.load(str(path))


def read_all(env):
    """path_id -> (ObjectReader, parsed object or None)."""
    out = {}
    for o in env.objects:
        try:
            d = o.read()
        except Exception:
            d = None
        out[o.path_id] = (o, d)
    return out


def obj_name(pair):
    return getattr(pair[1], "m_Name", None) if pair else None


# --------------------------------------------------------------------------- #
# mesh / skeleton extraction
# --------------------------------------------------------------------------- #
class MeshData:
    def __init__(self):
        self.positions = []
        self.normals = []
        self.uvs = []
        self.bone_indices = []
        self.bone_weights = []
        self.triangles = []
        self.bindposes = []
        self.bones = []          # (name, path_id)
        self.bone_parent = []    # index or -1
        self.mesh_name = ""


def extract_mesh(guy_objs):
    """Locate the SMR/mesh of the shipped runner and pull every channel out."""
    smrs = [(pid, pair) for pid, pair in guy_objs.items()
            if pair[0].type.name == "SkinnedMeshRenderer"]
    if not smrs:
        raise SystemExit("no SkinnedMeshRenderer in the guy bundle")

    chosen = None
    for pid, (reader, smr) in smrs:
        if smr is None:
            continue
        parent = smr.m_GameObject.read().m_Transform.read().m_Father
        pname = None
        if parent is not None:
            pname = parent.read().m_GameObject.read().m_Name
        if pname == PREFAB_ROOT:
            chosen = (pid, smr)
            break
    if chosen is None:
        chosen = smrs[0]
    smr = chosen[1]

    mesh = smr.m_Mesh.read()
    if mesh is None:
        raise SystemExit("SkinnedMeshRenderer has no mesh")

    handler = MeshHandler(mesh)
    handler.process()
    md = MeshData()
    md.mesh_name = mesh.m_Name
    md.positions = [(v[0], v[1], v[2]) for v in handler.m_Vertices]
    md.normals = [(v[0], v[1], v[2]) for v in (handler.m_Normals or [])]
    md.uvs = [(v[0], v[1]) for v in (handler.m_UV0 or [])]
    if handler.m_BoneIndices is None or handler.m_BoneWeights is None:
        raise SystemExit("mesh has no skin weights")
    md.bone_indices = [tuple(int(c) for c in w) for w in handler.m_BoneIndices]
    md.bone_weights = [tuple(float(c) for c in w) for w in handler.m_BoneWeights]
    md.triangles = [t for t in handler.get_triangles()]
    md.bindposes = [unity_matrix4x4(bp) for bp in (mesh.m_BindPose or [])]

    bones = []
    for bptr in smr.m_Bones:
        tr = bptr.read()
        if tr is None:
            bones.append((f"bone_{len(bones)}", None))
        else:
            bones.append((tr.m_GameObject.read().m_Name, tr))
    # parent index inside the joint set (from each bone's Transform.m_Father)
    by_reader = {}
    for i, (n, tr) in enumerate(bones):
        if tr is not None:
            by_reader[id(tr.object_reader)] = i
    for i, (n, tr) in enumerate(bones):
        md.bone_parent.append(-1)
        if tr is None:
            continue
        father = tr.m_Father
        if father is None:
            continue
        try:
            ftr = father.read()
        except Exception:
            ftr = None
        if ftr is None:
            continue
        j = by_reader.get(id(ftr.object_reader))
        if j is not None:
            md.bone_parent[i] = j
    return smr, mesh, md, bones


def find_material_texture(guy_objs, smr):
    """Return (material object, Texture2D object, texture path_id)."""
    mat = None
    for ptr in (smr.m_Materials or []):
        try:
            m = ptr.read()
        except Exception:
            m = None
        if m is not None and m.m_Name == MATERIAL_NAME:
            mat = m
            break
    if mat is None:
        for ptr in (smr.m_Materials or []):
            try:
                m = ptr.read()
            except Exception:
                m = None
            if m is not None:
                mat = m
                break
    tex = None
    tex_pid = None
    for key, val in (getattr(mat.m_SavedProperties, "m_TexEnvs", None) or []):
        if key == "_MainTex":
            tex_pid = val.m_Texture.m_PathID
            try:
                tex = val.m_Texture.read()
            except Exception:
                tex = None
    return mat, tex, tex_pid


# --------------------------------------------------------------------------- #
# rig (skin joints + the extra leaf bones the clips animate)
# --------------------------------------------------------------------------- #
# The shipped runner rig is larger than the 25-bone skin: the prefab also
# carries ``Bone L/R Toe0``, ``Bone L/R Finger01`` and ``Bone L/R Finger11`` as
# leaf children of the foot/finger bones, and the shipped clips animate them.
# They are exported as ordinary nodes (children of their skin-joint parents) so
# the clip channels have real targets; they carry no skin weights, so the
# skinned result is unchanged.
def build_rig(smr, md, joint_names):
    """Return (nodes, path_to_node, joint_nodes).

    ``nodes`` are bone nodes in mesh space (index i -> glTF node i+1); the first
    ``len(joint_names)`` are the skin joints in ``SkinnedMeshRenderer.m_Bones``
    order, the rest are the extra leaf bones.  ``path_to_node`` maps a clip's
    bone path (e.g. ``"Bone Pelvis/Bone Spine"``) to a node index.
    """
    nodes = [{"name": n,
              "parent": (md.bone_parent[i] if md.bone_parent[i] >= 0 else None),
              "trs": None, "extras": False}
             for i, n in enumerate(joint_names)]
    by_name = {n: i for i, n in enumerate(joint_names)}

    root_tr = smr.m_GameObject.read().m_Transform.read().m_Father.read()
    found = []

    def walk(tr, path):
        name = tr.m_GameObject.read().m_Name
        here = path + "/" + name if path else name
        found.append((here, name, tr, path))
        for child in tr.m_Children:
            walk(child.read(), here)

    walk(root_tr, "")
    extras = [(p, n, tr, parent) for (p, n, tr, parent) in found
              if n.startswith("Bone ") and n not in by_name]
    extras.sort(key=lambda e: e[0])
    for path, name, tr, parent_path in extras:
        parent_name = parent_path.rsplit("/", 1)[-1] if parent_path else None
        parent_idx = by_name.get(parent_name)
        by_name[name] = len(nodes)
        nodes.append({"name": name, "parent": parent_idx, "trs": None,
                      "extras": True, "transform": tr})

    rel = [None] * len(nodes)
    for i, nd in enumerate(nodes):
        rel[i] = nd["name"] if nd["parent"] is None else rel[nd["parent"]] + "/" + nd["name"]
    path_to_node = {rel[i]: i for i in range(len(nodes))}
    joint_nodes = list(range(len(joint_names)))
    return nodes, path_to_node, joint_nodes


# --------------------------------------------------------------------------- #
# clip decoding
# --------------------------------------------------------------------------- #
def clip_path(obj, *attrs):
    for a in attrs:
        v = getattr(obj, a, None)
        if v:
            return v
    return ""


def decode_clip(clip):
    """Decode one legacy AnimationClip into per-path key lists (Unity space)."""
    rot, pos, scl = {}, {}, {}

    for cc in (clip.m_CompressedRotationCurves or []):
        path = clip_path(cc, "m_Path", "path")
        times_pv = getattr(cc, "m_Times", None)
        vals_pv = getattr(cc, "m_Values", None)
        if not path or times_pv is None or vals_pv is None:
            continue
        deltas = _unpack_ints(times_pv)
        quats = _unpack_quats(vals_pv)
        keys = []
        t = 0
        prev = None
        for i in range(min(len(deltas), len(quats))):
            t += deltas[i]
            q = quats[i]
            if prev is not None and quat_dot(prev, q) < 0.0:
                q = quat_neg(q)
            prev = q
            keys.append((t * 0.01, q))   # centisecond deltas, prefix-summed
        if keys:
            rot[path] = keys

    for rc in (clip.m_RotationCurves or []):
        path = clip_path(rc, "path", "m_Path")
        curve = getattr(rc, "curve", None)
        if not path or curve is None:
            continue
        keys = []
        prev = None
        for k in (curve.m_Curve or []):
            val = k.value
            q = (val.x, val.y, val.z, val.w)
            if prev is not None and quat_dot(prev, q) < 0.0:
                q = quat_neg(q)
            prev = q
            keys.append((float(k.time), q))
        if keys:
            rot[path] = keys

    for group, dest in ((clip.m_PositionCurves, pos), (clip.m_ScaleCurves, scl)):
        for c in (group or []):
            path = clip_path(c, "path", "m_Path")
            curve = getattr(c, "curve", None)
            if not path or curve is None:
                continue
            keys = []
            for k in (curve.m_Curve or []):
                v = k.value
                ins = getattr(k, "inSlope", None)
                out = getattr(k, "outSlope", None)
                keys.append((float(k.time), (v.x, v.y, v.z),
                             (ins.x, ins.y, ins.z) if ins is not None else (0.0,) * 3,
                             (out.x, out.y, out.z) if out is not None else (0.0,) * 3))
            if keys:
                dest[path] = keys

    last = 0.0
    for group in (rot, pos, scl):
        for keys in group.values():
            last = max(last, keys[-1][0])
    sr = float(clip.m_SampleRate or DEFAULT_FPS)
    frames = int(round(last * sr))
    times = [k / sr for k in range(frames + 1)]
    return {
        "name": clip.m_Name,
        "sample_rate": sr,
        "duration": times[-1] if times else 0.0,
        "frames": frames + 1,
        "rot": rot, "pos": pos, "scl": scl,
        "times": times,
    }


def _channel_at_rest(values, rest, kind):
    for v in values:
        if kind == "rotation":
            if min(max(abs(v[i] - rest[i]) for i in range(4)),
                   max(abs(v[i] + rest[i]) for i in range(4))) > 1e-5:
                return False
        else:
            if max(abs(v[i] - rest[i]) for i in range(len(rest))) > 1e-5:
                return False
    return True


def build_clip_animation(cdata, path_to_node, rest_trs):
    """Resample a decoded clip onto its frame grid and convert to export space."""
    times = cdata["times"]
    by_node = {}
    dropped = []
    for path in sorted(set(cdata["rot"]) | set(cdata["pos"]) | set(cdata["scl"])):
        node = path_to_node.get(path)
        if node is None:
            dropped.append(path)      # e.g. a hair bone this rig does not have
            continue
        entry = by_node.setdefault(node, {})
        if path in cdata["rot"]:
            keys = cdata["rot"][path]
            entry["rotation"] = [mirror_rotation(sample_quats(keys, t)) for t in times]
        if path in cdata["pos"]:
            keys = cdata["pos"][path]
            entry["translation"] = [mirror_translation(sample_vector(keys, t)) for t in times]
        if path in cdata["scl"]:
            keys = cdata["scl"][path]
            entry["scale"] = [sample_vector(keys, t) for t in times]

    channels = []
    for node in sorted(by_node):
        rest = rest_trs[node]
        for path, rest_i in (("translation", 0), ("rotation", 1), ("scale", 2)):
            values = by_node[node].get(path)
            if values is None:
                continue
            if _channel_at_rest(values, rest[rest_i], path):
                continue      # identical to the bind pose: nothing to animate
            channels.append({"node": node + 1, "path": path, "values": values})
    return {"name": cdata["name"], "source": cdata["name"],
            "duration": cdata["duration"], "frames": cdata["frames"],
            "sample_rate": cdata["sample_rate"], "times": times,
            "channels": channels, "dropped": dropped,
            "paths": sorted(set(cdata["rot"]) | set(cdata["pos"]) | set(cdata["scl"]))}


# --------------------------------------------------------------------------- #
# glTF / GLB writer
# --------------------------------------------------------------------------- #
class GlbBuilder:
    def __init__(self):
        self.blob = bytearray()
        self.views = []
        self.accessors = []

    def add_view(self, data: bytes, target=None) -> int:
        pad = (-len(self.blob)) % 4
        self.blob += b"\0" * pad
        off = len(self.blob)
        self.blob += data
        view = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1

    def add_accessor(self, view, comp_type, count, type_str, mn=None, mx=None) -> int:
        acc = {"bufferView": view, "componentType": comp_type,
               "count": count, "type": type_str}
        if mn is not None:
            acc["min"] = mn
            acc["max"] = mx
        self.accessors.append(acc)
        return len(self.accessors) - 1


ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
FLOAT, UBYTE, USHORT, UINT = 5126, 5121, 5123, 5125


def pack_floats(values):
    return struct.pack("<%df" % len(values), *values)


def build_glb(md: MeshData, png_bytes: bytes, material_name: str,
              nodes: list, joint_nodes: list, ibm: list,
              animations: list) -> bytes:
    """Write a glTF 2.0 binary: one skinned primitive, the bone node tree and
    one glTF animation per decoded clip."""
    g = GlbBuilder()
    nverts = len(md.positions)

    # --- geometry ---------------------------------------------------------
    pos = bytearray()
    for p in md.positions:
        pos += struct.pack("<3f", -p[0], p[1], p[2])
    minv = [min(-p[0] for p in md.positions),
            min(p[1] for p in md.positions),
            min(p[2] for p in md.positions)]
    maxv = [max(-p[0] for p in md.positions),
            max(p[1] for p in md.positions),
            max(p[2] for p in md.positions)]
    a_pos = g.add_accessor(g.add_view(bytes(pos), ARRAY_BUFFER), FLOAT,
                           nverts, "VEC3", minv, maxv)

    attrs = {"POSITION": a_pos}
    if len(md.normals) == nverts:
        nb = bytearray()
        for n in md.normals:
            nb += struct.pack("<3f", -n[0], n[1], n[2])
        attrs["NORMAL"] = g.add_accessor(g.add_view(bytes(nb), ARRAY_BUFFER),
                                         FLOAT, nverts, "VEC3")
    if len(md.uvs) == nverts:
        ub = bytearray()
        for uv in md.uvs:
            ub += struct.pack("<2f", uv[0], 1.0 - uv[1])
        attrs["TEXCOORD_0"] = g.add_accessor(g.add_view(bytes(ub), ARRAY_BUFFER),
                                             FLOAT, nverts, "VEC2")

    # joints / weights (glTF images: v flipped above already)
    jb = bytearray()
    wb = bytearray()
    for idx, wgt in zip(md.bone_indices, md.bone_weights):
        total = sum(wgt[:4]) or 1.0
        jb += struct.pack("<4B", idx[0], idx[1], idx[2], idx[3])
        wb += struct.pack("<4f", *(w / total for w in wgt[:4]))
    attrs["JOINTS_0"] = g.add_accessor(g.add_view(bytes(jb), ARRAY_BUFFER),
                                       UBYTE, nverts, "VEC4")
    attrs["WEIGHTS_0"] = g.add_accessor(g.add_view(bytes(wb), ARRAY_BUFFER),
                                        FLOAT, nverts, "VEC4")

    ind = bytearray()
    nind = 0
    for tris in md.triangles:
        for a, b, c in tris:
            ind += struct.pack("<3H", a, c, b)  # reverse winding (CCW front)
            nind += 3
    a_idx = g.add_accessor(g.add_view(bytes(ind), ELEMENT_ARRAY_BUFFER),
                           USHORT, nind, "SCALAR")

    # --- inverse bind matrices -------------------------------------------
    ibb = bytearray()
    for m in ibm:
        for c in range(4):
            for r in range(4):
                ibb += struct.pack("<f", m[r][c])
    a_ibm = g.add_accessor(g.add_view(bytes(ibb)), FLOAT, len(ibm), "MAT4")

    # --- texture ----------------------------------------------------------
    a_img = g.add_view(png_bytes)
    json_doc = {
        "asset": {"version": "2.0", "generator": "tools/export_character.py"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "textures": [],
        "images": [],
        "samplers": [],
        "skins": [],
        "accessors": g.accessors,
        "bufferViews": g.views,
        "buffers": [{"byteLength": len(g.blob)}],
    }

    # nodes: 0 = mesh node, 1..N = bone nodes (index i -> glTF node i+1)
    json_doc["nodes"].append({"name": MESH_NAME, "mesh": 0, "skin": 0})
    for nd in nodes:
        t, q, s = nd["trs"]
        json_doc["nodes"].append({
            "name": nd["name"],
            "translation": [t[0], t[1], t[2]],
            "rotation": [q[0], q[1], q[2], q[3]],
            "scale": [s[0], s[1], s[2]],
        })
    for i, nd in enumerate(nodes):
        if nd["parent"] is not None:
            json_doc["nodes"][nd["parent"] + 1].setdefault("children", []).append(i + 1)
    roots = [i + 1 for i, nd in enumerate(nodes) if nd["parent"] is None]
    json_doc["scenes"][0]["nodes"] = [0] + roots
    json_doc["skins"].append({
        "name": "GuyDangerous",
        "joints": [i + 1 for i in joint_nodes],
        "inverseBindMatrices": a_ibm,
        "skeleton": roots[0] if roots else joint_nodes[0] + 1,
    })

    prim = {"attributes": attrs, "indices": a_idx, "material": 0, "mode": 4}
    json_doc["meshes"].append({"name": MESH_NAME, "primitives": [prim]})
    json_doc["samplers"].append({"magFilter": 9729, "minFilter": 9987,
                                 "wrapS": 10497, "wrapT": 10497})
    json_doc["images"].append({"name": material_name, "bufferView": a_img,
                               "mimeType": "image/png"})
    json_doc["textures"].append({"source": 0, "sampler": 0})
    json_doc["materials"].append({
        "name": material_name,
        "doubleSided": True,
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0.0,
            "roughnessFactor": 0.85,
        },
    })

    # --- animations -------------------------------------------------------
    # Every clip is resampled onto its own sample grid, so one shared input
    # accessor (the frame times) feeds every channel of the clip.  LINEAR is
    # used for all three paths: it is what raylib's GLTF pose evaluator handles
    # per bone without ambiguity (its cubic-spline path is vec3-only and it
    # keeps a single interpolation type per bone).
    gltf_anims = []
    for anim in animations:
        times = anim["times"]
        tacc = g.add_accessor(g.add_view(pack_floats(times), ARRAY_BUFFER), FLOAT,
                              len(times), "SCALAR", [times[0]], [times[-1]])
        samplers, channels = [], []
        for ch in anim["channels"]:
            values = ch["values"]
            dim = len(values[0])
            data = bytearray()
            for v in values:
                data += struct.pack("<%df" % dim, *v)
            oacc = g.add_accessor(g.add_view(bytes(data), ARRAY_BUFFER), FLOAT,
                                  len(values), "VEC3" if dim == 3 else "VEC4")
            samplers.append({"input": tacc, "output": oacc,
                             "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": ch["node"], "path": ch["path"]}})
        gltf_anims.append({"name": anim["name"], "samplers": samplers,
                           "channels": channels})
    if gltf_anims:
        json_doc["animations"] = gltf_anims

    json_doc["buffers"][0]["byteLength"] = len(g.blob)

    js = json.dumps(json_doc, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    blob = bytes(g.blob) + b"\0" * ((-len(g.blob)) % 4)

    total = 12 + 8 + len(js) + 8 + len(blob)
    out = bytearray()
    out += b"glTF" + struct.pack("<II", 2, total)
    out += struct.pack("<I", len(js)) + b"JSON" + js
    out += struct.pack("<I", len(blob)) + b"BIN\0" + blob
    return bytes(out)


# --------------------------------------------------------------------------- #
# static OBJ / MTL fallback
# --------------------------------------------------------------------------- #
def write_obj(out_dir: Path, md: MeshData, material_name: str):
    lines = [
        "# Temple Run 2 runner 'Guy Dangerous' - static bind-pose mesh\n",
        "# Unity axes with X negated (right-handed); +Z = character forward,\n",
        "# feet at y=0, units = Unity metres.  See README.md.\n",
        "mtllib guy.mtl\n",
        f"o {MESH_NAME}\n",
    ]
    for p in md.positions:
        lines.append(f"v {fmt(-p[0])} {fmt(p[1])} {fmt(p[2])}\n")
    for uv in md.uvs:
        lines.append(f"vt {fmt(uv[0])} {fmt(uv[1])}\n")
    for n in md.normals:
        lines.append(f"vn {fmt(-n[0])} {fmt(n[1])} {fmt(n[2])}\n")
    has_uv = len(md.uvs) == len(md.positions)
    has_n = len(md.normals) == len(md.positions)
    lines.append(f"usemtl {material_name}\n")
    for tris in md.triangles:
        for a, b, c in tris:
            f = []
            for i in (a, c, b):  # reverse winding, CCW front
                if has_uv and has_n:
                    f.append(f"{i + 1}/{i + 1}/{i + 1}")
                elif has_uv:
                    f.append(f"{i + 1}/{i + 1}")
                else:
                    f.append(f"{i + 1}")
            lines.append("f " + " ".join(f) + "\n")
    (out_dir / "guy.obj").write_text("".join(lines), encoding="utf-8")

    mtl = [
        "# Temple Run 2 runner material (extracted _MainTex -> guy.png)\n",
        f"newmtl {material_name}\n",
        "Ka 0.000000 0.000000 0.000000\n",
        "Kd 1.000000 1.000000 1.000000\n",
        "Ks 0.000000 0.000000 0.000000\n",
        "d 1.000000\n",
        "illum 1\n",
        "map_Kd guy.png\n",
    ]
    (out_dir / "guy.mtl").write_text("".join(mtl), encoding="utf-8")
    return len(md.triangles[0]) if md.triangles else 0


# --------------------------------------------------------------------------- #
# clips
# --------------------------------------------------------------------------- #
def collect_clips(guy_env_objs, shared_env_objs, path_to_node, rest_trs):
    """Decode every shipped AnimationClip into GLB-ready animations + a catalog."""
    catalog = {}
    for src, env_objs in (("guy", guy_env_objs), ("shared", shared_env_objs)):
        for pid, (reader, obj) in env_objs.items():
            if reader.type.name != "AnimationClip" or obj is None:
                continue
            name = obj.m_Name
            cdata = decode_clip(obj)
            anim = build_clip_animation(cdata, path_to_node, rest_trs)
            catalog[name] = {
                "source": name,
                "bundle": src,
                "path_id": pid,
                "legacy": bool(obj.m_Legacy),
                "sample_rate": cdata["sample_rate"],
                "duration": round(cdata["duration"], 6),
                "frames": cdata["frames"],
                "position_curves": len(obj.m_PositionCurves or []),
                "rotation_curves": len(obj.m_CompressedRotationCurves or []),
                "scale_curves": len(obj.m_ScaleCurves or []),
                "animated_bones": len(anim["paths"]),
                "unmapped_bones": anim["dropped"],
                "channels": len(anim["channels"]),
                "anim": anim,
            }
    return catalog


def build_animations_json(catalog, bones, extra_bones, bounds, yaw_offset,
                          scale, fps, file_map, glb_animations):
    out_clips = {}
    details = {}
    for src in CLIP_ORDER:
        if src not in catalog:
            continue
        friendly = CLIP_NAMES.get(src, src.lower())
        info = catalog[src]
        out_clips[friendly] = {
            "name": friendly,
            "clip": src,
            "fps": info["sample_rate"] or fps,
            "duration": info["duration"],
            "frames": info["frames"],
            "loop": friendly != "death",
            "baked": True,
            "rotation_encoding": "legacy-compressed-2020.3 (decoded)",
            "channels": info["channels"],
            "unmapped_bones": info["unmapped_bones"],
        }
    for src in sorted(catalog):
        details[src] = {k: v for k, v in catalog[src].items() if k != "anim"}
    doc = {
        "model": file_map["glb"],
        "obj": file_map["obj"],
        "png": file_map["png"],
        "yaw_offset_deg": yaw_offset,
        "scale": scale,
        "fps": fps,
        "bounds_min": [round(v, 6) for v in bounds[0]],
        "bounds_max": [round(v, 6) for v in bounds[1]],
        "up_axis": "Y",
        "forward_axis": "Z",
        "units": "unity-metres",
        "skeleton": {
            "joint_count": len(bones),
            "bind_pose": "exported in guy.glb / guy.obj",
            "bones": bones,
            "extra_bones": extra_bones,
        },
        "glb_animations": glb_animations,
        "clips": out_clips,
        "clip_details": details,
        "notes": [
            "Every shipped clip is decoded from the legacy "
            "m_CompressedRotationCurves (AssetStudio PackedQuatVector.UnpackQuats; "
            "times are centisecond deltas) and resampled onto the clip's own "
            "m_SampleRate grid, then written into guy.glb "
            "('animations' array, LINEAR).",
            "yaw_offset_deg is 0: the exported mesh already faces +Z (nose at "
            "z=+0.123, toes at z=+0.132 against back-of-head z=-0.084), and the "
            "renderer's track forward at yaw 0 is +Z. The props/track pieces are "
            "-Z-forward and use 180.",
            "Vertices are Unity axes with X negated; bone local transforms use "
            "S*L*S (translation -x, rotation (x,-y,-z,w), scale unchanged).",
            "Only the 25 skin joints drive deformation; the 6 extra leaf bones "
            "(Toe0 / Finger01 / Finger11) are exported and animated but raylib "
            "ignores channels whose target is not in skin.joints.",
            "raylib's LoadModelAnimations() resamples to 17 ms frames, so "
            "ModelAnimation.frameCount is duration*1000/17+1 (~58.8 fps), not "
            "the clip's 30 keyframes; the renderer steps those frames at "
            "1000/17 fps for true-speed playback.",
        ],
    }
    return doc


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #
README_TEMPLATE = """# Temple Run 2 runner assets ("Guy Dangerous")

Generated by `tools/export_character.py` (run it with
`recon/unity_assets/.venv/bin/python`).  Nothing here is hand-made: every value
comes from the shipped bundles listed under *Provenance*.

| file | what |
|---|---|
| `guy.glb` | skinned mesh + skeleton + albedo texture + all decoded clips |
| `guy.obj` / `guy.mtl` | static bind-pose mesh + material, `map_Kd guy.png` |
| `guy.png` | 1024x1024 albedo (the material's `_MainTex`) |
| `animations.json` | clip + bone catalogue and the runtime overrides |

## Node / axis conventions

* **Frame**: Unity coordinates with **X negated** - the standard left-handed
  Unity -> right-handed glTF/raylib conversion (identical to UnityPy's OBJ
  writer and to `tools/export_props.py`).  `Y` is up, `+Z` is the character's
  forward, `+X` is the character's **left** after the flip.
* **Origin**: the prefab root `guy_dangerous`; the soles are at `y = 0`, so the
  model can be placed directly at a ground point.
* **Units**: Unity metres.  Bounds `%(bounds_min)s .. %(bounds_max)s`
  (height `%(height).4f`, width `%(width).4f`, depth `%(depth).4f`).
* **Facing**: measured, not assumed.  The mesh faces `+Z`: the nose is the
  head's `+Z` extreme (`z = +0.123`) against the back of the head (`z = -0.084`),
  and the feet's `+Z` extreme is the toe (`z = +0.132`) against the heel
  (`z = -0.053`).  The renderer's track frame wants `forward = (sin yaw, cos yaw)`,
  i.e. `+Z` at `yaw = 0`, so `animations.json` sets **`yaw_offset_deg = 0.0`**
  (the shipped props/track pieces are `-Z`-forward and use `180.0`).
* **Winding**: triangles are reversed relative to the Unity mesh so front faces
  are counter-clockwise (CCW) once X is negated.  `guy.glb` also sets
  `doubleSided = true`.
* **UVs**: `guy.glb` writes glTF `TEXCOORD_0` bottom-up (`v' = 1 - v`, the glTF
  convention); the `guy.obj` fallback keeps Unity's `vt` values verbatim
  (matching `export_props.py`).
* **Node names**: `guy.glb` node 0 is the mesh node `%(mesh_name)s`; nodes
  `%(first_joint)d..%(last_joint)d` are the skeleton, named exactly like the
  Unity bones.  The first %(joint_count)d are the skin joints, ordered like the
  `SkinnedMeshRenderer.m_Bones` array (the same index space as the mesh's
  `JOINTS_0` and `inverseBindMatrices`); the remaining nodes are extra leaf bones
  (%(extra_bones)s) that the clips animate but the skin does not weight.  Joint
  locals are derived from the mesh's own `m_BindPose`, so the bind pose is
  reproduced exactly.

Skin joints (order): %(bone_list)s

## Animations

All %(clip_count)d shipped clips are decoded and baked into `guy.glb`
(`animations`), each named for its gameplay state (`run`, `jump`, `slide`,
`idle`, `death`, ...).  Decoding follows AssetStudio: the legacy bone rotations
live in `m_CompressedRotationCurves` (`PackedQuatVector.UnpackQuats` - a 3-bit
"which component is implicit" flag, the other three at 9/10 bits each) and the
key times are centisecond *deltas* (`PackedIntVector`, prefix-summed, `* 0.01`);
positions and scales are plain `AnimationCurve` keyframes and are evaluated with
their stored Hermite tangents.  Every curve of a clip is resampled onto that
clip's own `m_SampleRate` grid (`%(fps).0f` fps) and written as `LINEAR`
samplers, so playback timing, hierarchy and coordinates match the shipped data;
bone locals are converted with the same `S*L*S` mirror as the bind pose.

Each clip animates every rig bone (translation + rotation + scale); channels
identical to the bind pose are omitted.  Clips that reference a bone this rig
lacks (`Bone Hair001/002`, which belong to other runners) have those channels
dropped - recorded per clip as `unmapped_bones` in `animations.json`.

**Runtime note (raylib):** `LoadModelAnimations()` ignores the glTF key times
for playback - it resamples every clip to `GLTF_ANIMDELAY` = 17 ms frames, so
`ModelAnimation.frameCount` is `duration*1000/17 + 1` (~58.8 fps), not this
clip's %(fps).0f keyframes. The renderer steps those frames at `1000/17` fps;
driving `frameCount` at %(fps).0f fps would play every clip at 0.51x.

| state | clip | source | duration (s) | frames | bones | channels |
|---|---|---|---|---|---|---|
%(clip_table)s

## Provenance

* `%(guy_bundle)s`
  * `Mesh %(mesh_name)s` (path id `%(mesh_pid)s`) - 1351 verts / 1930 tris,
    25-bone skin, `SkinnedMeshRenderer %(smr_pid)s`
  * `Material %(material)s`, `Texture2D %(texture)s` (path id `%(tex_pid)s`,
    1024x1024)
  * clips %(guy_clips)s
* `%(shared_bundle)s`
  * clips %(shared_clips)s
"""


def write_readme(out_dir, md, bones, extra_bones, bounds, catalog, prov, fps):
    height = bounds[1][1] - bounds[0][1]
    width = bounds[1][0] - bounds[0][0]
    depth = bounds[1][2] - bounds[0][2]
    rows = []
    count = 0
    for src in CLIP_ORDER:
        if src not in catalog:
            continue
        info = catalog[src]
        count += 1
        rows.append("| `%s` | `%s` | %s | %.4f | %d | %d | %d |" % (
            CLIP_NAMES.get(src, src.lower()), src, info["bundle"],
            info["duration"], info["frames"], info["animated_bones"],
            info["channels"]))
    text = README_TEMPLATE % {
        "bounds_min": [round(v, 4) for v in bounds[0]],
        "bounds_max": [round(v, 4) for v in bounds[1]],
        "height": height,
        "width": width,
        "depth": depth,
        "fps": fps,
        "mesh_name": md.mesh_name,
        "first_joint": 1,
        "last_joint": len(bones) + len(extra_bones),
        "joint_count": len(bones),
        "extra_bones": ", ".join(f"`{b}`" for b in extra_bones) or "-",
        "bone_list": ", ".join(f"`{b}`" for b in bones),
        "clip_count": count,
        "clip_table": "\n".join(rows) if rows else "| - | - | - | - | - | - | - |",
        **prov,
    }
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    guy_env = load_env(args.bundles, GUY_BUNDLE)
    guy_objs = read_all(guy_env)
    try:
        shared_env = load_env(args.bundles, SHARED_BUNDLE)
        shared_objs = read_all(shared_env)
    except SystemExit:
        shared_objs = {}

    smr, mesh, md, bones_raw = extract_mesh(guy_objs)
    bone_names = [b[0] for b in bones_raw]
    if not md.bindposes:
        raise SystemExit("mesh has no bind poses")

    mat, tex, tex_pid = find_material_texture(guy_objs, smr)
    if tex is None:
        raise SystemExit("main texture missing")
    image = tex.image
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    (out_dir / "guy.png").write_bytes(png_bytes)

    # ---- bind-pose globals, mirrored into the export frame ---------------
    globals_mirrored = []
    worlds = []
    for i, bp in enumerate(md.bindposes):
        try:
            world = mat_invert(bp)
        except ValueError:
            raise SystemExit(f"bind pose {i} singular")
        worlds.append(world)
        globals_mirrored.append(mirror_x(world))

    node_locals = []
    for i, gm in enumerate(globals_mirrored):
        p = md.bone_parent[i]
        if p >= 0:
            local = mat_mul(mat_invert(globals_mirrored[p]), gm)
        else:
            local = gm
        node_locals.append(decompose(local))

    ibm = [mirror_x(bp) for bp in md.bindposes]

    # ---- rig: skin joints + the extra leaf bones the clips animate -------
    nodes, path_to_node, joint_nodes = build_rig(smr, md, bone_names)
    rest_trs = []
    for i, nd in enumerate(nodes):
        if nd["extras"]:
            tr = nd["transform"]
            lp, lr, ls = tr.m_LocalPosition, tr.m_LocalRotation, tr.m_LocalScale
            local = (mirror_translation((lp.x, lp.y, lp.z)),
                     mirror_rotation((lr.x, lr.y, lr.z, lr.w)),
                     (ls.x, ls.y, ls.z))
        else:
            local = node_locals[i]
        nd["trs"] = local
        rest_trs.append(local)
    extra_bones = [nd["name"] for nd in nodes[len(bone_names):]]

    # ---- clips -> glTF animations ----------------------------------------
    catalog = collect_clips(guy_objs, shared_objs, path_to_node, rest_trs)
    animations = []
    for src in CLIP_ORDER:
        info = catalog.get(src)
        if info is not None and info["anim"]["channels"]:
            info["anim"]["name"] = CLIP_NAMES.get(src, src.lower())
            animations.append(info["anim"])
    glb_animations = [a["name"] for a in animations]

    glb = build_glb(md, png_bytes, material_name=mat.m_Name if mat else MATERIAL_NAME,
                    nodes=nodes, joint_nodes=joint_nodes, ibm=ibm,
                    animations=animations)
    (out_dir / "guy.glb").write_bytes(glb)
    write_obj(out_dir, md, mat.m_Name if mat else MATERIAL_NAME)

    # ---- bounds in the export frame --------------------------------------
    xs = [-p[0] for p in md.positions]
    ys = [p[1] for p in md.positions]
    zs = [p[2] for p in md.positions]
    bounds = ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    guy_clips = sorted(k for k, v in catalog.items() if v["bundle"] == "guy")
    shared_clips = sorted(k for k, v in catalog.items() if v["bundle"] == "shared")

    prov = {
        "guy_bundle": GUY_BUNDLE,
        "shared_bundle": SHARED_BUNDLE,
        "mesh_pid": str(-169845926871220793),
        "smr_pid": str(smr.m_GameObject.m_PathID),
        "material": mat.m_Name if mat else MATERIAL_NAME,
        "texture": tex.m_Name,
        "tex_pid": str(tex_pid),
        "guy_clips": ", ".join(f"`{c}`" for c in guy_clips) or "-",
        "shared_clips": ", ".join(f"`{c}`" for c in shared_clips) or "-",
    }

    file_map = {"glb": "guy.glb", "obj": "guy.obj", "png": "guy.png"}
    doc = build_animations_json(catalog, bone_names, extra_bones, bounds,
                               yaw_offset=0.0, scale=1.0, fps=DEFAULT_FPS,
                               file_map=file_map, glb_animations=glb_animations)
    (out_dir / "animations.json").write_text(
        json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    write_readme(out_dir, md, bone_names, extra_bones, bounds, catalog, prov,
                 DEFAULT_FPS)

    tris = sum(len(t) for t in md.triangles)
    print(f"mesh {md.mesh_name}: {len(md.positions)} verts, {tris} tris, "
          f"{len(md.bindposes)} bind poses, {len(bone_names)} joints, "
          f"+{len(extra_bones)} extra bones")
    expect = {"Bone Pelvis": (0.0, 0.6208, 0.0335),
              "Bone Head": (0.0, 1.1169, 0.0309),
              "Bone L Toe0": (-0.1037, 0.003, 0.0662)}
    for i, name in enumerate(bone_names):
        if name in expect:
            w = worlds[i]
            got = (w[0][3], w[1][3], w[2][3])
            err = max(abs(got[k] - expect[name][k]) for k in range(3))
            print(f"  bindpose {name}: world={tuple(round(v, 4) for v in got)}"
                  f" (prefab pose diff {err:.4f})")
    for anim in animations:
        print(f"  clip {anim['source']:<16} {anim['frames']:>3} frames "
              f"@{anim['sample_rate']:.0f}fps  {len(anim['channels'])} channels"
              + (f"  dropped {anim['dropped']}" if anim["dropped"] else ""))
    print(f"clips: {len(catalog)} ({len(guy_clips)} guy + "
          f"{len(shared_clips)} shared), baked into GLB: {glb_animations}")
    print(f"bounds (export frame): {bounds[0]} .. {bounds[1]}")
    print(f"wrote {out_dir/'guy.glb'} ({len(glb)} bytes), guy.obj, guy.mtl, "
          f"guy.png ({len(png_bytes)} bytes), animations.json, README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
