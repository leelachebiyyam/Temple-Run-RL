#!/usr/bin/env python3
"""Export Temple Run 2 prop / pickup / UI assets from the shipped Unity data.

Reads the extracted APK serialized files and asset bundles under
recon/unity_assets/ (see recon/unity_assets/SUMMARY.md) and writes:

    resources/templerun2/props/<name>/model.obj
    resources/templerun2/props/<name>/model.mtl
    resources/templerun2/props/tex/<texture>.png      (shared texture pool)
    resources/templerun2/props/manifest.json
    resources/templerun2/ui/*.png
    resources/templerun2/ui/*.ttf
    resources/templerun2/ui/manifest.json

Geometry is exact: for each prop we locate the shipped prefab root GameObject,
walk its descendant GameObjects, skip scenery/effect/dummy/collision nodes, and
merge every visible mesh child with its full local->root transform baked in
(TRS composed down the hierarchy).  The result is expressed in the prefab root
frame with UnityPy's X-handedness flip (same convention as Mesh.export('obj')),
so the renderer's yaw_offset_deg/scale behave identically for every asset.
Every missing item is recorded in the manifest instead of silently dropped.

Usage:
    recon/unity_assets/.venv/bin/python tools/export_props.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict

import UnityPy
from PIL import Image
from PIL.Image import Transpose
from UnityPy.export.SpriteHelper import SpriteSettings
from UnityPy.export.Texture2DConverter import get_image_from_texture2d
from UnityPy.enums import SpritePackingRotation
from UnityPy.helpers.MeshHelper import MeshHandler

# --------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
UA = os.path.join(REPO, "recon", "unity_assets")
DATA = os.path.join(UA, "apk", "assets", "bin", "Data")
BUNDLES = os.path.join(UA, "bundles")
OUT = os.path.join(REPO, "resources", "templerun2")
PROPS_DIR = os.path.join(OUT, "props")
TEX_DIR = os.path.join(PROPS_DIR, "tex")
UI_DIR = os.path.join(OUT, "ui")

SANITIZE = re.compile(r"[^A-Za-z0-9_.-]+")

# Nodes that are never part of a prop's silhouette.  Names are matched
# case-insensitively; the spec's own root is always kept.
DEFAULT_EXCLUDE = re.compile(
    r"floating|scenery|_deco|deco_|^\s*deco|dummy|effect|particle|sound|shadow|"
    r"collision|collsion|trigger|cobweb|foliage|leaves|cloud|sky|lightning|moon|"
    r"pan_|water|terrain|splash|trail|emitter|bolt_|_vfx|vfx_|"
    r"prop_tombstone|prop_pumpkin|prop_river|prop_grass|grass_props",
    re.I,
)

# --------------------------------------------------------------------------- specs
# name, kind, root (prefab root GameObject name), file_hint, material fallback,
# note.  file_hint is a substring of the serialized-file name; the first root
# that contains a mesh wins.
PROPS = [
    # --- obstacles ---------------------------------------------------------
    dict(name="obstacle_jump_over", kind="obstacle",
         root="PirateCove_Docks_Trap_Log_A", file="region",
         material="Atlas_Opaque_Pirate01",
         note="Jump-over obstacle: the shipped Pirate Cove dock trap log."),
    dict(name="obstacle_jump_over_spikes", kind="obstacle",
         root="PirateCove_Docks_Trap_Log_Spikes_A", file="region",
         material="Batch_Atlas_Opaque_Pirate01",
         note="Jump-over obstacle variant: spiked dock trap log."),
    dict(name="obstacle_slide_under", kind="obstacle",
         root="PirateCove_Docks_Beam_A_prop", file="region",
         material="Batch_Atlas_Opaque_Pirate01",
         note="Slide-under obstacle: dock beam (low bar across the path)."),
    dict(name="obstacle_slide_under_beam", kind="obstacle",
         root="PirateCove_TreasureCave_Beam_D", mesh="PirateCove_TreasureCave_Beam_D",
         file="treasurecave", material="Batch_Atlas_Opaque_Pirate01",
         note="Slide-under obstacle variant: Treasure Cave fallen beam."),
    dict(name="obstacle_stumble", kind="obstacle",
         root="PirateCove_Cavern_Post_A", file="region",
         material="Batch_Atlas_Opaque_Pirate01",
         exclude="postrope|rope",
         note="Stumble obstacle: cavern post/trunk over the path."),
    dict(name="obstacle_gap_small", kind="obstacle",
         root="gap_small_a", file="base", material="machu_master_opaque_halloween",
         note="Machu gap-over piece (gap obstacle; the mesh spans the piece "
              "length, lateral/vertical extents are obstacle-scale)."),
    dict(name="obstacle_ledge_left", kind="obstacle",
         root="ledge_left_a", file="base", material="machu_master_opaque_halloween",
         note="Machu left ledge piece."),
    dict(name="obstacle_ledge_right", kind="obstacle",
         root="ledge_right_a", file="base", material="machu_master_opaque_halloween",
         note="Machu right ledge piece."),
    dict(name="obstacle_statue", kind="obstacle",
         root="PirateCove_MonkeyStatue_prop", file="town",
         material="Batch_Atlas_Opaque_Pirate01",
         note="Full statue: Pirate Cove monkey statue (monument scale)."),
    dict(name="obstacle_idol", kind="obstacle",
         root="idol", mesh="idol", file="region", material="IdolMaterial",
         note="Full statue: Temple idol."),
    dict(name="obstacle_statue_skull", kind="obstacle",
         root="Machu_skull_monkey", mesh="Machu_skull_monkey", file="region",
         material="skull_monkey",
         note="Machu Halloween skull-monkey statue."),
    dict(name="obstacle_fire_brazier", kind="obstacle",
         root="PirateCove_Torch_prop", file="region",
         material="Batch_Atlas_Opaque_Pirate01",
         note="Fire/brazier: Pirate Cove torch."),
    dict(name="obstacle_fire_candles", kind="obstacle",
         root="prop_candles", file="region", material="machu_master_opaque_halloween",
         note="Fire/brazier variant: Halloween candle cluster."),
    dict(name="obstacle_fire_blue", kind="obstacle",
         root="effect_fire_blue_a", file="region", material="effects_additive",
         note="Halloween blue flame mesh."),
    dict(name="obstacle_lantern", kind="obstacle",
         root="PirateCove_Cavern_Lantern_A", file="region",
         material="Batch_Atlas_Opaque_Pirate01",
         note="Pirate Cove cavern lantern."),
    dict(name="obstacle_saw", kind="obstacle",
         root="jump_over_b_saw", file="base", material="machu_master_opaque_halloween",
         note="Machu circular-saw obstacle (jump-over saw piece)."),
    # --- coins -------------------------------------------------------------
    dict(name="coin_gold", kind="coin", root="default_coin_gold",
         file="defaultcoinsetdata", material="coin_default",
         note="Default coin set: yellow / gold coin."),
    dict(name="coin_red", kind="coin", root="default_coin_red",
         file="defaultcoinsetdata", material="coin_default",
         note="Default coin set: red coin."),
    dict(name="coin_blue", kind="coin", root="default_coin_blue",
         file="defaultcoinsetdata", material="coin_default",
         note="Default coin set: blue coin."),
    dict(name="coin_mega", kind="coin", root="coin_mega", file="sharedassets0",
         material="PickUps01", note="Mega coin pickup."),
    # --- pickups -----------------------------------------------------------
    dict(name="pickup_magnet", kind="pickup", root="magnet", file="sharedassets0",
         material="PickUps01", note="Coin magnet powerup pickup."),
    dict(name="pickup_boost", kind="pickup", root="boost", file="sharedassets0",
         material="PickUps01", note="Boost / bolt powerup pickup."),
    dict(name="pickup_shield", kind="pickup", root="shield", file="sharedassets0",
         material="PickUps01", note="Shield powerup pickup."),
    dict(name="pickup_gem", kind="pickup", root="gem_a", mesh="gem_a",
         file="sharedassets0", material="Gem01", note="Gem pickup."),
    dict(name="pickup_coin_shadow", kind="pickup", root="coin_shadow",
         file="sharedassets0", material="global_effects_transparent",
         note="Blob shadow used under coins."),
]

# Props whose only shipped artwork is a 2D sprite (no 3D mesh in the data).
SPRITE_PROPS = [
    ("pickup_coin_bonus", "powerup_coin_bonus",
     "Coin bonus pickup: sprite only, no 3D mesh shipped."),
    ("pickup_score_bonus", "powerup_score_bonus",
     "Score bonus pickup: sprite only, no 3D mesh shipped."),
    ("pickup_gem_bonus", "powerup_gem_bonus",
     "Gem bonus pickup: sprite only, no 3D mesh shipped."),
]

UI_KEYWORDS = [
    "coin", "gem", "score", "multiplier", "power_meter", "meter", "magnet",
    "shield", "boost", "bolt", "bonus", "pause", "distance", "powerup",
]

UI_TEX_RE = re.compile(
    r"hud|__ui|icon|button|score|coin|gem|meter|powerup|pause|star|badge|banner|"
    r"bk-|font|atlas|interface|shield|boost|magnet|toggle|store|sale|window|"
    r"frame|glow|gradient|arrow|dialog|popup|bar\b",
    re.I,
)

# Bundle-resident (CAB-*) textures are only taken when they belong to the UI
# atlases; this keeps track/scenery atlases out of ui/.
UI_TEX_BUNDLE_RE = re.compile(
    r"^(HUD_|UI_|sactx-.*(interface|Interface|Social|DailyLogin|NewUIUX))|"
    r"Atlas$|^Font Texture$|^Poppins|^TempleRunSmallCaps",
)


def sanitize(name: str) -> str:
    return SANITIZE.sub("_", name).strip("_") or "unnamed"


def src_name(obj) -> str:
    """Readable source file for an object: bundle path or serialized file."""
    af = obj.assets_file
    parent = getattr(af, "parent", None)
    if parent is not None and getattr(parent, "name", None):
        return parent.name
    return af.name


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- math
IDENT = (1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0)


def mat_mul(a, b):
    return tuple(
        sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4))
        for r in range(4) for c in range(4)
    )


def local_matrix(t):
    p, q, s = t.m_LocalPosition, t.m_LocalRotation, t.m_LocalScale
    x, y, z, w = q.x, q.y, q.z, q.w
    r = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
         (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
         (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))
    sc = (s.x, s.y, s.z)
    return (r[0][0] * sc[0], r[0][1] * sc[1], r[0][2] * sc[2], p.x,
            r[1][0] * sc[0], r[1][1] * sc[1], r[1][2] * sc[2], p.y,
            r[2][0] * sc[0], r[2][1] * sc[1], r[2][2] * sc[2], p.z,
            0.0, 0.0, 0.0, 1.0)


def xform_point(m, v):
    x, y, z = v
    return (m[0] * x + m[1] * y + m[2] * z + m[3],
            m[4] * x + m[5] * y + m[6] * z + m[7],
            m[8] * x + m[9] * y + m[10] * z + m[11])


def xform_dir(m, v):
    x, y, z = v
    return (m[0] * x + m[1] * y + m[2] * z,
            m[4] * x + m[5] * y + m[6] * z,
            m[8] * x + m[9] * y + m[10] * z)


# --------------------------------------------------------------------------- env
class Assets:
    """Lazy per-file access to GameObjects, transforms, filters, renderers."""

    def __init__(self, env):
        self.env = env
        self.by_file = defaultdict(list)
        self.go_by_name = defaultdict(list)
        for o in env.objects:
            t = o.type.name
            if t not in ("GameObject", "Transform", "RectTransform", "MeshFilter",
                         "MeshRenderer", "SkinnedMeshRenderer", "Mesh",
                         "Material", "Texture2D", "Sprite", "Font", "SpriteAtlas"):
                continue
            fn = src_name(o)
            self.by_file[fn].append(o)
            if t == "GameObject":
                try:
                    self.go_by_name[o.read().m_Name].append((fn, o.path_id, o))
                except Exception:
                    continue
        self._trees = {}
        self._mesh_owner = None

    def tree(self, fname):
        t = self._trees.get(fname)
        if t is not None:
            return t
        go, tr, mf, mr, mesh, comp_go = {}, {}, {}, {}, {}, {}
        for o in self.by_file.get(fname, []):
            ty = o.type.name
            try:
                if ty == "GameObject":
                    g = o.read()
                    go[o.path_id] = g
                    for comp in (g.m_Component or []):
                        c = comp.component if hasattr(comp, "component") else comp
                        comp_go[c.m_PathID] = o.path_id
                elif ty in ("Transform", "RectTransform"):
                    tr[o.path_id] = o.read()
                elif ty == "MeshFilter":
                    mf[o.path_id] = o.read()
                elif ty in ("MeshRenderer", "SkinnedMeshRenderer"):
                    mr[o.path_id] = o.read()
                elif ty == "Mesh":
                    mesh[o.path_id] = o.read()
            except Exception:
                continue
        t = dict(go=go, tr=tr, mf=mf, mr=mr, mesh=mesh, comp_go=comp_go)
        self._trees[fname] = t
        return t

    def mesh_owners(self, mesh_name):
        """(file, GO pid) pairs for GameObjects that own a mesh of this name."""
        if self._mesh_owner is None:
            self._mesh_owner = defaultdict(list)
            for fn in self.by_file:
                t = self.tree(fn)
                for mfid, filt in t["mf"].items():
                    mo = t["mesh"].get(filt.m_Mesh.m_PathID)
                    gpid = t["comp_go"].get(mfid)
                    if mo is not None and gpid is not None:
                        self._mesh_owner[mo.m_Name].append((fn, gpid))
                for mrid, r in t["mr"].items():
                    mm = getattr(r, "m_Mesh", None)
                    if mm is None or not mm.m_PathID:
                        continue
                    mo = t["mesh"].get(mm.m_PathID)
                    gpid = t["comp_go"].get(mrid)
                    if mo is not None and gpid is not None:
                        self._mesh_owner[mo.m_Name].append((fn, gpid))
        return self._mesh_owner.get(mesh_name) or []

    def _has_resolvable_mesh(self, fname, root_pid):
        t = self.tree(fname)
        g = t["go"].get(root_pid)
        if g is None:
            return False
        stack = [g]
        seen = 0
        while stack and seen < 20000:
            seen += 1
            cur = stack.pop()
            trp = cur.m_Transform.m_PathID if cur.m_Transform else 0
            tr = t["tr"].get(trp)
            if tr is None:
                continue
            for comp in (cur.m_Component or []):
                c = comp.component if hasattr(comp, "component") else comp
                fid = c.m_PathID
                if fid in t["mf"] and t["mesh"].get(t["mf"][fid].m_Mesh.m_PathID):
                    return True
                if fid in t["mr"]:
                    mm = getattr(t["mr"][fid], "m_Mesh", None)
                    if mm is not None and mm.m_PathID and t["mesh"].get(mm.m_PathID):
                        return True
            for ch in (tr.m_Children or []):
                ct = t["tr"].get(ch.m_PathID)
                if ct and ct.m_GameObject:
                    ng = t["go"].get(ct.m_GameObject.m_PathID)
                    if ng:
                        stack.append(ng)
        return False

    def find_root(self, name, hint, mesh_name=None):
        pool = []
        if mesh_name:
            seen = set()
            for fn, gpid in self.mesh_owners(mesh_name):
                if (fn, gpid) in seen:
                    continue
                seen.add((fn, gpid))
                go = self.tree(fn)["go"].get(gpid)
                if go is not None:
                    pool.append((fn, gpid, go))
        if not pool:
            pool = list(self.go_by_name.get(name) or [])
        if hint:
            hit = [c for c in pool if hint.lower() in c[0].lower()]
            pool = hit or pool
        fallback = None
        for fn, pid, obj in pool:
            g = self.tree(fn)["go"].get(pid)
            if g is None or g.m_Transform is None:
                continue
            if fallback is None:
                fallback = (fn, pid, obj)
            if self._has_resolvable_mesh(fn, pid):
                return fn, pid, obj
        return fallback

# --------------------------------------------------------------------------- mesh parts
def mesh_geometry(mesh):
    h = MeshHandler(mesh)
    h.process()
    return h.m_Vertices or [], h.m_UV0 or [], h.m_Normals or [], h.get_triangles()


def collect_parts(assets, fname, root_pid, exclude):
    """Walk the prefab subtree, returning mesh parts with baked transforms."""
    t = assets.tree(fname)
    root = t["go"][root_pid]
    parts = []

    def walk(g, m, is_root=False):
        trp = g.m_Transform.m_PathID if g.m_Transform else 0
        tr = t["tr"].get(trp)
        if tr is None:
            return
        if not is_root:
            m = mat_mul(m, local_matrix(tr))
        mi = mr = None
        for comp in (g.m_Component or []):
            c = comp.component if hasattr(comp, "component") else comp
            if c.m_PathID in t["mf"]:
                mi = c.m_PathID
            if c.m_PathID in t["mr"]:
                mr = c.m_PathID
        mesh = None
        if mi is not None:
            mesh = t["mesh"].get(t["mf"][mi].m_Mesh.m_PathID)
        if mesh is None and mr is not None:
            mm = getattr(t["mr"][mr], "m_Mesh", None)
            if mm is not None and mm.m_PathID:
                mesh = t["mesh"].get(mm.m_PathID)
        if mesh is not None:
            mats = []
            if mr is not None:
                for p in (t["mr"][mr].m_Materials or []):
                    try:
                        mats.append(p.read().m_Name)
                    except Exception:
                        pass
            parts.append((mesh, m, mats))
        for ch in (tr.m_Children or []):
            ct = t["tr"].get(ch.m_PathID)
            if not ct or not ct.m_GameObject:
                continue
            ng = t["go"].get(ct.m_GameObject.m_PathID)
            if ng is None:
                continue
            if exclude.search(ng.m_Name or ""):
                continue
            walk(ng, m)

    walk(root, IDENT, True)
    return parts


def emit_obj(parts, mat_names_by_part):
    """Serialise baked parts to OBJ text (UnityPy X-handedness convention)."""
    lines = []
    vb = tb = nb = 0
    for mesh, m, _mats, matname in mat_names_by_part:
        verts, uvs, normals, tris = mesh_geometry(mesh)
        if not verts:
            continue
        lines.append("g %s" % mesh.m_Name)
        for v in verts:
            p = xform_point(m, v[:3])
            lines.append("v %.9G %.9G %.9G" % (-p[0], p[1], p[2]))
        for uv in uvs:
            lines.append("vt %.9G %.9G" % (uv[0], uv[1]))
        for n in normals:
            d = xform_dir(m, n[:3])
            ln = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) or 1.0
            lines.append("vn %.9G %.9G %.9G" % (-d[0] / ln, d[1] / ln, d[2] / ln))
        if matname:
            lines.append("usemtl %s" % matname)
        nv, nt, nn = len(verts), len(uvs), len(normals)
        submeshes = tris if tris and isinstance(tris[0], (list, tuple)) else [tris]
        for tri_list in submeshes:
            for a, b, c in tri_list:
                if nt and nn:
                    lines.append("f %d/%d/%d %d/%d/%d %d/%d/%d" % (
                        vb + c + 1, tb + c + 1, nb + c + 1,
                        vb + b + 1, tb + b + 1, nb + b + 1,
                        vb + a + 1, tb + a + 1, nb + a + 1))
                else:
                    lines.append("f %d %d %d" % (vb + c + 1, vb + b + 1, vb + a + 1))
        vb += nv
        tb += nt
        nb += nn
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- materials
def material_texture(material):
    try:
        texenvs = material.m_SavedProperties.m_TexEnvs
    except Exception:
        return None
    for key in ("_MainTex", "_BaseMap", "_BaseColorMap", "_Albedo"):
        for k, v in texenvs:
            if k == key and v.m_Texture and v.m_Texture.m_PathID:
                try:
                    return v.m_Texture.read()
                except Exception:
                    return None
    for k, v in texenvs:
        if v.m_Texture and v.m_Texture.m_PathID:
            try:
                return v.m_Texture.read()
            except Exception:
                return None
    return None


def material_color(material):
    try:
        for k, v in material.m_SavedProperties.m_Colors:
            if k in ("_Color", "_BaseColor"):
                return (v.r, v.g, v.b)
    except Exception:
        pass
    return (1.0, 1.0, 1.0)


def save_texture(tex):
    """Save a Texture2D once into the shared props/tex pool (returns rel path)."""
    name = sanitize(tex.m_Name or "tex")
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, name + ".png")
    if not os.path.exists(path):
        try:
            tex.image.save(path)
        except Exception as e:
            log("  ! texture save failed %s: %r" % (name, e))
            return None
    return "tex/%s.png" % name


# --------------------------------------------------------------------------- prop export
def export_prop(assets, materials, spec, props_index):
    name = spec["name"]
    entry = OrderedDict(name=name, kind=spec["kind"], note=spec["note"])
    if "material" in spec:
        entry["material"] = spec["material"]

    found = assets.find_root(spec["root"], spec.get("file"),
                             spec.get("mesh", spec["root"]))
    if found is None:
        entry["missing"] = "prefab root '%s' not found in shipped data" % spec["root"]
        props_index.append(entry)
        return False
    fname, pid, _obj = found
    entry["source_prefab"] = spec["root"]
    entry["source_file"] = fname

    exclude = DEFAULT_EXCLUDE
    if spec.get("exclude"):
        exclude = re.compile(DEFAULT_EXCLUDE.pattern + "|" + spec["exclude"], re.I)

    try:
        parts = collect_parts(assets, fname, pid, exclude)
    except Exception as e:
        entry["missing"] = "prefab walk failed: %r" % (e,)
        props_index.append(entry)
        return False
    if not parts:
        entry["missing"] = "no visible mesh under prefab root '%s'" % spec["root"]
        props_index.append(entry)
        return False

    # resolve material for each part
    mat_cache = {}
    resolved = []
    used_mats = OrderedDict()
    for mesh, m, mats in parts:
        mn = None
        for cand in (mats or []) + ([spec.get("material")] if spec.get("material") else []):
            if not cand:
                continue
            obj = mat_cache.get(cand)
            if obj is None:
                cands = materials.get(cand) or []
                obj = cands[0][2].read() if cands else None
                mat_cache[cand] = obj
            if obj is not None:
                mn = sanitize(obj.m_Name)
                used_mats[obj.m_Name] = obj
                break
        if mn is None:
            mn = sanitize(spec.get("material") or "default")
        resolved.append((mesh, m, mats, mn))

    outdir = os.path.join(PROPS_DIR, name)
    os.makedirs(outdir, exist_ok=True)
    objtext = emit_obj(parts, resolved)
    objtext = "mtllib model.mtl\n" + objtext
    with open(os.path.join(outdir, "model.obj"), "w") as fh:
        fh.write(objtext)

    # bounds (in the exported/flipped frame)
    xs, ys, zs = [], [], []
    for line in objtext.splitlines():
        if line.startswith("v "):
            p = line.split()
            xs.append(float(p[1])); ys.append(float(p[2])); zs.append(float(p[3]))
    if xs:
        entry["bounds_min"] = [min(xs), min(ys), min(zs)]
        entry["bounds_max"] = [max(xs), max(ys), max(zs)]
        entry["size"] = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
        entry["vertex_count"] = len(xs)
        # lateral = X (full track width is 5.1 m, half width 2.55), vertical = Y
        lat, vert = entry["size"][0], entry["size"][1]
        entry["obstacle_scale_ok"] = (lat <= 5.2 and vert <= 4.5)

    # materials + textures
    mtl = ["# generated by tools/export_props.py", ""]
    tex_rel = []
    for matname, mat in used_mats.items():
        r, g, b = material_color(mat)
        mtl += ["newmtl %s" % sanitize(matname),
                "Kd %.6f %.6f %.6f" % (r, g, b),
                "Ka 0.000000 0.000000 0.000000",
                "Ks 0.000000 0.000000 0.000000", "d 1.0"]
        tex = material_texture(mat)
        if tex is not None:
            rel = save_texture(tex)
            if rel:
                mtl.append("map_Kd ../%s" % rel)
                tex_rel.append("props/%s" % rel)
        mtl.append("")
    with open(os.path.join(outdir, "model.mtl"), "w") as fh:
        fh.write("\n".join(mtl))

    entry["obj"] = "props/%s/model.obj" % name
    entry["mtl"] = "props/%s/model.mtl" % name
    entry["textures"] = list(OrderedDict.fromkeys(tex_rel))
    entry["mesh_parts"] = [p[0].m_Name for p in resolved]
    entry["scale"] = 1.0
    entry["yaw_offset_deg"] = 180.0
    props_index.append(entry)
    return True


# --------------------------------------------------------------------------- UI helpers
def export_texture_obj(obj, outpath):
    try:
        obj.read().image.save(outpath)
        return True
    except Exception:
        return False


_ATLAS_SRC = {}


def _atlas_source_image(atlas_name, sad):
    tp = sad.texture
    ap = getattr(sad, "alphaTexture", None)
    apid = getattr(ap, "m_PathID", 0) or 0
    key = (atlas_name, getattr(tp, "m_PathID", 0), apid)
    img = _ATLAS_SRC.get(key)
    if img is not None:
        return img
    img = get_image_from_texture2d(tp.deref_parse_as_object(), False)
    if apid:
        alpha = get_image_from_texture2d(ap.deref_parse_as_object(), False)
        img = Image.merge("RGBA", (*img.split()[:3], alpha.split()[0]))
    _ATLAS_SRC[key] = img
    return img


def sprite_image_via_atlas(sprite, atlas_by_name):
    tags = getattr(sprite, "m_AtlasTags", None) or []
    if not tags:
        return None
    name = tags[0]
    ao = atlas_by_name.get(name)
    if ao is None:
        return None
    atlas = ao.read()
    key = getattr(sprite, "m_RenderDataKey", None)
    sad = next((v for k, v in atlas.m_RenderDataMap if k == key), None)
    if sad is None:
        return None
    img = _atlas_source_image(name, sad)
    tr = sad.textureRect
    out = img.crop((int(tr.x), int(tr.y), int(tr.x + tr.width), int(tr.y + tr.height)))
    st = SpriteSettings(sad.settingsRaw)
    if st.packed:
        r = st.packingRotation
        if r == SpritePackingRotation.kSPRFlipHorizontal:
            out = out.transpose(Transpose.FLIP_LEFT_RIGHT)
        elif r == SpritePackingRotation.kSPRFlipVertical:
            out = out.transpose(Transpose.FLIP_TOP_BOTTOM)
        elif r == SpritePackingRotation.kSPRRotate180:
            out = out.transpose(Transpose.ROTATE_180)
        elif r == SpritePackingRotation.kSPRRotate90:
            out = out.transpose(Transpose.ROTATE_270)
    return out


def make_sprite_writer(atlas_by_name):
    def writer(obj, path):
        try:
            sprite = obj.read()
        except Exception:
            return False
        img = None
        try:
            img = sprite.image
        except Exception:
            img = None
        if img is None:
            try:
                img = sprite_image_via_atlas(sprite, atlas_by_name)
            except Exception:
                img = None
        if img is None:
            return False
        try:
            img.save(path)
        except Exception:
            return False
        return True

    return writer


def try_export(writer, candidates, outpath):
    for c in candidates or []:
        try:
            if writer(c[2], outpath):
                return c
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- main
def main():
    os.makedirs(TEX_DIR, exist_ok=True)
    os.makedirs(UI_DIR, exist_ok=True)

    log("loading Unity environment ...")
    env = UnityPy.load(DATA, BUNDLES)
    assets = Assets(env)
    log("objects by file:", len(assets.by_file))

    materials = defaultdict(list)
    textures = defaultdict(list)
    sprites = defaultdict(list)
    fonts = defaultdict(list)
    for o in env.objects:
        ty = o.type.name
        try:
            if ty == "Material":
                materials[o.read().m_Name].append((src_name(o), o.path_id, o))
            elif ty == "Texture2D":
                textures[o.read().m_Name].append((src_name(o), o.path_id, o))
            elif ty == "Sprite":
                sprites[o.read().m_Name].append((src_name(o), o.path_id, o))
            elif ty == "Font":
                fonts[o.read().m_Name].append((src_name(o), o.path_id, o))
        except Exception:
            continue

    # ---- props ------------------------------------------------------------
    props_index = []
    for spec in PROPS:
        ok = export_prop(assets, materials, spec, props_index)
        log(("  [ok]   " if ok else "  [MISS] ") + spec["name"])

    # sprite-only pickups are exported into ui/ further below --------------
    sprite_prop_names = []

    # ---- UI ---------------------------------------------------------------
    atlas_by_name = {}
    for o in env.objects:
        if o.type.name == "SpriteAtlas":
            try:
                atlas_by_name.setdefault(o.read().m_Name, o)
            except Exception:
                continue
    sprite_writer = make_sprite_writer(atlas_by_name)

    ui_entries = []
    ui_missing = []
    exported = set()

    def add_sprite(name, cands, out_sname=None, alias=None):
        sname = out_sname or sanitize(name)
        path = os.path.join(UI_DIR, sname + ".png")
        if sname in exported and os.path.exists(path):
            return True
        ok = try_export(sprite_writer, cands, path)
        if ok is None:
            return False
        exported.add(sname)
        ent = OrderedDict(kind="sprite", name=name, file="ui/%s.png" % sname,
                          source_file=ok[0])
        if alias:
            ent["alias"] = alias
        ui_entries.append(ent)
        return True

    for name, cands in sorted(sprites.items()):
        if not re.match(r"^[A-Za-z]", name or ""):
            continue
        if not add_sprite(name, cands):
            ui_missing.append(OrderedDict(
                kind="sprite", name=name,
                missing="no readable texture for any sprite instance in shipped data"))

    if not add_sprite("Icon_Pause", sprites.get("Icon_Pause"),
                      out_sname="pause", alias="pause"):
        ui_missing.append(OrderedDict(
            kind="sprite", name="Icon_Pause", alias="pause",
            missing="Icon_Pause Sprite ships with m_RD.texture pathID 0 (no "
                    "texture) and no SpriteAtlas; no standalone pause art in "
                    "data. Pause button art is available in ui/tex_HUD_Base.png"))

    # sprite-only pickups
    for pname, sprite_name, note in SPRITE_PROPS:
        entry = OrderedDict(name=pname, kind="pickup", note=note,
                            source_sprite=sprite_name)
        if add_sprite(sprite_name, sprites.get(sprite_name)):
            entry["png"] = "ui/%s.png" % sanitize(sprite_name)
            entry["source_file"] = next(
                e["source_file"] for e in reversed(ui_entries)
                if e["file"] == entry["png"])
        else:
            entry["missing"] = "sprite '%s' has no readable texture" % sprite_name
        props_index.append(entry)

    # UI textures
    ui_tex_entries = []
    seen_tex = set()
    for name, cands in sorted(textures.items()):
        sname = sanitize(name)
        if sname in seen_tex:
            continue
        has_data = any(not c[0].endswith(".unity3d") for c in cands)
        if has_data:
            if not UI_TEX_RE.search(name or ""):
                continue
        elif not UI_TEX_BUNDLE_RE.search(name or ""):
            continue
        path = os.path.join(UI_DIR, "tex_%s.png" % sname)
        ok = try_export(export_texture_obj, cands, path)
        if ok is None:
            ui_missing.append(OrderedDict(
                kind="texture", name=name,
                missing="streamed texture data not readable in shipped data"))
            continue
        seen_tex.add(sname)
        ui_tex_entries.append(OrderedDict(kind="texture", name=name,
                                          file="ui/tex_%s.png" % sname,
                                          source_file=ok[0]))

    # UI fonts
    font_entries = []
    seen_font = set()
    for name, cands in sorted(fonts.items()):
        if not name or name in seen_font:
            continue
        for c in cands:
            try:
                f = c[2].read()
            except Exception:
                continue
            data = getattr(f, "m_FontData", None)
            if not data:
                continue
            sname = sanitize(name)
            ttf = os.path.join(UI_DIR, "font_%s.ttf" % sname)
            try:
                with open(ttf, "wb") as fh:
                    fh.write(bytes(bytearray(data)))
            except Exception as e:
                log("  ! font write failed %s: %r" % (name, e))
                continue
            fam = getattr(f, "m_FontNames", None) or []
            ent = OrderedDict(kind="font", name=name, file="ui/font_%s.ttf" % sname,
                              family=(fam[0] if fam else None), source_file=c[0])
            tex = getattr(f, "m_Texture", None)
            if tex is not None and getattr(tex, "m_PathID", 0):
                try:
                    atlas = os.path.join(UI_DIR, "font_%s.png" % sname)
                    tex.read().image.save(atlas)
                    ent["atlas"] = "ui/font_%s.png" % sname
                except Exception:
                    pass
            seen_font.add(name)
            font_entries.append(ent)
            break

    # ---- manifests --------------------------------------------------------
    with open(os.path.join(PROPS_DIR, "manifest.json"), "w") as fh:
        json.dump(OrderedDict(
            generated_by="tools/export_props.py",
            source="recon/unity_assets (APK assets/bin/Data + Bundles/20_03)",
            frame=("prefab root frame, Unity axes with X flipped at export "
                   "(UnityPy handedness convention); lateral=X, vertical=Y, "
                   "longitudinal=-Z"),
            tex_dir="props/tex",
            mtl_convention=("model.mtl references its diffuse texture as "
                            "'../tex/<name>.png' relative to the prop's own "
                            "directory, into the shared props/tex/ pool"),
            notes=[
                "Geometry is built from the prefab root GameObject: every "
                "visible mesh child has its local->root TRS baked in; scenery/"
                "effect/dummy/collision shadow nodes are excluded.",
                "Machu base-zone obstacle pieces (jump_over_a, slide_under_a, "
                "stumble_a) bake the obstacle together with the track floor and "
                "surrounding terrain into one 15-20 m mesh, so obstacle kinds "
                "use the shipped separable obstacle props instead.",
                "obstacle_scale_ok: lateral X <= 5.2 m (full track width, half "
                "width 2.55) and vertical Y <= 4.5 m.",
            ],
            props=props_index,
        ), fh, indent=1)

    with open(os.path.join(UI_DIR, "manifest.json"), "w") as fh:
        json.dump(OrderedDict(
            generated_by="tools/export_props.py",
            source="recon/unity_assets (APK assets/bin/Data + Bundles/20_03/HUDs)",
            keywords=UI_KEYWORDS,
            sprites=ui_entries,
            textures=ui_tex_entries,
            fonts=font_entries,
            missing=ui_missing,
        ), fh, indent=1)

    log("props:", len(props_index), "ui sprites:", len(ui_entries),
        "ui textures:", len(ui_tex_entries), "fonts:", len(font_entries),
        "ui missing:", len(ui_missing))


if __name__ == "__main__":
    main()
