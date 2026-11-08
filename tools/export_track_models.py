#!/usr/bin/env python3
"""Export Temple Run 2 track pieces from Unity asset bundles.

Reads ``recon/track/corpus.json`` (one row per ``TrackPieceInfo`` variation),
loads the variation's bundle with UnityPy, finds the prefab ``GameObject`` by
``prefab_path_id``, walks its ``Transform`` hierarchy and bakes every
``MeshFilter``/``MeshRenderer`` under it into a single OBJ per variation.

Layout (see ``resources/templerun2/ASSET_CONTRACT.md``)::

    resources/templerun2/models/track/<safe_name>/model.obj
    resources/templerun2/models/track/<safe_name>/model.mtl
    resources/templerun2/models/track/tex/<texture_name>.png   (shared)
    resources/templerun2/models/track/manifest.json

Textures are deduplicated across every variation into the single shared
``track/tex/`` directory; each MTL points at them with ``../tex/<name>.png``
(raylib resolves ``map_Kd`` relative to the MTL file, and the runtime loads the
OBJ through ``LoadModel()`` with a repo-root-relative path).

Conventions
-----------
* Vertices retain Unity axes in the prefab root frame; the runtime reflects X
  and applies the manifest's 180 degree yaw plus the segment heading. At
  heading h a local point (x, y, z) maps to
  (x*cos(h) - z*sin(h), y, -x*sin(h) - z*cos(h)), matching the reflected
  authored spline. Bounds are recorded in the original prefab frame.
* The prefab root's own local transform is the frame origin (not applied); the
  full local position/quaternion/scale chain is baked into every vertex.
* UVs retain Unity's V coordinate; obj_to_glb.py flips V once for glTF.
* Materials are deduplicated per variation.

Scenery filtering
-----------------
Piece prefabs bundle the track structure together with scenery (trees, props,
holiday decorations, effects) and, in several zones, terrain mass that reaches
far below/outside the path.  Two stages remove that:

1. ``select_nodes`` drops children whose name matches ``SKIP_NAME_TOKENS`` and,
   when any child matches ``PREFER_NAME_TOKENS`` (path/ground/wall structure),
   keeps only those.  Matching is substring-based, so the skip list has to
   cover both ``prop_*`` and ``*_prop`` spellings and the ``*_Background``
   backdrop meshes that would otherwise match the ``ground`` token.
2. ``Corridor`` clips triangles to the play corridor implied by the piece's
   authored spline: within ``CORRIDOR_LAT`` horizontally, ``CORRIDOR_UP``
   above and ``CORRIDOR_DOWN`` below the spine, and no further than
   ``CORRIDOR_MARGIN`` past either end.  If a piece would be clipped empty it
   is re-exported unclipped so no variation ships an empty model.

Bounds of every straight piece are validated against ``estimated_path_length``
(tolerance 10 percent) plus x/y sanity; the result lands in ``validation`` in
the manifest.

Materials
---------
A material PPtr that resolves to no asset (its serialized file is not in the
bundle set) is **not** substituted.  The OBJ still needs a ``usemtl`` name, so
the name is written, the MTL carries that ``newmtl`` plus a comment saying the
asset is missing (no invented ``Kd``/``Ks``/``Ns``/``d`` values), and the
manifest records ``unresolved_materials`` per entry plus a top-level summary
with the missing cab names and the dependency bundles named by the owning
bundle's ``*.manifest``.

Requires the UnityPy venv: ``recon/unity_assets/.venv/bin/python``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import traceback
from pathlib import Path

import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "recon/track/corpus.json"
DEFAULT_BUNDLES = ROOT / "recon/unity_assets/allbundles"
DEFAULT_OUT = ROOT / "resources/templerun2/models/track"

YAW_OFFSET_DEG = 180.0
SCALE = 1.0

_CAB_RE = re.compile(r"CAB-[0-9a-f]{32}")
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sanitize(name: str, fallback: str = "unnamed") -> str:
    s = _SAFE_RE.sub("_", (name or "").strip())
    s = s.strip("._")
    return s or fallback


def fmt(x: float) -> str:
    if not math.isfinite(x):
        x = 0.0
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def quat_rows(x: float, y: float, z: float, w: float):
    """Row-major 3x3 rotation matrix from a Unity quaternion (x, y, z, w)."""
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)),
        (2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)),
        (2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)),
    )


def local_matrix(tr):
    """Unity local->parent matrix ``T * R * S`` as a row-major 4x4 tuple."""
    p, r, s = tr.m_LocalPosition, tr.m_LocalRotation, tr.m_LocalScale
    rot = quat_rows(r.x, r.y, r.z, r.w)
    sc = (s.x, s.y, s.z)
    return (
        (rot[0][0] * sc[0], rot[0][1] * sc[1], rot[0][2] * sc[2], p.x),
        (rot[1][0] * sc[0], rot[1][1] * sc[1], rot[1][2] * sc[2], p.y),
        (rot[2][0] * sc[0], rot[2][1] * sc[1], rot[2][2] * sc[2], p.z),
        (0.0, 0.0, 0.0, 1.0),
    )


IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

# Prefab children whose name contains any of these are scenery/effects, not
# track structure, and are dropped before export.
#
# Matching is a plain substring test, so the tokens have to cover both spellings
# the shipped prefabs use: ``prop_`` for the ``prop_pumpkin`` style names and
# ``_prop`` for the far more common ``PirateCove_Barrel_prop (3)`` suffix (3945
# mesh instances across the corpus).  ``background`` is listed explicitly so the
# distant backdrop meshes (``<piece>_Background``) are dropped in this stage;
# otherwise the ``ground`` PREFER token below matches inside "Back*ground*" and
# hijacks the preferred set, discarding the piece's real road mesh.
#
# There is deliberately no ``shadow`` token: every ``<piece>_Shadow`` node in
# the corpus (10 distinct names) is the walkable ground/road surface, textured
# from the opaque world atlas or the sand atlas - e.g.
# ``PirateCove_Docks_Straight_A_Shadow`` -> ``Atlas_Opaque_Pirate01`` with
# ``Atlas_Opaque_PirateCove.png``, ``PirateCove_Beach_Straight_Organic_A_Shadow``
# -> ``Pirate_Ocean_Shore_A`` with ``Pirate_Sand_A.png``.  Dropping them left 29
# variations with no ground at all.
SKIP_NAME_TOKENS = (
    "floating_island", "island", "deco", "prop_", "_prop", "dummy_", "effect", "particle",
    "trail", "sound", "light", "candle", "pumpkin", "tombstone", "scenery",
    "sky", "cloud", "vine", "grass", "tree", "rock", "bush", "background",
    "collision", "trigger", "invis", "occluder", "blocker",
    "debug", "gizmo", "bounds",
)

# When a piece has any child matching these, only those children are exported
# (path / ground / wall structure); otherwise the non-scenery children are kept.
PREFER_NAME_TOKENS = (
    "path", "ground", "wall", "straight", "curve", "turn", "bridge", "ledge",
    "stairs", "track", "floor", "platform",
)

# Play-corridor envelope around the authored spline.  Triangles with any vertex
# outside are dropped, which removes terrain/scenery mass that the piece meshes
# embed (deep rock faces, surrounding cliffs, oversized arches) while keeping
# the runnable surface, its walls and the obstacles standing on it.
CORRIDOR_LAT = 4.0  # max horizontal distance from the spline polyline
CORRIDOR_UP = 4.0  # max height above the spline
CORRIDOR_DOWN = 3.0  # max depth below the spline
CORRIDOR_MARGIN = 0.35  # longitudinal slack past either spline end

# Initialised in main(); the PPtr resolver and the shared texture writer need
# process-wide access without threading an extra argument through every call.
BUNDLE_SET = None  # type: ignore[assignment]
TEX_STORE = None  # type: ignore[assignment]


def is_scenery(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in SKIP_NAME_TOKENS)


def is_preferred(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in PREFER_NAME_TOKENS)


class Corridor:
    """Play-corridor test built from a piece's authored spline."""

    def __init__(self, points, lat=CORRIDOR_LAT, up=CORRIDOR_UP, down=CORRIDOR_DOWN, margin=CORRIDOR_MARGIN):
        self.lat = lat
        self.up = up
        self.down = down
        self.margin = margin
        self.segs = []
        total = 0.0
        for a, b in zip(points, points[1:]):
            dx, dz = b[0] - a[0], b[2] - a[2]
            seg_len = math.hypot(dx, dz)
            self.segs.append((a[0], a[1], a[2], b[1], dx, dz, seg_len, total))
            total += seg_len
        self.total = total
        self.ok = len(self.segs) > 0

    def keep(self, x: float, y: float, z: float) -> bool:
        """True when a prefab-root-frame point lies inside the corridor."""
        if not self.ok:
            return True
        best_h = float("inf")
        best_y = 0.0
        best_along = 0.0
        for ax, ay, az, by, dx, dz, seg_len, cum in self.segs:
            if seg_len <= 1e-9:
                t_raw = 0.0
            else:
                # unclamped projection drives the longitudinal test; the
                # clamped one drives lateral/vertical distance
                t_raw = ((x - ax) * dx + (z - az) * dz) / (seg_len * seg_len)
            t = 0.0 if t_raw < 0.0 else (1.0 if t_raw > 1.0 else t_raw)
            h = math.hypot(x - (ax + dx * t), z - (az + dz * t))
            if h < best_h:
                best_h = h
                best_y = ay + (by - ay) * t
                best_along = cum + seg_len * t_raw
        if best_h > self.lat:
            return False
        dy = y - best_y
        if dy > self.up or dy < -self.down:
            return False
        return -self.margin <= best_along <= self.total + self.margin




def mat_mul(a, b):
    return tuple(
        tuple(a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] + a[r][3] * b[3][c] for c in range(4))
        for r in range(4)
    )


def xform(m, v):
    x, y, z = v
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    )


def xform_dir(m, v):
    x, y, z = v
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    )


# --------------------------------------------------------------------------- #
# bundle access
# --------------------------------------------------------------------------- #
def read_declared_dependencies(bundle_path: Path):
    """Parse the ``Dependencies:`` list from a sibling ``*.unity3d.manifest``.

    Unity writes a text manifest next to every asset bundle listing the other
    bundles it references.  When a material PPtr points at a cab that is not on
    disk, the declared dependency list is the only evidence of *which* bundle
    the missing asset lives in, so it is worth carrying into the manifest.
    """
    manifest = bundle_path.with_name(bundle_path.name + ".manifest")
    if not manifest.is_file():
        return []
    deps = []
    in_block = False
    try:
        for line in manifest.read_text(errors="replace").splitlines():
            if line.startswith("Dependencies:"):
                in_block = True
                continue
            if in_block:
                if line.startswith("- "):
                    deps.append(line[2:].strip().replace("\\", "/").rsplit("/", 1)[-1])
                elif line.strip() and not line.startswith((" ", "\t")):
                    break
    except OSError:
        return []
    return deps


class Bundle:
    """One ``.unity3d`` file plus per-serialized-file object lookups."""

    def __init__(self, path: Path):
        self.path = path
        self.env = UnityPy.load(str(path))
        self.declared_dependencies = read_declared_dependencies(path)
        self.files = {}  # cab name -> SerializedFile
        self.objects = {}  # (cab name, path_id) -> ObjectReader
        for bf in self.env.files.values():
            files = getattr(bf, "files", None)
            if not isinstance(files, dict):
                continue
            for name, sf in files.items():
                if name.endswith((".resS", ".resource")):
                    continue
                if not hasattr(sf, "objects"):
                    continue
                self.files[name] = sf
                for pid, obj in sf.objects.items():
                    self.objects[(name, pid)] = obj

    def get(self, cab: str, pid: int):
        return self.objects.get((cab, pid))


class BundleSet:
    """Every bundle on disk plus a ``cab name -> bundle`` index."""

    def __init__(self, root: Path):
        self.root = root
        self.bundles = {}  # bundle path -> Bundle
        self._by_cab = {}
        self._by_sf = {}  # id(SerializedFile) -> Bundle
        self._scan_all()

    def _scan_all(self):
        for path in sorted(self.root.rglob("*.unity3d")):
            self._load(path)

    def _load(self, path: Path) -> Bundle:
        b = self.bundles.get(Path(path))
        if b is None:
            b = Bundle(Path(path))
            self.bundles[Path(path)] = b
            for cab, sf in b.files.items():
                self._by_cab.setdefault(cab, b)
                self._by_sf[id(sf)] = b
        return b

    def for_sf(self, sf):
        return self._by_sf.get(id(sf))

    def for_cab(self, cab: str):
        return self._by_cab.get(cab)

    def for_path(self, path: Path):
        path = Path(path)
        if path in self.bundles:
            return self.bundles[path]
        return self._load(path)


def pptr(sf, ptr):
    """Resolve a PPtr to an ``ObjectReader`` (None for null/unresolvable)."""
    pid = getattr(ptr, "m_PathID", 0)
    fid = getattr(ptr, "m_FileID", 0)
    if not pid:
        return None
    bundle = BUNDLE_SET.for_sf(sf)
    if fid == 0:
        if bundle is None:
            return None
        return bundle.get(sf.name, pid)
    # external reference -> a dependency cab from another bundle
    try:
        externals = sf.externals
    except Exception:
        return None
    if fid < 1 or fid > len(externals):
        return None
    cabs = _CAB_RE.findall(externals[fid - 1].path or "")
    if cabs:
        dep = BUNDLE_SET.for_cab(cabs[-1])
        if dep is not None:
            return dep.get(cabs[-1], pid)
    # builtin resources ("Library/unity default resources" etc.)
    return None


def read_obj(reader):
    if reader is None:
        return None
    try:
        return reader.read()
    except Exception:
        return None


def describe_external(sf, ptr):
    """Describe an unresolvable external PPtr for the manifest.

    Returns ``{file_id, path_id, cab, dependency, owner_dependencies}`` where
    ``cab`` is the serialized-file name the reference points at (``None`` for
    builtin resources), ``owner_dependencies`` lists the bundles the owning
    bundle declares, and ``dependency`` names the missing bundle when the
    declared list pins it down unambiguously (exactly one entry).
    """
    fid = getattr(ptr, "m_FileID", 0)
    info = {
        "file_id": fid,
        "path_id": getattr(ptr, "m_PathID", 0),
        "cab": None,
        "dependency": None,
        "owner_dependencies": [],
    }
    if fid == 0:
        return info
    try:
        externals = sf.externals
    except Exception:
        return info
    if fid < 1 or fid > len(externals):
        return info
    path = externals[fid - 1].path or ""
    cabs = _CAB_RE.findall(path)
    if cabs:
        info["cab"] = cabs[-1]
    bundle = BUNDLE_SET.for_sf(sf)
    if bundle is not None and bundle.declared_dependencies:
        info["owner_dependencies"] = list(bundle.declared_dependencies)
        if len(bundle.declared_dependencies) == 1:
            info["dependency"] = bundle.declared_dependencies[0]
    return info


# --------------------------------------------------------------------------- #
# material / texture export
# --------------------------------------------------------------------------- #
def saved_properties(mat):
    sp = getattr(mat, "m_SavedProperties", None)
    colors, floats, texenvs = {}, {}, {}
    if sp is None:
        return colors, floats, texenvs
    for k, v in getattr(sp, "m_Colors", None) or []:
        colors[k] = v
    for k, v in getattr(sp, "m_Floats", None) or []:
        floats[k] = v
    for k, v in getattr(sp, "m_TexEnvs", None) or []:
        texenvs[k] = v
    return colors, floats, texenvs


class TextureStore:
    """Process-wide shared texture directory (``track/tex/``).

    Textures are deduplicated by ``path_id`` (asset identity), so every model
    references the same PNG through ``../tex/<name>.png``.
    """

    def __init__(self, tex_dir: Path):
        self.dir = tex_dir
        self.by_pid = {}  # path_id -> file name
        self.by_name = {}  # file name -> path_id
        self.written = 0
        self.failed = []

    def get(self, tex, path_id, warn_ctx):
        """Export (once) and return the PNG file name, or None on failure."""
        cached = self.by_pid.get(path_id)
        if cached:
            return cached
        base = sanitize(getattr(tex, "m_Name", "") or f"tex_{path_id}", "texture")
        fname = f"{base}.png"
        if fname in self.by_name and self.by_name[fname] != path_id:
            fname = f"{base}_{abs(path_id)}.png"
        try:
            image = tex.image
        except Exception as exc:  # unsupported texture format
            self.failed.append(f"{warn_ctx}: texture '{base}' decode failed: {exc}")
            return None
        if image is None:
            self.failed.append(f"{warn_ctx}: texture '{base}' has no decodable image")
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            image.save(str(self.dir / fname))
        except Exception as exc:
            self.failed.append(f"{warn_ctx}: texture '{base}' save failed: {exc}")
            return None
        self.by_pid[path_id] = fname
        self.by_name[fname] = path_id
        self.written += 1
        return fname


class VariationWriter:
    """Accumulates one variation's OBJ/MTL and dedups assets."""

    def __init__(self, out_dir: Path, textures: TextureStore):
        self.dir = out_dir
        self.textures = textures
        self.v = []  # (x, y, z)
        self.vt = []  # (u, v)
        self.vn = []  # (x, y, z)
        self.faces = []  # (material name, [((v, vt, vn) x3)]) 1-based, 0 = absent
        self.materials = {}  # name -> dict(props) | None
        self.material_order = []
        self.textures_used = []  # shared '<name>.png' files referenced
        self.unresolved = []  # material PPtrs that could not be resolved to an asset
        self.stats = {"renderers": 0, "meshes": 0, "tris": 0, "missing_materials": 0, "missing_textures": 0}
        self.warnings = []

    # -- materials -------------------------------------------------------- #
    def register_material(self, mat_reader, sf, ptr, warn_ctx):
        """Return the (deduped) material name used by the MTL."""
        mat = read_obj(mat_reader)
        if mat is None:
            name = sanitize(f"unresolved_mat_f{ptr.m_FileID}_p{ptr.m_PathID}", "material")
            if name not in self.materials:
                ref = describe_external(sf, ptr)
                self.materials[name] = None
                self.material_order.append(name)
                self.stats["missing_materials"] += 1
                entry = {"name": name, "object": warn_ctx}
                entry.update(ref)
                self.unresolved.append(entry)
                detail = f"cab {ref['cab']}" if ref["cab"] else f"file {ref['file_id']}"
                dep = f"; declared dependency {ref['dependency']}" if ref["dependency"] else ""
                self.warnings.append(
                    f"{warn_ctx}: material pid {ptr.m_PathID} unresolved ({detail}{dep})"
                )
            return name

        name = sanitize(mat.m_Name, "material")
        base = name
        n = 2
        while name in self.materials and self.materials[name] is not None:
            existing = self.materials[name]
            if existing.get("_pid") == ptr.m_PathID:
                return name
            name = f"{base}_{n}"
            n += 1

        colors, floats, texenvs = saved_properties(mat)
        info = {
            "_pid": ptr.m_PathID,
            "color": colors.get("_Color"),
            "spec": colors.get("_SpecColor"),
            "shininess": floats.get("_Shininess"),
            "map_kd": None,
        }

        main = texenvs.get("_MainTex")
        if main is not None:
            tex_ptr = main.m_Texture
            # the texture pptr is relative to the material's own serialized file
            try:
                tex_sf = mat.assets_file
            except Exception:
                tex_sf = sf
            tex_reader = pptr(tex_sf, tex_ptr)
            tex = read_obj(tex_reader)
            if tex is not None:
                png = self.textures.get(tex, tex_ptr.m_PathID, warn_ctx)
                if png:
                    info["map_kd"] = f"../tex/{png}"
                    if png not in self.textures_used:
                        self.textures_used.append(png)
            else:
                self.stats["missing_textures"] += 1
                self.warnings.append(f"{warn_ctx}: _MainTex pid {tex_ptr.m_PathID} unresolved")

        self.materials[name] = info
        self.material_order.append(name)
        return name

    # -- geometry --------------------------------------------------------- #
    def add_renderer(self, mesh, matrix, goname, material_names, warn_ctx, corridor=None):
        handler = MeshHandler(mesh)
        handler.process()
        verts = handler.m_Vertices
        if not verts:
            return False
        uvs = handler.m_UV0
        normals = handler.m_Normals
        world_v = [xform(matrix, (p[0], p[1], p[2])) for p in verts]
        world_n = None
        if normals:
            world_n = []
            for n in normals:
                d = xform_dir(matrix, (n[0], n[1], n[2]))
                ln = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) or 1.0
                world_n.append((d[0] / ln, d[1] / ln, d[2] / ln))
        inside = None
        if corridor is not None and corridor.ok:
            inside = [corridor.keep(*p) for p in world_v]

        triangles = handler.get_triangles()
        submeshes = mesh.m_SubMeshes or []
        kept = []  # (material name, [(a, b, c)] local vertex indices)
        for i, tris in enumerate(triangles):
            if not tris:
                continue
            mat_name = material_names[i] if i < len(material_names) else (material_names[-1] if material_names else None)
            bv = 0
            if i < len(submeshes):
                bv = getattr(submeshes[i], "baseVertex", 0) or 0
            out = []
            for a, b, c in tris:
                ia, ib, ic = a + bv, b + bv, c + bv
                if inside is not None and not (inside[ia] and inside[ib] and inside[ic]):
                    continue
                out.append((ia, ib, ic))
            if out:
                kept.append((mat_name, out))
        if not kept:
            return False

        # 1-based index of the first vertex of this mesh inside the OBJ
        first = len(self.v)
        used = sorted({i for _, tris in kept for tri in tris for i in tri})
        # Each local vertex gets an index into every attribute array it has.
        # The arrays are only parallel when *every* mesh in the piece carries
        # that attribute, so the attribute index has to be recorded per vertex
        # instead of assumed equal to the position index.
        remap = {}
        for local in used:
            vi = first + len(remap) + 1
            vt_i = 0
            vn_i = 0
            self.v.append(world_v[local])
            if uvs:
                # obj_to_glb converts the Unity V coordinate for glTF.
                self.vt.append((uvs[local][0], uvs[local][1]))
                vt_i = len(self.vt)
            if world_n is not None:
                self.vn.append(world_n[local])
                vn_i = len(self.vn)
            remap[local] = (vi, vt_i, vn_i)

        for mat_name, tris in kept:
            self.faces.append((mat_name, [(remap[a], remap[b], remap[c]) for a, b, c in tris]))
            self.stats["tris"] += len(tris)
        self.stats["meshes"] += 1
        return True


    # -- output ----------------------------------------------------------- #
    def write(self, header_comment: str):
        if not self.v or not self.faces:
            return False
        self.dir.mkdir(parents=True, exist_ok=True)

        lines = [f"# {header_comment}\n", "mtllib model.mtl\n", "o track\n"]
        for x, y, z in self.v:
            lines.append(f"v {fmt(x)} {fmt(y)} {fmt(z)}\n")
        for u, v in self.vt:
            lines.append(f"vt {fmt(u)} {fmt(v)}\n")
        for x, y, z in self.vn:
            lines.append(f"vn {fmt(x)} {fmt(y)} {fmt(z)}\n")

        def face_tok(v, vt, vn):
            if vt and vn:
                return f"{v}/{vt}/{vn}"
            if vt:
                return f"{v}/{vt}"
            if vn:
                return f"{v}//{vn}"
            return f"{v}"

        for mat_name, tris in self.faces:
            if mat_name:
                lines.append(f"usemtl {mat_name}\n")
            for a, b, c in tris:
                lines.append("f " + " ".join(face_tok(*t) for t in (a, b, c)) + "\n")
        (self.dir / "model.obj").write_text("".join(lines))

        # MTL
        mtl = ["# Temple Run 2 track material library\n"]
        for name in self.material_order:
            info = self.materials[name]
            mtl.append(f"newmtl {name}\n")
            if not info:
                # The PPtr pointed at a serialized file that is not on disk, so
                # no properties exist to export.  Write the name only (the OBJ
                # needs it for `usemtl`) and say so, rather than inventing a
                # plausible opaque-white material.
                ref = next((u for u in self.unresolved if u["name"] == name), None)
                where = ""
                if ref is not None:
                    where = f" (cab {ref['cab']})" if ref["cab"] else f" (file {ref['file_id']})"
                    if ref["dependency"]:
                        where += f"; missing dependency bundle {ref['dependency']}"
                mtl.append(f"# unresolved material - no asset available{where}\n\n")
                continue
            mtl.append("Ka 0 0 0\n")
            col = info.get("color")
            if col is not None:
                mtl.append(f"Kd {fmt(col.r)} {fmt(col.g)} {fmt(col.b)}\n")
                alpha = float(col.a)
            else:
                mtl.append("Kd 1 1 1\n")
                alpha = 1.0
            spec = info.get("spec")
            if spec is not None:
                mtl.append(f"Ks {fmt(spec.r)} {fmt(spec.g)} {fmt(spec.b)}\n")
            else:
                mtl.append("Ks 0 0 0\n")
            shin = info.get("shininess")
            mtl.append(f"Ns {fmt(float(shin) if shin is not None else 0.0)}\n")
            mtl.append(f"d {fmt(alpha)}\n")
            if info.get("map_kd"):
                mtl.append(f"map_Kd {info['map_kd']}\n")
            mtl.append("illum 1\n\n")
        (self.dir / "model.mtl").write_text("".join(mtl))

        lo = [min(v[i] for v in self.v) for i in range(3)]
        hi = [max(v[i] for v in self.v) for i in range(3)]
        return lo, hi


# --------------------------------------------------------------------------- #
# prefab walk
# --------------------------------------------------------------------------- #
def find_gameobject(bundle: Bundle, path_id: int):
    for (cab, pid), reader in bundle.objects.items():
        if pid == path_id and reader.type.name == "GameObject":
            return reader
    return None


def collect_nodes(tr_reader, matrix, out, depth=0):
    """Depth-first walk collecting every MeshRenderer under the prefab root."""
    tr = read_obj(tr_reader)
    if tr is None:
        return
    sf = tr_reader.assets_file
    matrix = mat_mul(matrix, local_matrix(tr))

    go_reader = pptr(sf, tr.m_GameObject)
    go = read_obj(go_reader)
    if go is not None:
        mf = mr = None
        for c in go.m_Component or []:
            cp = getattr(c, "component", c)
            r = pptr(sf, cp)
            if r is None:
                continue
            tname = r.type.name
            if tname == "MeshFilter" and mf is None:
                mf = r
            elif tname == "MeshRenderer" and mr is None:
                mr = r
        if mr is not None:
            out.append((go.m_Name or "node", mf, mr, matrix))

    for child in tr.m_Children or []:
        cr = pptr(sf, child)
        if cr is not None:
            collect_nodes(cr, matrix, out, depth + 1)


def select_nodes(nodes, warn_ctx, warnings):
    """Drop scenery children, preferring path/ground/wall structure."""
    kept = [n for n in nodes if not is_scenery(n[0])]
    dropped = len(nodes) - len(kept)
    preferred = [n for n in kept if is_preferred(n[0])]
    if preferred:
        if dropped or len(preferred) != len(kept):
            warnings.append(f"{warn_ctx}: kept {len(preferred)}/{len(nodes)} prefab MeshRenderers (scenery filtered)")
        return preferred
    if kept:
        if dropped:
            warnings.append(f"{warn_ctx}: no name-preferred structure; kept {len(kept)}/{len(nodes)} prefab MeshRenderers")
        return kept
    # Every MeshRenderer name matched a skip token.  Shipping an empty piece is
    # worse than shipping scenery, so fall back to the unfiltered set (the same
    # last-resort rule the corridor clip uses).
    warnings.append(f"{warn_ctx}: all {len(nodes)} prefab MeshRenderers matched scenery tokens; exported unfiltered")
    return nodes


def _emit_renderer(writer, mf_reader, mr_reader, matrix, ctx, goname, corridor=None):
    tag = f"{ctx['variation']}/{goname}"
    mr = read_obj(mr_reader)
    if mr is None:
        return False
    mr_sf = mr_reader.assets_file

    mat_names = []
    for mptr in mr.m_Materials or []:
        mat_reader = pptr(mr_sf, mptr)
        mat_names.append(writer.register_material(mat_reader, mr_sf, mptr, tag))

    mesh = None
    if mf_reader is not None:
        mf = read_obj(mf_reader)
        if mf is not None and getattr(mf, "m_Mesh", None) is not None:
            if mf.m_Mesh.m_PathID:
                mesh = read_obj(pptr(mf_reader.assets_file, mf.m_Mesh))
    if mesh is None:
        writer.warnings.append(f"{tag}: MeshRenderer without mesh")
        return False
    try:
        return bool(writer.add_renderer(mesh, matrix, goname, mat_names, tag, corridor))
    except Exception as exc:
        writer.warnings.append(f"{tag}: mesh '{getattr(mesh,'m_Name','')}' export failed: {exc}")
        return False


def export_prefab(writer, root_tr, ctx, corridor=None):
    """Export every selected renderer of one prefab into ``writer``."""
    nodes = []
    collect_nodes(root_tr, IDENTITY, nodes)
    selected = select_nodes(nodes, ctx["variation"], writer.warnings)
    writer.stats["renderers"] = len(selected)
    if not selected:
        return False
    any_mesh = False
    for goname, mf, mr, matrix in selected:
        if _emit_renderer(writer, mf, mr, matrix, ctx, goname, corridor):
            any_mesh = True
    return any_mesh


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
def flatten_corpus(corpus):
    rows = []
    idx = 0
    for piece in corpus["pieces"]:
        for var in piece["variations"]:
            rows.append(
                {
                    "var_index": idx,
                    "name": var.get("name") or "",
                    "prefab_path_id": var.get("prefab_path_id"),
                    "bundle": var.get("bundle") or "",
                    "matched": bool(var.get("matched")),
                    "piece": piece.get("key", ""),
                    "zone": piece.get("zone_name", ""),
                    "category": piece.get("category_name", ""),
                    "flags": piece.get("flags", {}),
                    "est_len": var.get("estimated_path_length") or 0.0,
                    "spline": var.get("spline_generated") or var.get("spline") or [],
                }
            )
            idx += 1
    return rows


def _is_straight(row) -> bool:
    flags = row.get("flags") or {}
    turning = ("IsTurnLeft", "IsTurnRight", "IsCurveLeft", "IsCurveRight", "IsJumpOver", "IsSlideUnder",
               "IsStumble", "IsJunction", "IsYJunction", "IsOptionalJunction", "IsZipLine", "IsWaterSlide")
    if any(flags.get(k) for k in turning):
        return False
    name = row["name"].lower()
    return ("straight" in name or "path" in name) and bool(row.get("est_len"))


def _validate(acc, row, lo, hi) -> None:
    """Sanity-check a straight piece against the authored path length."""
    acc["straights"] += 1
    est = row["est_len"]
    z_span = abs(hi[2] - lo[2])
    x_max = max(abs(lo[0]), abs(hi[0]))
    y_lo, y_hi = lo[1], hi[1]
    z_ok = abs(z_span - est) <= 0.1 * est
    y_ok = y_lo >= -3.0 and y_hi <= 5.0
    x_ok = x_max <= 4.5
    acc["z_ok"] += int(z_ok)
    acc["y_ok"] += int(y_ok)
    acc["x_ok"] += int(x_ok)
    acc["all_ok"] += int(z_ok and y_ok and x_ok)
    if not (z_ok and y_ok and x_ok):
        acc["failures"].append(
            {
                "var_index": row["var_index"],
                "variation": row["name"],
                "est_len": round(est, 2),
                "z_span": round(z_span, 2),
                "x_max": round(x_max, 2),
                "y": [round(y_lo, 2), round(y_hi, 2)],
                "z_ok": z_ok,
                "y_ok": y_ok,
                "x_ok": x_ok,
            }
        )


def bundle_path_for(rel: str) -> Path:
    rel = (rel or "").replace("\\", "/")
    if rel.startswith("bundles/"):
        rel = rel[len("bundles/") :]
    return BUNDLE_SET.root / rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--bundles", default=str(DEFAULT_BUNDLES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", default=None, help="regex filter on variation name")
    ap.add_argument("--limit", type=int, default=0, help="max variations to export (0 = all)")
    ap.add_argument("--lat", type=float, default=CORRIDOR_LAT, help="corridor max lateral distance from the spline")
    ap.add_argument("--up", type=float, default=CORRIDOR_UP, help="corridor max height above the spline")
    ap.add_argument("--down", type=float, default=CORRIDOR_DOWN, help="corridor max depth below the spline")
    ap.add_argument("--margin", type=float, default=CORRIDOR_MARGIN, help="corridor longitudinal slack")
    ap.add_argument("--no-clip", action="store_true", help="disable corridor clipping")
    args = ap.parse_args(argv)

    global BUNDLE_SET, TEX_STORE
    BUNDLE_SET = BundleSet(Path(args.bundles))

    corpus = json.loads(Path(args.corpus).read_text())
    rows = flatten_corpus(corpus)
    if args.only:
        rx = re.compile(args.only)
        rows = [r for r in rows if rx.search(r["name"])]
    if args.limit:
        rows = rows[: args.limit]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    TEX_STORE = TextureStore(out_root / "tex")

    # Drop per-model texture folders left by earlier layouts; textures now live
    # in the single shared track/tex/ directory.
    for child in out_root.iterdir():
        if child.is_dir() and (child / "tex").is_dir():
            shutil.rmtree(child / "tex")

    # Every manifest row gets its own folder so each var_index maps to a
    # distinct, existing model.obj.  The first variation to use a PieceName
    # keeps the plain sanitized name; later ones (including identical prefabs
    # repeated across zones) are suffixed with __v<var_index>.
    used_folder = set()
    row_folder = {}
    for r in rows:
        base = sanitize(r["name"], f"var_{r['var_index']}")
        folder = base
        if folder in used_folder:
            folder = f"{base}__v{r['var_index']}"
            n = 1
            while folder in used_folder:
                folder = f"{base}__v{r['var_index']}_{n}"
                n += 1
        used_folder.add(folder)
        row_folder[r["var_index"]] = folder

    seen_prefabs = set()  # (bundle, pid) exported at least once
    totals = {"renderers": 0, "tris": 0, "missing_materials": 0, "missing_textures": 0}
    validation = {"straights": 0, "z_ok": 0, "y_ok": 0, "x_ok": 0, "all_ok": 0, "failures": []}
    entries = []
    skipped = []
    warnings = []

    for i, r in enumerate(rows):
        folder = row_folder[r["var_index"]]
        print(f"[{i + 1}/{len(rows)}] var {r['var_index']:3d} {r['name']} ({folder})", flush=True)
        try:
            bpath = bundle_path_for(r["bundle"])
            if not bpath.exists():
                skipped.append({"var_index": r["var_index"], "variation": r["name"], "reason": f"bundle missing: {r['bundle']}"})
                continue
            bundle = BUNDLE_SET.for_path(bpath)
            go_reader = find_gameobject(bundle, r["prefab_path_id"])
            if go_reader is None:
                skipped.append({"var_index": r["var_index"], "variation": r["name"], "reason": "prefab GameObject not found"})
                continue

            go = read_obj(go_reader)
            root_tr = None
            for c in go.m_Component or []:
                cp = getattr(c, "component", c)
                rd = pptr(go_reader.assets_file, cp)
                if rd is not None and rd.type.name == "Transform":
                    root_tr = rd
                    break
            if root_tr is None:
                skipped.append({"var_index": r["var_index"], "variation": r["name"], "reason": "prefab root has no Transform"})
                continue

            corridor = None
            if not args.no_clip and len(r["spline"]) >= 2:
                corridor = Corridor(r["spline"], args.lat, args.up, args.down, args.margin)

            ctx = {"variation": r["name"]}
            writer = VariationWriter(out_root / folder, TEX_STORE)
            export_prefab(writer, root_tr, ctx, corridor)
            result = writer.write(f"Temple Run 2 track piece {r['name']} (var {r['var_index']})")
            warnings.extend(writer.warnings)
            if not result and corridor is not None:
                # Corridor clipped everything away: keep the unclipped model
                # rather than shipping an empty piece.
                warnings.append(f"{r['name']}: corridor clip removed all geometry; exported unclipped")
                writer = VariationWriter(out_root / folder, TEX_STORE)
                export_prefab(writer, root_tr, ctx, None)
                result = writer.write(f"Temple Run 2 track piece {r['name']} (var {r['var_index']})")
                warnings.extend(writer.warnings)
            if not result:
                skipped.append(
                    {
                        "var_index": r["var_index"],
                        "variation": r["name"],
                        "reason": "no mesh geometry under prefab",
                        "renderers": writer.stats["renderers"],
                    }
                )
                continue
            lo, hi = result
            if _is_straight(r):
                _validate(validation, r, lo, hi)
            cached = {
                "bounds_min": lo,
                "bounds_max": hi,
                "materials": list(writer.material_order),
                "textures": list(writer.textures_used),
                "stats": writer.stats,
                "unresolved": list(writer.unresolved),
            }
            seen_prefabs.add((r["bundle"], r["prefab_path_id"]))
            for k, v in writer.stats.items():
                totals[k] = totals.get(k, 0) + v
            entries.append(_entry(r, folder, cached))
        except Exception:
            skipped.append(
                {
                    "var_index": r["var_index"],
                    "variation": r["name"],
                    "reason": "exception: " + traceback.format_exc(limit=2).strip().splitlines()[-1],
                }
            )

    warnings.extend(TEX_STORE.failed)
    # Materials whose PPtr pointed at a serialized file that is not present in
    # the bundle set.  These are *not* substituted: the OBJ/MTL carries the
    # name only, and the manifest records where the missing asset should live.
    unresolved_all = [u for e in entries for u in e.get("unresolved_materials", [])]
    missing_deps = sorted({u["dependency"] for u in unresolved_all if u.get("dependency")})
    unresolved_summary = {
        "count": len(unresolved_all),
        "distinct_cabs": sorted({u["cab"] for u in unresolved_all if u.get("cab")}),
        "entries": len({e["var_index"] for e in entries if e.get("unresolved_materials")}),
        "materials": unresolved_all,
    }
    manifest = {
        "schema": "tr2-track-models/1",
        "source": "recon/track/corpus.json",
        "yaw_offset_deg": YAW_OFFSET_DEG,
        "scale": SCALE,
        "axes": "unity (no handedness flip); uv v flipped to OBJ convention",
        "textures": sorted(TEX_STORE.by_name),
        "texture_count": len(TEX_STORE.by_name),
        "count": len(entries),
        "distinct_prefabs": len(seen_prefabs),
        "distinct_objs": len({e["obj"] for e in entries}),
        "skipped_count": len(skipped),
        "unresolved_materials": unresolved_summary,
        "missing_dependency_bundles": missing_deps,
        "validation": validation,
        "entries": entries,
        "skipped": skipped,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    print("\n=== export summary ===")
    print(f"variations processed : {len(rows)}")
    print(f"manifest entries     : {len(entries)}")
    print(f"distinct prefabs     : {len(seen_prefabs)}")
    print(f"distinct model.obj   : {len({e['obj'] for e in entries})}")
    print(f"skipped              : {len(skipped)}")
    tris = totals["tris"]
    rend = totals["renderers"]
    miss_m = totals["missing_materials"]
    miss_t = totals["missing_textures"]
    print(f"renderers            : {rend}")
    print(f"triangles            : {tris}")
    print(f"shared textures      : {len(TEX_STORE.by_name)}")
    print(f"unresolved materials : {miss_m}")
    if miss_m:
        print(f"  missing bundles    : {', '.join(missing_deps) or '(unknown)'}")
        print(f"  missing cabs       : {', '.join(unresolved_summary['distinct_cabs'])}")
    print(f"unresolved textures  : {miss_t}")
    print(f"warnings             : {len(warnings)}")
    print(
        "straight validation  : "
        f"{validation['all_ok']}/{validation['straights']} pass "
        f"(z {validation['z_ok']}, y {validation['y_ok']}, x {validation['x_ok']})"
    )
    if skipped:
        print("--- skipped ---")
        for s in skipped:
            print(f"  var {s['var_index']:3d} {s['variation']}: {s['reason']}")
    if warnings:
        print("--- first warnings ---")
        for w in warnings[:20]:
            print("  " + w)
    return 0


def _entry(row, folder, cached):
    return {
        "variation": row["name"],
        "var_index": row["var_index"],
        "safe_name": folder,
        "obj": f"track/{folder}/model.obj",
        "scale": SCALE,
        "yaw_offset_deg": YAW_OFFSET_DEG,
        "bounds_min": [round(x, 6) for x in cached["bounds_min"]],
        "bounds_max": [round(x, 6) for x in cached["bounds_max"]],
        "materials": cached["materials"],
        "textures": [f"track/tex/{t}" for t in cached.get("textures", [])],
        "unresolved_materials": cached.get("unresolved", []),
        "piece": row["piece"],
        "zone": row["zone"],
        "category": row["category"],
    }


if __name__ == "__main__":
    sys.exit(main())
