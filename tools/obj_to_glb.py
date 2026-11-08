#!/usr/bin/env python3
"""Convert the shipped Temple Run 2 OBJ/MTL models to self-contained GLB.

Raylib 5.5 in this tree crashes in LoadModel() on every OBJ but loads GLB
correctly, so every track/prop model is also shipped as a binary glTF
(`model.glb`) written next to its source `model.obj`.

The converter is pure stdlib (os/json/struct/...) and produces glTF 2.0 with:

* POSITION   (float3)  - one primitive per OBJ `usemtl` group
* TEXCOORD_0 (float2)  - OBJ `vt`, V flipped into glTF's top-left origin
* indices    (uint32)  - triangles, per material
* one material per MTL entry, `pbrMetallicRoughness.baseColorFactor` from `Kd`
  (alpha from `d`/`Tr`) and `baseColorTexture` backed by the `map_Kd` PNG,
  embedded as an image bufferView.  Materials whose texture/mtl marks
  transparency use `alphaMode: BLEND`.
* a material with no MTL entry, or an MTL `newmtl` block that declares no
  properties (what ``tools/export_track_models.py`` writes for a material whose
  asset was not in the corpus), is **not** filled in with a white default.
  glTF's default material is used, the material carries
  ``extras.unresolved = true``, and the GLB records the names under
  ``extras.unresolvedMaterials``.

Usage
-----
    python3 tools/obj_to_glb.py                 # convert track + props + manifests
    python3 tools/obj_to_glb.py --track-only
    python3 tools/obj_to_glb.py --props-only
    python3 tools/obj_to_glb.py --dir resources/templerun2/props/coin_gold
    python3 tools/obj_to_glb.py --no-manifest

`--dir` converts a single model directory without touching any manifest.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "resources" / "templerun2" / "models"
TRACK_DIR = MODELS_ROOT / "track"
TRACK_MANIFEST = TRACK_DIR / "manifest.json"
PROPS_DIR = ROOT / "resources" / "templerun2" / "props"
PROPS_MANIFEST = PROPS_DIR / "manifest.json"

# glTF constants
COMP_FLOAT = 5126
COMP_UINT32 = 5125
TARGET_ARRAY = 34962
TARGET_INDEX = 34963
MAG_LINEAR = 9729
MIN_LINEAR = 9729
WRAP_REPEAT = 10497
GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

_TRANSPARENT_HINT = ("transparent", "additive")
_JPEG_MAGIC = b"\xff\xd8\xff"


# --------------------------------------------------------------------------- #
# OBJ / MTL parsing
# --------------------------------------------------------------------------- #
def _f3(parts, i):
    return (float(parts[i]), float(parts[i + 1]), float(parts[i + 2]))


def parse_obj(path):
    """Return (positions, uvs, prims).

    positions/uvs are the raw OBJ arrays.  prims is an ordered list of
    (material_name, faces) where each face is a list of (v, vt, vn) 0-based
    indices (-1 for absent), so one primitive is emitted per material.
    """
    positions = []
    uvs = []
    prims = []
    by_name = {}
    cur = "default"

    with path.open("r", errors="replace") as fh:
        for line in fh:
            if not line or line[0] not in "vfu#g o":
                continue
            if line.startswith("v "):
                p = line.split()
                if len(p) >= 4:
                    positions.append(_f3(p, 1))
            elif line.startswith("vt "):
                p = line.split()
                if len(p) >= 3:
                    uvs.append((float(p[1]), float(p[2])))
            elif line.startswith("usemtl"):
                cur = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else "default"
                if cur not in by_name:
                    by_name[cur] = len(prims)
                    prims.append((cur, []))
            elif line.startswith("f "):
                if cur not in by_name:
                    by_name[cur] = len(prims)
                    prims.append((cur, []))
                faces = prims[by_name[cur]][1]
                toks = line.split()[1:]
                verts = []
                for t in toks:
                    bits = t.split("/")
                    v = int(bits[0])
                    if len(bits) == 1:
                        # `f v ...`: no attribute indices at all.  Some exports
                        # (all the props) write `vt` parallel to `v` and
                        # reference only the position index, so reuse it.
                        vt = v if 0 < v <= len(uvs) else -1
                        vn = -1
                    else:
                        # `f v/vt`, `f v//vn`, `f v/vt/vn`: an empty slot means
                        # that attribute is absent for this vertex; do NOT fall
                        # back to the position index.
                        vt = int(bits[1]) if len(bits) > 1 and bits[1] else -1
                        vn = int(bits[2]) if len(bits) > 2 and bits[2] else -1
                    verts.append((v - 1 if v > 0 else v, vt - 1 if vt > 0 else vt, vn))
                # triangulate (fan) - our corpus is triangles, but be safe
                for k in range(1, len(verts) - 1):
                    faces.append((verts[0], verts[k], verts[k + 1]))

    # material order: keep first-seen, drop empty groups
    prims = [(m, f) for (m, f) in prims if f]
    return positions, uvs, prims


def parse_mtl(path):
    """Return {name: {'Kd', 'd', 'map_Kd', 'resolved'}}.

    ``resolved`` is False for a ``newmtl`` block that carries no property lines
    at all - the marker ``tools/export_track_models.py`` writes for a material
    whose asset was not available.  Such a material must not be turned into a
    plausible-looking white PBR material downstream.
    """
    mats = {}
    cur = None
    if not path.is_file():
        return mats
    with path.open("r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            key = p[0]
            if key == "newmtl" and len(p) >= 2:
                cur = s.split(None, 1)[1].strip()
                mats[cur] = {"Kd": [1.0, 1.0, 1.0], "d": 1.0, "map_Kd": None,
                             "resolved": False}
            elif cur is None:
                continue
            elif key == "Kd" and len(p) >= 4:
                try:
                    mats[cur]["Kd"] = [float(p[1]), float(p[2]), float(p[3])]
                    mats[cur]["resolved"] = True
                except ValueError:
                    pass
            elif key == "d" and len(p) >= 2:
                try:
                    mats[cur]["d"] = float(p[1])
                    mats[cur]["resolved"] = True
                except ValueError:
                    pass
            elif key == "Tr" and len(p) >= 2:
                try:
                    mats[cur]["d"] = 1.0 - float(p[1])
                    mats[cur]["resolved"] = True
                except ValueError:
                    pass
            elif key == "map_Kd" and len(p) >= 2:
                mats[cur]["map_Kd"] = p[-1]
                mats[cur]["resolved"] = True
            elif key in ("Ka", "Ks", "Ns", "Ke", "illum", "Ni"):
                mats[cur]["resolved"] = True
    return mats


# --------------------------------------------------------------------------- #
# GLB assembly
# --------------------------------------------------------------------------- #
class _Bin:
    def __init__(self):
        self.data = bytearray()
        self.views = []

    def add_view(self, blob, target=None):
        while len(self.data) % 4:
            self.data.append(0)
        off = len(self.data)
        self.data += blob
        view = {"buffer": 0, "byteOffset": off, "byteLength": len(blob)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1


def _mime_for(raw, rel):
    if raw[:3] == _JPEG_MAGIC or rel.lower().endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return "image/png"


def build_glb(positions, uvs, prims, mats, obj_dir, asset_name):
    """Return the GLB bytes for one parsed model."""
    binf = _Bin()
    accessors = []
    materials = []
    images = []
    textures = []
    texture_by_rel = {}
    gl_prims = []

    def add_accessor(blob, comp_type, type_str, count, target, mn=None, mx=None):
        bv = binf.add_view(blob, target)
        acc = {"bufferView": bv, "componentType": comp_type,
               "count": count, "type": type_str}
        if mn is not None:
            acc["min"] = mn
            acc["max"] = mx
        accessors.append(acc)
        return len(accessors) - 1

    unresolved_mats = []
    for mat_name, faces in prims:
        spec = mats.get(mat_name)
        # A primitive whose material has no MTL entry, or whose entry declares
        # no properties, has no authored appearance.  Do not synthesise one.
        unresolved = spec is None or not spec.get("resolved", True)
        key_index = {}
        out_pos = []
        out_uv = []
        indices = []
        has_uv = all(f[0][1] >= 0 and f[1][1] >= 0 and f[2][1] >= 0 for f in faces)

        for tri in faces:
            for (v, vt, _vn) in tri:
                k = (v, vt)
                vi = key_index.get(k)
                if vi is None:
                    vi = len(out_pos)
                    key_index[k] = vi
                    px, py, pz = positions[v]
                    out_pos.append((px, py, pz))
                    if vt >= 0:
                        u, vv = uvs[vt]
                        out_uv.append((u, 1.0 - vv))  # OBJ -> glTF V flip
                    else:
                        out_uv.append((0.0, 0.0))
                indices.append(vi)

        if not out_pos:
            continue

        lo = [min(p[i] for p in out_pos) for i in range(3)]
        hi = [max(p[i] for p in out_pos) for i in range(3)]

        pos_blob = struct.pack("<%df" % (len(out_pos) * 3),
                               *[c for p in out_pos for c in p])
        pos_acc = add_accessor(pos_blob, COMP_FLOAT, "VEC3", len(out_pos),
                               TARGET_ARRAY, lo, hi)

        attributes = {"POSITION": pos_acc}
        if has_uv:
            uv_blob = struct.pack("<%df" % (len(out_uv) * 2),
                                  *[c for uv in out_uv for c in uv])
            attributes["TEXCOORD_0"] = add_accessor(
                uv_blob, COMP_FLOAT, "VEC2", len(out_uv), TARGET_ARRAY)

        idx_blob = struct.pack("<%dI" % len(indices), *indices)
        idx_acc = add_accessor(idx_blob, COMP_UINT32, "SCALAR", len(indices),
                               TARGET_INDEX)

        # material
        if unresolved:
            # Name only + an explicit marker.  glTF's default material applies;
            # no baseColorFactor/texture is asserted because none is known.
            material = {"name": mat_name, "doubleSided": True,
                        "extras": {"unresolved": True}}
            if mat_name not in unresolved_mats:
                unresolved_mats.append(mat_name)
            materials.append(material)
            gl_prims.append({"attributes": attributes, "indices": idx_acc,
                             "material": len(materials) - 1, "mode": 4})
            continue

        kd = spec["Kd"]
        alpha = max(0.0, min(1.0, spec["d"]))
        pbr = {
            "baseColorFactor": [float(kd[0]), float(kd[1]), float(kd[2]), float(alpha)],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.9,
        }
        transparent = alpha < 1.0
        rel = spec.get("map_Kd")
        if rel:
            tex_rel = (obj_dir / rel).resolve()
            tidx = texture_by_rel.get(str(tex_rel))
            if tidx is None:
                try:
                    raw = tex_rel.read_bytes()
                except OSError:
                    raw = None
                if raw:
                    mime = _mime_for(raw, rel)
                    iv = binf.add_view(raw)
                    images.append({"bufferView": iv, "mimeType": mime,
                                   "name": tex_rel.stem})
                    textures.append({"sampler": 0, "source": len(images) - 1})
                    tidx = len(textures) - 1
                    texture_by_rel[str(tex_rel)] = tidx
                    if any(h in tex_rel.stem.lower() for h in _TRANSPARENT_HINT):
                        transparent = True
            if tidx is not None:
                pbr["baseColorTexture"] = {"index": tidx, "texCoord": 0}
        material = {"name": mat_name, "doubleSided": True,
                    "pbrMetallicRoughness": pbr}
        if transparent:
            material["alphaMode"] = "BLEND"
        materials.append(material)
        gl_prims.append({"attributes": attributes, "indices": idx_acc,
                         "material": len(materials) - 1, "mode": 4})

    doc = {
        "asset": {"version": "2.0", "generator": "tools/obj_to_glb.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": asset_name}],
        "meshes": [{"name": asset_name, "primitives": gl_prims}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": binf.views,
        "buffers": [{"byteLength": len(binf.data)}],
    }
    if unresolved_mats:
        doc["extras"] = {"unresolvedMaterials": unresolved_mats}
    if images:
        doc["images"] = images
        doc["textures"] = textures
        doc["samplers"] = [{"magFilter": MAG_LINEAR, "minFilter": MIN_LINEAR,
                            "wrapS": WRAP_REPEAT, "wrapT": WRAP_REPEAT}]

    json_bytes = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4:
        json_bytes += b" "
    bin_bytes = bytes(binf.data)
    while len(bin_bytes) % 4:
        bin_bytes += b"\x00"

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(json_bytes), CHUNK_JSON)
    out += json_bytes
    out += struct.pack("<II", len(bin_bytes), CHUNK_BIN)
    out += bin_bytes
    return bytes(out), len(gl_prims), len(materials), len(images), len(unresolved_mats)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def convert_model_dir(model_dir, verbose=False):
    """Convert <model_dir>/model.obj -> model.glb.  Returns a stats dict."""
    obj_path = model_dir / "model.obj"
    if not obj_path.is_file():
        return None
    glb_path = model_dir / "model.glb"
    positions, uvs, prims = parse_obj(obj_path)
    mats = parse_mtl(model_dir / "model.mtl")
    blob, nprim, nmat, ntex, nunres = build_glb(positions, uvs, prims, mats,
                                                model_dir, model_dir.name)
    glb_path.write_bytes(blob)
    if verbose:
        print(f"  {model_dir.name}: {len(positions)}v {len(prims)} prims "
              f"{ntex} tex -> {len(blob)} bytes")
    return {"dir": model_dir, "verts": len(positions), "prims": nprim,
            "materials": nmat, "textures": ntex, "bytes": len(blob),
            "unresolved": nunres}


def _iter_model_dirs(root):
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "model.obj").is_file():
            yield child


def convert_tree(root, verbose=False):
    stats = []
    for d in _iter_model_dirs(root):
        s = convert_model_dir(d, verbose)
        if s:
            stats.append(s)
    return stats


def update_manifest(manifest_path, key, rel_prefix, stats):
    """Add/correct a `glb` field on every manifest entry that was converted."""
    doc = json.loads(manifest_path.read_text())
    entries = doc.get(key)
    if not isinstance(entries, list):
        print(f"warning: {manifest_path} has no '{key}' list", file=sys.stderr)
        return 0
    done = {s["dir"].name: s for s in stats}
    n = 0
    for entry in entries:
        # The `obj` field is authoritative: variation names are reused across
        # prefabs, so only the exported folder disambiguates the model.
        folder = None
        obj = entry.get("obj")
        if isinstance(obj, str) and obj:
            parts = obj.replace("\\", "/").rstrip("/").split("/")
            if len(parts) >= 2:
                folder = parts[-2]
        if folder not in done:
            folder = (entry.get("safe_name") or entry.get("name")
                      or entry.get("variation"))
        s = done.get(folder)
        if s is None:
            continue
        entry["glb"] = f"{rel_prefix}/{s['dir'].name}/model.glb"
        n += 1
    manifest_path.write_text(json.dumps(doc, indent=1) + "\n")
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=None,
                    help="convert a single model directory (no manifest update)")
    ap.add_argument("--track-only", action="store_true")
    ap.add_argument("--props-only", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.dir is not None:
        s = convert_model_dir(args.dir.resolve(), verbose=True)
        if s is None:
            print(f"error: no model.obj under {args.dir}", file=sys.stderr)
            return 1
        return 0

    do_track = not args.props_only
    do_props = not args.track_only

    if do_track:
        ss = convert_tree(TRACK_DIR, args.verbose)
        total = sum(s["bytes"] for s in ss)
        n_unres = sum(1 for s in ss if s["unresolved"])
        print(f"track: {len(ss)} models -> model.glb ({total} bytes)")
        if n_unres:
            print(f"  {n_unres} models carry unresolved materials "
                  f"(flagged in GLB extras and per-material extras)")
        if not args.no_manifest and ss:
            n = update_manifest(TRACK_MANIFEST, "entries", "track", ss)
            print(f"  manifest: {n} entries tagged with glb")
    if do_props:
        ss = convert_tree(PROPS_DIR, args.verbose)
        total = sum(s["bytes"] for s in ss)
        n_unres = sum(1 for s in ss if s["unresolved"])
        print(f"props: {len(ss)} models -> model.glb ({total} bytes)")
        if n_unres:
            print(f"  {n_unres} models carry unresolved materials")
        if not args.no_manifest and ss:
            n = update_manifest(PROPS_MANIFEST, "props", "props", ss)
            print(f"  manifest: {n} entries tagged with glb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
