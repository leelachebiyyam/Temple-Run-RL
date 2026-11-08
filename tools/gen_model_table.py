#!/usr/bin/env python3
"""Generate ocean/templerun2/templerun2_models_generated.h.

Maps every TR2_VARIATION_TABLE var_index (0..N-1, N = corpus variation count)
to the shipped track model exported by tools/export_track_models.py.  The
model shipped to the runtime is the binary glTF `model.glb` (raylib 5.5 loads
GLB but crashes on these OBJ files); tools/obj_to_glb.py produces it beside
each `model.obj` and records it as the `glb` manifest field.

Inputs
------
* recon/track/corpus.json              - ordered variation list; its order is
                                         identical to TR2_VARIATION_TABLE, so
                                         the i-th variation is var_index i.
* resources/templerun2/models/track/manifest.json
                                       - per-variation export record
                                         (ASSET_CONTRACT.md), including the
                                         `glb` path written by
                                         tools/obj_to_glb.py. Optional: when it
                                         is absent/partial, exported models are
                                         still discovered by scanning
                                         resources/templerun2/models/track/ for
                                         unambiguous variation names
                                         (`model.glb`, falling back to
                                         `model.obj`).

Output
------
The header is fully self-contained (typedef + static table + static lookup).
It performs no I/O at runtime: entries with no exported model carry a NULL
obj_path and tr2_model_entry() returns NULL for them, which tells the caller to
use its procedural fallback.

Usage
-----
    python3 tools/gen_model_table.py [--check]

--check regenerates in memory and exits non-zero if the on-disk header differs.
"""

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "recon" / "track" / "corpus.json"
MANIFEST = ROOT / "resources" / "templerun2" / "models" / "track" / "manifest.json"
MODELS_ROOT = ROOT / "resources" / "templerun2" / "models"
TRACK_DIR = MODELS_ROOT / "track"
OUT = ROOT / "ocean" / "templerun2" / "templerun2_models_generated.h"

GUARD = "TEMPLERUN2_MODELS_GENERATED_H"
DEFAULT_SCALE = 1.0
DEFAULT_YAW = 180.0

_SANITIZE = re.compile(r"[^A-Za-z0-9_.-]")


def safe_name(name: str) -> str:
    """Sanitize a variation PieceName exactly as the exporter names folders."""
    return _SANITIZE.sub("_", name)


def cf(x: float) -> str:
    """Format a float as a C99 float literal."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        x = 0.0
    if x != x or x in (float("inf"), float("-inf")):  # NaN / inf -> 0
        x = 0.0
    s = f"{x:.6g}"
    if "e" in s or "E" in s or "." in s:
        return s + "f"
    return s + ".0f"


def cstr(s: str) -> str:
    """Format a Python string as a C string literal."""
    out = ['"']
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append("\\x%02x" % (ord(ch) & 0xFF))
    out.append('"')
    return "".join(out)


def load_variations():
    """Ordered variation rows; index == TR2_VARIATION_TABLE var_index."""
    data = json.loads(CORPUS.read_text())
    rows = []
    for piece in data["pieces"]:
        for var in piece["variations"]:
            rows.append({
                "var_index": len(rows),
                "name": var["name"],
                "piece_key": piece["key"],
                "prefab_path_id": var.get("prefab_path_id"),
            })
    return rows


def _manifest_entries(raw):
    """Accept several plausible manifest envelopes and return a list of dicts."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    if isinstance(raw, dict):
        for key in ("entries", "models", "variations", "pieces", "track", "items"):
            if isinstance(raw.get(key), list):
                return [e for e in raw[key] if isinstance(e, dict)]
        # dict keyed by var_index / variation name
        out = []
        for key, val in raw.items():
            if isinstance(val, dict):
                entry = dict(val)
                entry.setdefault("var_index", key)
                out.append(entry)
        if out:
            return out
        # a single entry
        if any(k in raw for k in ("obj", "obj_path", "variation", "name")):
            return [raw]
    return []


def _as_int(val):
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.lstrip("-").isdigit():
        return int(val)
    return None


def load_manifest():
    """Return lookup maps from the exporter manifest.

    Keyed by var_index (authoritative), then by prefab m_PathID, then by
    variation name (only usable for names that occur once in the corpus -
    30 variation names are reused by different prefabs, so those need
    var_index or prefab_path_id to disambiguate).
    """
    by_idx, by_name, by_pid = {}, {}, {}
    if not MANIFEST.exists():
        return by_idx, by_name, by_pid
    try:
        raw = json.loads(MANIFEST.read_text())
    except (OSError, ValueError) as exc:
        print(f"warning: cannot parse {MANIFEST}: {exc}", file=sys.stderr)
        return by_idx, by_name, by_pid

    for entry in _manifest_entries(raw):
        name = entry.get("variation", entry.get("name", entry.get("piece_name")))
        idx = _as_int(entry.get("var_index", entry.get("varIndex", entry.get("index"))))
        pid = _as_int(entry.get("prefab_path_id", entry.get("path_id",
                                                        entry.get("prefabPathId"))))
        if idx is not None:
            by_idx.setdefault(idx, entry)
        if isinstance(name, str) and name:
            by_name.setdefault(name, entry)
        if pid is not None:
            by_pid.setdefault(pid, entry)
    return by_idx, by_name, by_pid


def obj_bounds(path: Path):
    """Local-frame AABB from OBJ 'v' lines, or None."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    n = 0
    try:
        with path.open("r", errors="replace") as fh:
            for line in fh:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
                except ValueError:
                    continue
                for i, v in enumerate((x, y, z)):
                    lo[i] = min(lo[i], v)
                    hi[i] = max(hi[i], v)
                n += 1
    except OSError:
        return None
    if n == 0:
        return None
    return lo, hi


def glb_bounds(path: Path):
    """Local-frame AABB from a GLB's POSITION accessor min/max, or None."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 20 or data[:4] != b"glTF":
        return None
    clen, ctype = struct.unpack_from("<II", data, 12)
    if ctype != 0x4E4F534A:  # JSON chunk must come first
        return None
    try:
        doc = json.loads(data[20:20 + clen].decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    n = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            ai = prim.get("attributes", {}).get("POSITION")
            if ai is None:
                continue
            acc = doc.get("accessors", [])[ai] if ai < len(doc.get("accessors", [])) else {}
            mn, mx = acc.get("min"), acc.get("max")
            if not mn or not mx:
                continue
            for i in range(3):
                lo[i] = min(lo[i], float(mn[i]))
                hi[i] = max(hi[i], float(mx[i]))
            n += 1
    if n == 0:
        return None
    return lo, hi


def model_bounds(path: Path):
    """AABB for a shipped model path (.glb preferred, .obj fallback)."""
    if path.suffix.lower() == ".glb":
        return glb_bounds(path)
    return obj_bounds(path)


def _obj_field(entry):
    for key in ("glb", "glb_path", "model_glb",
                "obj", "obj_path", "model", "model_obj", "path"):
        val = entry.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _resolve_obj(raw_obj: str) -> str:
    """Normalize a manifest obj value to a repo-root-relative path."""
    p = raw_obj.replace("\\", "/").lstrip("/")
    if p.startswith("resources/templerun2/"):
        return p
    if p.startswith("models/"):
        return "resources/templerun2/" + p
    return "resources/templerun2/models/" + p


def _bounds(entry, obj_path):
    bmin = entry.get("bounds_min", entry.get("bmin"))
    bmax = entry.get("bounds_max", entry.get("bmax"))
    if isinstance(bmin, (list, tuple)) and isinstance(bmax, (list, tuple)) \
            and len(bmin) >= 3 and len(bmax) >= 3:
        try:
            lo = [float(bmin[i]) for i in range(3)]
            hi = [float(bmax[i]) for i in range(3)]
        except (TypeError, ValueError):
            lo = hi = None
        if lo is not None and all(hi[i] >= lo[i] for i in range(3)):
            return lo, hi
    if obj_path is not None:
        got = model_bounds(ROOT / obj_path)
        if got is not None:
            return got
    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]


def build_rows():
    """One row per var_index; obj_path None when nothing was exported."""
    variations = load_variations()
    by_idx, by_name, by_pid = load_manifest()

    name_counts = {}
    for v in variations:
        name_counts[v["name"]] = name_counts.get(v["name"], 0) + 1

    stats = {"total": len(variations), "exported": 0, "manifest": 0,
             "missing_files": 0, "scanned": 0, "by_index": 0, "by_name": 0,
             "by_pid": 0}
    rows = []
    for v in variations:
        i = v["var_index"]
        entry = by_idx.get(i)
        if entry is not None:
            stats["by_index"] += 1
        elif name_counts[v["name"]] == 1:
            entry = by_name.get(v["name"])
            if entry is not None:
                stats["by_name"] += 1
        if entry is None and v.get("prefab_path_id") is not None:
            entry = by_pid.get(v["prefab_path_id"])
            if entry is not None:
                stats["by_pid"] += 1

        obj_path = None
        if entry is not None:
            raw_obj = _obj_field(entry)
            if raw_obj:
                candidate = _resolve_obj(raw_obj)
                if (ROOT / candidate).is_file():
                    obj_path = candidate
                    stats["manifest"] += 1
                else:
                    # Manifest lists an export whose files are not on disk
                    # (broken/skipped export): treat the model as absent so
                    # the caller keeps its procedural fallback.
                    stats["missing_files"] += 1
                    print(f"warning: manifest model missing on disk, treating "
                          f"as absent: {candidate}", file=sys.stderr)

        if obj_path is None and name_counts[v["name"]] == 1:
            base = f"resources/templerun2/models/track/{safe_name(v['name'])}"
            for cand in ("model.glb", "model.obj"):
                guess = f"{base}/{cand}"
                if (ROOT / guess).is_file():
                    obj_path = guess
                    stats["scanned"] += 1
                    break

        scale = DEFAULT_SCALE
        yaw = DEFAULT_YAW
        if entry is not None:
            for key in ("scale", "model_scale"):
                if entry.get(key) is not None:
                    try:
                        scale = float(entry[key])
                    except (TypeError, ValueError):
                        pass
                    break
            for key in ("yaw_offset_deg", "yaw_offset", "yaw"):
                if entry.get(key) is not None:
                    try:
                        yaw = float(entry[key])
                    except (TypeError, ValueError):
                        pass
                    break

        lo, hi = _bounds(entry or {}, obj_path)
        if obj_path is not None:
            stats["exported"] += 1
        rows.append({
            "var_index": i,
            "name": v["name"],
            "piece_key": v["piece_key"],
            "obj_path": obj_path,
            "scale": scale,
            "yaw": yaw,
            "bmin": lo,
            "bmax": hi,
        })
    return rows, stats


def render(rows, stats):
    n = len(rows)
    header = []
    w = header.append
    w("/* Auto-generated by tools/gen_model_table.py - DO NOT EDIT.")
    w(" *")
    w(" * Maps TR2_VARIATION_TABLE var_index -> shipped track model")
    w(" * (resources/templerun2/models/track).  Generated from")
    w(" * recon/track/corpus.json plus the exporter manifest.")
    w(" *")
    w(" * obj_path is repo-root-relative and ready for LoadModel() (a .glb")
    w(" * path; LoadModel() dispatches on the extension).  It is NULL")
    w(" * for variations with no exported model, and tr2_model_entry() returns")
    w(" * NULL for those, so the caller keeps its procedural fallback.  This")
    w(" * header performs no file I/O at runtime.")
    w(" *")
    w(f" * variations: {stats['total']}  exported: {stats['exported']}"
      f"  (manifest {stats['manifest']}, scanned {stats['scanned']}, "
      f"manifest-listed-but-missing {stats['missing_files']})")
    w(" */")
    w(f"#ifndef {GUARD}")
    w(f"#define {GUARD}")
    w("")
    w("#include <stddef.h>")
    w("")
    w("#ifdef __cplusplus")
    w('extern "C" {')
    w("#endif")
    w("")
    w("typedef struct {")
    w("    int         var_index;      /* index into TR2_VARIATION_TABLE */")
    w("    const char *obj_path;       /* repo-relative model path (.glb), NULL if absent */")
    w("    float       scale;          /* model scale (manifest, default 1.0) */")
    w("    float       yaw_offset_deg; /* extra yaw in degrees (default 180.0) */")
    w("    float       bmin[3];        /* local-frame AABB min (x,y,z) */")
    w("    float       bmax[3];        /* local-frame AABB max (x,y,z) */")
    w("} Tr2ModelEntry;")
    w("")
    w(f"#define TR2_MODEL_COUNT {n}")
    w(f"#define TR2_MODEL_EXPORTED_COUNT {stats['exported']}")
    w("")
    w(f"static const Tr2ModelEntry TR2_MODEL_TABLE[TR2_MODEL_COUNT] = {{")
    for r in rows:
        obj = "NULL" if r["obj_path"] is None else cstr(r["obj_path"])
        name = r["name"].replace("*/", "* /")
        w(f"    {{ {r['var_index']:4d}, {obj:<72s}, "
          f"{cf(r['scale'])}, {cf(r['yaw'])}, "
          f"{{ {cf(r['bmin'][0])}, {cf(r['bmin'][1])}, {cf(r['bmin'][2])} }}, "
          f"{{ {cf(r['bmax'][0])}, {cf(r['bmax'][1])}, {cf(r['bmax'][2])} }} }},"
          f" /* {name} */")
    w("};")
    w("")
    w("/* NULL when var_index is out of range or has no exported model. */")
    w("static const Tr2ModelEntry *tr2_model_entry(int var_index) {")
    w("    if (var_index < 0 || var_index >= TR2_MODEL_COUNT) return NULL;")
    w("    if (TR2_MODEL_TABLE[var_index].obj_path == NULL) return NULL;")
    w("    return &TR2_MODEL_TABLE[var_index];")
    w("}")
    w("")
    w("#ifdef __cplusplus")
    w("}")
    w("#endif")
    w("")
    w(f"#endif /* {GUARD} */")
    w("")
    return "\n".join(header)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the on-disk header is stale")
    args = ap.parse_args()

    rows, stats = build_rows()
    text = render(rows, stats)

    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            print(f"{OUT} is stale; run tools/gen_model_table.py", file=sys.stderr)
            return 1
        print(f"{OUT} up to date ({stats['exported']}/{stats['total']} exported)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {stats['exported']}/{stats['total']} "
          f"variations exported (manifest {stats['manifest']}, "
          f"scanned {stats['scanned']}, manifest-listed-but-missing "
          f"{stats['missing_files']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
