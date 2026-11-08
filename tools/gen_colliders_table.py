#!/usr/bin/env python3
"""Generate ocean/templerun2/templerun2_colliders_data.h from shipped data.

Source of truth:
  * recon/collision/colliders.json   serialized BoxColliders + their
                                     TrackPieceCollisionInfo (Trigger/DeathType),
                                     size_world / center_rel_piece in the owning
                                     TrackPiece root frame.
  * recon/track/corpus.json          the shipped piece/variation table; a
                                     variation's `prefab_path_id` is the path id
                                     of its prefab root GameObject and its
                                     `bundle` is the zone bundle that ships it.

Mapping (variation -> its own colliders).  Two exact keys are used, never a
fuzzy/name-normalised guess:

  1. IDENTITY (preferred).  The collider is walked up its serialized Transform
     chain to its owning prefab root GameObject (colliders.json records the root
     GameObject *name* as `piece` but not its path id, so the bundles are
     re-scanned with UnityPy to recover root path id -> collider).  The collider
     belongs to the variation whose `prefab_path_id` equals that root path id and
     whose bundle basename matches.  This is the prefab's identity and is exact
     even when the scene instance name differs from the definition PieceName
     (e.g. `PirateCove_DOTD_Fort_Obstacle_Left_A` vs `PirateCove_Fort_LeftObstacle_A`).

  2. NAME+BUNDLE (fallback, no UnityPy needed).  `collider['piece']` equals the
     variation `name`, with the same bundle basename.

Rows that match neither key are reported, never guessed.

Volume fields (identical derivation to the hand-checked table this replaces):
    lane_center   = -center_rel_piece.x / TR2_LANE_WIDTH_M
    half_width    =  size_world.x / (2 * TR2_LANE_WIDTH_M)
    y_bottom/top  = center_rel_piece.y -/+ size_world.y/2
    forward_offset = -center_rel_piece.z      (spine runs toward -z)
    length        =  size_world.z
All rounded to 3 decimals.

The lateral sign is negated because track.h's tr2_tk_local_to_world maps an
authored piece-frame +X to the env *-lateral* axis (comment "local +X is the env
-lateral axis"); env lateral is +1 = right (runner fall direction: px<0 ->
WaterLeft).  The previous table used +x/1.7 and was mirrored.

kind is derived exactly as before (COLLISION.md sec.8.2):
    trigger 2 (Water)       -> WATER
    trigger 3 (MissedTurn)   -> LEDGE_LEFT / LEDGE_RIGHT for the Cavern
                                `*_Ledge_Left_A` / `*_Ledge_Right_A` edge boxes,
                                else GAP
    trigger 5 (Stumble)      -> STUMBLE
    trigger 6 (FallSaver)    -> FALL_SAVER   (not stamped as its own volume; its
                               geometry is folded into the GAP row of the same
                               prefab as the FallSaver rescue window, see below)
    trigger 10 (ZipLineLetGo)-> ZIPLINE      (not stamped: off-piece coords)
    trigger 4 (Obstacle):
        water-slide piece          -> WATER_BLOCK
        DeathType 2 (Fire)         -> FIRE
        DeathType 3 (MineDuck)     -> MINE_DUCK
        DeathType 5 (Mine)         -> MINE_WALL
        IsGenericObstacle          -> BLOCK
        y_bottom > 0.64            -> HIGH
        y_top < 2.0                -> LOW
        else                       -> BLOCK
Non-gameplay triggers (CameraTransition 1, Shadow 7, ZoneStart 8, ZoneStop 9,
AnimationStart 11, None 0) and MeshColliders (no size) are excluded.

FallSaver rescue window.  Every one of these prefabs also ships a
`Collision_FallSaver_A` trigger (TrackTrigger.FallSaver, 6) whose geometry the
runtime consumes in TRRunnerPawn.OnTriggerEnter (0x19658CC): a jumping /
just-landed runner that touches it is put back on the ground instead of dying.
It is not stamped as its own obstacle (it is not a hazard and must not reach the
observation), so its geometry is carried on the GAP row of the same prefab:

    save_offset    = fallsaver.forward_offset - gap.forward_offset
    save_length    = fallsaver.length
    save_half_width= fallsaver.half_width          (lane units)
    save_y_bottom  = fallsaver.y_bottom            (piece frame)
    save_y_top     = fallsaver.y_top

A GAP row with no FallSaver in its prefab carries zeros.  Only the FallSaver of
the same variation (same prefab identity / name+bundle key) is used, never one
from a sibling variation.

Run (UnityPy is only needed for the IDENTITY key; the venv already used by
recon/collision/extract_colliders.py):
    recon/unity_assets/.venv/bin/python tools/gen_colliders_table.py
Without UnityPy the generator still runs on the NAME+BUNDLE key alone:
    python3 tools/gen_colliders_table.py --no-identity
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "recon" / "track" / "corpus.json"
COLLIDERS = ROOT / "recon" / "collision" / "colliders.json"
UNITY_ASSETS = ROOT / "recon" / "unity_assets"
OUT = ROOT / "ocean" / "templerun2" / "templerun2_colliders_data.h"

LANE_WIDTH = 1.7

# TR2_VK_* kind ids (must match the header defines).
VK = {
    "LOW": 0, "HIGH": 1, "BLOCK": 2, "GAP": 3, "FIRE": 4, "STUMBLE": 5,
    "LEDGE_LEFT": 6, "LEDGE_RIGHT": 7, "MINE_DUCK": 8, "MINE_WALL": 9,
    "WATER": 10, "WATER_BLOCK": 11, "ZIPLINE": 12, "FALL_SAVER": 13,
}
KIND_NAME = {v: "TR2_VK_" + k for k, v in VK.items()}

# triggers stamped as gameplay volumes (everything else is excluded).
GAMEPLAY_TRIGGERS = {2, 3, 4, 5}
# stamped but currently a no-op in collision.h -> excluded to avoid dead rows.
NOOP_TRIGGERS = {6, 10}
NON_GAMEPLAY_TRIGGERS = {0, 1, 7, 8, 9, 11}

TRIGGER_NAMES = {
    0: "None", 1: "CameraTransition", 2: "Water", 3: "MissedTurn",
    4: "Obstacle", 5: "Stumble", 6: "FallSaver", 7: "Shadow",
    8: "ZoneStart", 9: "ZoneStop", 10: "ZipLineLetGo", 11: "AnimationStart",
}


def basename(bundle):
    return os.path.basename(bundle or "")


def cf(x):
    """C float literal, rounded to 3 decimals, no '-0.000'."""
    x = round(float(x) + 0.0, 3)
    if x == 0.0:
        x = 0.0
    return "%.3ff" % x


def trigger_death(entry):
    for node in entry.get("chain", []):
        tci = node.get("TrackPieceCollisionInfo")
        if tci:
            return tci.get("Trigger"), tci.get("DeathType")
    return None, None


def kind_for(entry, piece, trigger):
    """Reproduce the shipped kind mapping (see module docstring)."""
    size = entry["size_world"]
    ctr = entry["center_rel_piece"]
    y_bottom = round(ctr[1] - size[1] / 2.0, 3)
    y_top = round(ctr[1] + size[1] / 2.0, 3)
    flags = piece["flags"]
    name = entry.get("piece") or ""
    if trigger == 2:
        return VK["WATER"]
    if trigger == 3:
        if "Ledge_Left" in name:
            return VK["LEDGE_LEFT"]
        if "Ledge_Right" in name:
            return VK["LEDGE_RIGHT"]
        return VK["GAP"]
    if trigger == 5:
        return VK["STUMBLE"]
    if trigger == 6:
        return VK["FALL_SAVER"]
    if trigger == 10:
        return VK["ZIPLINE"]
    if trigger == 4:
        if flags.get("IsWaterSlide"):
            return VK["WATER_BLOCK"]
        death = trigger_death(entry)[1]
        if death == 2:
            return VK["FIRE"]
        if death == 3:
            return VK["MINE_DUCK"]
        if death == 5:
            return VK["MINE_WALL"]
        if flags.get("IsGenericObstacle"):
            return VK["BLOCK"]
        if y_bottom > 0.64:
            return VK["HIGH"]
        if y_top < 2.0:
            return VK["LOW"]
        return VK["BLOCK"]
    return None


def to_geom(entry):
    """Piece-frame box geometry of one collider (no trigger/kind)."""
    size = entry["size_world"]
    ctr = entry["center_rel_piece"]
    return {
        "lane_center": round(-ctr[0] / LANE_WIDTH, 3),
        "half_width": round(size[0] / (2.0 * LANE_WIDTH), 3),
        "y_bottom": round(ctr[1] - size[1] / 2.0, 3),
        "y_top": round(ctr[1] + size[1] / 2.0, 3),
        "forward_offset": round(-ctr[2], 3),
        "length": round(size[2], 3),
    }


def to_volume(entry, piece, trigger, death):
    size = entry["size_world"]
    ctr = entry["center_rel_piece"]
    vol = {
        "kind": kind_for(entry, piece, trigger),
        # authored piece-frame +X -> env -lateral (see docstring)
        "lane_center": round(-ctr[0] / LANE_WIDTH, 3),
        "half_width": round(size[0] / (2.0 * LANE_WIDTH), 3),
        "y_bottom": round(ctr[1] - size[1] / 2.0, 3),
        "y_top": round(ctr[1] + size[1] / 2.0, 3),
        "forward_offset": round(-ctr[2], 3),
        "length": round(size[2], 3),
        "trigger": trigger,
        "death_type": death,
    }
    # FallSaver rescue window, filled in by build() from the same prefab.
    vol["save_offset"] = 0.0
    vol["save_length"] = 0.0
    vol["save_half_width"] = 0.0
    vol["save_y_bottom"] = 0.0
    vol["save_y_top"] = 0.0
    return vol


# ------------------------------------------------------------------ bundles ---

def build_rootmap(entries, ua_root):
    """(bundle, collider path_id) -> owning prefab root GameObject path id."""
    try:
        import UnityPy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "UnityPy not importable (%s); run with "
            "recon/unity_assets/.venv/bin/python or pass --no-identity" % exc)

    by_bundle = {}
    for e in entries:
        by_bundle.setdefault(e["bundle"], []).append(e)

    rootmap = {}
    for bundle, rows in sorted(by_bundle.items()):
        path = os.path.join(str(ua_root), bundle)
        if not os.path.isfile(path):
            alt = path.replace("allbundles/", "bundles/")
            path = alt if os.path.isfile(alt) else path
        if not os.path.isfile(path):
            print("  warn: bundle missing on disk: %s" % bundle, file=sys.stderr)
            continue
        env = UnityPy.load(path)
        go, tr, comps = {}, {}, {}
        for o in env.objects:
            t = o.type.name
            try:
                if t == "GameObject":
                    go[o.path_id] = o.read().m_Name
                elif t in ("Transform", "RectTransform"):
                    tr[o.path_id] = o.read()
                elif t in ("BoxCollider", "SphereCollider", "CapsuleCollider",
                           "MeshCollider"):
                    comps[o.path_id] = o
            except Exception:
                pass
        go_of_tr = {}
        for tpid, tv in tr.items():
            go_of_tr[tpid] = getattr(getattr(tv, "m_GameObject", None),
                                     "path_id", None)
        tr_of_go = {}
        for tpid, gpid in go_of_tr.items():
            tr_of_go.setdefault(gpid, tpid)

        for e in rows:
            comp = comps.get(e["path_id"])
            if comp is None:
                rootmap[(bundle, e["path_id"])] = None
                continue
            try:
                gpid = getattr(getattr(comp.read(), "m_GameObject", None),
                               "path_id", None)
                trpid = tr_of_go.get(gpid)
                root = None
                depth = 0
                while trpid is not None and depth < 20:
                    tv = tr.get(trpid)
                    if tv is None:
                        break
                    root = go_of_tr.get(trpid)
                    trpid = getattr(getattr(tv, "m_Father", None), "path_id", None)
                    depth += 1
                rootmap[(bundle, e["path_id"])] = root
            except Exception:
                rootmap[(bundle, e["path_id"])] = None
    return rootmap


# --------------------------------------------------------------------- main ---

def build(corpus, colliders, rootmap):
    pieces = corpus["pieces"]
    variations = []  # global var index order == TR2_VARIATION_TABLE
    by_pid = {}
    by_name = {}
    for pi, piece in enumerate(pieces):
        for vi, var in enumerate(piece["variations"]):
            idx = len(variations)
            rec = {
                "var_idx": idx,
                "piece_idx": pi,
                "piece_key": piece["key"],
                "name": var["name"],
                "prefab_path_id": var.get("prefab_path_id"),
                "bundle": basename(var.get("bundle")),
            }
            variations.append(rec)
            if rec["prefab_path_id"] is not None:
                by_pid.setdefault((rec["bundle"], rec["prefab_path_id"]), rec)
            by_name.setdefault((rec["bundle"], rec["name"]), rec)

    vols = {}
    stats = {"identity": 0, "name": 0, "uncovered": [], "skipped_nosize": 0,
             "skipped_trigger": 0, "no_kind": 0, "fallsavers": 0}
    seen_rows = {}  # (var_idx, fields) dedup
    savers = {}     # var_idx -> [geometry of the prefab's FallSaver trigger(s)]

    for e in colliders:
        trigger, death = trigger_death(e)
        bundle = basename(e["bundle"])
        var = None
        how = None
        if rootmap is not None:
            root_pid = rootmap.get((e["bundle"], e["path_id"]))
            if root_pid is not None:
                var = by_pid.get((bundle, root_pid))
                how = "identity" if var else None
        if var is None:
            var = by_name.get((bundle, e.get("piece")))
            if var is not None:
                how = "name"
        if var is None:
            stats["uncovered"].append((e.get("piece"), bundle, trigger))
            continue
        if not e.get("size_world") or not e.get("center_rel_piece"):
            stats["skipped_nosize"] += 1
            continue
        if trigger == 6:
            # TrackTrigger.FallSaver: not a hazard, so no volume row -- its box
            # is the rescue window folded into the GAP row below.
            savers.setdefault(var["var_idx"], []).append(to_geom(e))
            stats["fallsavers"] += 1
            stats["skipped_trigger"] += 1
            continue
        if trigger in NON_GAMEPLAY_TRIGGERS or trigger is None:
            stats["skipped_trigger"] += 1
            continue
        if trigger in NOOP_TRIGGERS:
            stats["skipped_trigger"] += 1
            continue
        if trigger not in GAMEPLAY_TRIGGERS:
            stats["skipped_trigger"] += 1
            continue
        piece = pieces[var["piece_idx"]]
        vol = to_volume(e, piece, trigger, death)
        if vol["kind"] is None:
            stats["no_kind"] += 1
            continue
        key = (var["var_idx"], tuple(sorted(vol.items())))
        if key in seen_rows:
            continue
        seen_rows[key] = 1
        vols.setdefault(var["var_idx"], []).append((vol, e))
        stats[how] += 1

    # Fold each prefab's FallSaver into the GAP row it guards: the FallSaver is
    # authored at the pit's far edge, so pick the rescue window nearest that
    # edge.  Offsets are made relative to the GAP volume centre, the frame the
    # obstacle carries at runtime.
    for vidx, rows in vols.items():
        geom = savers.get(vidx)
        if not geom:
            continue
        for vol, _e in rows:
            if vol["kind"] != VK["GAP"]:
                continue
            far = vol["forward_offset"] + vol["length"] / 2.0
            best = min(geom, key=lambda g: abs(g["forward_offset"] - far))
            vol["save_offset"] = round(
                best["forward_offset"] - vol["forward_offset"], 3)
            vol["save_length"] = best["length"]
            vol["save_half_width"] = best["half_width"]
            vol["save_y_bottom"] = best["y_bottom"]
            vol["save_y_top"] = best["y_top"]

    return variations, vols, stats


def render(variations, vols, stats, used_identity):
    vsn = {v["var_idx"]: v for v in variations}
    header = ["""/* Canonical trigger / collision volumes for the Temple Run 2 rewrite.
 *
 * GENERATED by tools/gen_colliders_table.py from recon/collision/colliders.json
 * and recon/track/corpus.json.  Do not edit by hand.
 *
 * Per-variation rows: each variation of TR2_VARIATION_TABLE owns exactly the
 * gameplay trigger volumes of its own prefab, matched by %s
 * (see the generator docstring).  No union across the variations of a piece,
 * and no non-gameplay trigger is included.
 *
 * Frame / units, copied verbatim from COLLISION.md sec.8 and rounded to 3
 * decimals:
 *     lane_center    = -center_rel_piece.x / TR2_LANE_WIDTH_M   (env lane units;
 *                      authored +X is the env -lateral axis, +lane = right)
 *     half_width     =  size_world.x / (2 * TR2_LANE_WIDTH_M)
 *     y_bottom/y_top =  center_rel_piece.y -/+ size_world.y / 2
 *     forward_offset = -center_rel_piece.z
 *     length         =  size_world.z
 * kind/trigger/death reproduce the shipped TrackPieceCollisionInfo values and
 * the COLLISION.md sec.8.2 geometry rule (see the generator docstring).
 *
 * save_* on a GAP row carry the prefab's Collision_FallSaver_A trigger (the
 * TRRunnerPawn.OnTriggerEnter rescue window, 0x19658CC); see the generator
 * docstring.  Zero length = the prefab ships no FallSaver.
 *
 * Excluded: non-gameplay triggers (CameraTransition, Shadow, ZoneStart/ZoneStop,
 * ZipLineLetGo, AnimationStart, None), the FallSaver trigger as a volume of its
 * own (folded into its GAP row), ZipLine volumes, and MeshColliders (no box
 * size).
 */
#pragma once

#include <stddef.h>

/* TrackTrigger ids (COLLISION.md sec.2). */
#ifndef TR2_TRIG_NONE
#define TR2_TRIG_NONE 0
#define TR2_TRIG_CAMERATRANSITION 1
#define TR2_TRIG_WATER 2
#define TR2_TRIG_MISSEDTURN 3
#define TR2_TRIG_OBSTACLE 4
#define TR2_TRIG_STUMBLE 5
#define TR2_TRIG_FALLSAVER 6
#define TR2_TRIG_SHADOW 7
#define TR2_TRIG_ZONESTART 8
#define TR2_TRIG_ZONESTOP 9
#define TR2_TRIG_ZIPLINELETGO 10
#define TR2_TRIG_ANIMATIONSTART 11
#endif

/* Volume kinds. 0..7 match TR2_OBST_* in templerun2_types.h;
 * 8.. are source roles mapped explicitly to entity kinds during stamping. */
#ifndef TR2_VK_LOW
#define TR2_VK_LOW 0
#define TR2_VK_HIGH 1
#define TR2_VK_BLOCK 2
#define TR2_VK_GAP 3
#define TR2_VK_FIRE 4
#define TR2_VK_STUMBLE 5
#define TR2_VK_LEDGE_LEFT 6
#define TR2_VK_LEDGE_RIGHT 7
#define TR2_VK_MINE_DUCK 8
#define TR2_VK_MINE_WALL 9
#define TR2_VK_WATER 10
#define TR2_VK_WATER_BLOCK 11
#define TR2_VK_ZIPLINE 12
#define TR2_VK_FALL_SAVER 13
#define TR2_VK_COUNT 14
#endif

#define TR2_LANE_WIDTH_M 1.7f  /* metres per lane unit; default env->lane_width */

typedef struct {
    unsigned char kind;
    float lane_center;     /* lane units */
    float half_width;      /* lane units */
    float y_bottom, y_top; /* piece frame */
    float forward_offset;  /* metres from piece start */
    float length;
    unsigned char trigger;
    unsigned char death_type;
    /* FallSaver rescue window (TR2_VK_GAP rows; zeros when the prefab ships
     * none).  save_offset is the window centre relative to this volume centre
     * along the path, save_length its extent; save_half_width is lane units,
     * save_y_bottom/_top the piece-frame vertical span. */
    float save_offset;
    float save_length;
    float save_half_width;
    float save_y_bottom;
    float save_y_top;
} Tr2Volume;

/* ---- per-variation volume sets (keyed by TR2_VARIATION_TABLE index) ---- */
""" % ("the prefab GameObject path id (IDENTITY)" if used_identity
       else "the exact GameObject name + bundle basename (NAME+BUNDLE)")]

    for vidx in sorted(vols):
        v = vsn[vidx]
        rows = vols[vidx]
        header.append("static const Tr2Volume tr2_vv_%d[] = {  /* %s | piece %d %s */"
                      % (vidx, v["name"], v["piece_idx"], v["piece_key"]))
        for vol, e in rows:
            header.append("    { %s, %s, %s, %s, %s, %s, %s, %d, %d, %s, %s, %s, %s, %s },  /* %s */"
                          % (KIND_NAME[vol["kind"]], cf(vol["lane_center"]),
                             cf(vol["half_width"]), cf(vol["y_bottom"]),
                             cf(vol["y_top"]), cf(vol["forward_offset"]),
                             cf(vol["length"]), vol["trigger"],
                             vol["death_type"], cf(vol["save_offset"]),
                             cf(vol["save_length"]), cf(vol["save_half_width"]),
                             cf(vol["save_y_bottom"]), cf(vol["save_y_top"]),
                             e.get("piece") or ""))
        header.append("};")

    header.append("""
typedef struct {
    int var_idx;                 /* TR2_VARIATION_TABLE index */
    const Tr2Volume* vols;
    int count;
} Tr2VariationVolumes;

static const Tr2VariationVolumes TR2_VARIATION_VOLUMES[] = {""")
    for vidx in sorted(vols):
        header.append("    { %d, tr2_vv_%d, (int)(sizeof(tr2_vv_%d)/sizeof(tr2_vv_%d[0])) },"
                      % (vidx, vidx, vidx, vidx))
    header.append("""};
#define TR2_VARIATION_VOLUMES_COUNT \\
    (sizeof(TR2_VARIATION_VOLUMES) / sizeof(TR2_VARIATION_VOLUMES[0]))

/* Gameplay trigger volumes owned by the prefab of one variation (keyed by the
 * TR2_VARIATION_TABLE index carried in Tr2Segment.var_idx).  Returns the count
 * and sets *out to a static read-only array; sets *out = NULL and returns 0
 * when the variation has no gameplay volume (unknown index included). */
static int tr2_volumes_for_variation(int var_idx, const Tr2Volume** out) {
    int i;
    for (i = 0; i < (int)TR2_VARIATION_VOLUMES_COUNT; i++) {
        if (TR2_VARIATION_VOLUMES[i].var_idx == var_idx) {
            if (out) *out = TR2_VARIATION_VOLUMES[i].vols;
            return TR2_VARIATION_VOLUMES[i].count;
        }
    }
    if (out) *out = 0;
    return 0;
}""")

    # representative per kind, kept for the old tr2_volume_for_kind interface.
    first = {}
    for vidx in sorted(vols):
        for vol, e in vols[vidx]:
            first.setdefault(vol["kind"], vol)
    header.append("""
/* First volume of each kind (COLLISION.md sec.8.2 representative). */
static const Tr2Volume TR2_CANONICAL_VOLUMES[] = {""")
    for kind in sorted(first):
        vol = first[kind]
        header.append("    { %s, %s, %s, %s, %s, %s, %s, %d, %d, %s, %s, %s, %s, %s },"
                      % (KIND_NAME[kind], cf(vol["lane_center"]),
                         cf(vol["half_width"]), cf(vol["y_bottom"]),
                         cf(vol["y_top"]), cf(vol["forward_offset"]),
                         cf(vol["length"]), vol["trigger"], vol["death_type"],
                         cf(vol["save_offset"]), cf(vol["save_length"]),
                         cf(vol["save_half_width"]), cf(vol["save_y_bottom"]),
                         cf(vol["save_y_top"])))
    header.append("""};
#define TR2_CANONICAL_VOLUME_COUNT \\
    (sizeof(TR2_CANONICAL_VOLUMES) / sizeof(TR2_CANONICAL_VOLUMES[0]))

/* Representative volume of a given kind, or NULL when no volume of that kind
 * ships. */
static const Tr2Volume* tr2_volume_for_kind(int kind) {
    size_t i;
    for (i = 0; i < TR2_CANONICAL_VOLUME_COUNT; i++) {
        if ((int)TR2_CANONICAL_VOLUMES[i].kind == kind) {
            return &TR2_CANONICAL_VOLUMES[i];
        }
    }
    return 0;
}
""")
    return "\n".join(header) + "\n"


def report(stats, variations, vols):
    total_var = len(variations)
    with_vols = len(vols)
    rows = sum(len(v) for v in vols.values())
    print("variations            : %d" % total_var)
    print("variations with volumes: %d (%.1f%%)"
          % (with_vols, 100.0 * with_vols / max(1, total_var)))
    print("emitted volume rows    : %d" % rows)
    print("matched by IDENTITY    : %d" % stats["identity"])
    print("matched by NAME+BUNDLE : %d" % stats["name"])
    print("skipped (no box size)  : %d" % stats["skipped_nosize"])
    print("skipped (non-gameplay/no-op trigger): %d" % stats["skipped_trigger"])
    print("FallSaver windows folded into GAP rows: %d" % stats["fallsavers"])
    print("skipped (no kind)      : %d" % stats["no_kind"])
    uncovered = stats["uncovered"]
    print("uncovered colliders    : %d" % len(uncovered))
    by_trigger = {}
    for name, bundle, trig in uncovered:
        by_trigger.setdefault(trig, []).append((name, bundle))
    for trig in sorted(by_trigger, key=lambda t: (t is None, t)):
        rows_ = by_trigger[trig]
        print("  trigger %s (%s): %d"
              % (trig, TRIGGER_NAMES.get(trig, "?"), len(rows_)))
        for name, bundle in sorted(rows_)[:8]:
            print("      %s  [%s]" % (name, bundle))
        if len(rows_) > 8:
            print("      ... %d more" % (len(rows_) - 8))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-identity", action="store_true",
                    help="skip the UnityPy prefab-identity join (NAME+BUNDLE only)")
    ap.add_argument("--rootmap", type=Path,
                    help="read/write the (bundle,path_id)->root pid cache (JSON)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    corpus = json.loads(CORPUS.read_text())
    colliders = json.loads(COLLIDERS.read_text())

    rootmap = None
    used_identity = False
    if not args.no_identity:
        if args.rootmap and args.rootmap.exists():
            raw = json.loads(args.rootmap.read_text())
            rootmap = {(k.split("\t")[0], int(k.split("\t")[1])): v
                       for k, v in raw.items()}
        else:
            rootmap = build_rootmap(colliders["colliders"], UNITY_ASSETS)
            if args.rootmap:
                cache = {"%s\t%d" % (b, p): v
                         for (b, p), v in rootmap.items()}
                args.rootmap.write_text(json.dumps(cache))
        used_identity = True

    variations, vols, stats = build(corpus, colliders["colliders"], rootmap)
    args.out.write_text(render(variations, vols, stats, used_identity))
    print("wrote %s" % args.out)
    report(stats, variations, vols)
    return 0


if __name__ == "__main__":
    sys.exit(main())
