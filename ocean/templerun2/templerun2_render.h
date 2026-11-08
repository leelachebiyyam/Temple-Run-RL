/* Raylib 3D view for the data-driven Temple Run 2 rewrite.
 *
 * Per visible segment it draws the shipped track piece model when the
 * generated manifest (templerun2_models_generated.h) and its OBJ exist,
 * otherwise the procedural floor/walls/pillars sampled from tr2_seg_point.
 * Obstacle volumes (low log, overhead, block, pit, flame, stumble, ledges),
 * coins, powerups, the runner and the HUD are drawn as before; the runner,
 * HUD and atmosphere defer to their own headers when present.
 * A third-person chase camera sits behind the runner.
 *
 * Human play maps arrow keys / WASD onto the same four swipe actions the
 * policy uses:
 *   up / W / space   jump
 *   down / S         slide
 *   left / A         swipe left
 *   right / D        swipe right
 *
 * Header-only, C99, all symbols static.  Only reads Env plus the frozen
 * track sampler (tr2_seg_point / tr2_world_at).
 */
#pragma once

/* Smooth the rendered heading across chord and piece boundaries without
 * changing simulation geometry. The world sampler normally starts at seg_idx,
 * so sample retained earlier pieces explicitly for the trailing point. */
static float tr2_r_path_yaw(Env* env, float s) {
    const float window = 1.5f;
    float ax, ay, az, ayaw, bx, by, bz, byaw;
    float back = s - window;
    int first = env->seg_idx;
    int oldest = env->seg_gen - TR2_SEG_RING + 1;
    if (oldest < 0) oldest = 0;
    while (first > oldest && back < env->segments[first % TR2_SEG_RING].s0) {
        Tr2Segment* prev = &env->segments[(first - 1) % TR2_SEG_RING];
        if (!prev->valid) break;
        --first;
    }
    if (first < env->seg_idx) {
        Tr2Segment* seg = &env->segments[first % TR2_SEG_RING];
        float d = back - seg->s0;
        if (d < 0.0f) d = 0.0f;
        tr2_seg_point(seg, d, 0.0f, &ax, &az, &ayaw);
    } else {
        tr2_world_at(env, back, 0.0f, 0.0f, &ax, &ay, &az, &ayaw);
    }
    tr2_world_at(env, s + window, 0.0f, 0.0f, &bx, &by, &bz, &byaw);
    float dx = bx - ax, dz = bz - az;
    return dx * dx + dz * dz < 1e-6f ? byaw : atan2f(dx, dz);
}

/* ------------------------------------------------------------------ *
 * Optional shipped-asset modules.
 *
 * Each owning header is pulled in only when it exists on disk; otherwise
 * a static (weak-style) fallback is defined in this file so the renderer
 * still compiles and falls back to the procedural drawing below.  All
 * symbols here are static; the file stays C99 and header-only.
 * ------------------------------------------------------------------ */
#if defined(__has_include)
#  if __has_include("templerun2_models_generated.h")
#    include "templerun2_models_generated.h"
#    define TR2_R_HAVE_MODELS 1
#  endif
#  if __has_include("templerun2_character.h")
#    include "templerun2_character.h"
#    define TR2_R_HAVE_CHARACTER 1
#  endif
#  if __has_include("templerun2_hud.h")
#    include "templerun2_hud.h"
#    define TR2_R_HAVE_HUD 1
#  endif
#  if __has_include("templerun2_atmosphere.h")
#    include "templerun2_atmosphere.h"
#    define TR2_R_HAVE_ATMOSPHERE 1
#  endif
#endif

/* Animation-state labels passed to tr2_character_draw.  When the character
 * header is present its own values are used; otherwise this fallback mirrors
 * them (RUN=0, JUMP=1, SLIDE=2, IDLE=3, FALL=4, DEAD=5, ZIPLINE=6). */
#ifndef TR2_CHAR_STATE_RUN
#define TR2_CHAR_STATE_RUN   0
#define TR2_CHAR_STATE_JUMP  1
#define TR2_CHAR_STATE_SLIDE 2
#define TR2_CHAR_STATE_IDLE  3
#define TR2_CHAR_STATE_FALL  4
#define TR2_CHAR_STATE_DEAD  5
#define TR2_CHAR_STATE_ZIPLINE 6
#endif

/* Game-derived backdrop/materials (exported from the APK by UnityPy). */
static Texture2D tr2_r_sky;
static Texture2D tr2_r_sky_layers;
static Texture2D tr2_r_stone;
static Texture2D tr2_r_wood;
static int tr2_r_assets_loaded = 0;

static void tr2_r_load_assets(void) {
    if (tr2_r_assets_loaded) return;
    tr2_r_assets_loaded = 1;
    tr2_r_sky = LoadTexture("resources/templerun2/backdrop.png");
    tr2_r_sky_layers = LoadTexture("resources/templerun2/sky.png");
    tr2_r_stone = LoadTexture("resources/templerun2/stone.png");
    tr2_r_wood = LoadTexture("resources/templerun2/wood.png");
}

#include "rlgl.h"

/* Lateral (right) and forward unit vectors for a yaw, matching the
 * convention baked into tr2_seg_point: forward = (sin y, cos y),
 * lateral = (cos y, -sin y). */
static void tr2_r_frame(float yaw, float* fx, float* fz, float* lx, float* lz) {
    *fx = sinf(yaw);
    *fz = cosf(yaw);
    *lx = cosf(yaw);
    *lz = -sinf(yaw);
}


static void tr2_r_quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, Color color) {
    DrawTriangle3D(a, b, c, color);
    DrawTriangle3D(a, c, d, color);
}

static void tr2_r_box(Vector3 center, float w, float h, float d, float yaw,
        Color color) {
    rlPushMatrix();
    rlTranslatef(center.x, center.y, center.z);
    rlRotatef(yaw * RAD2DEG, 0.0f, 1.0f, 0.0f);
    DrawCube((Vector3){0.0f, 0.0f, 0.0f}, w, h, d, color);
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, w, h, d, (Color){0, 0, 0, 70});
    rlPopMatrix();
}


/* ------------------------------------------------- shipped piece models --- */

/* Manifest table for shipped track pieces.  When the generator has not run
 * yet there is no header, so every variation falls back procedurally. */
#ifndef TR2_R_HAVE_MODELS
typedef struct {
    int var_index;
    const char* obj_path;
    float scale;
    float yaw_offset_deg;
    float bmin[3], bmax[3];
} Tr2ModelEntry;
static const Tr2ModelEntry* tr2_model_entry(int var_index) {
    (void)var_index;
    return NULL;
}
#endif

#define TR2_R_MODEL_SLOTS 512

#define TR2_R_MODEL_CACHE_MAX 64

/* state: 0 = untried/evicted, 1 = loaded, 2 = unavailable. */
typedef struct {
    unsigned char state;
    Model model;
    unsigned long long last_used;
} Tr2ModelSlot;

static Tr2ModelSlot tr2_r_models[TR2_R_MODEL_SLOTS];
static int tr2_r_models_loaded = 0;
static unsigned long long tr2_r_model_clock = 0;

/* raylib's UnloadModel intentionally leaves material textures to the caller.
 * A glTF may reuse a texture across material maps, so unload each ID once. */
static void tr2_r_unload_piece(Model model) {
    unsigned int default_id = rlGetTextureIdDefault();
    for (int i = 0; i < model.materialCount; i++) {
        MaterialMap* maps = model.materials[i].maps;
        if (maps == NULL) continue;
        for (int j = 0; j <= MATERIAL_MAP_BRDF; j++) {
            unsigned int id = maps[j].texture.id;
            if (id == 0 || id == default_id) continue;
            int seen = 0;
            for (int k = 0; k <= i && !seen; k++) {
                MaterialMap* prior = model.materials[k].maps;
                if (prior == NULL) continue;
                int end = k == i ? j : MATERIAL_MAP_BRDF + 1;
                for (int l = 0; l < end; l++) {
                    if (prior[l].texture.id == id) { seen = 1; break; }
                }
            }
            if (!seen) rlUnloadTexture(id);
        }
    }
    UnloadModel(model);
}

static void tr2_r_evict_piece(void) {
    int oldest = -1;
    for (int i = 0; i < TR2_R_MODEL_SLOTS; i++) {
        if (tr2_r_models[i].state == 1 &&
            (oldest < 0 || tr2_r_models[i].last_used < tr2_r_models[oldest].last_used))
            oldest = i;
    }
    if (oldest >= 0) {
        tr2_r_unload_piece(tr2_r_models[oldest].model);
        tr2_r_models[oldest].state = 0;
        tr2_r_models_loaded--;
    }
}

/* Load the shipped model for a variation. Keep a bounded cache: each exported
 * model has its own GPU texture, and the full corpus exceeds typical VRAM. */
static Model* tr2_r_piece_model(int var_idx) {
    if (var_idx < 0 || var_idx >= TR2_R_MODEL_SLOTS) return NULL;
    Tr2ModelSlot* slot = &tr2_r_models[var_idx];
    if (slot->state == 0) {
        const Tr2ModelEntry* e = tr2_model_entry(var_idx);
        if (e == NULL || e->obj_path == NULL || !FileExists(e->obj_path)) {
            slot->state = 2;
        } else {
            /* obj_path is already a repo-root-relative path. */
            if (tr2_r_models_loaded >= TR2_R_MODEL_CACHE_MAX) tr2_r_evict_piece();
            Model m = LoadModel(e->obj_path);
            if (IsModelValid(m)) {
                slot->model = m;
                slot->state = 1;
                tr2_r_models_loaded++;
            } else {
                slot->state = 2;
            }
        }
    }
    if (slot->state == 1) slot->last_used = ++tr2_r_model_clock;
    return slot->state == 1 ? &slot->model : NULL;
}

/* Decode common track assets before gameplay instead of blocking a live
 * physics/render interval whenever an episode picks a different variation. */
static void tr2_r_warm_track_models(void) {
    int count = 0;
    const Tr2PieceEntry* pool = tr2_piece_pool(0, &count);
    for (int i = 0; i < count; ++i) {
        if (pool[i].odds <= 0.0f) continue;
        const tr2_piece_t* piece = &TR2_PIECE_TABLE[pool[i].piece_idx];
        for (int j = 0; j < piece->var_count; ++j) {
            if (tr2_r_models_loaded >= TR2_R_MODEL_CACHE_MAX) return;
            tr2_r_piece_model(piece->var_begin + j);
        }
    }
}

/* Track meshes retain Unity X. Reflect it for the renderer's right-handed
 * world, and align the first SourcePath point with the segment origin. */
static void tr2_r_draw_piece(Model* m, const Tr2ModelEntry* e,
        const Tr2Segment* seg) {
    float yaw_deg = seg->heading * RAD2DEG + (e != NULL ? e->yaw_offset_deg : 180.0f);
    float scale = (e != NULL && e->scale != 0.0f) ? e->scale : 1.0f;
    Vector3 pos = {seg->wx, seg->wy, seg->wz};
    if (seg->spline_count >= 2 && seg->spline_begin < TR2_TK_SPLINE_MAX) {
        const tr2_vec3_t* first = &TR2_SPLINE_POINTS[seg->spline_begin];
        float angle = yaw_deg * DEG2RAD, c = cosf(angle), s = sinf(angle);
        pos.x -= scale * (-c * first->x + s * first->z);
        pos.y -= scale * first->y;
        pos.z -= scale * (s * first->x + c * first->z);
    }
    DrawModelEx(*m, pos, (Vector3){0.0f, 1.0f, 0.0f},
        yaw_deg, (Vector3){-scale, scale, scale}, WHITE);
}

/* --------------------------------------------------- shipped prop models --- */

/* One row per mesh-bearing entry of resources/templerun2/props/manifest.json
 * (written by tools/export_props.py).  obj_path is repo-root-relative and
 * ready for LoadModel(); scale/yaw_offset_deg are the "scale" /
 * "yaw_offset_deg" the manifest records, so the renderer applies exactly what
 * the exporter captured (the OBJ local frame is Unity axes with its forward
 * at -Z, hence the 180 deg default).  Rows whose only shipped artwork is a 2D
 * sprite (pickup_coin_bonus / _score_bonus / _gem_bonus) are omitted here, so
 * the caller keeps its procedural primitive for those.  The manifest repeats
 * the AABB as one scalar per prop (a generator bug), so bounds are not used. */
typedef struct {
    const char* name;           /* manifest "name" */
    const char* obj_path;       /* LoadModel()-ready OBJ path */
    float scale;                /* manifest "scale" */
    float yaw_offset_deg;       /* manifest "yaw_offset_deg" */
} Tr2PropEntry;

enum {
    TR2_PROP_OBSTACLE_JUMP_OVER = 0,
    TR2_PROP_OBSTACLE_JUMP_OVER_FOREST,
    TR2_PROP_OBSTACLE_LOG,
    TR2_PROP_OBSTACLE_LOG_SPIKES,
    TR2_PROP_OBSTACLE_SLIDE_UNDER,
    TR2_PROP_OBSTACLE_JUMP_OR_SLIDE,
    TR2_PROP_OBSTACLE_STUMBLE,
    TR2_PROP_OBSTACLE_STUMBLE_B,
    TR2_PROP_OBSTACLE_STUMBLE_FOREST,
    TR2_PROP_OBSTACLE_GAP_SMALL,
    TR2_PROP_OBSTACLE_GAP_LARGE,
    TR2_PROP_OBSTACLE_BRIDGE_GAP,
    TR2_PROP_OBSTACLE_LEDGE_LEFT,
    TR2_PROP_OBSTACLE_LEDGE_RIGHT,
    TR2_PROP_OBSTACLE_STATUE,
    TR2_PROP_OBSTACLE_STATUE_TREASURE,
    TR2_PROP_OBSTACLE_STATUE_SKULL,
    TR2_PROP_OBSTACLE_IDOL,
    TR2_PROP_OBSTACLE_FIRE_BRAZIER,
    TR2_PROP_OBSTACLE_FIRE_CANDLES,
    TR2_PROP_OBSTACLE_FIRE_BLUE,
    TR2_PROP_OBSTACLE_LANTERN,
    TR2_PROP_OBSTACLE_SAW,
    TR2_PROP_COIN_GOLD,
    TR2_PROP_COIN_RED,
    TR2_PROP_COIN_BLUE,
    TR2_PROP_COIN_MEGA,
    TR2_PROP_PICKUP_MAGNET,
    TR2_PROP_PICKUP_BOOST,
    TR2_PROP_PICKUP_SHIELD,
    TR2_PROP_PICKUP_GEM,
    TR2_PROP_PICKUP_COIN_SHADOW,
    TR2_PROP_COUNT
};

#define TR2_PROP_ROOT "resources/templerun2/"

static const Tr2PropEntry TR2_PROP_TABLE[TR2_PROP_COUNT] = {
    { "obstacle_jump_over",
      TR2_PROP_ROOT "props/obstacle_jump_over/model.glb",          1.0f, 180.0f },
    { "obstacle_jump_over_forest",
      TR2_PROP_ROOT "props/obstacle_jump_over_forest/model.glb",   1.0f, 180.0f },
    { "obstacle_log",
      TR2_PROP_ROOT "props/obstacle_log/model.glb",                1.0f, 180.0f },
    { "obstacle_log_spikes",
      TR2_PROP_ROOT "props/obstacle_log_spikes/model.glb",         1.0f, 180.0f },
    { "obstacle_slide_under",
      TR2_PROP_ROOT "props/obstacle_slide_under/model.glb",        1.0f, 180.0f },
    { "obstacle_jump_or_slide",
      TR2_PROP_ROOT "props/obstacle_jump_or_slide/model.glb",      1.0f, 180.0f },
    { "obstacle_stumble",
      TR2_PROP_ROOT "props/obstacle_stumble/model.glb",            1.0f, 180.0f },
    { "obstacle_stumble_b",
      TR2_PROP_ROOT "props/obstacle_stumble_b/model.glb",          1.0f, 180.0f },
    { "obstacle_stumble_forest",
      TR2_PROP_ROOT "props/obstacle_stumble_forest/model.glb",     1.0f, 180.0f },
    { "obstacle_gap_small",
      TR2_PROP_ROOT "props/obstacle_gap_small/model.glb",          1.0f, 180.0f },
    { "obstacle_gap_large",
      TR2_PROP_ROOT "props/obstacle_gap_large/model.glb",          1.0f, 180.0f },
    { "obstacle_bridge_gap",
      TR2_PROP_ROOT "props/obstacle_bridge_gap/model.glb",         1.0f, 180.0f },
    { "obstacle_ledge_left",
      TR2_PROP_ROOT "props/obstacle_ledge_left/model.glb",         1.0f, 180.0f },
    { "obstacle_ledge_right",
      TR2_PROP_ROOT "props/obstacle_ledge_right/model.glb",        1.0f, 180.0f },
    { "obstacle_statue",
      TR2_PROP_ROOT "props/obstacle_statue/model.glb",             1.0f, 180.0f },
    { "obstacle_statue_treasure",
      TR2_PROP_ROOT "props/obstacle_statue_treasure/model.glb",    1.0f, 180.0f },
    { "obstacle_statue_skull",
      TR2_PROP_ROOT "props/obstacle_statue_skull/model.glb",       1.0f, 180.0f },
    { "obstacle_idol",
      TR2_PROP_ROOT "props/obstacle_idol/model.glb",               1.0f, 180.0f },
    { "obstacle_fire_brazier",
      TR2_PROP_ROOT "props/obstacle_fire_brazier/model.glb",       1.0f, 180.0f },
    { "obstacle_fire_candles",
      TR2_PROP_ROOT "props/obstacle_fire_candles/model.glb",       1.0f, 180.0f },
    { "obstacle_fire_blue",
      TR2_PROP_ROOT "props/obstacle_fire_blue/model.glb",          1.0f, 180.0f },
    { "obstacle_lantern",
      TR2_PROP_ROOT "props/obstacle_lantern/model.glb",            1.0f, 180.0f },
    { "obstacle_saw",
      TR2_PROP_ROOT "props/obstacle_saw/model.glb",                1.0f, 180.0f },
    { "coin_gold",
      TR2_PROP_ROOT "props/coin_gold/model.glb",                   1.0f, 180.0f },
    { "coin_red",
      TR2_PROP_ROOT "props/coin_red/model.glb",                    1.0f, 180.0f },
    { "coin_blue",
      TR2_PROP_ROOT "props/coin_blue/model.glb",                   1.0f, 180.0f },
    { "coin_mega",
      TR2_PROP_ROOT "props/coin_mega/model.glb",                   1.0f, 180.0f },
    { "pickup_magnet",
      TR2_PROP_ROOT "props/pickup_magnet/model.glb",               1.0f, 180.0f },
    { "pickup_boost",
      TR2_PROP_ROOT "props/pickup_boost/model.glb",                1.0f, 180.0f },
    { "pickup_shield",
      TR2_PROP_ROOT "props/pickup_shield/model.glb",               1.0f, 180.0f },
    { "pickup_gem",
      TR2_PROP_ROOT "props/pickup_gem/model.glb",                  1.0f, 180.0f },
    { "pickup_coin_shadow",
      TR2_PROP_ROOT "props/pickup_coin_shadow/model.glb",          1.0f, 180.0f }
};

/* One load attempt per manifest entry, cached for the process lifetime.
 * state: 0 = untried, 1 = loaded, 2 = unavailable (caller keeps its
 * procedural primitive). */
typedef struct {
    unsigned char state;
    Model model;
} Tr2PropSlot;

static Tr2PropSlot tr2_r_props[TR2_PROP_COUNT];

/* Load (once) the shipped OBJ registered for a manifest entry.  Returns NULL
 * when the file is missing/unreadable, so the caller draws its fallback. */
static Model* tr2_r_prop_model(int prop_idx) {
    if (prop_idx < 0 || prop_idx >= TR2_PROP_COUNT) return NULL;
    Tr2PropSlot* slot = &tr2_r_props[prop_idx];
    if (slot->state == 0) {
        if (!FileExists(TR2_PROP_TABLE[prop_idx].obj_path)) {
            slot->state = 2;
            return NULL;
        }
        Model m = LoadModel(TR2_PROP_TABLE[prop_idx].obj_path);
        if (IsModelValid(m)) {
            slot->model = m;
            slot->state = 1;
        } else {
            TraceLog(LOG_WARNING, "TR2 prop model missing: %s",
                TR2_PROP_TABLE[prop_idx].obj_path);
            slot->state = 2;
        }
    }
    return slot->state == 1 ? &slot->model : NULL;
}

/* Props already have Unity X reflected by the exporter. */
static void tr2_r_draw_prop(Model* m, const Tr2PropEntry* e, Vector3 pos,
        float yaw_rad, Color tint) {
    float yaw_deg = yaw_rad * RAD2DEG + (e != NULL ? e->yaw_offset_deg : 180.0f);
    float scale = (e != NULL && e->scale != 0.0f) ? e->scale : 1.0f;
    DrawModelEx(*m, pos, (Vector3){0.0f, 1.0f, 0.0f}, yaw_deg,
        (Vector3){scale, scale, scale}, tint);
}

/* The simplified 8-way obstacle model cannot tell the zone-alternate prefabs
 * apart (log, saw, forest stumble, treasure/idol statues, candles, ...); each
 * kind maps to its canonical shipped prefab, or -1 to keep the primitive. */
static int tr2_r_obstacle_prop(int kind) {
    switch (kind) {
        case TR2_OBST_LOW:         return TR2_PROP_OBSTACLE_JUMP_OVER;
        case TR2_OBST_HIGH:        return TR2_PROP_OBSTACLE_SLIDE_UNDER;
        case TR2_OBST_BLOCK:       return TR2_PROP_OBSTACLE_STATUE;
        case TR2_OBST_GAP:         return TR2_PROP_OBSTACLE_GAP_SMALL;
        case TR2_OBST_FIRE:        return TR2_PROP_OBSTACLE_FIRE_BRAZIER;
        case TR2_OBST_STUMBLE:     return TR2_PROP_OBSTACLE_STUMBLE;
        case TR2_OBST_LEDGE_LEFT:  return TR2_PROP_OBSTACLE_LEDGE_LEFT;
        case TR2_OBST_LEDGE_RIGHT: return TR2_PROP_OBSTACLE_LEDGE_RIGHT;
        default:                   return -1;
    }
}

/* Coin value 1/2/3 -> gold/red/blue shipped coin. */
static int tr2_r_coin_prop(int value) {
    if (value >= 3) return TR2_PROP_COIN_BLUE;
    if (value == 2) return TR2_PROP_COIN_RED;
    return TR2_PROP_COIN_GOLD;
}

/* Powerup kind -> shipped pickup, or -1 for the sprite-only bonus kinds. */
static int tr2_r_powerup_prop(int kind) {
    switch (kind) {
        case TR2_PU_MAGNET: return TR2_PROP_PICKUP_MAGNET;
        case TR2_PU_BOOST:  return TR2_PROP_PICKUP_BOOST;
        case TR2_PU_SHIELD: return TR2_PROP_PICKUP_SHIELD;
        case TR2_PU_GEM:    return TR2_PROP_PICKUP_GEM;
        default:            return -1;
    }
}

static void tr2_r_draw_track(Env* env, int first, int last) {
    float hw = env->track_half_width;
    float wall_h = 1.6f;
    Color stone = (Color){84, 72, 54, 255};
    Color stone_alt = (Color){92, 79, 58, 255};
    Color wall = (Color){96, 80, 58, 255};
    Color wall_top = (Color){126, 108, 78, 255};
    for (int i = first; i <= last; i++) {
        Tr2Segment* seg = &env->segments[i % TR2_SEG_RING];
        if (!seg->valid || seg->length <= 0.0f) continue;
        /* Shipped piece model, when the manifest/file exists: it stands in
         * for the whole procedural floor/wall/pillar block of this segment,
         * so it is drawn instead of, never on top of, the geometry below. */
        Model* shipped = tr2_r_piece_model(seg->var_idx);
        if (shipped != NULL) {
            tr2_r_draw_piece(shipped, tr2_model_entry(seg->var_idx), seg);
            continue;
        }
        int bridge = seg->mode == TR2_MODE_WATER
                  || seg->mode == TR2_MODE_ZIPLINE;
        Color floor = (seg->var_idx & 1) ? stone_alt : stone;
        int steps = (int)ceilf(seg->length / 3.0f);
        if (steps < 1) steps = 1;
        float plx = 0.0f, plz = 0.0f, prx = 0.0f, prz = 0.0f, py = 0.0f;
        for (int k = 0; k <= steps; k++) {
            float d = seg->length * (float)k / (float)steps;
            float h, lx, lz, rx, rz, y = tr2_seg_height(seg, d);
            tr2_seg_point(seg, d, -hw, &lx, &lz, &h);
            tr2_seg_point(seg, d, hw, &rx, &rz, &h);
            if (k > 0) {
                tr2_r_quad((Vector3){plx, py, plz}, (Vector3){prx, py, prz},
                    (Vector3){rx, y, rz}, (Vector3){lx, y, lz}, floor);

                /* Lane dividers: grooves at the two inter-lane boundaries. */
                for (int lane = 1; lane < TR2_LANES; lane++) {
                    float off = ((float)lane - 1.5f) * env->lane_width;
                    float ax, az, bx, bz, gh;
                    tr2_seg_point(seg, seg->length * (float)(k - 1) / (float)steps,
                        off, &ax, &az, &gh);
                    tr2_seg_point(seg, d, off, &bx, &bz, &gh);
                    DrawLine3D((Vector3){ax, py + 0.02f, az}, (Vector3){bx, y + 0.02f, bz},
                        (Color){150, 130, 94, 255});
                }

                if (!bridge) {
                    tr2_r_quad((Vector3){plx, py, plz}, (Vector3){lx, y, lz},
                        (Vector3){lx, y + wall_h, lz}, (Vector3){plx, py + wall_h, plz}, wall);
                    tr2_r_quad((Vector3){prx, py, prz}, (Vector3){rx, y, rz},
                        (Vector3){rx, y + wall_h, rz}, (Vector3){prx, py + wall_h, prz}, wall);
                    tr2_r_quad((Vector3){plx, py + wall_h, plz}, (Vector3){prx, py + wall_h, prz},
                        (Vector3){rx, y + wall_h, rz}, (Vector3){lx, y + wall_h, lz}, wall_top);
                }
            }
            plx = lx; plz = lz; prx = rx; prz = rz; py = y;
        }

        /* Roadside pillars. */
        int pillars = (int)(seg->length / 12.0f);
        for (int p = 0; p <= pillars; p++) {
            float d = seg->length * (float)p / (float)(pillars + 1);
            float lx, lz, rx, rz, h, y = tr2_seg_height(seg, d);
            tr2_seg_point(seg, d, -(hw + 0.35f), &lx, &lz, &h);
            tr2_seg_point(seg, d, hw + 0.35f, &rx, &rz, &h);
            tr2_r_box((Vector3){lx, y + 1.8f, lz}, 0.7f, 3.6f, 0.7f, h,
                (Color){120, 100, 72, 255});
            tr2_r_box((Vector3){rx, y + 1.8f, rz}, 0.7f, 3.6f, 0.7f, h,
                (Color){120, 100, 72, 255});
        }
    }
}

static void tr2_r_draw_low(Env* env, Tr2Obstacle* o, float wx, float wy,
        float wz, float yaw) {
    float lx = cosf(yaw), lz = -sinf(yaw);
    float half = o->half_width * env->lane_width;
    if (half <= 0.0f) half = 0.5f * env->lane_width;
    float yb = o->y_bottom;
    float yt = o->y_top > yb ? o->y_top : yb + 0.9f;
    float radius = 0.5f * (yt - yb);
    float cy = yb + radius;
    Vector3 a = {wx - lx * half, cy, wz - lz * half};
    Vector3 b = {wx + lx * half, cy, wz + lz * half};
    int stumbling = o->kind == TR2_OBST_STUMBLE;
    Color bark = stumbling ? (Color){150, 108, 52, 255} : (Color){124, 82, 40, 255};
    DrawCylinderEx(a, b, radius, radius, 12, bark);
    DrawCylinderWiresEx(a, b, radius, radius, 12, (Color){62, 38, 16, 255});
    DrawCylinderEx(a, b, radius * 1.04f, radius * 1.04f, 12,
        (Color){176, 132, 70, 80});
    if (stumbling) {
        for (int band = -1; band <= 1; band++) {
            Vector3 c = {wx + lx * (float)band * 0.35f * env->lane_width, cy,
                         wz + lz * (float)band * 0.35f * env->lane_width};
            DrawCylinderEx(c, (Vector3){c.x, cy, c.z}, radius * 1.02f,
                radius * 1.02f, 12, (Color){250, 214, 70, 255});
        }
    }
}

static void tr2_r_draw_high(Env* env, Tr2Obstacle* o, float wx, float wy,
        float wz, float yaw) {
    float lx = cosf(yaw), lz = -sinf(yaw);
    float half = o->half_width * env->lane_width;
    if (half <= 0.0f) half = 0.55f * env->lane_width;
    float yb = o->y_bottom;
    float yt = o->y_top > yb ? o->y_top : yb + 1.3f;
    float cy = 0.5f * (yb + yt);
    float thick = yt - yb;
    tr2_r_box((Vector3){wx, cy, wz}, 2.0f * half, thick,
        o->length > 0.0f ? o->length : 0.5f, yaw, (Color){54, 112, 46, 255});
    tr2_r_box((Vector3){wx, yt + 0.11f, wz}, 2.0f * half + 0.1f, 0.22f,
        (o->length > 0.0f ? o->length : 0.5f) + 0.12f, yaw,
        (Color){30, 78, 28, 255});
    for (int leaf = -1; leaf <= 1; leaf++) {
        Vector3 p = {wx + lx * (float)leaf * 0.4f * env->lane_width,
                     yt + 0.2f, wz + lz * (float)leaf * 0.4f * env->lane_width};
        DrawSphere(p, 0.24f, (Color){86, 158, 62, 255});
    }
}

static void tr2_r_draw_gap(Env* env, Tr2Obstacle* o, float wx, float wy,
        float wz, float yaw) {
    float len = o->length > 0.0f ? o->length : 2.0f;
    float yb = o->y_bottom;
    float yt = o->y_top;
    if (yt <= yb) { yb = -2.0f; yt = 0.0f; }
    float cy = 0.5f * (yb + yt);
    tr2_r_box((Vector3){wx, cy, wz}, 2.02f * env->track_half_width, yt - yb, len,
        yaw, (Color){6, 5, 4, 255});
    float fx = sinf(yaw), fz = cosf(yaw);
    for (int side = -1; side <= 1; side += 2) {
        Vector3 e = {wx + fx * len * 0.5f * (float)side, yt + 0.05f,
                     wz + fz * len * 0.5f * (float)side};
        tr2_r_box(e, 2.05f * env->track_half_width, 0.12f, 0.18f, yaw,
            (Color){206, 106, 40, 255});
    }
}

static void tr2_r_draw_fire(Env* env, Tr2Obstacle* o, float wx, float wy,
        float wz, float yaw) {
    float yb = o->y_bottom;
    float yt = o->y_top > yb ? o->y_top : yb + 1.8f;
    float h = yt;
    float lat_half = o->half_width * env->lane_width;
    if (lat_half <= 0.0f) lat_half = 0.5f * env->lane_width;
    Vector3 a = {wx, yb + 0.05f, wz};
    Vector3 b = {wx, h, wz};
    DrawCylinderEx(a, b, 0.42f, 0.14f, 10, (Color){246, 120, 34, 235});
    DrawSphere((Vector3){wx, h + 0.08f, wz}, 0.25f, (Color){255, 214, 80, 230});
    DrawSphere((Vector3){wx, yb + (h - yb) * 0.55f, wz}, 0.3f,
        (Color){255, 160, 44, 120});
    tr2_r_box((Vector3){wx, 0.25f, wz}, 1.6f * lat_half, 0.5f, 0.5f, yaw,
        (Color){86, 70, 58, 255});
}

static void tr2_r_draw_block(Env* env, Tr2Obstacle* o, float wx, float wy,
        float wz, float yaw) {
    float yb = o->y_bottom;
    float yt = o->y_top > yb ? o->y_top : yb + 3.0f;
    float half = o->half_width * env->lane_width;
    if (half <= 0.0f) half = 0.5f * env->lane_width;
    float len = o->length > 0.0f ? o->length : 1.2f;
    tr2_r_box((Vector3){wx, 0.5f * (yb + yt), wz}, 2.0f * half, yt - yb, len,
        yaw, (Color){132, 128, 120, 255});
    tr2_r_box((Vector3){wx, yb + 0.30f, wz}, 2.0f * half + 0.08f, 0.22f,
        len + 0.05f, yaw, (Color){190, 70, 52, 255});
    tr2_r_box((Vector3){wx, yb + 0.05f, wz}, 2.0f * half + 0.14f, 0.35f,
        len + 0.1f, yaw, (Color){92, 86, 78, 255});
}

/* Ledges (TR2_OBST_LEDGE_LEFT / _RIGHT): a safe rim on the inner side and a
 * dark drop beyond it on the indicated side. */
static void tr2_r_draw_ledge(Env* env, Tr2Obstacle* o, float wx, float wy,
        float wz, float yaw) {
    float lx = cosf(yaw), lz = -sinf(yaw);
    int side = o->kind == TR2_OBST_LEDGE_LEFT ? -1 : 1;
    float half = o->half_width * env->lane_width;
    if (half <= 0.0f) half = 0.75f * env->lane_width;
    float len = o->length > 0.0f ? o->length : 2.0f;
    /* Drop slab from the rim outward. */
    float ox = lx * (float)side * half;
    float oz = lz * (float)side * half;
    tr2_r_box((Vector3){wx + ox * 2.2f, -0.6f, wz + oz * 2.2f},
        half * 3.0f, 1.0f, len * 0.98f, yaw, (Color){8, 7, 6, 255});
    /* Warning rim exactly on the ledge boundary. */
    tr2_r_box((Vector3){wx + ox, 0.10f, wz + oz}, 0.30f, 0.2f, len, yaw,
        (Color){220, 170, 44, 255});
}

static void tr2_r_draw_obstacles(Env* env, int first, int last) {
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) {
        Tr2Obstacle* o = &env->obstacles[i];
        if (!o->active) continue;
        float rel = o->s - env->s;
        if (rel < -0.5f || rel > 110.0f) continue;
        /* These verified Machu prefabs bake the obstacle into the track mesh.
         * Do not overlay unrelated Pirate Cove props on top of them. */
        int owner = o->owner_segment_plus_one - 1;
        if (owner >= first && owner <= last) {
            Tr2Segment* seg = &env->segments[owner % TR2_SEG_RING];
            int vi = seg->var_idx;
            /* WaterSlide prefabs bake their duck/blocked-lane geometry into
             * the flume mesh; a generic primitive overlay would be wrong. */
            if (seg->valid && seg->mode == TR2_MODE_WATER) continue;
            if (seg->valid && vi >= 0 && vi < TR2_R_MODEL_SLOTS &&
                    tr2_r_models[vi].state == 1) {
                const char* name = TR2_VARIATION_TABLE[vi].name;
                if (strcmp(name, "jump_over_a") == 0 ||
                    strcmp(name, "slide_under_a") == 0 ||
                    strcmp(name, "stumble_a") == 0 ||
                    strcmp(name, "gap_small_a") == 0) continue;
            }
        }
        float wx, wy, wz, yaw;
        tr2_world_at(env, o->s, o->lane, 0.0f, &wx, &wy, &wz, &yaw);
        /* Shipped prefab first; the procedural primitive is only reached when
         * the matching OBJ is absent, so nothing is ever drawn twice. */
        int pi = tr2_r_obstacle_prop(o->kind);
        Model* pm = pi >= 0 ? tr2_r_prop_model(pi) : NULL;
        if (pm != NULL) {
            tr2_r_draw_prop(pm, &TR2_PROP_TABLE[pi], (Vector3){wx, wy, wz},
                yaw, WHITE);
            continue;
        }
        rlPushMatrix();
        rlTranslatef(0.0f, wy, 0.0f);
        switch (o->kind) {
            case TR2_OBST_LOW:
            case TR2_OBST_STUMBLE:
                tr2_r_draw_low(env, o, wx, wy, wz, yaw);
                break;
            case TR2_OBST_HIGH:
                tr2_r_draw_high(env, o, wx, wy, wz, yaw);
                break;
            case TR2_OBST_GAP:
                tr2_r_draw_gap(env, o, wx, wy, wz, yaw);
                break;
            case TR2_OBST_FIRE:
                tr2_r_draw_fire(env, o, wx, wy, wz, yaw);
                break;
            case TR2_OBST_LEDGE_LEFT:
            case TR2_OBST_LEDGE_RIGHT:
                tr2_r_draw_ledge(env, o, wx, wy, wz, yaw);
                break;
            case TR2_OBST_WATER:
                /* Water is part of the owning track mesh, not a statue. */
                break;
            default:
                tr2_r_draw_block(env, o, wx, wy, wz, yaw);
                break;
        }
        rlPopMatrix();
    }
}

static void tr2_r_draw_items(Env* env) {
    for (int i = 0; i < TR2_MAX_COINS; i++) {
        Tr2Coin* c = &env->coin_pool[i];
        if (!c->active) continue;
        float rel = c->s - env->s;
        if (rel < -4.0f || rel > 70.0f) continue;
        float wx, wy, wz, yaw;
        tr2_world_at(env, c->s, c->lane, c->y, &wx, &wy, &wz, &yaw);
        /* Shipped coin mesh (gold/red/blue by value) when present. */
        int cpi = tr2_r_coin_prop(c->value);
        Model* cmodel = tr2_r_prop_model(cpi);
        if (cmodel != NULL) {
            tr2_r_draw_prop(cmodel, &TR2_PROP_TABLE[cpi],
                (Vector3){wx, wy, wz}, yaw, WHITE);
            continue;
        }
        /* Disc facing the runner: axis along the path forward. */
        float fx = sinf(yaw), fz = cosf(yaw);
        Vector3 a = {wx - fx * 0.04f, wy, wz - fz * 0.04f};
        Vector3 b = {wx + fx * 0.04f, wy, wz + fz * 0.04f};
        Color face = c->value == 3 ? (Color){70, 150, 240, 255}
                   : c->value == 2 ? (Color){224, 72, 64, 255}
                                   : (Color){246, 196, 54, 255};
        Color edge = c->value == 3 ? (Color){30, 80, 160, 255}
                   : c->value == 2 ? (Color){140, 30, 24, 255}
                                   : (Color){170, 120, 20, 255};
        float radius = c->value >= 2 ? 0.34f : 0.26f;
        DrawCylinderEx(a, b, radius, radius, 12, face);
        DrawCylinderWiresEx(a, b, radius, radius, 12, edge);
    }
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) {
        Tr2Powerup* p = &env->powerups[i];
        if (!p->active) continue;
        float rel = p->s - env->s;
        if (rel < -4.0f || rel > 70.0f) continue;
        float wx, wy, wz, yaw;
        tr2_world_at(env, p->s, p->lane, 2.75f, &wx, &wy, &wz, &yaw);
        /* Shipped pickup mesh when present (bonus kinds ship sprites only). */
        int ppi = tr2_r_powerup_prop(p->kind);
        Model* pmodel = ppi >= 0 ? tr2_r_prop_model(ppi) : NULL;
        Color col;
        switch (p->kind) {
            case TR2_PU_MAGNET:     col = (Color){220, 60, 60, 255}; break;
            case TR2_PU_BOOST:      col = (Color){250, 210, 80, 255}; break;
            case TR2_PU_SHIELD:     col = (Color){70, 150, 235, 255}; break;
            case TR2_PU_GEM:        col = (Color){120, 220, 120, 255}; break;
            case TR2_PU_COIN_BONUS: col = (Color){246, 196, 54, 255}; break;
            default:                col = (Color){190, 140, 235, 255}; break;
        }
        float bob = sinf((float)env->tick * 0.08f) * 0.08f;
        if (pmodel != NULL) {
            tr2_r_draw_prop(pmodel, &TR2_PROP_TABLE[ppi],
                (Vector3){wx, wy + bob, wz}, yaw, WHITE);
            continue;
        }
        DrawSphere((Vector3){wx, wy + bob, wz}, 0.38f, col);
        DrawSphereWires((Vector3){wx, wy + bob, wz}, 0.46f, 8, 8,
            (Color){255, 255, 255, 160});
    }
}

static const char* tr2_r_death_name(int cause) {
    switch (cause) {
        case TR2_DEATH_FALL:       return "FELL";
        case TR2_DEATH_EATEN:      return "EATEN";
        case TR2_DEATH_FIRE:       return "BURNED";
        case TR2_DEATH_MINEDUCK:   return "MINE - DUCK";
        case TR2_DEATH_MINELEDGE:  return "MINE - LEDGE";
        case TR2_DEATH_MINE:       return "MINE";
        case TR2_DEATH_WHEEL:      return "WHEEL";
        case TR2_DEATH_WATERLEFT:  return "WATER - LEFT";
        case TR2_DEATH_LEDGE:      return "FELL OFF LEDGE";
        case TR2_DEATH_SCENERYTREE:return "TREE";
        case TR2_DEATH_SCENERYROCK:return "ROCK";
        case TR2_DEATH_DEATHTHROW: return "DEATH THROW";
        case TR2_DEATH_WATERRIGHT: return "WATER - RIGHT";
        case TR2_DEATH_NONE:       return "RUN ENDED";
        default:                   return "RUN OVER";
    }
}

/* Character state + animation clock handed to tr2_character_draw. */
static int tr2_r_char_state(Env* env) {
    if (env->dead) return TR2_CHAR_STATE_DEAD;
    if (env->mode == TR2_MODE_ZIPLINE) return TR2_CHAR_STATE_ZIPLINE;
    if (env->mode == TR2_MODE_WATER) return TR2_CHAR_STATE_SLIDE;
    if (env->sliding) return TR2_CHAR_STATE_SLIDE;
    if (env->mode == TR2_MODE_FALLING) return TR2_CHAR_STATE_FALL;
    if (!env->grounded) return TR2_CHAR_STATE_JUMP;
    return TR2_CHAR_STATE_RUN;
}

static float tr2_r_char_anim(Env* env, int state) {
    static const Env* previous_env = NULL;
    static int previous_state = -1;
    static int previous_tick = -1;
    static int state_start_tick = 0;
    if (env != previous_env || state != previous_state || env->tick < previous_tick) {
        state_start_tick = env->tick;
        previous_env = env;
        previous_state = state;
    }
    previous_tick = env->tick;
    return (float)(env->tick - state_start_tick) / 60.0f;
}

#ifndef TR2_R_HAVE_CHARACTER
/* Procedural stand-in for the shipped character model: the same capsule
 * figure the renderer used before the asset port (absent header only). */
static void tr2_character_init(void) {}
static void tr2_character_draw(Env* env, Vector3 pos, float yaw, float scale,
        int state, float anim_t) {
    (void)state;
    if (scale <= 0.0f) scale = 1.0f;
    float wx = pos.x, wy = pos.y, wz = pos.z;
    float fx = sinf(yaw), fz = cosf(yaw);
    float lx = cosf(yaw), lz = -sinf(yaw);
    Color shirt = {214, 86, 54, 255};
    Color skin = {226, 178, 132, 255};
    if (env->invuln_timer > 0.0f) shirt = (Color){255, 120, 120, 255};
    if (env->sliding) {
        Vector3 a = {wx - fx * 0.35f * scale, wy + 0.30f * scale,
                     wz - fz * 0.35f * scale};
        Vector3 b = {wx + fx * 0.45f * scale, wy + 0.30f * scale,
                     wz + fz * 0.45f * scale};
        DrawCapsule(a, b, 0.32f * scale, 8, 8, shirt);
        DrawSphere((Vector3){wx + fx * 0.66f * scale, wy + 0.34f * scale,
            wz + fz * 0.66f * scale}, 0.22f * scale, skin);
    } else {
        float bob = env->grounded ? sinf(anim_t * 30.0f) * 0.04f : 0.0f;
        float stride = env->grounded ? sinf(anim_t * 30.0f) : 0.0f;
        Vector3 a = {wx, wy + 0.42f * scale, wz};
        Vector3 b = {wx, wy + (1.22f + bob) * scale, wz};
        DrawCapsule(a, b, 0.34f * scale, 8, 8, shirt);
        DrawSphere((Vector3){wx, wy + (1.48f + bob) * scale, wz},
            0.24f * scale, skin);
        Vector3 l0 = {wx - lx * 0.12f * scale, wy + 0.06f * scale,
                      wz - lz * 0.12f * scale};
        Vector3 l1 = {wx - lx * 0.12f * scale + fx * stride * 0.28f * scale,
                      wy + 0.42f * scale,
                      wz - lz * 0.12f * scale + fz * stride * 0.28f * scale};
        Vector3 r0 = {wx + lx * 0.12f * scale, wy + 0.06f * scale,
                      wz + lz * 0.12f * scale};
        Vector3 r1 = {wx + lx * 0.12f * scale - fx * stride * 0.28f * scale,
                      wy + 0.42f * scale,
                      wz + lz * 0.12f * scale - fz * stride * 0.28f * scale};
        DrawCapsule(l0, l1, 0.1f * scale, 6, 6, (Color){62, 52, 42, 255});
        DrawCapsule(r0, r1, 0.1f * scale, 6, 6, (Color){62, 52, 42, 255});
        Vector3 arm0 = {wx - lx * 0.33f * scale, wy + 1.02f * scale,
                        wz - lz * 0.33f * scale};
        Vector3 arm1 = {wx - lx * 0.33f * scale - fx * stride * 0.24f * scale,
                        wy + 0.62f * scale,
                        wz - lz * 0.33f * scale - fz * stride * 0.24f * scale};
        Vector3 arm2 = {wx + lx * 0.33f * scale, wy + 1.02f * scale,
                        wz + lz * 0.33f * scale};
        Vector3 arm3 = {wx + lx * 0.33f * scale + fx * stride * 0.24f * scale,
                        wy + 0.62f * scale,
                        wz + lz * 0.33f * scale + fz * stride * 0.24f * scale};
        DrawCapsule(arm0, arm1, 0.08f * scale, 6, 6, skin);
        DrawCapsule(arm2, arm3, 0.08f * scale, 6, 6, skin);
    }
}
#endif

static void tr2_r_draw_player(Env* env) {
    float wx, wy, wz, yaw;
    tr2_world_at(env, env->s, env->px, env->y, &wx, &wy, &wz, &yaw);
    /* Draw the model along the smoothly averaged path heading so it sweeps
     * through a control-polygon bend instead of popping; its position still
     * comes from the sampler, so the trajectory and controls are unchanged. */
    yaw = tr2_r_path_yaw(env, env->s);
    float fx, fz, lx, lz;
    tr2_r_frame(yaw, &fx, &fz, &lx, &lz);
    (void)lx; (void)lz;

    /* Shipped character model; the owning header draws its own procedural
     * capsule fallback when its assets are absent. */
    int state = tr2_r_char_state(env);
    tr2_character_draw(env, (Vector3){wx, wy, wz}, yaw, 1.0f,
        state, tr2_r_char_anim(env, state));

    DrawCircle3D((Vector3){wx, wy - env->y + 0.02f, wz}, 0.45f, (Vector3){1.0f, 0.0f, 0.0f},
        90.0f, (Color){0, 0, 0, 70});
    if (env->shield_timer > 0.0f) {
        DrawSphereWires((Vector3){wx, wy + 0.9f, wz}, 0.85f, 8, 8,
            (Color){70, 150, 235, 180});
    }
    if (env->magnet_timer > 0.0f) {
        DrawCircle3D((Vector3){wx, wy - env->y + 0.05f, wz}, 2.2f, (Vector3){1.0f, 0.0f, 0.0f},
            90.0f, (Color){230, 80, 80, 90});
    }
    if (env->mode == TR2_MODE_ZIPLINE) {
        /* The runner hangs below the rope; draw it through the hand line. */
        DrawLine3D((Vector3){wx - fx * 3.0f, wy + 1.55f, wz - fz * 3.0f},
            (Vector3){wx + fx * 12.0f, wy + 1.75f, wz + fz * 12.0f},
            (Color){70, 55, 40, 255});
    }
}

#ifndef TR2_R_HAVE_HUD
/* Procedural HUD used when the shipped HUD header is absent. */
static void tr2_hud_init(void) {}
static void tr2_hud_draw(Env* env) {
    int w = GetScreenWidth();
    int h = GetScreenHeight();
    DrawRectangle(0, 0, w, 84, (Color){0, 0, 0, 120});
    DrawText(TextFormat("%d", (int)env->score), 18, 8, 38,
        (Color){255, 226, 140, 255});
    DrawText(TextFormat("%d m", (int)env->distance), 18, 50, 18,
        (Color){225, 225, 210, 255});
    DrawText(TextFormat("x%d", env->multiplier), 18, 68, 18,
        (Color){255, 190, 80, 255});

    DrawText(TextFormat("%d", env->coins), w - 92, 8, 34,
        (Color){246, 196, 54, 255});
    DrawText("COINS", w - 92, 46, 16, (Color){225, 225, 210, 255});
    DrawText(TextFormat("%.1f m/s", env->speed), w - 190, 74, 18,
        (Color){210, 210, 200, 255});

    int py = 12;
    int px = w / 2 - 60;
    if (env->magnet_timer > 0.0f) {
        DrawText(TextFormat("MAGNET %.0f", env->magnet_timer), px, py, 20,
            (Color){230, 80, 80, 255});
        py += 24;
    }
    if (env->shield_timer > 0.0f) {
        DrawText(TextFormat("SHIELD %.0f", env->shield_timer), px, py, 20,
            (Color){80, 165, 245, 255});
        py += 24;
    }
    if (env->invuln_timer > 0.0f) {
        DrawText(TextFormat("INVULN %.0f", env->invuln_timer), px, py, 20,
            (Color){255, 140, 140, 255});
        py += 24;
    }
    if (env->boost_distance_left > 0.0f) {
        DrawText(TextFormat("BOOST %d m", (int)env->boost_distance_left), px, py,
            20, (Color){255, 220, 90, 255});
        py += 24;
    }
    if (env->gems > 0) {
        DrawText(TextFormat("%d GEMS", env->gems), px, py, 20,
            (Color){130, 230, 130, 255});
        py += 24;
    }
    if (env->sliding) {
        DrawText("SLIDE", 18, h - 34, 20, (Color){150, 220, 255, 255});
    }

    if (env->dead) {
        const char* name = tr2_r_death_name(env->death_cause);
        int tw = MeasureText(name, 40);
        DrawText(name, (w - tw) / 2, h / 2 - 20, 40,
            (Color){240, 90, 70, 255});
    }
}
#endif

static void tr2_r_draw_hud(Env* env) {
    tr2_hud_draw(env);
}

static void tr2_r_human_controls(Env* env) {
    if (!IsWindowReady()) return;
    float* a = env->agents[0].actions;
    if (a == NULL) return;
    if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W) || IsKeyPressed(KEY_SPACE)) {
        a[0] = TR2_ACT_JUMP;
    } else if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
        a[0] = TR2_ACT_SLIDE;
    } else if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
        a[0] = TR2_ACT_LEFT;
    } else if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
        a[0] = TR2_ACT_RIGHT;
    } else {
        a[0] = TR2_ACT_NONE;
    }
}

static const Env* tr2_r_cam_env = NULL;
static int tr2_r_cam_tick = -1;

static void tr2_r_update_camera(Env* env, Tr2Client* c, float wx, float wz,
        float yaw, float dt) {
    float dist = 7.6f + 0.06f * env->speed;
    float height = 4.6f;
    /* puf_reset() rewinds env->tick inside puf_step() but c survives, so reseed
     * at an episode boundary instead of easing the chase pose across the map. */
    int restart = env != tr2_r_cam_env
        || (tr2_r_cam_tick >= 0 && env->tick < tr2_r_cam_tick);
    tr2_r_cam_env = env;
    tr2_r_cam_tick = env->tick;
    if (c->frame <= 1 || restart) {
        c->cam_x = wx - sinf(yaw) * dist;
        c->cam_z = wz - cosf(yaw) * dist;
        c->cam_y = height;
        c->cam_yaw = yaw;
        return;
    }
    /* Frame-rate independent easing: the 0.28 / 0.18 / 0.2 factors were tuned
     * per 60 Hz frame, so scale them by the real frame time.  Otherwise the
     * camera lag (and so the visible jitter) changes with the frame rate. */
    float sdt = dt > 0.0f ? dt * 60.0f : 1.0f;
    float kpos = 0.28f * sdt, kyaw = 0.18f * sdt, ky = 0.20f * sdt;
    if (kpos > 1.0f) kpos = 1.0f;
    if (kyaw > 1.0f) kyaw = 1.0f;
    if (ky > 1.0f) ky = 1.0f;
    float dyaw = yaw - c->cam_yaw;
    while (dyaw > PI) dyaw -= 2.0f * PI;
    while (dyaw < -PI) dyaw += 2.0f * PI;
    c->cam_yaw += dyaw * kyaw;
    float behind_x = wx - sinf(c->cam_yaw) * dist;
    float behind_z = wz - cosf(c->cam_yaw) * dist;
    c->cam_x += (behind_x - c->cam_x) * kpos;
    c->cam_z += (behind_z - c->cam_z) * kpos;
    c->cam_y += (height - c->cam_y) * ky;
}

#ifndef TR2_R_HAVE_ATMOSPHERE
/* No shipped atmosphere pass: leave camera and environment untouched. */
static void tr2_atmosphere_apply(Env* env, Camera3D* cam, float dt) {
    (void)env; (void)cam; (void)dt;
}
#endif

static void tr2_render_frame(Env* env, int read_input) {
    if (!IsWindowReady()) {
        InitWindow(540, 1170, "Temple Run 2 (PufferLib)");
        int monitor = GetCurrentMonitor();
        int available_height = GetMonitorHeight(monitor) - 100;
        if (available_height > 0 && available_height < 1170) {
            int width = 540 * available_height / 1170;
            SetWindowSize(width, available_height);
            Vector2 origin = GetMonitorPosition(monitor);
            SetWindowPosition((int)origin.x + (GetMonitorWidth(monitor) - width) / 2,
                (int)origin.y + 40);
        }
        SetTargetFPS(60);
        if (env->client == NULL) {
            env->client = (Tr2Client*)calloc(1, sizeof(Tr2Client));
        }
        tr2_r_load_assets();
        tr2_character_init();
        tr2_hud_init();
        tr2_r_warm_track_models();
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        if (env->client != NULL) {
            free(env->client);
            env->client = NULL;
        }
        if (IsWindowReady()) CloseWindow();
        return;
    }
    if (read_input) tr2_r_human_controls(env);

    Tr2Client* c = env->client;
    if (c == NULL) return;
    c->frame++;

    float frame_dt = GetFrameTime();
    float wx, wy, wz, ignored_yaw;
    tr2_world_at(env, env->s, env->px, env->y, &wx, &wy, &wz, &ignored_yaw);
    /* Feed the camera the smoothly averaged path heading, not the piecewise
     * constant chord tangent, so a bend rotates the chase offset instead of
     * jerking it sideways. */
    float path_yaw = tr2_r_path_yaw(env, env->s);
    (void)ignored_yaw;
    tr2_r_update_camera(env, c, wx, wz, path_yaw, frame_dt);

    /* The camera trails the runner, so retain the pieces behind the current
     * one as well. Never read a ring slot already reused by the generator. */
    int first = env->seg_idx;
    int oldest = env->seg_gen - TR2_SEG_RING + 1;
    if (oldest < 0) oldest = 0;
    while (first > oldest) {
        Tr2Segment* previous = &env->segments[(first - 1) % TR2_SEG_RING];
        if (!previous->valid || previous->s0 + previous->length < env->s - 12.0f) break;
        --first;
    }
    int last = env->seg_gen < env->seg_idx + TR2_SEG_AHEAD
        ? env->seg_gen : env->seg_idx + TR2_SEG_AHEAD;
    if (last < first) last = first;

    Camera3D cam;
    memset(&cam, 0, sizeof(cam));
    cam.position = (Vector3){c->cam_x, c->cam_y, c->cam_z};
    cam.target = (Vector3){wx + sinf(c->cam_yaw) * 12.0f, wy + 1.2f,
                           wz + cosf(c->cam_yaw) * 12.0f};
    cam.up = (Vector3){0.0f, 1.0f, 0.0f};
    cam.fovy = 62.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    /* Atmosphere pass may retune the camera / environment before it is used. */
    tr2_atmosphere_apply(env, &cam, frame_dt);

    BeginDrawing();
    ClearBackground((Color){30, 22, 38, 255});
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    if (tr2_r_assets_loaded && tr2_r_sky.id > 0) {
        /* The extracted atlas strip has transparent padding in its first
         * ten rows; exclude it instead of exposing the clear colour. */
        DrawTexturePro(tr2_r_sky,
            (Rectangle){0, 10, (float)tr2_r_sky.width, (float)tr2_r_sky.height - 10.0f},
            (Rectangle){0, 0, (float)sw, (float)sh}, (Vector2){0, 0}, 0.0f, WHITE);
    } else {
        DrawRectangleGradientV(0, 0, sw, sh / 2,
            (Color){70, 44, 90, 255}, (Color){196, 156, 138, 255});
    }
    /* Sky_Machu01 is an atlas: its upper-left transparent layer is the
     * Halloween moon visible in the captured Machu scene. */
    if (tr2_r_sky_layers.id > 0) {
        float moon_w = 0.35f * (float)sw;
        DrawTexturePro(tr2_r_sky_layers, (Rectangle){0, 0, 305, 325},
            (Rectangle){0.08f * sw, 0.005f * sh, moon_w, moon_w * (325.0f / 305.0f)},
            (Vector2){0, 0}, 0.0f, WHITE);
    }

    BeginMode3D(cam);
    rlDisableBackfaceCulling();
    tr2_r_draw_track(env, first, last);
    tr2_r_draw_player(env);
    tr2_r_draw_obstacles(env, first, last);
    tr2_r_draw_items(env);
    EndMode3D();

    tr2_r_draw_hud(env);
    EndDrawing();
    puf_web_vsync();
}

void puf_render(Env* env) {
    tr2_render_frame(env, 1);
}
