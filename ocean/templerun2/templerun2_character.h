/* Temple Run 2 — runner character renderer (header-only, C99, all symbols
 * static).  Loads the shipped runner model with its baked clips and falls back
 * to the procedural capsule figure when the asset (or a raylib GLB load) is
 * unavailable, so the rewrite TU always links and never crashes.
 *
 * Assets, probed with FileExists (all optional):
 *   resources/templerun2/character/guy.glb            preferred: skinned mesh
 *                                                     + baked clips
 *   resources/templerun2/character/guy.obj            static fallback mesh
 *   resources/templerun2/character/guy.png            diffuse map for the .obj
 *   resources/templerun2/character/animations.json    optional overrides
 *
 * animations.json (every key optional): top-level "yaw_offset_deg" (default
 * 180 — the standard Unity −Z-forward correction, matching the track models),
 * "scale" (default 1.0) and "fps" (default 30, the Unity m_SampleRate); per
 * clip {"name": "...", "fps": N}.
 *
 * ---------------------------------------------------------------------------
 * Animation state mapping (the `state` argument of tr2_character_draw):
 *
 *   TR2_CHAR_STATE_RUN    grounded, moving      -> "run"    (loop)
 *   TR2_CHAR_STATE_JUMP   jumping               -> "jump"   (one-shot)
 *   TR2_CHAR_STATE_SLIDE  slide/duck            -> "slide"  (one-shot)
 *   TR2_CHAR_STATE_IDLE   standing still        -> "idle"   (loop)
 *   TR2_CHAR_STATE_FALL   falling state         -> "fall" | "jump" (one-shot)
 *   TR2_CHAR_STATE_DEAD   run over              -> "death"  (play once, hold)
 *
 * Clip lookup per state is exact-name first, then case-insensitive substring
 * over the candidate list ("run"/"walk"/"sprint", "jump"/"leap"/"hop",
 * "slide"/"roll"/"duck", "idle"/"stand"/"breath", "fall"/"drop"/"air",
 * "death"/"die"/"dead"); a state that matches nothing reuses the run clip, and
 * a model with no usable clips at all falls back to clip 0. `anim_t` is
 * elapsed time in the current state; one-shot clips hold their final pose.
 *
 * When the model carries no skeleton (static .obj, or a .glb with no clips) the
 * same state drives a procedural bob / lean / stride / crouch on the mesh, and
 * when no model loads at all it drives the capsule figure that used to live in
 * templerun2_render.h — identical look, so the renderer can drop its own copy.
 *
 * Conventions: `pos` is the runner's feet/ground point, `yaw` uses the track
 * frame (`forward = (sin yaw, cos yaw)`, `lateral = (cos yaw, -sin yaw)`),
 * `scale` is a unitless multiplier (the model is metres, ~1.8 tall).  Must be
 * included after raylib.h and templerun2_types.h — true inside templerun2.h,
 * where templerun2_render.h pulls it in.
 */
#pragma once

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
#include "templerun2_types.h"

/* ---- animation states (the `state` argument) ----------------------------
 * Plain macros (not an enum) guarded with #ifndef so the renderer's own
 * fallback copies in templerun2_render.h cannot clash, whatever the include
 * order; both define the identical values. */
#ifndef TR2_CHAR_STATE_RUN
#define TR2_CHAR_STATE_RUN   0
#define TR2_CHAR_STATE_JUMP  1
#define TR2_CHAR_STATE_SLIDE 2
#define TR2_CHAR_STATE_IDLE  3
#define TR2_CHAR_STATE_FALL  4
#define TR2_CHAR_STATE_DEAD  5
#define TR2_CHAR_STATE_ZIPLINE 6
#endif
#ifndef TR2_CHAR_STATE_COUNT
#define TR2_CHAR_STATE_COUNT 7
#endif

#define TR2_CHAR_DIR   "resources/templerun2/character/"
#define TR2_CHAR_GLB   TR2_CHAR_DIR "guy.glb"
#define TR2_CHAR_OBJ   TR2_CHAR_DIR "guy.obj"
#define TR2_CHAR_PNG   TR2_CHAR_DIR "guy.png"
#define TR2_CHAR_JSON  TR2_CHAR_DIR "animations.json"

/* raylib 5.5 resamples glTF animations at 17 ms per frame. */
#define TR2_CHAR_DEFAULT_FPS (1000.0f / 17.0f)
#define TR2_CHAR_DEFAULT_YAW 0.0f

/* Candidate clip names per state, best match first (exact, then substring). */
static const char* const tr2_char_clip_names[TR2_CHAR_STATE_COUNT][4] = {
    {"run",   "walk",  "sprint", NULL},
    {"jump",  "leap",  "hop",    NULL},
    {"slide", "roll",  "duck",   NULL},
    {"idle",  "stand", "breath", NULL},
    {"fall",  "drop",  "air",    NULL},
    {"death", "die",   "dead",   NULL},
    {"zipline", "zipline_idle", "hang", NULL},
};

typedef struct {
    Model model;                            /* valid iff loaded */
    Matrix base_xform;                      /* loader transform, restored */
    ModelAnimation* anims;
    int anim_count;
    int clip[TR2_CHAR_STATE_COUNT];         /* state -> clip index, -1 none */
    int loaded;                             /* 1 = a mesh is on the GPU */
    int skinned;                            /* 1 = >=1 skeleton-valid clip */
    float yaw_offset_deg;
    float model_scale;
    float ground_y;                         /* bbox lift, model units */
} Tr2CharModel;

static Tr2CharModel tr2_char_model = {0};
static int tr2_char_ready = 0;

/* ---- tiny JSON probes (no dependency on a JSON parser) ------------------ */

/* Pointer just past `"key":` with leading spaces skipped, or NULL. */
static const char* tr2_char_json_value(const char* json, const char* key) {
    if (json == NULL || key == NULL) return NULL;
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    size_t n = strlen(pat);
    const char* p = json;
    while ((p = strstr(p, pat)) != NULL) {
        const char* q = p + n;
        while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n') q++;
        if (*q == ':') {
            q++;
            while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n') q++;
            return q;
        }
        p = q;
    }
    return NULL;
}

static float tr2_char_json_num(const char* json, const char* key, float def) {
    const char* v = tr2_char_json_value(json, key);
    if (v == NULL || !(*v == '-' || *v == '+' || (*v >= '0' && *v <= '9')))
        return def;
    return (float)atof(v);
}


/* ---- clip matching ------------------------------------------------------ */

/* 2 = exact name, 1 = substring, 0 = no match. */
static int tr2_char_clip_match(const char* name, const char* const* cands) {
    if (name == NULL || name[0] == '\0') return 0;
    char low[64];
    size_t n = strlen(name);
    if (n >= sizeof(low)) n = sizeof(low) - 1;
    for (size_t k = 0; k < n; k++) low[k] = (char)tolower((unsigned char)name[k]);
    low[n] = '\0';
    for (int i = 0; cands[i] != NULL; i++)
        if (strcmp(low, cands[i]) == 0) return 2;
    for (int i = 0; cands[i] != NULL; i++)
        if (strstr(low, cands[i]) != NULL) return 1;
    return 0;
}

static int tr2_char_find_clip(const ModelAnimation* anims, int count,
        const char* const* cands) {
    int best = -1;
    int best_score = 0;
    for (int i = 0; i < count; i++) {
        int score = tr2_char_clip_match(anims[i].name, cands);
        if (score > best_score) {
            best_score = score;
            best = i;
            if (score == 2) break;   /* cannot do better than exact */
        }
    }
    return best;
}

/* ---- load --------------------------------------------------------------- */

static void tr2_character_init(void) {
    if (tr2_char_ready) return;
    tr2_char_ready = 1;

    tr2_char_model.anims = NULL;
    tr2_char_model.anim_count = 0;
    tr2_char_model.loaded = 0;
    tr2_char_model.skinned = 0;
    tr2_char_model.yaw_offset_deg = TR2_CHAR_DEFAULT_YAW;
    tr2_char_model.model_scale = 1.0f;
    tr2_char_model.ground_y = 0.0f;
    for (int i = 0; i < TR2_CHAR_STATE_COUNT; i++) {
        tr2_char_model.clip[i] = -1;
    }

    /* Optional overrides from animations.json. */
    char* json = NULL;
    if (FileExists(TR2_CHAR_JSON)) {
        json = LoadFileText(TR2_CHAR_JSON);
        if (json != NULL) {
            tr2_char_model.yaw_offset_deg =
                tr2_char_json_num(json, "yaw_offset_deg", TR2_CHAR_DEFAULT_YAW);
            tr2_char_model.model_scale = tr2_char_json_num(json, "scale", 1.0f);
        }
    }
    if (!(tr2_char_model.model_scale > 0.0f)) tr2_char_model.model_scale = 1.0f;

    /* Preferred: skinned GLB with baked clips. */
    if (FileExists(TR2_CHAR_GLB)) {
        Model m = LoadModel(TR2_CHAR_GLB);
        if (IsModelValid(m) && m.meshCount > 0) {
            tr2_char_model.model = m;
            tr2_char_model.base_xform = m.transform;
            tr2_char_model.loaded = 1;
            tr2_char_model.anims =
                LoadModelAnimations(TR2_CHAR_GLB, &tr2_char_model.anim_count);
            if (tr2_char_model.anims == NULL) tr2_char_model.anim_count = 0;
        }
    }

    /* Fallback: static OBJ + optional PNG. */
    if (!tr2_char_model.loaded && FileExists(TR2_CHAR_OBJ)) {
        Model m = LoadModel(TR2_CHAR_OBJ);
        if (IsModelValid(m) && m.meshCount > 0) {
            tr2_char_model.model = m;
            tr2_char_model.base_xform = m.transform;
            tr2_char_model.loaded = 1;
            tr2_char_model.anims = NULL;
            tr2_char_model.anim_count = 0;
            /* LoadModel always creates >=1 material; paint the PNG onto it. */
            if (FileExists(TR2_CHAR_PNG) && m.materialCount > 0) {
                Texture2D tex = LoadTexture(TR2_CHAR_PNG);
                if (tex.id > 0) {
                    SetTextureFilter(tex, TEXTURE_FILTER_BILINEAR);
                    for (int i = 0; i < m.materialCount; i++)
                        SetMaterialTexture(&tr2_char_model.model.materials[i],
                            MATERIAL_MAP_DIFFUSE, tex);
                }
            }
        }
    }

    if (tr2_char_model.loaded) {
        BoundingBox bb = GetModelBoundingBox(tr2_char_model.model);
        if (bb.min.y < -0.01f) tr2_char_model.ground_y = -bb.min.y;
    }

    /* Resolve state -> clip, then verify the skeleton actually matches. */
    if (tr2_char_model.anim_count > 0) {
        for (int s = 0; s < TR2_CHAR_STATE_COUNT; s++) {
            int idx = tr2_char_find_clip(tr2_char_model.anims,
                    tr2_char_model.anim_count, tr2_char_clip_names[s]);
            if (idx >= 0) tr2_char_model.clip[s] = idx;
        }
        /* A state with no clip reuses the run clip; if even that is missing,
         * fall back to clip 0 so a single-clip file still animates. */
        if (tr2_char_model.clip[TR2_CHAR_STATE_RUN] < 0) {
            tr2_char_model.clip[TR2_CHAR_STATE_RUN] = 0;
        }
        for (int s = 0; s < TR2_CHAR_STATE_COUNT; s++) {
            if (tr2_char_model.clip[s] < 0) {
                tr2_char_model.clip[s] =
                    tr2_char_model.clip[TR2_CHAR_STATE_RUN];
            }
        }
        for (int s = 0; s < TR2_CHAR_STATE_COUNT; s++) {
            int idx = tr2_char_model.clip[s];
            if (idx >= 0 && IsModelAnimationValid(tr2_char_model.model,
                    tr2_char_model.anims[idx])) {
                tr2_char_model.skinned = 1;
                break;
            }
        }
    }

    if (json != NULL) UnloadFileText(json);
}

/* ---- playback ----------------------------------------------------------- */

static int tr2_char_anim_frame(int state, float t, float fps, int frames) {
    if (frames <= 0) return 0;
    if (!(fps > 0.0f)) fps = TR2_CHAR_DEFAULT_FPS;
    if (state != TR2_CHAR_STATE_RUN && state != TR2_CHAR_STATE_IDLE
            && state != TR2_CHAR_STATE_ZIPLINE) {
        int f = t > 0.0f ? (int)(t * fps) : 0;
        return f >= frames ? frames - 1 : f;
    }
    float ph = fmodf(t * fps, (float)frames);
    if (ph < 0.0f) ph += (float)frames;
    int f = (int)ph;
    return f >= frames ? frames - 1 : f;
}

/* ---- mesh path ---------------------------------------------------------- */

/* Minimal 4x4 helpers (raylib Matrix conventions) so this header does not need
 * raymath.h, which conflicts with CUDA float3 in the shared training TU. */
static Matrix tr2_char_rotx(float rad) {
    float c = cosf(rad), s = sinf(rad);
    Matrix m = {1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, c,    s,    0.0f,
                0.0f, -s,   c,    0.0f,
                0.0f, 0.0f, 0.0f, 1.0f};
    return m;
}

static Matrix tr2_char_roty(float rad) {
    float c = cosf(rad), s = sinf(rad);
    Matrix m = {c,    0.0f, s,    0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                -s,   0.0f, c,    0.0f,
                0.0f, 0.0f, 0.0f, 1.0f};
    return m;
}

static Matrix tr2_char_mulm(Matrix l, Matrix r) {
    Matrix o;
    o.m0  = l.m0 *r.m0  + l.m1 *r.m4  + l.m2 *r.m8  + l.m3 *r.m12;
    o.m1  = l.m0 *r.m1  + l.m1 *r.m5  + l.m2 *r.m9  + l.m3 *r.m13;
    o.m2  = l.m0 *r.m2  + l.m1 *r.m6  + l.m2 *r.m10 + l.m3 *r.m14;
    o.m3  = l.m0 *r.m3  + l.m1 *r.m7  + l.m2 *r.m11 + l.m3 *r.m15;
    o.m4  = l.m4 *r.m0  + l.m5 *r.m4  + l.m6 *r.m8  + l.m7 *r.m12;
    o.m5  = l.m4 *r.m1  + l.m5 *r.m5  + l.m6 *r.m9  + l.m7 *r.m13;
    o.m6  = l.m4 *r.m2  + l.m5 *r.m6  + l.m6 *r.m10 + l.m7 *r.m14;
    o.m7  = l.m4 *r.m3  + l.m5 *r.m7  + l.m6 *r.m11 + l.m7 *r.m15;
    o.m8  = l.m8 *r.m0  + l.m9 *r.m4  + l.m10*r.m8  + l.m11*r.m12;
    o.m9  = l.m8 *r.m1  + l.m9 *r.m5  + l.m10*r.m9  + l.m11*r.m13;
    o.m10 = l.m8 *r.m2  + l.m9 *r.m6  + l.m10*r.m10 + l.m11*r.m14;
    o.m11 = l.m8 *r.m3  + l.m9 *r.m7  + l.m10*r.m11 + l.m11*r.m15;
    o.m12 = l.m12*r.m0  + l.m13*r.m4  + l.m14*r.m8  + l.m15*r.m12;
    o.m13 = l.m12*r.m1  + l.m13*r.m5  + l.m14*r.m9  + l.m15*r.m13;
    o.m14 = l.m12*r.m2  + l.m13*r.m6  + l.m14*r.m10 + l.m15*r.m14;
    o.m15 = l.m12*r.m3  + l.m13*r.m7  + l.m14*r.m11 + l.m15*r.m15;
    return o;
}

static void tr2_char_draw_model(Vector3 pos, float yaw, float s, int state,
        float t) {
    float pitch_deg = 0.0f;
    float bob = 0.0f;
    if (!tr2_char_model.skinned) {
        switch (state) {
            case TR2_CHAR_STATE_RUN:
                pitch_deg = -6.0f;
                bob = sinf(t * 12.566f) * 0.03f;
                break;
            case TR2_CHAR_STATE_IDLE:
                bob = sinf(t * 2.2f) * 0.015f;
                break;
            case TR2_CHAR_STATE_JUMP: pitch_deg = 8.0f; break;
            case TR2_CHAR_STATE_FALL: pitch_deg = 4.0f; break;
            case TR2_CHAR_STATE_SLIDE: pitch_deg = -55.0f; bob = -0.10f; break;
            case TR2_CHAR_STATE_DEAD: pitch_deg = -82.0f; bob = -0.05f; break;
            default: break;
        }
    }

    if (tr2_char_model.skinned) {
        int idx = tr2_char_model.clip[state];
        if (idx < 0) idx = tr2_char_model.clip[TR2_CHAR_STATE_RUN];
        if (idx >= 0) {
            ModelAnimation anim = tr2_char_model.anims[idx];
            if (IsModelAnimationValid(tr2_char_model.model, anim)) {
                int frame = tr2_char_anim_frame(state, t,
                        TR2_CHAR_DEFAULT_FPS, anim.frameCount);
                UpdateModelAnimation(tr2_char_model.model, anim, frame);
            }
        }
    }

    Matrix rot = tr2_char_mulm(
        tr2_char_roty(yaw + tr2_char_model.yaw_offset_deg * DEG2RAD),
        tr2_char_rotx(pitch_deg * DEG2RAD));
    tr2_char_model.model.transform = tr2_char_mulm(tr2_char_model.base_xform, rot);
    Vector3 p = {pos.x,
                 pos.y + (bob + tr2_char_model.ground_y) * s,
                 pos.z};
    DrawModelEx(tr2_char_model.model, p, (Vector3){0.0f, 1.0f, 0.0f}, 0.0f,
        (Vector3){s, s, s}, WHITE);
    tr2_char_model.model.transform = tr2_char_model.base_xform;
}

/* ---- capsule path (asset missing) --------------------------------------- */

static void tr2_char_draw_capsule(Vector3 pos, float fx, float fz, float lx,
        float lz, float s, int state, float t, Color shirt, Color skin,
        Color pants) {
    float bob = 0.0f;
    float stride = 0.0f;
    float lean = 0.0f;
    switch (state) {
        case TR2_CHAR_STATE_RUN:
            bob = sinf(t * 12.566f) * 0.04f;
            stride = sinf(t * 12.566f);
            lean = 0.06f;
            break;
        case TR2_CHAR_STATE_IDLE:
            bob = sinf(t * 2.2f) * 0.02f;
            break;
        default:
            break;
    }

    if (state == TR2_CHAR_STATE_SLIDE || state == TR2_CHAR_STATE_DEAD) {
        float drop = state == TR2_CHAR_STATE_DEAD ? 0.22f : 0.30f;
        float side = state == TR2_CHAR_STATE_DEAD ? 0.10f : 0.0f;
        Vector3 a = {pos.x - fx * 0.35f * s + lx * side * s,
                     pos.y + drop * s, pos.z - fz * 0.35f * s + lz * side * s};
        Vector3 b = {pos.x + fx * 0.45f * s + lx * side * s,
                     pos.y + drop * s, pos.z + fz * 0.45f * s + lz * side * s};
        DrawCapsule(a, b, 0.32f * s, 8, 8, shirt);
        DrawSphere((Vector3){pos.x + fx * 0.66f * s + lx * side * s,
                             pos.y + (drop + 0.04f) * s,
                             pos.z + fz * 0.66f * s + lz * side * s},
                   0.22f * s, skin);
        return;
    }

    float hip = 0.42f + bob;
    float head = 1.48f + bob;
    float ax = pos.x + fx * lean * s;
    float az = pos.z + fz * lean * s;

    DrawCapsule((Vector3){ax, pos.y + hip * s, az},
                (Vector3){ax, pos.y + (1.22f + bob) * s, az},
                0.34f * s, 8, 8, shirt);
    DrawSphere((Vector3){ax, pos.y + head * s, az}, 0.24f * s, skin);

    if (state == TR2_CHAR_STATE_JUMP) {
        /* Knees up, arms raised. */
        DrawCapsule((Vector3){pos.x - lx * 0.12f * s, pos.y + 0.62f * s,
                              pos.z - lz * 0.12f * s},
                    (Vector3){pos.x - lx * 0.12f * s + fx * 0.30f * s,
                              pos.y + 0.50f * s,
                              pos.z - lz * 0.12f * s + fz * 0.30f * s},
                    0.10f * s, 6, 6, pants);
        DrawCapsule((Vector3){pos.x + lx * 0.12f * s, pos.y + 0.62f * s,
                              pos.z + lz * 0.12f * s},
                    (Vector3){pos.x + lx * 0.12f * s + fx * 0.30f * s,
                              pos.y + 0.50f * s,
                              pos.z + lz * 0.12f * s + fz * 0.30f * s},
                    0.10f * s, 6, 6, pants);
        DrawCapsule((Vector3){pos.x - lx * 0.33f * s, pos.y + 1.05f * s,
                              pos.z - lz * 0.33f * s},
                    (Vector3){pos.x - lx * 0.42f * s, pos.y + 1.45f * s,
                              pos.z - lz * 0.42f * s},
                    0.08f * s, 6, 6, skin);
        DrawCapsule((Vector3){pos.x + lx * 0.33f * s, pos.y + 1.05f * s,
                              pos.z + lz * 0.33f * s},
                    (Vector3){pos.x + lx * 0.42f * s, pos.y + 1.45f * s,
                              pos.z + lz * 0.42f * s},
                    0.08f * s, 6, 6, skin);
        return;
    }

    float swing = (state == TR2_CHAR_STATE_FALL) ? 0.22f : stride;
    /* Legs. */
    DrawCapsule((Vector3){pos.x - lx * 0.12f * s, pos.y + 0.06f * s,
                          pos.z - lz * 0.12f * s},
                (Vector3){pos.x - lx * 0.12f * s + fx * swing * 0.28f * s,
                          pos.y + 0.42f * s,
                          pos.z - lz * 0.12f * s + fz * swing * 0.28f * s},
                0.10f * s, 6, 6, pants);
    DrawCapsule((Vector3){pos.x + lx * 0.12f * s, pos.y + 0.06f * s,
                          pos.z + lz * 0.12f * s},
                (Vector3){pos.x + lx * 0.12f * s - fx * swing * 0.28f * s,
                          pos.y + 0.42f * s,
                          pos.z + lz * 0.12f * s - fz * swing * 0.28f * s},
                0.10f * s, 6, 6, pants);
    /* Arms. */
    DrawCapsule((Vector3){pos.x - lx * 0.33f * s, pos.y + 1.02f * s,
                          pos.z - lz * 0.33f * s},
                (Vector3){pos.x - lx * 0.33f * s - fx * stride * 0.24f * s,
                          pos.y + 0.62f * s,
                          pos.z - lz * 0.33f * s - fz * stride * 0.24f * s},
                0.08f * s, 6, 6, skin);
    DrawCapsule((Vector3){pos.x + lx * 0.33f * s, pos.y + 1.02f * s,
                          pos.z + lz * 0.33f * s},
                (Vector3){pos.x + lx * 0.33f * s + fx * stride * 0.24f * s,
                          pos.y + 0.62f * s,
                          pos.z + lz * 0.33f * s + fz * stride * 0.24f * s},
                0.08f * s, 6, 6, skin);
}

/* ---- public entry points ------------------------------------------------ */

static void tr2_character_draw(Env* env, Vector3 pos, float yaw, float scale,
        int state, float anim_t) {
    if (!tr2_char_ready) tr2_character_init();
    if (state < 0 || state >= TR2_CHAR_STATE_COUNT) state = TR2_CHAR_STATE_RUN;
    if (!(scale > 0.0f)) scale = 1.0f;
    float s = scale * tr2_char_model.model_scale;
    float t = anim_t;

    Color shirt = {214, 86, 54, 255};
    Color skin = {226, 178, 132, 255};
    Color pants = {62, 52, 42, 255};
    if (env != NULL && env->invuln_timer > 0.0f)
        shirt = (Color){255, 120, 120, 255};

    if (tr2_char_model.loaded) {
        tr2_char_draw_model(pos, yaw, s, state, t);
    } else {
        tr2_char_draw_capsule(pos, sinf(yaw), cosf(yaw), cosf(yaw), -sinf(yaw),
            s, state, t, shirt, skin, pants);
    }

    /* Contact shadow, matching the shadow the capsule figure used to carry. */
    DrawCircle3D((Vector3){pos.x, pos.y + 0.02f, pos.z}, 0.45f * s,
        (Vector3){1.0f, 0.0f, 0.0f}, 90.0f, (Color){0, 0, 0, 70});
}
