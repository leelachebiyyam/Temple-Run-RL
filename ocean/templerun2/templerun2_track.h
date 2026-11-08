/* Temple Run 2 — track generation, path transforms and entity placement.
 *
 * Data sources (included by templerun2.h before this file):
 *   templerun2_track_data.h    TR2_PIECE_TABLE / TR2_VARIATION_TABLE, TR2_FLAG_*
 *   templerun2_pieces_data.h   tr2_piece_pool / tr2_piece_entry / category / mode
 *   templerun2_pacing_data.h   Tr2PacingRow / tr2_pacing_lookup / coin+powerup
 *   templerun2_colliders_data.h tr2_volumes_for_variation / tr2_volume_for_kind
 *
 * The data-driven generator picks real shipped RegionID 100 pieces by category
 * (mandatory turn due -> turn, obstacle due -> obstacle, else straight / curve /
 * sub-mode), then chains them using the shipped GeneratedPath polyline,
 * SourceLastYRot and trigger volumes owned by the selected variation.
 *
 * Unity's piece-local forward is -Z and its +X points left. The renderer's
 * right-handed world reflects X: forward is (sin h, cos h), while positive
 * lane offsets point screen-right. Piece joins use the source end rotation.
 */
#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Sentinel for Tr2Segment.obstacle_kind: no entity was stamped. */
#define TR2_TK_ENTITY_NONE 255

#define TR2_TK_DEG 0.017453292519943295f   /* pi / 180 */
#define TR2_TK_TWO_PI 6.283185307179586f
#define TR2_TK_MIN_LEN 0.05f               /* ignore degenerate est_len */

/* Default geometry when a variation row is missing (metres). */
#define TR2_TK_LEN_STRAIGHT 5.12f
#define TR2_TK_LEN_TURN 10.24f
#define TR2_TK_LEN_CURVE 21.08f
#define TR2_TK_LEN_OBSTACLE 7.68f
#define TR2_TK_LEN_SUBMODE 5.16f

/* Turn clearance: no obstacle volumes within this many metres of a turn arc. */
#define TR2_TK_TURN_MARGIN 8.0f
/* Keep the run-up to a mandatory turn obstacle-free. */
#define TR2_TK_TURN_OBSTACLE_HOLD 12.0f
/* MinDistanceAfterTurnForObstacle (milestone default, TRACK.md 1.3). */
#define TR2_TK_AFTER_TURN_HOLD 10.0f

/* Coin / powerup pacing. */
#define TR2_TK_COIN_START 12.0f
#define TR2_TK_COIN_SPACING 2.2f
#define TR2_TK_COIN_Y 0.5f
#define TR2_TK_COIN_RUN_MIN 4

/* ------------------------------------------------------------------ rng --- */

/* xorshift32, unique to this module so it cannot clash with the runner etc. */
static unsigned int tr2_tk_step_rng(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float tr2_tk_rand(Env* env) {
    return (float)(tr2_tk_step_rng(&env->rng) >> 8) * (1.0f / 16777216.0f);
}

static int tr2_tk_rand_int(Env* env, int n) {
    if (n <= 1) return 0;
    return (int)(tr2_tk_step_rng(&env->rng) % (unsigned int)n);
}

static float tr2_tk_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Wrap an angle in degrees into (-180, 180]. */
static float tr2_tk_norm_deg(float d) {
    d = fmodf(d, 360.0f);
    if (d > 180.0f) d -= 360.0f;
    else if (d <= -180.0f) d += 360.0f;
    return d;
}

/* ------------------------------------------------------------- geometry --- */

/* Sample each variation's shipped GeneratedPath in the reflected world frame.
 * SourcePath is used only when the generated path has fewer than two points. */

#define TR2_TK_VARIATION_MAX \
    ((int)(sizeof(TR2_VARIATION_TABLE) / sizeof(TR2_VARIATION_TABLE[0])))
#define TR2_TK_SPLINE_MAX \
    ((int)(sizeof(TR2_SPLINE_POINTS) / sizeof(TR2_SPLINE_POINTS[0])))

/* Interpolate the authored spline of `seg` at path distance d (metres):
 * piece-local point (lx, ly, lz) relative to its first control point, plus the
 * local chord tangent (tx, tz) that sets the heading.  Returns 0 when the
 * segment carries no authored spline. */
static int tr2_tk_spline_at(const Tr2Segment* seg, float d, float* lx,
        float* ly, float* lz, float* tx, float* tz) {
    int n = (int)seg->spline_count;
    int b = (int)seg->spline_begin;
    if (n < 2 || b < 0 || b + n > TR2_TK_SPLINE_MAX) return 0;
    const tr2_vec3_t* P = &TR2_SPLINE_POINTS[b];
    float x0 = P[0].x, y0 = P[0].y, z0 = P[0].z;
    if (d < 0.0f) d = 0.0f;
    float acc = 0.0f;
    for (int i = 0; i + 1 < n; i++) {
        float dx = P[i + 1].x - P[i].x;
        float dy = P[i + 1].y - P[i].y;
        float dz = P[i + 1].z - P[i].z;
        float sl = sqrtf(dx * dx + dy * dy + dz * dz);
        if (sl < 1e-6f) continue;
        if (d <= acc + sl || i + 2 == n) {
            float t = (d - acc) / sl;
            if (t < 0.0f) t = 0.0f;
            else if (t > 1.0f) t = 1.0f;
            *lx = P[i].x + dx * t - x0;
            *ly = P[i].y + dy * t - y0;
            *lz = P[i].z + dz * t - z0;
            *tx = dx;
            *tz = dz;
            if (seg->alternate_path) { *lx = -*lx; *tx = -*tx; }
            return 1;
        }
        acc += sl;
    }
    *lx = P[n - 1].x - x0;
    *ly = P[n - 1].y - y0;
    *lz = P[n - 1].z - z0;
    *tx = P[n - 1].x - P[0].x;
    *tz = P[n - 1].z - P[0].z;
    if (seg->alternate_path) { *lx = -*lx; *tx = -*tx; }
    return 1;
}

/* Total authored polyline length of `seg` (metres); equals the variation's
 * EstimatedPathLength to < 1e-4 across the shipped table. */
static float tr2_tk_spline_length(const Tr2Segment* seg) {
    int n = (int)seg->spline_count;
    int b = (int)seg->spline_begin;
    if (n < 2 || b < 0 || b + n > TR2_TK_SPLINE_MAX) return 0.0f;
    const tr2_vec3_t* P = &TR2_SPLINE_POINTS[b];
    float L = 0.0f;
    for (int i = 0; i + 1 < n; i++) {
        float dx = P[i + 1].x - P[i].x;
        float dy = P[i + 1].y - P[i].y;
        float dz = P[i + 1].z - P[i].z;
        L += sqrtf(dx * dx + dy * dy + dz * dz);
    }
    return L;
}

/* Map an authored piece-local point plus a lateral offset (env metres) to
 * world space using the fixed piece-start heading (the game rotates the whole
 * control polygon by the piece transform; it does not follow the local
 * heading point-by-point). */
static void tr2_tk_local_to_world(const Tr2Segment* seg, float lx, float lz,
        float lat_m, float* x, float* z) {
    float sh = sinf(seg->heading), ch = cosf(seg->heading);
    float f = -lz;             /* local -Z is forward */
    float s = lx - lat_m;      /* Unity +X is left; positive lane is right */
    *x = seg->wx + sh * f + ch * s;
    *z = seg->wz + ch * f - sh * s;
}

/* End point and end heading of a segment, from its authored spline. */
static void tr2_seg_end(const Tr2Segment* seg, float* ex, float* ez,
        float* eh) {
    float lx, ly, lz, tx, tz;
    if (tr2_tk_spline_at(seg, seg->length, &lx, &ly, &lz, &tx, &tz)) {
        tr2_tk_local_to_world(seg, lx, lz, 0.0f, ex, ez);
        *eh = seg->heading + seg->yaw_delta;
        return;
    }
    /* No authored spline: straight placeholder (unreachable for the shipped
     * table, every variation carries >= 2 control points). */
    *ex = seg->wx + sinf(seg->heading) * seg->length;
    *ez = seg->wz + cosf(seg->heading) * seg->length;
    *eh = seg->heading + seg->yaw_delta;
}

/* Point on a segment at distance d with lateral offset in metres. */
static void tr2_seg_point(const Tr2Segment* seg, float d, float lat_m,
        float* x, float* z, float* yaw) {
    d = tr2_tk_clampf(d, 0.0f, seg->length);
    float lx, ly, lz, tx, tz;
    if (tr2_tk_spline_at(seg, d, &lx, &ly, &lz, &tx, &tz)) {
        tr2_tk_local_to_world(seg, lx, lz, 0.0f, x, z);
        *yaw = seg->heading + atan2f(tx, -tz);
        *x -= cosf(*yaw) * lat_m;
        *z += sinf(*yaw) * lat_m;
        return;
    }
    *x = seg->wx + sinf(seg->heading) * d - cosf(seg->heading) * lat_m;
    *z = seg->wz + cosf(seg->heading) * d + sinf(seg->heading) * lat_m;
    float t = seg->length > 0.0f ? d / seg->length : 0.0f;
    *yaw = seg->heading + seg->yaw_delta * t;
}

/* Authored elevation (world Y) of the segment at distance d. */
static float tr2_seg_height(const Tr2Segment* seg, float d) {
    d = tr2_tk_clampf(d, 0.0f, seg->length);
    float lx, ly, lz, tx, tz;
    if (tr2_tk_spline_at(seg, d, &lx, &ly, &lz, &tx, &tz)) return seg->wy + ly;
    return seg->wy;
}

/* --------------------------------------------------------- piece choice --- */

/* MaxTrackPieceDifficulty ramp: the shipped milestones raise difficulty with
 * distance (TRACK.md 1.3 / 2.4). */
static int tr2_tk_difficulty_cap(float s) {
    if (s < 250.0f) return 0;
    if (s < 1000.0f) return 1;
    if (s < 2000.0f) return 2;
    return 3;
}

/* Base (non-turn, non-obstacle) pieces: straights, curves and, when enabled,
 * sub-mode entry pieces. */
static int tr2_tk_base_eligible(const Tr2PieceEntry* e, Env* env) {
    if (e->category == TR2_CAT_STRAIGHT || e->category == TR2_CAT_CURVE) {
        return 1;
    }
    if (e->category == TR2_CAT_SUBMODE) return env->enable_submodes ? 1 : 0;
    return 0;
}

/* Do not generate full-width pits before the current jump can reach their
 * far rim or authored FallSaver. Reserve one decision interval plus a physics
 * frame for timing quantization; speed ramps only make this bound safer.
 * This is a generator constraint, not a policy action mask. */
static unsigned tr2_tk_playable_variations(const Env* env, int pidx, int count) {
    if (tr2_piece_category(pidx) != TR2_CAT_OBSTACLE)
        return (1u << count) - 1u;
    unsigned mask = 0;
    int first = TR2_PIECE_TABLE[pidx].var_begin;
    for (int i = 0; i < count; ++i) {
        const Tr2Volume* vols = NULL;
        int n = tr2_volumes_for_variation(first + i, &vols);
        int playable = 1;
        if (tr2_piece_category(pidx) == TR2_CAT_OBSTACLE) {
            for (int j = 0; j < n; ++j) {
                const Tr2Volume* v = &vols[j];
                if (v->kind != TR2_VK_GAP) continue;
                float centre = v->lane_center * TR2_LANE_WIDTH_M;
                float half = v->half_width * TR2_LANE_WIDTH_M;
                if (fabsf(centre) + env->lane_width >= half + env->player_radius)
                    continue; /* A safe side lane exists. */
                float speed = fmaxf(env->speed, env->min_speed);
                float jump = fminf(speed, env->jump_velocity);
                float gravity = 2.0f * env->gravity;
                float margin = (float)(env->decision_frames + 1) / 60.0f;
                float flight = 2.0f * jump / gravity - 1.0f / 60.0f;
                int clear = speed * (flight - margin) >= v->length;
                if (!clear && v->save_length > 0.0f) {
                    float reach = env->player_height - v->save_y_bottom;
                    float rescue_time = (jump + sqrtf(jump * jump + 2.0f * gravity * reach))
                        / gravity - 1.0f / 60.0f;
                    float rescue_distance = 0.5f * v->length + v->save_offset
                        - 0.5f * v->save_length;
                    clear = speed * (rescue_time - margin) >= rescue_distance;
                }
                if (!clear) { playable = 0; break; }
            }
        }
        if (playable) mask |= 1u << i;
    }
    return mask;
}

/* Weighted pick from the Zone-0 pool. cat < 0 selects a base piece (straight /
 * curve / sub-mode); otherwise exactly the requested category. SelectionOdds
 * gate the pool (odds are per-piece weights, as in SelectRandomPiece); the
 * difficulty cap is relaxed on a second pass if nothing qualifies. */
static int tr2_tk_choose_piece(Env* env, int cat, float s) {
    Tr2PieceEntry cand[128];
    int nc = 0;
    int cap = tr2_tk_difficulty_cap(s);
    int zones = cat == TR2_CAT_SUBMODE ? TR2_PIECE_ZONE_COUNT : 1;
    for (int pass = 0; pass < 2; pass++) {
        int relax = pass == 1;
        nc = 0;
        for (int z = 0; z < zones; z++) {
            int count = 0;
            const Tr2PieceEntry* pool = tr2_piece_pool(z, &count);
            if (pool == NULL) continue;
            for (int i = 0; i < count; i++) {
                const Tr2PieceEntry* e = &pool[i];
                if (e->odds <= 0.0f) continue;
                if (cat < 0) {
                    if (!tr2_tk_base_eligible(e, env)) continue;
                } else if (e->category != cat) {
                    continue;
                }
                if (!relax && e->difficulty > cap) continue;
                if (!tr2_tk_playable_variations(env, e->piece_idx,
                        tr2_piece_variation_count(e->piece_idx))) continue;
                if (nc < (int)(sizeof(cand) / sizeof(cand[0]))) cand[nc++] = *e;
            }
        }
        if (nc == 0) continue;
        float total = 0.0f;
        for (int i = 0; i < nc; i++) total += cand[i].odds;
        float r = tr2_tk_rand(env) * total;
        float acc = 0.0f;
        for (int i = 0; i < nc; i++) {
            acc += cand[i].odds;
            if (r < acc) return cand[i].piece_idx;
        }
        return cand[nc - 1].piece_idx;
    }
    return -1;
}

/* WaterSlide run: the shipped entrance (pool piece 69) and exit (70) bracket a
 * randomised chain of flume pieces.  Middle pieces are drawn from the Zone-3
 * pool by their shipped SelectionOdds, so short slides and long rivers with
 * duck/blocked-lane sections both occur. */
#define TR2_PIECE_WATERSLIDE_ENTER 69
#define TR2_PIECE_WATERSLIDE_EXIT  70

static int tr2_tk_choose_water(Env* env, int exit_due) {
    if (exit_due) return TR2_PIECE_WATERSLIDE_EXIT;
    int count = 0;
    const Tr2PieceEntry* pool = tr2_piece_pool(3, &count);
    const Tr2PieceEntry* cand[32];
    int nc = 0;
    for (int i = 0; i < count && nc < 32; i++) {
        int p = pool[i].piece_idx;
        if (p == TR2_PIECE_WATERSLIDE_ENTER
                || p == TR2_PIECE_WATERSLIDE_EXIT) continue;
        if (pool[i].odds <= 0.0f) continue;
        if (!tr2_tk_playable_variations(env, p,
                tr2_piece_variation_count(p))) continue;
        cand[nc++] = &pool[i];
    }
    if (nc == 0) return -1;
    float total = 0.0f;
    for (int i = 0; i < nc; i++) total += cand[i]->odds;
    float r = tr2_tk_rand(env) * total;
    float acc = 0.0f;
    for (int i = 0; i < nc; i++) {
        acc += cand[i]->odds;
        if (r < acc) return cand[i]->piece_idx;
    }
    return cand[nc - 1]->piece_idx;
}

/* Shipped GetVariation weights frequencies and excludes the last choice for
 * that piece. Some extracted multi-variation records have no positive
 * frequencies; pick uniformly among the remaining variations in that case. */
static int tr2_tk_choose_variation(Env* env, int pidx, int count) {
    if (count <= 1) return 0;
    int previous = (int)env->last_variation[pidx] - 1;
    int first = TR2_PIECE_TABLE[pidx].var_begin;
    unsigned mask = tr2_tk_playable_variations(env, pidx, count);
    assert(mask != 0);
    unsigned alternatives = previous >= 0 ? mask & ~(1u << previous) : mask;
    if (alternatives) mask = alternatives;
    int total = 0, remaining = 0;
    for (int i = 0; i < count; ++i) {
        if (!(mask & (1u << i))) continue;
        total += TR2_VARIATION_TABLE[first + i].frequency;
        ++remaining;
    }
    if (total > 0) {
        int pick = tr2_tk_rand_int(env, total);
        for (int i = 0; i < count; ++i) {
            if (!(mask & (1u << i))) continue;
            pick -= TR2_VARIATION_TABLE[first + i].frequency;
            if (pick < 0) return i;
        }
    }
    int pick = tr2_tk_rand_int(env, remaining);
    for (int i = 0; i < count; ++i)
        if ((mask & (1u << i)) && pick-- == 0) return i;
    return 0;
}

/* Apply a chosen piece (and a random variation) to a segment. */
static void tr2_tk_apply_piece(Env* env, Tr2Segment* seg, int pidx) {
    int vcount = tr2_piece_variation_count(pidx);
    int vlocal = tr2_tk_choose_variation(env, pidx, vcount);
    env->last_variation[pidx] = (uint8_t)(vlocal + 1);
    float len = 0.0f;
    float yaw_deg = 0.0f;
    if (vcount > 0) {
        len = tr2_piece_variation_len(pidx, vlocal);
        yaw_deg = tr2_piece_variation_yaw_deg(pidx, vlocal);
    }
    /* The Base junction's two authored paths share the same mesh and are
     * mirror images. The extracted table stores path 0; recover path 1 here. */
    seg->alternate_path = pidx == TR2_PIECE_R100_Z0_JUNCTION
        && tr2_tk_rand_int(env, 2) == 1;
    if (seg->alternate_path) yaw_deg = -yaw_deg;
    int cat = tr2_piece_category(pidx);
    uint32_t flags = TR2_PIECE_TABLE[pidx].flags;
    if (!(len > TR2_TK_MIN_LEN)) {
        switch (cat) {
        case TR2_CAT_TURN: len = TR2_TK_LEN_TURN; break;
        case TR2_CAT_CURVE: len = TR2_TK_LEN_CURVE; break;
        case TR2_CAT_OBSTACLE: len = TR2_TK_LEN_OBSTACLE; break;
        case TR2_CAT_SUBMODE: len = TR2_TK_LEN_SUBMODE; break;
        default: len = TR2_TK_LEN_STRAIGHT; break;
        }
    }

    int is_turn_flag = (flags & (TR2_FLAG_IsTurnLeft | TR2_FLAG_IsTurnRight)) != 0;
    int is_junc = (flags & (TR2_FLAG_IsJunction | TR2_FLAG_IsYJunction)) != 0;
    int optional = (flags & TR2_FLAG_IsOptionalJunction) != 0;
    int mandatory = is_turn_flag || (is_junc && !optional);

    float nd = tr2_tk_norm_deg(yaw_deg);
    int turn = 0;
    if (cat == TR2_CAT_TURN || (mandatory && (is_turn_flag || is_junc))) {
        if (fabsf(nd) < 1.0f) {
            if (flags & TR2_FLAG_IsTurnLeft) nd = -90.0f;
            else if (flags & TR2_FLAG_IsTurnRight) nd = 90.0f;
            else nd = 90.0f;
        }
        turn = nd < 0.0f ? -1 : 1;
    }

    /* The chosen variation's runtime GeneratedPath drives traversal. */
    int var_idx = (int)TR2_PIECE_TABLE[pidx].var_begin + vlocal;
    uint16_t sb = 0, sc = 0;
    if (vcount > 0 && var_idx >= 0 && var_idx < TR2_TK_VARIATION_MAX) {
        sb = TR2_VARIATION_TABLE[var_idx].spline_begin;
        sc = TR2_VARIATION_TABLE[var_idx].spline_count;
    }

    seg->piece_idx = pidx;
    seg->var_idx = var_idx;
    seg->spline_begin = sb;
    seg->spline_count = sc;

    float slen = tr2_tk_spline_length(seg);
    if (slen > TR2_TK_MIN_LEN) len = slen;
    seg->length = len;

    /* PositionPiece uses SourceLastYRot; reflection reverses its sign. */
    seg->yaw_delta = -tr2_tk_norm_deg(yaw_deg) * TR2_TK_DEG;
    seg->turn = (signed char)turn;
    seg->mode = (unsigned char)tr2_piece_mode(pidx);
    seg->valid = 1;
    seg->entity_spawned = 0;
    seg->obstacle_kind = TR2_TK_ENTITY_NONE;
}

/* -------------------------------------------------------- generation --- */

static void tr2_track_reset(Env* env) {
    memset(env->segments, 0, sizeof(env->segments));
    memset(env->last_variation, 0, sizeof(env->last_variation));
    for (int i = 0; i < TR2_SEG_RING; i++) {
        env->segments[i].obstacle_kind = TR2_TK_ENTITY_NONE;
    }
    env->seg_idx = 0;
    env->seg_gen = -1;
    env->piece_count = 0;
    env->water_run_left = -1;
    env->turn_acc = 0.0f;
    env->obstacle_acc = 0.0f;

    const Tr2PacingRow* row = tr2_pacing_lookup(0.0f);
    env->next_turn_gap = row->turn_min
        + (row->turn_max - row->turn_min) * tr2_tk_rand(env);
    /* Intro runway: the first mandatory turn never lands inside the opening
     * straight stretch (~100 m), so several obstacle pieces precede it. */
    if (env->next_turn_gap < TR2_FIRST_STRAIGHT_LENGTH) {
        env->next_turn_gap = TR2_FIRST_STRAIGHT_LENGTH;
    }
    env->next_obstacle_gap = row->obstacle_min
        + (row->obstacle_max - row->obstacle_min) * tr2_tk_rand(env);

    env->next_coin_s = TR2_TK_COIN_START;
    float ps = tr2_powerup_spacing(tr2_tk_rand_int(env, 1000));
    env->next_powerup_s = ps < 250.0f ? 250.0f : ps;

    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) env->obstacles[i].active = 0;
    for (int i = 0; i < TR2_MAX_COINS; i++) env->coin_pool[i].active = 0;
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) env->powerups[i].active = 0;

    tr2_track_generate(env);
}

/* Append pieces until the segment ring holds TR2_SEG_AHEAD beyond the player.
 * Category is chosen from the shipped pacing accumulators: mandatory turn due,
 * obstacle due, otherwise a base piece (straight / curve / sub-mode). */
static void tr2_track_generate(Env* env) {
    while (env->seg_gen < env->seg_idx + TR2_SEG_AHEAD) {
        int abs_idx = env->seg_gen + 1;
        Tr2Segment* seg = &env->segments[abs_idx % TR2_SEG_RING];
        Tr2Segment* prev = abs_idx == 0 ? NULL
            : &env->segments[(abs_idx - 1) % TR2_SEG_RING];
        memset(seg, 0, sizeof(*seg));
        seg->obstacle_kind = TR2_TK_ENTITY_NONE;

        seg->s0 = prev == NULL ? 0.0f : prev->s0 + prev->length;
        if (prev != NULL) {
            tr2_seg_end(prev, &seg->wx, &seg->wz, &seg->heading);
            seg->wy = tr2_seg_height(prev, prev->length);   /* join keeps Y */
        } else {
            seg->wx = 0.0f;
            seg->wz = 0.0f;
            seg->heading = 0.0f;
            seg->wy = 0.0f;
        }

        const Tr2PacingRow* row = tr2_pacing_lookup(seg->s0);
        int cat = -1;   /* base */
        if (abs_idx == 0) {
            cat = TR2_CAT_STRAIGHT;   /* start on a straight */
        } else if (env->turn_acc >= env->next_turn_gap) {
            cat = TR2_CAT_TURN;
        } else if (env->obstacle_acc >= env->next_obstacle_gap) {
            cat = TR2_CAT_OBSTACLE;
        }

        if (cat < 0 && tr2_tk_rand(env) < 0.35f) {
            cat = TR2_CAT_OBSTACLE;   /* keep obstacle density near the shipped 10-20 m */
        } else if (cat < 0 && env->enable_submodes
                && tr2_tk_rand(env) < 0.05f) {
            cat = TR2_CAT_SUBMODE;
        }
        int pidx = -1;
        if (env->water_run_left >= 0) {
            int exit_due = env->water_run_left == 0;
            pidx = tr2_tk_choose_water(env, exit_due);
            if (pidx >= 0)
                env->water_run_left = exit_due ? -1 : env->water_run_left - 1;
        }
        if (pidx < 0) pidx = tr2_tk_choose_piece(env, cat, seg->s0);
        if (pidx < 0 && cat != TR2_CAT_STRAIGHT) {
            pidx = tr2_tk_choose_piece(env, TR2_CAT_STRAIGHT, seg->s0);
        }
        if (pidx < 0) {
            /* No piece chosen: emit a plain straight only so the ring keeps
             * advancing.  This is NOT authored geometry (no shipped spline);
             * tr2_tk_choose_piece cannot reach it while TR2_PIECE_TABLE has a
             * valid straight, so it is a safety net, not a parity path. */
            seg->piece_idx = -1;
            seg->var_idx = 0;
            seg->spline_begin = 0;
            seg->spline_count = 0;
            seg->length = TR2_TK_LEN_STRAIGHT;
            seg->yaw_delta = 0.0f;
            seg->turn = 0;
            seg->mode = TR2_MODE_RUNNING;
            seg->valid = 1;
            seg->entity_spawned = 0;
        } else {
            tr2_tk_apply_piece(env, seg, pidx);
            if (env->enable_submodes
                    && seg->piece_idx == TR2_PIECE_WATERSLIDE_ENTER
                    && env->water_run_left < 0) {
                /* 2..5 middle flume pieces, then the authored exit piece. */
                env->water_run_left = 2 + tr2_tk_rand_int(env, 4);
            }
        }

        env->turn_acc += seg->length;
        env->obstacle_acc += seg->length;
        if (seg->turn != 0) {
            env->turn_acc = 0.0f;
            env->next_turn_gap = row->turn_min
                + (row->turn_max - row->turn_min) * tr2_tk_rand(env);
            if (!row->allow_turn_after_obstacle
                    && env->obstacle_acc > -TR2_TK_AFTER_TURN_HOLD) {
                env->obstacle_acc = -TR2_TK_AFTER_TURN_HOLD;
            }
        }
        if (seg->piece_idx >= 0
                && tr2_piece_category(seg->piece_idx) == TR2_CAT_OBSTACLE) {
            env->obstacle_acc = 0.0f;
            env->next_obstacle_gap = row->obstacle_min
                + (row->obstacle_max - row->obstacle_min) * tr2_tk_rand(env);
        }
        env->piece_count++;
        env->seg_gen = abs_idx;
    }
}

/* ------------------------------------------------------------- queries --- */

/* Convert path distance + lane offset (lane units) to world coordinates. */
static void tr2_world_at(Env* env, float s, float lane_units, float y,
        float* wx, float* wy, float* wz, float* yaw) {
    if (env->seg_gen < 0) {
        *wx = 0.0f;
        *wy = y;
        *wz = 0.0f;
        *yaw = 0.0f;
        return;
    }
    Tr2Segment* seg = NULL;
    for (int i = env->seg_idx; i <= env->seg_gen; i++) {
        Tr2Segment* cand = &env->segments[i % TR2_SEG_RING];
        if (!cand->valid) continue;
        seg = cand;
        if (s <= cand->s0 + cand->length) break;
    }
    if (seg == NULL) seg = &env->segments[env->seg_idx % TR2_SEG_RING];
    float d = s - seg->s0;
    if (d < 0.0f) d = 0.0f;
    tr2_seg_point(seg, d, lane_units * env->lane_width, wx, wz, yaw);
    *wy = tr2_seg_height(seg, d) + y;
}

/* True when s lies inside a mandatory turn arc or within its approach/exit
 * margins. */
static int tr2_near_turn(Env* env, float s) {
    for (int i = env->seg_idx; i <= env->seg_gen; i++) {
        Tr2Segment* seg = &env->segments[i % TR2_SEG_RING];
        if (!seg->valid || seg->turn == 0) continue;
        if (s > seg->s0 - TR2_TK_TURN_MARGIN
                && s < seg->s0 + seg->length + TR2_TK_TURN_MARGIN) {
            return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------ spawn --- */

static Tr2Obstacle* tr2_tk_alloc_obstacle(Env* env) {
    Tr2Obstacle* oldest = &env->obstacles[0];
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) {
        Tr2Obstacle* o = &env->obstacles[i];
        if (!o->active) return o;
        if (o->s < oldest->s) oldest = o;
    }
    return oldest;
}

static Tr2Coin* tr2_tk_alloc_coin(Env* env) {
    for (int i = 0; i < TR2_MAX_COINS; i++) {
        if (!env->coin_pool[i].active) return &env->coin_pool[i];
    }
    Tr2Coin* oldest = &env->coin_pool[0];
    for (int i = 1; i < TR2_MAX_COINS; i++) {
        if (env->coin_pool[i].s < oldest->s) oldest = &env->coin_pool[i];
    }
    return oldest;
}

static Tr2Powerup* tr2_tk_alloc_powerup(Env* env) {
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) {
        if (!env->powerups[i].active) return &env->powerups[i];
    }
    return &env->powerups[0];
}

static void tr2_tk_add_coin(Env* env, float s, float lane, float y) {
    if (tr2_near_turn(env, s)) return;
    Tr2Coin* c = tr2_tk_alloc_coin(env);
    c->s = s;
    c->lane = lane;
    c->y = y;
    c->value = (unsigned char)tr2_coin_value(s);
    c->active = 1;
}

/* A run of coins down a single safe lane inside one piece, at most
 * MaxCoinsPerRun (pacing->max_coins). */
static void tr2_tk_coin_run(Env* env, float s0, float len, float lane) {
    const Tr2PacingRow* row = tr2_pacing_lookup(s0);
    int maxn = row->max_coins > 0 ? row->max_coins : 25;
    int n = (int)(len / TR2_TK_COIN_SPACING);
    if (n < TR2_TK_COIN_RUN_MIN) n = TR2_TK_COIN_RUN_MIN;
    if (n > maxn) n = maxn;
    float start = s0 + 1.0f;
    for (int i = 0; i < n; i++) {
        float cs = start + (float)i * TR2_TK_COIN_SPACING;
        if (cs > s0 + len) break;
        tr2_tk_add_coin(env, cs, lane, TR2_TK_COIN_Y);
    }
}

/* Stamp one unspawned piece.  `abs_idx` is the segment's absolute ring index
 * (the caller's loop counter); every obstacle is tagged with abs_idx + 1 in
 * owner_segment_plus_one so the renderer can bind it to the owning variation.
 *
 * The volume set is the selected variation's own prefab (`seg->var_idx`), so no
 * other variation of the same piece can bleed its obstacles in.  The generator
 * (tools/gen_colliders_table.py) already drops non-gameplay triggers, no-op
 * ZipLine/FallSaver rows and mesh colliders, so every row here is a live
 * gameplay volume. */
static void tr2_tk_spawn_piece(Env* env, Tr2Segment* seg, int with_coins,
        int abs_idx) {
    const Tr2Volume* vols = NULL;
    int nv = tr2_volumes_for_variation(seg->var_idx, &vols);
    int blocked[TR2_LANES] = {0, 0, 0};
    int spawned = 0;
    /* Collider table values were normalized with TR2_LANE_WIDTH_M; preserve
     * their authored metre dimensions when the runner lane pitch differs. */
    float collider_scale = TR2_LANE_WIDTH_M / env->lane_width;

    for (int v = 0; v < nv && vols != NULL; v++) {
        const Tr2Volume* vol = &vols[v];
        int kind = vol->kind;
        if (kind == TR2_VK_MINE_DUCK) kind = TR2_OBST_HIGH;
        else if (kind == TR2_VK_WATER) kind = TR2_OBST_WATER;
        else if (kind == TR2_VK_WATER_BLOCK) kind = TR2_OBST_HIGH;
        else if (kind == TR2_VK_MINE_WALL) kind = TR2_OBST_BLOCK;
        /* A few authored rows sit behind the prefab origin
         * (mine_duck_b, forest_junction_b): clamp those into the piece
         * frame.  Positive offsets are left in place: turn/junction GAP and
         * MissedTurn walls are authored *past* the piece's estimated path
         * length on purpose, and belong past the turn, not on it. */
        float os = seg->s0 + vol->forward_offset;
        if (os < seg->s0) os = seg->s0;
        if (!spawned) {
            seg->obstacle_kind = (unsigned char)kind;
            spawned = 1;
        }
        float lane_center = vol->lane_center * collider_scale;
        float lane_half_width = vol->half_width * collider_scale;
        for (int l = 0; l < TR2_LANES; l++) {
            float lane_units = (float)l - 1.0f;
            if (fabsf(lane_units - lane_center) <= lane_half_width + 0.25f) {
                blocked[l] = 1;
            }
        }
        Tr2Obstacle* o = tr2_tk_alloc_obstacle(env);
        o->s = os;
        o->length = vol->length;
        o->lane = lane_center;
        o->half_width = lane_half_width;
        o->y_bottom = vol->y_bottom;
        o->y_top = vol->y_top;
        o->phase = kind == TR2_OBST_FIRE
            ? tr2_tk_rand(env) * TR2_TK_TWO_PI : 0.0f;
        o->kind = (unsigned char)kind;
        o->variant = (unsigned char)tr2_tk_rand_int(env, 4);
        o->trigger = vol->trigger;
        o->death_type = vol->death_type;
        o->save_offset = vol->save_offset;
        o->save_length = vol->save_length;
        o->save_half_width = vol->save_half_width * collider_scale;
        o->save_y_bottom = vol->save_y_bottom;
        o->save_y_top = vol->save_y_top;
        o->saved = 0;
        o->owner_segment_plus_one = abs_idx + 1;
        o->active = 1;
        o->passed = 0;
        o->trigger_entered = 0;
    }

    if (!with_coins) return;
    if (seg->s0 < env->next_coin_s) return;
    int free_n = 0;
    for (int l = 0; l < TR2_LANES; l++) {
        if (!blocked[l]) free_n++;
    }
    if (free_n > 0) {
        int pick = tr2_tk_rand_int(env, free_n);
        int lane = 0;
        for (int l = 0; l < TR2_LANES; l++) {
            if (blocked[l]) continue;
            if (pick == 0) { lane = l; break; }
            pick--;
        }
        tr2_tk_coin_run(env, seg->s0, seg->length, (float)lane - 1.0f);
    }
    const Tr2PacingRow* row = tr2_pacing_lookup(seg->s0);
    env->next_coin_s = seg->s0 + row->coin_gap * (0.8f + 0.4f * tr2_tk_rand(env));
}

/* Grow the track to stay ahead of the runner and stamp every newly generated
 * piece exactly once (gated by Tr2Segment.entity_spawned). */
static void tr2_spawn_ahead(Env* env) {
    tr2_track_generate(env);

    for (int i = env->seg_idx; i <= env->seg_gen; i++) {
        Tr2Segment* seg = &env->segments[i % TR2_SEG_RING];
        if (!seg->valid || seg->entity_spawned) continue;
        seg->entity_spawned = 1;
        if (seg->piece_idx < 0) continue;

        if (seg->turn == 0) {
            tr2_tk_spawn_piece(env, seg, seg->mode == TR2_MODE_RUNNING, i);
        }

        if (env->next_powerup_s < seg->s0 + seg->length) {
            float ps = env->next_powerup_s < seg->s0 ? seg->s0
                                                     : env->next_powerup_s;
            if (!tr2_near_turn(env, ps)) {
                Tr2Powerup* p = tr2_tk_alloc_powerup(env);
                p->s = ps;
                p->lane = (float)tr2_tk_rand_int(env, TR2_LANES) - 1.0f;
                p->kind = (unsigned char)tr2_powerup_kind(tr2_tk_rand_int(env, 1000));
                p->active = 1;
            }
            env->next_powerup_s = seg->s0 + seg->length
                + tr2_powerup_spacing(tr2_tk_rand_int(env, 1000));
        }
    }
}
