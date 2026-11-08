/* templerun2_observe.h — structured, egocentric observation (OBS_SIZE floats).
 *
 * Owner: observe module. Consumes the frozen Env layout from templerun2_types.h
 * and writes exactly OBS_SIZE = 168 normalised floats into
 * env->agents[0].observations. Every feature is egocentric (relative to the
 * runner along the path, px as the lateral origin) so a policy can be replayed
 * visually without a second state encoding.
 *
 * Final layout (238 floats == OBS_SIZE, asserted at the end of tr2_observe):
 *
 *   [  0.. 26] base (TR2_OBS_BASE = 27)
 *       0 px                      lateral player offset, -1..1
 *       1 y                       height / 3, clipped
 *       2 vy                      vertical velocity / 12, clipped
 *       3 grounded                0/1
 *       4 sliding                 0/1
 *       5 slide_timer             / slide_duration_cap, 0..1
 *       6 speed                   / max_speed, 0..1
 *       7 s                       / 5000, 0..1
 *       8 score                   / 5000, 0..1
 *       9 coins                   / 200, 0..1
 *      10 multiplier              / multiplier_max, 0..1
 *      11 magnet_timer            / magnet_time, 0..1
 *      12 shield_timer            / shield_time, 0..1
 *      13 boost_distance_left     / boost_distance, 0..1
 *      14 gems                    / 20, 0..1
 *      15 invuln_timer            / invuln_time, 0..1
 *      16 mode                    / TR2_MODE_DOWNHILL, 0..1
 *      17 stumble_timer           seconds remaining, clipped to 1
 *      18 turn_input              -1 / 0 / +1
 *      19 dead                    0/1
 *      20 piece type (piece_idx)  / 64, 0..1
 *      21 yaw (piece heading)     / pi, clipped -1..1
 *      22 end distance (seg_rem)  / 50, 0..1
 *      23 turn dir (current piece) -1 / 0 / +1
 *      24 mode_timer              / 4, 0..1
 *      25 current piece flags     bitfield / 32, 0..1
 *      26 turn_input_timer        / turn_window, 0..1
 *
 *   [ 27.. 29] turn (TR2_OBS_TURN = 3)
 *      27 turn present             0/1
 *      28 turn dir                 -1 / 0 / +1
 *      29 turn relative distance   / turn_window, 0..1
 *
 *   [ 30.. 53] TR2_LOOK_PIECES (6) * TR2_OBS_PIECE (4)
 *      30+4k rel   (ps->s0 - s) / 100, 0..1
 *      31+4k yaw   ps->yaw_delta / pi, clipped -1..1
 *      32+4k type  piece_idx / 64, 0..1
 *      33+4k flags piece flags bitfield / 32, 0..1
 *
 *   [ 54..203] TR2_LOOK_OBST (10) * TR2_OBS_OBST (15)
 *      54+15k rel    (o->s - s) / 50, 0..1
 *      55+15k lane   (o->lane - px) / 2, clipped -1..1
 *      56+15k..64+15k 9-kind onehot including water and left/right ledges
 *      65+15k half-width (metres) / 5
 *      66+15k length (metres) / 20
 *      67+15k bottom (metres) / 5, clipped -1..1
 *      68+15k top (metres) / 5, clipped -1..1
 *
 *   [204..227] TR2_LOOK_COINS (6) * TR2_OBS_COIN (4)
 *      204+4k rel   (c->s - s) / 30, 0..1
 *      205+4k lane  (c->lane - px) / 2, clipped -1..1
 *      206+4k y     c->y / 3, 0..1
 *      207+4k value c->value / 8, 0..1
 *
 *   [228..237] TR2_LOOK_POWERUPS (2) * TR2_OBS_POWERUP (5)
 *      228+5k rel   (p->s - s) / 50, 0..1
 *      229+5k lane  (p->lane - px) / 2, clipped -1..1
 *      230+5k magnet onehot (kind == TR2_PU_MAGNET)
 *      231+5k boost  onehot (kind == TR2_PU_BOOST)
 *      232+5k shield onehot (kind == TR2_PU_SHIELD)
 *                     (TR2_PU_GEM / TR2_PU_COIN_BONUS / TR2_PU_SCORE_BONUS are
 *                     instant pickups and yield an all-zero onehot; they are
 *                     still localised by the rel/lane slots.)
 *
 * Unfilled entity slots are written as zeros so the layout stays positional.
 */
#pragma once

#include <assert.h>
#include <stddef.h>

/* Saturating clip to [lo, hi]. */
static float tr2_obs_saturate(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* Clip to [-1, 1]; matches the egocentric convention used across the env. */
static float tr2_obs_clip(float v) {
    return tr2_obs_saturate(v, -1.0f, 1.0f);
}

/* v / span normalised to [0, 1] (span <= 0 -> 0). */
static float tr2_obs_norm(float v, float span) {
    if (span <= 0.0f) return 0.0f;
    return tr2_obs_saturate(v / span, 0.0f, 1.0f);
}

/* Per-piece bitfield: valid | entity_spawned | sub-mode | has obstacle. */
static float tr2_obs_piece_flags(const Tr2Segment* s) {
    int f = 0;
    if (s->valid) f |= 1;
    if (s->entity_spawned) f |= 2;
    if (s->mode != TR2_MODE_RUNNING) f |= 4;
    if (s->obstacle_kind != 255) f |= 8;
    return (float)f;
}

static void tr2_observe(Env* env) {
    obs_t* obs = env->agents[0].observations;
    if (obs == NULL) return;
    int idx = 0;

    Tr2Segment* seg = &env->segments[env->seg_idx % TR2_SEG_RING];
    Tr2Segment* next = &env->segments[(env->seg_idx + 1) % TR2_SEG_RING];
    float seg_rem = seg->s0 + seg->length - env->s;
    float turn_span = env->turn_window > 0.0f ? env->turn_window : 1.0f;

    /* --- base: 27 -------------------------------------------------------- */
    obs[idx++] = tr2_obs_clip(env->px);
    obs[idx++] = tr2_obs_clip(env->y / 3.0f);
    obs[idx++] = tr2_obs_clip(env->vy / 12.0f);
    obs[idx++] = (float)(env->grounded != 0);
    obs[idx++] = (float)(env->sliding != 0);
    obs[idx++] = tr2_obs_norm(env->slide_timer, env->slide_duration_cap);
    obs[idx++] = tr2_obs_norm(env->speed, env->max_speed);
    obs[idx++] = tr2_obs_norm(env->s, 5000.0f);
    obs[idx++] = tr2_obs_norm(env->score, 5000.0f);
    obs[idx++] = tr2_obs_norm((float)env->coins, 200.0f);
    obs[idx++] = env->multiplier_max > 0
        ? tr2_obs_norm((float)env->multiplier, (float)env->multiplier_max)
        : 0.0f;
    obs[idx++] = tr2_obs_norm(env->magnet_timer, env->magnet_time);
    obs[idx++] = tr2_obs_norm(env->shield_timer, env->shield_time);
    obs[idx++] = tr2_obs_norm(env->boost_distance_left, env->boost_distance);
    obs[idx++] = tr2_obs_norm((float)env->gems, 20.0f);
    obs[idx++] = tr2_obs_norm(env->invuln_timer, env->invuln_time);
    obs[idx++] = tr2_obs_norm((float)env->mode, (float)TR2_MODE_DOWNHILL);
    obs[idx++] = tr2_obs_clip(env->stumble_timer);
    obs[idx++] = (float)env->turn_input;
    obs[idx++] = (float)(env->dead != 0);
    obs[idx++] = tr2_obs_norm((float)seg->piece_idx, 64.0f);
    obs[idx++] = tr2_obs_clip(seg->heading / 3.14159265358979323846f);
    obs[idx++] = tr2_obs_norm(seg_rem, 50.0f);
    obs[idx++] = (float)seg->turn;
    obs[idx++] = tr2_obs_norm(env->mode_timer, 4.0f);
    obs[idx++] = tr2_obs_piece_flags(seg) / 32.0f;
    obs[idx++] = tr2_obs_norm(env->turn_input_timer, turn_span);

    /* --- turn: 3 --------------------------------------------------------- */
    int turn = seg->turn != 0 ? seg->turn : next->turn;
    float turn_s = seg->turn != 0 ? seg->s0 : next->s0;
    obs[idx++] = turn != 0 ? 1.0f : 0.0f;
    obs[idx++] = (float)turn;
    obs[idx++] = tr2_obs_norm(turn_s - env->s, turn_span);

    /* --- track pieces: 6 * 4 --------------------------------------------- */
    for (int k = 0; k < TR2_LOOK_PIECES; k++) {
        Tr2Segment* ps = &env->segments[(env->seg_idx + k) % TR2_SEG_RING];
        obs[idx++] = tr2_obs_norm(ps->s0 - env->s, 100.0f);
        obs[idx++] = tr2_obs_clip(ps->yaw_delta / 3.14159265358979323846f);
        obs[idx++] = tr2_obs_norm((float)ps->piece_idx, 64.0f);
        obs[idx++] = tr2_obs_piece_flags(ps) / 32.0f;
    }

    /* --- obstacles: nearest 10, 15 floats each --------------------------- */
    int obst_idx[TR2_LOOK_OBST];
    int n_obst = 0;
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) {
        Tr2Obstacle* o = &env->obstacles[i];
        if (!o->active || o->passed) continue;
        float rel = o->s - env->s;
        if (rel < -1.0f || rel > 90.0f) continue;
        int pos = n_obst;
        while (pos > 0 && env->obstacles[obst_idx[pos - 1]].s > o->s) pos--;
        if (pos >= TR2_LOOK_OBST) continue;
        if (n_obst < TR2_LOOK_OBST) n_obst++;
        for (int j = n_obst - 1; j > pos; j--) obst_idx[j] = obst_idx[j - 1];
        obst_idx[pos] = i;
    }
    for (int k = 0; k < TR2_LOOK_OBST; k++) {
        if (k < n_obst) {
            Tr2Obstacle* o = &env->obstacles[obst_idx[k]];
            obs[idx++] = tr2_obs_norm(o->s - env->s, 50.0f);
            obs[idx++] = tr2_obs_clip((o->lane - env->px) / 2.0f);
            for (int j = 0; j < TR2_OBST_KINDS; j++)
                obs[idx++] = (o->kind == (unsigned char)j) ? 1.0f : 0.0f;
            obs[idx++] = tr2_obs_norm(o->half_width * env->lane_width, 5.0f);
            obs[idx++] = tr2_obs_norm(o->length, 20.0f);
            obs[idx++] = tr2_obs_clip(o->y_bottom / 5.0f);
            obs[idx++] = tr2_obs_clip(o->y_top / 5.0f);
        } else {
            for (int f = 0; f < TR2_OBS_OBST; f++) obs[idx++] = 0.0f;
        }
    }

    /* --- coins: nearest 6, 4 floats each --------------------------------- */
    int coin_idx[TR2_LOOK_COINS];
    int n_coin = 0;
    for (int i = 0; i < TR2_MAX_COINS; i++) {
        Tr2Coin* c = &env->coin_pool[i];
        if (!c->active) continue;
        float rel = c->s - env->s;
        if (rel < -1.0f || rel > 45.0f) continue;
        int pos = n_coin;
        while (pos > 0 && env->coin_pool[coin_idx[pos - 1]].s > c->s) pos--;
        if (pos >= TR2_LOOK_COINS) continue;
        if (n_coin < TR2_LOOK_COINS) n_coin++;
        for (int j = n_coin - 1; j > pos; j--) coin_idx[j] = coin_idx[j - 1];
        coin_idx[pos] = i;
    }
    for (int k = 0; k < TR2_LOOK_COINS; k++) {
        if (k < n_coin) {
            Tr2Coin* c = &env->coin_pool[coin_idx[k]];
            obs[idx++] = tr2_obs_norm(c->s - env->s, 30.0f);
            obs[idx++] = tr2_obs_clip((c->lane - env->px) / 2.0f);
            obs[idx++] = tr2_obs_norm(c->y, 3.0f);
            obs[idx++] = tr2_obs_norm((float)c->value, 8.0f);
        } else {
            for (int f = 0; f < TR2_OBS_COIN; f++) obs[idx++] = 0.0f;
        }
    }

    /* --- powerups: nearest 2, 5 floats each ------------------------------ */
    int pu_idx[TR2_LOOK_POWERUPS];
    int n_pu = 0;
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) {
        Tr2Powerup* p = &env->powerups[i];
        if (!p->active) continue;
        float rel = p->s - env->s;
        if (rel < -1.0f || rel > 60.0f) continue;
        int pos = n_pu;
        while (pos > 0 && env->powerups[pu_idx[pos - 1]].s > p->s) pos--;
        if (pos >= TR2_LOOK_POWERUPS) continue;
        if (n_pu < TR2_LOOK_POWERUPS) n_pu++;
        for (int j = n_pu - 1; j > pos; j--) pu_idx[j] = pu_idx[j - 1];
        pu_idx[pos] = i;
    }
    for (int k = 0; k < TR2_LOOK_POWERUPS; k++) {
        if (k < n_pu) {
            Tr2Powerup* p = &env->powerups[pu_idx[k]];
            obs[idx++] = tr2_obs_norm(p->s - env->s, 50.0f);
            obs[idx++] = tr2_obs_clip((p->lane - env->px) / 2.0f);
            obs[idx++] = (p->kind == TR2_PU_MAGNET) ? 1.0f : 0.0f;
            obs[idx++] = (p->kind == TR2_PU_BOOST) ? 1.0f : 0.0f;
            obs[idx++] = (p->kind == TR2_PU_SHIELD) ? 1.0f : 0.0f;
        } else {
            for (int f = 0; f < TR2_OBS_POWERUP; f++) obs[idx++] = 0.0f;
        }
    }

    assert(idx == OBS_SIZE);
}
