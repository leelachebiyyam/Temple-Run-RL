/* Deterministic mechanics tests for ocean/templerun2.
 *
 * The env is header-only; this translation unit is the build. Every module
 * header (templerun2_track.h, templerun2_runner.h, templerun2_collision.h,
 * templerun2_items.h, templerun2_observe.h, templerun2_render.h) and the data
 * headers must exist first. raylib is linked because templerun2_render.h
 * includes raylib.h.
 *
 * Build from the repo root (config/templerun2.ini is read relative to
 * the cwd):
 *   clang -O2 -I. -Isrc -I./raylib-5.5_macos/include \
 *     -I./ocean/templerun2 -DPLATFORM_DESKTOP \
 *     tests/test_templerun2_env.c \
 *     ./raylib-5.5_macos/lib/libraylib.a -framework Cocoa -framework IOKit \
 *     -framework CoreVideo -framework OpenGL -lm -lpthread \
 *     -o /tmp/test_templerun2
 *   /tmp/test_templerun2
 *
 * The checks only touch the frozen contract: the Env / Tr2* struct fields, the
 * TR2_OBST_* / TR2_ACT_* / TR2_DEATH_* enums, OBS_SIZE and the puf_* entry
 * points.  They assert documented qualitative behaviour, never an internal
 * formula or a private helper.
 */
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "ocean/templerun2/templerun2.h"


typedef struct {
    Env env;
    obs_t obs[OBS_SIZE];
    float action, reward, terminal;
} Harness;

/* ------------------------------------------------------------------ harness */

/* tr2_config() reads every tuning key through dict_get(), which exits on a
 * miss, so the harness must present a complete dict. Load the shipped config
 * for realistic values, then backfill anything the ini does not define so the
 * test also runs from a different cwd. */
static void fill_missing(Dict* d) {
    static const struct { const char* k; double v; } defs[] = {
        {"lane_width", 0.85}, {"track_half_width", 1.12},
        {"player_radius", 0.44}, {"player_height", 1.04},
        {"slide_height", 0.64}, {"min_speed", 10.0}, {"max_speed", 21.0},
        {"force_scale", 0.1}, {"jump_velocity", 10.0}, {"gravity", 25.0},
        {"slam_velocity", 20.0}, {"slide_duration_factor", 0.07},
        {"slide_duration_cap", 0.75}, {"stumble_speed", 0.85},
        {"lane_lerp", 0.1}, {"turn_window", 7.0}, {"magnet_time", 10.0},
        {"shield_time", 10.0}, {"boost_distance", 250.0},
        {"boost_speed", 1.55}, {"invuln_time", 4.0}, {"magnet_radius", 34.64},
        {"coin_red_m", 2500.0}, {"coin_blue_m", 5000.0},
        {"score_distance", 1.0}, {"score_coin", 1.0}, {"multiplier_m", 0.0},
        {"multiplier_max", 1},
        {"reward_clear", 0.05}, {"death_penalty", 8.0},
        {"reward_coin", 0.15},
        {"swipe_cost", 0.0},
        {"episode_length", 200000}, {"enable_submodes", 1},
    };
    for (size_t i = 0; i < sizeof(defs) / sizeof(defs[0]); i++) {
        if (!dict_find(d, defs[i].k)) dict_set(d, defs[i].k, defs[i].v);
    }
}

static void harness_init(Harness* h, unsigned int seed) {
    Ini ini = {0};
    puf_ini_load_env(&ini, "templerun2", 0, NULL);
    Dict* kwargs = puf_ini_section(&ini, "env", 1);
    fill_missing(kwargs);
    /* Mechanics tests advance one 60 Hz frame; decision batching is tested
     * separately below. */
    dict_set(kwargs, "decision_frames", 1);

    memset(h, 0, sizeof(*h));
    h->env.rng = seed;
    puf_init(&h->env, kwargs);
    h->env.agents[0].observations = h->obs;
    h->env.agents[0].actions = &h->action;
    h->env.agents[0].rewards = &h->reward;
    h->env.agents[0].terminals = &h->terminal;
    puf_reset(&h->env);
    puf_ini_free(&ini);
}

/* Same env, but with the anti-spam shaping pinned to an exact value so the
 * reward arithmetic below does not depend on the shipped config value. */
static void harness_init_costs(Harness* h, unsigned int seed, float cost) {
    harness_init(h, seed);
    h->env.swipe_cost = cost;
}

/* Deactivate every pooled entity.  The track stamps entity volumes for every
 * generated piece, so mechanics tests scrub them and keep only what they place
 * themselves. */
static void clear_all(Harness* h) {
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) h->env.obstacles[i].active = 0;
    for (int i = 0; i < TR2_MAX_COINS; i++) h->env.coin_pool[i].active = 0;
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) h->env.powerups[i].active = 0;
    h->env.next_coin_s = 1e9f;
    h->env.next_powerup_s = 1e9f;
}

/* Deactivate every pooled entity and suppress mandatory turns and sub-mode
 * pieces so a test can leave a clear running path alone: hazard generation is
 * scrubbed, not the runner's ability to react to it. */
static void clear_track_hazards(Harness* h) {
    clear_all(h);
    h->env.next_turn_gap = 1e9f;
    h->env.turn_acc = 0.0f;
    h->env.enable_submodes = 0;
    for (int i = 0; i < TR2_SEG_RING; i++) h->env.segments[i].turn = 0;
}

/* Deactivate every pooled entity except the named slot (pass -1 to drop all of
 * a pool).  The kept slot is left untouched so a picked-up entity stays
 * consumed. */
static void suppress_except(Harness* h, int obst, int coin, int pu) {
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) {
        if (i != obst) h->env.obstacles[i].active = 0;
    }
    for (int i = 0; i < TR2_MAX_COINS; i++) {
        if (i != coin) h->env.coin_pool[i].active = 0;
    }
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) {
        if (i != pu) h->env.powerups[i].active = 0;
    }
}

/* Place one obstacle in the centre lane `rel` metres ahead of the runner.
 * The vertical span is the documented clearance for each kind: LOW clears on a
 * jump, HIGH on a slide, GAP on a jump (a fall otherwise). */
static void put_obstacle(Env* env, int kind, float rel) {
    Tr2Obstacle* o = &env->obstacles[0];
    memset(o, 0, sizeof(*o));
    o->s = env->s + rel;
    o->lane = 0.0f;
    o->half_width = 0.5f;
    o->length = 1.0f;
    o->kind = (unsigned char)kind;
    o->trigger = 4; /* TrackTrigger.Obstacle */
    o->death_type = TR2_DEATH_SCENERYROCK;
    if (kind == TR2_OBST_HIGH) {
        o->y_bottom = 0.9f;
        o->y_top = 2.6f;
        o->length = 1.2f;
    } else if (kind == TR2_OBST_GAP) {
        o->y_bottom = -2.0f;
        o->y_top = 0.47f;
        o->length = 3.0f;
        o->death_type = TR2_DEATH_FALL;
    } else { /* LOW */
        o->y_bottom = 0.0f;
        o->y_top = 0.9f;
    }
    o->active = 1;
}

/* Swipe for the next mandatory turn when it is inside the turn window. */
static void follow_turn_action(Harness* h) {
    const Tr2Segment* next =
        &h->env.segments[(h->env.seg_idx + 1) % TR2_SEG_RING];
    h->action = -1.0f;
    if (next->turn != 0 && next->s0 - h->env.s < h->env.turn_window - 1.0f) {
        h->action = next->turn < 0 ? (float)TR2_ACT_LEFT : (float)TR2_ACT_RIGHT;
    }
}

/* Run until the runner's own `rel` to the tracked obstacle, or `max_steps`.
 * Every other pooled entity is dropped each step so only slot 0 is live. */
static void step_until(Harness* h, float rel, int max_steps) {
    for (int i = 0; i < max_steps && h->env.obstacles[0].s - h->env.s > rel;
            i++) {
        suppress_except(h, 0, -1, -1);
        h->action = -1.0f;
        puf_step(&h->env);
    }
}

/* ------------------------------------------------ deterministic state diff */

static int fbits(float x, float y) { return memcmp(&x, &y, sizeof(float)) == 0; }

static int same_obst(const Tr2Obstacle* x, const Tr2Obstacle* y) {
    return fbits(x->s, y->s) && fbits(x->length, y->length)
        && fbits(x->lane, y->lane) && fbits(x->half_width, y->half_width)
        && fbits(x->y_bottom, y->y_bottom) && fbits(x->y_top, y->y_top)
        && fbits(x->phase, y->phase) && x->kind == y->kind
        && x->variant == y->variant && x->trigger == y->trigger
        && x->death_type == y->death_type && x->active == y->active
        && x->passed == y->passed;
}

static int same_coin(const Tr2Coin* x, const Tr2Coin* y) {
    return fbits(x->s, y->s) && fbits(x->y, y->y) && fbits(x->lane, y->lane)
        && x->value == y->value && x->active == y->active;
}

static int same_powerup(const Tr2Powerup* x, const Tr2Powerup* y) {
    return fbits(x->s, y->s) && fbits(x->lane, y->lane) && x->kind == y->kind
        && x->active == y->active;
}

static int same_seg(const Tr2Segment* x, const Tr2Segment* y) {
    return fbits(x->s0, y->s0) && fbits(x->length, y->length)
        && fbits(x->heading, y->heading) && fbits(x->wx, y->wx)
        && fbits(x->wz, y->wz) && fbits(x->yaw_delta, y->yaw_delta)
        && x->turn == y->turn && x->piece_idx == y->piece_idx
        && x->var_idx == y->var_idx && x->mode == y->mode
        && x->obstacle_kind == y->obstacle_kind
        && x->entity_spawned == y->entity_spawned && x->valid == y->valid;
}

static int same_state(const Env* a, const Env* b) {
#define EQ(field) do { \
        assert(memcmp(&a->field, &b->field, sizeof(a->field)) == 0); \
    } while (0)
    EQ(s); EQ(px); EQ(lane); EQ(target_lane); EQ(y); EQ(vy);
    EQ(grounded); EQ(sliding); EQ(slide_timer); EQ(stumble_timer); EQ(speed);
    EQ(mode); EQ(mode_timer); EQ(turn_input_timer); EQ(turn_input); EQ(dead);
    EQ(death_cause); EQ(last_death_cause); EQ(distance); EQ(score); EQ(coins);
    EQ(gems); EQ(multiplier); EQ(magnet_timer); EQ(shield_timer);
    EQ(boost_distance_left); EQ(invuln_timer); EQ(episode_return); EQ(tick);
    EQ(seg_idx); EQ(seg_gen); EQ(piece_count); EQ(turn_acc); EQ(obstacle_acc);
    EQ(next_turn_gap); EQ(next_obstacle_gap); EQ(next_coin_s);
    EQ(next_powerup_s); EQ(rng);
#undef EQ
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) {
        if (!same_obst(&a->obstacles[i], &b->obstacles[i])) return 0;
    }
    for (int i = 0; i < TR2_MAX_COINS; i++) {
        if (!same_coin(&a->coin_pool[i], &b->coin_pool[i])) return 0;
    }
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) {
        if (!same_powerup(&a->powerups[i], &b->powerups[i])) return 0;
    }
    for (int i = 0; i < TR2_SEG_RING; i++) {
        if (!same_seg(&a->segments[i], &b->segments[i])) return 0;
    }
    return 1;
}

/* -------------------------------------------------------------------- tests */

static void test_obs_contract(void) {
    assert(OBS_SIZE == 238);
    Harness h;
    harness_init(&h, 9);
    for (int j = 0; j < OBS_SIZE; j++) assert(isfinite(h.obs[j]));
    for (int i = 0; i < 1000; i++) {
        h.action = (float)((i * 7 + 3) % TR2_ACTIONS);
        puf_step(&h.env);
        for (int j = 0; j < OBS_SIZE; j++) {
            assert(isfinite(h.obs[j]));
        }
    }
    printf("ok OBS_SIZE == 238, all observations finite\n");
}

/* Hazards of the same kind and centre can require different timing or lanes.
 * Their dimensions must be visible to the policy. */
static void test_obstacle_geometry_observation(void) {
    Harness h;
    harness_init(&h, 9);
    clear_track_hazards(&h);
    put_obstacle(&h.env, TR2_OBST_LOW, 8.0f);
    Tr2Obstacle *o = &h.env.obstacles[0];
    o->half_width = 0.5f;
    o->length = 1.0f;
    o->y_bottom = -0.5f;
    o->y_top = 0.5f;
    tr2_observe(&h.env);
    float width = h.obs[65], length = h.obs[66], bottom = h.obs[67], top = h.obs[68];
    o->half_width = 1.0f;
    o->length = 3.0f;
    o->y_bottom = -1.0f;
    o->y_top = 0.9f;
    tr2_observe(&h.env);
    assert(h.obs[65] > width && h.obs[66] > length);
    assert(h.obs[67] < bottom && h.obs[68] > top);
    printf("ok policy can distinguish obstacle dimensions\n");
}

static void test_determinism(void) {
    Harness a, b;
    harness_init(&a, 42);
    harness_init(&b, 42);
    for (int i = 0; i < 3000; i++) {
        a.action = (float)(i % TR2_ACTIONS);
        b.action = a.action;
        puf_step(&a.env);
        puf_step(&b.env);

        assert(memcmp(a.obs, b.obs, OBS_SIZE * sizeof(obs_t)) == 0);
        assert(same_state(&a.env, &b.env));
        assert(memcmp(&a.env.log, &b.env.log, sizeof(Log)) == 0);
        assert(a.terminal == b.terminal);
        for (int j = 0; j < OBS_SIZE; j++) assert(isfinite(a.obs[j]));
    }
    printf("ok determinism (%.0f episodes, %.0f m)\n", (double)a.env.log.n,
        a.env.log.n > 0.0f ? (double)(a.env.log.perf / a.env.log.n) : 0.0);
}

/* Decision batching must preserve 60 Hz physics and spend only one gesture.
 * In particular, a later interval must not repeat or cancel a jump. */
static void test_swipe_decision_interval(void) {
    const int gestures[] = {TR2_ACT_JUMP, TR2_ACT_SLIDE, TR2_ACT_LEFT};
    for (int k = 0; k < 3; ++k) {
        Harness batched, frames;
        harness_init_costs(&batched, 11, 0.01f);
        harness_init_costs(&frames, 11, 0.01f);
        clear_track_hazards(&batched);
        clear_track_hazards(&frames);
        batched.env.decision_frames = 6;
        batched.action = gestures[k];
        puf_step(&batched.env);
        float total = 0.0f;
        for (int f = 0; f < 6; ++f) {
            frames.action = f == 0 ? gestures[k] : TR2_ACT_NONE;
            puf_step(&frames.env);
            total += frames.reward;
        }
        assert(same_state(&batched.env, &frames.env));
        assert(memcmp(batched.obs, frames.obs, sizeof(batched.obs)) == 0);
        assert(fabsf(batched.reward - total) < 1e-6f);
        assert(fabsf(batched.reward + 0.01f) < 1e-6f);
        assert(batched.env.agents[0].actions == &batched.action);
    }
    Harness fatal;
    harness_init_costs(&fatal, 11, 0.01f);
    clear_track_hazards(&fatal);
    fatal.env.decision_frames = 6;
    put_obstacle(&fatal.env, TR2_OBST_LOW, 0.1f);
    fatal.action = TR2_ACT_SLIDE;
    puf_step(&fatal.env);
    assert(fatal.terminal > 0.5f);
    assert(fatal.env.tick == 0); /* Do not step the reset episode. */
    assert(fatal.reward < -fatal.env.death_penalty);
    assert(fatal.env.agents[0].actions == &fatal.action);
    printf("ok swipe intervals preserve physics, rewards and terminal boundaries\n");
}

static void test_speed(void) {
    Harness h;
    harness_init(&h, 5);
    clear_all(&h);
    assert(fabsf(h.env.speed - h.env.min_speed) < 1e-3f);

    float first = h.env.speed, prev = h.env.speed;
    int steps = 0;
    for (int i = 0; i < 400; i++) {
        clear_all(&h);
        follow_turn_action(&h);
        puf_step(&h.env);
        assert(h.terminal < 0.5f);
        assert(h.env.speed <= h.env.max_speed + 1e-3f);
        assert(h.env.speed >= h.env.min_speed - 1e-3f);
        assert(h.env.speed >= prev - 1e-3f);
        prev = h.env.speed;
        steps = i + 1;
        if (h.env.speed >= h.env.max_speed - 1e-3f) break;
    }
    assert(steps > 1);
    assert(h.env.speed > first);
    printf("ok speed ramp (%.2f -> %.2f in %d steps)\n", (double)first,
        (double)h.env.speed, steps);
}

static float one_step_delta(Harness* h, float ratio) {
    harness_init(h, 5);
    clear_all(h);
    h->env.speed = h->env.max_speed * ratio;
    float before = h->env.speed;
    h->action = -1.0f;
    puf_step(&h->env);
    assert(h->terminal < 0.5f);
    return h->env.speed - before;
}

static void test_force_buckets(void) {
    Harness h;
    float d50 = one_step_delta(&h, 0.50f);
    float d70 = one_step_delta(&h, 0.70f);
    float d90 = one_step_delta(&h, 0.90f);
    float d99 = one_step_delta(&h, 0.99f);
    assert(d50 > 0.0f && d70 > 0.0f && d90 > 0.0f); /* still speeding up */
    assert(d99 >= 0.0f);
    assert(d50 >= d70);
    assert(d70 >= d90);
    assert(d90 >= d99);
    printf("ok force buckets (d50 %.5f >= d70 %.5f >= d90 %.5f >= d99 %.5f)\n",
        (double)d50, (double)d70, (double)d90, (double)d99);

    Harness cap;
    harness_init(&cap, 5);
    clear_all(&cap);
    cap.env.speed = cap.env.max_speed;
    float before = cap.env.speed;
    cap.action = -1.0f;
    puf_step(&cap.env);
    assert(cap.terminal < 0.5f);
    assert(cap.env.speed <= cap.env.max_speed + 1e-3f);
    assert(cap.env.speed >= before - 1e-3f);
    printf("ok speed capped at max (%.2f)\n", (double)cap.env.speed);
}

static void test_obstacles(void) {
    const int kinds[] = {TR2_OBST_LOW, TR2_OBST_HIGH, TR2_OBST_GAP};
    const int action[] = {TR2_ACT_JUMP, TR2_ACT_SLIDE, TR2_ACT_JUMP};

    for (int k = 0; k < 3; k++) {
        Harness h;
        harness_init(&h, 11);
        clear_all(&h);
        h.env.swipe_cost = 0.0f;
        h.env.reward_clear = 1.0f;
        /* The 3 m synthetic gap needs enough speed for the jump to span it. */
        if (kinds[k] == TR2_OBST_GAP) h.env.speed = h.env.max_speed;
        put_obstacle(&h.env, kinds[k], 8.0f);
        float obs_s = h.env.obstacles[0].s;
        float obs_len = h.env.obstacles[0].length;
        step_until(&h, kinds[k] == TR2_OBST_GAP ? 2.8f : 2.0f, 600);
        suppress_except(&h, 0, -1, -1);
        h.action = (float)action[k];
        puf_step(&h.env);
        assert(h.terminal < 0.5f);
        for (int i = 0; i < 80 && h.env.s <= obs_s + obs_len; i++) {
            suppress_except(&h, 0, -1, -1);
            h.action = -1.0f;
            puf_step(&h.env);
            assert(h.terminal < 0.5f);
        }
        assert(h.terminal < 0.5f);
        assert(h.env.s > obs_s + obs_len);
        assert(fabsf(h.env.episode_return - 1.0f) < 1e-5f);
        printf("ok obstacle kind %d cleared\n", kinds[k]);
    }

    for (int k = 0; k < 3; k++) {
        Harness h;
        harness_init(&h, 11);
        clear_all(&h);
        put_obstacle(&h.env, kinds[k], 8.0f);
        for (int i = 0; i < 400 && h.terminal < 0.5f; i++) {
            suppress_except(&h, 0, -1, -1);
            h.action = -1.0f;
            puf_step(&h.env);
        }
        assert(h.terminal > 0.5f);
        assert(h.env.last_death_cause >= TR2_DEATH_FALL);
        assert(h.env.last_death_cause <= TR2_DEATH_WATERRIGHT);
        printf("ok obstacle kind %d kills when ignored (cause %d)\n", kinds[k],
            h.env.last_death_cause);
    }
}

/* The shipped jump-over collider must distinguish a jump from a slide. */
static void test_authored_jump_over(void) {
    int variation = -1;
    for (int i = 0; i < TR2_TK_VARIATION_MAX; ++i)
        if (strcmp(TR2_VARIATION_TABLE[i].name, "jump_over_a") == 0) {
            variation = i;
            break;
        }
    assert(variation >= 0);
    for (int slide = 0; slide < 2; ++slide) {
        Harness h;
        harness_init_costs(&h, 11, 0.0f);
        clear_track_hazards(&h);
        Tr2Segment seg = {0};
        seg.s0 = 5.0f;
        seg.length = 9.6f;
        seg.var_idx = variation;
        tr2_tk_spawn_piece(&h.env, &seg, 0, 0);
        float end = h.env.obstacles[0].s + h.env.obstacles[0].length;
        step_until(&h, 2.0f, 600);
        suppress_except(&h, 0, -1, -1);
        h.action = slide ? TR2_ACT_SLIDE : TR2_ACT_JUMP;
        puf_step(&h.env);
        for (int i = 0; i < 80 && h.terminal < 0.5f && h.env.s < end; ++i) {
            suppress_except(&h, 0, -1, -1);
            h.action = TR2_ACT_NONE;
            puf_step(&h.env);
        }
        if (slide) assert(h.terminal > 0.5f);
        else assert(h.terminal < 0.5f && h.env.s >= end);
    }
    printf("ok authored jump-over clears on jump, kills on slide\n");
}

/* Progress itself is not a reward. A cleanly passed obstacle pays once,
 * including when the runner was already in a safe lane; a collision does not. */
static void test_dodge_reward(void) {
    Harness h;
    harness_init_costs(&h, 23, 0.0f);
    clear_track_hazards(&h);
    h.env.reward_clear = 1.0f;
    h.action = TR2_ACT_NONE;
    for (int i = 0; i < 20; ++i) {
        clear_track_hazards(&h);
        puf_step(&h.env);
        assert(h.reward == 0.0f && h.terminal < 0.5f);
    }
    put_obstacle(&h.env, TR2_OBST_LOW, 3.0f);
    h.env.obstacles[0].lane = -1.0f;
    h.env.obstacles[0].half_width = 0.25f;
    float total = 0.0f;
    for (int i = 0; i < 50; ++i) {
        suppress_except(&h, 0, -1, -1);
        puf_step(&h.env);
        total += h.reward;
        assert(h.terminal < 0.5f);
    }
    assert(fabsf(total - 1.0f) < 1e-5f);
    Harness hit;
    harness_init_costs(&hit, 23, 0.0f);
    clear_track_hazards(&hit);
    hit.env.reward_clear = 1.0f;
    put_obstacle(&hit.env, TR2_OBST_LOW, 3.0f);
    hit.action = TR2_ACT_NONE;
    total = 0.0f;
    for (int i = 0; i < 50 && hit.terminal < 0.5f; ++i) {
        suppress_except(&hit, 0, -1, -1);
        puf_step(&hit.env);
        total += hit.reward;
    }
    assert(hit.terminal > 0.5f && total < 0.0f);
    printf("ok clean obstacle reward and collision penalty\n");
}

static void test_turn(void) {
    Harness h;
    harness_init(&h, 7);
    int turned = 0;
    for (int i = 0; i < 5000; i++) {
        clear_all(&h);
        const Tr2Segment* next =
            &h.env.segments[(h.env.seg_idx + 1) % TR2_SEG_RING];
        float to_next = next->s0 - h.env.s;
        h.action = -1.0f;
        if (next->turn != 0 && to_next < 6.0f) {
            h.action =
                next->turn < 0 ? (float)TR2_ACT_LEFT : (float)TR2_ACT_RIGHT;
            turned = 1;
        }
        puf_step(&h.env);
        assert(h.terminal < 0.5f);
        if (turned && h.env.seg_idx > 1) break;
    }
    assert(turned);
    printf("ok turn follow (%.1f m)\n", (double)h.env.s);

    Harness miss;
    harness_init(&miss, 7);
    for (int i = 0; i < 3000 && !miss.env.off_track; i++) {
        clear_all(&miss);
        miss.action = -1.0f; /* never swipe */
        puf_step(&miss.env);
        assert(miss.terminal < 0.5f);
    }
    assert(miss.env.off_track);
    for (int i = 0; i < 20; i++) {
        clear_all(&miss);
        puf_step(&miss.env);
        assert(miss.terminal < 0.5f);
    }
    assert(miss.env.mode == TR2_MODE_FALLING);
    for (int i = 0; i < 60 && miss.terminal < 0.5f; i++) {
        clear_all(&miss);
        puf_step(&miss.env);
    }
    assert(miss.terminal > 0.5f);
    assert(miss.env.last_death_cause >= TR2_DEATH_FALL);
    assert(miss.env.last_death_cause <= TR2_DEATH_WATERRIGHT);
    printf("ok missed turn kills (cause %d)\n", miss.env.last_death_cause);
}

/* Consumer behaviour: a clear track can be left alone.  Idle never changes
 * lane, never jumps, never slides, and the runner keeps advancing. */
static void test_idle_on_clear_track(void) {
    Harness h;
    harness_init(&h, 13);
    clear_track_hazards(&h);
    float s0 = h.env.s;
    for (int i = 0; i < 600; i++) {   /* ~10 s, >100 m of clear track */
        clear_track_hazards(&h);
        h.action = (float)TR2_ACT_NONE;
        puf_step(&h.env);
        assert(h.terminal < 0.5f);
        assert(h.env.last_action == TR2_ACT_NONE);
        assert(h.env.target_lane == TR2_CENTER_LANE);
        assert(h.env.lane == TR2_CENTER_LANE);
        assert(fabsf(h.env.px) < 1e-6f);
        assert(h.env.grounded && !h.env.sliding);
        assert(h.env.y == 0.0f && h.env.vy == 0.0f);
        assert(h.env.turn_input == 0 && h.env.off_track == 0);
    }
    assert(h.env.s > s0 + 100.0f);
    assert(h.env.distance > 100.0f);
    printf("ok idle on clear track (%.1f m, lane %d)\n", (double)h.env.s,
        h.env.lane);
}

/* Repeated outward swipes must stay within the game's 0.85 m lateral range. */
static void test_lane_world_bounds(void) {
    for (int dir = 0; dir < 2; ++dir) {
        Harness h;
        harness_init(&h, 31);
        clear_track_hazards(&h);
        for (int i = 0; i < 80; ++i) {
            clear_track_hazards(&h);
            h.action = dir ? TR2_ACT_RIGHT : TR2_ACT_LEFT;
            puf_step(&h.env);
            assert(h.terminal < 0.5f);
            float x, y, z, yaw, cx, cy, cz, cyaw;
            tr2_world_at(&h.env, h.env.s, h.env.px, 0.0f, &x, &y, &z, &yaw);
            tr2_world_at(&h.env, h.env.s, 0.0f, 0.0f, &cx, &cy, &cz, &cyaw);
            assert(hypotf(x - cx, z - cz) <= 0.851f);
        }
        assert(fabsf(h.env.px) > 0.99f);
    }
    printf("ok outward swipes stay within playable lateral range\n");
}

/* Each gesture does exactly its physical thing from the centre lane, and idle
 * does none of them; idle also never cancels an in-progress gesture. */
static void test_gestures_do_their_thing(void) {
    Harness h;

    harness_init(&h, 31);
    clear_track_hazards(&h);
    h.action = (float)TR2_ACT_LEFT;
    puf_step(&h.env);
    assert(h.env.target_lane == TR2_LANE_MIN);
    assert(h.env.px < 0.0f);
    assert(h.env.grounded && !h.env.sliding);

    harness_init(&h, 31);
    clear_track_hazards(&h);
    h.action = (float)TR2_ACT_RIGHT;
    puf_step(&h.env);
    assert(h.env.target_lane == TR2_LANE_MAX);
    assert(h.env.px > 0.0f);

    harness_init(&h, 31);
    clear_track_hazards(&h);
    h.action = (float)TR2_ACT_JUMP;
    puf_step(&h.env);
    assert(!h.env.grounded && h.env.y > 0.0f && !h.env.sliding);

    harness_init(&h, 31);
    clear_track_hazards(&h);
    h.action = (float)TR2_ACT_SLIDE;
    puf_step(&h.env);
    assert(h.env.sliding && h.env.slide_timer > 0.0f);
    assert(h.env.grounded && h.env.y == 0.0f);

    /* Idle in mid-air neither re-jumps nor cancels the jump. */
    harness_init(&h, 31);
    clear_track_hazards(&h);
    h.action = (float)TR2_ACT_JUMP;
    puf_step(&h.env);
    float y1 = h.env.y;
    clear_track_hazards(&h);
    h.action = (float)TR2_ACT_NONE;
    puf_step(&h.env);
    assert(!h.env.grounded);
    assert(fabsf(h.env.y - y1) > 1e-4f);   /* the arc keeps integrating */
    assert(!h.env.sliding);

    /* Idle during a slide lets it finish; it does not extend or cut it. */
    harness_init(&h, 31);
    clear_track_hazards(&h);
    h.action = (float)TR2_ACT_SLIDE;
    puf_step(&h.env);
    float slide_left = h.env.slide_timer;
    clear_track_hazards(&h);
    h.action = (float)TR2_ACT_NONE;
    puf_step(&h.env);
    assert(h.env.sliding);
    assert(h.env.slide_timer < slide_left);
    printf("ok four gestures act, idle acts on none\n");
}

/* Deliberate repeated swipes still chain lane changes; the shipped 120 ms
 * debounce still absorbs repeats inside its window. */
static void test_repeated_swipes(void) {
    Harness h;
    harness_init(&h, 23);
    clear_track_hazards(&h);

    h.action = (float)TR2_ACT_RIGHT;
    puf_step(&h.env);
    assert(h.env.target_lane == TR2_LANE_MAX);
    for (int i = 0; i < 10; i++) {   /* longer than the 120 ms debounce */
        clear_track_hazards(&h);
        h.action = (float)TR2_ACT_NONE;
        puf_step(&h.env);
    }
    assert(h.env.target_lane == TR2_LANE_MAX);

    h.action = (float)TR2_ACT_LEFT;
    puf_step(&h.env);
    assert(h.env.target_lane == 1);
    for (int i = 0; i < 10; i++) {
        clear_track_hazards(&h);
        h.action = (float)TR2_ACT_NONE;
        puf_step(&h.env);
    }
    h.action = (float)TR2_ACT_LEFT;
    puf_step(&h.env);
    assert(h.env.target_lane == TR2_LANE_MIN);
    assert(h.terminal < 0.5f && h.env.off_track == 0);

    /* A swipe into the wall is legal and leaves the run alive. */
    for (int i = 0; i < 10; i++) {
        clear_track_hazards(&h);
        h.action = (float)TR2_ACT_NONE;
        puf_step(&h.env);
    }
    h.action = (float)TR2_ACT_LEFT;
    puf_step(&h.env);
    assert(h.env.target_lane == TR2_LANE_MIN);
    assert(h.terminal < 0.5f);

    /* Repeated gestures inside the debounce window collapse to one lane. */
    Harness g;
    harness_init(&g, 23);
    clear_track_hazards(&g);
    for (int i = 0; i < 3; i++) {
        g.action = (float)TR2_ACT_RIGHT;
        puf_step(&g.env);
    }
    assert(g.env.target_lane == TR2_LANE_MAX);
    assert(g.terminal < 0.5f);
    printf("ok repeated swipes chain, debounce preserved\n");
}

/* A swipe inside the turn window is buffered as a turn command (not a lane
 * change) and that single buffered swipe carries the turn; the opposite
 * direction is an ordinary lane change and the missed turn still kills, so
 * nothing auto-steers. */
static void test_turn_swipe_buffers(void) {
    Harness h;
    harness_init(&h, 29);
    h.env.enable_submodes = 0;   /* a mandatory turn must be earned by the swipe */
    int found = 0, dir = 0, lane_before = 0, seg_before = 0;
    for (int i = 0; i < 6000; i++) {
        clear_all(&h);
        const Tr2Segment* next =
            &h.env.segments[(h.env.seg_idx + 1) % TR2_SEG_RING];
        float to_next = next->s0 - h.env.s;
        h.action = (float)TR2_ACT_NONE;
        if (next->turn != 0 && to_next > 0.6f
                && to_next < h.env.turn_window - 1.0f) {
            dir = next->turn;
            lane_before = h.env.target_lane;
            seg_before = h.env.seg_idx;
            h.action =
                (float)(dir < 0 ? TR2_ACT_LEFT : TR2_ACT_RIGHT);
            puf_step(&h.env);
            found = 1;
            break;
        }
        puf_step(&h.env);
        assert(h.terminal < 0.5f);
    }
    assert(found);
    assert(h.env.off_track == 0);
    assert(h.env.turn_input == dir);           /* buffered for piece entry */
    assert(h.env.turn_input_timer > 0.0f);
    assert(h.env.target_lane == lane_before);  /* turn, not a lane change */

    /* Idling afterwards: the buffered swipe alone carries the turn. */
    int taken = 0;
    for (int i = 0; i < 600 && h.terminal < 0.5f; i++) {
        clear_all(&h);
        h.action = (float)TR2_ACT_NONE;
        puf_step(&h.env);
        if (h.env.seg_idx > seg_before) {
            taken = 1;
            break;
        }
    }
    assert(taken);
    assert(h.env.off_track == 0);
    assert(h.env.turn_input == 0);             /* consumed at piece entry */
    assert(h.env.target_lane == lane_before);
    printf("ok turn swipe buffers then carries the turn (dir %d)\n", dir);

    /* Wrong-direction swipe near the same kind of turn: lane change, no
     * autopilot, still lethal when the turn is missed. */
    Harness w;
    harness_init(&w, 29);
    w.env.enable_submodes = 0;
    int wfound = 0, wdir = 0, wlane = 0;
    for (int i = 0; i < 6000; i++) {
        clear_all(&w);
        const Tr2Segment* next =
            &w.env.segments[(w.env.seg_idx + 1) % TR2_SEG_RING];
        float to_next = next->s0 - w.env.s;
        w.action = (float)TR2_ACT_NONE;
        if (next->turn != 0 && to_next > 0.6f
                && to_next < w.env.turn_window - 1.0f) {
            wdir = next->turn;
            wlane = w.env.target_lane;
            w.action = (float)(wdir < 0 ? TR2_ACT_RIGHT : TR2_ACT_LEFT);
            puf_step(&w.env);
            wfound = 1;
            break;
        }
        puf_step(&w.env);
        assert(w.terminal < 0.5f);
    }
    assert(wfound);
    int expect = wlane + (wdir < 0 ? 1 : -1);
    if (expect < TR2_LANE_MIN) expect = TR2_LANE_MIN;
    if (expect > TR2_LANE_MAX) expect = TR2_LANE_MAX;
    assert(w.env.target_lane == expect);
    assert(w.env.turn_input == 0);
    assert(w.env.off_track == 0);
    for (int i = 0; i < 900 && w.terminal < 0.5f; i++) {
        clear_all(&w);
        w.action = (float)TR2_ACT_NONE;
        puf_step(&w.env);
    }
    assert(w.terminal > 0.5f);
    assert(w.env.last_death_cause >= TR2_DEATH_FALL);
    assert(w.env.last_death_cause <= TR2_DEATH_WATERRIGHT);
    printf("ok wrong swipe is a lane change; missed turn still kills\n");
}

/* swipe_cost anti-spam shaping: a 0.0 value leaves the reward arithmetic
 * untouched, a positive value charges each gesture-input step and never the
 * idle action, and it never gates the gesture itself. */
static void test_swipe_cost_boundaries(void) {
    Harness idle, jump, charged;
    harness_init_costs(&idle, 17, 0.0f);
    clear_track_hazards(&idle);
    harness_init_costs(&jump, 17, 0.0f);
    clear_track_hazards(&jump);
    harness_init_costs(&charged, 17, 0.25f);
    clear_track_hazards(&charged);

    idle.action = (float)TR2_ACT_NONE;
    jump.action = (float)TR2_ACT_JUMP;
    charged.action = (float)TR2_ACT_JUMP;
    puf_step(&idle.env);
    puf_step(&jump.env);
    puf_step(&charged.env);
    /* Zero cost: neither idle nor jumping on a clear track earns reward. */
    assert(fabsf(idle.reward - jump.reward) < 1e-3f);
    /* Charged: exactly one swipe_cost this step, and the jump still happens. */
    assert(fabsf((idle.reward - charged.reward) - 0.25f) < 1e-3f);
    assert(!charged.env.grounded && charged.env.y > 0.0f);
    assert(charged.env.episode_return <= -0.2f);

    /* Idle never pays, however long it is held. */
    Harness a, b;
    harness_init_costs(&a, 19, 0.25f);
    clear_track_hazards(&a);
    harness_init_costs(&b, 19, 0.0f);
    clear_track_hazards(&b);
    for (int i = 0; i < 60; i++) {
        clear_track_hazards(&a);
        clear_track_hazards(&b);
        a.action = (float)TR2_ACT_NONE;
        b.action = (float)TR2_ACT_NONE;
        puf_step(&a.env);
        puf_step(&b.env);
    }
    assert(fbits(a.reward, b.reward));
    assert(fbits(a.env.episode_return, b.env.episode_return));

    /* An input the world swallows (slide while already sliding) still pays:
     * the cost tracks real gesture input, not whether it changed the world. */
    Harness s;
    harness_init_costs(&s, 19, 0.25f);
    clear_track_hazards(&s);
    s.action = (float)TR2_ACT_SLIDE;
    puf_step(&s.env);
    assert(s.env.sliding);
    s.action = (float)TR2_ACT_SLIDE;
    puf_step(&s.env);
    assert(s.env.sliding && s.reward < -0.2f);
    printf("ok swipe_cost (charged vs zero, idle free, gestures never gated)\n");
}

static void test_pickups(void) {
    Harness h;
    harness_init(&h, 3);
    h.env.reward_coin = 0.15f;
    clear_all(&h);
    Tr2Coin* c = &h.env.coin_pool[0];
    memset(c, 0, sizeof(*c));
    c->s = h.env.s + 2.0f;
    c->lane = 0.0f;
    c->y = 0.5f;
    c->value = 1;
    c->active = 1;
    int before = h.env.coins;
    for (int i = 0; i < 60 && h.env.coins == before; i++) {
        suppress_except(&h, -1, 0, -1);
        h.action = -1.0f;
        puf_step(&h.env);
    }
    assert(h.env.coins == before + 1);
    assert(h.env.score > h.env.distance);
    assert(fabsf(h.reward - 0.15f) < 1e-6f);
    printf("ok coin pickup\n");

    Harness m;
    harness_init(&m, 3);
    clear_all(&m);
    Tr2Powerup* p = &m.env.powerups[0];
    memset(p, 0, sizeof(*p));
    p->s = m.env.s + 2.0f;
    p->lane = 0.0f;
    p->kind = TR2_PU_MAGNET;
    p->active = 1;
    for (int i = 0; i < 60 && m.env.magnet_timer <= 0.0f; i++) {
        suppress_except(&m, -1, -1, 0);
        m.action = -1.0f;
        puf_step(&m.env);
    }
    assert(m.env.magnet_timer > 0.0f);
    assert(m.reward == 0.0f);
    printf("ok magnet pickup\n");

    Harness s;
    harness_init(&s, 3);
    clear_all(&s);
    Tr2Powerup* q = &s.env.powerups[0];
    memset(q, 0, sizeof(*q));
    q->s = s.env.s + 2.0f;
    q->lane = 0.0f;
    q->kind = TR2_PU_SHIELD;
    q->active = 1;
    for (int i = 0; i < 60 && s.env.shield_timer <= 0.0f; i++) {
        suppress_except(&s, -1, -1, 0);
        s.action = -1.0f;
        puf_step(&s.env);
    }
    assert(s.env.shield_timer > 0.0f);
    printf("ok shield pickup\n");
}

static void test_shield_obstacle_recovery(void) {
    Harness h;
    harness_init(&h, 7);
    clear_all(&h);
    h.env.shield_timer = 10.0f;
    put_obstacle(&h.env, TR2_OBST_LOW, 1.0f);
    h.action = -1.0f;
    for (int i = 0; i < 30 && h.env.shield_timer > 0.0f; i++) {
        suppress_except(&h, 0, -1, -1);
        puf_step(&h.env);
    }
    assert(h.terminal < 0.5f && h.env.shield_timer == 0.0f);
    assert(h.env.invuln_timer > 0.0f && h.env.invuln_timer <= 0.2f);
    for (int i = 0; i < 20; i++) {
        suppress_except(&h, -1, -1, -1);
        puf_step(&h.env);
    }
    assert(h.terminal < 0.5f && h.env.invuln_timer <= 0.0f);
    put_obstacle(&h.env, TR2_OBST_LOW, 1.0f);
    for (int i = 0; i < 30 && h.terminal < 0.5f; i++) {
        suppress_except(&h, 0, -1, -1);
        puf_step(&h.env);
    }
    assert(h.terminal > 0.5f);
    printf("ok shield grants brief obstacle protection\n");
}

static void test_repeated_stumble(void) {
    Harness h;
    harness_init(&h, 7);
    clear_all(&h);
    h.action = -1.0f;
    put_obstacle(&h.env, TR2_OBST_STUMBLE, 1.0f);
    for (int i = 0; i < 30 && h.env.stumble_timer <= 0.0f; i++) {
        suppress_except(&h, 0, -1, -1);
        puf_step(&h.env);
    }
    assert(h.terminal < 0.5f && h.env.stumble_timer > 0.0f);
    for (int i = 0; i < 75; i++) {
        suppress_except(&h, -1, -1, -1);
        puf_step(&h.env);
        assert(h.terminal < 0.5f);
    }
    put_obstacle(&h.env, TR2_OBST_STUMBLE, 1.0f);
    for (int i = 0; i < 30 && h.terminal < 0.5f; i++) {
        suppress_except(&h, 0, -1, -1);
        puf_step(&h.env);
    }
    assert(h.terminal > 0.5f && h.env.last_death_cause == TR2_DEATH_EATEN);
    printf("ok repeated stumble causes monkey death\n");
}

/* turn_left_a is an authored two-leg path: straight 5.12, then left 5.12.
 * A lane offset must rotate with the second leg, not with the piece origin. */
static void test_authored_turn_geometry(void) {
    Tr2Segment seg = {0};
    seg.length = 10.24f;
    seg.spline_begin = TR2_VARIATION_TABLE[38].spline_begin;
    seg.spline_count = TR2_VARIATION_TABLE[38].spline_count;
    float x, z, yaw, lane_x, lane_z, lane_yaw;
    tr2_seg_point(&seg, 7.68f, 0.0f, &x, &z, &yaw);
    tr2_seg_point(&seg, 7.68f, 1.0f, &lane_x, &lane_z, &lane_yaw);
    assert(fabsf(x - 2.56f) < 1e-4f);
    assert(fabsf(z - 5.12f) < 1e-4f);
    assert(fabsf(yaw - 0.5f * PI) < 1e-4f);
    assert(fabsf(lane_x - x) < 1e-4f);
    assert(fabsf(lane_z - z - 1.0f) < 1e-4f);
    printf("ok authored turn geometry and lateral offset\n");
}

/* A captured GeneratedPath sample bends away from the SourcePath chord. */
static void test_smoothed_path(void) {
    Tr2Segment seg = {0};
    seg.spline_begin = TR2_VARIATION_TABLE[14].spline_begin;
    seg.spline_count = TR2_VARIATION_TABLE[14].spline_count;
    seg.length = tr2_tk_spline_length(&seg);
    float x, z, yaw;
    tr2_seg_point(&seg, 3.403667f, 0.0f, &x, &z, &yaw);
    assert(fabsf(x + 0.29351f) < 0.001f);
    assert(fabsf(z - 3.36075f) < 0.001f);
    printf("ok generated path curvature\n");
}

/* A right-angle piece join must not make the rendered runner/camera snap. */
static void test_render_heading_at_piece_join(void) {
    Env env = {0};
    env.seg_gen = 1;
    env.segments[0].valid = 1;
    env.segments[0].length = 10.0f;
    env.segments[1].valid = 1;
    env.segments[1].s0 = 10.0f;
    env.segments[1].length = 10.0f;
    env.segments[1].wz = 10.0f;
    env.segments[1].heading = PI * 0.5f;
    env.seg_idx = 0;
    float before = tr2_r_path_yaw(&env, 9.9999f);
    env.seg_idx = 1;
    float after = tr2_r_path_yaw(&env, 10.0001f);
    assert(fabsf(before - PI * 0.25f) < 0.001f);
    assert(fabsf(after - before) < 0.001f);
    printf("ok rendered heading across right-angle piece join\n");
}

/* A junction has two actual exits; both must be generated and their paths
 * must lead to opposite sides of the same junction mesh. */
static void test_junction_branches(void) {
    Env env = {0};
    env.rng = 12345;
    int left = 0, right = 0;
    for (int i = 0; i < 64; ++i) {
        Tr2Segment seg = {0};
        tr2_tk_apply_piece(&env, &seg, TR2_PIECE_R100_Z0_JUNCTION);
        float x, z, heading;
        tr2_seg_end(&seg, &x, &z, &heading);
        assert(fabsf(fabsf(x) - 2.56f) < 1e-3f);
        assert(fabsf(z - 2.56f) < 1e-3f);
        if (seg.turn < 0) { ++left; assert(x > 0.0f); }
        else { ++right; assert(seg.turn > 0 && x < 0.0f); }
    }
    assert(left > 0 && right > 0);
    printf("ok generated junctions use both branches\n");
}

/* Reusing a piece type must not always reproduce the same visual variation. */
static void test_course_variations(void) {
    Env env = {0};
    env.rng = 193;
    int previous = -1;
    for (int i = 0; i < 32; ++i) {
        Tr2Segment seg = {0};
        tr2_tk_apply_piece(&env, &seg, TR2_PIECE_R100_Z0_STRAIGHT_FLAT);
        assert(seg.var_idx != previous);
        previous = seg.var_idx;
    }
    printf("ok piece variations do not immediately repeat\n");
}

/* A long pit must not be offered below its physical crossing speed. Once
 * reachable it remains part of the random course, rather than being removed. */
static void test_gap_generation_reachability(void) {
    Harness h;
    harness_init(&h, 11);
    h.env.decision_frames = 2;
    int pidx = -1;
    for (int i = 0; i < TR2_TK_VARIATION_MAX; ++i)
        if (!strcmp(TR2_VARIATION_TABLE[i].name, "gap_large_a")) {
            pidx = TR2_VARIATION_TABLE[i].piece;
            break;
        }
    assert(pidx >= 0);
    h.env.speed = 10.0f;
    for (int i = 0; i < 32; ++i) {
        Tr2Segment seg = {0};
        tr2_tk_apply_piece(&h.env, &seg, pidx);
        assert(strcmp(TR2_VARIATION_TABLE[seg.var_idx].name, "gap_large_a") != 0);
    }
    h.env.speed = 21.0f;
    int long_gaps = 0;
    for (int i = 0; i < 32; ++i) {
        Tr2Segment seg = {0};
        tr2_tk_apply_piece(&h.env, &seg, pidx);
        long_gaps += strcmp(TR2_VARIATION_TABLE[seg.var_idx].name, "gap_large_a") == 0;
    }
    assert(long_gaps > 0);
    printf("ok long gaps generated only after becoming reachable\n");
}

/* Water exempts rising entry, not standing/sliding or descending re-entry. */
static void test_water_jump_entry(void) {
    int vi = -1;
    for (int i = 0; i < TR2_TK_VARIATION_MAX; ++i)
        if (!strcmp(TR2_VARIATION_TABLE[i].name, "curve_water_a")) vi = i;
    assert(vi >= 0);
    for (int jumping = 0; jumping < 2; ++jumping) {
        Harness h;
        harness_init(&h, 3);
        clear_track_hazards(&h);
        Tr2Segment seg = {0};
        seg.s0 = 8.0f;
        seg.var_idx = vi;
        tr2_tk_spawn_piece(&h.env, &seg, 0, 0);
        Tr2Obstacle* o = NULL;
        for (int i = 0; i < TR2_MAX_OBSTACLES; ++i)
            if (h.env.obstacles[i].active
                    && h.env.obstacles[i].trigger == TR2_TRIG_WATER)
                o = &h.env.obstacles[i];
        assert(o && o->kind == TR2_OBST_WATER);
        o->owner_segment_plus_one = 0; /* isolate the trigger in a flat frame */
        h.env.s = o->s;
        h.env.px = o->lane;
        h.env.y = 0.2f;
        h.env.grounded = !jumping;
        h.env.vy = jumping ? 5.0f : 0.0f;
        h.env.shield_timer = 10.0f;
        tr2_observe(&h.env);
        assert(h.obs[56 + TR2_OBST_WATER] == 1.0f);
        tr2_collision_step(&h.env, h.env.s - 0.1f);
        if (!jumping) {
            assert(h.env.dead && h.env.death_cause == o->death_type);
        } else {
            assert(!h.env.dead && o->trigger_entered);
            h.env.vy = -5.0f;
            tr2_collision_step(&h.env, h.env.s);
            assert(!h.env.dead); /* same overlap is not a fresh entry */
            h.env.px += 20.0f;
            tr2_collision_step(&h.env, h.env.s);
            h.env.px = o->lane;
            tr2_collision_step(&h.env, h.env.s);
            assert(h.env.dead);
        }
    }
    printf("ok water rising entry, continuous overlap and descending re-entry\n");
}

/* A zipline is caught by jumping; walking into it misses the rope and falls.
 * While attached the runner hangs at rope height instead of running on air. */
static void test_zipline_grab_and_hang(void) {
    float walked_reward = 0.0f, walked_terminal = 0.0f;
    float air_reward = 0.0f, air_terminal = 0.0f;
    Env walked = {0};
    walked.agents[0].rewards = &walked_reward;
    walked.agents[0].terminals = &walked_terminal;
    walked.seg_gen = 0;
    walked.segments[0].valid = 1;
    walked.segments[0].mode = TR2_MODE_ZIPLINE;
    walked.segments[0].length = 30.0f;
    walked.seg_idx = 0;
    walked.enable_submodes = 1;
    walked.grounded = 1;
    tr2_runner_apply_mode(&walked);
    assert(walked.dead && walked.death_cause == TR2_DEATH_FALL);

    Env air = {0};
    air.agents[0].rewards = &air_reward;
    air.agents[0].terminals = &air_terminal;
    air.seg_gen = 0;
    air.segments[0].valid = 1;
    air.segments[0].mode = TR2_MODE_ZIPLINE;
    air.segments[0].length = 30.0f;
    air.seg_idx = 0;
    air.enable_submodes = 1;
    air.grounded = 0;
    air.y = 0.6f;
    air.vy = 2.0f;
    air.lane = TR2_CENTER_LANE;
    air.target_lane = TR2_CENTER_LANE;
    air.speed = 10.0f;
    air.min_speed = 10.0f;
    air.max_speed = 21.0f;
    air.agents[0].actions = NULL;   /* idle every frame */
    tr2_runner_apply_mode(&air);
    assert(!air.dead && air.mode == TR2_MODE_ZIPLINE);
    assert(fabsf(air.y - TR2_RUNNER_ZIPLINE_HANG) < 1e-6f);
    for (int i = 0; i < 30; i++) {
        tr2_runner_step(&air);
        assert(!air.dead);
        assert(fabsf(air.y - TR2_RUNNER_ZIPLINE_HANG) < 1e-6f);
    }
    printf("ok zipline needs an airborne grab and hangs at rope height\n");
}

/* WaterSlide duck obstacles sit just above the flume.  The prone water pose
 * clears them only while the duck gesture is held; standing hits them. */
static void test_water_duck_under(void) {
    for (int duck = 0; duck < 2; ++duck) {
        Harness h;
        harness_init(&h, 7);
        clear_all(&h);
        Tr2Obstacle* o = &h.env.obstacles[0];
        memset(o, 0, sizeof(*o));
        o->s = h.env.s + 0.4f;
        o->length = 1.0f;
        o->lane = 0.0f;
        o->half_width = 1.4f;
        o->y_bottom = 0.228f;
        o->y_top = 1.333f;
        o->kind = TR2_OBST_HIGH;
        o->active = 1;
        h.env.mode = TR2_MODE_WATER;
        h.env.grounded = 1;
        h.env.sliding = duck;
        tr2_collision_step(&h.env, h.env.s);
        if (duck) assert(!h.env.dead);
        else assert(h.env.dead);
    }
    printf("ok water slide ducks under flume obstacles, standing hits\n");
}

/* A WaterSlide runs entrance -> 2..5 middle flume pieces -> exit, so short
 * slides and long rivers both occur instead of isolated one-piece splashes. */
static void test_water_slide_chain(void) {
    Env env = {0};
    env.rng = 5;
    env.enable_submodes = 1;
    env.speed = 10.0f;
    env.min_speed = 10.0f;
    env.max_speed = 21.0f;
    tr2_track_reset(&env);
    memset(env.segments, 0, sizeof(env.segments));
    env.seg_gen = -1;
    env.seg_idx = 0;
    env.water_run_left = 3;
    tr2_track_generate(&env);
    for (int i = 0; i < 3; ++i) {
        const Tr2Segment* s = &env.segments[i % TR2_SEG_RING];
        assert(s->valid && s->mode == TR2_MODE_WATER);
        assert(s->piece_idx != TR2_PIECE_WATERSLIDE_ENTER);
        assert(s->piece_idx != TR2_PIECE_WATERSLIDE_EXIT);
    }
    assert(env.segments[3 % TR2_SEG_RING].piece_idx
            == TR2_PIECE_WATERSLIDE_EXIT);
    printf("ok waterslide chains middle flume pieces before the exit\n");
}

int main(void) {
    test_obs_contract();
    test_obstacle_geometry_observation();
    test_water_jump_entry();
    test_zipline_grab_and_hang();
    test_water_duck_under();
    test_water_slide_chain();
    test_determinism();
    test_swipe_decision_interval();
    test_gap_generation_reachability();
    test_speed();
    test_force_buckets();
    test_obstacles();
    test_authored_jump_over();
    test_dodge_reward();
    test_turn();
    test_idle_on_clear_track();
    test_lane_world_bounds();
    test_gestures_do_their_thing();
    test_repeated_swipes();
    test_turn_swipe_buffers();
    test_swipe_cost_boundaries();
    test_authored_turn_geometry();
    test_smoothed_path();
    test_render_heading_at_piece_join();
    test_junction_branches();
    test_course_variations();
    test_pickups();
    test_shield_obstacle_recovery();
    test_repeated_stumble();
    printf("all templerun2 tests passed\n");
    return 0;
}
