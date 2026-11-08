/* Deterministic gap semantics tests for ocean/templerun2.
 *
 * The shipped `gap_large_a` prefab carries a MissedTurn pit (Trigger 3,
 * DeathType Fall, 7.385 m long, top 0.387) and a `Collision_FallSaver_A`
 * trigger at its far edge.  TRRunnerPawn.OnTriggerEnter's MissedTurn branch
 * (0x19653C8) has no unconditional Kill -- it clears the ground flags and hands
 * the runner to Update's ground-dropped-away block (COLLISION.md sec.6.1) --
 * and its gate `_JumpActive && !IsMandatoryTurn() && IsJumpingUpPhase`
 * (0x1965438..5458) lets a rising jump over a non-turn piece pass untouched.
 * The FallSaver branch (0x19658CC) rescues a runner that touches its window
 * while `_JumpActive` or within fallSaverPostLandGrace (0.1 s, 0x238) of a
 * landing.  templerun2_collision.h models the opening as a hole: it takes the
 * support away over the pit, the runner falls onto the pit floor, and that fall
 * is the death.
 *
 * What the shipped numbers imply (speed * 2*v/g, v = clamp(speed, 8, 10) = 10,
 * air gravity 2 * 25 => 0.383 s airtime): a jump spans ~0.383 * speed metres,
 * so the 7.385 m opening is only clearable above ~19.3 m/s and only the
 * FallSaver can save an attempt at 18 m/s.  These tests therefore do NOT assert
 * that every speed clears; they pin the source behaviour at both ends.
 *
 * clang -O2 -I. -Isrc -I./raylib-5.5_macos/include -I./ocean/templerun2 \
 *   -DPLATFORM_DESKTOP tests/test_templerun2_gap.c \
 *   ./raylib-5.5_macos/lib/libraylib.a -framework Cocoa -framework IOKit \
 *   -framework CoreVideo -framework OpenGL -lm -lpthread -o /tmp/test_tr2_gap
 * /tmp/test_tr2_gap
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

/* One decision per frame: these tests exercise 60 Hz physics, not batching. */
static void harness_init(Harness* h, unsigned int seed) {
    Ini ini = {0};
    puf_ini_load_env(&ini, "templerun2", 0, NULL);
    Dict* kwargs = puf_ini_section(&ini, "env", 1);
    fill_missing(kwargs);
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

/* Scrub the generated hazard stream and leave a clear running path: only the
 * volume a test places itself is kept live (its slot index is returned by
 * place_gap). */
static void clear_track(Harness* h) {
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) h->env.obstacles[i].active = 0;
    for (int i = 0; i < TR2_MAX_COINS; i++) h->env.coin_pool[i].active = 0;
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) h->env.powerups[i].active = 0;
    h->env.next_coin_s = 1e9f;
    h->env.next_powerup_s = 1e9f;
    h->env.next_turn_gap = 1e9f;
    h->env.turn_acc = 0.0f;
    h->env.enable_submodes = 0;
    for (int i = 0; i < TR2_SEG_RING; i++) h->env.segments[i].turn = 0;
}

static int variation_index(const char* name) {
    for (int i = 0; i < TR2_TK_VARIATION_MAX; ++i)
        if (strcmp(TR2_VARIATION_TABLE[i].name, name) == 0) return i;
    return -1;
}

/* Stamp one gap_large_a prefab 4 m ahead of the runner and return the slot the
 * generator gives every asset-stamped volume. */
static int place_gap(Harness* h) {
    int variation = variation_index("gap_large_a");
    assert(variation >= 0);
    Tr2Segment seg = {0};
    seg.s0 = 4.0f;
    seg.length = 13.44f;
    seg.var_idx = variation;
    tr2_tk_spawn_piece(&h->env, &seg, 0, 0);
    int slot = -1;
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) {
        if (!h->env.obstacles[i].active) continue;
        slot = i;
        break;
    }
    assert(slot >= 0);
    /* The test places the volume at an absolute s on its own stretch of track,
     * so drop the owning-segment binding: the authored span is then used
     * directly, with no local path height to rectify against. */
    h->env.obstacles[slot].owner_segment_plus_one = 0;
    return slot;
}

static void suppress_except(Harness* h, int keep) {
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++)
        if (i != keep) h->env.obstacles[i].active = 0;
    for (int i = 0; i < TR2_MAX_COINS; i++) h->env.coin_pool[i].active = 0;
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) h->env.powerups[i].active = 0;
}

/* ------------------------------------------------------------------- driver */

enum { GAP_IDLE = 0, GAP_JUMP = 1, GAP_SLIDE = 2 };

typedef struct {
    int survived;        /* reached past the far edge alive */
    float reward;        /* episode_return at the end (survived) or just before
                          * the killing step (the harness resets on death) */
    int cause;           /* last_death_cause when it died */
} GapOutcome;

/* Hold `speed` exactly (max_speed = speed => the pacing force is zero, so the
 * approach does not accelerate) and run one attempt at the opening.  A jump is
 * issued once, when the runner is `lead` metres from the opening's near edge. */
static GapOutcome run_gap(float speed, float lead, int act) {
    Harness h;
    harness_init(&h, 7);
    clear_track(&h);
    h.env.reward_clear = 1.0f;
    h.env.swipe_cost = 0.0f;
    h.env.max_speed = speed;   /* no pacing force: keep the speed pinned */
    h.env.speed = speed;
    int slot = place_gap(&h);

    float near = h.env.obstacles[slot].s - h.env.obstacles[slot].length * 0.5f;
    float far = h.env.obstacles[slot].s + h.env.obstacles[slot].length * 0.5f;

    GapOutcome out = {-1, 0.0f, TR2_DEATH_NONE};
    int jumped = 0;
    for (int step = 0; step < 400; ++step) {
        suppress_except(&h, slot);
        h.action = TR2_ACT_NONE;
        if (act == GAP_JUMP && !jumped && h.env.grounded
                && h.env.s >= near - lead) {
            h.action = TR2_ACT_JUMP;
            jumped = 1;
        } else if (act == GAP_SLIDE && h.env.grounded
                && h.env.s >= near - 2.0f && !h.env.sliding) {
            h.action = TR2_ACT_SLIDE;
        }
        float reward_before = h.env.episode_return;
        puf_step(&h.env);
        if (h.terminal > 0.5f) {
            out.survived = 0;
            out.reward = reward_before;   /* puf_step reset the env on death */
            out.cause = h.env.last_death_cause;
            return out;
        }
        if (h.env.s > far + 0.5f) {
            out.survived = 1;
            out.reward = h.env.episode_return;
            out.cause = TR2_DEATH_NONE;
            return out;
        }
    }
    out.survived = 0;
    out.reward = h.env.episode_return;
    out.cause = h.env.last_death_cause;
    return out;
}

/* -------------------------------------------------------------------- tests */

/* The generated row must be the shipped `Collision_Gap_A` box plus its
 * FallSaver window, not a stand-in. */
static void test_gap_volume_geometry(void) {
    Harness h;
    harness_init(&h, 3);
    clear_track(&h);
    int slot = place_gap(&h);
    const Tr2Obstacle* o = &h.env.obstacles[slot];
    assert(o->kind == TR2_OBST_GAP);
    assert(o->trigger == 3);                 /* TrackTrigger.MissedTurn */
    assert(o->death_type == TR2_DEATH_FALL);
    assert(fabsf(o->length - 7.385f) < 0.001f);
    assert(fabsf(o->y_top - 0.387f) < 0.001f);
    assert(fabsf(o->y_bottom + 2.221f) < 0.001f);
    /* Collision_FallSaver_A at the far edge: centre 10.4, length 0.5. */
    assert(o->save_length > 0.0f);
    assert(fabsf(o->save_length - 0.5f) < 0.001f);
    assert(fabsf(o->save_offset - 3.653f) < 0.001f);
    printf("ok gap_large_a volume + FallSaver window generated\n");
}

/* Idle and slide into the opening both fall; neither is a jump. */
static void test_idle_and_slide_fall(void) {
    const int acts[] = {GAP_IDLE, GAP_SLIDE};
    for (int i = 0; i < 2; i++) {
        GapOutcome o = run_gap(21.0f, 0.0f, acts[i]);
        assert(!o.survived);
        assert(o.cause == TR2_DEATH_FALL);
    }
    printf("ok idle and slide fall into the gap (cause Fall)\n");
}

/* A timely jump clears the opening at max speed and pays the clear reward. */
static void test_jump_clears_at_max_speed(void) {
    int cleared = 0;
    float best = -1.0f;
    for (float lead = 0.05f; lead <= 4.0f; lead += 0.05f) {
        GapOutcome o = run_gap(21.0f, lead, GAP_JUMP);
        if (o.survived && o.reward > 0.5f) {
            cleared++;
            if (best < 0.0f) best = lead;
        }
    }
    assert(cleared > 0);
    assert(best >= 0.0f && best <= 1.0f);   /* the jump is later than an early hop */
    printf("ok %d leads clear gap_large_a at 21 m/s (first at %.2f m)\n",
            cleared, best);
}

/* The FallSaver is a net, not immunity: a jump that comes down short is
 * rescued (stumble, no clear reward), one that comes down earlier is not. */
static void test_fallsaver_rescues_short_jump(void) {
    int rescued = 0, fell = 0;
    for (float lead = 0.05f; lead <= 6.0f; lead += 0.05f) {
        GapOutcome o = run_gap(21.0f, lead, GAP_JUMP);
        if (o.survived && o.reward < 0.0f) rescued++;
        if (!o.survived) fell++;
    }
    assert(rescued > 0);
    assert(fell > 0);   /* a jump far too early still dies: no permanent immunity */
    printf("ok FallSaver rescued %d short jumps, %d early jumps still fell\n",
            rescued, fell);
}

/* Source physics limit: at low speed the jump spans 0.383 * speed metres, less
 * than the 7.385 m opening, and the FallSaver window is out of reach, so no
 * timing gets across.  (This is why the shipped difficulty ramp keeps
 * difficulty-2 pieces off the track until the runner is fast.) */
static void test_low_speed_is_impossible(void) {
    float speeds[] = {10.0f, 14.0f};
    for (int s = 0; s < 2; s++) {
        int survived = 0;
        for (float lead = 0.0f; lead <= 8.0f; lead += 0.05f) {
            GapOutcome o = run_gap(speeds[s], lead, GAP_JUMP);
            if (o.survived) survived++;
        }
        assert(survived == 0);
        printf("ok gap_large_a unreachable by jump at %.0f m/s\n", speeds[s]);
    }
}

int main(void) {
    test_gap_volume_geometry();
    test_idle_and_slide_fall();
    test_jump_clears_at_max_speed();
    test_fallsaver_rescues_short_jump();
    test_low_speed_is_impossible();
    printf("all templerun2 gap tests passed\n");
    return 0;
}
