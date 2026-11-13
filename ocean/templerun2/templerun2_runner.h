/* templerun2_runner.h — TRRunnerPawn update: speed/force integration, lane
 * servo, jump / slam / slide, stumble timer, mandatory turns and sub-mode
 * pieces.
 *
 * Owner: runner module.  Ported from recon/il2cpp/asm/PLAYER.md (§1-§8) and
 * recon/decompiler/out/Assembly-CSharp_decompiled/TRRunnerPawn.cs.
 *
 * Integration order mirrors TRRunnerPawn.Update (0x19614DC):
 *
 *   1  read action (the input controller's AddForce lands in the same step)
 *   2  lane servo        AddOffsetFromCenterOfTrack((target - px) * 0.1)
 *   3  sub-mode          CurrentState from the piece under the runner
 *   4  force -> speed    PlayerVelocity += PlayerForce * dt
 *   5  advance s         PositionOnTrack += direction * speed * dt  (skipped
 *                        while OffTrack; the KillTimer countdown runs instead)
 *   6  gravity / jump    CurrentJumpVelocity += (onGround ? 1 : 2) * g * dt
 *   7  piece crossing    mandatory-turn validation at the piece boundary
 *                        (a missed turn goes OffTrack with a 1.0 s KillTimer)
 *   8  death causes      water drift, fall; KillTimer expiry on off-track
 *
 * StumbleKillTimer accumulates every frame; the collision module reads it in
 * Stumble() to decide the monkey (Eaten) death (TR2_STUMBLE_MONKEY_TIME).
 *
 * This header is included after templerun2_collision.h (see templerun2.h) and
 * reuses its fall/death entry points, so a fall bypasses shield / boost
 * exactly as in the shipped game.  All functions are static; the step is a
 * fixed 1/60 s because Env carries no dt and the shipped game runs at 60 Hz.
 */
#pragma once

#include <math.h>

#define TR2_RUNNER_DT (1.0f / 60.0f)

/* TRRunnerPawn prefab MaxJumpVelocity=10 / MinJumpVelocity=8 (PLAYER.md §8). */
#define TR2_RUNNER_JUMP_MIN 8.0f

/* Height of the runner's root below the zipline rope while hanging.  The rope
 * itself is drawn by the renderer; physics keeps the runner attached at this
 * clearance until the zipline piece ends, then normal gravity drops it. */
#define TR2_RUNNER_ZIPLINE_HANG 1.70f

/* A buffered turn command stays valid this far past the piece boundary (m). */
#define TR2_RUNNER_TURN_GRACE 1.0f

/* StumbleKillTimer is Reset/Resurrect'd to 2 x StumbleDeathByMonkeyTime
 * (`fadd s0,s0,s0`, 0x196BA48 / 0x19687C8 / 0x196CD10), so a fresh run starts
 * outside the monkey window; only a repeat stumble inside TR2_STUMBLE_MONKEY_TIME
 * seconds of the previous one is lethal. */
#define TR2_RUNNER_STUMBLE_SAFE (2.0f * TR2_STUMBLE_MONKEY_TIME)

/* UpdateKillTimer (0x1964ED4, COLLISION.md sec.6.2): a missed turn sets
 * KillTimer = 1.0 s (OnTrackPieceTurnEvent), the runner enters Falling once it
 * has been off the track for more than 0.25 s, and the run ends when the
 * KillTimer reaches 0. */
#define TR2_RUNNER_KILL_TIMER 1.0f
#define TR2_RUNNER_FALL_DELAY 0.25f

static float tr2_runner_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static int tr2_runner_clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* TRTrackPacing.RunnerSpeed force buckets (PLAYER.md §1.7).  CalcRunnerForce
 * returns ForceToAdd * 0.1 and Update does PlayerVelocity += Force*dt, so
 * d(speed)/dt is the bucket * force_scale.  The bucket test is a strict `>`
 * on each threshold, so the percentages asked for are exact. */
static float tr2_runner_force(float ratio) {
    if (ratio < 0.50f) return 30.0f;
    if (ratio < 0.65f) return 10.0f;
    if (ratio < 0.85f) return 1.5f;
    if (ratio < 1.00f) return 0.5f;
    return 0.0f;
}

/* Jump() jump table (§2): Riding / ZipLine / Falling / Dead return false, a
 * stumble (active or buffered) blocks the jump, WaterSlide is handled by the
 * caller (it stops the under-water slide instead of jumping). */
static int tr2_runner_can_jump(Env* env) {
    if (env->stumble_timer > 0.0f) return 0;
    switch (env->mode) {
        case TR2_MODE_RIDING:
        case TR2_MODE_ZIPLINE:
        case TR2_MODE_FALLING:
        case TR2_MODE_DEAD:
            return 0;
        default:
            return 1;
    }
}

/* Jump() strength term `s9`: 1.0 running, 0.75 for the Luge / DownHillSlide
 * table entries (Direction.y >= 0 branch; the negative branch only differs
 * when the runner is already dropping). */
static float tr2_runner_jump_strength(Env* env) {
    if (env->mode == TR2_MODE_LUGE || env->mode == TR2_MODE_DOWNHILL)
        return 0.75f;
    return 1.0f;
}

/* Duck() slide duration per state (§4). */
static float tr2_runner_slide_duration(Env* env) {
    switch (env->mode) {
        case TR2_MODE_RIDING:
        case TR2_MODE_WATER:
        case TR2_MODE_LUGE:
            return 1.0f;
        case TR2_MODE_DOWNHILL:
            return 0.75f;
        default:
            return fminf(env->speed * env->slide_duration_factor,
                    env->slide_duration_cap);
    }
}

/* TRGameRunner loads Stats.ScoreMultiplier once at run start from
 * GameProfile.GetScoreMultiplier(): base 10 plus objectives/artifacts.  It
 * does not ramp with distance, so the run keeps a single constant. */
static void tr2_runner_update_multiplier(Env* env) {
    env->multiplier = env->score_multiplier > 0 ? env->score_multiplier : 10;
}

/* The piece under the runner selects the state (Tr2Segment.mode).  A dropped
 * runner keeps Falling instead of being re-tagged by the piece it left. */
static void tr2_runner_apply_mode(Env* env) {
    if (env->off_track) return;   /* off-track owns the Falling state */
    if (!env->grounded && env->y < -1.0f) {
        env->mode = TR2_MODE_FALLING;
        env->mode_timer = 0.0f;
        return;
    }
    unsigned char want = TR2_MODE_RUNNING;
    Tr2Segment* seg = &env->segments[env->seg_idx % TR2_SEG_RING];
    if (env->enable_submodes && seg->mode != TR2_MODE_RUNNING)
        want = seg->mode;
    if (want != env->mode) {
        env->mode = want;
        env->mode_timer = 0.0f;
        if (want == TR2_MODE_ZIPLINE) {
            /* Jump() to catch the rope: the shipped zipline attach only
             * happens while airborne.  A grounded runner that walks into the
             * zipline piece misses the rope and falls. */
            if (env->grounded) {
                tr2_collision_fall(env, TR2_DEATH_FALL);
                return;
            }
            env->y = TR2_RUNNER_ZIPLINE_HANG;
            env->vy = 0.0f;
            env->grounded = 0;
        }
        if (want != TR2_MODE_RUNNING && want != TR2_MODE_FALLING) {
            float remain = seg->s0 + seg->length - env->s;
            float v = env->speed > env->min_speed ? env->speed : env->min_speed;
            env->mode_timer = (remain > 0.0f && v > 0.0f) ? remain / v : 0.0f;
        }
    } else if (env->mode != TR2_MODE_RUNNING) {
        env->mode_timer -= TR2_RUNNER_DT;
        if (env->mode_timer < 0.0f) env->mode_timer = 0.0f;
    }
}

/* Mandatory turns are validated when the runner crosses into the turn piece:
 * the swipe must already be buffered (Turn() ran inside the approach window).
 * An invincible runner (boost / god) and sub-mode pieces auto-steer, exactly
 * like AutoTurn; a runner that neither auto-steers nor buffered the right
 * swipe goes OffTrack for a 1.0 s KillTimer instead of dying outright. */
static void tr2_runner_cross_pieces(Env* env) {
    while (env->seg_idx < env->seg_gen) {
        Tr2Segment* cur = &env->segments[env->seg_idx % TR2_SEG_RING];
        if (env->s < cur->s0 + cur->length) break;
        int nxt = env->seg_idx + 1;
        Tr2Segment* ns = &env->segments[nxt % TR2_SEG_RING];
        if (ns->turn != 0) {
            /* OnTrackPieceTurnEvent auto-selects the turn while the runner is
             * invincible (PowerComp.IsInvincible(), COLLISION.md sec.5/6.4),
             * i.e. env->invuln_timer -- not the boost distance. */
            int auto_steer = env->invuln_timer > 0.0f
                || (env->enable_submodes && env->mode != TR2_MODE_RUNNING);
            if (!auto_steer && env->turn_input != ns->turn) {
                /* MISSED TURN (COLLISION.md sec.6.2/6.4): go off-track with a
                 * 1.0 s KillTimer instead of dying on the piece boundary. */
                env->off_track = 1;
                env->off_track_time = 0.0f;
                env->kill_timer = TR2_RUNNER_KILL_TIMER;
                return;
            }
            env->turn_input = 0;          /* consumed at piece entry */
            env->turn_input_timer = 0.0f;
        }
        env->seg_idx = nxt;
        tr2_track_generate(env);
    }
}

static void tr2_runner_reset(Env* env) {
    env->s = 0.0f;
    env->px = 0.0f;
    env->lane = TR2_CENTER_LANE;
    env->target_lane = TR2_CENTER_LANE;
    env->y = 0.0f;
    env->vy = 0.0f;
    env->grounded = 1;
    env->sliding = 0;
    env->slide_timer = 0.0f;
    env->stumble_timer = 0.0f;
    env->stumble_kill_timer = TR2_RUNNER_STUMBLE_SAFE;
    env->off_track = 0;
    env->kill_timer = 0.0f;
    env->off_track_time = 0.0f;
    env->speed = env->min_speed;
    env->mode = TR2_MODE_RUNNING;
    env->mode_timer = 0.0f;
    env->turn_input = 0;
    env->turn_input_timer = 0.0f;
    env->lane_cooldown = 0.0f;
    env->last_action = TR2_ACT_NONE;
    env->dead = 0;
    env->death_cause = TR2_DEATH_NONE;
    env->distance = 0.0f;
    env->score = 0.0f;
    env->score_remainder = 0.0f;
    env->coins_for_bonus = 0;
    env->coins_for_meter = 0;
    env->multiplier = env->score_multiplier > 0 ? env->score_multiplier : 10;
    env->episode_return = 0.0f;
    env->tick = 0;
    if (env->agents[0].rewards != NULL) env->agents[0].rewards[0] = 0.0f;
}

static void tr2_runner_step(Env* env) {
    if (env->dead) return;

    float* actions = env->agents[0].actions;
    int raw = TR2_ACT_NONE;
    if (actions != NULL) {
        raw = (int)lroundf(actions[0]);
        /* A negative value (legacy callers) means "no swipe"; the harness and
         * the renderer both use TR2_ACT_NONE == 0. */
        if (raw < 0) raw = TR2_ACT_NONE;
        else raw = tr2_runner_clampi(raw, 0, TR2_ACTIONS - 1);
    }
    /* The policy samples one of five classes: TR2_ACT_NONE leaves the runner
     * alone (no lane change, no jump, no slide), the other four are the
     * physical gestures.  Repeated lane swipes stay rate-limited by
     * lane_cooldown below. */
    int action = raw;
    env->last_action = raw;

    /* Anti-spam shaping (config `swipe_cost`, 0 disables it exactly): each
     * step that spends a real gesture input pays the cost; the idle action
     * never does.  It is a scalar leaf penalty, not an action gate -- every
     * gesture stays selectable. */
    if (action != TR2_ACT_NONE && env->swipe_cost != 0.0f) {
        env->agents[0].rewards[0] -= env->swipe_cost;
        env->episode_return -= env->swipe_cost;
    }

    /* Turn() sets TimeSinceLastTurn = 0 and _JustTurned; the buffered swipe
     * expires once the approach window has been travelled. */
    if (env->turn_input_timer > 0.0f) {
        env->turn_input_timer -= TR2_RUNNER_DT;
        if (env->turn_input_timer <= 0.0f) {
            env->turn_input_timer = 0.0f;
            env->turn_input = 0;
        }
    }
    /* UpdateActions order: a slide started by the action below is not yet
     * ticked, so it lasts the full min(speed*factor, cap). */
    if (env->sliding) {
        env->slide_timer -= TR2_RUNNER_DT;
        if (env->slide_timer <= 0.0f) {
            env->sliding = 0;
            env->slide_timer = 0.0f;
        }
    }
    if (env->stumble_timer > 0.0f) {
        env->stumble_timer -= TR2_RUNNER_DT;
        if (env->stumble_timer < 0.0f) env->stumble_timer = 0.0f;
    }
    /* StumbleKillTimer += Time.deltaTime every Update frame (0x19624D4); a
     * normal Stumble() zeroes it, re-entering the monkey window. */
    env->stumble_kill_timer += TR2_RUNNER_DT;
    if (env->lane_cooldown > 0.0f) env->lane_cooldown -= TR2_RUNNER_DT;

    /* --- input -------------------------------------------------------- */
    Tr2Segment* next = &env->segments[(env->seg_idx + 1) % TR2_SEG_RING];
    float to_next = next->s0 - env->s;
    if (action == TR2_ACT_NONE) {
        /* Idle: no gesture this step.  The track is left alone, the runner
         * keeps the current lane / jump / slide state. */
    } else if (action == TR2_ACT_LEFT || action == TR2_ACT_RIGHT) {
        int dir = (action == TR2_ACT_LEFT) ? -1 : 1;
        int turn_ahead = next->turn == dir
            && to_next <= env->turn_window
            && to_next >= -TR2_RUNNER_TURN_GRACE;
        if (!turn_ahead && next->turn == dir && to_next > env->turn_window
                && tr2_near_turn(env, env->s))
            turn_ahead = 1;   /* helper confirms the swipe lands in the zone */
        if (turn_ahead) {
            float v = env->speed > env->min_speed ? env->speed : env->min_speed;
            env->turn_input = dir;
            env->turn_input_timer = v > 0.0f
                ? env->turn_window / v + 0.1f : 0.5f;
        } else if (env->mode != TR2_MODE_ZIPLINE && env->lane_cooldown <= 0.0f) {
            /* Swipes have a short debounce in the shipped controller; without
             * it a noisy policy visibly oscillates between lanes. */
            env->target_lane = tr2_runner_clampi(env->target_lane + dir,
                    TR2_LANE_MIN, TR2_LANE_MAX);
            env->lane_cooldown = 0.12f;
        }
    } else if (action == TR2_ACT_JUMP) {
        if (env->mode == TR2_MODE_WATER) {
            /* Jump() -> StopUnderWaterSliding(true): no jump, ends the slide. */
            env->sliding = 0;
            env->slide_timer = 0.0f;
        } else if (env->grounded && tr2_runner_can_jump(env)) {
            float vmax = env->jump_velocity;
            float vmin = fminf(TR2_RUNNER_JUMP_MIN, vmax);
            float v = tr2_runner_clampf(env->speed, vmin, vmax);
            env->vy = v * tr2_runner_jump_strength(env);
            env->grounded = 0;
            env->sliding = 0;        /* Jump() calls StopDucking() */
            env->slide_timer = 0.0f;
        }
    } else if (action == TR2_ACT_SLIDE) {
        /* Duck() and SlamDown() both bail out in ZipLine; SlamDown() further
         * only fires while airborne and *clamps* CurrentJumpVelocity to -20
         * (fmin, so an already-faster drop keeps its speed), it does not
         * assign. */
        if (env->mode != TR2_MODE_ZIPLINE) {
            if (!env->grounded) {
                env->vy = fminf(env->vy, -env->slam_velocity);  /* SlamDown() */
            } else if (!env->sliding && env->stumble_timer <= 0.0f) {
                env->sliding = 1;
                env->slide_timer = tr2_runner_slide_duration(env);
            }
        }
    }

    /* --- lane servo: (target - px) * lane_lerp (0.1) ------------------ */
    {
        float target = (float)env->target_lane - 1.0f;
        if (env->mode == TR2_MODE_ZIPLINE) target = 0.0f;
        env->px += (target - env->px) * env->lane_lerp;
        if (fabsf(target - env->px) < 0.001f) env->px = target;
        env->px = tr2_runner_clampf(env->px, -1.0f, 1.0f);
        env->lane = tr2_runner_clampi((int)lroundf(env->px + 1.0f),
                TR2_LANE_MIN, TR2_LANE_MAX);
    }

    /* --- sub-mode from the piece under the runner --------------------- */
    tr2_runner_apply_mode(env);
    if (env->dead) return;

    /* --- force -> speed (PlayerVelocity integration) ------------------ */
    if (env->grounded && !env->sliding && env->mode != TR2_MODE_ZIPLINE) {
        float ratio = env->max_speed > 0.0f
            ? env->speed / env->max_speed : 1.0f;
        float force = tr2_runner_force(ratio);
        if (force > 0.0f)
            env->speed += force * env->force_scale * TR2_RUNNER_DT;
        if (env->speed > env->max_speed) env->speed = env->max_speed;
    }
    if (env->speed < env->min_speed) env->speed = env->min_speed;

    /* --- advance along the path / off-track countdown ----------------- */
    if (env->off_track) {
        /* UpdateKillTimer (0x1964ED4, COLLISION.md sec.6.2): the along-track
         * follow (Update steps 16/17 else-branch) is skipped, so s / distance
         * / score freeze; the runner enters Falling 0.25 s after leaving the
         * track and the run ends when KillTimer reaches 0. */
        env->off_track_time += TR2_RUNNER_DT;
        env->kill_timer -= TR2_RUNNER_DT;
        if (env->mode != TR2_MODE_FALLING
                && env->off_track_time > TR2_RUNNER_FALL_DELAY) {
            env->mode = TR2_MODE_FALLING;
            env->mode_timer = 0.0f;
        }
        if (env->kill_timer <= 0.0f) {
            env->kill_timer = 0.0f;
            tr2_collision_fall(env, TR2_DEATH_FALL);
            return;
        }
    } else {
        float boost = env->boost_distance_left > 0.0f ? env->boost_speed : 1.0f;
        float move_speed = env->speed * boost;
        float ds = move_speed * TR2_RUNNER_DT;
        env->s += ds;
        env->distance += ds;

        tr2_runner_update_multiplier(env);
        {
            /* UpdateStats: whole metres are scored at the run multiplier and
             * the fractional remainder carries to the next frame. */
            env->score_remainder += ds * env->score_distance;
            int metres = (int)env->score_remainder;
            if (metres > 0) {
                env->score_remainder -= (float)metres;
                env->score += (float)(env->multiplier * metres);
            }
        }

        if (env->boost_distance_left > 0.0f) {
            env->boost_distance_left -= ds;
            if (env->boost_distance_left < 0.0f) env->boost_distance_left = 0.0f;
        }
    }

    /* The ground can drop away inside a gap. Do not clamp a falling runner
     * back onto the track plane; land only while crossing real support. */
    if (env->mode == TR2_MODE_ZIPLINE) {
        /* Attached to the rope: no gravity until the piece ends. */
        env->y = TR2_RUNNER_ZIPLINE_HANG;
        env->vy = 0.0f;
        env->grounded = 0;
    } else if (env->grounded) {
        if (env->y > env->floor_y) {
            env->grounded = 0;
            env->vy = 0.0f;
        } else {
            env->y = env->floor_y;
            env->vy = 0.0f;
        }
    } else {
        float y0 = env->y;
        env->vy -= env->gravity * 2.0f * TR2_RUNNER_DT;
        env->y += env->vy * TR2_RUNNER_DT;
        if (env->y <= env->floor_y && y0 >= env->floor_y) {
            env->y = env->floor_y;
            env->vy = 0.0f;
            env->grounded = 1;
            env->last_jump_land_time = (float)env->tick * TR2_RUNNER_DT;
        }
    }

    /* --- piece crossing: mandatory-turn validation at piece start ----- */
    if (!env->off_track) tr2_runner_cross_pieces(env);
    if (env->dead) return;

    /* --- death causes ------------------------------------------------- */
    if (env->mode == TR2_MODE_WATER && fabsf(env->px) > 1.0f) {
        tr2_collision_fall(env, env->px < 0.0f
                ? TR2_DEATH_WATERLEFT : TR2_DEATH_WATERRIGHT);
        return;
    }
    if (env->y < -6.0f) tr2_collision_fall(env, TR2_DEATH_FALL);
}
