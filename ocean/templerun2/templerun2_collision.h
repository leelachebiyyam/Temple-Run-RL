/* Trigger volumes, death / stumble / shield logic.
 *
 * Ported from recon/collision/COLLISION.md and TRRunnerPawn.OnTriggerEnter.
 *
 * The runner is a trigger sphere (radius `player_radius`, SerializeField
 * sphere centre 0.6 standing / 0.2 ducking) that rides the pawn root, so in
 * the track frame it occupies:
 *
 *      standing on ground   0.16 .. 1.04      (= player_height)
 *      ducking  on ground  -0.24 .. 0.64      (= slide_height)
 *      jumping by h         0.16+h .. 1.04+h
 *
 * i.e. span = [top - 2*radius, top], top = y + player_height (standing) or
 * y + slide_height (sliding).  Each Tr2Obstacle is an axis-aligned trigger box
 * with a longitudinal interval [s - length/2, s + length/2], a lateral interval
 * [lane - half_width, lane + half_width] (lane units) and a vertical interval
 * [y_bottom, y_top] (piece frame, origin on the track surface).  A hit is an
 * overlap on all three axes. Clearability depends on the actual bounds, jump
 * trajectory and lateral position, not just the coarse obstacle kind. Some
 * shipped volumes exceed the simulated jump apex; for example the two
 * jump_or_slide fire volumes can currently be slid under but not jumped over.
 * STUMBLE slows the runner; GAP and ledge collisions cause a fall.
 *
 * GAP volumes are pits, not walls.  The shipped `Collision_Gap_A` box is the
 * *opening*: `OnTriggerEnter`'s MissedTurn branch (0x19653C8) has no
 * unconditional Kill -- it clears the ground flags and hands the runner to
 * Update's ground-dropped-away block (COLLISION.md sec.6.1), whose Kill is the
 * free fall.  So the volume is modelled the way the shipped geometry is
 * authored: while the runner is over it the support under it is the pit floor
 * (`y_bottom`, rectified into the local path frame), the runner falls onto that
 * floor and that fall is the death. A jump still airborne beyond the far edge
 * can land on solid track; a short jump can touch the FallSaver below.
 * This support model approximates the original mesh-ground checks.
 * The source OnTriggerEnter upward-jump gate is evaluated at first contact.
 *
 * The shipped FallSaver (`TrackTrigger.FallSaver`, 6) is that pit's rescue
 * window and is carried on the GAP row (`Tr2Obstacle.save_*`).  OnTriggerEnter's
 * FallSaver branch (0x19658CC) saves a runner that touches it while `_JumpActive`
 * or within `fallSaverPostLandGrace` (0.1 s, 0x238) of a landing: state ->
 * Running, `Stumble(0, false)`, ground restored.  That is what makes a jump
 * that comes down just short of the far edge survive.
 *
 * Heights: an obstacle's `y_bottom`/`y_top` are authored in the owning piece's
 * frame (origin at the piece root) while `env->y` is the runner's height above
 * the local path surface, so the path's local height is subtracted before the
 * two are compared (`tr2_col_path_rel`).  Without it a dipped spline reads
 * wrong: bridge_gap_a's path falls to -1.04 over its own opening, lifting the
 * authored `y_top = -0.423` to +0.617 above the surface there.
 *
 * Every volume above is a *static* trigger BoxCollider in the shipped assets
 * (recon/collision/COLLISION.md sec.8.2); the fire boxes are no exception, so
 * the action that clears them is the one their geometry encodes (jump/duck),
 * not a timed pass.
 *
 * Shield absorbs exactly one hit.  A shielded obstacle hit runs
 * `PowerComp.Invincible(0.2f, draw:false)` then `PowerComp.EndShield(true)`
 * (OnTriggerEnter, COLLISION.md sec.2/5), so it grants only 0.2 s of
 * invulnerability; a shielded *stumble* runs `EndShield(true)` alone and
 * grants none (TRRunnerPawn.Stumble).  Invulnerability (boost power-up grants
 * 4 s, `invuln_time`; god-mode is represented by the same timer) absorbs
 * obstacles and stumbles without being consumed.  Water trigger volumes are
 * *not* shield-blocked: OnTriggerEnter's Water branch tests invincibility
 * only (COLLISION.md sec.2/5); these volumes retain a separate Water entity
 * kind and are handled as trigger-entry events.
 *
 * Falls never consult the shield.  Boost/`invuln_timer` does not stop the
 * free-fall death, but it *does* clear a ledge-edge volume unharmed because
 * `TRRunnerPawn.CheckForLedgeDeath` early-outs on `PowerComp.IsInvincible()`
 * (COLLISION.md sec.5/6.3).
 */
#pragma once

/* PowerComp.Invincible(time) constant used by the shielded-obstacle branch
 * (`mov w1,wzr` around `fmov s0,#0.2`; const @0x799384 = 0.2). */
#define TR2_COL_SHIELD_IFRAMES 0.2f

/* TRRunnerPawn.fallSaverPostLandGrace(0x238); `str w23,[x19,#0x238]` with
 * w23 = 0x3DCCCCCD in Awake (PLAYER.md sec.8).  Read by the FallSaver branch
 * of OnTriggerEnter (0x1965cac: `fcmp (Time.time - LastJumpLandTime), grace`). */
#define TR2_COL_FALL_SAVER_GRACE 0.1f

/* TRRunnerPawn.LastJumpLandTime(0x438) is `str w8,#-0x40800000` = -4.0f in
 * Awake (PLAYER.md sec.8); the FallSaver grace test only counts a landing once
 * the stamp is >= 0. */
#define TR2_COL_LAST_JUMP_LAND_INIT (-4.0f)

/* In a WaterSlide the runner lies on the flume.  Holding the duck/swipe-down
 * gesture goes lower again, under the shipped water-block volumes whose
 * lowest face is 0.228 m above the local flume frame.  A standing runner still
 * hits them, so the duck input remains required. */
#define TR2_COL_WATER_DUCK_TOP 0.20f

/* One physics frame.  TR2_RUNNER_DT is the same value, but templerun2_runner.h
 * is included *after* this header, so the clock is spelled out here. */
#define TR2_COL_DT (1.0f / 60.0f)

/* Tr2Obstacle.saved carries the two per-opening bits a GAP needs (the field is
 * otherwise unused for non-gap kinds):
 *   bit 0  a FallSaver rescued the runner out of this opening
 *   bit 1  the opening was first touched airborne, i.e. `_JumpActive` at
 *          OnTriggerEnter -- the FallSaver's `_JumpActive` test
 */
#define TR2_COL_GAP_SAVED 0x01u
#define TR2_COL_GAP_JUMP 0x02u

/* Time.time equivalent: puf_step advances env->tick once per 60 Hz frame
 * inside the decision loop, so tick * dt is the clock LastJumpLandTime and
 * fallSaverPostLandGrace live on. */
static float tr2_col_now(Env* env) {
    return (float)env->tick * TR2_COL_DT;
}

/* Height of the local path surface under the runner, relative to the owning
 * piece root: an obstacle's y_bottom/y_top are authored in that piece frame,
 * env->y is measured above the surface.  Zero for an unowned volume or a
 * straight path (tr2_seg_height == seg->wy then). */
static float tr2_col_path_rel(Env* env, const Tr2Obstacle* o) {
    if (o->owner_segment_plus_one <= 0) return 0.0f;
    int idx = (o->owner_segment_plus_one - 1) % TR2_SEG_RING;
    const Tr2Segment* seg = &env->segments[idx];
    if (!seg->valid) return 0.0f;
    float d = env->s - seg->s0;
    if (d < 0.0f) d = 0.0f;
    if (d > seg->length) d = seg->length;
    return tr2_seg_height(seg, d) - seg->wy;
}

/* TrackPieceDefinition.IsMandatoryTurn() of the piece under the runner
 * (TP_IsMandatoryTurn.asm -> get_IsMandatoryTurn).  The OnTriggerEnter jump
 * exemption only applies off mandatory turns. */
static int tr2_col_mandatory_turn(Env* env) {
    return env->segments[env->seg_idx % TR2_SEG_RING].turn != 0;
}

static void tr2_collision_die(Env* env, int cause) {
    if (env->dead) return;
    env->dead = 1;
    env->death_cause = cause;
    env->last_death_cause = cause;
    env->agents[0].rewards[0] -= env->death_penalty;
    env->episode_return -= env->death_penalty;
}

/* Obstacle-class hit (everything except a fall).  Invulnerability wins first
 * (boost 4 s / god), otherwise a shield is consumed for one hit and hands out
 * the shipped post-hit i-window: PowerComp.Invincible(0.2f, draw:false). */
static void tr2_collision_hit(Env* env, int cause) {
    if (env->invuln_timer > 0.0f) return;
    if (env->shield_timer > 0.0f) {
        env->shield_timer = 0.0f;
        /* RunnerPowers.Invincible uses max(timeLeft, time). */
        if (env->invuln_timer < TR2_COL_SHIELD_IFRAMES)
            env->invuln_timer = TR2_COL_SHIELD_IFRAMES;
        env->agents[0].rewards[0] -= 1.0f;
        env->episode_return -= 1.0f;
        return;
    }
    tr2_collision_die(env, cause);
}

/* Fall / off-track death: no shield, no invulnerability. */
static void tr2_collision_fall(Env* env, int cause) {
    tr2_collision_die(env, cause);
}

/* TRRunnerPawn.Stumble: a second unprotected stumble within ten seconds is
 * lethal; otherwise the runner slows and locks control for StumbleDuration.
 * IgnoreStumble's state test is bit(state-1) of 0x6F (PLAYER.md sec.7): Dead / Riding / ZipLine /
 * WaterSlide / Luge / DownHillSlide are immune outright (a shield is not even
 * consumed).  Only Falling passes the mask, and the shipped
 * `if (CurrentState != Running) return true` then bails before any slowdown,
 * so only a Running pawn is actually slowed.  Invincibility (boost / god)
 * makes the runner stumble-proof; a shield absorbs one stumble with no
 * i-frames.
 *
 * SetVelocityAfterStumble(0.85): `_CurrentSpeed = max(_CurrentSpeed*0.85,
 * MinSpeed)` first, then `StumbleDuration = 0.75 + 0.25 *
 * clamp01(_CurrentSpeed / MaxSpeed)` (PLAYER.md sec.7). */
static void tr2_collision_stumble(Env* env) {
    switch (env->mode) {
        case TR2_MODE_DEAD:
        case TR2_MODE_RIDING:
        case TR2_MODE_ZIPLINE:
        case TR2_MODE_WATER:
        case TR2_MODE_LUGE:
        case TR2_MODE_DOWNHILL:
            return;
        default:   /* Running, Falling */
            break;
    }
    if (env->invuln_timer > 0.0f) return;
    if (env->shield_timer > 0.0f) {
        env->shield_timer = 0.0f;
        return;
    }
    if (env->stumble_kill_timer < TR2_STUMBLE_MONKEY_TIME) {
        tr2_collision_die(env, TR2_DEATH_EATEN);
        return;
    }
    if (env->mode != TR2_MODE_RUNNING) return;
    env->speed *= env->stumble_speed;
    if (env->speed < env->min_speed) env->speed = env->min_speed;
    float pct = env->max_speed > 0.0f ? env->speed / env->max_speed : 1.0f;
    if (pct > 1.0f) pct = 1.0f;
    env->stumble_timer = 0.75f + 0.25f * pct;
    env->agents[0].rewards[0] -= 0.5f;
    env->episode_return -= 0.5f;
    env->stumble_kill_timer = 0.0f;
}

static void tr2_collision_reset(Env* env) {
    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) {
        Tr2Obstacle* o = &env->obstacles[i];
        o->s = 0.0f;
        o->length = 0.0f;
        o->lane = 0.0f;
        o->half_width = 0.0f;
        o->y_bottom = 0.0f;
        o->y_top = 0.0f;
        o->phase = 0.0f;
        o->kind = 0;
        o->variant = 0;
        o->trigger = 0;
        o->death_type = 0;
        o->active = 0;
        o->passed = 0;
        o->trigger_entered = 0;
        o->owner_segment_plus_one = 0;
        o->save_offset = 0.0f;
        o->save_length = 0.0f;
        o->save_half_width = 0.0f;
        o->save_y_bottom = 0.0f;
        o->save_y_top = 0.0f;
        o->saved = 0;
    }
    env->last_hit_kind = 0;
    /* Ground support is solid until a pit says otherwise; LastJumpLandTime
     * starts negative so the FallSaver grace is off until a landing stamps
     * it (TRRunnerPawn.Awake, PLAYER.md sec.8). */
    env->floor_y = 0.0f;
    env->last_jump_land_time = TR2_COL_LAST_JUMP_LAND_INIT;
}

static void tr2_collision_step(Env* env, float prev_s) {
    /* Runner pose span, in the local track frame (see header comment). */
    int ducking = env->sliding && env->grounded;
    float pose_top = ducking ? env->slide_height : env->player_height;
    if (ducking && env->mode == TR2_MODE_WATER)
        pose_top = TR2_COL_WATER_DUCK_TOP;
    float top = env->y + pose_top;
    float bottom = top - 2.0f * env->player_radius;

    /* Support under the runner this frame: solid track unless a GAP pit below
     * takes it away.  tr2_runner_step reads it on the next frame's
     * gravity/landing block, which is what turns "over the opening" into a
     * fall. */
    env->floor_y = 0.0f;
    int on_turn = tr2_col_mandatory_turn(env);

    for (int i = 0; i < TR2_MAX_OBSTACLES; i++) {
        Tr2Obstacle* o = &env->obstacles[i];
        if (!o->active) continue;

        float o0 = o->s - o->length * 0.5f;
        float o1 = o->s + o->length * 0.5f;

        if (o1 < prev_s - 0.2f) {
            o->active = 0;
            continue;
        }
        if (o->passed) continue;
        if (o0 > env->s + 0.2f) continue;

        int is_gap = (o->kind == TR2_OBST_GAP);
        int is_ledge = (o->kind == TR2_OBST_LEDGE_LEFT
                || o->kind == TR2_OBST_LEDGE_RIGHT);

        /* Lateral overlap, metres: |px - lane| * lane_width vs volume half
         * width plus the runner sphere radius. */
        float lat_m = fabsf(env->px - o->lane) * env->lane_width;
        float half_m = o->half_width * env->lane_width;
        int lateral = lat_m < half_m + env->player_radius;

        /* Swept longitudinal test against the frame's travel [prev_s, s]. */
        int longitudinal = (o0 <= env->s) && (o1 >= prev_s);

        if (is_gap) {
            /* --- MissedTurn / GAP: a pit, not a wall (header comment) ----- */
            float path_rel = tr2_col_path_rel(env, o);
            float gap_floor = o->y_bottom - path_rel;
            /* the runner's body is inside the opening this frame (the swept
             * test below also holds on the two frames either side of it) */
            int over_now = (o0 <= env->s) && (env->s <= o1);

            if (!lateral || !longitudinal || env->s > o1) {
                if (env->s > o1) {
                    /* Left the opening.  Below the rim means it went in. */
                    if (lateral && !(o->saved & TR2_COL_GAP_SAVED)
                            && env->y < 0.0f) {
                        int cause = o->death_type;
                        if (cause == TR2_DEATH_NONE) cause = TR2_DEATH_FALL;
                        env->last_hit_kind = o->kind;
                        tr2_collision_fall(env, cause);
                        return;
                    }
                    o->passed = 1;
                    if (!(o->saved & TR2_COL_GAP_SAVED)) {
                        env->agents[0].rewards[0] += env->reward_clear;
                        env->episode_return += env->reward_clear;
                    }
                }
                continue;
            }

            /* OnTriggerEnter fires once per overlap (Unity), so the gate is
             * evaluated on the frame the runner's body first enters the box:
             * `_JumpActive && !IsMandatoryTurn() && IsJumpingUpPhase`
             * (0x1965438..5458) -> return untouched.  Any other first contact
             * clears the ground flags (`strb wzr,[x19,#0x244]`, 0x1965464),
             * i.e. the runner drops.  Bit 1 records that first contact for
             * the FallSaver's `_JumpActive` test. */
            int entering = over_now && lateral && (prev_s < o0);
            if (entering) {
                o->saved = (unsigned char)((o->saved & TR2_COL_GAP_SAVED)
                        | (env->grounded ? 0 : TR2_COL_GAP_JUMP));
                int rising = (o->saved & TR2_COL_GAP_JUMP)
                        && env->vy > 0.0f && !on_turn;
                if (!rising) {
                    env->grounded = 0;
                    env->last_hit_kind = o->kind;
                }
            }

            /* FallSaver (0x19658CC): `inGrace || _JumpActive` saves a runner
             * whose collider touches the rescue window.  The box sits at the
             * far edge, so this is a jump coming down just short of solid
             * ground; a runner that ran or slid into the opening is not
             * jumping and is not saved by it. */
            if (over_now && !(o->saved & TR2_COL_GAP_SAVED)
                    && o->save_length > 0.0f) {
                float sv = o->s + o->save_offset;
                float sv0 = sv - o->save_length * 0.5f;
                float sv1 = sv + o->save_length * 0.5f;
                int sv_lat = lat_m
                        < o->save_half_width * env->lane_width + env->player_radius;
                int sv_vert = (bottom < o->save_y_top - path_rel)
                        && (top > o->save_y_bottom - path_rel);
                int in_grace = env->last_jump_land_time >= 0.0f
                        && (tr2_col_now(env) - env->last_jump_land_time)
                                <= TR2_COL_FALL_SAVER_GRACE;
                if (((o->saved & TR2_COL_GAP_JUMP) || in_grace)
                        && sv_lat && sv_vert
                        && (sv0 <= env->s) && (sv1 >= env->s)) {
                    /* Stumble(0, false) + IsOnGround = true + Running. */
                    o->saved |= TR2_COL_GAP_SAVED;
                    env->grounded = 1;
                    env->y = 0.0f;
                    env->vy = 0.0f;
                    env->floor_y = 0.0f;   /* the window is solid ground */
                    env->last_hit_kind = o->kind;
                    tr2_collision_stumble(env);
                    if (env->dead) return;
                }
            }

            /* The opening has no support: hand the pit floor to
             * tr2_runner_step and let the fall resolve on it (Update's
             * ground-dropped-away Kill, COLLISION.md sec.6.1).  A FallSaver
             * rescue cancels it for the rest of this crossing. */
            if (over_now && !(o->saved & TR2_COL_GAP_SAVED)
                    && gap_floor < 0.0f) {
                if (gap_floor < env->floor_y) env->floor_y = gap_floor;
                if (env->y <= gap_floor + 1e-3f) {
                    int cause = o->death_type;
                    if (cause == TR2_DEATH_NONE) cause = TR2_DEATH_FALL;
                    env->last_hit_kind = o->kind;
                    tr2_collision_fall(env, cause);
                    return;
                }
            }
            continue;
        }

        /* Vertical overlap of [bottom, top] with [y_bottom, y_top]. */
        int vertical = (bottom < o->y_top) && (top > o->y_bottom);
        if (o->trigger == TR2_TRIG_WATER) {
            float path_rel = tr2_col_path_rel(env, o);
            int water_overlap = bottom < o->y_top - path_rel
                && top > o->y_bottom - path_rel;
            if (!lateral || !longitudinal || !water_overlap)
                o->trigger_entered = 0;
            if (!o->trigger_entered && lateral && longitudinal && water_overlap) {
                o->trigger_entered = 1;
                /* OnTriggerEnter 0x19652DC: rising jumps on non-turn pieces,
                 * riding and invincibility bypass Water. The exemption is an
                 * entry event; descending later inside the same box is not
                 * a second collision. Shields do not protect against water. */
                int rising = !env->grounded && env->vy > 0.0f && !on_turn;
                if (!rising && env->mode != TR2_MODE_RIDING && env->invuln_timer <= 0.0f) {
                    env->last_hit_kind = o->kind;
                    tr2_collision_die(env, o->death_type);
                    return;
                }
            }
            if (env->s > o1) {
                o->passed = 1;
                env->agents[0].rewards[0] += env->reward_clear;
                env->episode_return += env->reward_clear;
            }
            continue;
        }

        if (lateral && longitudinal && vertical) {
            o->passed = 1;
            if (o->kind == TR2_OBST_STUMBLE) {
                env->last_hit_kind = o->kind;
                tr2_collision_stumble(env);
            } else if (is_ledge) {
                /* Ledge-edge falls record the volume's own DeathType. */
                int cause = o->death_type;
                if (cause == TR2_DEATH_NONE) cause = TR2_DEATH_FALL;
                env->last_hit_kind = o->kind;
                /* CheckForLedgeDeath early-outs on PowerComp.IsInvincible(),
                 * so boost/god clear a ledge-edge volume unharmed
                 * (COLLISION.md sec.5/6.3). */
                if (env->invuln_timer <= 0.0f) tr2_collision_fall(env, cause);
            } else {
                int cause = o->death_type;
                if (cause == TR2_DEATH_NONE) cause = TR2_DEATH_SCENERYROCK;
                env->last_hit_kind = o->kind;
                tr2_collision_hit(env, cause);
            }
            if (env->dead) return;
        } else if (env->s > o1) {
            /* Cleared the volume without a hit. */
            o->passed = 1;
            env->agents[0].rewards[0] += env->reward_clear;
            env->episode_return += env->reward_clear;
        }
    }
}
