/* Temple Run 2 — data-driven C rewrite (RegionID 100 gameplay core).
 *
 * Module layout:
 *   templerun2_track.h      piece selection, path/splines, entity placement
 *   templerun2_runner.h     TRRunnerPawn update, jump/slide/turn, sub-modes
 *   templerun2_collision.h  trigger volumes, death/stumble/shield logic
 *   templerun2_items.h      coins, powerups, scoring
 *   templerun2_observe.h    238-float structured observation
 *   templerun2_render.h     raylib 3D renderer
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;
#include "pufferenv.h"

#include "templerun2_track_data.h"
#include "templerun2_types.h"
#include "templerun2_pacing_data.h"
#include "templerun2_pieces_data.h"
#include "templerun2_colliders_data.h"

#include "templerun2_track.h"
#include "templerun2_collision.h"
#include "templerun2_items.h"
#include "templerun2_runner.h"
#include "templerun2_observe.h"
#include "templerun2_render.h"

static float tr2_cfg(Dict* kwargs, const char* key, float fallback) {
    double v = dict_get(kwargs, key);
    return v == 0.0 ? fallback : (float)v;
}

static void tr2_config(Env* env, Dict* kwargs) {
    env->lane_width = tr2_cfg(kwargs, "lane_width", 0.85f);
    env->track_half_width = tr2_cfg(kwargs, "track_half_width", 1.12f);
    env->player_radius = tr2_cfg(kwargs, "player_radius", 0.44f);
    env->player_height = tr2_cfg(kwargs, "player_height", 1.04f);
    env->slide_height = tr2_cfg(kwargs, "slide_height", 0.64f);
    env->min_speed = tr2_cfg(kwargs, "min_speed", 10.0f);
    env->max_speed = tr2_cfg(kwargs, "max_speed", 21.0f);
    env->force_scale = tr2_cfg(kwargs, "force_scale", 0.1f);
    env->jump_velocity = tr2_cfg(kwargs, "jump_velocity", 10.0f);
    env->gravity = tr2_cfg(kwargs, "gravity", 25.0f);
    env->slam_velocity = tr2_cfg(kwargs, "slam_velocity", 20.0f);
    env->slide_duration_factor = tr2_cfg(kwargs, "slide_duration_factor", 0.07f);
    env->slide_duration_cap = tr2_cfg(kwargs, "slide_duration_cap", 0.75f);
    env->stumble_speed = tr2_cfg(kwargs, "stumble_speed", 0.85f);
    env->lane_lerp = tr2_cfg(kwargs, "lane_lerp", 0.1f);
    env->turn_window = tr2_cfg(kwargs, "turn_window", 7.0f);
    env->magnet_time = tr2_cfg(kwargs, "magnet_time", 10.0f);
    env->shield_time = tr2_cfg(kwargs, "shield_time", 10.0f);
    env->boost_distance = tr2_cfg(kwargs, "boost_distance", 250.0f);
    env->boost_speed = tr2_cfg(kwargs, "boost_speed", 1.55f);
    env->invuln_time = tr2_cfg(kwargs, "invuln_time", 4.0f);
    env->magnet_radius = tr2_cfg(kwargs, "magnet_radius", 34.64f);
    env->coin_red_m = tr2_cfg(kwargs, "coin_red_m", 2500.0f);
    env->coin_blue_m = tr2_cfg(kwargs, "coin_blue_m", 5000.0f);
    env->score_distance = tr2_cfg(kwargs, "score_distance", 1.0f);
    env->score_coin = tr2_cfg(kwargs, "score_coin", 1.0f);
    env->multiplier_m = tr2_cfg(kwargs, "multiplier_m", 0.0f);
    env->multiplier_max = (int)tr2_cfg(kwargs, "multiplier_max", 1.0f);
    env->reward_clear = tr2_cfg(kwargs, "reward_clear", 0.125f);
    env->reward_coin = tr2_cfg(kwargs, "reward_coin", 0.01875f);
    env->swipe_cost = tr2_cfg(kwargs, "swipe_cost", 0.0f);
    env->death_penalty = tr2_cfg(kwargs, "death_penalty", 1.0f);
    env->episode_length = (int)tr2_cfg(kwargs, "episode_length", 200000.0f);
    env->enable_submodes = (int)tr2_cfg(kwargs, "enable_submodes", 1.0f);
    DictItem* decision = dict_find(kwargs, "decision_frames");
    env->decision_frames = decision ? (int)decision->value : 2;
    if (env->decision_frames < 1) env->decision_frames = 1;
    if (env->lane_width <= 0.0f || env->lane_width > 0.85f) env->lane_width = 0.85f;
    if (env->episode_length <= 0) env->episode_length = 200000;
}

static void tr2_env_reset(Env* env) {
    tr2_runner_reset(env);
    tr2_collision_reset(env);
    tr2_items_reset(env);
    tr2_track_reset(env);
    tr2_spawn_ahead(env);
    tr2_observe(env);
}

void puf_init(Env* env, Dict* kwargs) {
    unsigned int rng = env->rng;
    memset(env, 0, sizeof(*env));
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    tr2_config(env, kwargs);
    env->rng = rng != 0 ? rng * 2654435761u : 0x9E3779B9u;
    if (env->rng == 0) env->rng = 0x9E3779B9u;
    tr2_env_reset(env);
}

void puf_reset(Env* env) {
    tr2_env_reset(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "coins", log->coins);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

void puf_step(Env* env) {
    float* actions = env->agents[0].actions;
    float gesture = actions ? actions[0] : TR2_ACT_NONE;
    float command = gesture;
    env->agents[0].actions = &command;
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    for (int frame = 0; frame < env->decision_frames; ++frame) {
        /* A swipe is an event, not a button held throughout the interval. */
        command = frame == 0 ? gesture : TR2_ACT_NONE;
        float prev_s = env->s;
        tr2_runner_step(env);
        if (!env->dead) tr2_collision_step(env, prev_s);
        if (!env->dead) tr2_items_step(env, prev_s);
        tr2_spawn_ahead(env);
        env->tick += 1;
        int draw_intermediate = 1;
        if (env->client) {
            double now = GetTime();
            double* deadline = &env->client->playback_deadline;
            if (*deadline == 0.0 || now - *deadline > 0.25) *deadline = now;
            *deadline += 1.0 / 60.0;
            /* Catch up after a slow asset upload/inference by dropping only
             * an intermediate picture, never a physics step or policy event. */
            draw_intermediate = now <= *deadline;
        }
        if (env->dead || env->tick >= env->episode_length) {
            env->log.perf += env->distance;
            env->log.score += env->score;
            env->log.coins += (float)env->coins;
            env->log.episode_return += env->episode_return;
            env->log.episode_length += (float)env->tick;
            env->log.n += 1.0f;
            float terminal_reward = env->agents[0].rewards[0];
            puf_reset(env);
            env->agents[0].rewards[0] = terminal_reward;
            env->agents[0].terminals[0] = 1.0f;
            break;
        }
        if (env->client && draw_intermediate && frame + 1 < env->decision_frames)
            tr2_render_frame(env, 0);
    }
    env->agents[0].actions = actions;
    tr2_observe(env);
}

void puf_close(Env* env) {
    if (env->client != NULL) {
        free(env->client);
        env->client = NULL;
    }
    if (IsWindowReady()) CloseWindow();
}
