/* Coins, powerups and item scoring.
 *
 * Owns coin valuation/pickup, the six powerups, the coin/gem counters and the
 * magnet / shield / invulnerability timers.  The runner adds *distance* score;
 * this module only ever adds item score.
 *
 * Faithful numbers (recon/README.md, device/Config/Powers.txt,
 * unity_assets/data/key/BonusItemProtoData.json):
 *   coins        1/2/3 by the shared tr2_coin_value() (2x from 2500 m,
 *                3x from 5000 m; DistanceToChangeDouble/TripleCoins)
 *   mega coin    50 (BonusItemProtoData.MegaCoinValue)
 *   magnet       10 s, radius sqrt(1200) = 34.64 (VacuumRangeRadius2)
 *   shield       10 s, absorbs one hit
 *   boost        250 m, grants 4 s of invulnerability
 *   gem          +1 gem
 *   coin bonus   +50 coins
 *   score bonus  +500 points
 */
#pragma once

#include <math.h>

/* Fixed 60 Hz simulation step, matching TR2_RUNNER_DT; TR2_DT is honored if a
 * data header earlier in the include chain publishes it. */
#ifdef TR2_DT
#define TR2_ITEMS_DT TR2_DT
#else
#define TR2_ITEMS_DT (1.0f / 60.0f)
#endif

#define TR2_COIN_MEGA 50
#define TR2_COIN_BONUS_COINS 50
#define TR2_SCORE_BONUS_POINTS 500.0f


/* Coin worth.  1/2/3 is what the generator stamped into coin->value using the
 * shared distance helper tr2_coin_value() (pacing_data.h; 2x from 2500 m, 3x
 * from 5000 m); an unstamped slot falls back to that same helper.  A pooled
 * value of TR2_COIN_MEGA (50) marks a mega coin outright. */
static int tr2_item_value(const Tr2Coin* coin) {
    if (coin->value >= TR2_COIN_MEGA) return TR2_COIN_MEGA;
    if (coin->value >= 1 && coin->value <= 3) return coin->value;
    return tr2_coin_value(coin->s);
}

static void tr2_apply_powerup(Env* env, int kind) {
    switch (kind) {
    case TR2_PU_MAGNET:
        env->magnet_timer = env->magnet_time;
        break;
    case TR2_PU_BOOST:
        env->boost_distance_left = env->boost_distance;
        /* Boost grants the shipped 4 s invincibility, independent of distance. */
        if (env->invuln_timer < env->invuln_time) env->invuln_timer = env->invuln_time;
        break;
    case TR2_PU_SHIELD:
        env->shield_timer = env->shield_time;
        break;
    case TR2_PU_GEM:
        env->gems += 1;
        break;
    case TR2_PU_COIN_BONUS:
        env->coins += TR2_COIN_BONUS_COINS;
        break;
    case TR2_PU_SCORE_BONUS:
        env->score += TR2_SCORE_BONUS_POINTS;
        break;
    default:
        break;
    }
}

/* The runner's coin multiplier (TRGameRunner.AddCoinsToScore) scales both the
 * coin count and the score. */
static void tr2_collect_coin(Env* env, Tr2Coin* coin) {
    int value = tr2_item_value(coin);
    int mult = env->multiplier > 0 ? env->multiplier : 1;
    coin->active = 0;
    env->coins += value;
    env->score += env->score_coin * (float)value * (float)mult;
    float reward = env->reward_coin * (float)value;
    env->agents[0].rewards[0] += reward;
    env->episode_return += reward;
}

static void tr2_items_reset(Env* env) {
    env->coins = 0;
    env->gems = 0;
    env->multiplier = 1;
    env->magnet_timer = 0.0f;
    env->shield_timer = 0.0f;
    env->boost_distance_left = 0.0f;
    env->invuln_timer = 0.0f;
    for (int i = 0; i < TR2_MAX_COINS; i++) {
        env->coin_pool[i].active = 0;
        env->coin_pool[i].value = 0;
    }
    for (int i = 0; i < TR2_MAX_POWERUPS; i++) {
        env->powerups[i].active = 0;
    }
}

static void tr2_items_step(Env* env, float prev_s) {
    /* Timers tick every step.  Shield *consumption* is the collision module's
     * job (it clears shield_timer and grants i-frames); the runner drains
     * boost_distance_left as it moves, since boost is measured in metres. */
    if (env->magnet_timer > 0.0f) {
        env->magnet_timer -= TR2_ITEMS_DT;
        if (env->magnet_timer < 0.0f) env->magnet_timer = 0.0f;
    }
    if (env->shield_timer > 0.0f) {
        env->shield_timer -= TR2_ITEMS_DT;
        if (env->shield_timer < 0.0f) env->shield_timer = 0.0f;
    }
    if (env->invuln_timer > 0.0f) {
        env->invuln_timer -= TR2_ITEMS_DT;
        if (env->invuln_timer < 0.0f) env->invuln_timer = 0.0f;
    }

    float player_top = env->y + (env->sliding ? env->slide_height : env->player_height);
    float magnet_r2 = env->magnet_radius * env->magnet_radius;
    int magnet_on = env->magnet_timer > 0.0f;

    for (int i = 0; i < TR2_MAX_COINS; i++) {
        Tr2Coin* coin = &env->coin_pool[i];
        if (!coin->active) continue;
        if (coin->s < prev_s - 1.0f) {   /* swept past without being reached */
            coin->active = 0;
            continue;
        }
        float ds = coin->s - env->s;
        float lat_m = fabsf(env->px - coin->lane) * env->lane_width;
        int touch = coin->s >= prev_s - 0.6f && coin->s <= env->s + 0.6f
            && lat_m < env->player_radius + 0.45f * env->lane_width
            && coin->y > env->y - 0.4f
            && coin->y < player_top + 0.6f;
        int vacuum = 0;
        if (!touch && magnet_on && ds > -1.0f) {
            float mid = env->y + 0.5f * (player_top - env->y);
            float dy = coin->y - mid;
            vacuum = ds * ds + lat_m * lat_m + dy * dy < magnet_r2;
        }
        if (touch || vacuum) tr2_collect_coin(env, coin);
    }

    for (int i = 0; i < TR2_MAX_POWERUPS; i++) {
        Tr2Powerup* pu = &env->powerups[i];
        if (!pu->active) continue;
        if (pu->s < env->s - 2.0f) {     /* left behind, free the slot */
            pu->active = 0;
            continue;
        }
        if (pu->kind >= TR2_PU_KINDS) continue;
        if (pu->s < prev_s - 1.0f || pu->s > env->s + 1.0f) continue;
        float lat_m = fabsf(env->px - pu->lane) * env->lane_width;
        if (lat_m > env->player_radius + 0.55f * env->lane_width) continue;
        pu->active = 0;
        tr2_apply_powerup(env, pu->kind);
    }
}
