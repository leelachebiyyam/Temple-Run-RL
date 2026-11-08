/* Shipped TRTrackPacing milestone rows for RegionID 100 (Machu) plus the
 * coin / powerup pacing helpers that read BonusItemProtoData.
 *
 * Evidence:
 *   recon/unity_assets/data/key/bundles/TRTrackPacing.json
 *     RegionID 100 asset (bundle .../Machu_Halloween/2/
 *     machu_halloweenregion.unity3d, path_id -8413602224705284033). Its 8
 *     "Milestones" entries are reproduced 1:1, in shipped order, in
 *     TR2_PACING_TABLE below (Distance / DoubleObstaclePercent kept as the
 *     serialized floats).
 *   recon/unity_assets/data/key/BonusItemProtoData.json  (sharedassets0.assets
 *     path_id 492) — DistanceToChangeDoubleCoins / TripleCoins, probabilities.
 *   config/templerun2_rewrite.ini — powerup_spacing_min/max 400/800.
 *   ocean/templerun2/templerun2_track.h:140 — reference first segment 100.0f.
 */
#pragma once

#include <stddef.h>

typedef struct {
    float distance;
    float obstacle_min, obstacle_max;
    float turn_min, turn_max;
    float double_pct;
    float coin_gap;
    int max_coins;
    int allow_turn_after_obstacle;
} Tr2PacingRow;

/* One row per TRTrackPacing.Milestones entry for RegionID 100.
 * Fields map: distance=Distance, obstacle_min/max=Min/MaxDistanceBetweenObstacles,
 * turn_min/max=Min/MaxDistanceBetweenTurns, double_pct=DoubleObstaclePercent,
 * coin_gap=MinDistanceBetweenCoinRuns, max_coins=MaxCoinsPerRun,
 * allow_turn_after_obstacle=AllowTurnAfterObstacle. */
static const Tr2PacingRow TR2_PACING_TABLE[] = {
    /* Distance    obst_min obst_max turn_min turn_max double   coin max allow */
    {0.0f,        10.0f,  20.0f,   25.0f,   50.0f,   0.000f,  80.0f, 25, 0},
    {250.0f,      10.0f,  20.0f,   25.0f,   50.0f,   0.000f,  70.0f, 25, 0},
    {500.0f,      10.0f,  15.0f,   20.0f,   45.0f,   0.050f,  60.0f, 25, 0},
    {1000.0f,     10.0f,  10.0f,   25.0f,   45.0f,   0.050f,  60.0f, 25, 0},
    {1500.0f,      7.5f,  10.0f,   15.0f,   40.0f,   0.100f,  50.0f, 25, 1},
    {2000.0f,      5.0f,   5.0f,   15.0f,   40.0f,   0.150f,  50.0f, 25, 1},
    {3250.0f,      5.0f,   5.0f,   10.0f,   30.0f,   0.175f,  40.0f, 25, 1},
    {4500.0f,      0.0f,   2.5f,    5.0f,   25.0f,   0.200f,  40.0f, 25, 1},
};
#define TR2_PACING_COUNT (sizeof(TR2_PACING_TABLE) / sizeof(TR2_PACING_TABLE[0]))

/* Active milestone row for travelled distance s (last row whose distance <= s). */
static const Tr2PacingRow* tr2_pacing_lookup(float s) {
    const Tr2PacingRow* row = &TR2_PACING_TABLE[0];
    for (size_t i = 1; i < TR2_PACING_COUNT; i++) {
        if (s < TR2_PACING_TABLE[i].distance) break;
        row = &TR2_PACING_TABLE[i];
    }
    return row;
}

/* First generated segment is a straight. */
/* Evidence: reference server first-segment length 100.0f
 * (ocean/templerun2/templerun2_track.h:140, abs_idx == 0); the Machu base pool
 * leads with the "Straight Intro"/Straight pieces (recon/track/TRACK.md:711). */
static const float TR2_FIRST_STRAIGHT_LENGTH = 100.0f;

/* ------------------------------------------------------------------ coins */

/* Coin multiplier thresholds (metres travelled).
 * Evidence: BonusItemProtoData.json "DistanceToChangeDoubleCoins": 2500.0,
 * "DistanceToChangeTripleCoins": 5000.0 (sharedassets0.assets path_id 492);
 * mirrored by config/templerun2_rewrite.ini coin_red_m=2500 / coin_blue_m=5000. */
#define TR2_COIN_DOUBLE_DISTANCE 2500.0f
#define TR2_COIN_TRIPLE_DISTANCE 5000.0f

/* Coin pickup value: 1x below 2500 m, 2x from 2500 m, 3x from 5000 m. */
static int tr2_coin_value(float s) {
    if (s >= TR2_COIN_TRIPLE_DISTANCE) return 3;
    if (s >= TR2_COIN_DOUBLE_DISTANCE) return 2;
    return 1;
}

/* --------------------------------------------------------------- powerups */

/* Bonus-item spacing, metres.
 * Evidence: config/templerun2_rewrite.ini powerup_spacing_min=400.0 /
 * powerup_spacing_max=800.0 (the range the reference env samples in
 * ocean/templerun2/templerun2_track.h:335). BonusItemProtoData's own defaults
 * (DefaultDistanceBetweenBonusItems 500.0, MinDistanceBetweenBonusItems 5.0,
 * MaxTimeBetweenBonusItems 15.0) sit inside / gate this range. */
#define TR2_POWERUP_SPACING_MIN 400.0f
#define TR2_POWERUP_SPACING_MAX 800.0f

/* Spawn weights.
 * Evidence: BonusItemProtoData.json "ProbabilityVacuum": 100 (magnet),
 * "ProbabilityBoost": 100, "ProbabilityShield": 100, "ProbabilityGem": 100;
 * ProbabilityBolt/CoinBonus are 0, so only these four are eligible. */
#define TR2_POWERUP_MAGNET_WEIGHT 100
#define TR2_POWERUP_BOOST_WEIGHT  100
#define TR2_POWERUP_SHIELD_WEIGHT 100
#define TR2_POWERUP_GEM_WEIGHT    100
#define TR2_POWERUP_WEIGHT_TOTAL \
    (TR2_POWERUP_MAGNET_WEIGHT + TR2_POWERUP_BOOST_WEIGHT \
     + TR2_POWERUP_SHIELD_WEIGHT + TR2_POWERUP_GEM_WEIGHT)

/* Reduce any caller-supplied random integer to a float in [0,1). */
static float tr2_pacing_unit(int rng) {
    return (float)((unsigned)rng % 65536u) * (1.0f / 65536.0f);
}

/* Uniform distance in metres until the next bonus item, [400, 800). */
static float tr2_powerup_spacing(int rng) {
    return TR2_POWERUP_SPACING_MIN
        + (TR2_POWERUP_SPACING_MAX - TR2_POWERUP_SPACING_MIN)
          * tr2_pacing_unit(rng);
}

/* Weighted powerup kind: magnet / boost / shield / gem at equal 100 weights.
 * Returns 0/1/2/3, i.e. TR2_PU_MAGNET / TR2_PU_BOOST / TR2_PU_SHIELD /
 * TR2_PU_GEM (templerun2_types.h); this header stays standalone so it does not
 * pull in that type file. */
static int tr2_powerup_kind(int rng) {
    int pick = (int)(tr2_pacing_unit(rng) * (float)TR2_POWERUP_WEIGHT_TOTAL);
    if (pick < TR2_POWERUP_MAGNET_WEIGHT) return 0;  /* TR2_PU_MAGNET */
    pick -= TR2_POWERUP_MAGNET_WEIGHT;
    if (pick < TR2_POWERUP_BOOST_WEIGHT) return 1;   /* TR2_PU_BOOST */
    pick -= TR2_POWERUP_BOOST_WEIGHT;
    if (pick < TR2_POWERUP_SHIELD_WEIGHT) return 2;  /* TR2_PU_SHIELD */
    return 3;                                        /* TR2_PU_GEM */
}
