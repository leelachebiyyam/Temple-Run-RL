/* RegionID 100 (Machu / "Spooky Summit") shipped track-piece pools for Zones
 * 0..4 (Base, MineCart, Narrows, WaterSlide, Forest).
 *
 * Every row indexes TR2_PIECE_TABLE / TR2_VARIATION_TABLE in
 * templerun2_track_data.h, which templerun2.h includes first.  `odds` and
 * `difficulty` are the piece's authored SelectionOdds / DifficultyLevel;
 * `category` is derived from the piece's real flag bitmask exactly like the
 * game's TrackPieceDefinition getters (recon/track/TRACK.md 1.2):
 *
 *   IsTurn     = IsJunction | IsYJunction | IsTurnLeft | IsTurnRight
 *   IsAnyCurve = IsCurveLeft | IsCurveRight
 *   IsObstacle = IsSlideUnder | IsJumpOver | IsStumble | IsZipLine
 *              | IsGenericObstacle | IsLedgeLeft | IsLedgeRight
 *                (every Machu zone has RegionZoneInfo.IsLedgeZone == 0, so
 *                 LedgeLeftOrRightIsObstacle is set)
 *   submode    = the piece's gameplay StateType != Running (the IsZipLine
 *                flag, or the zone's RegionZoneInfo.Gameplay: 1 Ride,
 *                2 WaterSlide, 3 Luge, 4 DownHillSlide)
 *
 * Category priority: SUBMODE, TURN, CURVE, OBSTACLE, STRAIGHT.  tr2_piece_category
 * recomputes this from TR2_PIECE_TABLE at run time and is authoritative.
 *
 * Every piece of RegionID 100 appears exactly once, in its zone pool, in
 * TR2_PIECE_TABLE order (including odds==0 compound / authoring-only pieces).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "templerun2_types.h"   /* TR2_MODE_* */

enum {
    TR2_CAT_STRAIGHT = 0,
    TR2_CAT_TURN = 1,
    TR2_CAT_OBSTACLE = 2,
    TR2_CAT_CURVE = 3,
    TR2_CAT_SUBMODE = 4,
};

typedef struct {
    int piece_idx;      /* TR2_PIECE_TABLE index */
    float odds;         /* authored SelectionOdds (per-1000 pool gate) */
    int difficulty;     /* authored DifficultyLevel */
    int category;       /* TR2_CAT_* */
} Tr2PieceEntry;

/* Zone 0 - Base (26 pieces). */
static const Tr2PieceEntry TR2_PIECE_POOL_Z0[] = {
    { 17, 1.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z0_P0  Machu Base Straight Flat */
    { 18, 0.7500f, 1, TR2_CAT_TURN},  /* R100_Z0_P1  Machu Base Junction */
    { 19, 0.1000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z0_P2  Machu Base Bridge */
    { 20, 1.0000f, 0, TR2_CAT_OBSTACLE},  /* R100_Z0_P3  Machu Base Jump Over */
    { 21, 1.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z0_P4  Machu Base Straight Intro */
    { 22, 1.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z0_P5  Machu Base Straight */
    { 23, 1.0000f, 0, TR2_CAT_TURN},  /* R100_Z0_P6  Machu Base Turn Left */
    { 24, 1.0000f, 0, TR2_CAT_TURN},  /* R100_Z0_P7  Machu Base Turn Right */
    { 25, 0.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z0_P8  Machu Base Temple Start */
    { 26, 0.5000f, 2, TR2_CAT_SUBMODE},  /* R100_Z0_P9  Machu Base ZipLine */
    { 27, 1.0000f, 3, TR2_CAT_OBSTACLE},  /* R100_Z0_P10  Machu_Halloween Base Ledge Right Start */
    { 28, 0.0000f, 3, TR2_CAT_OBSTACLE},  /* R100_Z0_P11  Machu_Halloween Base Ledge Right Middle */
    { 29, 0.0000f, 3, TR2_CAT_OBSTACLE},  /* R100_Z0_P12  Machu_Halloween Base Ledge Right End */
    { 30, 1.0000f, 0, TR2_CAT_CURVE},  /* R100_Z0_P13  Machu Base Curve Left */
    { 31, 1.0000f, 0, TR2_CAT_CURVE},  /* R100_Z0_P14  Machu Base Curve Right */
    { 32, 1.0000f, 3, TR2_CAT_OBSTACLE},  /* R100_Z0_P15  Machu Base Ledge Left Start */
    { 33, 0.0000f, 3, TR2_CAT_OBSTACLE},  /* R100_Z0_P16  Machu_Halloween Base Ledge Left Middle */
    { 34, 0.0000f, 3, TR2_CAT_OBSTACLE},  /* R100_Z0_P17  Machu_Halloween Base Ledge Left End */
    { 35, 1.0000f, 0, TR2_CAT_OBSTACLE},  /* R100_Z0_P18  Machu Base Stumble */
    { 36, 1.0000f, 2, TR2_CAT_OBSTACLE},  /* R100_Z0_P19  Machu Base Jump or Slide */
    { 37, 0.7500f, 1, TR2_CAT_OBSTACLE},  /* R100_Z0_P20  Machu Base Slide Under */
    { 38, 1.0000f, 2, TR2_CAT_OBSTACLE},  /* R100_Z0_P21  Machu Base Jump Over Long */
    { 39, 0.4000f, 3, TR2_CAT_OBSTACLE},  /* R100_Z0_P22  Machu Base Run Between */
    { 40, 0.3000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z0_P23  Machu Base Stairs Up Start */
    { 41, 0.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z0_P24  Machu_Halloween Base Stairs Up Middle */
    { 42, 0.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z0_P25  Machu_Halloween Base Stairs Up End */
};

/* Zone 1 - MineCart (16 pieces). */
static const Tr2PieceEntry TR2_PIECE_POOL_Z1[] = {
    { 43, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P0  Machu MineCart Straight */
    { 44, 1.0000f, 3, TR2_CAT_SUBMODE},  /* R100_Z1_P1  Machu MineCart Entrance */
    { 45, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P2  Machu MineCart Exit */
    { 46, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P3  Machu MineCart Ledge Left Start */
    { 47, 0.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P4  Machu_Halloween MineCart Ledge Left Middle */
    { 48, 0.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P5  Machu_Halloween MineCart Ledge Left End */
    { 49, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P6  Machu MineCart Turn Left */
    { 50, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P7  Machu MineCart Turn Right */
    { 51, 0.2000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P8  Machu MineCart Curve Right Long */
    { 52, 0.4000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P9  Machu MineCart Curve Left */
    { 53, 0.4000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P10  Machu MineCart Curve Right */
    { 54, 0.4000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P11  Machu MineCart Junction */
    { 55, 0.7500f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P12  Machu MineCart Duck */
    { 56, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P13  Machu MineCart Ledge Right Start */
    { 57, 0.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P14  Machu_Halloween MineCart Ledge Right Middle */
    { 58, 0.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z1_P15  Machu_Halloween MineCart Ledge Right End */
};

/* Zone 2 - Narrows (10 pieces). */
static const Tr2PieceEntry TR2_PIECE_POOL_Z2[] = {
    { 59, 1.0000f, 3, TR2_CAT_STRAIGHT},  /* R100_Z2_P0  Machu Narrows Entrance */
    { 60, 1.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z2_P1  Machu Narrows Exit */
    { 61, 1.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z2_P2  Machu Narrows Straight */
    { 62, 1.0000f, 0, TR2_CAT_TURN},  /* R100_Z2_P3  Machu Narrows Turn Left */
    { 63, 1.0000f, 0, TR2_CAT_TURN},  /* R100_Z2_P4  Machu Narrows Turn Right */
    { 64, 1.0000f, 1, TR2_CAT_OBSTACLE},  /* R100_Z2_P5  Machu Narrows Jump Or Slide */
    { 65, 0.1000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z2_P6  Machu Narrows Straight Rare */
    { 66, 0.7500f, 1, TR2_CAT_TURN},  /* R100_Z2_P7  Machu Narrows Junction */
    { 67, 1.0000f, 1, TR2_CAT_OBSTACLE},  /* R100_Z2_P8  Machu Narrows Stumble */
    { 68, 1.0000f, 1, TR2_CAT_OBSTACLE},  /* R100_Z2_P9  Machu Narrows Jump */
};

/* Zone 3 - WaterSlide (17 pieces). */
static const Tr2PieceEntry TR2_PIECE_POOL_Z3[] = {
    { 69, 1.0000f, 2, TR2_CAT_SUBMODE},  /* R100_Z3_P0  Machu WaterSlide Entrance */
    { 70, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z3_P1  Machu WaterSlide Exit */
    { 71, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z3_P2  Machu WaterSlide Straight */
    { 72, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z3_P3  Machu WaterSlide CurveLeft45 */
    { 73, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z3_P4  Machu WaterSlide CurveRight45 */
    { 74, 0.5000f, 1, TR2_CAT_SUBMODE},  /* R100_Z3_P5  Machu WaterSlide Duck */
    { 75, 0.2500f, 1, TR2_CAT_SUBMODE},  /* R100_Z3_P6  Machu WaterSlide Duck Left */
    { 76, 0.2500f, 1, TR2_CAT_SUBMODE},  /* R100_Z3_P7  Machu WaterSlide Duck Right */
    { 77, 0.2500f, 1, TR2_CAT_SUBMODE},  /* R100_Z3_P8  Machu WaterSlide Duck Center */
    { 78, 0.2500f, 1, TR2_CAT_SUBMODE},  /* R100_Z3_P9  Machu WaterSlide Duck Side */
    { 79, 0.7500f, 0, TR2_CAT_SUBMODE},  /* R100_Z3_P10  Machu WaterSlide Tilt Right */
    { 80, 0.7500f, 0, TR2_CAT_SUBMODE},  /* R100_Z3_P11  Machu WaterSlide Tilt Left */
    { 81, 0.7500f, 0, TR2_CAT_SUBMODE},  /* R100_Z3_P12  Machu WaterSlide CurveLeft90 */
    { 82, 0.7500f, 0, TR2_CAT_SUBMODE},  /* R100_Z3_P13  Machu WaterSlide CurveRight90 */
    { 83, 0.7500f, 1, TR2_CAT_SUBMODE},  /* R100_Z3_P14  Machu WaterSlide Rock */
    { 84, 1.0000f, 0, TR2_CAT_SUBMODE},  /* R100_Z3_P15  Machu WaterSlide Tilt Tutorial */
    { 85, 0.7500f, 1, TR2_CAT_SUBMODE},  /* R100_Z3_P16  Machu WaterSlide Junction */
};

/* Zone 4 - Forest (9 pieces). */
static const Tr2PieceEntry TR2_PIECE_POOL_Z4[] = {
    { 86, 1.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z4_P0  Machu Forest Entrance */
    { 87, 1.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z4_P1  Machu Forest Exit */
    { 88, 1.0000f, 0, TR2_CAT_STRAIGHT},  /* R100_Z4_P2  Machu Forest Straight */
    { 89, 1.0000f, 0, TR2_CAT_TURN},  /* R100_Z4_P3  Machu Forest Turn Left */
    { 90, 1.0000f, 0, TR2_CAT_TURN},  /* R100_Z4_P4  Machu Forest Turn Right */
    { 91, 0.7500f, 1, TR2_CAT_TURN},  /* R100_Z4_P5  Machu Forest Junction */
    { 92, 1.0000f, 2, TR2_CAT_OBSTACLE},  /* R100_Z4_P6  Machu Forest Slide Under */
    { 93, 1.0000f, 1, TR2_CAT_OBSTACLE},  /* R100_Z4_P7  Machu Forest Jump Over */
    { 94, 1.0000f, 0, TR2_CAT_OBSTACLE},  /* R100_Z4_P8  Machu Forest Stumble */
};

#define TR2_PIECE_POOL_Z0_COUNT (sizeof(TR2_PIECE_POOL_Z0) / sizeof(TR2_PIECE_POOL_Z0[0]))
#define TR2_PIECE_POOL_Z1_COUNT (sizeof(TR2_PIECE_POOL_Z1) / sizeof(TR2_PIECE_POOL_Z1[0]))
#define TR2_PIECE_POOL_Z2_COUNT (sizeof(TR2_PIECE_POOL_Z2) / sizeof(TR2_PIECE_POOL_Z2[0]))
#define TR2_PIECE_POOL_Z3_COUNT (sizeof(TR2_PIECE_POOL_Z3) / sizeof(TR2_PIECE_POOL_Z3[0]))
#define TR2_PIECE_POOL_Z4_COUNT (sizeof(TR2_PIECE_POOL_Z4) / sizeof(TR2_PIECE_POOL_Z4[0]))

static const size_t TR2_PIECE_POOL_SIZES[] = {
    TR2_PIECE_POOL_Z0_COUNT,
    TR2_PIECE_POOL_Z1_COUNT,
    TR2_PIECE_POOL_Z2_COUNT,
    TR2_PIECE_POOL_Z3_COUNT,
    TR2_PIECE_POOL_Z4_COUNT,
};

static const Tr2PieceEntry* const TR2_PIECE_POOLS[] = {
    TR2_PIECE_POOL_Z0,
    TR2_PIECE_POOL_Z1,
    TR2_PIECE_POOL_Z2,
    TR2_PIECE_POOL_Z3,
    TR2_PIECE_POOL_Z4,
};
#define TR2_PIECE_ZONE_COUNT (sizeof(TR2_PIECE_POOLS) / sizeof(TR2_PIECE_POOLS[0]))

/* RegionZoneInfo.Gameplay (TRGameRunnerGameplay) per (region, zone):
 * 0 Run, 1 Ride, 2 WaterSlide, 3 Luge, 4 DownHillSlide, extracted from the
 * shipped zone assets (recon/unity_assets/data/key/zones/, one JSON per zone). */
static int tr2_zone_gameplay(int region_id, int zone_id) {
    switch (region_id * 16 + zone_id) {
        case 3: return 2;   /* region 0, zone 3 */
        case 1600: return 0;   /* region 100, zone 0 */
        case 1601: return 1;   /* region 100, zone 1 */
        case 1602: return 0;   /* region 100, zone 2 */
        case 1603: return 2;   /* region 100, zone 3 */
        case 1604: return 0;   /* region 100, zone 4 */
        case 1665: return 0;   /* region 104, zone 1 */
        case 1666: return 0;   /* region 104, zone 2 */
        case 1667: return 0;   /* region 104, zone 3 */
        case 1668: return 0;   /* region 104, zone 4 */
        case 1669: return 0;   /* region 104, zone 5 */
        case 1670: return 0;   /* region 104, zone 6 */
        case 1671: return 0;   /* region 104, zone 7 */
        case 1672: return 0;   /* region 104, zone 8 */
        case 1673: return 0;   /* region 104, zone 9 */
        case 1674: return 4;   /* region 104, zone 10 */
        default: return 0;    /* Run */
    }
}

/* Gameplay StateType required while traversing piece_idx (TR2_MODE_*). */
static int tr2_piece_mode(int piece_idx) {
    if (piece_idx < 0 || piece_idx >= TR2_PIECE_COUNT) return TR2_MODE_RUNNING;
    if (TR2_PIECE_TABLE[piece_idx].flags & TR2_FLAG_IsZipLine)
        return TR2_MODE_ZIPLINE;
    switch (tr2_zone_gameplay(TR2_PIECE_TABLE[piece_idx].region_id,
                              TR2_PIECE_TABLE[piece_idx].zone_id)) {
        case 1: return TR2_MODE_RIDING;
        case 2: return TR2_MODE_WATER;
        case 3: return TR2_MODE_LUGE;
        case 4: return TR2_MODE_DOWNHILL;
        default: return TR2_MODE_RUNNING;
    }
}

/* Selection category of piece_idx (TR2_CAT_*); -1 when out of range. */
static int tr2_piece_category(int piece_idx) {
    uint32_t f;
    if (piece_idx < 0 || piece_idx >= TR2_PIECE_COUNT) return -1;
    if (tr2_piece_mode(piece_idx) != TR2_MODE_RUNNING) return TR2_CAT_SUBMODE;
    f = (uint32_t)TR2_PIECE_TABLE[piece_idx].flags;
    if (f & (TR2_FLAG_IsJunction | TR2_FLAG_IsYJunction
             | TR2_FLAG_IsTurnLeft | TR2_FLAG_IsTurnRight))
        return TR2_CAT_TURN;
    if (f & (TR2_FLAG_IsCurveLeft | TR2_FLAG_IsCurveRight))
        return TR2_CAT_CURVE;
    if (f & (TR2_FLAG_IsSlideUnder | TR2_FLAG_IsJumpOver | TR2_FLAG_IsStumble
             | TR2_FLAG_IsZipLine | TR2_FLAG_IsGenericObstacle
             | TR2_FLAG_IsLedgeLeft | TR2_FLAG_IsLedgeRight))
        return TR2_CAT_OBSTACLE;
    return TR2_CAT_STRAIGHT;
}

/* RegionID 100 zone pool (zone 0..4); out-of-range falls back to zone 0.
 * *count receives the pool length (may be NULL).  Never returns NULL. */
static const Tr2PieceEntry* tr2_piece_pool(int zone, int* count) {
    if (zone < 0 || zone >= (int)TR2_PIECE_ZONE_COUNT) zone = 0;
    if (count) *count = (int)TR2_PIECE_POOL_SIZES[zone];
    return TR2_PIECE_POOLS[zone];
}

/* Pool row describing piece_idx, or NULL when it is not in RegionID 100. */
static const Tr2PieceEntry* tr2_piece_entry(int piece_idx) {
    size_t z;
    if (piece_idx < 0 || piece_idx >= TR2_PIECE_COUNT) return 0;
    for (z = 0; z < TR2_PIECE_ZONE_COUNT; z++) {
        size_t i;
        for (i = 0; i < TR2_PIECE_POOL_SIZES[z]; i++) {
            if (TR2_PIECE_POOLS[z][i].piece_idx == piece_idx)
                return &TR2_PIECE_POOLS[z][i];
        }
    }
    return 0;
}

/* Number of authored variations of piece_idx (0 when out of range). */
static int tr2_piece_variation_count(int piece_idx) {
    if (piece_idx < 0 || piece_idx >= TR2_PIECE_COUNT) return 0;
    return (int)TR2_PIECE_TABLE[piece_idx].var_count;
}

/* Estimated path length (world units) of local variation var_idx. */
static float tr2_piece_variation_len(int piece_idx, int var_idx) {
    int v;
    if (piece_idx < 0 || piece_idx >= TR2_PIECE_COUNT) return 0.0f;
    if (var_idx < 0 || var_idx >= (int)TR2_PIECE_TABLE[piece_idx].var_count)
        return 0.0f;
    v = (int)TR2_PIECE_TABLE[piece_idx].var_begin + var_idx;
    return TR2_VARIATION_TABLE[v].est_len;
}

/* End heading of local variation var_idx, degrees (0 forward, 90 right,
 * 270 left) - the raw SourceLastYRot kept in TR2_VARIATION_TABLE.end_yaw. */
static float tr2_piece_variation_yaw_deg(int piece_idx, int var_idx) {
    int v;
    if (piece_idx < 0 || piece_idx >= TR2_PIECE_COUNT) return 0.0f;
    if (var_idx < 0 || var_idx >= (int)TR2_PIECE_TABLE[piece_idx].var_count)
        return 0.0f;
    v = (int)TR2_PIECE_TABLE[piece_idx].var_begin + var_idx;
    return TR2_VARIATION_TABLE[v].end_yaw;
}
