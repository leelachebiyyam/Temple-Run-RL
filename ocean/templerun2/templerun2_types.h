/* Shared, frozen types for the data-driven Temple Run 2 rewrite.
 *
 * Module owners MUST NOT change this file. If a field is missing, message the
 * main integrator; do not edit another module's file.
 */
#pragma once

#include <stdint.h>

#define TR2_LANES 3
#define TR2_LANE_MIN 0
#define TR2_LANE_MAX 2
#define TR2_CENTER_LANE 1

/* One discrete head, five classes.  Index 0 is an explicit idle ("no swipe")
 * so the policy can leave a clear track alone; 1..4 are the four physical
 * gestures (left, right, up/jump, down/slide).  This restores the legacy
 * analytic env's 0 = no swipe, 1..4 = gesture layout; the renderer's keyboard
 * path writes TR2_ACT_NONE on every frame without a key press, so one press is
 * one gesture.  tr2_runner_step still decodes a negative value as idle for
 * legacy callers. */
#define TR2_ACTIONS 5
#define NUM_ATNS 1
#define ACT_SIZES {TR2_ACTIONS}

enum {
    TR2_ACT_NONE = 0,    /* idle: no gesture this step (sampled) */
    TR2_ACT_LEFT = 1,    /* swipe left / turn left */
    TR2_ACT_RIGHT = 2,   /* swipe right / turn right */
    TR2_ACT_JUMP = 3,    /* swipe up */
    TR2_ACT_SLIDE = 4,   /* swipe down */
};

/* Entity kinds used by collision and rendering. */
enum {
    TR2_OBST_LOW = 0,
    TR2_OBST_HIGH = 1,
    TR2_OBST_BLOCK = 2,
    TR2_OBST_GAP = 3,
    TR2_OBST_FIRE = 4,
    TR2_OBST_STUMBLE = 5,
    TR2_OBST_LEDGE_LEFT = 6,
    TR2_OBST_LEDGE_RIGHT = 7,
    TR2_OBST_WATER = 8,
    TR2_OBST_KINDS = 9,
};

enum {
    TR2_PU_MAGNET = 0,
    TR2_PU_BOOST = 1,
    TR2_PU_SHIELD = 2,
    TR2_PU_GEM = 3,
    TR2_PU_COIN_BONUS = 4,
    TR2_PU_SCORE_BONUS = 5,
    TR2_PU_KINDS = 6,
};

/* TRRunnerPawn.StateType */
enum {
    TR2_MODE_RUNNING = 0,
    TR2_MODE_DEAD = 1,
    TR2_MODE_RIDING = 2,
    TR2_MODE_ZIPLINE = 3,
    TR2_MODE_WATER = 4,
    TR2_MODE_FALLING = 5,
    TR2_MODE_LUGE = 6,
    TR2_MODE_DOWNHILL = 7,
};

/* DeathTypes, kept numeric-compatible with dump.cs. */
enum {
    TR2_DEATH_FALL = 0,
    TR2_DEATH_EATEN = 1,
    TR2_DEATH_FIRE = 2,
    TR2_DEATH_MINEDUCK = 3,
    TR2_DEATH_MINELEDGE = 4,
    TR2_DEATH_MINE = 5,
    TR2_DEATH_WHEEL = 6,
    TR2_DEATH_WATERLEFT = 7,
    TR2_DEATH_LEDGE = 8,
    TR2_DEATH_SCENERYTREE = 9,
    TR2_DEATH_SCENERYROCK = 10,
    TR2_DEATH_DEATHTHROW = 11,
    TR2_DEATH_NONE = 12,
    TR2_DEATH_WATERRIGHT = 13,
};

/* TRRunnerPawn.StumbleDeathByMonkeyTime (public serialized float; ..ctor
 * 0x1972E38 `mov w10,#0x41200000` = 10.0).  Stumble() kills with Eaten while
 * StumbleKillTimer is below this value, so a repeat stumble inside the window
 * is lethal (PLAYER.md sec.7, COLLISION.md sec.4).  The timer restarts at
 * 2x this (safe) on Reset/Resurrect and at 0 on a normal stumble. */
#define TR2_STUMBLE_MONKEY_TIME 10.0f

#define TR2_SEG_RING 96
#define TR2_SEG_AHEAD 16
#define TR2_MAX_OBSTACLES 256
#define TR2_MAX_COINS 512
#define TR2_MAX_POWERUPS 32
#define TR2_LOOK_OBST 10
#define TR2_LOOK_COINS 6
#define TR2_LOOK_PIECES 6
#define TR2_LOOK_POWERUPS 2

/* Structured observation layout (238 floats):
 *   27 base (24 state + mode timer, current piece flags, turn timer)
 *    3 turn
 *    6 * 4 pieces
 *   10 * 15 obstacles
 *    6 * 4 coins
 *    2 * 5 powerups
 */
#define TR2_OBS_BASE 27
#define TR2_OBS_TURN 3
#define TR2_OBS_PIECE 4
#define TR2_OBS_OBST 15
#define TR2_OBS_COIN 4
#define TR2_OBS_POWERUP 5
#define OBS_SIZE \
    (TR2_OBS_BASE + TR2_OBS_TURN \
     + TR2_LOOK_PIECES * TR2_OBS_PIECE \
     + TR2_LOOK_OBST * TR2_OBS_OBST \
     + TR2_LOOK_COINS * TR2_OBS_COIN \
     + TR2_LOOK_POWERUPS * TR2_OBS_POWERUP)

struct Log {
    float perf;
    float score;
    float coins;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct {
    float s;               /* absolute path distance */
    float length;          /* extent along the path */
    float lane;            /* lane centre in lane units (-1,0,1) */
    float half_width;      /* lateral half extent in lane units */
    float y_bottom;        /* vertical span, piece frame */
    float y_top;
    float phase;           /* flame/periodic phase */
    unsigned char kind;
    unsigned char variant;
    unsigned char trigger; /* TrackTrigger value */
    unsigned char death_type;
    unsigned char active;
    unsigned char passed;
    unsigned char trigger_entered;
    int owner_segment_plus_one; /* 0 = not piece-owned, else abs segment + 1 */
    /* Authored FallSaver window relative to this gap's centre. */
    float save_offset;
    float save_length;
    float save_half_width;
    float save_y_bottom;
    float save_y_top;
    unsigned char saved;
} Tr2Obstacle;

typedef struct {
    float s;
    float y;
    float lane;
    unsigned char value;
    unsigned char active;
} Tr2Coin;

typedef struct {
    float s;
    float lane;
    unsigned char kind;
    unsigned char active;
} Tr2Powerup;

/* One shipped track piece, placed on the world path. */
typedef struct {
    float s0;
    float length;
    float heading;         /* yaw at piece start */
    float wx, wz;          /* world position of the piece start */
    float yaw_delta;       /* authored SourceLastYRot (radians) */
    float wy;              /* world elevation (Y) of the piece start */
    signed char turn;      /* mandatory turn: 0 none, -1 left, +1 right */
    int piece_idx;         /* TR2_PIECE_TABLE index */
    int var_idx;           /* TR2_VARIATION_TABLE index */
    uint16_t spline_begin; /* first index in TR2_SPLINE_POINTS (authored SourcePath) */
    uint16_t spline_count; /* authored control points; 0 = no authored spline */
    unsigned char alternate_path; /* second branch of a symmetric junction */
    unsigned char mode;    /* TR2_MODE_* required for this piece */
    unsigned char obstacle_kind; /* spawned entity kind, 255 none */
    unsigned char entity_spawned;
    unsigned char valid;
} Tr2Segment;

typedef struct {
    int frame;
    float cam_x, cam_y, cam_z;
    float cam_yaw;
    double playback_deadline; /* wall-clock target across episode resets */
} Tr2Client;

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;

    /* tuning */
    float lane_width;
    float track_half_width;
    float player_radius;
    float player_height;
    float slide_height;
    float min_speed;
    float max_speed;
    float force_scale;
    float jump_velocity;
    float gravity;
    float slam_velocity;
    float slide_duration_factor;
    float slide_duration_cap;
    float stumble_speed;
    float lane_lerp;
    float turn_window;
    float magnet_time;
    float shield_time;
    float boost_distance;
    float boost_speed;
    float invuln_time;
    float magnet_radius;
    float coin_red_m;
    float coin_blue_m;
    float score_distance;
    float score_coin;
    float multiplier_m;
    int multiplier_max;
    float reward_clear;
    float reward_coin;
    /* Shaping only: charged once per step whose sampled action is one of the
     * four gestures, never for TR2_ACT_NONE.  0.0 disables it. */
    float swipe_cost;
    float death_penalty;
    int episode_length;
    int enable_submodes;
    int decision_frames; /* 60 Hz physics frames per policy swipe decision */

    /* runner */
    float s;
    float px;              /* PlayerXOffset, normalized -1..1 */
    int lane;              /* nearest lane 0..2 */
    int target_lane;
    float y;
    float vy;
    int grounded;
    float floor_y; /* Current support height, relative to the path surface. */
    float last_jump_land_time; /* Physics tick/60; FallSaver landing grace. */
    int sliding;
    float slide_timer;
    float stumble_timer;
    float stumble_kill_timer; /* StumbleKillTimer: seconds since last stumble */
    int off_track;            /* OffTrack: missed turn / dead end */
    float kill_timer;         /* KillTimer: off-track death countdown (s) */
    float off_track_time;     /* seconds since OffTrack was set */
    float speed;
    unsigned char mode;
    float mode_timer;
    float turn_input_timer;
    float lane_cooldown;
    int last_action;
    int turn_input;        /* -1/0/+1 */
    int dead;
    int death_cause;
    int last_death_cause;
    int last_hit_kind;

    float distance;
    float score;
    int coins;
    int gems;
    int multiplier;
    float magnet_timer;
    float shield_timer;
    float boost_distance_left;
    float invuln_timer;
    float episode_return;
    int tick;

    /* track */
    Tr2Segment segments[TR2_SEG_RING];
    int seg_idx;
    int seg_gen;
    int piece_count;
    uint8_t last_variation[TR2_PIECE_COUNT]; /* local index + 1; 0 = unseen */
    float turn_acc;
    float obstacle_acc;
    float next_turn_gap;
    float next_obstacle_gap;
    float next_coin_s;
    float next_powerup_s;

    /* entities */
    Tr2Obstacle obstacles[TR2_MAX_OBSTACLES];
    Tr2Coin coin_pool[TR2_MAX_COINS];
    Tr2Powerup powerups[TR2_MAX_POWERUPS];

    Tr2Client* client;
};

/* Module interfaces (implemented in the corresponding header). */
static void tr2_track_reset(Env* env);
static void tr2_track_generate(Env* env);
static void tr2_seg_point(const Tr2Segment* seg, float d, float lat_m,
        float* x, float* z, float* yaw);
static void tr2_seg_end(const Tr2Segment* seg, float* ex, float* ez,
        float* eh);
static float tr2_seg_height(const Tr2Segment* seg, float d);
static void tr2_world_at(Env* env, float s, float lane_units, float y,
        float* wx, float* wy, float* wz, float* yaw);
static int tr2_near_turn(Env* env, float s);
static void tr2_spawn_ahead(Env* env);

static void tr2_runner_reset(Env* env);
static void tr2_runner_step(Env* env);

static void tr2_collision_reset(Env* env);
static void tr2_collision_step(Env* env, float prev_s);

static void tr2_items_reset(Env* env);
static void tr2_items_step(Env* env, float prev_s);

static void tr2_observe(Env* env);
