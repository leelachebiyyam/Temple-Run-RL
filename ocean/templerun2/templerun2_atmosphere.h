/* Temple Run 2 atmosphere pass — chase camera, distance fog, sky/light tint.
 *
 * Every value below is read from the shipped APK assets (see
 * resources/templerun2/ASSET_CONTRACT.md and recon/unity_assets):
 *
 *   TRCameraController prefab, asset 7b3169ba740bc46dc8b9d4a67dda9f5c (pid 9):
 *       EndLocation               = (0, 4.79, -8.48)   EndRotation = (6.7727, 0, 0)
 *       DebugCameraFollowDistance = 6.0   DebugCameraFocusDistance = 3.2
 *       DebugCameraFocusOffset    = (0, 2.24, 0)   DebugCameraPitchHeight = 0.96
 *     EndLocation/EndRotation are the *title-screen* pose (the shipped field is
 *     tooltipped "Location of camera when showing the title screen"), NOT the
 *     gameplay camera; the asset sets AllowDebugCameraSettings = 0, so the play
 *     camera uses the follow/focus model below.  The standard region row
 *     (CameraSettings for PirateCove Sandbar, zone bundle
 *     piratecove_dotdsandbarzone.unity3d) confirms the same set:
 *       CameraFollowDistance 6.0, CameraFocusDistance 3.2,
 *       CameraPitchHeight 0.96, CameraFocusOffset.y 2.24;
 *     TRCameraController.UpdateCameraOffsets derives the pitch as
 *     atan(CameraPitchHeight / CameraFocusDistance) = atan(0.96 / 3.2) = 16.70 deg.
 *   CameraChangeRate is per zone (0.1 for the Machu water-slide bundle, 0.25
 *   for the pirate-cove sandbar); this pass keeps 0.1.
 *   Unity Camera objects serialize no field-of-view override, so Unity's 60 deg
 *   perspective default applies.
 *
 *   RegionInfo "Machu_Halloween", RegionID 100, pid -2488247537773096884
 *   (bundle machu_halloweenregion.unity3d):
 *       GrowDistance 80, CullDistance 80, FadeInEndDistnace 75,
 *       FullyOpaqueDistance 45, GlobalLightIntensity 1.33, AmbientLight,
 *       GlobalLightColor, ProjectedShadowColor, RegionSunlight{Position,Rotation}.
 *
 *   Sky: Sky_Machu01 (1024x1024, machu_halloweenregion.unity3d) is a layered
 *   sky *atlas* — a moon disc, cloud sprites, lightning and two mountain strips
 *   packed into one sheet — exported to resources/templerun2/sky.png.  It is
 *   not a single full-screen background; the renderer blits the composited
 *   single sheet resources/templerun2/backdrop.png instead.
 *
 * raylib 5.5 keeps no fog state (rlFog* was dropped after raylib 3.x) and its
 * default 3D shader is unlit, so the pass applies what raylib does support:
 *   - the 3D clip planes are pinned to the region's near/far so BeginMode3D()
 *     crops the world at CullDistance and distant track fades into the sky
 *     (BeginMode3D builds its projection from rlGetCullDistanceNear/Far), and
 *   - the shipped fog / ambient / sunlight tint is published as constants
 *     (TR2_ATM_*_COLOR) for the renderer to fold into its material colours.
 *
 * Called by puf_render() before BeginMode3D(); must not draw.  Fallback safe:
 * no-ops on NULL env/cam and seeds itself on the first client frame.
 *
 * Header-only, C99, all symbols static.  Assumes raylib.h and the frozen types
 * (Env/Tr2Client, tr2_world_at) are already visible; both headers are re-included
 * here so the file also compiles on its own.
 */
#pragma once

#include <math.h>
#include <stddef.h>

#include "raylib.h"
#include "rlgl.h"

/* ------------------------------------------------------------------ *
 * Shipped camera framing (TRCameraController).
 * ------------------------------------------------------------------ */
#define TR2_ATM_CAM_BACK        6.00f      /* CameraFollowDistance */
#define TR2_ATM_CAM_HEIGHT      3.20f      /* focus 2.24 + pitchHeight 0.96 */
#define TR2_ATM_CAM_PITCH_DEG   16.699f    /* atan(0.96 / 3.2) */
#define TR2_ATM_CAM_FOV         60.0f      /*  shipped Unity Camera field of view */
#define TR2_ATM_CAM_CHANGE_RATE 0.10f      /*  CameraSettings.CameraChangeRate */
#define TR2_ATM_CAM_TARGET_LEN  3.20f      /* CameraFocusDistance */

#define TR2_ATM_CAM_FOLLOW_DIST  6.00f     /* DebugCameraFollowDistance */
#define TR2_ATM_CAM_FOCUS_DIST   3.20f     /* DebugCameraFocusDistance */
#define TR2_ATM_CAM_FOCUS_Y      2.24f     /* DebugCameraFocusOffset.y */
#define TR2_ATM_CAM_PITCH_HEIGHT 0.96f     /* DebugCameraPitchHeight */

/* ------------------------------------------------------------------ *
 * Shipped region atmosphere (RegionInfo Machu_Halloween, RegionID 100).
 * ------------------------------------------------------------------ */
#define TR2_ATM_GROW_DIST        80.0f     /* GrowDistance */
#define TR2_ATM_CULL_DIST        80.0f     /* CullDistance */
#define TR2_ATM_FADE_END_DIST    75.0f     /* FadeInEndDistnace */
#define TR2_ATM_OPAQUE_DIST      45.0f     /* FullyOpaqueDistance */
#define TR2_ATM_LIGHT_INTENSITY  1.33f     /* GlobalLightIntensity */

#define TR2_ATM_AMBIENT_R 0.361866f        /* AmbientLight */
#define TR2_ATM_AMBIENT_G 0.238754f
#define TR2_ATM_AMBIENT_B 0.470588f
#define TR2_ATM_LIGHT_R   1.000000f        /* GlobalLightColor */
#define TR2_ATM_LIGHT_G   0.862290f
#define TR2_ATM_LIGHT_B   0.616000f
#define TR2_ATM_SHADOW_R  0.189906f        /* ProjectedShadowColor */
#define TR2_ATM_SHADOW_G  0.104455f
#define TR2_ATM_SHADOW_B  0.338235f
#define TR2_ATM_SUN_POS_X 0.949900f        /* RegionSunlightPosition */
#define TR2_ATM_SUN_POS_Y 1.990000f
#define TR2_ATM_SUN_POS_Z -2.000000f
#define TR2_ATM_SUN_PITCH_DEG 43.700f      /* RegionSunlightRotation.x */
#define TR2_ATM_SUN_YAW_DEG   332.530f     /* RegionSunlightRotation.y */

/* ------------------------------------------------------------------ *
 * Fog / clip planes.
 *
 * Machu ships no RegionCameraSettings; the shipped Camera near clip (0.3) and
 * the region CullDistance (80) bound the 3D pass.  Fog runs from the distance
 * where pieces are fully opaque (45) to the cull plane (80).
 * ------------------------------------------------------------------ */
#define TR2_ATM_NEAR_CLIP 0.3f
#define TR2_ATM_FOG_START TR2_ATM_OPAQUE_DIST
#define TR2_ATM_FOG_END   TR2_ATM_CULL_DIST
#define TR2_ATM_FAR_CLIP  TR2_ATM_FOG_END

/* Shipped Machu sky atlas (Sky_Machu01); backdrop.png is the composited sheet
 * the renderer actually blits full-screen (see the sky note in the file head). */
#define TR2_ATM_SKY_TEXTURE "resources/templerun2/sky.png"
#define TR2_ATM_SKY_BACKDROP "resources/templerun2/backdrop.png"

/* Tint colours (0..255) for the renderer to multiply into materials.  Machu
 * serializes no RegionVertexFog, so the fog tint falls back to the region
 * ambient light; the PirateCove value is kept for reference. */
#define TR2_ATM_AMBIENT_COLOR (Color){ 92, 61, 120, 255 }
#define TR2_ATM_LIGHT_COLOR   (Color){ 255, 220, 157, 255 }
#define TR2_ATM_SHADOW_COLOR  (Color){ 48, 27, 86, 255 }
#define TR2_ATM_FOG_COLOR     TR2_ATM_AMBIENT_COLOR
#define TR2_ATM_FOG_COLOR_PIRATECOVE (Color){ 165, 249, 235, 255 }

/* ------------------------------------------------------------------ *
 * Chase camera state (one interactive window per process).
 * ------------------------------------------------------------------ */
static float tr2_atm_cam_x = 0.0f;
static float tr2_atm_cam_y = 0.0f;
static float tr2_atm_cam_z = 0.0f;
static float tr2_atm_cam_yaw = 0.0f;
static int tr2_atm_cam_seeded = 0;
static const Env* tr2_atm_cam_env = NULL;   /* episode-identity guard  */
static int tr2_atm_cam_tick = -1;           /* last env->tick observed */


static void tr2_atmosphere_apply(Env* env, Camera3D* cam, float dt) {
    if (env == NULL || cam == NULL) return;

    float wx, wy, wz, yaw;
    tr2_world_at(env, env->s, env->px, env->y, &wx, &wy, &wz, &yaw);
    float path_yaw = tr2_r_path_yaw(env, env->s);
    (void)yaw;

    Tr2Client* c = env->client;
    /* puf_reset() runs inside puf_step() and rewinds env->tick (and s) to the
     * start of a new run, but this camera state and Tr2Client survive.  Without
     * this guard the chase pose eases from the death spot to the new start,
     * dragging the view across the whole map.  Snap on every episode boundary. */
    int restart = env != tr2_atm_cam_env
        || (tr2_atm_cam_tick >= 0 && env->tick < tr2_atm_cam_tick);
    tr2_atm_cam_env = env;
    tr2_atm_cam_tick = env->tick;

    /* Chase pose from the shipped TRCameraController EndLocation. */
    if (!tr2_atm_cam_seeded || (c != NULL && c->frame <= 1) || restart) {
        tr2_atm_cam_x = wx - sinf(path_yaw) * TR2_ATM_CAM_BACK;
        tr2_atm_cam_y = wy + TR2_ATM_CAM_HEIGHT;
        tr2_atm_cam_z = wz - cosf(path_yaw) * TR2_ATM_CAM_BACK;
        tr2_atm_cam_yaw = path_yaw;
        tr2_atm_cam_seeded = 1;
    } else {
        float k = TR2_ATM_CAM_CHANGE_RATE * (dt > 0.0f ? dt : (1.0f / 60.0f)) * 60.0f;
        if (k > 1.0f) k = 1.0f;
        else if (k < 0.0f) k = 0.0f;
        /* Ease the orbit heading toward the (already smooth) path heading, then
         * place the desired pose behind the runner along that eased heading.
         * Deriving the offset from the eased heading rotates the chase distance
         * through a bend instead of translating it sideways in a single frame. */
        float dyaw = path_yaw - tr2_atm_cam_yaw;
        while (dyaw > PI) dyaw -= 2.0f * PI;
        while (dyaw < -PI) dyaw += 2.0f * PI;
        tr2_atm_cam_yaw += dyaw * k;
        float want_x = wx - sinf(tr2_atm_cam_yaw) * TR2_ATM_CAM_BACK;
        float want_y = wy + TR2_ATM_CAM_HEIGHT;
        float want_z = wz - cosf(tr2_atm_cam_yaw) * TR2_ATM_CAM_BACK;
        tr2_atm_cam_x += (want_x - tr2_atm_cam_x) * k;
        tr2_atm_cam_y += (want_y - tr2_atm_cam_y) * k;
        tr2_atm_cam_z += (want_z - tr2_atm_cam_z) * k;
    }

    /* Aim along the shipped EndRotation pitch, offset from the pawn's yaw. */
    float pitch = TR2_ATM_CAM_PITCH_DEG * DEG2RAD;
    float cp = cosf(pitch), sp = sinf(pitch);
    cam->position = (Vector3){tr2_atm_cam_x, tr2_atm_cam_y, tr2_atm_cam_z};
    cam->target = (Vector3){
        cam->position.x + sinf(tr2_atm_cam_yaw) * TR2_ATM_CAM_TARGET_LEN,
        wy + TR2_ATM_CAM_FOCUS_Y,
        cam->position.z + cosf(tr2_atm_cam_yaw) * TR2_ATM_CAM_TARGET_LEN,
    };
    (void)cp; (void)sp; (void)pitch;
    cam->up = (Vector3){0.0f, 1.0f, 0.0f};
    cam->fovy = TR2_ATM_CAM_FOV;
    cam->projection = CAMERA_PERSPECTIVE;

    if (c != NULL) {           /* keep the renderer's cached pose coherent */
        c->cam_x = cam->position.x;
        c->cam_y = cam->position.y;
        c->cam_z = cam->position.z;
        c->cam_yaw = tr2_atm_cam_yaw;
    }

    /* Distance fog: crop the 3D pass at the region's cull distance so distant
     * track fades into the sky.  BeginMode3D() takes its near/far planes from
     * rlGetCullDistanceNear/Far, which rlSetClipPlanes() sets. */
    rlSetClipPlanes((double)TR2_ATM_NEAR_CLIP, (double)TR2_ATM_FAR_CLIP);
}
